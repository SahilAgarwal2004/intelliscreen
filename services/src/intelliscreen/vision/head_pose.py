"""3D Head Pose Estimation using Perspective-n-Point (PnP) geometry.

Resolves 3D orientation (yaw, pitch, roll) and translation of the candidate's head
by matching 2D facial landmark anchors to a canonical anthropometric 3D model.
Classifies looking-away events and projects 3D orientation axes for visual explainability.
"""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import HeadPoseAngles
from intelliscreen.vision.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    FaceLandmarkDetector,
    FacialLandmarkResult,
)
from intelliscreen.vision.preprocessor import PreprocessedFrame

logger = get_logger("head_pose")


# ============================================================================
# Domain Results & Perception Output
# ============================================================================

@dataclass
class HeadPoseResult:
    """Perception output from 3D Head Pose Estimation."""

    pose_detected: bool
    yaw: float
    pitch: float
    roll: float
    confidence: float
    is_looking_away: bool
    direction_label: str
    euler_angles: HeadPoseAngles
    rvec: np.ndarray  # Shape (3, 1) rotation vector
    tvec: np.ndarray  # Shape (3, 1) translation vector
    rotation_matrix: np.ndarray  # Shape (3, 3) rotation matrix
    is_missing: bool = False
    latency_ms: float = 0.0


# ============================================================================
# HeadPoseEstimator Pipeline
# ============================================================================

class HeadPoseEstimator:
    """Estimates continuous 3D head pose orientation and translation via Perspective-n-Point."""

    def __init__(
        self,
        yaw_threshold: float | None = None,
        pitch_threshold: float | None = None,
        model_3d: np.ndarray | None = None,
    ) -> None:
        """Initialize the head pose estimator.

        Args:
            yaw_threshold: Angular deviation (degrees) beyond which candidate is looking away.
            pitch_threshold: Angular vertical deviation (degrees) beyond which candidate is looking away.
            model_3d: 3D anthropometric face model array of shape (6, 3). Defaults to CANONICAL_FACE_MODEL_3D.
        """
        settings = get_settings()
        self.yaw_threshold = (
            yaw_threshold if yaw_threshold is not None else settings.head_yaw_threshold
        )
        self.pitch_threshold = (
            pitch_threshold if pitch_threshold is not None else settings.head_pitch_threshold
        )
        self.model_3d = (
            model_3d if model_3d is not None else CANONICAL_FACE_MODEL_3D
        ).astype(np.float64)

        self._landmark_detector: FaceLandmarkDetector | None = None
        logger.info(
            f"HeadPoseEstimator initialized (yaw_thresh={self.yaw_threshold}°, "
            f"pitch_thresh={self.pitch_threshold}°)"
        )

    def estimate(
        self,
        frame: PreprocessedFrame,
        landmarks_result: FacialLandmarkResult | None = None,
    ) -> HeadPoseResult:
        """Estimate 3D head pose from preprocessed frame and landmark anchors.

        Args:
            frame: Standardized PreprocessedFrame.
            landmarks_result: Optional pre-extracted FacialLandmarkResult. If None,
                              internal FaceLandmarkDetector is invoked.

        Returns:
            HeadPoseResult: Decomposed Euler angles, PnP transform matrices, and directional labels.
        """
        t0 = time.perf_counter()
        h, w = frame.processed_shape[:2]

        # 1. Obtain facial landmarks if not provided
        if landmarks_result is None:
            if self._landmark_detector is None:
                self._landmark_detector = FaceLandmarkDetector()
            landmarks_result = self._landmark_detector.detect(frame)

        # 2. Absence handling if no landmarks are detected
        if (
            not landmarks_result.landmarks_detected
            or landmarks_result.is_missing
            or len(landmarks_result.pnp_landmarks_2d) < 6
        ):
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_empty_result(latency_ms)

        # 3. Setup intrinsic camera matrix (pinhole model approximation)
        # focal_length ~ width (approx. 60 deg horizontal FoV)
        focal_length = float(w)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, float(w) / 2.0],
                [0.0, focal_length, float(h) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 4. Resolve Perspective-n-Point (PnP)
        image_points = landmarks_result.pnp_landmarks_2d[:6].astype(np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.model_3d,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as exc:
            logger.warning(f"cv2.solvePnP execution failed ({exc}). Returning empty pose.")
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_empty_result(latency_ms)

        if not success:
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_empty_result(latency_ms)

        # 5. Convert Rodrigues rotation vector to 3x3 rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 6. Extract Euler angles via RQ decomposition
        # OpenCV RQDecomp3x3 decomposes into rotations about X, Y, Z axes
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        # Map to intuitive convention conforming to HeadPoseAngles schema:
        # pitch: negative=down, positive=up (rotation about X)
        # yaw: negative=left, positive=right (rotation about Y)
        # roll: negative=tilt left, positive=tilt right (rotation about Z)
        pitch = float(round(-angles[0], 2))
        yaw = float(round(angles[1], 2))
        roll = float(round(angles[2], 2))

        # 7. Compute reprojection confidence
        projected_points, _ = cv2.projectPoints(
            self.model_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        reproj_error = float(
            np.mean(np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1))
        )
        confidence = float(round(max(0.0, min(1.0, 1.0 - reproj_error / 35.0)), 3))

        # 8. Domain model & Threshold Evaluation
        euler_angles = HeadPoseAngles(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=confidence,
        )
        is_looking_away = euler_angles.is_looking_away(
            yaw_threshold=self.yaw_threshold,
            pitch_threshold=self.pitch_threshold,
        )

        direction_label = self._classify_direction(yaw, pitch)
        latency_ms = (time.perf_counter() - t0) * 1000

        return HeadPoseResult(
            pose_detected=True,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=confidence,
            is_looking_away=is_looking_away,
            direction_label=direction_label,
            euler_angles=euler_angles,
            rvec=rvec,
            tvec=tvec,
            rotation_matrix=rotation_matrix,
            is_missing=False,
            latency_ms=round(latency_ms, 2),
        )

    def project_pose_axes(
        self,
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        length: float = 50.0,
    ) -> np.ndarray:
        """Project and draw 3D coordinate axes protruding from the candidate's nose tip.

        Axes convention:
        - X-axis: Red (Lateral / Right)
        - Y-axis: Green (Vertical / Down)
        - Z-axis: Blue (Depth / Forward pointing towards viewer)

        Args:
            image: BGR image array.
            rvec: Rodrigues rotation vector (3, 1).
            tvec: Translation vector (3, 1).
            length: Length of coordinate axes in 3D world units (mm).

        Returns:
            np.ndarray: Modified image with 3D coordinate axes drawn.
        """
        output = image.copy()
        h, w = image.shape[:2]

        focal_length = float(w)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, float(w) / 2.0],
                [0.0, focal_length, float(h) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D points: Origin (nose tip) and 3 orthogonal axis endpoints
        axes_3d = np.array(
            [
                [0.0, 0.0, 0.0],          # Nose origin
                [length, 0.0, 0.0],        # X-axis (Right)
                [0.0, length, 0.0],        # Y-axis (Down)
                [0.0, 0.0, -length],       # Z-axis (Forward pointing out from face)
            ],
            dtype=np.float64,
        )

        try:
            pts_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
            pts = pts_2d.reshape(-1, 2).astype(int)

            origin = tuple(pts[0])
            pt_x = tuple(pts[1])
            pt_y = tuple(pts[2])
            pt_z = tuple(pts[3])

            # Draw axes: X = Red (0, 0, 255), Y = Green (0, 255, 0), Z = Blue (255, 0, 0)
            cv2.arrowedLine(output, origin, pt_x, (0, 0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(output, origin, pt_y, (0, 255, 0), 2, tipLength=0.2)
            cv2.arrowedLine(output, origin, pt_z, (255, 0, 0), 2, tipLength=0.2)
        except Exception as exc:
            logger.warning(f"Failed to project coordinate axes: {exc}")

        return output

    def _classify_direction(self, yaw: float, pitch: float) -> str:
        """Classify discrete gaze/head orientation from angular values."""
        yaw_dev = abs(yaw) > self.yaw_threshold
        pitch_dev = abs(pitch) > self.pitch_threshold

        if not yaw_dev and not pitch_dev:
            return "LOOKING_FORWARD"

        if yaw_dev and pitch_dev:
            horiz = "LEFT" if yaw < 0 else "RIGHT"
            vert = "UP" if pitch > 0 else "DOWN"
            return f"LOOKING_{vert}_{horiz}"

        if yaw_dev:
            return "LOOKING_LEFT" if yaw < 0 else "LOOKING_RIGHT"

        return "LOOKING_UP" if pitch > 0 else "LOOKING_DOWN"

    def _build_empty_result(self, latency_ms: float) -> HeadPoseResult:
        """Return standardized empty result when head pose cannot be solved."""
        return HeadPoseResult(
            pose_detected=False,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            confidence=0.0,
            is_looking_away=False,
            direction_label="UNKNOWN",
            euler_angles=HeadPoseAngles(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            is_missing=True,
            latency_ms=round(latency_ms, 2),
        )
