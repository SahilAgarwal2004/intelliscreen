"""Gaze Estimation and Screen Deviation Analysis for IntelliScreen.

Fuses 3D head pose orientation with 2D/3D iris position ratios to derive absolute
candidate line-of-sight gaze vectors. Classifies discrete gaze direction, tracks
off-screen gaze deviations, and supports resting baseline calibration.
"""

from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import GazeDirection, GazeVector
from intelliscreen.vision.head_pose import HeadPoseEstimator, HeadPoseResult
from intelliscreen.vision.landmarks import FaceLandmarkDetector, FacialLandmarkResult
from intelliscreen.vision.preprocessor import PreprocessedFrame

logger = get_logger("gaze_estimator")

# Scaling sensitivities to convert normalized iris displacement (Δratio) to degrees
# Human ocular range: approx +/- 30 degrees horizontally, +/- 25 degrees vertically
DEFAULT_IRIS_YAW_SCALE: float = 70.0
DEFAULT_IRIS_PITCH_SCALE: float = 60.0


# ============================================================================
# Domain Results & Perception Output
# ============================================================================

@dataclass
class GazeResult:
    """Perception output from Gaze Estimation and Deviation Analysis."""

    gaze_detected: bool
    yaw: float  # Absolute gaze horizontal angle in degrees (negative=left, positive=right)
    pitch: float  # Absolute gaze vertical angle in degrees (negative=down, positive=up)
    direction: GazeDirection
    is_deviated: bool
    confidence: float
    gaze_vector: GazeVector
    left_eye_ratio: tuple[float, float]  # (horizontal_ratio, vertical_ratio)
    right_eye_ratio: tuple[float, float]
    head_pose_yaw: float
    head_pose_pitch: float
    is_missing: bool = False
    latency_ms: float = 0.0
    left_iris_center: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    right_iris_center: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )


# ============================================================================
# GazeEstimator Pipeline
# ============================================================================

class GazeEstimator:
    """Fuses 3D head pose and iris vectorization to compute gaze deviation."""

    def __init__(
        self,
        yaw_threshold: float | None = None,
        pitch_threshold: float | None = None,
        iris_yaw_scale: float = DEFAULT_IRIS_YAW_SCALE,
        iris_pitch_scale: float = DEFAULT_IRIS_PITCH_SCALE,
    ) -> None:
        """Initialize the gaze estimator.

        Args:
            yaw_threshold: Threshold (degrees) beyond which horizontal gaze is deviated.
            pitch_threshold: Threshold (degrees) beyond which vertical gaze is deviated.
            iris_yaw_scale: Angular multiplier mapping horizontal iris displacement to degrees.
            iris_pitch_scale: Angular multiplier mapping vertical iris displacement to degrees.
        """
        settings = get_settings()
        self.yaw_threshold = (
            yaw_threshold if yaw_threshold is not None else settings.gaze_yaw_threshold
        )
        self.pitch_threshold = (
            pitch_threshold if pitch_threshold is not None else settings.gaze_pitch_threshold
        )
        self.iris_yaw_scale = iris_yaw_scale
        self.iris_pitch_scale = iris_pitch_scale

        # Resting baseline calibration offset (degrees)
        self._baseline_yaw: float = 0.0
        self._baseline_pitch: float = 0.0

        # Subordinate pipeline estimators for automatic resolution
        self._landmark_detector: FaceLandmarkDetector | None = None
        self._head_pose_estimator: HeadPoseEstimator | None = None

        logger.info(
            f"GazeEstimator initialized (yaw_thresh={self.yaw_threshold}°, "
            f"pitch_thresh={self.pitch_threshold}°)"
        )

    def calibrate_baseline(self, yaw: float, pitch: float) -> None:
        """Set the candidate's natural resting gaze baseline.

        Subsequent gaze angle calculations will be measured relative to this offset.

        Args:
            yaw: Neutral resting yaw in degrees.
            pitch: Neutral resting pitch in degrees.
        """
        self._baseline_yaw = float(yaw)
        self._baseline_pitch = float(pitch)
        logger.info(f"Gaze baseline calibrated to ({self._baseline_yaw:.1f}°, {self._baseline_pitch:.1f}°)")

    def reset_calibration(self) -> None:
        """Reset resting baseline calibration back to absolute camera origin (0, 0)."""
        self._baseline_yaw = 0.0
        self._baseline_pitch = 0.0
        logger.info("Gaze baseline calibration reset to zero.")

    def estimate(
        self,
        frame: PreprocessedFrame,
        landmarks_result: FacialLandmarkResult | None = None,
        head_pose_result: HeadPoseResult | None = None,
    ) -> GazeResult:
        """Estimate gaze angles and detect screen deviation.

        Args:
            frame: Standardized PreprocessedFrame.
            landmarks_result: Optional precomputed facial landmarks.
            head_pose_result: Optional precomputed 3D head pose.

        Returns:
            GazeResult: Continuous gaze angles, discrete direction, and deviation flag.
        """
        t0 = time.perf_counter()

        # 1. Resolve landmarks if omitted
        if landmarks_result is None:
            if self._landmark_detector is None:
                self._landmark_detector = FaceLandmarkDetector()
            landmarks_result = self._landmark_detector.detect(frame)

        # 2. Absence handling if landmarks or face is missing
        if (
            not landmarks_result.landmarks_detected
            or landmarks_result.is_missing
            or len(landmarks_result.landmark_points_2d) < 468
        ):
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_empty_result(latency_ms)

        # 3. Resolve head pose if omitted
        if head_pose_result is None:
            if self._head_pose_estimator is None:
                self._head_pose_estimator = HeadPoseEstimator()
            head_pose_result = self._head_pose_estimator.estimate(frame, landmarks_result)

        head_yaw = head_pose_result.yaw if head_pose_result.pose_detected else 0.0
        head_pitch = head_pose_result.pitch if head_pose_result.pose_detected else 0.0

        # 4. Extract Iris and Eye Corner coordinates
        pts = landmarks_result.landmark_points_2d
        num_landmarks = len(pts)

        # Left Eye (Subject's Left / Viewer's Right):
        # 362: inner corner, 263: outer corner, 386: upper eyelid, 374: lower eyelid
        # 473: left iris center (fallback to corner midpoint)
        l_inner = pts[362]
        l_outer = pts[263]
        l_upper = pts[386]
        l_lower = pts[374]
        l_iris = pts[473] if num_landmarks > 473 else (l_inner + l_outer) / 2.0

        # Right Eye (Subject's Right / Viewer's Left):
        # 33: outer corner, 133: inner corner, 159: upper eyelid, 145: lower eyelid
        # 468: right iris center (fallback to corner midpoint)
        r_outer = pts[33]
        r_inner = pts[133]
        r_upper = pts[159]
        r_lower = pts[145]
        r_iris = pts[468] if num_landmarks > 468 else (r_outer + r_inner) / 2.0

        # 5. Compute Normalized Iris Position Ratios
        l_rx, l_ry = self._compute_iris_ratios(l_inner, l_outer, l_upper, l_lower, l_iris, is_left=True)
        r_rx, r_ry = self._compute_iris_ratios(r_outer, r_inner, r_upper, r_lower, r_iris, is_left=False)

        # Average eye position ratios
        avg_rx = (l_rx + r_rx) / 2.0
        avg_ry = (l_ry + r_ry) / 2.0

        # 6. Compute Iris Deflection Angles (degrees)
        # Shift relative to center (0.5).
        # Horizontal: rx > 0.5 -> looking left (+dx in image coordinates)
        # Vertical: ry < 0.5 -> looking up (-dy in image coordinates)
        delta_x = avg_rx - 0.5
        delta_y = 0.5 - avg_ry

        iris_yaw = delta_x * self.iris_yaw_scale
        iris_pitch = delta_y * self.iris_pitch_scale

        # 7. Fuse Head Pose with Eye Deflection and Apply Calibration
        raw_yaw = head_yaw + iris_yaw
        raw_pitch = head_pitch + iris_pitch

        gaze_yaw = float(round(raw_yaw - self._baseline_yaw, 2))
        gaze_pitch = float(round(raw_pitch - self._baseline_pitch, 2))

        # 8. Direction Classification and Deviation Check
        direction = self._classify_gaze_direction(gaze_yaw, gaze_pitch)
        gaze_vector = GazeVector(
            yaw=gaze_yaw,
            pitch=gaze_pitch,
            direction=direction,
            confidence=head_pose_result.confidence,
        )
        is_deviated = gaze_vector.is_deviated(
            yaw_threshold=self.yaw_threshold,
            pitch_threshold=self.pitch_threshold,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return GazeResult(
            gaze_detected=True,
            yaw=gaze_yaw,
            pitch=gaze_pitch,
            direction=direction,
            is_deviated=is_deviated,
            confidence=head_pose_result.confidence,
            gaze_vector=gaze_vector,
            left_eye_ratio=(round(l_rx, 3), round(l_ry, 3)),
            right_eye_ratio=(round(r_rx, 3), round(r_ry, 3)),
            head_pose_yaw=round(head_yaw, 2),
            head_pose_pitch=round(head_pitch, 2),
            is_missing=False,
            latency_ms=round(latency_ms, 2),
            left_iris_center=l_iris.astype(np.float32),
            right_iris_center=r_iris.astype(np.float32),
        )

    def draw_gaze_rays(
        self,
        image: np.ndarray,
        gaze: GazeResult,
        length: float = 60.0,
    ) -> np.ndarray:
        """Draw projected line-of-sight gaze rays from pupil centers.

        Args:
            image: BGR image canvas.
            gaze: Computed GazeResult.
            length: Ray length in pixels.

        Returns:
            np.ndarray: Modified image with gaze vectors overlaid.
        """
        output = image.copy()
        if not gaze.gaze_detected or gaze.is_missing:
            return output

        # Project 2D vector components from spherical angles
        rad_yaw = math.radians(gaze.yaw)
        rad_pitch = math.radians(gaze.pitch)

        dx = math.sin(rad_yaw) * length
        dy = -math.sin(rad_pitch) * length

        # Green if focused on screen, Orange/Red if deviated
        color = (0, 0, 255) if gaze.is_deviated else (0, 255, 0)

        for iris_center in (gaze.left_iris_center, gaze.right_iris_center):
            p1 = (int(iris_center[0]), int(iris_center[1]))
            p2 = (int(iris_center[0] + dx), int(iris_center[1] + dy))
            cv2.circle(output, p1, 3, (0, 255, 255), -1)
            cv2.arrowedLine(output, p1, p2, color, 2, tipLength=0.25)

        return output

    def _compute_iris_ratios(
        self,
        corner1: np.ndarray,
        corner2: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        iris: np.ndarray,
        is_left: bool,
    ) -> tuple[float, float]:
        """Compute horizontal and vertical normalized iris positions [0, 1]."""
        # Horizontal span: corner1 to corner2
        h_span = float(corner2[0] - corner1[0])
        if abs(h_span) < 1e-4:
            rx = 0.5
        else:
            rx = float((iris[0] - corner1[0]) / h_span)

        # Vertical span: upper eyelid to lower eyelid
        v_span = float(lower[1] - upper[1])
        if abs(v_span) < 1e-4:
            ry = 0.5
        else:
            ry = float((iris[1] - upper[1]) / v_span)

        # Clamp to realistic anatomical range [0.05, 0.95]
        rx = max(0.05, min(0.95, rx))
        ry = max(0.05, min(0.95, ry))

        return rx, ry

    def _classify_gaze_direction(self, yaw: float, pitch: float) -> GazeDirection:
        """Classify continuous gaze angles into discrete GazeDirection enum."""
        yaw_dev = abs(yaw) > self.yaw_threshold
        pitch_dev = abs(pitch) > self.pitch_threshold

        if not yaw_dev and not pitch_dev:
            return GazeDirection.CENTER

        if yaw_dev and pitch_dev:
            if yaw < 0 and pitch > 0:
                return GazeDirection.UP_LEFT
            if yaw > 0 and pitch > 0:
                return GazeDirection.UP_RIGHT
            if yaw < 0 and pitch < 0:
                return GazeDirection.DOWN_LEFT
            return GazeDirection.DOWN_RIGHT

        if yaw_dev:
            return GazeDirection.LEFT if yaw < 0 else GazeDirection.RIGHT

        return GazeDirection.UP if pitch > 0 else GazeDirection.DOWN

    def _build_empty_result(self, latency_ms: float) -> GazeResult:
        """Return standardized empty result when gaze cannot be computed."""
        return GazeResult(
            gaze_detected=False,
            yaw=0.0,
            pitch=0.0,
            direction=GazeDirection.CENTER,
            is_deviated=False,
            confidence=0.0,
            gaze_vector=GazeVector(
                yaw=0.0,
                pitch=0.0,
                direction=GazeDirection.CENTER,
                confidence=0.0,
            ),
            left_eye_ratio=(0.5, 0.5),
            right_eye_ratio=(0.5, 0.5),
            head_pose_yaw=0.0,
            head_pose_pitch=0.0,
            is_missing=True,
            latency_ms=round(latency_ms, 2),
            left_iris_center=np.zeros(2, dtype=np.float32),
            right_iris_center=np.zeros(2, dtype=np.float32),
        )
