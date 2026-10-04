"""Dense Facial Landmark Detection, Eye Aspect Ratio (EAR), and Anatomical Mapping.

Extracts 478 3D landmarks (face mesh + iris) using MediaPipe Task API with robust
fallback to 5-point geometric inference. Provides canonical anatomical region
mapping, Eye Aspect Ratio (EAR) blink detection, and key anchor subsets for 3D
Head Pose (solvePnP) and Gaze Vectorization.
"""

from dataclasses import dataclass, field
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.exceptions import DetectionError, ModelInferenceError
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import NormalizedLandmark
from intelliscreen.vision.face_detector import DetectedFace
from intelliscreen.vision.preprocessor import PreprocessedFrame

logger = get_logger("facial_landmarks")

DEFAULT_LANDMARK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
    "weights",
)


# ============================================================================
# Canonical Landmark Indices (MediaPipe 468 + 10 Iris = 478)
# ============================================================================

# Left Eye (Subject's Left / Camera-Right)
LEFT_EYE_INDICES: list[int] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

# Right Eye (Subject's Right / Camera-Left)
RIGHT_EYE_INDICES: list[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]

# Nose Bridge (Descending from glabella to nose apex)
NOSE_BRIDGE_INDICES: list[int] = [168, 6, 197, 195, 5, 4]

# Chin (Apex and lower mandibular contour)
CHIN_INDICES: list[int] = [
    152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10
]

# Mouth (Outer and inner lips contour)
MOUTH_INDICES: list[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185
]

# Iris landmarks (Refined 478-landmark topology)
LEFT_IRIS_INDICES: list[int] = [473, 474, 475, 476, 477]
RIGHT_IRIS_INDICES: list[int] = [468, 469, 470, 471, 472]

# Canonical 6-point indices for Eye Aspect Ratio (EAR) computation:
# Order: [p1_corner, p2_upper1, p3_upper2, p4_corner, p5_lower2, p6_lower1]
LEFT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)

# Canonical 6 Anthropometric PnP Keypoints for 3D Head Pose:
# 1: Nose Tip, 152: Chin, 263: Left Eye Outer Corner, 33: Right Eye Outer Corner,
# 291: Left Mouth Corner, 61: Right Mouth Corner
PNP_6POINT_INDICES: list[int] = [1, 152, 263, 33, 291, 61]

# Generic 3D Anthropometric Facial Model (in millimeters, coordinate frame centered at nose tip)
CANONICAL_FACE_MODEL_3D: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (index 1)
        (0.0, -330.0, -65.0),     # Chin (index 152)
        (225.0, 170.0, -135.0),   # Left eye outer corner (index 263)
        (-225.0, 170.0, -135.0),  # Right eye outer corner (index 33)
        (150.0, -150.0, -125.0),  # Left mouth corner (index 291)
        (-150.0, -150.0, -125.0), # Right mouth corner (index 61)
    ],
    dtype=np.float32,
)


# ============================================================================
# Mathematical Helper Functions
# ============================================================================

def compute_eye_aspect_ratio(
    eye_points: np.ndarray,
) -> float:
    """Calculate the Eye Aspect Ratio (EAR) for a 6-point eye contour.

    Follows the Soukupová and Čech (2016) formulation:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_points: An array of shape (6, 2) representing [p1, p2, p3, p4, p5, p6]
                    in order: [corner1, upper1, upper2, corner2, lower2, lower1].

    Returns:
        float: Scalar EAR value. Typically > 0.25 when open and < 0.20 when closed.
    """
    if eye_points.shape[0] < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    dist_v1 = float(np.linalg.norm(p2 - p6))
    dist_v2 = float(np.linalg.norm(p3 - p5))
    dist_h = float(np.linalg.norm(p1 - p4))

    if dist_h < 1e-6:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(round(ear, 4))


# ============================================================================
# Domain Results & Perception Output
# ============================================================================

@dataclass
class FacialLandmarkResult:
    """Structured perception output for dense facial landmark extraction."""

    landmarks_detected: bool
    landmarks: list[NormalizedLandmark]
    landmark_points_2d: np.ndarray  # Shape (N, 2), pixel coordinates (x, y)
    landmark_points_3d: np.ndarray  # Shape (N, 3), scaled coordinates (x, y, z*width)
    normalized_points_2d: np.ndarray  # Shape (N, 2), range [0, 1]
    normalized_points_3d: np.ndarray  # Shape (N, 3), range [0, 1]

    # Anatomical Region Index Mappings
    left_eye: list[int] = field(default_factory=lambda: list(LEFT_EYE_INDICES))
    right_eye: list[int] = field(default_factory=lambda: list(RIGHT_EYE_INDICES))
    nose_bridge: list[int] = field(default_factory=lambda: list(NOSE_BRIDGE_INDICES))
    chin: list[int] = field(default_factory=lambda: list(CHIN_INDICES))
    mouth: list[int] = field(default_factory=lambda: list(MOUTH_INDICES))
    left_iris: list[int] = field(default_factory=lambda: list(LEFT_IRIS_INDICES))
    right_iris: list[int] = field(default_factory=lambda: list(RIGHT_IRIS_INDICES))

    # Eye Aspect Ratio (EAR) & Blink Metrics
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    is_blinking: bool = False

    # Key Anatomical Anchor Subsets for PnP & Gaze
    pnp_landmarks_2d: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    pnp_landmarks_3d: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    gaze_anchors: dict[str, np.ndarray] = field(default_factory=dict)

    # Diagnostic & Pipeline State
    is_missing: bool = True
    is_fallback: bool = False
    latency_ms: float = 0.0

    def get_region_points_2d(self, region: str | list[int]) -> np.ndarray:
        """Extract 2D pixel coordinates for a named anatomical region or index list.

        Args:
            region: String region name (e.g. 'left_eye', 'chin') or list of integer indices.

        Returns:
            np.ndarray: Array of shape (K, 2) containing pixel coordinates.
        """
        if not self.landmarks_detected or len(self.landmark_points_2d) == 0:
            return np.empty((0, 2), dtype=np.float32)

        if isinstance(region, str):
            if not hasattr(self, region):
                raise ValueError(f"Unknown anatomical region '{region}'.")
            indices = getattr(self, region)
        else:
            indices = region

        return self.landmark_points_2d[indices]

    def get_region_points_3d(self, region: str | list[int]) -> np.ndarray:
        """Extract 3D coordinates for a named anatomical region or index list.

        Args:
            region: String region name or list of integer indices.

        Returns:
            np.ndarray: Array of shape (K, 3) containing 3D coordinates.
        """
        if not self.landmarks_detected or len(self.landmark_points_3d) == 0:
            return np.empty((0, 3), dtype=np.float32)

        if isinstance(region, str):
            if not hasattr(self, region):
                raise ValueError(f"Unknown anatomical region '{region}'.")
            indices = getattr(self, region)
        else:
            indices = region

        return self.landmark_points_3d[indices]

    def get_pnp_6points(self) -> np.ndarray:
        """Return 2D pixel coordinates for the canonical 6 anthropometric PnP anchors.

        Order: [Nose tip, Chin, Left eye outer, Right eye outer, Left mouth, Right mouth].
        """
        return self.pnp_landmarks_2d


# ============================================================================
# FaceLandmarkDetector Pipeline
# ============================================================================

class FaceLandmarkDetector:
    """Unified facial landmark detector utilizing MediaPipe Task API with robust geometric fallback."""

    def __init__(
        self,
        model_path: str | None = None,
        blink_ear_threshold: float | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        backend: str = "mediapipe",
    ) -> None:
        """Initialize the facial landmark detector.

        Args:
            model_path: Path to the MediaPipe face_landmarker.task file.
            blink_ear_threshold: EAR threshold below which eyes are flagged as blinking.
            min_detection_confidence: Confidence threshold for face detection.
            min_presence_confidence: Confidence threshold for face presence.
            min_tracking_confidence: Confidence threshold for landmark tracking.
            backend: Landmark detection backend ('mediapipe' or 'fallback').
        """
        settings = get_settings()
        self.blink_ear_threshold = (
            blink_ear_threshold
            if blink_ear_threshold is not None
            else getattr(settings, "landmark_blink_threshold", 0.20)
        )
        self.backend = backend.lower()
        self.model_path = model_path or os.path.join(
            DEFAULT_WEIGHTS_DIR, "face_landmarker.task"
        )
        self._landmarker: vision.FaceLandmarker | None = None

        if self.backend == "mediapipe":
            self._init_mediapipe_landmarker(
                min_detection_confidence=min_detection_confidence,
                min_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        elif self.backend == "fallback":
            logger.info("FaceLandmarkDetector initialized in pure fallback mode.")
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'mediapipe' or 'fallback'."
            )

    def _ensure_model_exists(self) -> None:
        """Verify local existence of face_landmarker.task or download from Google CDN."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(
                f"Downloading MediaPipe Face Landmarker model from {DEFAULT_LANDMARK_URL}..."
            )
            try:
                urllib.request.urlretrieve(DEFAULT_LANDMARK_URL, self.model_path)
                logger.info(
                    f"Downloaded Face Landmarker model ({os.path.getsize(self.model_path)} bytes)."
                )
            except Exception as exc:
                raise ModelInferenceError(
                    f"Unable to download Face Landmarker model to {self.model_path}: {exc}"
                ) from exc

    def _init_mediapipe_landmarker(
        self,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """Instantiate MediaPipe FaceLandmarker task runner."""
        try:
            self._ensure_model_exists()
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info(f"MediaPipe FaceLandmarker successfully initialized from {self.model_path}")
        except Exception as exc:
            logger.warning(
                f"MediaPipe FaceLandmarker initialization failed ({exc}). Operating in fallback mode."
            )
            self.backend = "fallback"
            self._landmarker = None

    def detect(
        self,
        frame: PreprocessedFrame,
        primary_face: DetectedFace | None = None,
    ) -> FacialLandmarkResult:
        """Extract dense 3D facial landmarks from a preprocessed video frame.

        Args:
            frame: PreprocessedFrame containing RGB, BGR, and grayscale image representations.
            primary_face: Optional localized face detection from FaceDetector (YuNet or Haar).

        Returns:
            FacialLandmarkResult: Comprehensive landmark coordinates, EAR metrics, and anatomical subsets.
        """
        t0 = time.perf_counter()
        h, w = frame.processed_shape[:2]

        # 1. Try MediaPipe Task API inference if enabled and initialized
        if self.backend == "mediapipe" and self._landmarker is not None:
            try:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame.rgb_image,
                )
                mp_res = self._landmarker.detect(mp_image)

                if mp_res.face_landmarks and len(mp_res.face_landmarks) > 0:
                    raw_lms = mp_res.face_landmarks[0]
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return self._build_result_from_mediapipe(raw_lms, w, h, latency_ms)
            except Exception as exc:
                logger.warning(f"MediaPipe landmark detection error: {exc}. Attempting fallback.")

        # 2. If MediaPipe returned no face or failed, attempt geometric fallback with primary_face
        if primary_face is not None and primary_face.box.area > 0:
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_fallback_result(primary_face, w, h, latency_ms)

        # 3. No face detected by any backend
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._build_empty_result(latency_ms)

    def _build_result_from_mediapipe(
        self,
        raw_lms: list,
        width: int,
        height: int,
        latency_ms: float,
    ) -> FacialLandmarkResult:
        """Construct FacialLandmarkResult from raw MediaPipe NormalizedLandmarks."""
        num_landmarks = len(raw_lms)

        # Arrays for normalized and pixel-space coordinates
        norm_2d = np.empty((num_landmarks, 2), dtype=np.float32)
        norm_3d = np.empty((num_landmarks, 3), dtype=np.float32)
        pts_2d = np.empty((num_landmarks, 2), dtype=np.float32)
        pts_3d = np.empty((num_landmarks, 3), dtype=np.float32)
        domain_landmarks: list[NormalizedLandmark] = []

        w_float, h_float = float(width), float(height)

        for i, lm in enumerate(raw_lms):
            lx, ly, lz = float(lm.x), float(lm.y), float(lm.z)
            norm_2d[i] = (lx, ly)
            norm_3d[i] = (lx, ly, lz)
            pts_2d[i] = (lx * w_float, ly * h_float)
            pts_3d[i] = (lx * w_float, ly * h_float, lz * w_float)

            vis = getattr(lm, "visibility", None)
            domain_landmarks.append(
                NormalizedLandmark(
                    x=lx,
                    y=ly,
                    z=lz,
                    visibility=float(vis) if vis is not None else None,
                )
            )

        # Compute Eye Aspect Ratios (EAR)
        left_ear_pts = pts_2d[list(LEFT_EYE_EAR_INDICES)]
        right_ear_pts = pts_2d[list(RIGHT_EYE_EAR_INDICES)]
        left_ear = compute_eye_aspect_ratio(left_ear_pts)
        right_ear = compute_eye_aspect_ratio(right_ear_pts)
        avg_ear = float(round((left_ear + right_ear) / 2.0, 4))
        is_blinking = bool(avg_ear < self.blink_ear_threshold)

        # Extract PnP 6-point subset
        pnp_2d = pts_2d[PNP_6POINT_INDICES].copy()
        pnp_3d = pts_3d[PNP_6POINT_INDICES].copy()

        # Extract Gaze Vectorization Anchors
        gaze_anchors: dict[str, np.ndarray] = {
            "left_iris_center_2d": pts_2d[473].copy() if num_landmarks > 473 else pts_2d[LEFT_EYE_EAR_INDICES[0]],
            "right_iris_center_2d": pts_2d[468].copy() if num_landmarks > 468 else pts_2d[RIGHT_EYE_EAR_INDICES[0]],
            "left_eye_inner_corner_2d": pts_2d[362].copy(),
            "left_eye_outer_corner_2d": pts_2d[263].copy(),
            "right_eye_inner_corner_2d": pts_2d[133].copy(),
            "right_eye_outer_corner_2d": pts_2d[33].copy(),
            "left_iris_center_3d": pts_3d[473].copy() if num_landmarks > 473 else pts_3d[LEFT_EYE_EAR_INDICES[0]],
            "right_iris_center_3d": pts_3d[468].copy() if num_landmarks > 468 else pts_3d[RIGHT_EYE_EAR_INDICES[0]],
        }

        return FacialLandmarkResult(
            landmarks_detected=True,
            landmarks=domain_landmarks,
            landmark_points_2d=pts_2d,
            landmark_points_3d=pts_3d,
            normalized_points_2d=norm_2d,
            normalized_points_3d=norm_3d,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            is_blinking=is_blinking,
            pnp_landmarks_2d=pnp_2d,
            pnp_landmarks_3d=pnp_3d,
            gaze_anchors=gaze_anchors,
            is_missing=False,
            is_fallback=False,
            latency_ms=round(latency_ms, 2),
        )

    def _build_fallback_result(
        self,
        primary_face: DetectedFace,
        width: int,
        height: int,
        latency_ms: float,
    ) -> FacialLandmarkResult:
        """Synthesize anatomical anchors from YuNet 5 keypoints or bounding box geometry."""
        box = primary_face.box
        kp = primary_face.keypoints
        w_float, h_float = float(width), float(height)

        # Deduce key anatomical coordinates
        if "nose_tip" in kp and "left_eye" in kp and "right_eye" in kp:
            nose_tip = np.array(kp["nose_tip"], dtype=np.float32)
            left_eye_outer = np.array(kp["left_eye"], dtype=np.float32)
            right_eye_outer = np.array(kp["right_eye"], dtype=np.float32)
            left_mouth = np.array(kp.get("left_mouth", (box.xmax - 0.2 * box.width, box.ymax - 0.2 * box.height)), dtype=np.float32)
            right_mouth = np.array(kp.get("right_mouth", (box.xmin + 0.2 * box.width, box.ymax - 0.2 * box.height)), dtype=np.float32)
            chin_x = float((left_mouth[0] + right_mouth[0]) / 2.0)
            chin_y = float(min(box.ymax, max(left_mouth[1], right_mouth[1]) + 0.25 * box.height))
            chin = np.array((chin_x, chin_y), dtype=np.float32)
        else:
            # Fallback when keypoints are missing (e.g. Haar Cascade)
            cx, cy = box.center
            bw, bh = box.width, box.height
            nose_tip = np.array((cx, cy + 0.05 * bh), dtype=np.float32)
            chin = np.array((cx, box.ymax - 0.05 * bh), dtype=np.float32)
            left_eye_outer = np.array((cx + 0.3 * bw, cy - 0.15 * bh), dtype=np.float32)
            right_eye_outer = np.array((cx - 0.3 * bw, cy - 0.15 * bh), dtype=np.float32)
            left_mouth = np.array((cx + 0.2 * bw, cy + 0.25 * bh), dtype=np.float32)
            right_mouth = np.array((cx - 0.2 * bw, cy + 0.25 * bh), dtype=np.float32)

        # Construct PnP 6 points: [Nose tip, Chin, Left eye, Right eye, Left mouth, Right mouth]
        pnp_2d = np.array(
            [nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth],
            dtype=np.float32,
        )
        pnp_3d = np.hstack([pnp_2d, np.zeros((6, 1), dtype=np.float32)])

        # Construct a dense array of 478 landmarks to ensure safe index access
        pts_2d = np.zeros((478, 2), dtype=np.float32)
        pts_3d = np.zeros((478, 3), dtype=np.float32)
        norm_2d = np.zeros((478, 2), dtype=np.float32)
        norm_3d = np.zeros((478, 3), dtype=np.float32)

        # Place known anchors into canonical index slots
        anchor_slots = {
            1: nose_tip,
            152: chin,
            263: left_eye_outer,
            33: right_eye_outer,
            291: left_mouth,
            61: right_mouth,
            168: (left_eye_outer + right_eye_outer) / 2.0,  # Glabella / bridge
            362: (left_eye_outer * 0.7 + right_eye_outer * 0.3),  # Left eye inner
            133: (right_eye_outer * 0.7 + left_eye_outer * 0.3),  # Right eye inner
            473: left_eye_outer,   # Left iris proxy
            468: right_eye_outer,  # Right iris proxy
        }

        # Seed the entire array with face center so any unpopulated index returns face center
        face_center_2d = np.array(box.center, dtype=np.float32)
        pts_2d[:] = face_center_2d

        for idx, pt in anchor_slots.items():
            pts_2d[idx] = pt

        # Fill regions with reasonable approximations around anchors
        for idx in LEFT_EYE_INDICES:
            if idx not in anchor_slots:
                pts_2d[idx] = left_eye_outer
        for idx in RIGHT_EYE_INDICES:
            if idx not in anchor_slots:
                pts_2d[idx] = right_eye_outer
        for idx in CHIN_INDICES:
            if idx not in anchor_slots:
                pts_2d[idx] = chin
        for idx in MOUTH_INDICES:
            if idx not in anchor_slots:
                pts_2d[idx] = (left_mouth + right_mouth) / 2.0
        for idx in NOSE_BRIDGE_INDICES:
            if idx not in anchor_slots:
                pts_2d[idx] = nose_tip

        # Populate 3D and normalized arrays
        pts_3d[:, :2] = pts_2d
        norm_2d[:, 0] = pts_2d[:, 0] / max(w_float, 1.0)
        norm_2d[:, 1] = pts_2d[:, 1] / max(h_float, 1.0)
        norm_3d[:, :2] = norm_2d

        domain_landmarks = [
            NormalizedLandmark(x=float(norm_2d[i, 0]), y=float(norm_2d[i, 1]), z=0.0)
            for i in range(478)
        ]

        # Estimated default open eye EAR for fallback
        left_ear = 0.28
        right_ear = 0.28
        avg_ear = 0.28
        is_blinking = False

        gaze_anchors = {
            "left_iris_center_2d": left_eye_outer.copy(),
            "right_iris_center_2d": right_eye_outer.copy(),
            "left_eye_inner_corner_2d": pts_2d[362].copy(),
            "left_eye_outer_corner_2d": left_eye_outer.copy(),
            "right_eye_inner_corner_2d": pts_2d[133].copy(),
            "right_eye_outer_corner_2d": right_eye_outer.copy(),
            "left_iris_center_3d": np.append(left_eye_outer, 0.0),
            "right_iris_center_3d": np.append(right_eye_outer, 0.0),
        }

        return FacialLandmarkResult(
            landmarks_detected=True,
            landmarks=domain_landmarks,
            landmark_points_2d=pts_2d,
            landmark_points_3d=pts_3d,
            normalized_points_2d=norm_2d,
            normalized_points_3d=norm_3d,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            is_blinking=is_blinking,
            pnp_landmarks_2d=pnp_2d,
            pnp_landmarks_3d=pnp_3d,
            gaze_anchors=gaze_anchors,
            is_missing=False,
            is_fallback=True,
            latency_ms=round(latency_ms, 2),
        )

    def _build_empty_result(self, latency_ms: float) -> FacialLandmarkResult:
        """Return standardized empty result when no face is present."""
        return FacialLandmarkResult(
            landmarks_detected=False,
            landmarks=[],
            landmark_points_2d=np.empty((0, 2), dtype=np.float32),
            landmark_points_3d=np.empty((0, 3), dtype=np.float32),
            normalized_points_2d=np.empty((0, 2), dtype=np.float32),
            normalized_points_3d=np.empty((0, 3), dtype=np.float32),
            left_ear=0.0,
            right_ear=0.0,
            avg_ear=0.0,
            is_blinking=False,
            pnp_landmarks_2d=np.empty((0, 2), dtype=np.float32),
            pnp_landmarks_3d=np.empty((0, 3), dtype=np.float32),
            gaze_anchors={},
            is_missing=True,
            is_fallback=False,
            latency_ms=round(latency_ms, 2),
        )
