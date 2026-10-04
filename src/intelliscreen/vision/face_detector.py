"""High-performance Face Detection pipeline for IntelliScreen.

Supports OpenCV YuNet (modern neural detector) with automatic model management
and Haar Cascade as an offline, parameter-free fallback.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import time
import urllib.request

import cv2
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.exceptions import DetectionError, ModelInferenceError
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import BoundingBox
from intelliscreen.vision.preprocessor import PreprocessedFrame

logger = get_logger("face_detector")

DEFAULT_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
    "weights",
)


@dataclass
class DetectedFace:
    """Localized face instance with confidence and anatomical anchor keypoints."""

    box: BoundingBox
    confidence: float
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return self.box.area


@dataclass
class FaceDetectionResult:
    """Structured perception output for face detection."""

    face_detected: bool
    face_count: int
    faces: list[DetectedFace]
    primary_face: DetectedFace | None
    is_multiple_faces: bool
    is_missing: bool
    latency_ms: float


class BaseFaceDetector(ABC):
    """Abstract interface for face detection backends."""

    @abstractmethod
    def detect(self, frame: PreprocessedFrame) -> FaceDetectionResult:
        """Detect faces within the given preprocessed video frame."""


class YuNetFaceDetector(BaseFaceDetector):
    """OpenCV FaceDetectorYN neural network inference backend."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        nms_threshold: float = 0.3,
    ) -> None:
        settings = get_settings()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.face_confidence_threshold
        )
        self.nms_threshold = nms_threshold

        self.model_path = model_path or os.path.join(
            DEFAULT_WEIGHTS_DIR, "face_detection_yunet_2023mar.onnx"
        )
        self._ensure_model_exists()

        try:
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (settings.vision_frame_width, settings.vision_frame_height),
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
            self._current_input_size = (
                settings.vision_frame_width,
                settings.vision_frame_height,
            )
            logger.info(f"YuNet face detector initialized from {self.model_path}")
        except Exception as exc:
            raise ModelInferenceError(
                f"Failed to initialize YuNet face detector: {exc}"
            ) from exc

    def _ensure_model_exists(self) -> None:
        """Download YuNet ONNX weights if not present locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Downloading YuNet model weights from {DEFAULT_YUNET_URL}...")
            try:
                urllib.request.urlretrieve(DEFAULT_YUNET_URL, self.model_path)
                logger.info(f"Downloaded YuNet weights ({os.path.getsize(self.model_path)} bytes)")
            except Exception as exc:
                raise ModelInferenceError(
                    f"Unable to download YuNet model to {self.model_path}: {exc}"
                ) from exc

    def detect(self, frame: PreprocessedFrame) -> FaceDetectionResult:
        """Run YuNet inference on the preprocessed BGR frame."""
        t0 = time.perf_counter()
        img = frame.bgr_image
        h, w = img.shape[:2]

        if self._current_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        try:
            status, raw_faces = self.detector.detect(img)
        except Exception as exc:
            raise DetectionError(f"YuNet detection execution failed: {exc}") from exc

        faces: list[DetectedFace] = []
        if raw_faces is not None and len(raw_faces) > 0:
            for item in raw_faces:
                conf = float(item[14])
                if conf < self.confidence_threshold:
                    continue

                x, y, bw, bh = float(item[0]), float(item[1]), float(item[2]), float(item[3])

                # Clamp to frame boundary
                xmin = max(0.0, x)
                ymin = max(0.0, y)
                xmax = min(float(w), x + bw)
                ymax = min(float(h), y + bh)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # 5 keypoints: right eye, left eye, nose tip, right mouth, left mouth
                keypoints = {
                    "right_eye": (float(item[4]), float(item[5])),
                    "left_eye": (float(item[6]), float(item[7])),
                    "nose_tip": (float(item[8]), float(item[9])),
                    "right_mouth": (float(item[10]), float(item[11])),
                    "left_mouth": (float(item[12]), float(item[13])),
                }

                box = BoundingBox(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    confidence=conf,
                    label="face",
                )
                faces.append(DetectedFace(box=box, confidence=conf, keypoints=keypoints))

        latency_ms = (time.perf_counter() - t0) * 1000
        primary_face = self._select_primary_face(faces, w, h)

        return FaceDetectionResult(
            face_detected=len(faces) > 0,
            face_count=len(faces),
            faces=faces,
            primary_face=primary_face,
            is_multiple_faces=len(faces) > 1,
            is_missing=len(faces) == 0,
            latency_ms=round(latency_ms, 2),
        )

    def _select_primary_face(
        self, faces: list[DetectedFace], width: int, height: int
    ) -> DetectedFace | None:
        """Select the candidate face based on bounding box area and proximity to screen center."""
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]

        cx, cy = width / 2.0, height / 2.0

        def score_face(face: DetectedFace) -> float:
            fcx, fcy = face.box.center
            dist_sq = (fcx - cx) ** 2 + (fcy - cy) ** 2
            norm_dist = np.sqrt(dist_sq) / np.sqrt(cx**2 + cy**2)
            # Favor larger bounding box area and lower distance to screen center
            return float(face.box.area * (1.0 - 0.4 * norm_dist))

        return max(faces, key=score_face)


class HaarCascadeFaceDetector(BaseFaceDetector):
    """Fallback offline OpenCV Haar Cascade face detector."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise ModelInferenceError(f"Failed to load Haar Cascade from {cascade_path}")
        logger.info("Haar Cascade face detector initialized as fallback")

    def detect(self, frame: PreprocessedFrame) -> FaceDetectionResult:
        """Run Haar cascade on the grayscale frame."""
        t0 = time.perf_counter()
        gray = frame.gray_image
        h, w = gray.shape[:2]

        raw_faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        faces: list[DetectedFace] = []
        if len(raw_faces) > 0:
            for x, y, bw, bh in raw_faces:
                box = BoundingBox(
                    xmin=float(x),
                    ymin=float(y),
                    xmax=float(min(w, x + bw)),
                    ymax=float(min(h, y + bh)),
                    confidence=0.85,  # Haar cascade does not provide continuous confidence
                    label="face",
                )
                faces.append(DetectedFace(box=box, confidence=0.85))

        latency_ms = (time.perf_counter() - t0) * 1000
        primary_face = max(faces, key=lambda f: f.box.area) if faces else None

        return FaceDetectionResult(
            face_detected=len(faces) > 0,
            face_count=len(faces),
            faces=faces,
            primary_face=primary_face,
            is_multiple_faces=len(faces) > 1,
            is_missing=len(faces) == 0,
            latency_ms=round(latency_ms, 2),
        )


class FaceDetector:
    """Unified face detection manager with backend selection and automated fallback."""

    def __init__(
        self,
        backend: str = "yunet",
        confidence_threshold: float | None = None,
        model_path: str | None = None,
    ) -> None:
        self.backend_name = backend.lower()
        if self.backend_name == "yunet":
            try:
                self.backend: BaseFaceDetector = YuNetFaceDetector(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                )
            except Exception as exc:
                logger.warning(
                    f"YuNet initialization failed ({exc}). Falling back to Haar Cascade detector."
                )
                self.backend = HaarCascadeFaceDetector()
                self.backend_name = "haar"
        elif self.backend_name == "haar":
            self.backend = HaarCascadeFaceDetector()
        else:
            raise ValueError(f"Unknown face detector backend '{backend}'. Choose 'yunet' or 'haar'.")

    def detect(self, frame: PreprocessedFrame) -> FaceDetectionResult:
        """Run face detection using configured backend."""
        return self.backend.detect(frame)
