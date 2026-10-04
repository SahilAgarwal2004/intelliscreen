"""Computer vision perception package for IntelliScreen."""

from intelliscreen.vision.face_detector import (
    BaseFaceDetector,
    DetectedFace,
    FaceDetectionResult,
    FaceDetector,
    HaarCascadeFaceDetector,
    YuNetFaceDetector,
)
from intelliscreen.vision.preprocessor import (
    FrameQualityMetrics,
    PreprocessedFrame,
    VideoPreprocessor,
)

__all__ = [
    # Preprocessing
    "FrameQualityMetrics",
    "PreprocessedFrame",
    "VideoPreprocessor",
    # Face Detection
    "BaseFaceDetector",
    "DetectedFace",
    "FaceDetectionResult",
    "FaceDetector",
    "YuNetFaceDetector",
    "HaarCascadeFaceDetector",
]
