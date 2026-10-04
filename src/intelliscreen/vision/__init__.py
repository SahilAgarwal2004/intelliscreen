"""Computer vision perception package for IntelliScreen."""

from intelliscreen.vision.face_detector import (
    BaseFaceDetector,
    DetectedFace,
    FaceDetectionResult,
    FaceDetector,
    HaarCascadeFaceDetector,
    YuNetFaceDetector,
)
from intelliscreen.vision.gaze import (
    GazeEstimator,
    GazeResult,
)
from intelliscreen.vision.head_pose import (
    HeadPoseEstimator,
    HeadPoseResult,
)
from intelliscreen.vision.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    CHIN_INDICES,
    FaceLandmarkDetector,
    FacialLandmarkResult,
    LEFT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    MOUTH_INDICES,
    NOSE_BRIDGE_INDICES,
    PNP_6POINT_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_IRIS_INDICES,
    compute_eye_aspect_ratio,
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
    # Facial Landmark Detection
    "FacialLandmarkResult",
    "FaceLandmarkDetector",
    "compute_eye_aspect_ratio",
    "CANONICAL_FACE_MODEL_3D",
    "PNP_6POINT_INDICES",
    "LEFT_EYE_INDICES",
    "RIGHT_EYE_INDICES",
    "NOSE_BRIDGE_INDICES",
    "CHIN_INDICES",
    "MOUTH_INDICES",
    "LEFT_IRIS_INDICES",
    "RIGHT_IRIS_INDICES",
    # 3D Head Pose Estimation
    "HeadPoseEstimator",
    "HeadPoseResult",
    # Gaze Estimation & Deviation Tracking
    "GazeEstimator",
    "GazeResult",
]
