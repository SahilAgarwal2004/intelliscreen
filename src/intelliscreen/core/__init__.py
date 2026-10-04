"""Core abstractions, schemas, logging, and exceptions for IntelliScreen."""

from intelliscreen.core.exceptions import (
    ConfigurationError,
    DetectionError,
    FramePreprocessingError,
    IntelliScreenError,
    ModelInferenceError,
    NLPServiceError,
    SessionError,
    TemporalEngineError,
    VisionPipelineError,
)
from intelliscreen.core.logging import get_logger, setup_logger
from intelliscreen.core.schemas import (
    AnswerEvaluation,
    BehaviorEvent,
    BoundingBox,
    DetectedObject,
    EventType,
    FrameObservation,
    GazeDirection,
    GazeVector,
    HeadPoseAngles,
    NormalizedLandmark,
    QuestionContext,
    SessionState,
    SuspicionScore,
    SuspicionSeverity,
    TranscriptSegment,
)

__all__ = [
    # Schemas
    "BoundingBox",
    "NormalizedLandmark",
    "HeadPoseAngles",
    "GazeDirection",
    "GazeVector",
    "DetectedObject",
    "FrameObservation",
    "EventType",
    "BehaviorEvent",
    "TranscriptSegment",
    "QuestionContext",
    "AnswerEvaluation",
    "SuspicionSeverity",
    "SuspicionScore",
    "SessionState",
    # Logging
    "setup_logger",
    "get_logger",
    # Exceptions
    "IntelliScreenError",
    "ConfigurationError",
    "VisionPipelineError",
    "FramePreprocessingError",
    "ModelInferenceError",
    "TemporalEngineError",
    "NLPServiceError",
    "SessionError",
]
