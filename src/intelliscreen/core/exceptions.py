"""Domain-specific exception hierarchy for IntelliScreen."""


class IntelliScreenError(Exception):
    """Base exception for all domain errors within IntelliScreen."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(IntelliScreenError):
    """Raised when configuration validation or environment variable parsing fails."""


class VisionPipelineError(IntelliScreenError):
    """Raised when an error occurs during video decoding, preprocessing, or vision inference."""


class FramePreprocessingError(VisionPipelineError):
    """Raised when a video frame cannot be normalized, resized, or converted."""


class DetectionError(VisionPipelineError):
    """Raised when face or object localization fails due to corruption or invalid geometry."""


class ModelInferenceError(VisionPipelineError):
    """Raised when a vision, audio, or NLP model fails during forward inference."""


class TemporalEngineError(IntelliScreenError):
    """Raised when state accumulation or sliding window calculations encounter invalid timestamps."""


class NLPServiceError(IntelliScreenError):
    """Raised when transcription or semantic answer evaluation fails."""


class SessionError(IntelliScreenError):
    """Raised when an interview session operation is invalid or a session is not found."""
