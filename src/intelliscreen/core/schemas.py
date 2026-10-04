"""Core domain models, data transfer objects, and schemas for IntelliScreen."""

from enum import Enum
import time
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field, model_validator


# ============================================================================
# 1. Geometric & Visual Perception Schemas
# ============================================================================

class BoundingBox(BaseModel):
    """Normalized or absolute 2D bounding box."""

    xmin: float = Field(description="Left coordinate.")
    ymin: float = Field(description="Top coordinate.")
    xmax: float = Field(description="Right coordinate.")
    ymax: float = Field(description="Bottom coordinate.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence.")
    label: str | None = Field(default=None, description="Optional class label.")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.xmax < self.xmin:
            raise ValueError(f"xmax ({self.xmax}) must be greater than or equal to xmin ({self.xmin})")
        if self.ymax < self.ymin:
            raise ValueError(f"ymax ({self.ymax}) must be greater than or equal to ymin ({self.ymin})")
        return self

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0


class NormalizedLandmark(BaseModel):
    """3D facial or iris landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float | None = Field(default=None, ge=0.0, le=1.0)


class HeadPoseAngles(BaseModel):
    """Euler angles for head rotation in degrees."""

    yaw: float = Field(description="Horizontal rotation: negative=left, positive=right.")
    pitch: float = Field(description="Vertical rotation: negative=down, positive=up.")
    roll: float = Field(description="Lateral tilt: negative=tilt left, positive=tilt right.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def is_looking_away(self, yaw_threshold: float = 30.0, pitch_threshold: float = 25.0) -> bool:
        """Check whether head pose exceeds angular bounds."""
        return abs(self.yaw) > yaw_threshold or abs(self.pitch) > pitch_threshold


class GazeDirection(str, Enum):
    """Discrete gaze direction classifications."""

    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class GazeVector(BaseModel):
    """Continuous gaze deviation and discrete classification."""

    yaw: float = Field(description="Gaze horizontal angle in degrees.")
    pitch: float = Field(description="Gaze vertical angle in degrees.")
    direction: GazeDirection = Field(default=GazeDirection.CENTER)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def is_deviated(self, yaw_threshold: float = 25.0, pitch_threshold: float = 20.0) -> bool:
        """Check whether gaze vector exceeds center screen threshold."""
        return abs(self.yaw) > yaw_threshold or abs(self.pitch) > pitch_threshold


class DetectedObject(BaseModel):
    """Detected secondary object in frame (e.g. phone, additional person)."""

    label: str = Field(description="Object category label (e.g. 'cell phone', 'person').")
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class FrameObservation(BaseModel):
    """Structured perception output for a single processed video frame."""

    frame_index: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0, description="Session elapsed time in milliseconds.")
    face_detected: bool = Field(description="True if at least one face was detected.")
    face_count: int = Field(ge=0, description="Total faces detected in the frame.")
    primary_face_box: BoundingBox | None = None
    head_pose: HeadPoseAngles | None = None
    gaze: GazeVector | None = None
    detected_objects: list[DetectedObject] = Field(default_factory=list)


# ============================================================================
# 2. Behavioral Event Schemas
# ============================================================================

class EventType(str, Enum):
    """Discrete behavioral events derived from temporal observation streams."""

    FACE_MISSING = "FACE_MISSING"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOOKING_LEFT = "LOOKING_LEFT"
    LOOKING_RIGHT = "LOOKING_RIGHT"
    LOOKING_UP = "LOOKING_UP"
    LOOKING_DOWN = "LOOKING_DOWN"
    PROLONGED_OFF_SCREEN_GAZE = "PROLONGED_OFF_SCREEN_GAZE"
    PHONE_DETECTED = "PHONE_DETECTED"
    HEAD_TURN = "HEAD_TURN"
    POSSIBLE_ABSENCE = "POSSIBLE_ABSENCE"


class BehaviorEvent(BaseModel):
    """Aggregated behavioral event over a temporal interval."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    start_time: float = Field(ge=0.0, description="Start timestamp in seconds.")
    end_time: float = Field(ge=0.0, description="End timestamp in seconds.")
    duration: float = Field(ge=0.0, description="Event duration in seconds.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score.")
    source: str = Field(description="Subsystem generating this event (e.g. 'gaze_engine').")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_duration_and_times(self) -> "BehaviorEvent":
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) cannot be earlier than start_time ({self.start_time})"
            )
        computed_duration = round(self.end_time - self.start_time, 3)
        if abs(self.duration - computed_duration) > 0.05:
            # Reconcile minor floating-point difference
            self.duration = computed_duration
        return self


# ============================================================================
# 3. Audio & Semantic NLP Schemas
# ============================================================================

class TranscriptSegment(BaseModel):
    """Transcribed speech segment with temporal bounds."""

    segment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    start_time: float = Field(ge=0.0, description="Segment start in seconds.")
    end_time: float = Field(ge=0.0, description="Segment end in seconds.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    speaker: str = "candidate"


class QuestionContext(BaseModel):
    """Structured technical interview question definition."""

    question_id: str
    question: str
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    reference_answer: str
    expected_concepts: list[str] = Field(default_factory=list)


class AnswerEvaluation(BaseModel):
    """Semantic scoring and concept recall analysis for an interview answer."""

    question_id: str
    candidate_answer: str
    semantic_similarity: float = Field(ge=0.0, le=1.0, description="Embedding cosine similarity.")
    concept_coverage: float = Field(ge=0.0, le=1.0, description="Recall ratio of expected concepts.")
    completeness: float = Field(ge=0.0, le=1.0, description="Depth and completeness score.")
    overall_score: float = Field(ge=0.0, le=100.0, description="Composite answer quality score.")
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)


# ============================================================================
# 4. Suspicion Scoring & Session State
# ============================================================================

class SuspicionSeverity(str, Enum):
    """Risk tiers for suspicion evaluation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SuspicionScore(BaseModel):
    """Explainable proctoring assessment score."""

    timestamp: float = Field(description="Timestamp of score generation in seconds.")
    score: float = Field(ge=0.0, le=100.0, description="Composite suspicion index (0-100).")
    severity: SuspicionSeverity
    confidence: float = Field(ge=0.0, le=1.0, description="Reliability of assessment.")
    contributing_events: list[str] = Field(
        default_factory=list,
        description="Names of events contributing to the score.",
    )
    explanation_summary: str = Field(description="Human-readable explanation of findings.")
    mitigating_factors: list[str] = Field(
        default_factory=list,
        description="Contextual observations that reduced suspicion.",
    )


class SessionState(BaseModel):
    """Living state of an interview/assessment session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str
    created_at: float = Field(default_factory=time.time)
    active_question_id: str | None = None
    is_active: bool = True
    total_frames_processed: int = 0
    events_recorded: list[BehaviorEvent] = Field(default_factory=list)
    latest_suspicion: SuspicionScore | None = None
