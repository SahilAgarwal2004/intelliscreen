"""Pytest fixtures for IntelliScreen unit and integration tests."""

import pytest

from intelliscreen.config.settings import Settings
from intelliscreen.core.schemas import (
    BehaviorEvent,
    BoundingBox,
    DetectedObject,
    EventType,
    FrameObservation,
    GazeDirection,
    GazeVector,
    HeadPoseAngles,
    QuestionContext,
    SessionState,
    SuspicionScore,
    SuspicionSeverity,
)


@pytest.fixture
def sample_settings() -> Settings:
    """Fixture returning baseline default settings."""
    return Settings()


@pytest.fixture
def sample_bounding_box() -> BoundingBox:
    """Fixture returning a valid normalized bounding box."""
    return BoundingBox(
        xmin=0.2,
        ymin=0.15,
        xmax=0.6,
        ymax=0.75,
        confidence=0.95,
        label="face",
    )


@pytest.fixture
def sample_head_pose() -> HeadPoseAngles:
    """Fixture returning centered head pose."""
    return HeadPoseAngles(
        yaw=2.5,
        pitch=-1.0,
        roll=0.5,
        confidence=0.98,
    )


@pytest.fixture
def sample_gaze_vector() -> GazeVector:
    """Fixture returning forward-looking gaze vector."""
    return GazeVector(
        yaw=-1.2,
        pitch=0.8,
        direction=GazeDirection.CENTER,
        confidence=0.94,
    )


@pytest.fixture
def sample_frame_observation(
    sample_bounding_box: BoundingBox,
    sample_head_pose: HeadPoseAngles,
    sample_gaze_vector: GazeVector,
) -> FrameObservation:
    """Fixture returning a complete single frame observation."""
    return FrameObservation(
        frame_index=120,
        timestamp_ms=4000.0,
        face_detected=True,
        face_count=1,
        primary_face_box=sample_bounding_box,
        head_pose=sample_head_pose,
        gaze=sample_gaze_vector,
        detected_objects=[
            DetectedObject(
                label="cell phone",
                confidence=0.89,
                box=BoundingBox(xmin=0.8, ymin=0.7, xmax=0.95, ymax=0.95, confidence=0.89),
            )
        ],
    )


@pytest.fixture
def sample_behavior_event() -> BehaviorEvent:
    """Fixture returning a valid prolonged gaze event."""
    return BehaviorEvent(
        event_type=EventType.PROLONGED_OFF_SCREEN_GAZE,
        start_time=12.0,
        end_time=16.5,
        duration=4.5,
        confidence=0.91,
        source="gaze_engine",
        metadata={"direction": "LEFT", "max_yaw": 38.2},
    )


@pytest.fixture
def sample_question_context() -> QuestionContext:
    """Fixture returning a sample OS interview question."""
    return QuestionContext(
        question_id="os_001",
        question="What is the difference between a process and a thread?",
        topic="Operating Systems",
        difficulty="medium",
        reference_answer=(
            "A process is an executing instance of a program with its own dedicated memory space, "
            "whereas a thread is a lightweight unit of execution within a process that shares memory and resources."
        ),
        expected_concepts=["memory space", "lightweight", "shared resources", "context switch"],
    )


@pytest.fixture
def sample_session_state() -> SessionState:
    """Fixture returning an active interview session."""
    return SessionState(
        candidate_id="cand_test_99",
        active_question_id="os_001",
    )
