"""Unit tests for domain schemas, validations, and serialization."""

import pytest
from pydantic import ValidationError

from intelliscreen.core.schemas import (
    AnswerEvaluation,
    BehaviorEvent,
    BoundingBox,
    EventType,
    FrameObservation,
    GazeDirection,
    GazeVector,
    HeadPoseAngles,
    SessionState,
    SuspicionScore,
    SuspicionSeverity,
)


def test_bounding_box_geometry():
    """Verify BoundingBox properties and validation."""
    box = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0, confidence=0.9)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.center == (30.0, 50.0)

    # Inverted x coordinates should raise ValidationError
    with pytest.raises(ValidationError):
        BoundingBox(xmin=60.0, ymin=20.0, xmax=50.0, ymax=80.0)

    # Inverted y coordinates should raise ValidationError
    with pytest.raises(ValidationError):
        BoundingBox(xmin=10.0, ymin=90.0, xmax=50.0, ymax=80.0)


def test_head_pose_deviation():
    """Verify HeadPose deviation logic."""
    centered_pose = HeadPoseAngles(yaw=5.0, pitch=2.0, roll=0.0)
    assert not centered_pose.is_looking_away(yaw_threshold=30.0, pitch_threshold=25.0)

    turned_pose_right = HeadPoseAngles(yaw=35.0, pitch=0.0, roll=0.0)
    assert turned_pose_right.is_looking_away(yaw_threshold=30.0, pitch_threshold=25.0)

    turned_pose_left = HeadPoseAngles(yaw=-35.0, pitch=0.0, roll=0.0)
    assert turned_pose_left.is_looking_away(yaw_threshold=30.0, pitch_threshold=25.0)

    downward_tilt = HeadPoseAngles(yaw=0.0, pitch=-30.0, roll=0.0)
    assert downward_tilt.is_looking_away(yaw_threshold=30.0, pitch_threshold=25.0)


def test_gaze_vector_deviation():
    """Verify GazeVector deviation logic."""
    center_gaze = GazeVector(yaw=5.0, pitch=-4.0, direction=GazeDirection.CENTER)
    assert not center_gaze.is_deviated(yaw_threshold=25.0, pitch_threshold=20.0)

    deviated_gaze = GazeVector(yaw=28.0, pitch=-5.0, direction=GazeDirection.RIGHT)
    assert deviated_gaze.is_deviated(yaw_threshold=25.0, pitch_threshold=20.0)


def test_frame_observation_serialization(sample_frame_observation: FrameObservation):
    """Verify FrameObservation serialization and deserialization."""
    data = sample_frame_observation.model_dump()
    assert data["frame_index"] == 120
    assert data["face_count"] == 1
    assert data["primary_face_box"]["confidence"] == 0.95
    assert len(data["detected_objects"]) == 1

    # Roundtrip from json
    json_str = sample_frame_observation.model_dump_json()
    reconstructed = FrameObservation.model_validate_json(json_str)
    assert reconstructed.frame_index == sample_frame_observation.frame_index
    assert reconstructed.primary_face_box.xmax == sample_frame_observation.primary_face_box.xmax


def test_behavior_event_duration_validation():
    """Verify BehaviorEvent duration reconciliation and negative time rejection."""
    event = BehaviorEvent(
        event_type=EventType.PHONE_DETECTED,
        start_time=10.0,
        end_time=13.5,
        duration=0.0,  # Will be reconciled to 3.5
        confidence=0.88,
        source="object_detector",
    )
    assert event.duration == 3.5

    # End time before start time must fail
    with pytest.raises(ValidationError):
        BehaviorEvent(
            event_type=EventType.FACE_MISSING,
            start_time=15.0,
            end_time=12.0,
            duration=3.0,
            confidence=0.9,
            source="face_detector",
        )


def test_answer_evaluation_schema():
    """Verify AnswerEvaluation score constraints."""
    eval_result = AnswerEvaluation(
        question_id="os_001",
        candidate_answer="A process has separate memory while threads share memory.",
        semantic_similarity=0.88,
        concept_coverage=0.75,
        completeness=0.80,
        overall_score=81.0,
        matched_concepts=["separate memory", "share memory"],
        missing_concepts=["context switch"],
    )
    assert eval_result.overall_score == 81.0
    assert len(eval_result.matched_concepts) == 2

    # Score out of bounds
    with pytest.raises(ValidationError):
        AnswerEvaluation(
            question_id="os_001",
            candidate_answer="...",
            semantic_similarity=1.5,  # Invalid: > 1.0
            concept_coverage=0.5,
            completeness=0.5,
            overall_score=50.0,
        )


def test_session_state_lifecycle(sample_session_state: SessionState, sample_behavior_event: BehaviorEvent):
    """Verify SessionState recording events and updating suspicion."""
    assert sample_session_state.is_active is True
    assert len(sample_session_state.events_recorded) == 0

    sample_session_state.events_recorded.append(sample_behavior_event)
    sample_session_state.total_frames_processed += 100

    suspicion = SuspicionScore(
        timestamp=16.5,
        score=72.0,
        severity=SuspicionSeverity.HIGH,
        confidence=0.85,
        contributing_events=["PROLONGED_OFF_SCREEN_GAZE"],
        explanation_summary="Prolonged gaze deviation detected for 4.5 seconds to the left.",
        mitigating_factors=[],
    )
    sample_session_state.latest_suspicion = suspicion

    assert len(sample_session_state.events_recorded) == 1
    assert sample_session_state.total_frames_processed == 100
    assert sample_session_state.latest_suspicion.severity == SuspicionSeverity.HIGH
