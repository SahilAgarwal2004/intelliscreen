"""Unit tests for the Temporal Event Engine and Sliding Window Buffer."""

import pytest

from intelliscreen.core.exceptions import TemporalEngineError
from intelliscreen.core.schemas import (
    BehaviorEvent,
    BoundingBox,
    DetectedObject,
    EventType,
    FrameObservation,
    GazeDirection,
    GazeVector,
    HeadPoseAngles,
)
from intelliscreen.temporal.engine import (
    TemporalEngineConfig,
    TemporalEventEngine,
)
from intelliscreen.temporal.trackers import (
    FaceAbsenceTracker,
    GazeDeviationTracker,
    HeadTurnTracker,
    MultipleFacesTracker,
    PhoneDetectionTracker,
)
from intelliscreen.temporal.window import (
    SlidingWindowBuffer,
    TemporalWindowSummary,
)


# ============================================================================
# Helper Factories
# ============================================================================

def make_observation(
    frame_index: int,
    timestamp_ms: float,
    face_detected: bool = True,
    face_count: int = 1,
    yaw: float = 0.0,
    pitch: float = 0.0,
    gaze_yaw: float = 0.0,
    gaze_pitch: float = 0.0,
    gaze_direction: GazeDirection = GazeDirection.CENTER,
    detected_objects: list[DetectedObject] | None = None,
) -> FrameObservation:
    """Helper to generate mock FrameObservation instances."""
    primary_box = (
        BoundingBox(xmin=100.0, ymin=100.0, xmax=300.0, ymax=300.0, confidence=0.95)
        if face_detected
        else None
    )
    head_pose = (
        HeadPoseAngles(yaw=yaw, pitch=pitch, roll=0.0, confidence=0.9)
        if face_detected
        else None
    )
    gaze = (
        GazeVector(
            yaw=gaze_yaw,
            pitch=gaze_pitch,
            direction=gaze_direction,
            confidence=0.9,
        )
        if face_detected
        else None
    )
    return FrameObservation(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        face_detected=face_detected,
        face_count=face_count,
        primary_face_box=primary_box,
        head_pose=head_pose,
        gaze=gaze,
        detected_objects=detected_objects or [],
    )


# ============================================================================
# 1. Sliding Window Buffer Tests
# ============================================================================

class TestSlidingWindowBuffer:
    """Tests for time-bounded window maintenance and summary statistics."""

    def test_window_pruning(self) -> None:
        buffer = SlidingWindowBuffer(window_duration_sec=5.0)

        # Ingest 10 observations spaced 1 second apart (0.0s to 9.0s)
        for i in range(10):
            obs = make_observation(frame_index=i, timestamp_ms=float(i * 1000))
            buffer.add_observation(obs)

        # Current time is 9.0s, window is 5.0s -> records before 4.0s should be pruned
        obs_in_window = buffer.get_observations()
        timestamps = [o.timestamp_ms / 1000.0 for o in obs_in_window]

        assert min(timestamps) >= 4.0
        assert max(timestamps) == 9.0

    def test_event_duration_and_counts(self) -> None:
        buffer = SlidingWindowBuffer(window_duration_sec=30.0)

        ev1 = BehaviorEvent(
            event_type=EventType.PHONE_DETECTED,
            start_time=5.0,
            end_time=8.5,
            duration=3.5,
            confidence=0.9,
            source="test",
        )
        ev2 = BehaviorEvent(
            event_type=EventType.PHONE_DETECTED,
            start_time=12.0,
            end_time=14.0,
            duration=2.0,
            confidence=0.85,
            source="test",
        )
        ev3 = BehaviorEvent(
            event_type=EventType.LOOKING_LEFT,
            start_time=15.0,
            end_time=19.0,
            duration=4.0,
            confidence=0.88,
            source="test",
        )

        buffer.add_event(ev1)
        buffer.add_event(ev2)
        buffer.add_event(ev3)

        assert buffer.get_cumulative_duration(EventType.PHONE_DETECTED) == 5.5
        assert buffer.get_cumulative_duration(EventType.LOOKING_LEFT) == 4.0
        assert buffer.get_cumulative_duration(EventType.FACE_MISSING) == 0.0

        counts = buffer.get_event_counts()
        assert counts[EventType.PHONE_DETECTED.value] == 2
        assert counts[EventType.LOOKING_LEFT.value] == 1

    def test_window_summary_critical_flag(self) -> None:
        buffer = SlidingWindowBuffer(window_duration_sec=10.0)

        # Ingest observations with a detected phone
        phone_box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=100)
        phone_obj = DetectedObject(label="cell phone", confidence=0.85, box=phone_box)

        for i in range(10):
            obs = make_observation(
                frame_index=i,
                timestamp_ms=float(i * 1000),
                detected_objects=[phone_obj] if i >= 5 else [],
            )
            buffer.add_observation(obs)

        summary = buffer.get_summary()
        assert isinstance(summary, TemporalWindowSummary)
        assert summary.total_frames == 10
        assert summary.phone_detected_ratio == 0.5
        assert summary.is_critical_flag is True


# ============================================================================
# 2. Condition Tracker Tests
# ============================================================================

class TestConditionTrackers:
    """Tests for discrete condition state machines and hysteresis timers."""

    def test_face_absence_and_possible_absence(self) -> None:
        tracker = FaceAbsenceTracker(min_duration_sec=3.0, extended_absence_sec=10.0)

        # Absence for 2.0s (< 3.0s threshold) -> no event
        evs = tracker.step(time_sec=0.0, active=True)
        assert len(evs) == 0
        evs = tracker.step(time_sec=2.0, active=True)
        assert len(evs) == 0

        # Absence reaches 3.5s (>= 3.0s threshold) -> FACE_MISSING triggered
        evs = tracker.step(time_sec=3.5, active=True)
        assert len(evs) == 1
        assert evs[0].event_type == EventType.FACE_MISSING
        assert evs[0].start_time == 0.0
        assert evs[0].duration == 3.5

        # Absence reaches 10.5s (>= 10.0s threshold) -> POSSIBLE_ABSENCE triggered
        evs = tracker.step(time_sec=10.5, active=True)
        assert len(evs) == 1
        assert evs[0].event_type == EventType.POSSIBLE_ABSENCE

        # Candidate returns at 11.0s (condition inactive) -> debounces and finalizes
        tracker.step(time_sec=11.0, active=False)
        tracker.step(time_sec=11.1, active=False)
        final_evs = tracker.step(time_sec=11.2, active=False)  # Debounce limit exceeded
        assert len(final_evs) == 1
        assert final_evs[0].event_type == EventType.FACE_MISSING
        assert final_evs[0].duration == 10.5
        assert final_evs[0].metadata["finalized"] is True

    def test_phone_detection_tracker_with_debounce(self) -> None:
        tracker = PhoneDetectionTracker(min_duration_sec=2.0, debounce_frames=2)

        # Phone visible from 0.0s to 2.5s
        tracker.step(time_sec=0.0, active=True)
        tracker.step(time_sec=1.0, active=True)
        evs = tracker.step(time_sec=2.5, active=True)
        assert len(evs) == 1
        assert evs[0].event_type == EventType.PHONE_DETECTED

        # Temporary single-frame drop (e.g. motion blur occlusion)
        drop_evs = tracker.step(time_sec=2.6, active=False)
        assert len(drop_evs) == 0
        assert tracker.is_in_streak is True

        # Returns on next frame
        re_evs = tracker.step(time_sec=2.7, active=True)
        assert len(re_evs) == 0

        # Disappears completely
        tracker.step(time_sec=3.0, active=False)
        tracker.step(time_sec=3.1, active=False)
        fin = tracker.step(time_sec=3.2, active=False)
        assert len(fin) == 1
        assert fin[0].event_type == EventType.PHONE_DETECTED
        assert fin[0].duration == 2.7

    def test_multiple_faces_tracker(self) -> None:
        tracker = MultipleFacesTracker(min_duration_sec=1.0)
        tracker.step(time_sec=0.0, active=True)
        evs = tracker.step(time_sec=1.2, active=True)
        assert len(evs) == 1
        assert evs[0].event_type == EventType.MULTIPLE_FACES

    def test_gaze_deviation_directional_and_prolonged(self) -> None:
        tracker = GazeDeviationTracker(
            prolonged_threshold_sec=4.0,
            directional_threshold_sec=2.0,
        )

        # Glance away for 1.0s (normal thinking time) -> no event
        evs0 = tracker.step(time_sec=0.0, is_deviated=True, direction=GazeDirection.LEFT)
        evs1 = tracker.step(time_sec=1.0, is_deviated=True, direction=GazeDirection.LEFT)
        assert len(evs0) == 0
        assert len(evs1) == 0

        # Continuous left glance reaches 2.5s (>= 2.0s directional threshold)
        evs2 = tracker.step(time_sec=2.5, is_deviated=True, direction=GazeDirection.LEFT)
        assert len(evs2) == 1
        assert evs2[0].event_type == EventType.LOOKING_LEFT

        # Glancing continues to 4.5s (>= 4.0s prolonged threshold)
        evs3 = tracker.step(time_sec=4.5, is_deviated=True, direction=GazeDirection.LEFT)
        assert len(evs3) == 1
        assert evs3[0].event_type == EventType.PROLONGED_OFF_SCREEN_GAZE

    def test_head_turn_tracker(self) -> None:
        tracker = HeadTurnTracker(min_duration_sec=2.0)
        tracker.step(time_sec=0.0, active=True)
        evs = tracker.step(time_sec=2.2, active=True)
        assert len(evs) == 1
        assert evs[0].event_type == EventType.HEAD_TURN


# ============================================================================
# 3. Unified Temporal Event Engine Tests
# ============================================================================

class TestTemporalEventEngine:
    """Tests for integrated observation stream processing and session lifecycle."""

    def test_engine_initialization(self) -> None:
        config = TemporalEngineConfig(
            face_absence_alert_sec=2.5,
            phone_presence_alert_sec=1.5,
        )
        engine = TemporalEventEngine(config=config)
        assert engine.config.face_absence_alert_sec == 2.5
        assert engine.config.phone_presence_alert_sec == 1.5

    def test_normal_interview_sequence_no_alerts(self) -> None:
        """Verify standard attentive candidate produces zero suspicious events."""
        engine = TemporalEventEngine()

        # 30 frames @ 10fps (3.0 seconds of looking forward)
        for i in range(30):
            obs = make_observation(
                frame_index=i,
                timestamp_ms=float(i * 100),
                face_detected=True,
                yaw=0.0,
                pitch=0.0,
                gaze_yaw=0.0,
                gaze_pitch=0.0,
                gaze_direction=GazeDirection.CENTER,
            )
            events = engine.process_observation(obs)
            assert len(events) == 0

        assert len(engine.get_all_events()) == 0
        summary = engine.get_window_summary()
        assert summary.total_frames == 30
        assert summary.face_absence_ratio == 0.0
        assert summary.is_critical_flag is False

    def test_phone_presence_detection_stream(self) -> None:
        """Verify phone held in view for 2.5s triggers PHONE_DETECTED."""
        engine = TemporalEventEngine(
            config=TemporalEngineConfig(phone_presence_alert_sec=2.0)
        )

        phone_box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=100)
        phone_obj = DetectedObject(label="cell phone", confidence=0.92, box=phone_box)

        triggered_events: list[BehaviorEvent] = []

        # 30 frames @ 10fps (3.0 seconds total)
        for i in range(30):
            has_phone = i >= 5  # Phone appears from frame 5 (0.5s) to frame 30 (3.0s) -> 2.5s duration
            obs = make_observation(
                frame_index=i,
                timestamp_ms=float(i * 100),
                detected_objects=[phone_obj] if has_phone else [],
            )
            evs = engine.process_observation(obs)
            triggered_events.extend(evs)

        # Phone was present for 2.5s, threshold is 2.0s -> PHONE_DETECTED must have fired
        phone_alerts = [e for e in triggered_events if e.event_type == EventType.PHONE_DETECTED]
        assert len(phone_alerts) >= 1
        assert phone_alerts[0].confidence > 0.8

    def test_non_monotonic_timestamp_error(self) -> None:
        """Verify engine rejects observations that travel backwards in time."""
        engine = TemporalEventEngine()
        obs1 = make_observation(frame_index=1, timestamp_ms=5000.0)
        obs2 = make_observation(frame_index=2, timestamp_ms=4000.0)  # Earlier timestamp!

        engine.process_observation(obs1)
        with pytest.raises(TemporalEngineError):
            engine.process_observation(obs2)

    def test_flush_finalization(self) -> None:
        """Verify flush closes active streaks at interview completion."""
        engine = TemporalEventEngine(
            config=TemporalEngineConfig(phone_presence_alert_sec=1.0)
        )

        phone_obj = DetectedObject(
            label="cell phone",
            confidence=0.9,
            box=BoundingBox(xmin=1, ymin=1, xmax=10, ymax=10),
        )

        for i in range(15):  # 1.5 seconds of phone
            obs = make_observation(
                frame_index=i,
                timestamp_ms=float(i * 100),
                detected_objects=[phone_obj],
            )
            engine.process_observation(obs)

        # Event fired, but hasn't finalized because phone was still visible
        flushed = engine.flush(current_time_sec=2.0)
        finalized_phones = [
            e for e in flushed
            if e.event_type == EventType.PHONE_DETECTED and e.metadata.get("finalized")
        ]
        assert len(finalized_phones) == 1
        assert finalized_phones[0].end_time == 2.0

    def test_engine_reset(self) -> None:
        """Verify reset clears all buffers, trackers, and history."""
        engine = TemporalEventEngine()
        obs = make_observation(frame_index=0, timestamp_ms=0.0)
        engine.process_observation(obs)

        assert engine._total_observations_processed == 1
        engine.reset()
        assert engine._total_observations_processed == 0
        assert len(engine.get_all_events()) == 0
        assert engine.get_window_summary().total_frames == 0
