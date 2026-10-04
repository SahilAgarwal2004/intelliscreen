"""Unified Temporal Event Engine for IntelliScreen.

Transforms discrete, noisy FrameObservation perception streams into
validated, aggregated BehaviorEvents and rolling statistical window metrics.
"""

from dataclasses import dataclass
import time
from typing import Any

from intelliscreen.config.settings import get_settings
from intelliscreen.core.exceptions import TemporalEngineError
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import (
    BehaviorEvent,
    EventType,
    FrameObservation,
    GazeDirection,
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

logger = get_logger("temporal_engine")


@dataclass
class TemporalEngineConfig:
    """Configurable timing thresholds and hysteresis parameters for TemporalEventEngine."""

    face_absence_alert_sec: float = 3.0
    extended_absence_sec: float = 10.0
    multiple_faces_alert_sec: float = 1.0
    phone_presence_alert_sec: float = 2.0
    prolonged_look_away_sec: float = 4.0
    directional_look_away_sec: float = 2.0
    head_turn_sec: float = 2.0
    sliding_window_sec: float = 30.0
    debounce_frames: int = 2

    @classmethod
    def from_settings(cls) -> "TemporalEngineConfig":
        """Instantiate configuration directly from centralized system settings."""
        settings = get_settings()
        return cls(
            face_absence_alert_sec=settings.temporal_face_absence_alert_sec,
            multiple_faces_alert_sec=settings.temporal_multiple_faces_alert_sec,
            phone_presence_alert_sec=settings.temporal_phone_presence_alert_sec,
            prolonged_look_away_sec=settings.temporal_suspicious_look_away_min_sec,
            sliding_window_sec=settings.temporal_sliding_window_sec,
        )


class TemporalEventEngine:
    """Stateful temporal event engine aggregating FrameObservations into BehaviorEvents."""

    def __init__(self, config: TemporalEngineConfig | None = None) -> None:
        """Initialize condition state trackers and sliding window buffer.

        Args:
            config: Operational temporal parameters.
        """
        self.config = config or TemporalEngineConfig.from_settings()

        # 1. Condition Trackers
        self.absence_tracker = FaceAbsenceTracker(
            min_duration_sec=self.config.face_absence_alert_sec,
            extended_absence_sec=self.config.extended_absence_sec,
            debounce_frames=self.config.debounce_frames,
        )
        self.multiple_faces_tracker = MultipleFacesTracker(
            min_duration_sec=self.config.multiple_faces_alert_sec,
            debounce_frames=self.config.debounce_frames,
        )
        self.phone_tracker = PhoneDetectionTracker(
            min_duration_sec=self.config.phone_presence_alert_sec,
            debounce_frames=self.config.debounce_frames,
        )
        self.gaze_tracker = GazeDeviationTracker(
            prolonged_threshold_sec=self.config.prolonged_look_away_sec,
            directional_threshold_sec=self.config.directional_look_away_sec,
            debounce_frames=self.config.debounce_frames,
        )
        self.head_turn_tracker = HeadTurnTracker(
            min_duration_sec=self.config.head_turn_sec,
            debounce_frames=self.config.debounce_frames,
        )

        # 2. Sliding Window Buffer
        self.window_buffer = SlidingWindowBuffer(
            window_duration_sec=self.config.sliding_window_sec,
        )

        # Session persistence
        self._all_events: list[BehaviorEvent] = []
        self._last_processed_time_sec: float = -1.0
        self._total_observations_processed: int = 0

        logger.info(
            f"TemporalEventEngine initialized (absence_thresh={self.config.face_absence_alert_sec}s, "
            f"phone_thresh={self.config.phone_presence_alert_sec}s, "
            f"look_away_thresh={self.config.prolonged_look_away_sec}s, "
            f"window={self.config.sliding_window_sec}s)"
        )

    def process_observation(self, observation: FrameObservation) -> list[BehaviorEvent]:
        """Ingest a single FrameObservation and return any newly triggered or finalized BehaviorEvents.

        Args:
            observation: Canonical FrameObservation domain object.

        Returns:
            list[BehaviorEvent]: High-level behavioral events emitted during this step.
        """
        t_sec = observation.timestamp_ms / 1000.0

        if self._last_processed_time_sec >= 0.0 and t_sec < self._last_processed_time_sec:
            raise TemporalEngineError(
                f"Non-monotonic timestamp received: {t_sec:.3f}s is earlier than "
                f"previous timestamp {self._last_processed_time_sec:.3f}s."
            )
        self._last_processed_time_sec = t_sec
        self._total_observations_processed += 1

        new_events: list[BehaviorEvent] = []

        # 1. Face Absence & Possible Absence
        is_missing = not observation.face_detected
        absence_events = self.absence_tracker.step(
            time_sec=t_sec,
            active=is_missing,
            confidence=1.0,
            metadata={"frame_index": observation.frame_index},
        )
        new_events.extend(absence_events)

        # 2. Multiple Faces
        has_multiple_faces = observation.face_count > 1
        multi_events = self.multiple_faces_tracker.step(
            time_sec=t_sec,
            active=has_multiple_faces,
            confidence=1.0,
            metadata={
                "frame_index": observation.frame_index,
                "face_count": observation.face_count,
            },
        )
        new_events.extend(multi_events)

        # 3. Cell Phone Presence
        phone_objs = [
            obj for obj in observation.detected_objects
            if obj.label.lower() in ("cell phone", "phone")
        ]
        has_phone = len(phone_objs) > 0
        phone_conf = max((obj.confidence for obj in phone_objs), default=0.0) if has_phone else 0.0
        phone_events = self.phone_tracker.step(
            time_sec=t_sec,
            active=has_phone,
            confidence=phone_conf,
            metadata={
                "frame_index": observation.frame_index,
                "detected_count": len(phone_objs),
            },
        )
        new_events.extend(phone_events)

        # 4. 3D Head Turn
        is_head_turning = (
            observation.head_pose.is_looking_away()
            if observation.head_pose is not None
            else False
        )
        head_meta: dict[str, Any] = {"frame_index": observation.frame_index}
        if observation.head_pose is not None:
            head_meta["yaw"] = observation.head_pose.yaw
            head_meta["pitch"] = observation.head_pose.pitch
            head_meta["roll"] = observation.head_pose.roll
        head_events = self.head_turn_tracker.step(
            time_sec=t_sec,
            active=is_head_turning,
            confidence=observation.head_pose.confidence if observation.head_pose else 1.0,
            metadata=head_meta,
        )
        new_events.extend(head_events)

        # 5. Gaze Screen Deviation & Direction
        is_gaze_deviated = (
            observation.gaze.is_deviated()
            if observation.gaze is not None
            else False
        )
        gaze_direction = (
            observation.gaze.direction
            if observation.gaze is not None
            else GazeDirection.CENTER
        )
        gaze_meta: dict[str, Any] = {
            "frame_index": observation.frame_index,
            "direction": gaze_direction.value,
        }
        if observation.gaze is not None:
            gaze_meta["yaw"] = observation.gaze.yaw
            gaze_meta["pitch"] = observation.gaze.pitch
        gaze_events = self.gaze_tracker.step(
            time_sec=t_sec,
            is_deviated=is_gaze_deviated,
            direction=gaze_direction,
            confidence=observation.gaze.confidence if observation.gaze else 1.0,
            metadata=gaze_meta,
        )
        new_events.extend(gaze_events)

        # Update sliding window and session history
        self.window_buffer.add_observation(observation)
        for ev in new_events:
            self.window_buffer.add_event(ev)
            self._all_events.append(ev)

        return new_events

    def flush(self, current_time_sec: float | None = None) -> list[BehaviorEvent]:
        """Finalize any active condition streaks upon interview completion.

        Args:
            current_time_sec: Session termination timestamp. If None, uses last processed time.

        Returns:
            list[BehaviorEvent]: Any finalized events closed by the flush operation.
        """
        now = (
            current_time_sec
            if current_time_sec is not None
            else max(0.0, self._last_processed_time_sec)
        )
        flushed: list[BehaviorEvent] = []

        flushed.extend(self.absence_tracker.flush(now))
        flushed.extend(self.multiple_faces_tracker.flush(now))
        flushed.extend(self.phone_tracker.flush(now))
        flushed.extend(self.head_turn_tracker.flush(now))
        flushed.extend(self.gaze_tracker.flush(now))

        for ev in flushed:
            self.window_buffer.add_event(ev)
            self._all_events.append(ev)

        return flushed

    def get_active_events(self) -> list[BehaviorEvent]:
        """Return list of currently active behavioral events."""
        active: list[BehaviorEvent] = []
        trackers = [
            self.absence_tracker,
            self.multiple_faces_tracker,
            self.phone_tracker,
            self.head_turn_tracker,
            self.gaze_tracker.prolonged_tracker,
        ]
        for tr in trackers:
            if tr._active_event is not None:
                active.append(tr._active_event)
        return active

    def get_all_events(self) -> list[BehaviorEvent]:
        """Return all behavioral events recorded across the session."""
        return list(self._all_events)

    def get_window_summary(self, current_time_sec: float | None = None) -> TemporalWindowSummary:
        """Retrieve aggregated behavioral metrics over the active sliding window."""
        return self.window_buffer.get_summary(current_time_sec)

    def reset(self) -> None:
        """Reset all trackers, sliding window buffer, and session event logs."""
        self.absence_tracker.reset()
        self.multiple_faces_tracker.reset()
        self.phone_tracker.reset()
        self.gaze_tracker.reset()
        self.head_turn_tracker.reset()
        self.window_buffer.reset()
        self._all_events.clear()
        self._last_processed_time_sec = -1.0
        self._total_observations_processed = 0
