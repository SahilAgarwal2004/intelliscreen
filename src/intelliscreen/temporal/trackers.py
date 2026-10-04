"""Behavioral condition trackers and temporal state machines for IntelliScreen."""

from dataclasses import dataclass, field
import uuid
from typing import Any

from intelliscreen.core.schemas import (
    BehaviorEvent,
    EventType,
    GazeDirection,
)


@dataclass
class ConditionStreak:
    """Internal state tracking an ongoing contiguous temporal condition."""

    start_time: float
    last_active_time: float
    consecutive_frames: int = 0
    gap_count: int = 0
    alert_fired: bool = False
    confidence_samples: list[float] = field(default_factory=list)
    metadata_accumulator: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_active_time - self.start_time)

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_samples:
            return 1.0
        return round(sum(self.confidence_samples) / len(self.confidence_samples), 2)


class BaseConditionTracker:
    """Generic temporal tracker with hysteresis debouncing and event finalization."""

    def __init__(
        self,
        event_type: EventType,
        min_duration_sec: float,
        debounce_frames: int = 2,
        source: str = "temporal_engine",
    ) -> None:
        self.event_type = event_type
        self.min_duration_sec = float(min_duration_sec)
        self.debounce_frames = int(debounce_frames)
        self.source = source

        self._streak: ConditionStreak | None = None
        self._active_event: BehaviorEvent | None = None

    @property
    def is_in_streak(self) -> bool:
        return self._streak is not None

    @property
    def is_alert_active(self) -> bool:
        return self._streak is not None and self._streak.alert_fired

    def step(
        self,
        time_sec: float,
        active: bool,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[BehaviorEvent]:
        """Update tracker state with current frame status and return newly triggered/finalized events.

        Args:
            time_sec: Elapsed session time in seconds.
            active: True if condition is satisfied in the current frame.
            confidence: Frame observation confidence.
            metadata: Additional contextual properties.

        Returns:
            list[BehaviorEvent]: Any events triggered or finalized during this step.
        """
        emitted: list[BehaviorEvent] = []
        meta = metadata or {}

        if active:
            if self._streak is None:
                # Start new condition streak
                self._streak = ConditionStreak(
                    start_time=time_sec,
                    last_active_time=time_sec,
                    consecutive_frames=1,
                    gap_count=0,
                    alert_fired=False,
                    confidence_samples=[confidence],
                    metadata_accumulator=dict(meta),
                )
            else:
                # Extend existing streak
                self._streak.last_active_time = time_sec
                self._streak.consecutive_frames += 1
                self._streak.gap_count = 0
                self._streak.confidence_samples.append(confidence)
                self._streak.metadata_accumulator.update(meta)

            # Check if duration threshold has been reached for the first time
            if (
                not self._streak.alert_fired
                and self._streak.duration >= self.min_duration_sec
            ):
                self._streak.alert_fired = True
                ev = BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=self.event_type,
                    start_time=round(self._streak.start_time, 3),
                    end_time=round(self._streak.last_active_time, 3),
                    duration=round(self._streak.duration, 3),
                    confidence=self._streak.avg_confidence,
                    source=self.source,
                    metadata=dict(self._streak.metadata_accumulator),
                )
                self._active_event = ev
                emitted.append(ev)

        else:
            # Condition is inactive in current frame
            if self._streak is not None:
                self._streak.gap_count += 1
                if self._streak.gap_count > self.debounce_frames:
                    # Debounce threshold exceeded: finalize and terminate streak
                    if self._streak.alert_fired:
                        final_ev = BehaviorEvent(
                            event_id=self._active_event.event_id if self._active_event else str(uuid.uuid4()),
                            event_type=self.event_type,
                            start_time=round(self._streak.start_time, 3),
                            end_time=round(self._streak.last_active_time, 3),
                            duration=round(self._streak.duration, 3),
                            confidence=self._streak.avg_confidence,
                            source=self.source,
                            metadata={
                                **self._streak.metadata_accumulator,
                                "finalized": True,
                                "total_frames": self._streak.consecutive_frames,
                            },
                        )
                        emitted.append(final_ev)

                    self._streak = None
                    self._active_event = None

        return emitted

    def flush(self, current_time_sec: float) -> list[BehaviorEvent]:
        """Finalize any ongoing active streak upon session completion."""
        emitted: list[BehaviorEvent] = []
        if self._streak is not None and self._streak.alert_fired:
            end_t = max(self._streak.start_time, current_time_sec)
            dur = round(end_t - self._streak.start_time, 3)
            final_ev = BehaviorEvent(
                event_id=self._active_event.event_id if self._active_event else str(uuid.uuid4()),
                event_type=self.event_type,
                start_time=round(self._streak.start_time, 3),
                end_time=round(end_t, 3),
                duration=dur,
                confidence=self._streak.avg_confidence,
                source=self.source,
                metadata={
                    **self._streak.metadata_accumulator,
                    "finalized": True,
                    "flushed": True,
                },
            )
            emitted.append(final_ev)

        self._streak = None
        self._active_event = None
        return emitted

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._streak = None
        self._active_event = None


class FaceAbsenceTracker(BaseConditionTracker):
    """Tracks continuous face absence and escalates to possible candidate absence."""

    def __init__(
        self,
        min_duration_sec: float = 3.0,
        extended_absence_sec: float = 10.0,
        debounce_frames: int = 2,
    ) -> None:
        super().__init__(
            event_type=EventType.FACE_MISSING,
            min_duration_sec=min_duration_sec,
            debounce_frames=debounce_frames,
            source="face_absence_engine",
        )
        self.extended_absence_sec = float(extended_absence_sec)
        self._extended_absence_fired: bool = False

    def step(
        self,
        time_sec: float,
        active: bool,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[BehaviorEvent]:
        emitted = super().step(time_sec, active, confidence, metadata)

        # Check for secondary escalation to POSSIBLE_ABSENCE
        if self._streak is not None and active:
            if (
                not self._extended_absence_fired
                and self._streak.duration >= self.extended_absence_sec
            ):
                self._extended_absence_fired = True
                ev = BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.POSSIBLE_ABSENCE,
                    start_time=round(self._streak.start_time, 3),
                    end_time=round(self._streak.last_active_time, 3),
                    duration=round(self._streak.duration, 3),
                    confidence=1.0,
                    source="face_absence_engine",
                    metadata={
                        **(metadata or {}),
                        "escalation": "extended_face_absence",
                    },
                )
                emitted.append(ev)

        if not active and self._streak is None:
            self._extended_absence_fired = False

        return emitted

    def reset(self) -> None:
        super().reset()
        self._extended_absence_fired = False


class MultipleFacesTracker(BaseConditionTracker):
    """Tracks continuous presence of secondary faces in the video frame."""

    def __init__(
        self,
        min_duration_sec: float = 1.0,
        debounce_frames: int = 2,
    ) -> None:
        super().__init__(
            event_type=EventType.MULTIPLE_FACES,
            min_duration_sec=min_duration_sec,
            debounce_frames=debounce_frames,
            source="multiple_faces_engine",
        )


class PhoneDetectionTracker(BaseConditionTracker):
    """Tracks continuous presence of unauthorized cell phone devices."""

    def __init__(
        self,
        min_duration_sec: float = 2.0,
        debounce_frames: int = 2,
    ) -> None:
        super().__init__(
            event_type=EventType.PHONE_DETECTED,
            min_duration_sec=min_duration_sec,
            debounce_frames=debounce_frames,
            source="unauthorized_device_engine",
        )


class HeadTurnTracker(BaseConditionTracker):
    """Tracks continuous abnormal head yaw or pitch rotation."""

    def __init__(
        self,
        min_duration_sec: float = 2.0,
        debounce_frames: int = 2,
    ) -> None:
        super().__init__(
            event_type=EventType.HEAD_TURN,
            min_duration_sec=min_duration_sec,
            debounce_frames=debounce_frames,
            source="head_turn_engine",
        )


class GazeDeviationTracker:
    """Tracks continuous gaze screen deviation and directional classification."""

    def __init__(
        self,
        prolonged_threshold_sec: float = 4.0,
        directional_threshold_sec: float = 2.0,
        debounce_frames: int = 2,
    ) -> None:
        self.prolonged_tracker = BaseConditionTracker(
            event_type=EventType.PROLONGED_OFF_SCREEN_GAZE,
            min_duration_sec=prolonged_threshold_sec,
            debounce_frames=debounce_frames,
            source="gaze_engine",
        )
        self.directional_threshold_sec = directional_threshold_sec
        self.debounce_frames = debounce_frames

        self._active_direction: GazeDirection = GazeDirection.CENTER
        self._direction_streak: ConditionStreak | None = None

    def step(
        self,
        time_sec: float,
        is_deviated: bool,
        direction: GazeDirection,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[BehaviorEvent]:
        """Update both prolonged deviation tracker and directional trackers."""
        emitted: list[BehaviorEvent] = []
        meta = metadata or {}

        # 1. Prolonged Off-Screen Gaze Tracker
        prolonged_events = self.prolonged_tracker.step(
            time_sec=time_sec,
            active=is_deviated,
            confidence=confidence,
            metadata={**meta, "direction": direction.value},
        )
        emitted.extend(prolonged_events)

        # 2. Directional Gaze Tracking (LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN)
        dir_map: dict[GazeDirection, EventType] = {
            GazeDirection.LEFT: EventType.LOOKING_LEFT,
            GazeDirection.RIGHT: EventType.LOOKING_RIGHT,
            GazeDirection.UP: EventType.LOOKING_UP,
            GazeDirection.DOWN: EventType.LOOKING_DOWN,
        }

        target_event_type = dir_map.get(direction)

        if is_deviated and target_event_type is not None:
            if (
                self._direction_streak is None
                or self._active_direction != direction
            ):
                # Started new directional streak
                self._active_direction = direction
                self._direction_streak = ConditionStreak(
                    start_time=time_sec,
                    last_active_time=time_sec,
                    consecutive_frames=1,
                    gap_count=0,
                    alert_fired=False,
                    confidence_samples=[confidence],
                    metadata_accumulator=dict(meta),
                )
            else:
                self._direction_streak.last_active_time = time_sec
                self._direction_streak.consecutive_frames += 1
                self._direction_streak.gap_count = 0
                self._direction_streak.confidence_samples.append(confidence)
                self._direction_streak.metadata_accumulator.update(meta)

            if (
                not self._direction_streak.alert_fired
                and self._direction_streak.duration >= self.directional_threshold_sec
            ):
                self._direction_streak.alert_fired = True
                ev = BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=target_event_type,
                    start_time=round(self._direction_streak.start_time, 3),
                    end_time=round(self._direction_streak.last_active_time, 3),
                    duration=round(self._direction_streak.duration, 3),
                    confidence=self._direction_streak.avg_confidence,
                    source="gaze_engine",
                    metadata={
                        **self._direction_streak.metadata_accumulator,
                        "direction": direction.value,
                    },
                )
                emitted.append(ev)
        else:
            if self._direction_streak is not None:
                self._direction_streak.gap_count += 1
                if self._direction_streak.gap_count > self.debounce_frames:
                    if self._direction_streak.alert_fired and target_event_type is not None:
                        final_ev = BehaviorEvent(
                            event_id=str(uuid.uuid4()),
                            event_type=target_event_type,
                            start_time=round(self._direction_streak.start_time, 3),
                            end_time=round(self._direction_streak.last_active_time, 3),
                            duration=round(self._direction_streak.duration, 3),
                            confidence=self._direction_streak.avg_confidence,
                            source="gaze_engine",
                            metadata={
                                **self._direction_streak.metadata_accumulator,
                                "finalized": True,
                            },
                        )
                        emitted.append(final_ev)

                    self._direction_streak = None
                    self._active_direction = GazeDirection.CENTER

        return emitted

    def flush(self, current_time_sec: float) -> list[BehaviorEvent]:
        """Finalize any ongoing gaze deviation events."""
        emitted = self.prolonged_tracker.flush(current_time_sec)

        dir_map: dict[GazeDirection, EventType] = {
            GazeDirection.LEFT: EventType.LOOKING_LEFT,
            GazeDirection.RIGHT: EventType.LOOKING_RIGHT,
            GazeDirection.UP: EventType.LOOKING_UP,
            GazeDirection.DOWN: EventType.LOOKING_DOWN,
        }

        if (
            self._direction_streak is not None
            and self._direction_streak.alert_fired
            and self._active_direction in dir_map
        ):
            ev_type = dir_map[self._active_direction]
            end_t = max(self._direction_streak.start_time, current_time_sec)
            final_ev = BehaviorEvent(
                event_id=str(uuid.uuid4()),
                event_type=ev_type,
                start_time=round(self._direction_streak.start_time, 3),
                end_time=round(end_t, 3),
                duration=round(end_t - self._direction_streak.start_time, 3),
                confidence=self._direction_streak.avg_confidence,
                source="gaze_engine",
                metadata={
                    **self._direction_streak.metadata_accumulator,
                    "finalized": True,
                    "flushed": True,
                },
            )
            emitted.append(final_ev)

        self._direction_streak = None
        self._active_direction = GazeDirection.CENTER
        return emitted

    def reset(self) -> None:
        """Reset gaze trackers."""
        self.prolonged_tracker.reset()
        self._direction_streak = None
        self._active_direction = GazeDirection.CENTER
