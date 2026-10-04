"""Sliding window buffer and temporal aggregation metrics for IntelliScreen."""

from collections import deque
from dataclasses import dataclass, field
import time

from intelliscreen.config.settings import get_settings
from intelliscreen.core.schemas import (
    BehaviorEvent,
    EventType,
    FrameObservation,
)


@dataclass
class TemporalWindowSummary:
    """Statistical summary of behavioral metrics over a sliding temporal window."""

    window_start_time: float
    window_end_time: float
    window_duration_sec: float
    total_frames: int
    event_counts: dict[str, int] = field(default_factory=dict)
    cumulative_durations: dict[str, float] = field(default_factory=dict)
    face_absence_ratio: float = 0.0
    gaze_deviation_ratio: float = 0.0
    phone_detected_ratio: float = 0.0
    multiple_faces_ratio: float = 0.0
    is_critical_flag: bool = False


class SlidingWindowBuffer:
    """Maintains a bounded time-indexed buffer of FrameObservations and BehaviorEvents."""

    def __init__(self, window_duration_sec: float | None = None) -> None:
        """Initialize the sliding window buffer.

        Args:
            window_duration_sec: Horizon in seconds for retaining observations and events.
        """
        settings = get_settings()
        self.window_duration_sec = (
            float(window_duration_sec)
            if window_duration_sec is not None
            else settings.temporal_sliding_window_sec
        )

        self._observations: deque[FrameObservation] = deque()
        self._events: deque[BehaviorEvent] = deque()
        self._latest_timestamp_sec: float = 0.0

    @property
    def window_duration(self) -> float:
        return self.window_duration_sec

    def add_observation(self, observation: FrameObservation) -> None:
        """Append a new frame observation and prune expired historical records."""
        t_sec = observation.timestamp_ms / 1000.0
        if t_sec > self._latest_timestamp_sec:
            self._latest_timestamp_sec = t_sec

        self._observations.append(observation)
        self._prune(self._latest_timestamp_sec)

    def add_event(self, event: BehaviorEvent) -> None:
        """Append an aggregated behavioral event."""
        if event.end_time > self._latest_timestamp_sec:
            self._latest_timestamp_sec = event.end_time

        self._events.append(event)
        self._prune(self._latest_timestamp_sec)

    def _prune(self, current_time_sec: float) -> None:
        """Evict observations and events that fell outside the sliding window horizon."""
        cutoff_sec = current_time_sec - self.window_duration_sec

        while self._observations and (self._observations[0].timestamp_ms / 1000.0) < cutoff_sec:
            self._observations.popleft()

        while self._events and self._events[0].end_time < cutoff_sec:
            self._events.popleft()

    def get_observations(self) -> list[FrameObservation]:
        """Return all observations currently inside the sliding window."""
        return list(self._observations)

    def get_events(
        self,
        event_type: EventType | None = None,
        min_duration_sec: float = 0.0,
    ) -> list[BehaviorEvent]:
        """Retrieve events in window filtered by type and duration."""
        res: list[BehaviorEvent] = []
        for ev in self._events:
            if event_type is not None and ev.event_type != event_type:
                continue
            if ev.duration < min_duration_sec:
                continue
            res.append(ev)
        return res

    def get_cumulative_duration(self, event_type: EventType) -> float:
        """Calculate total seconds spent in a specific behavioral event state within window."""
        total = 0.0
        for ev in self._events:
            if ev.event_type == event_type:
                total += ev.duration
        return round(total, 3)

    def get_event_counts(self) -> dict[str, int]:
        """Return frequency dictionary of all event types in the window."""
        counts: dict[str, int] = {}
        for ev in self._events:
            val = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
            counts[val] = counts.get(val, 0) + 1
        return counts

    def get_summary(self, current_time_sec: float | None = None) -> TemporalWindowSummary:
        """Generate high-level aggregated metrics and behavioral ratios for the window."""
        now = (
            current_time_sec
            if current_time_sec is not None
            else self._latest_timestamp_sec
        )
        self._prune(now)

        total_frames = len(self._observations)
        if total_frames == 0:
            return TemporalWindowSummary(
                window_start_time=max(0.0, now - self.window_duration_sec),
                window_end_time=now,
                window_duration_sec=self.window_duration_sec,
                total_frames=0,
            )

        window_start = self._observations[0].timestamp_ms / 1000.0
        window_end = self._observations[-1].timestamp_ms / 1000.0
        actual_duration = max(0.001, window_end - window_start)

        missing_count = 0
        gaze_dev_count = 0
        phone_count = 0
        multi_face_count = 0

        for obs in self._observations:
            if not obs.face_detected:
                missing_count += 1
            if obs.face_count > 1:
                multi_face_count += 1
            if obs.gaze is not None and obs.gaze.is_deviated():
                gaze_dev_count += 1
            if any(obj.label == "cell phone" for obj in obs.detected_objects):
                phone_count += 1

        absence_ratio = round(missing_count / total_frames, 3)
        gaze_ratio = round(gaze_dev_count / total_frames, 3)
        phone_ratio = round(phone_count / total_frames, 3)
        multi_ratio = round(multi_face_count / total_frames, 3)

        # Cumulative event durations
        durations: dict[str, float] = {}
        counts = self.get_event_counts()
        for ev in self._events:
            val = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
            durations[val] = round(durations.get(val, 0.0) + ev.duration, 3)

        # Critical threshold flag (e.g. phone visible or prolonged candidate absence)
        is_critical = (
            phone_count > 0
            or absence_ratio > 0.50
            or multi_ratio > 0.30
            or durations.get(EventType.PHONE_DETECTED.value, 0.0) > 0.0
            or durations.get(EventType.POSSIBLE_ABSENCE.value, 0.0) > 0.0
        )

        return TemporalWindowSummary(
            window_start_time=round(window_start, 3),
            window_end_time=round(window_end, 3),
            window_duration_sec=round(actual_duration, 3),
            total_frames=total_frames,
            event_counts=counts,
            cumulative_durations=durations,
            face_absence_ratio=absence_ratio,
            gaze_deviation_ratio=gaze_ratio,
            phone_detected_ratio=phone_ratio,
            multiple_faces_ratio=multi_ratio,
            is_critical_flag=is_critical,
        )

    def reset(self) -> None:
        """Clear all buffered observations and events."""
        self._observations.clear()
        self._events.clear()
        self._latest_timestamp_sec = 0.0
