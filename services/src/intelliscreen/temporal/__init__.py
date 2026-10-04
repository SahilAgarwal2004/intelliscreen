"""Temporal Event Engine and Aggregation System for IntelliScreen."""

from intelliscreen.temporal.engine import (
    TemporalEngineConfig,
    TemporalEventEngine,
)
from intelliscreen.temporal.trackers import (
    BaseConditionTracker,
    ConditionStreak,
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

__all__ = [
    # Engine & Configuration
    "TemporalEngineConfig",
    "TemporalEventEngine",
    # Condition Trackers
    "BaseConditionTracker",
    "ConditionStreak",
    "FaceAbsenceTracker",
    "MultipleFacesTracker",
    "PhoneDetectionTracker",
    "HeadTurnTracker",
    "GazeDeviationTracker",
    # Sliding Window
    "SlidingWindowBuffer",
    "TemporalWindowSummary",
]
