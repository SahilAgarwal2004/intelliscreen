"""Explainable Suspicion Scoring Engine for MCQ & Text-Based Exam Proctoring."""

from dataclasses import dataclass
import time

from intelliscreen.config.settings import get_settings
from intelliscreen.core.schemas import (
    BehaviorEvent,
    EventType,
    SuspicionScore,
    SuspicionSeverity,
)
from intelliscreen.temporal.window import TemporalWindowSummary


@dataclass
class ProctoringEvaluation:
    """Real-time proctoring evaluation result for a candidate during an MCQ test."""

    score: float
    severity: SuspicionSeverity
    confidence: float
    contributing_factors: list[str]
    explanation: str
    mitigating_factors: list[str]
    is_malpractice_flagged: bool


class ProctoringScorer:
    """Evaluates behavioral events, temporal ratios, and test incidents to compute suspicion."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(
        self,
        window_summary: TemporalWindowSummary,
        active_events: list[BehaviorEvent],
        test_incidents: list[str] | None = None,
        timestamp: float | None = None,
    ) -> ProctoringEvaluation:
        """Calculate composite suspicion score (0-100) and generate explainable audit findings."""
        score = 0.0
        contributing_factors: list[str] = []
        mitigating_factors: list[str] = []

        # -------------------------------------------------------------
        # 1. Critical Hard Violations (Immediate Escalation)
        # -------------------------------------------------------------
        if window_summary.phone_detected_ratio > 0.0:
            # Phone visibility is a high-severity violation in an online exam
            phone_penalty = min(50.0, 30.0 + window_summary.phone_detected_ratio * 40.0)
            score += phone_penalty
            contributing_factors.append(
                f"Unauthorized mobile phone detected in candidate workspace (presence ratio: {window_summary.phone_detected_ratio:.1%})"
            )

        if window_summary.multiple_faces_ratio > 0.0:
            # Secondary person presence
            score += 35.0
            contributing_factors.append("Multiple individuals detected in candidate camera frame")

        # -------------------------------------------------------------
        # 2. Prolonged Gaze Deviations & Off-Screen Look-Away
        # -------------------------------------------------------------
        if window_summary.gaze_deviation_ratio > 0.15:
            # Scale penalty based on proportion of time spent looking away from screen
            gaze_penalty = min(35.0, window_summary.gaze_deviation_ratio * 50.0)
            score += gaze_penalty
            contributing_factors.append(
                f"Sustained off-screen gaze deviation ({window_summary.gaze_deviation_ratio:.1%} of recent 30s window)"
            )
        else:
            if window_summary.gaze_deviation_ratio > 0.0:
                mitigating_factors.append(
                    "Brief gaze deviations (<2.5s) classified as natural cognitive pause / reading question"
                )

        # -------------------------------------------------------------
        # 3. Candidate Desk Absence / Camera Disappearance
        # -------------------------------------------------------------
        if window_summary.face_absence_ratio > 0.10:
            absence_penalty = min(40.0, window_summary.face_absence_ratio * 50.0)
            score += absence_penalty
            contributing_factors.append(
                f"Candidate absent from camera view ({window_summary.face_absence_ratio:.1%} of recent 30s window)"
            )

        # -------------------------------------------------------------
        # 4. Multi-Signal Correlation (Compound Malpractice)
        # -------------------------------------------------------------
        # Compound: Phone visible + Looking Downward
        looking_down = any(e.event_type == EventType.LOOKING_DOWN for e in active_events)
        if window_summary.phone_detected_ratio > 0.0 and looking_down:
            score += 15.0
            contributing_factors.append(
                "High-confidence correlated malpractice: simultaneous downward gaze with phone presence"
            )

        # -------------------------------------------------------------
        # 5. Optional Exam Incidents (e.g. Tab Switching / Fullscreen Exit)
        # -------------------------------------------------------------
        if test_incidents:
            for incident in test_incidents:
                score += 15.0
                contributing_factors.append(f"Client-side exam incident: {incident}")

        # Clamp composite score between 0.0 and 100.0
        final_score = round(min(100.0, max(0.0, score)), 1)

        # Determine Severity Tier based on Settings
        if final_score >= self.settings.scoring_high_threshold:
            severity = SuspicionSeverity.CRITICAL
        elif final_score >= self.settings.scoring_medium_threshold:
            severity = SuspicionSeverity.HIGH
        elif final_score >= self.settings.scoring_low_threshold:
            severity = SuspicionSeverity.MEDIUM
        else:
            severity = SuspicionSeverity.LOW

        # Generate Human-Auditable Explanation Summary
        if not contributing_factors:
            explanation = "Normal exam behavior observed. Candidate remains focused on screen."
            confidence = 0.95
        else:
            explanation = f"Suspicion elevated ({severity.value}): " + "; ".join(contributing_factors) + "."
            confidence = 0.88

        is_malpractice = severity in (SuspicionSeverity.HIGH, SuspicionSeverity.CRITICAL)

        return ProctoringEvaluation(
            score=final_score,
            severity=severity,
            confidence=confidence,
            contributing_factors=contributing_factors,
            explanation=explanation,
            mitigating_factors=mitigating_factors,
            is_malpractice_flagged=is_malpractice,
        )

    def to_domain_score(
        self, evaluation: ProctoringEvaluation, timestamp: float | None = None
    ) -> SuspicionScore:
        """Convert evaluation to canonical SuspicionScore domain model."""
        return SuspicionScore(
            timestamp=timestamp if timestamp is not None else time.time(),
            score=evaluation.score,
            severity=evaluation.severity,
            confidence=evaluation.confidence,
            contributing_events=evaluation.contributing_factors,
            explanation_summary=evaluation.explanation,
            mitigating_factors=evaluation.mitigating_factors,
        )
