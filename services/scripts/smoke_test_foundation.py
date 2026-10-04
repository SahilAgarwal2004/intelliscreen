#!/usr/bin/env python3
"""Smoke test script for IntelliScreen Foundation (Feature 1)."""

import json
from intelliscreen.config.settings import get_settings
from intelliscreen.core.logging import setup_logger
from intelliscreen.core.schemas import (
    BehaviorEvent,
    BoundingBox,
    DetectedObject,
    EventType,
    FrameObservation,
    GazeDirection,
    GazeVector,
    HeadPoseAngles,
    SessionState,
    SuspicionScore,
    SuspicionSeverity,
)

logger = setup_logger("smoke_test", log_level="INFO")


def run_smoke_test() -> None:
    logger.info("Initializing IntelliScreen Foundation Smoke Test...")

    # 1. Configuration & Settings Check
    settings = get_settings()
    logger.info(
        f"Settings loaded: Env={settings.env}, TargetFPS={settings.vision_target_fps}, "
        f"GazeYawThresh={settings.gaze_yaw_threshold}°, SuspiciousMinSec={settings.temporal_suspicious_look_away_min_sec}s"
    )

    # 2. Session Initialization
    session = SessionState(candidate_id="cand_alex_402")
    logger.info(f"Initialized active session: {session.session_id} for candidate {session.candidate_id}")

    # 3. Simulate Frame Observation Stream
    # Frame 1: Normal center gaze
    frame_normal = FrameObservation(
        frame_index=1,
        timestamp_ms=100.0,
        face_detected=True,
        face_count=1,
        primary_face_box=BoundingBox(xmin=0.25, ymin=0.15, xmax=0.65, ymax=0.75, confidence=0.98),
        head_pose=HeadPoseAngles(yaw=2.0, pitch=1.0, roll=0.0, confidence=0.97),
        gaze=GazeVector(yaw=3.0, pitch=-1.0, direction=GazeDirection.CENTER, confidence=0.95),
    )
    session.total_frames_processed += 1

    # Frame 2: Gaze deviating to bottom-left with phone detection
    frame_suspicious = FrameObservation(
        frame_index=45,
        timestamp_ms=4500.0,
        face_detected=True,
        face_count=1,
        primary_face_box=BoundingBox(xmin=0.25, ymin=0.18, xmax=0.65, ymax=0.78, confidence=0.96),
        head_pose=HeadPoseAngles(yaw=-28.0, pitch=-22.0, roll=-2.0, confidence=0.94),
        gaze=GazeVector(yaw=-32.0, pitch=-24.0, direction=GazeDirection.DOWN_LEFT, confidence=0.92),
        detected_objects=[
            DetectedObject(
                label="cell phone",
                confidence=0.88,
                box=BoundingBox(xmin=0.70, ymin=0.65, xmax=0.92, ymax=0.95, confidence=0.88),
            )
        ],
    )
    session.total_frames_processed += 1

    logger.info(
        f"Processed Frame #{frame_suspicious.frame_index}: "
        f"Gaze Yaw={frame_suspicious.gaze.yaw}°, Gaze Deviated={frame_suspicious.gaze.is_deviated()}, "
        f"Head Turn={frame_suspicious.head_pose.is_looking_away()}, Objects={[o.label for o in frame_suspicious.detected_objects]}"
    )

    # 4. Synthesize Behavior Event
    event = BehaviorEvent(
        event_type=EventType.PROLONGED_OFF_SCREEN_GAZE,
        start_time=1.5,
        end_time=5.8,
        duration=4.3,
        confidence=0.92,
        source="temporal_engine",
        metadata={"direction": "DOWN_LEFT", "mean_yaw": -30.5, "phone_concurrent": True},
    )
    session.events_recorded.append(event)
    logger.info(f"Recorded Behavior Event: {event.event_type.value} (Duration: {event.duration}s, Conf: {event.confidence})")

    # 5. Suspicion Assessment
    suspicion = SuspicionScore(
        timestamp=5.8,
        score=78.5,
        severity=SuspicionSeverity.HIGH,
        confidence=0.91,
        contributing_events=[event.event_type.value, "PHONE_DETECTED"],
        explanation_summary=(
            "Candidate exhibited sustained downward-left gaze deviation for 4.3s (exceeding 4.0s threshold) "
            "co-occurring with a secondary handheld object ('cell phone', conf: 0.88)."
        ),
        mitigating_factors=[],
    )
    session.latest_suspicion = suspicion
    logger.info(f"Generated Suspicion Verdict: Score={suspicion.score}/100, Tier={suspicion.severity.value}")
    logger.info(f"Explanation: {suspicion.explanation_summary}")

    # 6. JSON Export Validation
    exported_json = session.model_dump_json(indent=2)
    assert len(exported_json) > 100
    logger.info("Session JSON serialization verified successfully.")
    print("\n--- SERIALIZED SESSION SUMMARY SAMPLE ---")
    print(exported_json[:600] + "\n... [truncated] ...\n}")


if __name__ == "__main__":
    run_smoke_test()
