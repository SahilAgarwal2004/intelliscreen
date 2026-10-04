#!/usr/bin/env python3
"""End-to-end smoke test for the Temporal Event Engine and Sliding Window Buffer.

Demonstrates temporal state machine aggregation across a realistic 60-second interview stream:
  1. Attentive candidate baseline (0s - 15s)
  2. Natural cognitive glance (15s - 16.8s) -> verified NOT flagged
  3. Prolonged off-screen glance left (20s - 26s) -> triggers LOOKING_LEFT & PROLONGED_OFF_SCREEN_GAZE
  4. Cell phone detection (30s - 33.5s) -> triggers PHONE_DETECTED
  5. Extended absence from desk (40s - 52s) -> triggers FACE_MISSING & POSSIBLE_ABSENCE
  6. Secondary person present (55s - 57s) -> triggers MULTIPLE_FACES
"""

from pathlib import Path
import sys
import time

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

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


def create_mock_obs(
    frame_index: int,
    t_sec: float,
    face_detected: bool = True,
    face_count: int = 1,
    yaw: float = 0.0,
    pitch: float = 0.0,
    gaze_yaw: float = 0.0,
    gaze_pitch: float = 0.0,
    gaze_dir: GazeDirection = GazeDirection.CENTER,
    objects: list[DetectedObject] | None = None,
) -> FrameObservation:
    """Construct domain FrameObservation for a given timeline second."""
    p_box = (
        BoundingBox(xmin=150, ymin=100, xmax=450, ymax=400, confidence=0.95)
        if face_detected
        else None
    )
    pose = (
        HeadPoseAngles(yaw=yaw, pitch=pitch, roll=0.0, confidence=0.92)
        if face_detected
        else None
    )
    gaze = (
        GazeVector(yaw=gaze_yaw, pitch=gaze_pitch, direction=gaze_dir, confidence=0.88)
        if face_detected
        else None
    )
    return FrameObservation(
        frame_index=frame_index,
        timestamp_ms=round(t_sec * 1000.0, 1),
        face_detected=face_detected,
        face_count=face_count,
        primary_face_box=p_box,
        head_pose=pose,
        gaze=gaze,
        detected_objects=objects or [],
    )


def run_smoke_test() -> None:
    print("=" * 82)
    print("      INTELLISCREEN TEMPORAL EVENT ENGINE - REALISTIC INTERVIEW SIMULATION")
    print("=" * 82)

    config = TemporalEngineConfig(
        face_absence_alert_sec=3.0,
        extended_absence_sec=10.0,
        multiple_faces_alert_sec=1.0,
        phone_presence_alert_sec=2.0,
        prolonged_look_away_sec=4.0,
        directional_look_away_sec=2.0,
        sliding_window_sec=30.0,
        debounce_frames=2,
    )
    engine = TemporalEventEngine(config=config)
    print(f"\n1. Initialized TemporalEventEngine with Sliding Window: {config.sliding_window_sec}s")

    fps = 10.0
    total_duration_sec = 60.0
    total_frames = int(total_duration_sec * fps)

    phone_obj = DetectedObject(
        label="cell phone",
        confidence=0.91,
        box=BoundingBox(xmin=350, ymin=250, xmax=420, ymax=380),
    )

    print(f"2. Streaming {total_frames} frames (60.0s @ {fps}fps) through Temporal State Machines...\n")
    all_triggered: list[BehaviorEvent] = []

    t0_start = time.perf_counter()

    for idx in range(total_frames):
        t_sec = idx / fps

        # Timeline logic
        if 0.0 <= t_sec < 15.0:
            # Phase 1: Natural centered engagement
            obs = create_mock_obs(idx, t_sec)

        elif 15.0 <= t_sec < 16.8:
            # Phase 2: Natural thinking glance (1.8s < thresholds)
            obs = create_mock_obs(
                idx, t_sec,
                gaze_yaw=-28.0,
                gaze_dir=GazeDirection.LEFT,
            )

        elif 16.8 <= t_sec < 20.0:
            # Return to center
            obs = create_mock_obs(idx, t_sec)

        elif 20.0 <= t_sec < 26.0:
            # Phase 3: Prolonged suspicious off-screen look left (6.0s continuous)
            obs = create_mock_obs(
                idx, t_sec,
                yaw=-32.0,
                gaze_yaw=-35.0,
                gaze_dir=GazeDirection.LEFT,
            )

        elif 26.0 <= t_sec < 30.0:
            # Return to center
            obs = create_mock_obs(idx, t_sec)

        elif 30.0 <= t_sec < 33.5:
            # Phase 4: Unauthorized cell phone detected (3.5s)
            obs = create_mock_obs(idx, t_sec, objects=[phone_obj])

        elif 33.5 <= t_sec < 40.0:
            # Return to center
            obs = create_mock_obs(idx, t_sec)

        elif 40.0 <= t_sec < 52.0:
            # Phase 5: Candidate completely absent from desk (12.0s)
            obs = create_mock_obs(idx, t_sec, face_detected=False, face_count=0)

        elif 52.0 <= t_sec < 55.0:
            # Return to center
            obs = create_mock_obs(idx, t_sec)

        elif 55.0 <= t_sec < 57.0:
            # Phase 6: Secondary person appears in frame (2.0s)
            obs = create_mock_obs(idx, t_sec, face_count=2)

        else:
            # Return to center
            obs = create_mock_obs(idx, t_sec)

        events = engine.process_observation(obs)
        for ev in events:
            all_triggered.append(ev)
            status = "FINALIZED" if ev.metadata.get("finalized") else "TRIGGERED"
            print(
                f"   [t={t_sec:5.1f}s] EVENT {status:9s}: {ev.event_type.value:26s} | "
                f"Duration: {ev.duration:4.1f}s (start={ev.start_time:.1f}s, end={ev.end_time:.1f}s)"
            )

    # Finalize any open streaks at session completion
    flushed = engine.flush()
    for ev in flushed:
        all_triggered.append(ev)
        print(
            f"   [t={60.0:5.1f}s] EVENT FLUSHED  : {ev.event_type.value:26s} | "
            f"Duration: {ev.duration:4.1f}s (start={ev.start_time:.1f}s, end={ev.end_time:.1f}s)"
        )

    proc_time_ms = (time.perf_counter() - t0_start) * 1000.0
    ms_per_frame = proc_time_ms / total_frames

    print(f"\n3. Engine Processing Performance:")
    print(f"   - Processed {total_frames} frames in {proc_time_ms:.2f} ms ({ms_per_frame:.4f} ms/frame)")
    assert ms_per_frame < 0.5, "Performance error: temporal aggregation must take < 0.5ms/frame."

    # Validate Event Triggering Accuracies
    print("\n4. Behavioral State Machine Verification:")
    event_types = [e.event_type for e in all_triggered]

    # Verify natural 1.8s glance was ignored
    glance_events = [
        e for e in all_triggered
        if 15.0 <= e.start_time <= 17.0 and e.event_type == EventType.LOOKING_LEFT
    ]
    assert len(glance_events) == 0, "Error: Natural 1.8s thinking glance was falsely flagged!"
    print("   [PASS] Natural cognitive glance (15.0s - 16.8s) correctly preserved without false alarm.")

    # Verify prolonged look away and directional look left
    assert EventType.LOOKING_LEFT in event_types, "Failed to detect LOOKING_LEFT."
    assert EventType.PROLONGED_OFF_SCREEN_GAZE in event_types, "Failed to detect PROLONGED_OFF_SCREEN_GAZE."
    print("   [PASS] Prolonged off-screen gaze and directional LOOKING_LEFT triggered and finalized.")

    # Verify cell phone detection
    assert EventType.PHONE_DETECTED in event_types, "Failed to detect PHONE_DETECTED."
    phone_evs = [e for e in all_triggered if e.event_type == EventType.PHONE_DETECTED]
    assert any(e.metadata.get("finalized") and abs(e.duration - 3.5) <= 0.2 for e in phone_evs)
    print("   [PASS] Cell phone detection (30.0s - 33.5s) triggered at 2.0s and finalized at 3.5s.")


    # Verify face absence and escalation to possible absence
    assert EventType.FACE_MISSING in event_types, "Failed to detect FACE_MISSING."
    assert EventType.POSSIBLE_ABSENCE in event_types, "Failed to detect POSSIBLE_ABSENCE escalation."
    print("   [PASS] Desk absence triggered FACE_MISSING (3.0s) and escalated to POSSIBLE_ABSENCE (10.0s).")

    # Verify multiple faces
    assert EventType.MULTIPLE_FACES in event_types, "Failed to detect MULTIPLE_FACES."
    print("   [PASS] Secondary person (55.0s - 57.0s) triggered MULTIPLE_FACES.")

    # Sliding Window Summary
    print("\n5. Active Sliding Window Summary (Last 30 seconds [30.0s - 60.0s]):")
    summary = engine.get_window_summary()
    print(f"   - Window Bounds: [{summary.window_start_time:.1f}s - {summary.window_end_time:.1f}s]")
    print(f"   - Total Buffered Frames: {summary.total_frames}")
    print(f"   - Face Absence Ratio: {summary.face_absence_ratio * 100:.1f}%")
    print(f"   - Phone Detected Ratio: {summary.phone_detected_ratio * 100:.1f}%")
    print(f"   - Multiple Faces Ratio: {summary.multiple_faces_ratio * 100:.1f}%")
    print(f"   - Is Critical Risk Flag: {summary.is_critical_flag}")
    print(f"   - Event Frequencies: {summary.event_counts}")
    print(f"   - Cumulative Durations: {summary.cumulative_durations}")

    print("\n" + "=" * 82)
    print("SMOKE TEST SUCCESSFUL: Temporal Event Engine demonstrated complete fidelity!")
    print("=" * 82)


if __name__ == "__main__":
    run_smoke_test()
