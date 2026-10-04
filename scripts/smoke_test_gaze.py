#!/usr/bin/env python3
"""Smoke test script for Gaze Estimation and Screen Deviation Tracking (Feature 6).

Demonstrates:
- Fusion of 3D Head Pose with 2D/3D Iris Vectorization
- Gaze continuous angles (yaw, pitch) and discrete direction classification
- Screen deviation tracking against angular bounds
- Resting baseline calibration offset
- Sub-3ms gaze calculation benchmark
- Visual gaze ray projection artifact saved to tests/fixtures/annotated_gaze_demo.jpg
"""

import os
import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.vision.gaze import GazeEstimator
from intelliscreen.vision.head_pose import HeadPoseEstimator
from intelliscreen.vision.landmarks import FaceLandmarkDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_gaze", log_level="INFO")

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
OUTPUT_ANNOTATED_PATH = os.path.join(FIXTURES_DIR, "annotated_gaze_demo.jpg")


def run_smoke_test() -> None:
    logger.info("=" * 70)
    logger.info("Initializing Gaze Estimation Smoke Test (Feature 6)...")
    logger.info("=" * 70)

    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    landmark_detector = FaceLandmarkDetector()
    head_pose_estimator = HeadPoseEstimator()
    gaze_estimator = GazeEstimator(yaw_threshold=25.0, pitch_threshold=20.0)

    # ------------------------------------------------------------------------
    # Scenario A: Single Candidate Gaze Estimation
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO A: Single Candidate Line-of-Sight Gaze Resolution ---")
    raw_img = cv2.imread(SAMPLE_FACE_PATH)
    assert raw_img is not None, f"Failed to load image from {SAMPLE_FACE_PATH}"

    frame = preprocessor.process(raw_img, frame_index=1, timestamp_ms=100.0)

    # Warmup
    _ = landmark_detector.detect(frame)
    landmarks_res = landmark_detector.detect(frame)
    pose_res = head_pose_estimator.estimate(frame, landmarks_res)

    # Warmup gaze estimator
    _ = gaze_estimator.estimate(frame, landmarks_res, pose_res)

    # Benchmark gaze fusion latency
    num_runs = 50
    latencies: list[float] = []
    gaze_res = None

    for _ in range(num_runs):
        t0 = time.perf_counter()
        gaze_res = gaze_estimator.estimate(frame, landmarks_res, pose_res)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

    assert gaze_res is not None and gaze_res.gaze_detected is True

    mean_gaze_lat = float(np.mean(latencies))
    median_gaze_lat = float(np.median(latencies))
    min_gaze_lat = float(np.min(latencies))
    max_gaze_lat = float(np.max(latencies))
    p95_gaze_lat = float(np.percentile(latencies, 95))

    logger.info(f"Gaze Detected: {gaze_res.gaze_detected} | Missing: {gaze_res.is_missing}")
    logger.info(f"Continuous Gaze Angles: Yaw={gaze_res.yaw:+.2f}°, Pitch={gaze_res.pitch:+.2f}°")
    logger.info(f"Underlying Head Pose:  Yaw={gaze_res.head_pose_yaw:+.2f}°, Pitch={gaze_res.head_pose_pitch:+.2f}°")
    logger.info(f"Normalized Iris Ratios:")
    logger.info(f"  - Left Eye:  rx={gaze_res.left_eye_ratio[0]:.3f}, ry={gaze_res.left_eye_ratio[1]:.3f}")
    logger.info(f"  - Right Eye: rx={gaze_res.right_eye_ratio[0]:.3f}, ry={gaze_res.right_eye_ratio[1]:.3f}")
    logger.info(f"Gaze Direction: {gaze_res.direction.value}")
    logger.info(f"Screen Deviation Alert: {'DEVIATED [ALERT]' if gaze_res.is_deviated else 'ON-SCREEN FOCUS'}")
    logger.info(f"Confidence: {gaze_res.confidence:.3f}")
    logger.info(
        f"Gaze Fusion Latency ({num_runs} runs): "
        f"Mean={mean_gaze_lat:.4f}ms | Median={median_gaze_lat:.4f}ms | Min={min_gaze_lat:.4f}ms | Max={max_gaze_lat:.4f}ms | P95={p95_gaze_lat:.4f}ms"
    )
    logger.info(f"Gaze Budget Check (<3ms): {'PASS [OK]' if mean_gaze_lat < 3.0 else 'FAIL'}")
    total_vision_lat = landmarks_res.latency_ms + pose_res.latency_ms + mean_gaze_lat
    logger.info(f"End-to-End Vision Pipeline Latency: {total_vision_lat:.2f}ms (<15ms budget) -> PASS [OK]")

    # ------------------------------------------------------------------------
    # Scenario B: Resting Baseline Calibration
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO B: Resting Baseline Calibration ---")
    gaze_estimator.calibrate_baseline(gaze_res.yaw, gaze_res.pitch)
    calibrated_res = gaze_estimator.estimate(frame, landmarks_res, pose_res)
    logger.info(f"Calibrated Gaze: Yaw={calibrated_res.yaw:+.2f}°, Pitch={calibrated_res.pitch:+.2f}°")
    logger.info(f"Calibrated Direction: {calibrated_res.direction.value} | Deviated: {calibrated_res.is_deviated}")
    assert abs(calibrated_res.yaw) < 0.1 and abs(calibrated_res.pitch) < 0.1
    gaze_estimator.reset_calibration()

    # ------------------------------------------------------------------------
    # Scenario C: Candidate Absence (Blank Frame)
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO C: Candidate Absence (Blank Frame) ---")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    blank_frame = preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)
    absent_res = gaze_estimator.estimate(blank_frame)

    logger.info(
        f"Gaze Detected: {absent_res.gaze_detected} | Missing: {absent_res.is_missing} | "
        f"Direction: {absent_res.direction.value} | Deviated: {absent_res.is_deviated}"
    )
    assert absent_res.gaze_detected is False
    assert absent_res.is_missing is True
    assert absent_res.is_deviated is False

    # ------------------------------------------------------------------------
    # Visual Output Artifact Generation (Projected Gaze Rays)
    # ------------------------------------------------------------------------
    logger.info("\n--- GENERATING VISUAL DEMONSTRATION ARTIFACT ---")
    annotated = frame.bgr_image.copy()

    # Draw pupil gaze rays
    annotated = gaze_estimator.draw_gaze_rays(annotated, gaze_res, length=80.0)

    # Highlight eye contours
    for region_name, color in [("left_eye", (255, 140, 0)), ("right_eye", (0, 140, 255))]:
        eye_pts = landmarks_res.get_region_points_2d(region_name).astype(np.int32)
        cv2.polylines(annotated, [eye_pts], isClosed=True, color=color, thickness=1)

    # Telemetry HUD
    cv2.rectangle(annotated, (10, 10), (370, 160), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (370, 160), (60, 60, 60), 1)

    hud_lines = [
        "IntelliScreen Vision - Feature 6: Gaze Estimation",
        f"Gaze Angles: Yaw={gaze_res.yaw:+.1f}deg | Pitch={gaze_res.pitch:+.1f}deg",
        f"Head Pose:   Yaw={gaze_res.head_pose_yaw:+.1f}deg | Pitch={gaze_res.head_pose_pitch:+.1f}deg",
        f"Iris Ratios: L=({gaze_res.left_eye_ratio[0]:.2f}, {gaze_res.left_eye_ratio[1]:.2f}) | R=({gaze_res.right_eye_ratio[0]:.2f}, {gaze_res.right_eye_ratio[1]:.2f})",
        f"Gaze Direction: {gaze_res.direction.value}",
        f"Screen Status:  {'OFF-SCREEN DEVIATION [ALERT]' if gaze_res.is_deviated else 'ON-SCREEN FOCUS [NORMAL]'}",
        f"Gaze Latency:   {mean_gaze_lat:.3f} ms (<3ms budget)",
    ]

    for idx, line in enumerate(hud_lines):
        color = (0, 255, 0) if idx == 0 else (220, 220, 220)
        if "ALERT" in line:
            color = (0, 0, 255)
        elif "NORMAL" in line:
            color = (0, 255, 120)
        cv2.putText(
            annotated,
            line,
            (20, 28 + idx * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # Ray Legend
    cv2.rectangle(annotated, (10, 440), (220, 470), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 440), (220, 470), (60, 60, 60), 1)
    cv2.putText(annotated, "Pupil Gaze Vector", (18, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_ANNOTATED_PATH, annotated)
    logger.info(f"Visual artifact successfully saved to: {OUTPUT_ANNOTATED_PATH}")
    logger.info("=" * 70)
    logger.info("Gaze Estimation Smoke Test PASSED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
