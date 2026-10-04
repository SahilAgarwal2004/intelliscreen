#!/usr/bin/env python3
"""Smoke test script for 3D Head Pose Estimation (Feature 5).

Demonstrates:
- 3D Head Pose resolution via Perspective-n-Point (solvePnP)
- Euler angles decomposition (yaw, pitch, roll) and direction classification
- Sub-5ms PnP latency benchmark
- Visual 3D coordinate axes projection (X=Red, Y=Green, Z=Blue) from nose tip
- Absence / missing face handling
"""

import os
import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.vision.head_pose import HeadPoseEstimator
from intelliscreen.vision.landmarks import FaceLandmarkDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_head_pose", log_level="INFO")

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
OUTPUT_ANNOTATED_PATH = os.path.join(FIXTURES_DIR, "annotated_head_pose_demo.jpg")


def run_smoke_test() -> None:
    logger.info("=" * 70)
    logger.info("Initializing 3D Head Pose Estimation Smoke Test (Feature 5)...")
    logger.info("=" * 70)

    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    landmark_detector = FaceLandmarkDetector()
    head_pose_estimator = HeadPoseEstimator(yaw_threshold=30.0, pitch_threshold=25.0)

    # ------------------------------------------------------------------------
    # Scenario A: Single Candidate Head Pose Estimation
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO A: Single Candidate Head Pose Resolution ---")
    raw_img = cv2.imread(SAMPLE_FACE_PATH)
    assert raw_img is not None, f"Failed to load image from {SAMPLE_FACE_PATH}"

    frame = preprocessor.process(raw_img, frame_index=1, timestamp_ms=100.0)
    # Warmup landmark detector
    _ = landmark_detector.detect(frame)
    landmarks_res = landmark_detector.detect(frame)
    assert landmarks_res.landmarks_detected is True

    # Warmup
    _ = head_pose_estimator.estimate(frame, landmarks_res)

    # Benchmark PnP resolution latency
    num_runs = 50
    latencies: list[float] = []
    pose_res = None

    for _ in range(num_runs):
        t0 = time.perf_counter()
        pose_res = head_pose_estimator.estimate(frame, landmarks_res)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

    assert pose_res is not None and pose_res.pose_detected is True

    mean_pnp_lat = float(np.mean(latencies))
    median_pnp_lat = float(np.median(latencies))
    min_pnp_lat = float(np.min(latencies))
    max_pnp_lat = float(np.max(latencies))
    p95_pnp_lat = float(np.percentile(latencies, 95))

    logger.info(f"Pose Detected: {pose_res.pose_detected} | Missing: {pose_res.is_missing}")
    logger.info(f"Euler Angles: Yaw={pose_res.yaw:.2f}°, Pitch={pose_res.pitch:.2f}°, Roll={pose_res.roll:.2f}°")
    logger.info(f"Confidence: {pose_res.confidence:.3f}")
    logger.info(f"Direction Classification: {pose_res.direction_label}")
    logger.info(f"Looking Away Alert: {pose_res.is_looking_away}")
    logger.info(
        f"PnP Resolution Latency ({num_runs} runs): "
        f"Mean={mean_pnp_lat:.4f}ms | Median={median_pnp_lat:.4f}ms | Min={min_pnp_lat:.4f}ms | Max={max_pnp_lat:.4f}ms | P95={p95_pnp_lat:.4f}ms"
    )
    logger.info(f"PnP Budget Check (<5ms): {'PASS [OK]' if mean_pnp_lat < 5.0 else 'FAIL'}")
    logger.info(f"End-to-End Vision Frame Budget Check (<15ms): {'PASS [OK]' if landmarks_res.latency_ms + mean_pnp_lat < 15.0 else 'FAIL'}")

    logger.info("\n[Transform Vectors]")
    logger.info(f"  - Rodrigues rvec: {pose_res.rvec.ravel().tolist()}")
    logger.info(f"  - Translation tvec: {pose_res.tvec.ravel().tolist()}")

    # ------------------------------------------------------------------------
    # Scenario B: Candidate Missing / Away from Screen
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO B: Candidate Absence (Blank Frame) ---")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    blank_frame = preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)
    absent_res = head_pose_estimator.estimate(blank_frame)

    logger.info(
        f"Pose Detected: {absent_res.pose_detected} | Missing: {absent_res.is_missing} | "
        f"Direction: {absent_res.direction_label} | Latency: {absent_res.latency_ms:.2f}ms"
    )
    assert absent_res.pose_detected is False
    assert absent_res.is_missing is True

    # ------------------------------------------------------------------------
    # Visual Output Artifact Generation (3D Coordinate Axes Projection)
    # ------------------------------------------------------------------------
    logger.info("\n--- GENERATING VISUAL DEMONSTRATION ARTIFACT ---")
    annotated = frame.bgr_image.copy()

    # Draw 3D coordinate axes from nose tip
    annotated = head_pose_estimator.project_pose_axes(
        annotated,
        pose_res.rvec,
        pose_res.tvec,
        length=75.0,
    )

    # Highlight nose tip origin
    nose_2d = landmarks_res.pnp_landmarks_2d[0].astype(int)
    cv2.circle(annotated, tuple(nose_2d), 4, (0, 255, 255), -1)

    # Telemetry HUD
    cv2.rectangle(annotated, (10, 10), (340, 140), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (340, 140), (60, 60, 60), 1)

    hud_lines = [
        "IntelliScreen Vision - Feature 5: 3D Head Pose",
        f"Yaw: {pose_res.yaw:+.1f}deg (Horiz) | Pitch: {pose_res.pitch:+.1f}deg (Vert)",
        f"Roll: {pose_res.roll:+.1f}deg (Tilt) | Conf: {pose_res.confidence:.2f}",
        f"Direction: {pose_res.direction_label}",
        f"Looking Away: {'YES [ALERT]' if pose_res.is_looking_away else 'NO (Focused)'}",
        f"PnP Latency: {mean_pnp_lat:.3f} ms (<5ms budget)",
    ]

    for idx, line in enumerate(hud_lines):
        color = (0, 255, 0) if idx == 0 else (220, 220, 220)
        if "ALERT" in line:
            color = (0, 0, 255)
        cv2.putText(
            annotated,
            line,
            (20, 30 + idx * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # Legend for 3D Axes
    cv2.rectangle(annotated, (10, 430), (240, 470), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 430), (240, 470), (60, 60, 60), 1)
    cv2.putText(annotated, "3D Axes: ", (18, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, "X(R)", (90, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, "Y(D)", (135, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(annotated, "Z(Fwd)", (180, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_ANNOTATED_PATH, annotated)
    logger.info(f"Visual artifact successfully saved to: {OUTPUT_ANNOTATED_PATH}")
    logger.info("=" * 70)
    logger.info("3D Head Pose Smoke Test PASSED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
