#!/usr/bin/env python3
"""Smoke test script for Dense Facial Landmark Detection (Feature 4).

Demonstrates:
- Dense 478 3D landmark extraction on test image
- Sub-15ms CPU inference benchmarking
- Eye Aspect Ratio (EAR) computation and blink detection
- Anatomical region extraction (eyes, nose, chin, mouth, irises)
- Canonical PnP 6-point anchors for 3D Head Pose and Gaze Vectorization
- Visual artifact generation with annotated facial mesh and metrics overlay
"""

import os
import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.vision.face_detector import FaceDetector
from intelliscreen.vision.landmarks import FaceLandmarkDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_landmarks", log_level="INFO")

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
OUTPUT_ANNOTATED_PATH = os.path.join(FIXTURES_DIR, "annotated_landmarks_demo.jpg")


def run_smoke_test() -> None:
    logger.info("=" * 70)
    logger.info("Initializing Dense Facial Landmark Smoke Test (Feature 4)...")
    logger.info("=" * 70)

    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    face_detector = FaceDetector(backend="yunet", confidence_threshold=0.6)
    landmark_detector = FaceLandmarkDetector(blink_ear_threshold=0.20)

    # ------------------------------------------------------------------------
    # Scenario A: Single Candidate Present (Standard Operation)
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO A: Single Candidate Landmark Extraction ---")
    raw_img = cv2.imread(SAMPLE_FACE_PATH)
    assert raw_img is not None, f"Failed to load image from {SAMPLE_FACE_PATH}"

    frame = preprocessor.process(raw_img, frame_index=1, timestamp_ms=100.0)
    detection_res = face_detector.detect(frame)
    primary_face = detection_res.primary_face

    # Warmup
    _ = landmark_detector.detect(frame, primary_face=primary_face)

    # Benchmark latency over multiple iterations
    num_runs = 20
    latencies: list[float] = []
    result = None

    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = landmark_detector.detect(frame, primary_face=primary_face)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

    assert result is not None and result.landmarks_detected is True

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    logger.info(f"Landmarks Detected: {result.landmarks_detected} | Missing: {result.is_missing} | Fallback: {result.is_fallback}")
    logger.info(f"Total Dense Landmarks: {len(result.landmarks)} (468 face mesh + 10 iris)")
    logger.info(
        f"Inference Latency ({num_runs} runs): "
        f"Mean={mean_lat:.2f}ms | Median={median_lat:.2f}ms | Min={min_lat:.2f}ms | Max={max_lat:.2f}ms | P95={p95_lat:.2f}ms"
    )
    logger.info(f"CPU Performance Budget Check (<15ms): {'PASS [OK]' if mean_lat < 15.0 else 'FAIL'}")

    logger.info(f"\n[Eye Aspect Ratio (EAR) & Blink Metrics]")
    logger.info(f"  - Left Eye EAR:  {result.left_ear:.4f}")
    logger.info(f"  - Right Eye EAR: {result.right_ear:.4f}")
    logger.info(f"  - Average EAR:   {result.avg_ear:.4f}")
    logger.info(f"  - Blink Status:  {'BLINKING' if result.is_blinking else 'EYES OPEN'}")

    logger.info(f"\n[Anatomical Region Keypoints]")
    logger.info(f"  - Left Eye Points:    {len(result.left_eye)} pts")
    logger.info(f"  - Right Eye Points:   {len(result.right_eye)} pts")
    logger.info(f"  - Nose Bridge Points: {len(result.nose_bridge)} pts")
    logger.info(f"  - Chin Points:        {len(result.chin)} pts")
    logger.info(f"  - Mouth Points:       {len(result.mouth)} pts")
    logger.info(f"  - Left Iris Center:   {result.gaze_anchors['left_iris_center_2d'].tolist()}")
    logger.info(f"  - Right Iris Center:  {result.gaze_anchors['right_iris_center_2d'].tolist()}")

    logger.info(f"\n[Canonical 6 PnP Anchors for 3D Head Pose]")
    pnp_pts = result.get_pnp_6points()
    anchor_labels = [
        "1. Nose Tip",
        "2. Chin Apex",
        "3. Left Eye Outer Corner",
        "4. Right Eye Outer Corner",
        "5. Left Mouth Corner",
        "6. Right Mouth Corner",
    ]
    for label, pt in zip(anchor_labels, pnp_pts):
        logger.info(f"  - {label:26s}: x={pt[0]:.1f}, y={pt[1]:.1f}")

    # ------------------------------------------------------------------------
    # Scenario B: Candidate Missing / Away from Screen
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO B: Candidate Absence (Blank Frame) ---")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    blank_frame = preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)
    absent_res = landmark_detector.detect(blank_frame)

    logger.info(
        f"Landmarks Detected: {absent_res.landmarks_detected} | "
        f"Missing: {absent_res.is_missing} | Points: {len(absent_res.landmarks)} | Latency: {absent_res.latency_ms:.2f}ms"
    )
    assert absent_res.landmarks_detected is False
    assert absent_res.is_missing is True

    # ------------------------------------------------------------------------
    # Scenario C: Robust Fallback Inference
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO C: Geometric Fallback Backend ---")
    fallback_detector = FaceLandmarkDetector(backend="fallback")
    fallback_res = fallback_detector.detect(frame, primary_face=primary_face)

    logger.info(
        f"Fallback Detected: {fallback_res.landmarks_detected} | "
        f"Is Fallback: {fallback_res.is_fallback} | Anchors: {fallback_res.pnp_landmarks_2d.shape}"
    )
    assert fallback_res.landmarks_detected is True
    assert fallback_res.is_fallback is True

    # ------------------------------------------------------------------------
    # Visual Output Artifact Generation
    # ------------------------------------------------------------------------
    logger.info("\n--- GENERATING VISUAL DEMONSTRATION ARTIFACT ---")
    annotated = frame.bgr_image.copy()

    # 1. Draw subtle mesh points
    for pt in result.landmark_points_2d:
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 1, (180, 180, 180), -1)

    # 2. Draw eye contours
    for region_name, color in [("left_eye", (255, 140, 0)), ("right_eye", (0, 140, 255))]:
        eye_pts = result.get_region_points_2d(region_name).astype(np.int32)
        cv2.polylines(annotated, [eye_pts], isClosed=True, color=color, thickness=1)

    # 3. Draw iris centers
    left_iris = result.gaze_anchors["left_iris_center_2d"].astype(int)
    right_iris = result.gaze_anchors["right_iris_center_2d"].astype(int)
    cv2.circle(annotated, tuple(left_iris), 4, (0, 255, 0), -1)
    cv2.circle(annotated, tuple(right_iris), 4, (0, 255, 0), -1)

    # 4. Draw mouth contour
    mouth_pts = result.get_region_points_2d("mouth").astype(np.int32)
    cv2.polylines(annotated, [mouth_pts], isClosed=True, color=(0, 0, 255), thickness=1)

    # 5. Draw chin contour
    chin_pts = result.get_region_points_2d("chin").astype(np.int32)
    cv2.polylines(annotated, [chin_pts], isClosed=False, color=(0, 255, 255), thickness=1)

    # 6. Draw canonical 6 PnP keypoints (prominent magenta rings)
    for i, pt in enumerate(pnp_pts):
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, (255, 0, 255), 2)
        cv2.putText(
            annotated,
            str(i + 1),
            (int(pt[0]) + 8, int(pt[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # 7. Telemetry & Metrics HUD Overlay
    cv2.rectangle(annotated, (10, 10), (320, 130), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (320, 130), (60, 60, 60), 1)

    hud_lines = [
        f"IntelliScreen Vision - Feature 4",
        f"Dense Landmarks: {len(result.landmarks)} pts (478 topology)",
        f"CPU Latency: {mean_lat:.2f} ms (<15ms budget)",
        f"Left EAR: {result.left_ear:.3f} | Right: {result.right_ear:.3f}",
        f"Avg EAR:  {result.avg_ear:.3f}  ({'BLINK' if result.is_blinking else 'OPEN'})",
    ]
    for idx, line in enumerate(hud_lines):
        color = (0, 255, 0) if idx == 0 else (220, 220, 220)
        cv2.putText(
            annotated,
            line,
            (20, 32 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(OUTPUT_ANNOTATED_PATH, annotated)
    logger.info(f"Visual artifact successfully saved to: {OUTPUT_ANNOTATED_PATH}")
    logger.info("=" * 70)
    logger.info("Dense Facial Landmark Smoke Test PASSED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
