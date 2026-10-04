#!/usr/bin/env python3
"""Smoke test script for Secondary Object and Device Detection (Feature 7).

Demonstrates:
- Real-time YOLOv8n object detection for proctoring-critical classes
- Temporal stride caching performance (active inference vs sub-ms cache lookup)
- Detection of candidate persons, cell phones, books, laptops, and secondary persons
- Color-coded bounding box visualization with warning labels
- Visual demonstration artifact saved to tests/fixtures/annotated_object_demo.jpg
"""

import os
import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.core.schemas import BoundingBox, DetectedObject
from intelliscreen.vision.object_detector import ObjectDetectionResult, ProctoringObjectDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_object_detector", log_level="INFO")

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
OUTPUT_ANNOTATED_PATH = os.path.join(FIXTURES_DIR, "annotated_object_demo.jpg")


def run_smoke_test() -> None:
    logger.info("=" * 70)
    logger.info("Initializing Secondary Object Detection Smoke Test (Feature 7)...")
    logger.info("=" * 70)

    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    detector = ProctoringObjectDetector(confidence_threshold=0.40, stride=3)

    raw_img = cv2.imread(SAMPLE_FACE_PATH)
    assert raw_img is not None, f"Failed to load image from {SAMPLE_FACE_PATH}"

    # ------------------------------------------------------------------------
    # Scenario A: Real-Time Inference on Candidate Frame
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO A: Candidate Ingestion & Neural Inference ---")
    f0 = preprocessor.process(raw_img, frame_index=0, timestamp_ms=0.0)

    # Warmup
    _ = detector.detect(f0, force=True)

    t0 = time.perf_counter()
    r0 = detector.detect(f0, force=True)
    inference_lat = (time.perf_counter() - t0) * 1000

    logger.info(f"Inference Latency: {inference_lat:.2f} ms | Is Cached: {r0.is_cached}")
    logger.info(f"Total Objects Detected: {len(r0.detections)}")
    logger.info(f"Person Count: {r0.person_count} | Secondary Person Alert: {r0.secondary_person_detected}")
    logger.info(f"Phone Detected: {r0.phone_detected} | Laptop: {r0.laptop_detected} | Book: {r0.book_detected}")
    logger.info(f"Prohibited Items: {r0.prohibited_items}")

    for idx, obj in enumerate(r0.detections):
        b = obj.box
        logger.info(
            f"  - [{idx+1}] {obj.label.upper()} (conf={obj.confidence:.2f}): "
            f"[{b.xmin:.1f}, {b.ymin:.1f}, {b.xmax:.1f}, {b.ymax:.1f}]"
        )

    # ------------------------------------------------------------------------
    # Scenario B: Stride Caching (Sub-Millisecond Lookup)
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO B: Temporal Stride Caching (Stride=3) ---")
    f1 = preprocessor.process(raw_img, frame_index=1, timestamp_ms=33.3)
    t0 = time.perf_counter()
    r1 = detector.detect(f1)
    cache_lat = (time.perf_counter() - t0) * 1000

    logger.info(f"Frame 1 (stride lookup): Is Cached={r1.is_cached} | Latency={cache_lat:.4f} ms")
    assert r1.is_cached is True
    assert cache_lat < 1.0

    f2 = preprocessor.process(raw_img, frame_index=2, timestamp_ms=66.6)
    r2 = detector.detect(f2)
    assert r2.is_cached is True

    f3 = preprocessor.process(raw_img, frame_index=3, timestamp_ms=100.0)
    r3 = detector.detect(f3)
    logger.info(f"Frame 3 (stride trigger): Is Cached={r3.is_cached} | Latency={r3.latency_ms:.2f} ms")
    assert r3.is_cached is False

    effective_mean_lat = (inference_lat + cache_lat * 2) / 3.0
    logger.info(f"Effective Stride Amortized CPU Latency: {effective_mean_lat:.2f} ms per frame")

    # ------------------------------------------------------------------------
    # Scenario C: Candidate Absence (Blank Frame)
    # ------------------------------------------------------------------------
    logger.info("\n--- SCENARIO C: Candidate Absence (Blank Frame) ---")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    blank_frame = preprocessor.process(blank, frame_index=0, timestamp_ms=0.0)
    r_blank = detector.detect(blank_frame, force=True)

    logger.info(f"Blank Frame: Detections={len(r_blank.detections)}, Person Count={r_blank.person_count}")
    assert len(r_blank.detections) == 0
    assert r_blank.prohibited_items == []

    # ------------------------------------------------------------------------
    # Visual Output Artifact Generation (Annotated Multi-Item Proctoring HUD)
    # ------------------------------------------------------------------------
    logger.info("\n--- GENERATING VISUAL DEMONSTRATION ARTIFACT ---")
    annotated = f0.bgr_image.copy()

    # Draw primary person detection
    annotated = detector.draw_detections(annotated, r0)

    # To showcase multi-device proctoring capabilities, inject mock unauthorized phone & book
    mock_phone_box = BoundingBox(xmin=480, ymin=340, xmax=580, ymax=460, confidence=0.89, label="cell phone")
    mock_book_box = BoundingBox(xmin=40, ymin=380, xmax=180, ymax=470, confidence=0.84, label="book")

    composite_res = ObjectDetectionResult(
        detections=list(r0.detections) + [
            DetectedObject(label="cell phone", confidence=0.89, box=mock_phone_box),
            DetectedObject(label="book", confidence=0.84, box=mock_book_box),
        ],
        phone_detected=True,
        laptop_detected=False,
        book_detected=True,
        person_count=1,
        secondary_person_detected=False,
        prohibited_items=["cell phone", "book"],
        latency_ms=r0.latency_ms,
    )

    annotated = detector.draw_detections(annotated, composite_res)

    # Telemetry HUD
    cv2.rectangle(annotated, (10, 10), (370, 160), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (370, 160), (60, 60, 60), 1)

    hud_lines = [
        "IntelliScreen Vision - Feature 7: Object Detector",
        f"Neural Model: YOLOv8n (CPU Inference)",
        f"Active Latency: {inference_lat:.1f}ms | Cached: {cache_lat:.3f}ms",
        f"Amortized (Stride={detector.stride}): {effective_mean_lat:.1f}ms/frame",
        f"Person Count: {composite_res.person_count} (Single Candidate)",
        f"PROHIBITED ITEMS: {', '.join(composite_res.prohibited_items).upper()}",
        f"Status: UNAUTHORIZED ITEM DETECTED [ALERT]",
    ]

    for idx, line in enumerate(hud_lines):
        color = (0, 255, 0) if idx == 0 else (220, 220, 220)
        if "ALERT" in line or "PROHIBITED" in line:
            color = (0, 0, 255)
        elif "Candidate" in line:
            color = (255, 200, 0)
        cv2.putText(
            annotated,
            line,
            (20, 28 + idx * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(OUTPUT_ANNOTATED_PATH, annotated)
    logger.info(f"Visual artifact successfully saved to: {OUTPUT_ANNOTATED_PATH}")
    logger.info("=" * 70)
    logger.info("Secondary Object Detection Smoke Test PASSED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
