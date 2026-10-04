#!/usr/bin/env python3
"""Smoke test script for Face Detection pipeline (Feature 3)."""

import os
import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.vision.face_detector import FaceDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_face_detector", log_level="INFO")

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


def run_smoke_test() -> None:
    logger.info("Initializing Face Detection Smoke Test...")

    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    detector = FaceDetector(backend="yunet", confidence_threshold=0.6)

    # ---------------------------------------------------------
    # Scenario A: Single Candidate Present
    # ---------------------------------------------------------
    img = cv2.imread(SAMPLE_FACE_PATH)
    frame_single = preprocessor.process(img, frame_index=1, timestamp_ms=100.0)

    t0 = time.perf_counter()
    res_single = detector.detect(frame_single)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info("--- SCENARIO A: Single Candidate ---")
    logger.info(
        f"Face Count: {res_single.face_count}, Missing: {res_single.is_missing}, "
        f"Multiple: {res_single.is_multiple_faces} | Latency: {latency_ms:.2f} ms"
    )
    if res_single.primary_face:
        box = res_single.primary_face.box
        logger.info(
            f"Primary Face Box: [{box.xmin:.1f}, {box.ymin:.1f}, {box.xmax:.1f}, {box.ymax:.1f}] | "
            f"Conf: {res_single.primary_face.confidence:.3f}"
        )
        logger.info(f"Anatomical Keypoints: {res_single.primary_face.keypoints}")

    # ---------------------------------------------------------
    # Scenario B: Candidate Missing / Away from Screen
    # ---------------------------------------------------------
    blank = np.full((480, 640, 3), 110, dtype=np.uint8)
    frame_empty = preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)
    res_empty = detector.detect(frame_empty)

    logger.info("--- SCENARIO B: Candidate Absence ---")
    logger.info(
        f"Face Count: {res_empty.face_count}, Missing: {res_empty.is_missing}, "
        f"Multiple: {res_empty.is_multiple_faces} | Latency: {res_empty.latency_ms} ms"
    )
    assert res_empty.is_missing is True

    # ---------------------------------------------------------
    # Scenario C: Multiple Persons in Frame
    # ---------------------------------------------------------
    resized = cv2.resize(img, (220, 220))
    multi_canvas = np.full((480, 640, 3), 120, dtype=np.uint8)
    multi_canvas[130:350, 70:290] = resized
    multi_canvas[130:350, 350:570] = resized

    frame_multi = preprocessor.process(multi_canvas, frame_index=3, timestamp_ms=300.0)
    res_multi = detector.detect(frame_multi)

    logger.info("--- SCENARIO C: Multiple Faces Detected ---")
    logger.info(
        f"Face Count: {res_multi.face_count}, Missing: {res_multi.is_missing}, "
        f"Multiple: {res_multi.is_multiple_faces} | Latency: {res_multi.latency_ms} ms"
    )
    assert res_multi.is_multiple_faces is True
    assert res_multi.face_count == 2

    # ---------------------------------------------------------
    # Visual Output Artifact Generation
    # ---------------------------------------------------------
    output_bgr = frame_single.bgr_image.copy()
    if res_single.primary_face:
        b = res_single.primary_face.box
        cv2.rectangle(
            output_bgr,
            (int(b.xmin), int(b.ymin)),
            (int(b.xmax), int(b.ymax)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output_bgr,
            f"Face: {res_single.primary_face.confidence:.2f}",
            (int(b.xmin), max(20, int(b.ymin) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for name, pt in res_single.primary_face.keypoints.items():
            cv2.circle(output_bgr, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    annotated_path = os.path.join(FIXTURES_DIR, "annotated_detection_demo.jpg")
    cv2.imwrite(annotated_path, output_bgr)
    logger.info(f"Visual demonstration artifact written to {annotated_path}")
    logger.info("Face Detector smoke test completed successfully.")


if __name__ == "__main__":
    run_smoke_test()
