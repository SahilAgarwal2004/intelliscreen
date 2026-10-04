#!/usr/bin/env python3
"""Smoke test script for Video Preprocessing & Quality Assessment (Feature 2)."""

import time
import cv2
import numpy as np

from intelliscreen.core.logging import setup_logger
from intelliscreen.core.schemas import BoundingBox
from intelliscreen.vision.preprocessor import VideoPreprocessor

logger = setup_logger("smoke_test_preprocessor", log_level="INFO")


def create_synthetic_frame(
    width: int = 1920,
    height: int = 1080,
    brightness: int = 140,
    blur: bool = False,
) -> np.ndarray:
    """Generate a realistic test frame containing geometric shapes and text."""
    frame = np.full((height, width, 3), brightness, dtype=np.uint8)

    # Draw simulated candidate head oval
    center = (width // 2, height // 2)
    axes = (width // 8, height // 4)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (70, 100, 200), -1)

    # Draw simulated eye regions
    cv2.circle(frame, (center[0] - 60, center[1] - 40), 20, (255, 255, 255), -1)
    cv2.circle(frame, (center[0] + 60, center[1] - 40), 20, (255, 255, 255), -1)
    cv2.circle(frame, (center[0] - 60, center[1] - 40), 8, (20, 20, 20), -1)
    cv2.circle(frame, (center[0] + 60, center[1] - 40), 8, (20, 20, 20), -1)

    # Add high-contrast text for edge detection / sharpness
    cv2.putText(
        frame,
        "IntelliScreen Proctoring Stream (1080p)",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if blur:
        frame = cv2.GaussianBlur(frame, (45, 45), 0)

    return frame


def run_smoke_test() -> None:
    logger.info("Initializing Video Preprocessor Smoke Test...")
    preprocessor = VideoPreprocessor(target_width=640, target_height=480)

    # 1. Process standard 1080p frame
    raw_1080p = create_synthetic_frame(width=1920, height=1080, brightness=130, blur=False)
    t0 = time.perf_counter()
    processed = preprocessor.process(raw_1080p, frame_index=1, timestamp_ms=33.3)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"Input Frame: {processed.original_shape[1]}x{processed.original_shape[0]} px -> "
        f"Processed: {processed.processed_shape[1]}x{processed.processed_shape[0]} px | "
        f"Latency: {latency_ms:.2f} ms"
    )
    logger.info(
        f"Quality Metrics (Standard): Brightness={processed.quality.brightness}, "
        f"Contrast={processed.quality.contrast}, BlurScore={processed.quality.blur_score}, "
        f"LowLight={processed.quality.is_low_light}, Blurry={processed.quality.is_blurry}"
    )

    # 2. Process Low-Light Frame
    dark_raw = create_synthetic_frame(width=1280, height=720, brightness=25, blur=False)
    processed_dark = preprocessor.process(dark_raw, frame_index=2, timestamp_ms=66.7)
    logger.info(
        f"Quality Metrics (Dark): Brightness={processed_dark.quality.brightness}, "
        f"LowLight={processed_dark.quality.is_low_light}"
    )
    assert processed_dark.quality.is_low_light is True

    # 3. Process Motion-Blurred Frame
    blurred_raw = create_synthetic_frame(width=1280, height=720, brightness=130, blur=True)
    processed_blurred = preprocessor.process(blurred_raw, frame_index=3, timestamp_ms=100.0)
    logger.info(
        f"Quality Metrics (Blurred): BlurScore={processed_blurred.quality.blur_score}, "
        f"Blurry={processed_blurred.quality.is_blurry}"
    )
    assert processed_blurred.quality.is_blurry is True

    # 4. Demonstrate Bounding Box Coordinate Mapping
    # Suppose a face detector detects a face in processed 640x480 pixel space at [200, 100, 440, 380]
    proc_box = BoundingBox(xmin=200.0, ymin=100.0, xmax=440.0, ymax=380.0, confidence=0.96, label="face")
    orig_box = processed.map_box_to_original(proc_box)

    logger.info("Coordinate Inversion Demonstration:")
    logger.info(
        f"  Processed Space (640x480): [{proc_box.xmin:.1f}, {proc_box.ymin:.1f}, {proc_box.xmax:.1f}, {proc_box.ymax:.1f}]"
    )
    logger.info(
        f"  Original Space  (1920x1080): [{orig_box.xmin:.1f}, {orig_box.ymin:.1f}, {orig_box.xmax:.1f}, {orig_box.ymax:.1f}]"
    )

    logger.info("Video Preprocessor smoke test passed cleanly.")


if __name__ == "__main__":
    run_smoke_test()
