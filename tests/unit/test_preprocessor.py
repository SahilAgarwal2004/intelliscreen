"""Unit tests for VideoPreprocessor and PreprocessedFrame."""

import cv2
import numpy as np
import pytest

from intelliscreen.core.exceptions import FramePreprocessingError
from intelliscreen.core.schemas import BoundingBox
from intelliscreen.vision.preprocessor import VideoPreprocessor


def test_normal_bgr_frame_preprocessing():
    """Verify that a 1080p BGR frame is correctly resized and transformed."""
    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    raw_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)

    processed = preprocessor.process(raw_frame, frame_index=1, timestamp_ms=33.3)

    assert processed.frame_index == 1
    assert processed.timestamp_ms == 33.3
    assert processed.rgb_image.shape == (480, 640, 3)
    assert processed.bgr_image.shape == (480, 640, 3)
    assert processed.gray_image.shape == (480, 640)
    assert processed.original_shape == (1080, 1920, 3)
    assert processed.processed_shape == (480, 640, 3)
    assert pytest.approx(processed.scale_x, 0.001) == 640 / 1920
    assert pytest.approx(processed.scale_y, 0.001) == 480 / 1080


def test_rgb_input_flag():
    """Verify that passing an RGB frame with is_rgb=True correctly translates to BGR."""
    preprocessor = VideoPreprocessor(target_width=320, target_height=240)
    # Red pixel in RGB: (255, 0, 0)
    rgb_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_frame[:, :] = [255, 0, 0]

    processed = preprocessor.process(rgb_frame, is_rgb=True)

    # In BGR, red is channel 2: [0, 0, 255]
    assert processed.bgr_image[0, 0, 0] == 0
    assert processed.bgr_image[0, 0, 2] == 255
    # In RGB, red is channel 0: [255, 0, 0]
    assert processed.rgb_image[0, 0, 0] == 255
    assert processed.rgb_image[0, 0, 2] == 0


def test_grayscale_input_conversion():
    """Verify that a 2D grayscale array is converted into 3-channel representations."""
    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    gray_frame = np.full((200, 200), 100, dtype=np.uint8)

    processed = preprocessor.process(gray_frame)

    assert processed.gray_image.shape == (480, 640)
    assert processed.bgr_image.shape == (480, 640, 3)
    assert processed.rgb_image.shape == (480, 640, 3)


def test_process_bytes_success():
    """Verify decoding and processing raw image bytes (e.g. JPEG)."""
    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    dummy_image = np.full((300, 400, 3), 150, dtype=np.uint8)
    _, encoded_bytes = cv2.imencode(".jpg", dummy_image)

    processed = preprocessor.process_bytes(encoded_bytes.tobytes(), frame_index=5, timestamp_ms=166.7)

    assert processed.frame_index == 5
    assert processed.rgb_image.shape == (480, 640, 3)
    assert processed.original_shape == (300, 400, 3)


def test_invalid_frame_inputs():
    """Verify that malformed inputs raise FramePreprocessingError."""
    preprocessor = VideoPreprocessor()

    # Non-array input
    with pytest.raises(FramePreprocessingError, match="Expected numpy.ndarray"):
        preprocessor.process("not_an_image")  # type: ignore

    # Empty array
    with pytest.raises(FramePreprocessingError, match="empty"):
        preprocessor.process(np.array([]))

    # Invalid dimension (4D array)
    with pytest.raises(FramePreprocessingError, match="Invalid frame dimensions"):
        preprocessor.process(np.zeros((1, 2, 3, 4)))

    # Too small dimensions
    with pytest.raises(FramePreprocessingError, match="too small"):
        preprocessor.process(np.zeros((5, 5, 3), dtype=np.uint8))

    # Corrupt / empty bytes
    with pytest.raises(FramePreprocessingError, match="empty image byte buffer"):
        preprocessor.process_bytes(b"")

    with pytest.raises(FramePreprocessingError, match="Failed to decode"):
        preprocessor.process_bytes(b"corrupt_non_image_payload")


def test_quality_metrics_low_light():
    """Verify detection of underexposed / dark frames."""
    preprocessor = VideoPreprocessor(low_light_threshold=40.0)

    # Very dark frame (mean intensity = 10)
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    processed_dark = preprocessor.process(dark_frame)
    assert processed_dark.quality.is_low_light is True
    assert processed_dark.quality.brightness == 10.0

    # Normal lit frame (mean intensity = 130)
    lit_frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    processed_lit = preprocessor.process(lit_frame)
    assert processed_lit.quality.is_low_light is False
    assert processed_lit.quality.brightness == 130.0


def test_quality_metrics_blur():
    """Verify blur detection via Laplacian variance."""
    preprocessor = VideoPreprocessor(blur_threshold=30.0)

    # Sharp image with high frequency edges (checkerboard pattern)
    sharp_frame = np.zeros((480, 640), dtype=np.uint8)
    sharp_frame[::20, :] = 255
    sharp_frame[:, ::20] = 255
    processed_sharp = preprocessor.process(sharp_frame)
    assert processed_sharp.quality.is_blurry is False
    assert processed_sharp.quality.blur_score > 30.0

    # Extremely blurred frame
    blurred_frame = cv2.GaussianBlur(sharp_frame, (51, 51), 0)
    processed_blurred = preprocessor.process(blurred_frame)
    assert processed_blurred.quality.is_blurry is True
    assert processed_blurred.quality.blur_score < processed_sharp.quality.blur_score


def test_bounding_box_mapping_to_original():
    """Verify re-mapping detected boxes from processed frame to original resolution."""
    preprocessor = VideoPreprocessor(target_width=640, target_height=480)
    # Original is 1920x1080
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    processed = preprocessor.process(raw_frame)

    # 1. Normalized box (0.2, 0.2, 0.8, 0.8)
    norm_box = BoundingBox(xmin=0.2, ymin=0.2, xmax=0.8, ymax=0.8, confidence=0.95)
    mapped_norm = processed.map_box_to_original(norm_box)
    assert pytest.approx(mapped_norm.xmin, 0.1) == 0.2 * 1920
    assert pytest.approx(mapped_norm.ymin, 0.1) == 0.2 * 1080
    assert pytest.approx(mapped_norm.xmax, 0.1) == 0.8 * 1920
    assert pytest.approx(mapped_norm.ymax, 0.1) == 0.8 * 1080

    # 2. Processed pixel box (320, 240, 640, 480)
    pixel_box = BoundingBox(xmin=320.0, ymin=240.0, xmax=640.0, ymax=480.0, confidence=0.9)
    mapped_pixel = processed.map_box_to_original(pixel_box)
    assert pytest.approx(mapped_pixel.xmin, 0.1) == 960.0
    assert pytest.approx(mapped_pixel.ymin, 0.1) == 540.0
    assert pytest.approx(mapped_pixel.xmax, 0.1) == 1920.0
    assert pytest.approx(mapped_pixel.ymax, 0.1) == 1080.0
