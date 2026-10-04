"""Unit tests for FaceDetector, YuNet, and Haar Cascade backends."""

import os
import cv2
import numpy as np
import pytest

from intelliscreen.vision.face_detector import FaceDetector, HaarCascadeFaceDetector, YuNetFaceDetector
from intelliscreen.vision.preprocessor import VideoPreprocessor

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


@pytest.fixture(scope="module")
def preprocessor() -> VideoPreprocessor:
    return VideoPreprocessor(target_width=640, target_height=480)


@pytest.fixture(scope="module")
def sample_face_frame(preprocessor: VideoPreprocessor):
    """Load and preprocess the standard test face image."""
    assert os.path.exists(SAMPLE_FACE_PATH), f"Fixture not found at {SAMPLE_FACE_PATH}"
    img = cv2.imread(SAMPLE_FACE_PATH)
    return preprocessor.process(img, frame_index=1, timestamp_ms=100.0)


@pytest.fixture(scope="module")
def empty_frame(preprocessor: VideoPreprocessor):
    """Generate and preprocess a blank neutral image with no face."""
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    return preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)


@pytest.fixture(scope="module")
def multi_face_frame(preprocessor: VideoPreprocessor):
    """Generate a frame with two faces side-by-side."""
    img = cv2.imread(SAMPLE_FACE_PATH)
    resized = cv2.resize(img, (240, 240))
    canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Paste face 1 on left
    canvas[120:360, 60:300] = resized
    # Paste face 2 on right
    canvas[120:360, 340:580] = resized
    return preprocessor.process(canvas, frame_index=3, timestamp_ms=300.0)


def test_yunet_single_face_detection(sample_face_frame):
    """Verify YuNet reliably detects a single face with anatomical keypoints."""
    detector = FaceDetector(backend="yunet", confidence_threshold=0.6)
    result = detector.detect(sample_face_frame)

    assert result.face_detected is True
    assert result.face_count == 1
    assert result.is_missing is False
    assert result.is_multiple_faces is False
    assert result.primary_face is not None
    assert result.primary_face.confidence >= 0.6
    assert result.latency_ms > 0.0

    # Verify 5 anatomical keypoints
    kp = result.primary_face.keypoints
    assert "right_eye" in kp
    assert "left_eye" in kp
    assert "nose_tip" in kp
    assert "right_mouth" in kp
    assert "left_mouth" in kp


def test_face_absence_detection(empty_frame):
    """Verify detector correctly identifies empty scene as missing face."""
    detector = FaceDetector(backend="yunet")
    result = detector.detect(empty_frame)

    assert result.face_detected is False
    assert result.face_count == 0
    assert result.primary_face is None
    assert result.is_missing is True
    assert result.is_multiple_faces is False


def test_multiple_faces_detection(multi_face_frame):
    """Verify detector correctly flags multiple face presence in candidate frame."""
    detector = FaceDetector(backend="yunet", confidence_threshold=0.5)
    result = detector.detect(multi_face_frame)

    assert result.face_detected is True
    assert result.face_count == 2
    assert result.is_multiple_faces is True
    assert result.is_missing is False
    assert result.primary_face is not None


def test_confidence_threshold_filtering(sample_face_frame):
    """Verify that detections below the confidence threshold are discarded."""
    # An unattainable threshold of 0.999 should filter out all faces
    detector = FaceDetector(backend="yunet", confidence_threshold=0.999)
    result = detector.detect(sample_face_frame)

    assert result.face_detected is False
    assert result.face_count == 0
    assert result.is_missing is True


def test_haar_cascade_backend(sample_face_frame):
    """Verify offline Haar Cascade fallback detects face."""
    detector = FaceDetector(backend="haar")
    result = detector.detect(sample_face_frame)

    assert result.face_detected is True
    assert result.face_count >= 1
    assert result.primary_face is not None


def test_invalid_backend_error():
    """Verify that specifying an unsupported backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown face detector backend"):
        FaceDetector(backend="non_existent_engine")
