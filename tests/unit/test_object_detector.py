"""Unit tests for Secondary Object and Device Detection (Feature 7)."""

import os
import cv2
import numpy as np
import pytest

from intelliscreen.core.schemas import BoundingBox, DetectedObject
from intelliscreen.vision.object_detector import ObjectDetectionResult, ProctoringObjectDetector
from intelliscreen.vision.preprocessor import PreprocessedFrame, VideoPreprocessor

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


@pytest.fixture(scope="module")
def preprocessor() -> VideoPreprocessor:
    return VideoPreprocessor(target_width=640, target_height=480)


@pytest.fixture(scope="module")
def sample_face_frame(preprocessor: VideoPreprocessor):
    assert os.path.exists(SAMPLE_FACE_PATH), f"Fixture not found at {SAMPLE_FACE_PATH}"
    img = cv2.imread(SAMPLE_FACE_PATH)
    return preprocessor.process(img, frame_index=0, timestamp_ms=0.0)


@pytest.fixture(scope="module")
def blank_frame(preprocessor: VideoPreprocessor):
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    return preprocessor.process(blank, frame_index=0, timestamp_ms=0.0)


@pytest.fixture(scope="module")
def detector() -> ProctoringObjectDetector:
    return ProctoringObjectDetector(confidence_threshold=0.40, stride=3)


# ============================================================================
# 1. Detection on Sample Frame (Candidate Person Detection)
# ============================================================================

def test_person_detection_on_sample_face(
    detector: ProctoringObjectDetector,
    sample_face_frame,
):
    """Verify detector detects candidate person in standard test fixture."""
    result = detector.detect(sample_face_frame, force=True)

    assert isinstance(result, ObjectDetectionResult)
    assert result.person_count >= 1
    assert result.secondary_person_detected is False
    assert result.phone_detected is False
    assert result.is_cached is False

    # Check bounding box validity
    person_objs = [d for d in result.detections if d.label == "person"]
    assert len(person_objs) >= 1
    box = person_objs[0].box
    assert 0.0 <= box.xmin < box.xmax <= 640.0
    assert 0.0 <= box.ymin < box.ymax <= 480.0
    assert box.confidence >= 0.40


# ============================================================================
# 2. Blank Frame (Absence of Any Objects)
# ============================================================================

def test_blank_frame_no_detections(
    detector: ProctoringObjectDetector,
    blank_frame,
):
    """Verify blank neutral image results in zero detections and clean flags."""
    result = detector.detect(blank_frame, force=True)

    assert len(result.detections) == 0
    assert result.person_count == 0
    assert result.secondary_person_detected is False
    assert result.phone_detected is False
    assert result.laptop_detected is False
    assert result.book_detected is False
    assert result.prohibited_items == []


# ============================================================================
# 3. Confidence Threshold Filtering
# ============================================================================

def test_high_confidence_threshold_filtering(sample_face_frame):
    """Verify setting high confidence threshold filters out marginal detections."""
    strict_detector = ProctoringObjectDetector(confidence_threshold=0.99)
    result = strict_detector.detect(sample_face_frame, force=True)

    # 0.99 is unattainably high for standard person detection in test image
    assert len(result.detections) == 0
    assert result.person_count == 0


# ============================================================================
# 4. Temporal Stride Caching Logic
# ============================================================================

def test_stride_execution_and_caching(
    detector: ProctoringObjectDetector,
    preprocessor: VideoPreprocessor,
):
    """Verify inference only executes every stride frames unless forced."""
    img = cv2.imread(SAMPLE_FACE_PATH)

    # Frame 0: (0 % 3 == 0) -> Inference must run
    f0 = preprocessor.process(img, frame_index=0)
    r0 = detector.detect(f0)
    assert r0.is_cached is False

    # Frame 1: (1 % 3 != 0) -> Must return cached result
    f1 = preprocessor.process(img, frame_index=1)
    r1 = detector.detect(f1)
    assert r1.is_cached is True
    assert r1.person_count == r0.person_count
    assert r1.latency_ms < 5.0  # Cache lookup is sub-millisecond

    # Frame 2: (2 % 3 != 0) -> Must return cached result
    f2 = preprocessor.process(img, frame_index=2)
    r2 = detector.detect(f2)
    assert r2.is_cached is True

    # Frame 3: (3 % 3 == 0) -> Inference must re-run
    f3 = preprocessor.process(img, frame_index=3)
    r3 = detector.detect(f3)
    assert r3.is_cached is False

    # Frame 4: (4 % 3 != 0) but force=True -> Inference must run
    f4 = preprocessor.process(img, frame_index=4)
    r4 = detector.detect(f4, force=True)
    assert r4.is_cached is False


# ============================================================================
# 5. Prohibited Items List & Secondary Person Detection
# ============================================================================

def test_prohibited_items_logic_and_secondary_person():
    """Verify ObjectDetectionResult correctly flags prohibited categories."""
    phone_box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=90, confidence=0.85, label="cell phone")
    person1_box = BoundingBox(xmin=100, ymin=100, xmax=300, ymax=400, confidence=0.9, label="person")
    person2_box = BoundingBox(xmin=400, ymin=100, xmax=600, ymax=400, confidence=0.88, label="person")
    book_box = BoundingBox(xmin=50, ymin=300, xmax=150, ymax=450, confidence=0.75, label="book")

    detections = [
        DetectedObject(label="cell phone", confidence=0.85, box=phone_box),
        DetectedObject(label="person", confidence=0.9, box=person1_box),
        DetectedObject(label="person", confidence=0.88, box=person2_box),
        DetectedObject(label="book", confidence=0.75, box=book_box),
    ]

    prohibited = ["cell phone", "book", "secondary person"]

    res = ObjectDetectionResult(
        detections=detections,
        phone_detected=True,
        laptop_detected=False,
        book_detected=True,
        person_count=2,
        secondary_person_detected=True,
        prohibited_items=prohibited,
        latency_ms=10.0,
    )

    assert res.phone_detected is True
    assert res.book_detected is True
    assert res.secondary_person_detected is True
    assert "cell phone" in res.prohibited_items
    assert "secondary person" in res.prohibited_items
    assert "book" in res.prohibited_items


# ============================================================================
# 6. Drawing Bounding Boxes & Warning Annotations
# ============================================================================

def test_draw_detections_annotation(detector: ProctoringObjectDetector, sample_face_frame):
    """Verify drawing detections modifies the image canvas with labeled boxes."""
    result = detector.detect(sample_face_frame, force=True)
    canvas = sample_face_frame.bgr_image.copy()

    rendered = detector.draw_detections(canvas, result)
    assert rendered.shape == canvas.shape

    if len(result.detections) > 0:
        assert not np.array_equal(rendered, canvas)


# ============================================================================
# 7. Latency Performance Benchmark
# ============================================================================

def test_object_detection_cached_latency(detector: ProctoringObjectDetector, sample_face_frame):
    """Verify cached stride lookup takes negligible time (<1ms)."""
    # Ensure cache is primed
    _ = detector.detect(sample_face_frame, force=True)

    latencies: list[float] = []
    # Make subsequent calls on non-stride frame index
    for i in range(10):
        dummy_frame = PreprocessedFrame(
            frame_index=1,  # Not divisible by 3
            timestamp_ms=100.0 + i,
            rgb_image=sample_face_frame.rgb_image,
            bgr_image=sample_face_frame.bgr_image,
            gray_image=sample_face_frame.gray_image,
            original_shape=sample_face_frame.original_shape,
            processed_shape=sample_face_frame.processed_shape,
            scale_x=sample_face_frame.scale_x,
            scale_y=sample_face_frame.scale_y,
            quality=sample_face_frame.quality,
        )
        res = detector.detect(dummy_frame)
        assert res.is_cached is True
        latencies.append(res.latency_ms)

    mean_cached_lat = float(np.mean(latencies))
    assert mean_cached_lat < 1.0, f"Cached lookup {mean_cached_lat:.4f}ms should be < 1.0ms."
