"""Unit tests for dense Facial Landmark Detection, EAR computation, and anatomical mapping."""

import os
import cv2
import numpy as np
import pytest

from intelliscreen.core.schemas import BoundingBox, NormalizedLandmark
from intelliscreen.vision.face_detector import DetectedFace, FaceDetector
from intelliscreen.vision.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    CHIN_INDICES,
    FaceLandmarkDetector,
    FacialLandmarkResult,
    LEFT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    MOUTH_INDICES,
    NOSE_BRIDGE_INDICES,
    PNP_6POINT_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_IRIS_INDICES,
    compute_eye_aspect_ratio,
)
from intelliscreen.vision.preprocessor import VideoPreprocessor

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


@pytest.fixture(scope="module")
def preprocessor() -> VideoPreprocessor:
    """Preprocess images to standard 640x480 resolution."""
    return VideoPreprocessor(target_width=640, target_height=480)


@pytest.fixture(scope="module")
def sample_face_frame(preprocessor: VideoPreprocessor):
    """Load and preprocess the standard test face image."""
    assert os.path.exists(SAMPLE_FACE_PATH), f"Fixture not found at {SAMPLE_FACE_PATH}"
    img = cv2.imread(SAMPLE_FACE_PATH)
    return preprocessor.process(img, frame_index=1, timestamp_ms=100.0)


@pytest.fixture(scope="module")
def blank_frame(preprocessor: VideoPreprocessor):
    """Generate and preprocess a blank neutral image with no face."""
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    return preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)


@pytest.fixture(scope="module")
def sample_detected_face() -> DetectedFace:
    """Fixture returning a mock DetectedFace with YuNet keypoints."""
    return DetectedFace(
        box=BoundingBox(xmin=200.0, ymin=100.0, xmax=440.0, ymax=380.0, confidence=0.95, label="face"),
        confidence=0.95,
        keypoints={
            "right_eye": (265.0, 200.0),
            "left_eye": (375.0, 200.0),
            "nose_tip": (320.0, 250.0),
            "right_mouth": (280.0, 310.0),
            "left_mouth": (360.0, 310.0),
        },
    )


@pytest.fixture(scope="module")
def landmark_detector() -> FaceLandmarkDetector:
    """Singleton instance of the FaceLandmarkDetector."""
    return FaceLandmarkDetector(blink_ear_threshold=0.20)


# ============================================================================
# 1. Detection on Sample Face Fixture
# ============================================================================

def test_dense_landmark_detection_on_sample_face(landmark_detector, sample_face_frame):
    """Verify detector extracts 478 dense 3D landmarks on a valid face image."""
    result = landmark_detector.detect(sample_face_frame)

    assert isinstance(result, FacialLandmarkResult)
    assert result.landmarks_detected is True
    assert result.is_missing is False
    assert result.is_fallback is False

    # Check dense landmarks count (478: 468 mesh + 10 iris)
    assert len(result.landmarks) == 478
    assert isinstance(result.landmarks[0], NormalizedLandmark)

    # Check 2D and 3D coordinate arrays
    assert result.landmark_points_2d.shape == (478, 2)
    assert result.landmark_points_3d.shape == (478, 3)
    assert result.normalized_points_2d.shape == (478, 2)
    assert result.normalized_points_3d.shape == (478, 3)

    # Coordinates must be within normalized bounds [0, 1]
    assert np.all(result.normalized_points_2d[:, 0] >= 0.0)
    assert np.all(result.normalized_points_2d[:, 0] <= 1.0)
    assert np.all(result.normalized_points_2d[:, 1] >= 0.0)
    assert np.all(result.normalized_points_2d[:, 1] <= 1.0)

    # Pixel coordinates must align with frame dimensions (640x480)
    assert np.all(result.landmark_points_2d[:, 0] >= 0.0)
    assert np.all(result.landmark_points_2d[:, 0] <= 640.0)
    assert np.all(result.landmark_points_2d[:, 1] >= 0.0)
    assert np.all(result.landmark_points_2d[:, 1] <= 480.0)


def test_landmark_detector_performance_under_budget(landmark_detector, sample_face_frame):
    """Verify landmark detection inference latency is well under the 15ms CPU budget."""
    # Warmup
    _ = landmark_detector.detect(sample_face_frame)

    latencies: list[float] = []
    for _ in range(5):
        res = landmark_detector.detect(sample_face_frame)
        latencies.append(res.latency_ms)

    mean_latency = float(np.mean(latencies))
    assert mean_latency < 15.0, f"Mean latency {mean_latency:.2f}ms exceeds 15ms target."


# ============================================================================
# 2. Missing Face & Absence Handling
# ============================================================================

def test_missing_face_detection(landmark_detector, blank_frame):
    """Verify detector cleanly identifies missing face on a blank frame."""
    result = landmark_detector.detect(blank_frame)

    assert result.landmarks_detected is False
    assert result.is_missing is True
    assert len(result.landmarks) == 0
    assert result.landmark_points_2d.shape == (0, 2)
    assert result.landmark_points_3d.shape == (0, 3)
    assert result.left_ear == 0.0
    assert result.right_ear == 0.0
    assert result.avg_ear == 0.0
    assert result.is_blinking is False
    assert result.pnp_landmarks_2d.shape == (0, 2)


# ============================================================================
# 3. Eye Aspect Ratio (EAR) & Blink Calculation
# ============================================================================

def test_eye_aspect_ratio_calculation():
    """Verify EAR calculation formula on synthetic geometry."""
    # Construct an open eye: width = 20, height = 8
    # [p1_corner, p2_upper1, p3_upper2, p4_corner, p5_lower2, p6_lower1]
    open_eye = np.array(
        [
            [10.0, 20.0],  # p1: left corner
            [15.0, 16.0],  # p2: upper 1
            [25.0, 16.0],  # p3: upper 2
            [30.0, 20.0],  # p4: right corner
            [25.0, 24.0],  # p5: lower 2
            [15.0, 24.0],  # p6: lower 1
        ],
        dtype=np.float32,
    )
    # dist_v1 = ||(15,16) - (15,24)|| = 8.0
    # dist_v2 = ||(25,16) - (25,24)|| = 8.0
    # dist_h  = ||(10,20) - (30,20)|| = 20.0
    # EAR = (8 + 8) / (2 * 20) = 16 / 40 = 0.40
    open_ear = compute_eye_aspect_ratio(open_eye)
    assert pytest.approx(open_ear, abs=1e-3) == 0.40

    # Construct a closed eye (blinking): vertical height collapsed to 1.0
    closed_eye = np.array(
        [
            [10.0, 20.0],
            [15.0, 19.5],
            [25.0, 19.5],
            [30.0, 20.0],
            [25.0, 20.5],
            [15.0, 20.5],
        ],
        dtype=np.float32,
    )
    closed_ear = compute_eye_aspect_ratio(closed_eye)
    assert pytest.approx(closed_ear, abs=1e-3) == 0.05
    assert closed_ear < 0.20


def test_ear_metrics_on_sample_face(landmark_detector, sample_face_frame):
    """Verify EAR values on candidate with open eyes."""
    result = landmark_detector.detect(sample_face_frame)

    assert result.left_ear > 0.25, f"Left EAR {result.left_ear} should be > 0.25 for open eye."
    assert result.right_ear > 0.25, f"Right EAR {result.right_ear} should be > 0.25 for open eye."
    assert result.avg_ear > 0.25
    assert result.is_blinking is False


def test_ear_zero_division_guard():
    """Verify compute_eye_aspect_ratio returns 0.0 for degenerate zero-width eyes."""
    degenerate_eye = np.zeros((6, 2), dtype=np.float32)
    ear = compute_eye_aspect_ratio(degenerate_eye)
    assert ear == 0.0


# ============================================================================
# 4. Anatomical Region Mapping & Access
# ============================================================================

def test_anatomical_region_indices_and_points(landmark_detector, sample_face_frame):
    """Verify all anatomical region indices are present and point queries return correct arrays."""
    result = landmark_detector.detect(sample_face_frame)

    # Check attribute index lists
    assert result.left_eye == LEFT_EYE_INDICES
    assert result.right_eye == RIGHT_EYE_INDICES
    assert result.nose_bridge == NOSE_BRIDGE_INDICES
    assert result.chin == CHIN_INDICES
    assert result.mouth == MOUTH_INDICES
    assert result.left_iris == LEFT_IRIS_INDICES
    assert result.right_iris == RIGHT_IRIS_INDICES

    # Check get_region_points_2d
    left_eye_pts = result.get_region_points_2d("left_eye")
    assert left_eye_pts.shape == (len(LEFT_EYE_INDICES), 2)

    right_eye_pts = result.get_region_points_2d("right_eye")
    assert right_eye_pts.shape == (len(RIGHT_EYE_INDICES), 2)

    nose_pts = result.get_region_points_2d("nose_bridge")
    assert nose_pts.shape == (len(NOSE_BRIDGE_INDICES), 2)

    chin_pts = result.get_region_points_2d("chin")
    assert chin_pts.shape == (len(CHIN_INDICES), 2)

    mouth_pts = result.get_region_points_2d("mouth")
    assert mouth_pts.shape == (len(MOUTH_INDICES), 2)

    # Check 3D point retrieval
    left_eye_3d = result.get_region_points_3d("left_eye")
    assert left_eye_3d.shape == (len(LEFT_EYE_INDICES), 3)

    # Check custom index list query
    custom_pts = result.get_region_points_2d([1, 152])
    assert custom_pts.shape == (2, 2)


def test_invalid_region_name_raises_error(landmark_detector, sample_face_frame):
    """Verify querying an invalid anatomical region name raises ValueError."""
    result = landmark_detector.detect(sample_face_frame)
    with pytest.raises(ValueError, match="Unknown anatomical region"):
        result.get_region_points_2d("non_existent_organ")


# ============================================================================
# 5. Key Anatomical Anchor Subsets for PnP and Gaze
# ============================================================================

def test_pnp_6point_anchors(landmark_detector, sample_face_frame):
    """Verify canonical 6 anthropometric PnP anchors are extracted with correct order."""
    result = landmark_detector.detect(sample_face_frame)

    assert result.pnp_landmarks_2d.shape == (6, 2)
    assert result.pnp_landmarks_3d.shape == (6, 3)

    pnp_points = result.get_pnp_6points()
    assert np.array_equal(pnp_points, result.pnp_landmarks_2d)

    # Index 0 is Nose tip, Index 1 is Chin.
    # On an upright face, Chin y must be greater than Nose tip y.
    nose_y = result.pnp_landmarks_2d[0, 1]
    chin_y = result.pnp_landmarks_2d[1, 1]
    assert chin_y > nose_y, "Chin must be vertically below nose tip on sample face."

    # Index 2 is Left eye outer (camera right), Index 3 is Right eye outer (camera left).
    # Camera right x should be greater than camera left x.
    left_eye_x = result.pnp_landmarks_2d[2, 0]
    right_eye_x = result.pnp_landmarks_2d[3, 0]
    assert left_eye_x > right_eye_x, "Left eye outer corner should be to the right of right eye outer corner."

    # Validate CANONICAL_FACE_MODEL_3D reference shape
    assert CANONICAL_FACE_MODEL_3D.shape == (6, 3)
    assert CANONICAL_FACE_MODEL_3D.dtype == np.float32


def test_gaze_anchors_subset(landmark_detector, sample_face_frame):
    """Verify gaze vectorization anchors (iris centers and eye corners) are populated."""
    result = landmark_detector.detect(sample_face_frame)

    assert "left_iris_center_2d" in result.gaze_anchors
    assert "right_iris_center_2d" in result.gaze_anchors
    assert "left_eye_inner_corner_2d" in result.gaze_anchors
    assert "left_eye_outer_corner_2d" in result.gaze_anchors
    assert "right_eye_inner_corner_2d" in result.gaze_anchors
    assert "right_eye_outer_corner_2d" in result.gaze_anchors

    left_iris = result.gaze_anchors["left_iris_center_2d"]
    right_iris = result.gaze_anchors["right_iris_center_2d"]

    assert left_iris.shape == (2,)
    assert right_iris.shape == (2,)
    # Left iris (camera right) x should be greater than right iris x
    assert left_iris[0] > right_iris[0]


# ============================================================================
# 6. Fallback Mode & Robustness
# ============================================================================

def test_geometric_fallback_with_primary_face(sample_face_frame, sample_detected_face):
    """Verify fallback detector synthesizes valid landmarks from primary_face."""
    detector = FaceLandmarkDetector(backend="fallback")
    result = detector.detect(sample_face_frame, primary_face=sample_detected_face)

    assert result.landmarks_detected is True
    assert result.is_missing is False
    assert result.is_fallback is True

    # Check PnP anchors shape
    assert result.pnp_landmarks_2d.shape == (6, 2)
    assert result.pnp_landmarks_3d.shape == (6, 3)

    # Check that chin is below nose tip
    nose_y = result.pnp_landmarks_2d[0, 1]
    chin_y = result.pnp_landmarks_2d[1, 1]
    assert chin_y > nose_y

    # Check fallback EAR values
    assert result.avg_ear == 0.28
    assert result.is_blinking is False

    # Check safe region access on fallback result
    pts = result.get_region_points_2d("left_eye")
    assert pts.shape == (len(LEFT_EYE_INDICES), 2)


def test_invalid_backend_configuration():
    """Verify initializing with invalid backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        FaceLandmarkDetector(backend="invalid_mesh_engine")
