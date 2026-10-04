"""Unit tests for Gaze Estimation and Screen Deviation Analysis."""

import os
import cv2
import numpy as np
import pytest

from intelliscreen.core.schemas import GazeDirection, GazeVector
from intelliscreen.vision.gaze import GazeEstimator, GazeResult
from intelliscreen.vision.head_pose import HeadPoseEstimator, HeadPoseResult
from intelliscreen.vision.landmarks import FaceLandmarkDetector, FacialLandmarkResult
from intelliscreen.vision.preprocessor import VideoPreprocessor

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


@pytest.fixture(scope="module")
def preprocessor() -> VideoPreprocessor:
    return VideoPreprocessor(target_width=640, target_height=480)


@pytest.fixture(scope="module")
def landmark_detector() -> FaceLandmarkDetector:
    return FaceLandmarkDetector()


@pytest.fixture(scope="module")
def head_pose_estimator() -> HeadPoseEstimator:
    return HeadPoseEstimator()


@pytest.fixture(scope="module")
def gaze_estimator() -> GazeEstimator:
    return GazeEstimator(yaw_threshold=25.0, pitch_threshold=20.0)


@pytest.fixture(scope="module")
def sample_face_frame(preprocessor: VideoPreprocessor):
    assert os.path.exists(SAMPLE_FACE_PATH), f"Fixture not found at {SAMPLE_FACE_PATH}"
    img = cv2.imread(SAMPLE_FACE_PATH)
    return preprocessor.process(img, frame_index=1, timestamp_ms=100.0)


@pytest.fixture(scope="module")
def blank_frame(preprocessor: VideoPreprocessor):
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    return preprocessor.process(blank, frame_index=2, timestamp_ms=200.0)


@pytest.fixture(scope="module")
def sample_landmarks_res(landmark_detector: FaceLandmarkDetector, sample_face_frame) -> FacialLandmarkResult:
    return landmark_detector.detect(sample_face_frame)


@pytest.fixture(scope="module")
def sample_head_pose_res(
    head_pose_estimator: HeadPoseEstimator,
    sample_face_frame,
    sample_landmarks_res: FacialLandmarkResult,
) -> HeadPoseResult:
    return head_pose_estimator.estimate(sample_face_frame, sample_landmarks_res)


# ============================================================================
# 1. Gaze Estimation on Sample Face Fixture
# ============================================================================

def test_gaze_estimation_sample_face(
    gaze_estimator: GazeEstimator,
    sample_face_frame,
    sample_landmarks_res: FacialLandmarkResult,
    sample_head_pose_res: HeadPoseResult,
):
    """Verify gaze estimation produces valid continuous angles, ratios, and direction enum."""
    result = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)

    assert isinstance(result, GazeResult)
    assert result.gaze_detected is True
    assert result.is_missing is False

    # Check GazeVector domain model
    assert isinstance(result.gaze_vector, GazeVector)
    assert pytest.approx(result.yaw, abs=1e-2) == result.gaze_vector.yaw
    assert pytest.approx(result.pitch, abs=1e-2) == result.gaze_vector.pitch
    assert result.direction == result.gaze_vector.direction
    assert isinstance(result.direction, GazeDirection)

    # Confidence must reflect high landmark + PnP quality
    assert result.confidence > 0.80

    # Iris ratios must be normalized within valid ocular limits [0.05, 0.95]
    lx, ly = result.left_eye_ratio
    rx, ry = result.right_eye_ratio
    assert 0.05 <= lx <= 0.95
    assert 0.05 <= ly <= 0.95
    assert 0.05 <= rx <= 0.95
    assert 0.05 <= ry <= 0.95

    # Iris centers must be valid 2D coordinates within image bounds
    assert result.left_iris_center.shape == (2,)
    assert result.right_iris_center.shape == (2,)
    assert 0.0 <= result.left_iris_center[0] <= 640.0
    assert 0.0 <= result.left_iris_center[1] <= 480.0


def test_gaze_estimation_automatic_cascade(
    gaze_estimator: GazeEstimator,
    sample_face_frame,
):
    """Verify estimator cascades internal landmark and head pose detection if omitted."""
    result = gaze_estimator.estimate(sample_face_frame)
    assert result.gaze_detected is True
    assert result.is_missing is False
    assert result.confidence > 0.80


# ============================================================================
# 2. Missing Face & Absence Handling
# ============================================================================

def test_gaze_absence_handling(
    gaze_estimator: GazeEstimator,
    blank_frame,
):
    """Verify estimator returns default non-deviated values on blank frame."""
    result = gaze_estimator.estimate(blank_frame)

    assert result.gaze_detected is False
    assert result.is_missing is True
    assert result.yaw == 0.0
    assert result.pitch == 0.0
    assert result.direction == GazeDirection.CENTER
    assert result.is_deviated is False
    assert result.confidence == 0.0
    assert result.left_eye_ratio == (0.5, 0.5)
    assert result.right_eye_ratio == (0.5, 0.5)


# ============================================================================
# 3. Gaze Deviation Logic & Direction Classification
# ============================================================================

def test_gaze_deviation_thresholds():
    """Verify is_deviated triggers when continuous angles exceed thresholds."""
    estimator = GazeEstimator(yaw_threshold=25.0, pitch_threshold=20.0)

    # Case 1: Centered gaze
    vec_center = GazeVector(yaw=5.0, pitch=-4.0, direction=GazeDirection.CENTER)
    assert not vec_center.is_deviated(yaw_threshold=25.0, pitch_threshold=20.0)

    # Case 2: Excessive horizontal deviation (looking right)
    vec_right = GazeVector(yaw=30.0, pitch=2.0, direction=GazeDirection.RIGHT)
    assert vec_right.is_deviated(yaw_threshold=25.0, pitch_threshold=20.0)

    # Case 3: Excessive vertical deviation (looking down at notes)
    vec_down = GazeVector(yaw=-2.0, pitch=-24.0, direction=GazeDirection.DOWN)
    assert vec_down.is_deviated(yaw_threshold=25.0, pitch_threshold=20.0)


def test_gaze_direction_enum_classification():
    """Verify classification across all GazeDirection enum states."""
    estimator = GazeEstimator(yaw_threshold=25.0, pitch_threshold=20.0)

    assert estimator._classify_gaze_direction(yaw=0.0, pitch=0.0) == GazeDirection.CENTER
    assert estimator._classify_gaze_direction(yaw=-30.0, pitch=0.0) == GazeDirection.LEFT
    assert estimator._classify_gaze_direction(yaw=35.0, pitch=0.0) == GazeDirection.RIGHT
    assert estimator._classify_gaze_direction(yaw=0.0, pitch=25.0) == GazeDirection.UP
    assert estimator._classify_gaze_direction(yaw=0.0, pitch=-25.0) == GazeDirection.DOWN
    assert estimator._classify_gaze_direction(yaw=-30.0, pitch=25.0) == GazeDirection.UP_LEFT
    assert estimator._classify_gaze_direction(yaw=30.0, pitch=25.0) == GazeDirection.UP_RIGHT
    assert estimator._classify_gaze_direction(yaw=-30.0, pitch=-25.0) == GazeDirection.DOWN_LEFT
    assert estimator._classify_gaze_direction(yaw=30.0, pitch=-25.0) == GazeDirection.DOWN_RIGHT


# ============================================================================
# 4. Resting Baseline Calibration
# ============================================================================

def test_resting_baseline_calibration(
    gaze_estimator: GazeEstimator,
    sample_face_frame,
    sample_landmarks_res: FacialLandmarkResult,
    sample_head_pose_res: HeadPoseResult,
):
    """Verify resting baseline calibration shifts computed gaze relative to offset."""
    # Reset calibration first
    gaze_estimator.reset_calibration()
    uncalibrated = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)

    # Calibrate candidate's current resting pose as neutral center (0, 0)
    gaze_estimator.calibrate_baseline(uncalibrated.yaw, uncalibrated.pitch)
    calibrated = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)

    # Calibrated gaze should be centered near 0.0 degrees
    assert pytest.approx(calibrated.yaw, abs=1e-2) == 0.0
    assert pytest.approx(calibrated.pitch, abs=1e-2) == 0.0
    assert calibrated.direction == GazeDirection.CENTER
    assert calibrated.is_deviated is False

    # Reset calibration and verify return to uncalibrated
    gaze_estimator.reset_calibration()
    reset_res = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)
    assert pytest.approx(reset_res.yaw, abs=1e-2) == uncalibrated.yaw
    assert pytest.approx(reset_res.pitch, abs=1e-2) == uncalibrated.pitch


# ============================================================================
# 5. Drawing Gaze Rays
# ============================================================================

def test_draw_gaze_rays(
    gaze_estimator: GazeEstimator,
    sample_face_frame,
    sample_landmarks_res: FacialLandmarkResult,
    sample_head_pose_res: HeadPoseResult,
):
    """Verify gaze rays draw arrows originating from iris centers on the image."""
    result = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)
    assert result.gaze_detected is True

    canvas = sample_face_frame.bgr_image.copy()
    rendered = gaze_estimator.draw_gaze_rays(canvas, result, length=50.0)

    assert rendered.shape == canvas.shape
    # Pixels must differ due to drawn arrows
    assert not np.array_equal(rendered, canvas)

    # Check that pixel difference occurred near left iris center
    l_iris_x, l_iris_y = int(result.left_iris_center[0]), int(result.left_iris_center[1])
    diff_iris = np.sum(np.abs(rendered[l_iris_y - 10 : l_iris_y + 10, l_iris_x - 10 : l_iris_x + 10].astype(float) -
                              canvas[l_iris_y - 10 : l_iris_y + 10, l_iris_x - 10 : l_iris_x + 10].astype(float)))
    assert diff_iris > 0.0


# ============================================================================
# 6. Gaze Latency Performance Benchmark
# ============================================================================

def test_gaze_latency_under_3ms_budget(
    gaze_estimator: GazeEstimator,
    sample_face_frame,
    sample_landmarks_res: FacialLandmarkResult,
    sample_head_pose_res: HeadPoseResult,
):
    """Verify gaze fusion calculation alone takes under 3ms on CPU."""
    # Warmup
    _ = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)

    latencies: list[float] = []
    for _ in range(50):
        res = gaze_estimator.estimate(sample_face_frame, sample_landmarks_res, sample_head_pose_res)
        latencies.append(res.latency_ms)

    mean_latency = float(np.mean(latencies))
    assert mean_latency < 3.0, f"Gaze latency {mean_latency:.4f}ms exceeds 3.0ms budget."
