"""Unit tests for 3D Head Pose Estimation (PnP)."""

import os
import cv2
import numpy as np
import pytest

from intelliscreen.core.schemas import HeadPoseAngles
from intelliscreen.vision.head_pose import HeadPoseEstimator, HeadPoseResult
from intelliscreen.vision.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    FaceLandmarkDetector,
    FacialLandmarkResult,
)
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
    return HeadPoseEstimator(yaw_threshold=30.0, pitch_threshold=25.0)


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
def sample_landmarks_result(landmark_detector: FaceLandmarkDetector, sample_face_frame) -> FacialLandmarkResult:
    return landmark_detector.detect(sample_face_frame)


# ============================================================================
# 1. Pose Estimation on Sample Face Fixture
# ============================================================================

def test_head_pose_estimation_sample_face(
    head_pose_estimator: HeadPoseEstimator,
    sample_face_frame,
    sample_landmarks_result: FacialLandmarkResult,
):
    """Verify solvePnP resolves reasonable head pose angles and high confidence."""
    result = head_pose_estimator.estimate(sample_face_frame, sample_landmarks_result)

    assert isinstance(result, HeadPoseResult)
    assert result.pose_detected is True
    assert result.is_missing is False

    # Check Euler angles structure and schema
    assert isinstance(result.euler_angles, HeadPoseAngles)
    assert pytest.approx(result.yaw, abs=1e-2) == result.euler_angles.yaw
    assert pytest.approx(result.pitch, abs=1e-2) == result.euler_angles.pitch
    assert pytest.approx(result.roll, abs=1e-2) == result.euler_angles.roll

    # Sample face is candidate turned over shoulder; angles should be within reasonable bounds
    assert -50.0 < result.yaw < 50.0
    assert -45.0 < result.pitch < 45.0
    assert -45.0 < result.roll < 45.0

    # High confidence (> 0.8) due to small reprojection error
    assert result.confidence > 0.80

    # Transform matrices shape
    assert result.rvec.shape == (3, 1)
    assert result.tvec.shape == (3, 1)
    assert result.rotation_matrix.shape == (3, 3)

    # Direction label must be populated
    assert result.direction_label in [
        "LOOKING_FORWARD",
        "LOOKING_LEFT",
        "LOOKING_RIGHT",
        "LOOKING_UP",
        "LOOKING_DOWN",
        "LOOKING_UP_LEFT",
        "LOOKING_UP_RIGHT",
        "LOOKING_DOWN_LEFT",
        "LOOKING_DOWN_RIGHT",
    ]


def test_head_pose_estimator_without_precomputed_landmarks(
    head_pose_estimator: HeadPoseEstimator,
    sample_face_frame,
):
    """Verify estimator runs internal landmark detection if landmarks_result is omitted."""
    result = head_pose_estimator.estimate(sample_face_frame, landmarks_result=None)
    assert result.pose_detected is True
    assert result.is_missing is False
    assert result.confidence > 0.80


# ============================================================================
# 2. Missing Face & Absence Handling
# ============================================================================

def test_head_pose_absence_handling(
    head_pose_estimator: HeadPoseEstimator,
    blank_frame,
):
    """Verify estimator returns standardized empty result on blank frame with no face."""
    result = head_pose_estimator.estimate(blank_frame)

    assert result.pose_detected is False
    assert result.is_missing is True
    assert result.yaw == 0.0
    assert result.pitch == 0.0
    assert result.roll == 0.0
    assert result.confidence == 0.0
    assert result.is_looking_away is False
    assert result.direction_label == "UNKNOWN"
    assert np.array_equal(result.rotation_matrix, np.eye(3))


# ============================================================================
# 3. Looking Away Logic & Thresholds
# ============================================================================

def test_looking_away_logic_thresholds():
    """Verify is_looking_away flags when yaw or pitch exceeds angular limits."""
    estimator = HeadPoseEstimator(yaw_threshold=20.0, pitch_threshold=15.0)

    # Case 1: Centered pose
    angles_center = HeadPoseAngles(yaw=5.0, pitch=-3.0, roll=1.0)
    assert not angles_center.is_looking_away(yaw_threshold=20.0, pitch_threshold=15.0)

    # Case 2: Excessive yaw looking right
    angles_right = HeadPoseAngles(yaw=28.0, pitch=2.0, roll=0.0)
    assert angles_right.is_looking_away(yaw_threshold=20.0, pitch_threshold=15.0)

    # Case 3: Excessive pitch looking down
    angles_down = HeadPoseAngles(yaw=2.0, pitch=-22.0, roll=0.0)
    assert angles_down.is_looking_away(yaw_threshold=20.0, pitch_threshold=15.0)


def test_direction_classification():
    """Verify discrete direction labels for various angular orientations."""
    estimator = HeadPoseEstimator(yaw_threshold=25.0, pitch_threshold=20.0)

    assert estimator._classify_direction(yaw=5.0, pitch=2.0) == "LOOKING_FORWARD"
    assert estimator._classify_direction(yaw=-35.0, pitch=0.0) == "LOOKING_LEFT"
    assert estimator._classify_direction(yaw=40.0, pitch=0.0) == "LOOKING_RIGHT"
    assert estimator._classify_direction(yaw=0.0, pitch=25.0) == "LOOKING_UP"
    assert estimator._classify_direction(yaw=0.0, pitch=-30.0) == "LOOKING_DOWN"
    assert estimator._classify_direction(yaw=-30.0, pitch=25.0) == "LOOKING_UP_LEFT"
    assert estimator._classify_direction(yaw=30.0, pitch=-25.0) == "LOOKING_DOWN_RIGHT"


# ============================================================================
# 4. 3D Pose Axes Projection
# ============================================================================

def test_project_pose_axes(
    head_pose_estimator: HeadPoseEstimator,
    sample_face_frame,
    sample_landmarks_result: FacialLandmarkResult,
):
    """Verify 3D coordinate axes projection modifies the image canvas at nose tip."""
    result = head_pose_estimator.estimate(sample_face_frame, sample_landmarks_result)
    assert result.pose_detected is True

    original_img = sample_face_frame.bgr_image.copy()
    axes_img = head_pose_estimator.project_pose_axes(
        original_img,
        result.rvec,
        result.tvec,
        length=60.0,
    )

    assert axes_img.shape == original_img.shape
    # Image must have been modified by drawing axes lines
    assert not np.array_equal(axes_img, original_img)

    # Nose tip coordinate region should contain red, green, or blue line pixels
    nose_x = int(sample_landmarks_result.pnp_landmarks_2d[0, 0])
    nose_y = int(sample_landmarks_result.pnp_landmarks_2d[0, 1])

    # Check a 30x30 bounding box around the nose tip
    roi_original = original_img[nose_y - 15 : nose_y + 15, nose_x - 15 : nose_x + 15]
    roi_axes = axes_img[nose_y - 15 : nose_y + 15, nose_x - 15 : nose_x + 15]
    diff = np.sum(np.abs(roi_axes.astype(float) - roi_original.astype(float)))
    assert diff > 0.0, "Expected pixel differences near nose origin from drawn axes."


# ============================================================================
# 5. PnP Execution Performance Benchmark
# ============================================================================

def test_pnp_latency_under_5ms_budget(
    head_pose_estimator: HeadPoseEstimator,
    sample_face_frame,
    sample_landmarks_result: FacialLandmarkResult,
):
    """Verify PnP execution alone completes in well under 5ms on CPU."""
    # Warmup
    _ = head_pose_estimator.estimate(sample_face_frame, sample_landmarks_result)

    latencies: list[float] = []
    for _ in range(20):
        res = head_pose_estimator.estimate(sample_face_frame, sample_landmarks_result)
        latencies.append(res.latency_ms)

    mean_latency = float(np.mean(latencies))
    assert mean_latency < 5.0, f"PnP latency {mean_latency:.3f}ms exceeds 5.0ms target budget."
