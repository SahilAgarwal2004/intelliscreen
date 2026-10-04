"""Unit tests for the Unified Vision Pipeline Orchestrator."""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from intelliscreen.core.exceptions import VisionPipelineError
from intelliscreen.core.schemas import FrameObservation
from intelliscreen.vision.pipeline import (
    VisionPipeline,
    VisionPipelineConfig,
    VisionPipelineResult,
)


@pytest.fixture
def sample_face_bgr() -> np.ndarray:
    """Fixture providing a genuine sample face BGR image."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_face.jpg"
    assert fixture_path.exists(), f"Sample face fixture not found at {fixture_path}"
    image = cv2.imread(str(fixture_path))
    assert image is not None, "Failed to decode sample face image."
    return image


@pytest.fixture
def black_frame() -> np.ndarray:
    """Fixture providing a synthetic black image (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestVisionPipelineConfig:
    """Tests for VisionPipelineConfig instantiation and settings mapping."""

    def test_default_config(self) -> None:
        config = VisionPipelineConfig()
        assert config.target_width == 640
        assert config.target_height == 480
        assert config.target_fps == 10.0
        assert config.enable_objects is True
        assert config.object_stride == 3

    def test_from_settings(self) -> None:
        config = VisionPipelineConfig.from_settings()
        assert config.target_width > 0
        assert config.target_height > 0
        assert config.target_fps > 0


class TestVisionPipelineExecution:
    """Tests for VisionPipeline perception execution and observation generation."""

    def test_pipeline_initialization(self) -> None:
        config = VisionPipelineConfig(enable_objects=False)
        pipeline = VisionPipeline(config=config)
        assert pipeline.preprocessor is not None
        assert pipeline.face_detector is not None
        assert pipeline.landmark_detector is not None
        assert pipeline.head_pose_estimator is not None
        assert pipeline.gaze_estimator is not None
        assert pipeline.object_detector is None

    def test_empty_black_frame(self, black_frame: np.ndarray) -> None:
        """Verify pipeline handles frames with no face or objects without error."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        result = pipeline.process_frame(black_frame, frame_index=0, timestamp_ms=0.0)

        assert isinstance(result, VisionPipelineResult)
        obs = result.observation
        assert isinstance(obs, FrameObservation)
        assert obs.frame_index == 0
        assert obs.timestamp_ms == 0.0
        assert obs.face_detected is False
        assert obs.face_count == 0
        assert obs.primary_face_box is None
        assert obs.head_pose is None
        assert obs.gaze is None
        assert obs.detected_objects == []
        assert result.is_suspicious_instant is True

    def test_sample_face_observation_aggregation(self, sample_face_bgr: np.ndarray) -> None:
        """Verify pipeline correctly detects and aggregates face, pose, gaze, and objects."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        result = pipeline.process_frame(sample_face_bgr)

        assert isinstance(result, VisionPipelineResult)
        obs = result.observation

        # Verify face detection
        assert obs.face_detected is True
        assert obs.face_count >= 1
        assert obs.primary_face_box is not None
        assert obs.primary_face_box.width > 50
        assert obs.primary_face_box.height > 50

        # Verify 3D Head Pose
        assert obs.head_pose is not None
        assert abs(obs.head_pose.yaw) < 90.0
        assert abs(obs.head_pose.pitch) < 90.0
        assert abs(obs.head_pose.roll) < 90.0
        assert obs.head_pose.confidence > 0.0

        # Verify Gaze Vector
        assert obs.gaze is not None
        assert obs.gaze.direction is not None
        assert obs.gaze.confidence > 0.0

        # Verify Detailed Sub-results
        assert result.face_result.face_detected is True
        assert result.landmark_result.landmarks_detected is True
        assert len(result.landmark_result.landmarks) == 478
        assert result.head_pose_result.pose_detected is True
        assert result.gaze_result.gaze_detected is True

        # Verify Latency Breakdown
        assert result.total_latency_ms > 0.0
        assert "preprocessor" in result.component_latencies
        assert "face_detector" in result.component_latencies
        assert "landmarks" in result.component_latencies
        assert "head_pose" in result.component_latencies
        assert "gaze" in result.component_latencies

    def test_process_observation_shortcut(self, sample_face_bgr: np.ndarray) -> None:
        """Verify lightweight process_observation returns a validated FrameObservation."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        obs = pipeline.process_observation(sample_face_bgr, frame_index=42, timestamp_ms=4200.0)

        assert isinstance(obs, FrameObservation)
        assert obs.frame_index == 42
        assert obs.timestamp_ms == 4200.0
        assert obs.face_detected is True

    def test_auto_increment_frame_counter_and_timestamps(self, black_frame: np.ndarray) -> None:
        """Verify sequential frames automatically advance indices and timestamps."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(target_fps=10.0, enable_objects=False)
        )

        r0 = pipeline.process_frame(black_frame)
        r1 = pipeline.process_frame(black_frame)
        r2 = pipeline.process_frame(black_frame)

        assert r0.observation.frame_index == 0
        assert r0.observation.timestamp_ms == 0.0

        assert r1.observation.frame_index == 1
        assert pytest.approx(r1.observation.timestamp_ms, 0.1) == 100.0

        assert r2.observation.frame_index == 2
        assert pytest.approx(r2.observation.timestamp_ms, 0.1) == 200.0

    def test_gaze_calibration_forwarding(self) -> None:
        """Verify pipeline forwards calibration to internal gaze estimator."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        assert pipeline.gaze_estimator._baseline_yaw == 0.0
        assert pipeline.gaze_estimator._baseline_pitch == 0.0

        pipeline.calibrate_gaze(yaw=7.5, pitch=-3.2)
        assert pipeline.gaze_estimator._baseline_yaw == 7.5
        assert pipeline.gaze_estimator._baseline_pitch == -3.2

        pipeline.reset_gaze_calibration()
        assert pipeline.gaze_estimator._baseline_yaw == 0.0
        assert pipeline.gaze_estimator._baseline_pitch == 0.0

    def test_diagnostics_and_reset(self, sample_face_bgr: np.ndarray) -> None:
        """Verify cumulative performance diagnostics and reset behavior."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        pipeline.process_frame(sample_face_bgr)
        pipeline.process_frame(sample_face_bgr)

        diag = pipeline.get_diagnostics()
        assert diag["total_frames_processed"] == 2
        assert diag["average_latency_ms"] > 0.0
        assert diag["face_detection_rate"] == 1.0

        pipeline.reset()
        diag_after = pipeline.get_diagnostics()
        assert diag_after["total_frames_processed"] == 0
        assert diag_after["average_latency_ms"] == 0.0
        assert diag_after["face_detection_rate"] == 0.0

    def test_annotate_frame(self, sample_face_bgr: np.ndarray) -> None:
        """Verify annotate_frame produces valid annotated image canvas."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False)
        )
        result = pipeline.process_frame(sample_face_bgr)

        annotated = pipeline.annotate_frame(result=result)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (480, 640, 3)
        assert annotated.dtype == np.uint8

        # Verify error handling when result is omitted
        with pytest.raises(VisionPipelineError):
            pipeline.annotate_frame(result=None)

    def test_pipeline_object_stride_caching(self, sample_face_bgr: np.ndarray) -> None:
        """Verify object detection executes inference on frame 0 and caches frames 1 and 2."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=True, object_stride=3)
        )
        r0 = pipeline.process_frame(sample_face_bgr)
        r1 = pipeline.process_frame(sample_face_bgr)
        r2 = pipeline.process_frame(sample_face_bgr)
        r3 = pipeline.process_frame(sample_face_bgr)

        assert r0.object_result.is_cached is False
        assert r1.object_result.is_cached is True
        assert r2.object_result.is_cached is True
        assert r3.object_result.is_cached is False

    def test_pipeline_force_object_inference(self, sample_face_bgr: np.ndarray) -> None:
        """Verify force_object_inference bypasses temporal stride cache."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=True, object_stride=3)
        )
        r0 = pipeline.process_frame(sample_face_bgr)
        r1 = pipeline.process_frame(sample_face_bgr, force_object_inference=True)

        assert r0.object_result.is_cached is False
        assert r1.object_result.is_cached is False

    def test_pipeline_quality_assessment(self) -> None:
        """Verify pipeline measures low-light and blur conditions."""
        pipeline = VisionPipeline(
            config=VisionPipelineConfig(enable_objects=False, low_light_threshold=40.0)
        )
        dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        r_dark = pipeline.process_frame(dark_frame)

        assert r_dark.preprocessed.quality.is_low_light is True
        assert r_dark.preprocessed.quality.brightness < 40.0

