"""Unified Vision Pipeline Orchestrator and Observation Aggregator for IntelliScreen.

Coordinates the end-to-end computer vision perception stack:
  VideoPreprocessor -> FaceDetector -> FaceLandmarkDetector ->
  HeadPoseEstimator -> GazeEstimator -> ProctoringObjectDetector
and produces unified, validated FrameObservation domain models.
"""

from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.exceptions import VisionPipelineError
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import (
    BoundingBox,
    DetectedObject,
    FrameObservation,
    GazeVector,
    HeadPoseAngles,
)
from intelliscreen.vision.face_detector import (
    DetectedFace,
    FaceDetectionResult,
    FaceDetector,
)
from intelliscreen.vision.gaze import (
    GazeEstimator,
    GazeResult,
)
from intelliscreen.vision.head_pose import (
    HeadPoseEstimator,
    HeadPoseResult,
)
from intelliscreen.vision.landmarks import (
    FaceLandmarkDetector,
    FacialLandmarkResult,
)
from intelliscreen.vision.object_detector import (
    ObjectDetectionResult,
    ProctoringObjectDetector,
)
from intelliscreen.vision.preprocessor import (
    PreprocessedFrame,
    VideoPreprocessor,
)

logger = get_logger("vision_pipeline")


@dataclass
class VisionPipelineConfig:
    """Configurable options and operational thresholds for VisionPipeline."""

    target_width: int = 640
    target_height: int = 480
    target_fps: float = 10.0
    face_backend: str = "yunet"
    face_confidence: float = 0.5
    landmark_backend: str = "mediapipe"
    head_yaw_threshold: float = 30.0
    head_pitch_threshold: float = 25.0
    gaze_yaw_threshold: float = 25.0
    gaze_pitch_threshold: float = 20.0
    gaze_iris_yaw_scale: float = 45.0
    gaze_iris_pitch_scale: float = 35.0
    enable_objects: bool = True
    object_stride: int = 3
    object_confidence: float = 0.4
    low_light_threshold: float = 40.0
    blur_threshold: float = 40.0

    @classmethod
    def from_settings(cls) -> "VisionPipelineConfig":
        """Instantiate configuration directly from centralized system settings."""
        settings = get_settings()
        return cls(
            target_width=settings.vision_frame_width,
            target_height=settings.vision_frame_height,
            target_fps=settings.vision_target_fps,
            face_confidence=settings.face_confidence_threshold,
            head_yaw_threshold=settings.head_yaw_threshold,
            head_pitch_threshold=settings.head_pitch_threshold,
            gaze_yaw_threshold=settings.gaze_yaw_threshold,
            gaze_pitch_threshold=settings.gaze_pitch_threshold,
            gaze_iris_yaw_scale=settings.gaze_iris_yaw_scale,
            gaze_iris_pitch_scale=settings.gaze_iris_pitch_scale,
            object_stride=settings.vision_object_detection_stride,
            object_confidence=settings.object_confidence_threshold,
        )



@dataclass
class VisionPipelineResult:
    """Consolidated multi-model output for a single processed video frame."""

    observation: FrameObservation
    preprocessed: PreprocessedFrame
    face_result: FaceDetectionResult
    landmark_result: FacialLandmarkResult
    head_pose_result: HeadPoseResult
    gaze_result: GazeResult
    object_result: ObjectDetectionResult
    total_latency_ms: float
    component_latencies: dict[str, float] = field(default_factory=dict)

    @property
    def is_suspicious_instant(self) -> bool:
        """Instantaneous heuristic check for candidate divergence or unauthorized objects."""
        if not self.observation.face_detected:
            return True
        if self.observation.face_count > 1:
            return True
        if self.head_pose_result.is_looking_away:
            return True
        if self.gaze_result.is_deviated:
            return True
        if self.object_result.phone_detected or self.object_result.book_detected:
            return True
        return False


class VisionPipeline:
    """Unified single-frame and streaming perception pipeline orchestrator.

    Integrates frame preprocessing, face localization, 478-point landmark mesh,
    PnP 3D head pose estimation, gaze ray vectorization, and YOLOv8n proctoring
    object detection into an aggregated FrameObservation domain model.
    """

    def __init__(
        self,
        config: VisionPipelineConfig | None = None,
        landmark_detector: FaceLandmarkDetector | None = None,
        head_pose_estimator: HeadPoseEstimator | None = None,
        gaze_estimator: GazeEstimator | None = None,
        object_detector: ProctoringObjectDetector | None = None,
    ) -> None:
        """Initialize the unified vision perception pipeline.

        Args:
            config: Operational configuration settings.
            landmark_detector: Optional injected FaceLandmarkDetector.
            head_pose_estimator: Optional injected HeadPoseEstimator.
            gaze_estimator: Optional injected GazeEstimator.
            object_detector: Optional injected ProctoringObjectDetector.
        """
        self.config = config or VisionPipelineConfig.from_settings()

        # 1. Video Preprocessor
        self.preprocessor = VideoPreprocessor(
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            low_light_threshold=self.config.low_light_threshold,
            blur_threshold=self.config.blur_threshold,
        )

        # 2. Face Detector
        self.face_detector = FaceDetector(
            backend=self.config.face_backend,
            confidence_threshold=self.config.face_confidence,
        )

        # 3. Dense Landmark Detector (MediaPipe 478-point mesh)
        self.landmark_detector = landmark_detector or FaceLandmarkDetector(
            backend=self.config.landmark_backend,
        )

        # 4. 3D Head Pose Estimator (solvePnP)
        self.head_pose_estimator = head_pose_estimator or HeadPoseEstimator(
            yaw_threshold=self.config.head_yaw_threshold,
            pitch_threshold=self.config.head_pitch_threshold,
        )
        self.head_pose_estimator._landmark_detector = self.landmark_detector

        # 5. Gaze Estimator & Screen Deviation
        self.gaze_estimator = gaze_estimator or GazeEstimator(
            yaw_threshold=self.config.gaze_yaw_threshold,
            pitch_threshold=self.config.gaze_pitch_threshold,
            iris_yaw_scale=self.config.gaze_iris_yaw_scale,
            iris_pitch_scale=self.config.gaze_iris_pitch_scale,
        )
        self.gaze_estimator._landmark_detector = self.landmark_detector
        self.gaze_estimator._head_pose_estimator = self.head_pose_estimator


        # 6. Secondary Object Detector (YOLOv8n with temporal stride caching)
        if self.config.enable_objects:
            self.object_detector = object_detector or ProctoringObjectDetector(
                confidence_threshold=self.config.object_confidence,
                stride=self.config.object_stride,
            )
        else:
            self.object_detector = None

        # Internal session tracking state
        self._frame_count: int = 0
        self._cumulative_latency_ms: float = 0.0
        self._face_detected_count: int = 0

        logger.info(
            f"VisionPipeline initialized successfully (resolution={self.config.target_width}x{self.config.target_height}, "
            f"fps={self.config.target_fps}, objects_enabled={self.config.enable_objects})"
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
        is_rgb: bool = False,
        force_object_inference: bool = False,
    ) -> VisionPipelineResult:
        """Execute unified multi-stage perception on an ingested video frame.

        Args:
            frame: Input image array (BGR or RGB).
            frame_index: Sequential index. If None, auto-increments from internal counter.
            timestamp_ms: Session timestamp in ms. If None, computed from target FPS.
            is_rgb: True if input array is RGB; False if BGR.
            force_object_inference: If True, bypass object detector stride caching.

        Returns:
            VisionPipelineResult: Consolidated perception output and domain observation.
        """
        t_start = time.perf_counter()
        component_latencies: dict[str, float] = {}

        # 1. Resolve frame index and timestamp
        if frame_index is None:
            idx = self._frame_count
        else:
            idx = int(frame_index)

        if timestamp_ms is None:
            ts_ms = (idx / max(1.0, self.config.target_fps)) * 1000.0
        else:
            ts_ms = float(timestamp_ms)

        # 2. Frame Ingestion and Preprocessing
        t0 = time.perf_counter()
        preprocessed = self.preprocessor.process(
            frame=frame,
            frame_index=idx,
            timestamp_ms=ts_ms,
            is_rgb=is_rgb,
        )
        component_latencies["preprocessor"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 3. Primary Face Detection
        t0 = time.perf_counter()
        face_result = self.face_detector.detect(preprocessed)
        component_latencies["face_detector"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Dense Facial Landmarks, 3D Head Pose, and Gaze Estimation
        if face_result.face_detected:
            # 4a. Facial Landmarks
            t0 = time.perf_counter()
            landmark_result = self.landmark_detector.detect(
                preprocessed,
                primary_face=face_result.primary_face,
            )
            component_latencies["landmarks"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # 4b. 3D Head Pose
            t0 = time.perf_counter()
            head_pose_result = self.head_pose_estimator.estimate(
                preprocessed,
                landmarks_result=landmark_result,
            )
            component_latencies["head_pose"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # 4c. Gaze Estimation
            t0 = time.perf_counter()
            gaze_result = self.gaze_estimator.estimate(
                preprocessed,
                landmarks_result=landmark_result,
                head_pose_result=head_pose_result,
            )
            component_latencies["gaze"] = round((time.perf_counter() - t0) * 1000.0, 2)
        else:
            landmark_result = self.landmark_detector._build_empty_result(0.0)
            head_pose_result = self.head_pose_estimator._build_empty_result(0.0)
            gaze_result = self.gaze_estimator._build_empty_result(0.0)
            component_latencies["landmarks"] = 0.0
            component_latencies["head_pose"] = 0.0
            component_latencies["gaze"] = 0.0

        # 5. Proctoring Object Detection
        t0 = time.perf_counter()
        if self.object_detector is not None:
            object_result = self.object_detector.detect(
                preprocessed,
                force=force_object_inference,
            )
        else:
            object_result = ObjectDetectionResult(
                detections=[],
                phone_detected=False,
                laptop_detected=False,
                book_detected=False,
                person_count=0,
                secondary_person_detected=False,
                prohibited_items=[],
                latency_ms=0.0,
                is_cached=False,
            )
        component_latencies["object_detector"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 6. Aggregate into canonical FrameObservation domain model
        primary_box: BoundingBox | None = None
        if face_result.face_detected and face_result.primary_face is not None:
            primary_box = face_result.primary_face.box

        head_pose_angles: HeadPoseAngles | None = None
        if head_pose_result.pose_detected:
            head_pose_angles = head_pose_result.euler_angles


        gaze_vector: GazeVector | None = None
        if gaze_result.gaze_detected:
            gaze_vector = gaze_result.gaze_vector

        observation = FrameObservation(
            frame_index=idx,
            timestamp_ms=ts_ms,
            face_detected=face_result.face_detected,
            face_count=face_result.face_count,
            primary_face_box=primary_box,
            head_pose=head_pose_angles,
            gaze=gaze_vector,
            detected_objects=list(object_result.detections),
        )

        total_lat = round((time.perf_counter() - t_start) * 1000.0, 2)

        # Update cumulative diagnostics
        self._frame_count += 1
        self._cumulative_latency_ms += total_lat
        if observation.face_detected:
            self._face_detected_count += 1

        return VisionPipelineResult(
            observation=observation,
            preprocessed=preprocessed,
            face_result=face_result,
            landmark_result=landmark_result,
            head_pose_result=head_pose_result,
            gaze_result=gaze_result,
            object_result=object_result,
            total_latency_ms=total_lat,
            component_latencies=component_latencies,
        )

    def process_observation(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
    ) -> FrameObservation:
        """Lightweight perception helper returning only the canonical FrameObservation."""
        result = self.process_frame(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        return result.observation

    def calibrate_gaze(self, yaw: float, pitch: float) -> None:
        """Calibrate the natural resting gaze baseline for the candidate."""
        self.gaze_estimator.calibrate_baseline(yaw=yaw, pitch=pitch)

    def reset_gaze_calibration(self) -> None:
        """Reset gaze calibration back to standard camera normal."""
        self.gaze_estimator.reset_calibration()

    def get_diagnostics(self) -> dict[str, Any]:
        """Retrieve aggregated pipeline performance statistics."""
        avg_lat = (
            self._cumulative_latency_ms / max(1, self._frame_count)
            if self._frame_count > 0
            else 0.0
        )
        detection_rate = (
            self._face_detected_count / max(1, self._frame_count)
            if self._frame_count > 0
            else 0.0
        )
        return {
            "total_frames_processed": self._frame_count,
            "average_latency_ms": round(avg_lat, 2),
            "face_detection_rate": round(detection_rate, 3),
            "object_detector_enabled": self.object_detector is not None,
            "target_resolution": (self.config.target_width, self.config.target_height),
        }

    def reset(self) -> None:
        """Reset internal frame counter and diagnostic metrics."""
        self._frame_count = 0
        self._cumulative_latency_ms = 0.0
        self._face_detected_count = 0
        if self.object_detector is not None:
            self.object_detector.reset()

    def annotate_frame(
        self,
        frame: np.ndarray | None = None,
        result: VisionPipelineResult | None = None,
        draw_face_box: bool = True,
        draw_pose_axes: bool = True,
        draw_gaze_rays: bool = True,
        draw_objects: bool = True,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Render multi-modal perception diagnostics onto the video frame.

        Args:
            frame: Optional base image. If None, uses result.preprocessed.bgr_image.
            result: VisionPipelineResult containing all perception outputs.
            draw_face_box: If True, draws bounding box around detected faces.
            draw_pose_axes: If True, projects 3D Cartesian pose axes from nose bridge.
            draw_gaze_rays: If True, projects gaze vector rays from pupils.
            draw_objects: If True, draws bounding boxes for secondary objects.
            draw_hud: If True, draws status HUD with FPS, angles, and alerts.

        Returns:
            np.ndarray: Annotated BGR image array.
        """
        if result is None:
            raise VisionPipelineError("Cannot annotate frame without VisionPipelineResult.")

        canvas = (
            frame.copy()
            if frame is not None
            else result.preprocessed.bgr_image.copy()
        )
        h, w = canvas.shape[:2]

        # 1. Draw Object Detections
        if draw_objects and self.object_detector is not None and result.object_result.detections:
            canvas = self.object_detector.draw_detections(canvas, result.object_result)

        # 2. Draw Face Bounding Box & Keypoints
        if draw_face_box and result.face_result.face_detected:
            for face in result.face_result.faces:
                bx = face.box
                is_suspicious = (
                    result.head_pose_result.is_looking_away
                    or result.gaze_result.is_deviated
                )
                box_color = (0, 165, 255) if is_suspicious else (0, 255, 0)  # Orange if deviated, Green if normal
                cv2.rectangle(
                    canvas,
                    (int(bx.xmin), int(bx.ymin)),
                    (int(bx.xmax), int(bx.ymax)),
                    box_color,
                    2,
                )
                label = f"Face: {face.confidence:.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (int(bx.xmin), max(15, int(bx.ymin) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw 3D Head Pose Axes
        if draw_pose_axes and result.head_pose_result.pose_detected:
            canvas = self.head_pose_estimator.project_pose_axes(
                canvas,
                result.head_pose_result.rvec,
                result.head_pose_result.tvec,
                length=50.0,
            )

        # 4. Draw 3D Gaze Direction Rays
        if draw_gaze_rays and result.gaze_result.gaze_detected:
            canvas = self.gaze_estimator.draw_gaze_rays(
                canvas,
                result.gaze_result,
                length=45.0,
            )


        # 5. Draw Diagnostics HUD
        if draw_hud:
            canvas = self._draw_hud(canvas, result)

        return canvas

    def _draw_hud(self, canvas: np.ndarray, result: VisionPipelineResult) -> np.ndarray:
        """Render top status banner and proctoring HUD overlay."""
        h, w = canvas.shape[:2]

        # Top semi-transparent banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Pipeline latency & frame info
        obs = result.observation
        frame_text = f"Frame: #{obs.frame_index} | Latency: {result.total_latency_ms:.1f}ms"
        cv2.putText(
            canvas,
            frame_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Pose & Gaze status
        if obs.head_pose is not None:
            pose_status = "LOOKING AWAY" if result.head_pose_result.is_looking_away else "CENTER"
            pose_color = (0, 0, 255) if result.head_pose_result.is_looking_away else (0, 255, 0)
            pose_str = f"Head: Y={obs.head_pose.yaw:+.1f} P={obs.head_pose.pitch:+.1f} [{pose_status}]"
        else:
            pose_str = "Head: NO FACE"
            pose_color = (0, 165, 255)

        cv2.putText(
            canvas,
            pose_str,
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            pose_color,
            1,
            cv2.LINE_AA,
        )

        # Gaze & Direction status
        if obs.gaze is not None:
            gaze_status = "OFF-SCREEN" if result.gaze_result.is_deviated else "ON-SCREEN"
            gaze_color = (0, 0, 255) if result.gaze_result.is_deviated else (0, 255, 0)
            gaze_str = f"Gaze: {obs.gaze.direction.value} [{gaze_status}]"
        else:
            gaze_str = "Gaze: N/A"
            gaze_color = (150, 150, 150)

        cv2.putText(
            canvas,
            gaze_str,
            (w - 240, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            gaze_color,
            1,
            cv2.LINE_AA,
        )

        # Object / Security Alert
        obj_res = result.object_result
        if obj_res.phone_detected:
            alert_str = "! ALERT: PHONE DETECTED !"
            alert_color = (0, 0, 255)
        elif obj_res.secondary_person_detected or obs.face_count > 1:
            alert_str = "! ALERT: MULTIPLE PERSONS !"
            alert_color = (0, 0, 255)
        elif obj_res.book_detected:
            alert_str = "! ALERT: BOOK DETECTED !"
            alert_color = (0, 0, 255)
        elif not obs.face_detected:
            alert_str = "! WARNING: FACE MISSING !"
            alert_color = (0, 165, 255)
        else:
            alert_str = "SECURE / NORMAL"
            alert_color = (0, 255, 0)

        cv2.putText(
            canvas,
            alert_str,
            (w - 240, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            alert_color,
            1,
            cv2.LINE_AA,
        )

        return canvas
