"""Video frame ingestion, validation, normalization, and quality assessment."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from intelliscreen.config.settings import get_settings
from intelliscreen.core.exceptions import FramePreprocessingError
from intelliscreen.core.schemas import BoundingBox


@dataclass(frozen=True)
class FrameQualityMetrics:
    """Quantitative image quality and illumination metrics."""

    brightness: float
    contrast: float
    blur_score: float
    is_low_light: bool
    is_blurry: bool


@dataclass
class PreprocessedFrame:
    """Standardized multi-format frame representation for computer vision inference."""

    frame_index: int
    timestamp_ms: float
    rgb_image: np.ndarray
    bgr_image: np.ndarray
    gray_image: np.ndarray
    original_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int]
    scale_x: float
    scale_y: float
    quality: FrameQualityMetrics

    def map_box_to_original(self, box: BoundingBox) -> BoundingBox:
        """Map a bounding box from processed frame coordinates back to original frame coordinates."""
        orig_h, orig_w = self.original_shape[:2]
        proc_h, proc_w = self.processed_shape[:2]

        # If coordinates are already normalized (0.0 to 1.0), scale by original dimensions
        if box.xmax <= 1.0 and box.ymax <= 1.0:
            return BoundingBox(
                xmin=box.xmin * orig_w,
                ymin=box.ymin * orig_h,
                xmax=box.xmax * orig_w,
                ymax=box.ymax * orig_h,
                confidence=box.confidence,
                label=box.label,
            )

        # If coordinates are in processed pixel space
        inv_scale_x = 1.0 / self.scale_x if self.scale_x > 0 else 1.0
        inv_scale_y = 1.0 / self.scale_y if self.scale_y > 0 else 1.0

        return BoundingBox(
            xmin=max(0.0, box.xmin * inv_scale_x),
            ymin=max(0.0, box.ymin * inv_scale_y),
            xmax=min(float(orig_w), box.xmax * inv_scale_x),
            ymax=min(float(orig_h), box.ymax * inv_scale_y),
            confidence=box.confidence,
            label=box.label,
        )


class VideoPreprocessor:
    """Ingests raw video frames, validates dimensions, resizes, and assesses quality."""

    def __init__(
        self,
        target_width: int | None = None,
        target_height: int | None = None,
        low_light_threshold: float = 40.0,
        blur_threshold: float = 40.0,
    ) -> None:
        settings = get_settings()
        self.target_width = target_width or settings.vision_frame_width
        self.target_height = target_height or settings.vision_frame_height
        self.low_light_threshold = low_light_threshold
        self.blur_threshold = blur_threshold

    def process(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
        is_rgb: bool = False,
    ) -> PreprocessedFrame:
        """Preprocess an input numpy image array into standardized formats and measure quality."""
        self._validate_raw_frame(frame)

        orig_h, orig_w = frame.shape[:2]
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        original_shape = (orig_h, orig_w, channels)

        # Standardize color representations
        if channels == 1:
            gray = frame.copy()
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif is_rgb:
            rgb = frame.copy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            bgr = frame.copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Spatial Resizing
        if (orig_w, orig_h) != (self.target_width, self.target_height):
            bgr_resized = cv2.resize(
                bgr,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            rgb_resized = cv2.resize(
                rgb,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            gray_resized = cv2.resize(
                gray,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            bgr_resized = bgr
            rgb_resized = rgb
            gray_resized = gray

        scale_x = self.target_width / orig_w
        scale_y = self.target_height / orig_h
        proc_h, proc_w = gray_resized.shape[:2]
        processed_shape = (proc_h, proc_w, 3)

        # Quality Metrics Computation
        quality = self._calculate_quality_metrics(gray_resized)

        return PreprocessedFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            rgb_image=rgb_resized,
            bgr_image=bgr_resized,
            gray_image=gray_resized,
            original_shape=original_shape,
            processed_shape=processed_shape,
            scale_x=scale_x,
            scale_y=scale_y,
            quality=quality,
        )

    def process_bytes(
        self,
        raw_bytes: bytes,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
    ) -> PreprocessedFrame:
        """Decode raw image bytes (e.g. from multipart HTTP upload) and preprocess."""
        if not raw_bytes:
            raise FramePreprocessingError("Received empty image byte buffer.")

        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise FramePreprocessingError("Failed to decode image from provided byte stream.")

        return self.process(frame, frame_index=frame_index, timestamp_ms=timestamp_ms)

    def _validate_raw_frame(self, frame: Any) -> None:
        """Ensure input frame is a non-empty, valid numpy array with proper dimensions."""
        if not isinstance(frame, np.ndarray):
            raise FramePreprocessingError(f"Expected numpy.ndarray, received {type(frame).__name__}")

        if frame.size == 0:
            raise FramePreprocessingError("Input frame array is empty (size=0).")

        if frame.ndim not in (2, 3):
            raise FramePreprocessingError(
                f"Invalid frame dimensions: ndim={frame.ndim}. Expected 2 (grayscale) or 3 (color)."
            )

        if frame.shape[0] < 10 or frame.shape[1] < 10:
            raise FramePreprocessingError(
                f"Frame dimensions too small: {frame.shape[1]}x{frame.shape[0]} px."
            )

    def _calculate_quality_metrics(self, gray_frame: np.ndarray) -> FrameQualityMetrics:
        """Compute illumination, contrast, and focus metrics on grayscale frame."""
        brightness = float(np.mean(gray_frame))
        contrast = float(np.std(gray_frame))

        # Variance of the Laplacian acts as a focus/sharpness measure
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        blur_score = float(laplacian.var())

        is_low_light = brightness < self.low_light_threshold
        is_blurry = blur_score < self.blur_threshold

        return FrameQualityMetrics(
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            blur_score=round(blur_score, 2),
            is_low_light=is_low_light,
            is_blurry=is_blurry,
        )
