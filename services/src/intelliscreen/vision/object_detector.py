"""Secondary Object and Unauthorized Device Detection for IntelliScreen.

Detects unauthorized items (cell phones, laptops, books/notes) and secondary persons
in the candidate's camera feed using Ultralytics YOLOv8n. Implements temporal stride
caching to conserve CPU resources while maintaining real-time proctoring alerts.
"""

from dataclasses import dataclass, field
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from intelliscreen.config.settings import get_settings
from intelliscreen.core.logging import get_logger
from intelliscreen.core.schemas import BoundingBox, DetectedObject
from intelliscreen.vision.preprocessor import PreprocessedFrame

logger = get_logger("object_detector")

DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "models",
    "weights",
)
DEFAULT_YOLO_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "yolov8n.pt")

# Proctoring-relevant COCO class indices:
# 0: person, 63: laptop, 67: cell phone, 73: book
PROCTORING_COCO_CLASSES: dict[int, str] = {
    0: "person",
    63: "laptop",
    67: "cell phone",
    73: "book",
}
TARGET_CLASS_IDS: list[int] = list(PROCTORING_COCO_CLASSES.keys())


# ============================================================================
# Domain Results & Perception Output
# ============================================================================

@dataclass
class ObjectDetectionResult:
    """Structured perception output for secondary object detection."""

    detections: list[DetectedObject]
    phone_detected: bool
    laptop_detected: bool
    book_detected: bool
    person_count: int
    secondary_person_detected: bool
    prohibited_items: list[str]
    latency_ms: float
    is_cached: bool = False


# ============================================================================
# ProctoringObjectDetector Pipeline
# ============================================================================

class ProctoringObjectDetector:
    """Detects unauthorized physical devices and secondary persons with stride caching."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        stride: int | None = None,
    ) -> None:
        """Initialize the proctoring object detector.

        Args:
            model_path: Path to the YOLOv8n weights file.
            confidence_threshold: Minimum detection confidence score.
            stride: Process inference every N frames to conserve CPU.
        """
        settings = get_settings()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.object_confidence_threshold
        )
        self.stride = (
            stride
            if stride is not None
            else settings.vision_object_detection_stride
        )
        self.model_path = model_path or DEFAULT_YOLO_PATH

        self._cached_result: ObjectDetectionResult | None = None
        self._last_processed_frame_index: int = -1

        try:
            self.model = YOLO(self.model_path)
            logger.info(
                f"ProctoringObjectDetector initialized from {self.model_path} "
                f"(conf={self.confidence_threshold}, stride={self.stride})"
            )
        except Exception as exc:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {exc}")
            raise

    def detect(
        self,
        frame: PreprocessedFrame,
        force: bool = False,
    ) -> ObjectDetectionResult:
        """Run object detection on the preprocessed frame with temporal stride caching.

        Args:
            frame: Standardized PreprocessedFrame.
            force: If True, bypass stride caching and execute inference immediately.

        Returns:
            ObjectDetectionResult: List of detected objects, category flags, and alerts.
        """
        t0 = time.perf_counter()
        frame_idx = frame.frame_index

        # Check stride cache
        if (
            not force
            and self._cached_result is not None
            and (frame_idx % self.stride != 0)
        ):
            # Return cached detection with minimal lookup latency
            cache_lat = (time.perf_counter() - t0) * 1000
            return ObjectDetectionResult(
                detections=list(self._cached_result.detections),
                phone_detected=self._cached_result.phone_detected,
                laptop_detected=self._cached_result.laptop_detected,
                book_detected=self._cached_result.book_detected,
                person_count=self._cached_result.person_count,
                secondary_person_detected=self._cached_result.secondary_person_detected,
                prohibited_items=list(self._cached_result.prohibited_items),
                latency_ms=round(cache_lat, 2),
                is_cached=True,
            )

        # Execute neural inference on BGR image
        img = frame.bgr_image
        h, w = img.shape[:2]

        try:
            results = self.model(
                img,
                classes=TARGET_CLASS_IDS,
                conf=self.confidence_threshold,
                verbose=False,
                device="cpu",
            )
        except Exception as exc:
            logger.warning(f"YOLO inference error: {exc}. Returning empty result.")
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._build_empty_result(latency_ms)

        detections: list[DetectedObject] = []
        person_count = 0
        phone_detected = False
        laptop_detected = False
        book_detected = False

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box_data in boxes:
                cls_id = int(box_data.cls[0].item())
                conf = float(box_data.conf[0].item())

                if conf < self.confidence_threshold:
                    continue

                label = PROCTORING_COCO_CLASSES.get(cls_id, self.model.names.get(cls_id, "unknown"))
                xyxy = box_data.xyxy[0].tolist()

                xmin = max(0.0, float(xyxy[0]))
                ymin = max(0.0, float(xyxy[1]))
                xmax = min(float(w), float(xyxy[2]))
                ymax = min(float(h), float(xyxy[3]))

                if xmax <= xmin or ymax <= ymin:
                    continue

                bbox = BoundingBox(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    confidence=conf,
                    label=label,
                )
                detections.append(DetectedObject(label=label, confidence=conf, box=bbox))

                if label == "person":
                    person_count += 1
                elif label == "cell phone":
                    phone_detected = True
                elif label == "laptop":
                    laptop_detected = True
                elif label == "book":
                    book_detected = True

        secondary_person_detected = person_count > 1

        # Compile prohibited items list
        prohibited_items: list[str] = []
        if phone_detected:
            prohibited_items.append("cell phone")
        if laptop_detected:
            prohibited_items.append("laptop")
        if book_detected:
            prohibited_items.append("book")
        if secondary_person_detected:
            prohibited_items.append("secondary person")

        latency_ms = (time.perf_counter() - t0) * 1000

        result = ObjectDetectionResult(
            detections=detections,
            phone_detected=phone_detected,
            laptop_detected=laptop_detected,
            book_detected=book_detected,
            person_count=person_count,
            secondary_person_detected=secondary_person_detected,
            prohibited_items=prohibited_items,
            latency_ms=round(latency_ms, 2),
            is_cached=False,
        )

        self._cached_result = result
        self._last_processed_frame_index = frame_idx
        return result

    def draw_detections(
        self,
        image: np.ndarray,
        result: ObjectDetectionResult,
    ) -> np.ndarray:
        """Draw bounding boxes and warning labels onto the image.

        Color conventions:
        - Cell Phone / Secondary Person: Red (0, 0, 255) [High Alert]
        - Laptop / Book: Yellow (0, 255, 255) [Unauthorized Material]
        - Primary Person: Cyan / Green (255, 200, 0) [Neutral Candidate]

        Args:
            image: BGR image canvas.
            result: Computed ObjectDetectionResult.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        output = image.copy()
        person_idx = 0

        for obj in result.detections:
            box = obj.box
            x1, y1, x2, y2 = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)

            if obj.label == "cell phone":
                color = (0, 0, 255)  # Red
                title = f"PHONE: {obj.confidence:.2f}"
            elif obj.label == "person":
                person_idx += 1
                if person_idx > 1:
                    color = (0, 0, 255)  # Secondary person is Red
                    title = f"EXTRA PERSON: {obj.confidence:.2f}"
                else:
                    color = (255, 200, 0)  # Primary candidate is Cyan
                    title = f"Candidate: {obj.confidence:.2f}"
            elif obj.label == "book":
                color = (0, 255, 255)  # Yellow
                title = f"BOOK/NOTES: {obj.confidence:.2f}"
            elif obj.label == "laptop":
                color = (0, 215, 255)  # Amber Yellow
                title = f"SECONDARY SCREEN: {obj.confidence:.2f}"
            else:
                color = (200, 200, 200)
                title = f"{obj.label}: {obj.confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw filled header label
            text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tw, th = text_size
            cv2.rectangle(
                output,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 6, max(th + 6, y1)),
                color,
                -1,
            )
            text_color = (0, 0, 0) if color in [(0, 255, 255), (0, 215, 255), (255, 200, 0)] else (255, 255, 255)
            cv2.putText(
                output,
                title,
                (x1 + 3, max(th, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return output

    def _build_empty_result(self, latency_ms: float) -> ObjectDetectionResult:
        """Return standardized empty result when no objects are detected or on failure."""
        return ObjectDetectionResult(
            detections=[],
            phone_detected=False,
            laptop_detected=False,
            book_detected=False,
            person_count=0,
            secondary_person_detected=False,
            prohibited_items=[],
            latency_ms=round(latency_ms, 2),
            is_cached=False,
        )
