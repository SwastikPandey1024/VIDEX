"""Object detection provider implementations (YOLO26, MockDetector).

Ensures all detections strictly preserve FrameTimestamp provenance and metadata
from DecodedFrame / Frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    Frame,
    FrameTimestamp,
    TimestampSource,
)
from videx.ingestion.base import DecodedFrame


@dataclass
class YOLO26DetectorConfig:
    """Configuration for YOLO26 object detection provider."""

    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    half_precision: bool = False
    classes: list[int] | None = None
    max_detections: int = 300
    export_mask: bool = False
    custom_class_names: dict[int, str] = field(default_factory=dict)


class YOLO26Detector:
    """YOLO26 object detection provider implementing DetectionProvider protocol.

    Wraps the YOLO inference engine behind a clean boundary, supporting CPU execution,
    configured confidence thresholds, class filtering, and exact timestamp inheritance.
    """

    def __init__(
        self,
        config: YOLO26DetectorConfig | None = None,
        model: object | None = None,
    ) -> None:
        self.config = config or YOLO26DetectorConfig()
        self._model: Any = model
        self._provider_name = f"yolo26_{Path(self.config.model_path).stem}"
        self._is_warm = model is not None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def warmup(self) -> None:
        """Load model weights and perform dummy forward pass if ultralytics is available."""
        if self._is_warm:
            return

        try:
            import importlib

            ultralytics_mod = importlib.import_module("ultralytics")
            yolo_cls = ultralytics_mod.YOLO
            self._model = yolo_cls(self.config.model_path)
            # Dummy forward pass on 640x640 blank array
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(
                dummy,
                device=self.config.device,
                conf=self.config.confidence_threshold,
                verbose=False,
            )
            self._is_warm = True
        except (ImportError, AttributeError):
            # Model execution will raise RuntimeError upon actual detection attempt
            self._model = None
            self._is_warm = False

    def detect(
        self,
        frame_data: bytes | DecodedFrame,
        frame_meta: Frame | None = None,
    ) -> list[Detection]:
        """Run YOLO26 inference on a frame, inheriting exact timestamp and frame provenance."""
        # 1. Resolve image array and metadata
        frame_array, f_meta, frame_ts = _resolve_frame_input(frame_data, frame_meta)
        if frame_ts is None:
            frame_ts = getattr(f_meta, "frame_timestamp", None)
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=f_meta.frame_number,
                pts_seconds=f_meta.timestamp_seconds,
                timestamp_source=TimestampSource.DERIVED,
            )

        # 2. Check model availability
        if self._model is None:
            try:
                import importlib

                ultralytics_mod = importlib.import_module("ultralytics")
                yolo_cls = ultralytics_mod.YOLO
                self._model = yolo_cls(self.config.model_path)
            except (ImportError, AttributeError) as err:
                raise RuntimeError(
                    f"Ultralytics is required for YOLO26Detector '{self.provider_name}' "
                    "but is not installed in the environment."
                ) from err

        # 3. Execute inference
        results = self._model(
            frame_array,
            device=self.config.device,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            max_det=self.config.max_detections,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        res0 = results[0]
        boxes = res0.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = res0.names or self.config.custom_class_names

        # 4. Map results to domain Detection schemas
        for i in range(len(boxes)):
            box_item = boxes.xyxy[i]
            xyxy = box_item.tolist() if hasattr(box_item, "tolist") else list(box_item)
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = xyxy
            w = max(x2 - x1, 1e-4)
            h = max(y2 - y1, 1e-4)

            mask_coords = None
            if self.config.export_mask and hasattr(res0, "masks") and res0.masks is not None:
                try:
                    poly = res0.masks.xy[i]
                    mask_coords = poly.tolist()
                except Exception:
                    mask_coords = None

            # Prepare provenance attributes
            attr: dict[str, Any] = {
                "source_provider": self.provider_name,
                "model_path": self.config.model_path,
            }
            if frame_ts is not None:
                attr["timestamp_source"] = frame_ts.timestamp_source.value
                attr["is_repaired"] = frame_ts.is_repaired
                if frame_ts.repair_reason:
                    attr["repair_reason"] = frame_ts.repair_reason
                if frame_ts.original_pts_seconds is not None:
                    attr["original_pts_seconds"] = frame_ts.original_pts_seconds

            detections.append(
                Detection(
                    frame_id=f_meta.frame_id,
                    video_id=f_meta.video_id,
                    frame_number=f_meta.frame_number,
                    timestamp_seconds=f_meta.timestamp_seconds,
                    frame_timestamp=frame_ts,
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        width=w,
                        height=h,
                        coordinate_type=CoordinateType.PIXEL,
                    ),
                    mask=mask_coords,
                    provider=self.provider_name,
                    attributes=attr,
                )
            )

        return detections

    def detect_batch(
        self,
        batch: list[tuple[bytes | DecodedFrame, Frame | None]],
    ) -> list[list[Detection]]:
        return [self.detect(frame_data, frame_meta) for frame_data, frame_meta in batch]


class MockDetector:
    """Deterministic, configurable mock detector for CI testing and unit tests.

    Emits programmed detections or runs simple color/pattern checks without neural net weights.
    """

    def __init__(
        self,
        provider_name: str = "mock_detector",
        canned_detections: dict[int, list[dict[str, Any]]] | None = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        self._provider_name = provider_name
        self.canned_detections = canned_detections or {}
        self.confidence_threshold = confidence_threshold

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def warmup(self) -> None:
        pass

    def detect(
        self,
        frame_data: bytes | DecodedFrame,
        frame_meta: Frame | None = None,
    ) -> list[Detection]:
        _, f_meta, frame_ts = _resolve_frame_input(frame_data, frame_meta)
        if frame_ts is None:
            frame_ts = getattr(f_meta, "frame_timestamp", None)
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=f_meta.frame_number,
                pts_seconds=f_meta.timestamp_seconds,
                timestamp_source=TimestampSource.DERIVED,
            )

        items = self.canned_detections.get(f_meta.frame_number, [])
        detections: list[Detection] = []

        for item in items:
            conf = float(item.get("confidence", 0.90))
            if conf < self.confidence_threshold:
                continue

            attr: dict[str, Any] = {
                "mock": True,
                "timestamp_source": frame_ts.timestamp_source.value,
                "is_repaired": frame_ts.is_repaired,
            }
            if frame_ts.repair_reason:
                attr["repair_reason"] = frame_ts.repair_reason
            if frame_ts.original_pts_seconds is not None:
                attr["original_pts_seconds"] = frame_ts.original_pts_seconds

            detections.append(
                Detection(
                    frame_id=f_meta.frame_id,
                    video_id=f_meta.video_id,
                    frame_number=f_meta.frame_number,
                    timestamp_seconds=f_meta.timestamp_seconds,
                    frame_timestamp=frame_ts,
                    class_name=str(item.get("class_name", "object")),
                    class_id=int(item.get("class_id", 0)),
                    confidence=conf,
                    bbox=BoundingBox(
                        x=float(item.get("x", 10.0)),
                        y=float(item.get("y", 10.0)),
                        width=float(item.get("width", 50.0)),
                        height=float(item.get("height", 50.0)),
                        coordinate_type=CoordinateType.PIXEL,
                    ),
                    mask=item.get("mask"),
                    provider=self.provider_name,
                    attributes=attr,
                )
            )

        return detections

    def detect_batch(
        self,
        batch: list[tuple[bytes | DecodedFrame, Frame | None]],
    ) -> list[list[Detection]]:
        return [self.detect(f_data, f_meta) for f_data, f_meta in batch]


def _resolve_frame_input(
    frame_data: bytes | DecodedFrame,
    frame_meta: Frame | None,
) -> tuple[np.ndarray, Frame, FrameTimestamp | None]:
    """Helper normalizing inputs into a NumPy ndarray, Frame metadata, and FrameTimestamp."""
    if isinstance(frame_data, DecodedFrame):
        if not isinstance(frame_data.frame_array, np.ndarray):
            raise ValueError(
                f"DecodedFrame array must be np.ndarray, got {type(frame_data.frame_array)}"
            )
        img = frame_data.frame_array
        frame_ts = frame_data.frame_timestamp
        f_meta = frame_meta or Frame(
            video_id=frame_data.video_id or uuid4(),
            scene_id=frame_data.scene_id,
            frame_number=frame_data.frame_index,
            timestamp_seconds=frame_data.timestamp_seconds,
            frame_timestamp=frame_ts,
            width=frame_data.width,
            height=frame_data.height,
        )
        return img, f_meta, frame_ts

    if not isinstance(frame_data, bytes):
        raise TypeError(f"frame_data must be DecodedFrame or bytes, got {type(frame_data)}")

    buf = np.frombuffer(frame_data, dtype=np.uint8)
    decoded_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded_img is None or not isinstance(decoded_img, np.ndarray):
        raise ValueError("Failed to decode image bytes into OpenCV matrix")
    img = decoded_img

    h, w = img.shape[:2]
    frame_ts = getattr(frame_meta, "frame_timestamp", None)
    f_meta = frame_meta or Frame(
        video_id=uuid4(),
        frame_number=0,
        timestamp_seconds=0.0,
        frame_timestamp=frame_ts,
        width=w,
        height=h,
    )
    return img, f_meta, frame_ts
