"""Object detection provider implementations (YOLO26, MockDetector).

Ensures all detections strictly preserve FrameTimestamp provenance and metadata
from DecodedFrame / Frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from videx.config import Settings

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
from videx.providers.base import DetectionProvider

logger = logging.getLogger(__name__)


@dataclass
class YOLO26DetectorConfig:
    """Configuration for YOLO26 object detection provider."""

    model_path: str = "yolo26n.pt"
    onnx_model_path: str = "models/yolo26n.onnx"
    backend: Literal["pytorch", "onnx"] = "pytorch"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    half_precision: bool = False
    classes: list[int] | None = None
    max_detections: int = 300
    export_mask: bool = False
    custom_class_names: dict[int, str] = field(default_factory=dict)


def _resolve_model_path(path_str: str) -> str:
    """Resolve model path, preferring existing path or searching models/ directory."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    models_candidate = Path("models") / p
    if models_candidate.exists():
        return str(models_candidate)
    return path_str


class YOLO26Detector:
    """YOLO26 object detection provider implementing DetectionProvider protocol.

    Wraps the YOLO inference engine behind a clean boundary, supporting PyTorch FP32
    (default) and opt-in ONNX Runtime FP32 execution, CPU execution, configured confidence
    thresholds, class filtering, and exact timestamp inheritance.
    """

    def __init__(
        self,
        config: YOLO26DetectorConfig | None = None,
        model: object | None = None,
    ) -> None:
        self.config = config or YOLO26DetectorConfig()
        self._model: Any = model
        self._is_warm = model is not None

    def _resolve_active_model_path(self) -> str:
        """Resolve the model path based on the selected backend."""
        if self.config.backend == "onnx":
            if self.config.model_path.lower().endswith(".onnx"):
                raw_path = self.config.model_path
            else:
                raw_path = self.config.onnx_model_path
        else:
            raw_path = self.config.model_path
        return _resolve_model_path(raw_path)

    @property
    def provider_name(self) -> str:
        model_stem = Path(self._resolve_active_model_path()).stem
        if self.config.backend == "onnx":
            return f"yolo26_onnx_{model_stem}"
        return f"yolo26_{model_stem}"

    def _load_model(self) -> None:
        """Lazily loads the backend model and validates dependencies and model paths."""
        if self.config.backend == "onnx":
            # 1. Validate onnxruntime dependency
            try:
                import importlib

                importlib.import_module("onnxruntime")
            except ImportError as err:
                raise RuntimeError(
                    "ONNX Runtime is required when detector backend is 'onnx', "
                    "but 'onnxruntime' is not installed in the environment. "
                    "Install it via 'pip install onnxruntime'."
                ) from err

            # 2. Validate ultralytics dependency
            try:
                import importlib

                ultralytics_mod = importlib.import_module("ultralytics")
                yolo_cls = ultralytics_mod.YOLO
            except (ImportError, AttributeError) as err:
                raise RuntimeError(
                    f"Ultralytics is required for YOLO26Detector '{self.provider_name}' "
                    "but is not installed in the environment."
                ) from err

            # 3. Validate ONNX model file exists
            resolved_path = self._resolve_active_model_path()
            p = Path(resolved_path)
            if not p.is_file():
                raise FileNotFoundError(
                    f"ONNX model file not found at '{resolved_path}' "
                    f"(configured as '{self.config.onnx_model_path}'). "
                    "Please provide a valid path or export the ONNX model first."
                )
            if not resolved_path.lower().endswith(".onnx"):
                raise ValueError(
                    f"Expected an ONNX model file (.onnx) for backend='onnx', "
                    f"but got '{resolved_path}'."
                )

            logger.info("Loading ONNX detector from %s", resolved_path)
            self._model = yolo_cls(resolved_path)

        elif self.config.backend == "pytorch":
            try:
                import importlib

                ultralytics_mod = importlib.import_module("ultralytics")
                yolo_cls = ultralytics_mod.YOLO
            except (ImportError, AttributeError) as err:
                raise RuntimeError(
                    f"Ultralytics is required for YOLO26Detector '{self.provider_name}' "
                    "but is not installed in the environment."
                ) from err

            resolved_path = self._resolve_active_model_path()
            logger.info("Loading PyTorch detector from %s", resolved_path)
            self._model = yolo_cls(resolved_path)
        else:
            raise ValueError(
                f"Unsupported detector backend '{self.config.backend}'. "
                "Must be 'pytorch' or 'onnx'."
            )

    def warmup(self) -> None:
        """Load model weights and perform dummy forward pass if backend is available."""
        if self._is_warm:
            return

        try:
            if self._model is None:
                self._load_model()
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(
                dummy,
                device=self.config.device,
                conf=self.config.confidence_threshold,
                verbose=False,
            )
            self._is_warm = True
        except (ImportError, AttributeError):
            # Note on exception behavior: Actionable configuration and dependency errors
            # raised by _load_model() (FileNotFoundError, ValueError, RuntimeError) are
            # intentionally NOT caught here and propagate directly to the caller.
            # Only non-fatal attribute/import mismatches during dummy execution reset the
            # warmup flag to allow graceful fallback when probing optional components.
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

        # 2. Check model availability (lazy loading)
        if self._model is None:
            self._load_model()

        # 3. Execute inference
        try:
            results = self._model(
                frame_array,
                device=self.config.device,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.classes,
                max_det=self.config.max_detections,
                verbose=False,
            )
        except Exception as err:
            logger.error(
                "Inference failed on detector '%s' (backend=%s): %s",
                self.provider_name,
                self.config.backend,
                err,
            )
            raise RuntimeError(
                f"Inference failed on detector '{self.provider_name}' "
                f"using backend '{self.config.backend}': {err}"
            ) from err

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
                "model_path": self._resolve_active_model_path(),
                "backend": self.config.backend,
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


def create_detector(
    settings: Settings | None = None,
    **kwargs: object,
) -> DetectionProvider:
    """Create a configured DetectionProvider from application settings.

    Args:
        settings: VIDEX Settings instance or None (calls get_settings() if None).
        **kwargs: Overrides for YOLO26DetectorConfig fields.

    Returns:
        Configured DetectionProvider (YOLO26Detector).
    """
    if settings is None:
        try:
            from videx.config import get_settings

            settings = get_settings()
        except Exception:
            settings = None

    backend = getattr(settings, "detector_backend", "pytorch") if settings else "pytorch"
    conf_thresh = getattr(settings, "detector_confidence_threshold", 0.25) if settings else 0.25
    iou_thresh = getattr(settings, "detector_iou_threshold", 0.45) if settings else 0.45
    device = getattr(settings, "detector_device", "cpu") if settings else "cpu"
    pt_path = (
        getattr(settings, "detector_model_path", "models/yolo26n.pt")
        if settings
        else "models/yolo26n.pt"
    )
    onnx_path = (
        getattr(settings, "detector_onnx_model_path", "models/yolo26n.onnx")
        if settings
        else "models/yolo26n.onnx"
    )

    cfg_kwargs: dict[str, object] = {
        "model_path": pt_path,
        "onnx_model_path": onnx_path,
        "backend": backend,
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "device": device,
    }
    cfg_kwargs.update(kwargs)

    config = YOLO26DetectorConfig(**cfg_kwargs)  # type: ignore[arg-type]
    return YOLO26Detector(config=config)
