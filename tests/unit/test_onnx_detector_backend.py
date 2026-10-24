"""Unit tests for the opt-in ONNX FP32 detector backend.

Verifies:
- Configuration defaults (PyTorch default, ONNX opt-in).
- Provider naming and backend selection.
- Lazy-loading behavior (no eager loading on detector instantiation).
- Missing model file and extension validation.
- Missing dependency handling (onnxruntime / ultralytics) with actionable errors.
- Schema conformity, bounding box coordinate math, and FrameTimestamp preservation.
- Error handling on inference failure.
- create_detector factory function wiring with Settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videx.config import Settings
from videx.domain.schemas import CoordinateType, FrameTimestamp, TimestampSource
from videx.ingestion.base import DecodedFrame
from videx.perception.detection import (
    YOLO26Detector,
    YOLO26DetectorConfig,
    create_detector,
)


class MockBoxes:
    def __init__(
        self,
        xyxy: list[list[float]] | None = None,
        conf: list[float] | None = None,
        cls: list[int] | None = None,
    ) -> None:
        self.xyxy = xyxy if xyxy is not None else [[50.0, 100.0, 150.0, 300.0]]
        self.conf = conf if conf is not None else [0.92]
        self.cls = cls if cls is not None else [0]

    def __len__(self) -> int:
        return len(self.xyxy)


class MockResult:
    def __init__(
        self,
        boxes: MockBoxes | None = None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.boxes = boxes if boxes is not None else MockBoxes()
        self.names = names if names is not None else {0: "person", 1: "bicycle"}
        self.masks = None


class MockYOLOModel:
    def __init__(self, result: MockResult | None = None) -> None:
        self._result = result or MockResult()
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, *args: object, **kwargs: Any) -> list[MockResult]:
        self.call_count += 1
        self.last_kwargs = kwargs
        return [self._result]


class TestONNXDetectorBackendConfig:
    def test_config_defaults(self) -> None:
        cfg = YOLO26DetectorConfig()
        assert cfg.backend == "pytorch"
        assert cfg.model_path == "yolo26n.pt"
        assert cfg.onnx_model_path == "models/yolo26n.onnx"
        assert cfg.confidence_threshold == 0.25
        assert cfg.iou_threshold == 0.45
        assert cfg.device == "cpu"

    def test_config_onnx_opt_in(self) -> None:
        cfg = YOLO26DetectorConfig(
            backend="onnx",
            onnx_model_path="custom/model.onnx",
        )
        assert cfg.backend == "onnx"
        assert cfg.onnx_model_path == "custom/model.onnx"


class TestONNXDetectorProviderNaming:
    def test_provider_name_pytorch(self) -> None:
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="pytorch",
                model_path="models/yolo26n.pt",
            ),
            model=MockYOLOModel(),
        )
        assert detector.provider_name == "yolo26_yolo26n"

    def test_provider_name_onnx(self) -> None:
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="models/yolo26n.onnx",
            ),
            model=MockYOLOModel(),
        )
        assert detector.provider_name == "yolo26_onnx_yolo26n"


class TestONNXDetectorLazyLoading:
    def test_lazy_loading_does_not_initialize_model_on_init(self) -> None:
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="non_existent_model.onnx",
            )
        )
        # Should not throw on construction
        assert detector._model is None
        assert detector._is_warm is False

    def test_missing_onnx_model_raises_file_not_found(self, tmp_path: Path) -> None:
        non_existent = str(tmp_path / "absent_detector.onnx")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=non_existent,
            )
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            detector._load_model()

    def test_invalid_extension_for_onnx_backend_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_text("fake weights")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                model_path=str(fake_pt),
                onnx_model_path=str(fake_pt),
            )
        )
        with pytest.raises(ValueError, match="Expected an ONNX model file"):
            detector._load_model()

    def test_missing_onnxruntime_raises_actionable_runtime_error(
        self, tmp_path: Path
    ) -> None:
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake onnx data")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(fake_onnx),
            )
        )

        def mock_import(name: str) -> Any:
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(
                RuntimeError,
                match="ONNX Runtime is required when detector backend is 'onnx'",
            ):
                detector._load_model()

    def test_missing_ultralytics_raises_runtime_error(self, tmp_path: Path) -> None:
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake onnx data")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(fake_onnx),
            )
        )

        def mock_import(name: str) -> Any:
            if name == "ultralytics":
                raise ImportError("No module named 'ultralytics'")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(
                RuntimeError,
                match="Ultralytics is required for YOLO26Detector",
            ):
                detector._load_model()

    def test_unsupported_backend_raises_value_error(self) -> None:
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="invalid_backend",  # type: ignore[arg-type]
            )
        )
        with pytest.raises(ValueError, match="Unsupported detector backend 'invalid_backend'"):
            detector._load_model()

    def test_warmup_propagates_missing_model_file_not_found(self, tmp_path: Path) -> None:
        non_existent = str(tmp_path / "absent_detector.onnx")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=non_existent,
            )
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            detector.warmup()
        assert detector._is_warm is False
        assert detector._model is None

    def test_warmup_propagates_missing_onnxruntime_runtime_error(self, tmp_path: Path) -> None:
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake onnx data")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(fake_onnx),
            )
        )

        def mock_import(name: str) -> Any:
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="ONNX Runtime is required"):
                detector.warmup()
        assert detector._is_warm is False
        assert detector._model is None

    def test_detect_propagates_missing_model_file_not_found(self, tmp_path: Path) -> None:
        non_existent = str(tmp_path / "absent_detector.onnx")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=non_existent,
            )
        )
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        decoded = DecodedFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            frame_array=dummy_frame,
            width=100,
            height=100,
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            detector.detect(decoded)

    def test_warmup_handles_dummy_inference_attribute_error_gracefully(self) -> None:
        class FlakyModel:
            def __call__(self, *args: object, **kwargs: object) -> Any:
                raise AttributeError("Simulated unsupported inference attribute")

        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="models/yolo26n.onnx",
            ),
            model=FlakyModel(),
        )
        detector._is_warm = False
        detector.warmup()
        assert detector._is_warm is False
        assert detector._model is None


class TestONNXDetectorInferenceAndSchema:
    def test_detection_output_schema_and_coordinates(self) -> None:
        mock_engine = MockYOLOModel(
            result=MockResult(
                boxes=MockBoxes(
                    xyxy=[[40.0, 60.0, 160.0, 260.0]],
                    conf=[0.89],
                    cls=[0],
                ),
                names={0: "person"},
            )
        )
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="models/yolo26n.onnx",
                confidence_threshold=0.3,
                iou_threshold=0.5,
            ),
            model=mock_engine,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = FrameTimestamp(
            frame_index=15,
            pts_seconds=0.6,
            timestamp_source=TimestampSource.CONTAINER,
        )
        decoded = DecodedFrame(
            frame_index=15,
            timestamp_seconds=0.6,
            frame_array=dummy_frame,
            width=640,
            height=480,
            frame_timestamp=ts,
        )

        detections = detector.detect(decoded)
        assert len(detections) == 1
        d = detections[0]

        # Verify domain schema & coordinates
        assert d.frame_number == 15
        assert d.timestamp_seconds == 0.6
        assert d.frame_timestamp == ts
        assert d.class_id == 0
        assert d.class_name == "person"
        assert d.confidence == pytest.approx(0.89, abs=1e-3)
        assert d.provider == "yolo26_onnx_yolo26n"

        # Coordinates: xyxy [40, 60, 160, 260] -> x=40, y=60, w=120, h=200
        assert d.bbox.coordinate_type == CoordinateType.PIXEL
        assert d.bbox.x == 40.0
        assert d.bbox.y == 60.0
        assert d.bbox.width == 120.0
        assert d.bbox.height == 200.0

        # Provenance attributes
        assert d.attributes["source_provider"] == "yolo26_onnx_yolo26n"
        assert d.attributes["backend"] == "onnx"
        assert d.attributes["timestamp_source"] == "container"

    def test_inference_error_wrapping(self) -> None:
        class FailingEngine:
            def __call__(self, *args: object, **kwargs: object) -> Any:
                raise RuntimeError("ONNX ExecutionProvider CUDA error")

        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="models/yolo26n.onnx",
            ),
            model=FailingEngine(),
        )

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        decoded = DecodedFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            frame_array=dummy_frame,
            width=100,
            height=100,
        )

        with pytest.raises(RuntimeError, match="Inference failed on detector"):
            detector.detect(decoded)


class TestDetectorFactory:
    def test_create_detector_default_settings(self) -> None:
        det = create_detector()
        assert isinstance(det, YOLO26Detector)
        assert det.config.backend == "pytorch"
        assert det.config.model_path == Settings().detector_model_path

    def test_create_detector_with_onnx_settings(self) -> None:
        settings = Settings(
            detector_backend="onnx",
            detector_onnx_model_path="models/yolo26n.onnx",
            detector_confidence_threshold=0.35,
            detector_iou_threshold=0.55,
        )
        det = create_detector(settings=settings)
        assert isinstance(det, YOLO26Detector)
        assert det.config.backend == "onnx"
        assert det.config.onnx_model_path == "models/yolo26n.onnx"
        assert det.config.confidence_threshold == 0.35
        assert det.config.iou_threshold == 0.55

    def test_create_detector_kwargs_override(self) -> None:
        settings = Settings(detector_backend="pytorch")
        det = create_detector(settings=settings, backend="onnx", confidence_threshold=0.4)
        assert isinstance(det, YOLO26Detector)
        assert det.config.backend == "onnx"
        assert det.config.confidence_threshold == 0.4
