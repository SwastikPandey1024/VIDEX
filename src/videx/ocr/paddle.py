"""PaddleOCR provider implementation and specialized routes.

Supports both general Latin/multilingual and dedicated Indic/Devanagari routes.
Includes dynamic import isolation and dependency injection for zero-download testing.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from videx.domain.schemas import BoundingBox, Frame, OCRObservation
from videx.ingestion.base import DecodedFrame
from videx.ocr.base import PaddleOCRConfig, resolve_frame_input
from videx.ocr.normalization import normalize_text
from videx.providers.base import FrameBytes


class PaddleOCRProvider:
    """Production OCR adapter wrapping PaddleOCR.

    Maintains strict separation between Latin/multilingual models and specialized
    Indic language models. Preserves authoritative FrameTimestamp temporal provenance.
    """

    def __init__(
        self,
        config: PaddleOCRConfig | None = None,
        injected_engine: object | None = None,
    ) -> None:
        self.config = config or PaddleOCRConfig()
        self._provider_name = f"paddle_{self.config.language}"
        self._engine: Any = injected_engine
        self._is_initialized = injected_engine is not None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return (self.config.language,)

    def warmup(self) -> None:
        """Initialize and warm up PaddleOCR engine."""
        if self._engine is not None:
            return

        try:
            try:
                import torch  # noqa: F401  # Pre-import on Windows to avoid shm.dll conflicts
            except ImportError:
                pass

            paddleocr_mod = importlib.import_module("paddleocr")
            paddle_cls = paddleocr_mod.PaddleOCR

            import inspect

            sig = inspect.signature(paddle_cls.__init__)
            param_names = set(sig.parameters.keys())

            kwargs: dict[str, Any] = {}
            if "lang" in param_names:
                kwargs["lang"] = self.config.language
            if "use_angle_cls" in param_names:
                kwargs["use_angle_cls"] = self.config.use_angle_cls
            if "device" in param_names:
                kwargs["device"] = (
                    self.config.device
                    if self.config.device
                    else ("gpu" if self.config.use_gpu else "cpu")
                )
            elif "use_gpu" in param_names:
                kwargs["use_gpu"] = self.config.use_gpu or (self.config.device == "cuda")

            # Disable mkldnn on CPU by default to prevent oneDNN PIR runtime error on Windows
            if "enable_mkldnn" in param_names or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["enable_mkldnn"] = False

            if "show_log" in param_names:
                kwargs["show_log"] = False

            if self.config.custom_model_dir:
                if "det_model_dir" in param_names:
                    kwargs["det_model_dir"] = self.config.custom_model_dir
                elif "text_detection_model_dir" in param_names:
                    kwargs["text_detection_model_dir"] = self.config.custom_model_dir

            kwargs.update(self.config.extra_options)

            self._engine = paddle_cls(**kwargs)
            self._is_initialized = True
        except (ImportError, AttributeError, Exception) as err:
            raise RuntimeError(
                f"PaddleOCR is required for '{self.provider_name}' but is not installed "
                f"or failed to initialize: {err}"
            ) from err

    def detect_text(self, frame: DecodedFrame | Frame) -> list[OCRObservation]:
        """Detect and recognise text in a single video frame.

        Args:
            frame: DecodedFrame with raw numpy pixel array or Frame domain model.

        Returns:
            List of OCRObservation records with validated coordinates and timestamps.
        """
        resolved = resolve_frame_input(frame)

        # Ensure image is in BGR numpy array format
        img_array = resolved.frame_array
        if img_array is None:
            return []

        if not self._is_initialized or self._engine is None:
            self.warmup()

        # Execute OCR engine
        # PaddleOCR 3.x / PaddleX: predict() returns [OCRResult, ...]
        # PaddleOCR 2.x legacy: ocr() returns [ [ [ [x1,y1],... ], (text, confidence) ], ... ]
        if hasattr(self._engine, "predict"):
            raw_results = self._engine.predict(img_array)
        else:
            raw_results = self._engine.ocr(img_array, cls=self.config.use_angle_cls)

        observations: list[OCRObservation] = []
        if not raw_results or not raw_results[0]:
            return observations

        page_results = raw_results[0]

        # Check for PaddleX / PaddleOCR 3.x dict-like result structure
        if isinstance(page_results, dict) or (
            hasattr(page_results, "__getitem__") and "rec_texts" in page_results
        ):
            rec_texts = page_results.get("rec_texts", [])
            rec_scores = page_results.get("rec_scores", [])
            rec_polys = page_results.get("rec_polys", [])

            for text_str, conf, poly_pts in zip(rec_texts, rec_scores, rec_polys, strict=False):
                conf_val = float(conf)
                if conf_val < self.config.rec_thresh:
                    continue

                pts_arr = np.array(poly_pts, dtype=np.float32)
                min_x = float(np.min(pts_arr[:, 0]))
                max_x = float(np.max(pts_arr[:, 0]))
                min_y = float(np.min(pts_arr[:, 1]))
                max_y = float(np.max(pts_arr[:, 1]))

                width = max(max_x - min_x, 1.0)
                height = max(max_y - min_y, 1.0)

                bbox = BoundingBox(
                    x=min_x,
                    y=min_y,
                    width=width,
                    height=height,
                )
                polygon_list = [[float(pt[0]), float(pt[1])] for pt in pts_arr]
                norm_text = normalize_text(text_str)

                attr: dict[str, Any] = {
                    "source_provider": self.provider_name,
                    "model_name": self.config.model_name,
                    "timestamp_source": resolved.frame_timestamp.timestamp_source.value,
                    "is_repaired": resolved.frame_timestamp.is_repaired,
                }
                if resolved.frame_timestamp.repair_reason:
                    attr["repair_reason"] = resolved.frame_timestamp.repair_reason
                if resolved.frame_timestamp.original_pts_seconds is not None:
                    attr["original_pts_seconds"] = resolved.frame_timestamp.original_pts_seconds

                obs = OCRObservation(
                    observation_id=uuid4(),
                    frame_id=resolved.frame_id,
                    video_id=resolved.video_id,
                    frame_number=resolved.frame_number,
                    timestamp_seconds=resolved.timestamp_seconds,
                    frame_timestamp=resolved.frame_timestamp,
                    text=text_str,
                    normalized_text=norm_text,
                    confidence=conf_val,
                    bbox=bbox,
                    polygon=polygon_list,
                    language=self.config.language,
                    script=self.config.script,
                    provider=self.provider_name,
                    recognition_confidence=conf_val,
                    detection_confidence=conf_val,
                    attributes=attr,
                )
                observations.append(obs)
            return observations

        # Legacy 2.x list-of-lines structure
        for line in page_results:
            if not line or len(line) < 2:
                continue

            poly_pts, (text_str, conf) = line[0], line[1]
            conf_val = float(conf)
            if conf_val < self.config.rec_thresh:
                continue

            pts_arr = np.array(poly_pts, dtype=np.float32)
            min_x = float(np.min(pts_arr[:, 0]))
            max_x = float(np.max(pts_arr[:, 0]))
            min_y = float(np.min(pts_arr[:, 1]))
            max_y = float(np.max(pts_arr[:, 1]))

            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)

            bbox = BoundingBox(
                x=min_x,
                y=min_y,
                width=width,
                height=height,
            )
            polygon_list = [[float(pt[0]), float(pt[1])] for pt in poly_pts]

            norm_text = normalize_text(text_str)

            attr = {
                "source_provider": self.provider_name,
                "model_name": self.config.model_name,
                "timestamp_source": resolved.frame_timestamp.timestamp_source.value,
                "is_repaired": resolved.frame_timestamp.is_repaired,
            }
            if resolved.frame_timestamp.repair_reason:
                attr["repair_reason"] = resolved.frame_timestamp.repair_reason
            if resolved.frame_timestamp.original_pts_seconds is not None:
                attr["original_pts_seconds"] = resolved.frame_timestamp.original_pts_seconds

            obs = OCRObservation(
                observation_id=uuid4(),
                frame_id=resolved.frame_id,
                video_id=resolved.video_id,
                frame_number=resolved.frame_number,
                timestamp_seconds=resolved.timestamp_seconds,
                frame_timestamp=resolved.frame_timestamp,
                text=text_str,
                normalized_text=norm_text,
                confidence=conf_val,
                bbox=bbox,
                polygon=polygon_list,
                language=self.config.language,
                script=self.config.script,
                provider=self.provider_name,
                recognition_confidence=conf_val,
                detection_confidence=conf_val,
                attributes=attr,
            )
            observations.append(obs)

        return observations

    def detect_text_batch(
        self, frames: Sequence[DecodedFrame | Frame]
    ) -> list[list[OCRObservation]]:
        """Batch process frames."""
        return [self.detect_text(f) for f in frames]

    def recognise(self, frame_data: FrameBytes, frame_meta: Frame) -> list[OCRObservation]:
        """Legacy protocol compatibility: decodes JPEG bytes into array."""
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decoded = DecodedFrame(
            frame_index=frame_meta.frame_number,
            timestamp_seconds=frame_meta.timestamp_seconds,
            width=img.shape[1] if img is not None else frame_meta.width,
            height=img.shape[0] if img is not None else frame_meta.height,
            frame_array=img if img is not None else np.zeros((10, 10, 3), dtype=np.uint8),
            video_id=frame_meta.video_id,
            frame_timestamp=frame_meta.frame_timestamp,
        )
        return self.detect_text(decoded)


class PaddleGeneralOCRProvider(PaddleOCRProvider):
    """General Latin/English OCR route using standard PP-OCR multilingual baseline.

    Explicitly designated for Chinese, English, Latin-script, and common European languages.
    """

    def __init__(
        self,
        config: PaddleOCRConfig | None = None,
        injected_engine: object | None = None,
    ) -> None:
        cfg = config or PaddleOCRConfig(
            model_name="PP-OCRv6_mobile",
            language="en",
            script="Latin",
        )
        super().__init__(cfg, injected_engine=injected_engine)
        self._provider_name = cfg.provider_name or "paddle_general"


class PaddleIndicOCRProvider(PaddleOCRProvider):
    """Dedicated Hindi and Indic language OCR route using Devanagari model family.

    Rule: Never represent PP-OCRv6 general model as the Hindi engine.
    This route explicitly loads the dedicated Devanagari / Indic multilingual model.
    """

    def __init__(
        self,
        config: PaddleOCRConfig | None = None,
        injected_engine: object | None = None,
    ) -> None:
        cfg = config or PaddleOCRConfig(
            model_name="devanagari_PP-OCRv5_mobile_rec",
            language="hi",
            script="Devanagari",
            use_angle_cls=True,
        )
        super().__init__(cfg, injected_engine=injected_engine)
        self._provider_name = cfg.provider_name or "paddle_indic"
