"""VIDEX OCR Evidence Pipeline package.

Provides multilingual text detection and recognition, language- and script-aware routing,
conservative normalization, temporal text fusion, and structured evidence propagation.
"""

from videx.ocr.base import (
    MockOCRConfig,
    OCRProvider,
    PaddleOCRConfig,
    ResolvedFrameInput,
    resolve_frame_input,
)
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.normalization import normalize_text
from videx.ocr.paddle import (
    PaddleGeneralOCRProvider,
    PaddleIndicOCRProvider,
    PaddleOCRProvider,
)
from videx.ocr.pipeline import OCRPipeline, OCRPipelineResult
from videx.ocr.router import OCRRouter, OCRRouterConfig
from videx.ocr.scheduler import (
    OCRScheduler,
    OCRSchedulerConfig,
    OCRSchedulingStrategy,
)

__all__ = [
    "MockOCRConfig",
    "MockOCRProvider",
    "OCRPipeline",
    "OCRPipelineResult",
    "OCRProvider",
    "OCRRouter",
    "OCRRouterConfig",
    "OCRScheduler",
    "OCRSchedulerConfig",
    "OCRSchedulingStrategy",
    "PaddleGeneralOCRProvider",
    "PaddleIndicOCRProvider",
    "PaddleOCRConfig",
    "PaddleOCRProvider",
    "ResolvedFrameInput",
    "TemporalOCRFusion",
    "TemporalOCRFusionConfig",
    "normalize_text",
    "resolve_frame_input",
]
