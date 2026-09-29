"""Base types, configurations, and frame resolution utilities for OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from videx.domain.schemas import Frame, FrameTimestamp, TimestampSource
from videx.ingestion.base import DecodedFrame
from videx.providers.base import OCRProvider

__all__ = [
    "MockOCRConfig",
    "OCRProvider",
    "PaddleOCRConfig",
    "ResolvedFrameInput",
    "resolve_frame_input",
]


@dataclass(frozen=True)
class PaddleOCRConfig:
    """Configuration for PaddleOCR runtime engines.

    Distinguishes model family, language, script, and inference thresholds.
    """

    model_name: str = "PP-OCRv6_mobile"
    language: str = "en"
    script: str = "Latin"
    device: str = "cpu"
    use_angle_cls: bool = True
    det_db_thresh: float = 0.3
    rec_thresh: float = 0.5
    use_gpu: bool = False
    enable_mkldnn: bool = False
    provider_name: str | None = None
    custom_model_dir: str | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockOCRConfig:
    """Configuration for deterministic testing mock OCR provider."""

    provider_name: str = "mock_ocr"
    supported_languages: tuple[str, ...] = ("en", "hi")
    canned_results_by_frame: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    canned_texts: list[str] | None = None
    canned_bboxes: list[Any] | None = None
    canned_confidence: float | None = None
    default_text: str = "SAMPLE TEXT"
    default_language: str = "en"
    default_script: str = "Latin"
    default_confidence: float = 0.95
    latency_ms: float = 0.0
    language: str | None = None
    script: str | None = None

    def __post_init__(self) -> None:
        if self.language is not None:
            self.default_language = self.language
        if self.script is not None:
            self.default_script = self.script


@dataclass(frozen=True)
class ResolvedFrameInput:
    """Extracted frame metadata and raw array for OCR engines.

    Guarantees strict temporal provenance inheritance from ingestion.
    """

    frame_id: UUID
    video_id: UUID
    frame_number: int
    timestamp_seconds: float
    frame_timestamp: FrameTimestamp
    width: int
    height: int
    frame_array: np.ndarray | None = None


def resolve_frame_input(frame: DecodedFrame | Frame) -> ResolvedFrameInput:
    """Extract authoritative frame metadata and image array without independent clock drift.

    Rule: OCR must NEVER calculate its own timestamp from frame_number / fps.
    It MUST inherit the authoritative FrameTimestamp from ingestion.

    Args:
        frame: Either DecodedFrame or Frame domain model.

    Returns:
        ResolvedFrameInput with validated temporal provenance.
    """
    if isinstance(frame, DecodedFrame):
        f_meta = frame.to_domain_frame()
        frame_ts = frame.frame_timestamp
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=frame.frame_index,
                pts_seconds=frame.timestamp_seconds,
                timestamp_source=TimestampSource.DERIVED,
            )
        arr = frame.frame_array if isinstance(frame.frame_array, np.ndarray) else None
        return ResolvedFrameInput(
            frame_id=f_meta.frame_id,
            video_id=f_meta.video_id,
            frame_number=frame.frame_index,
            timestamp_seconds=frame_ts.pts_seconds,
            frame_timestamp=frame_ts,
            width=frame.width,
            height=frame.height,
            frame_array=arr,
        )

    if isinstance(frame, Frame):
        frame_ts = frame.frame_timestamp
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=frame.frame_number,
                pts_seconds=frame.timestamp_seconds,
                timestamp_source=TimestampSource.DERIVED,
            )
        return ResolvedFrameInput(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_number=frame.frame_number,
            timestamp_seconds=frame_ts.pts_seconds,
            frame_timestamp=frame_ts,
            width=frame.width,
            height=frame.height,
            frame_array=None,
        )

    raise TypeError(
        f"Expected DecodedFrame or Frame, got unsupported input type: {type(frame).__name__}"
    )
