"""Deterministic Mock OCR provider for hermetic testing and verification."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from videx.domain.schemas import BoundingBox, Frame, OCRObservation
from videx.ingestion.base import DecodedFrame
from videx.ocr.base import MockOCRConfig, resolve_frame_input
from videx.ocr.normalization import normalize_text
from videx.providers.base import FrameBytes


class MockOCRProvider:
    """Hermetic deterministic OCR provider for tests and CI execution."""

    def __init__(self, config: MockOCRConfig | None = None) -> None:
        self.config = config or MockOCRConfig()
        self._warmed_up = False

    @property
    def provider_name(self) -> str:
        return self.config.provider_name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self.config.supported_languages

    def warmup(self) -> None:
        """Simulate model load / context allocation."""
        self._warmed_up = True

    def detect_text(self, frame: DecodedFrame | Frame) -> list[OCRObservation]:
        """Generate deterministic OCR observations for a frame."""
        resolved = resolve_frame_input(frame)

        if self.config.latency_ms > 0:
            time.sleep(self.config.latency_ms / 1000.0)

        # Look up canned results for this specific frame index
        canned_entries = self.config.canned_results_by_frame.get(resolved.frame_number)

        if canned_entries is None and self.config.canned_texts is not None:
            # Construct canned entries from canned_texts / canned_bboxes
            canned_entries = []
            for i, text in enumerate(self.config.canned_texts):
                box = None
                if self.config.canned_bboxes and i < len(self.config.canned_bboxes):
                    box = self.config.canned_bboxes[i]
                conf = (
                    self.config.canned_confidence
                    if self.config.canned_confidence is not None
                    else self.config.default_confidence
                )
                canned_entries.append({
                    "text": text,
                    "bbox": box,
                    "confidence": conf,
                    "language": self.config.default_language,
                    "script": self.config.default_script,
                })

        if canned_entries is None:
            return []

        observations: list[OCRObservation] = []
        for entry in canned_entries:
            text = entry.get("text", self.config.default_text)
            norm_text = entry.get("normalized_text", normalize_text(text))
            conf = float(entry.get("confidence", self.config.default_confidence))
            lang = entry.get("language", self.config.default_language)
            script = entry.get("script", self.config.default_script)
            bbox = entry.get("bbox")
            polygon = entry.get("polygon")

            # Default bounding box if none specified
            if bbox is None and polygon is None:
                bbox = BoundingBox(x=10.0, y=10.0, width=120.0, height=30.0)

            # Build polygon from bbox if needed
            if polygon is None and bbox is not None:
                polygon = [
                    [bbox.x, bbox.y],
                    [bbox.x + bbox.width, bbox.y],
                    [bbox.x + bbox.width, bbox.y + bbox.height],
                    [bbox.x, bbox.y + bbox.height],
                ]

            attr: dict[str, Any] = {
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
                text=text,
                normalized_text=norm_text,
                confidence=conf,
                bbox=bbox,
                polygon=polygon,
                language=lang,
                script=script,
                provider=self.provider_name,
                recognition_confidence=conf,
                detection_confidence=conf,
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
        """Legacy protocol compatibility method."""
        return self.detect_text(frame_meta)
