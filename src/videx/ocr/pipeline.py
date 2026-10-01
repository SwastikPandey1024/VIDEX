"""Integrated OCR Evidence Pipeline coordinating scheduling, routing,
fusion, and evidence records.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from videx.domain.schemas import Evidence, EvidenceType, Frame, OCRObservation, TextObservation
from videx.ingestion.base import DecodedFrame
from videx.ocr.base import resolve_frame_input
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.router import OCRRouter
from videx.ocr.scheduler import OCRScheduler, OCRSchedulerConfig
from videx.providers.base import OCRProvider


@dataclass
class OCRPipelineResult:
    """Consolidated results of an OCR pipeline execution over a video timeline."""

    video_id: UUID
    raw_observations: list[OCRObservation]
    fused_observations: list[TextObservation]
    evidence_records: list[Evidence]
    frames_evaluated: int
    frames_processed: int
    total_latency_ms: float

    @property
    def mean_latency_per_processed_frame_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_latency_ms / self.frames_processed


class OCRPipeline:
    """Coordinates frame selection, multilingual routing, temporal fusion, and evidence creation."""

    def __init__(
        self,
        router: OCRRouter | OCRProvider | None = None,
        scheduler: OCRScheduler | None = None,
        fusion: TemporalOCRFusion | None = None,
    ) -> None:
        if router is None:
            self._router = OCRRouter()
        elif isinstance(router, OCRRouter):
            self._router = router
        else:
            # Wrap standalone provider into a simple router
            self._router = OCRRouter(
                providers={"general": router},
            )

        self._scheduler = scheduler or OCRScheduler(OCRSchedulerConfig())
        self._fusion = fusion or TemporalOCRFusion(TemporalOCRFusionConfig())

        self._raw_observations: list[OCRObservation] = []
        self._frames_evaluated = 0
        self._frames_processed = 0
        self._total_latency_ms = 0.0
        self._video_id: UUID | None = None

    def reset(self) -> None:
        """Reset internal accumulator state for evaluating a new video sequence."""
        self._raw_observations.clear()
        self._frames_evaluated = 0
        self._frames_processed = 0
        self._total_latency_ms = 0.0
        self._video_id = None
        cfg = getattr(self._fusion, "config", TemporalOCRFusionConfig())
        self._fusion = TemporalOCRFusion(cfg)

    def process_frames(
        self,
        frames: Sequence[DecodedFrame | Frame],
        language: str | None = None,
        script: str | None = None,
        reset: bool = True,
    ) -> tuple[list[OCRObservation], list[TextObservation], list[Evidence]]:
        """Process a sequence of frames and return (raw_obs, fused_obs, evidence_records)."""
        if reset:
            self.reset()
        for frame in frames:
            self.process_frame(frame, language=language, script=script)
        result = self.finalize()
        return result.raw_observations, result.fused_observations, result.evidence_records

    def process_frame(
        self,
        frame: DecodedFrame | Frame,
        language: str | None = None,
        script: str | None = None,
    ) -> list[OCRObservation]:
        """Evaluate a frame according to scheduling policy and extract OCR text."""
        resolved = resolve_frame_input(frame)
        self._frames_evaluated += 1
        if self._video_id is None:
            self._video_id = resolved.video_id

        # Check scheduling policy
        if not self._scheduler.should_process(resolved.frame_number, resolved.timestamp_seconds):
            # Still update fusion timeline to track elapsed frame gap
            self._fusion.update([], current_frame=resolved.frame_number)
            return []

        self._frames_processed += 1
        t0 = time.perf_counter()
        observations = self._router.detect_text(frame, language=language, script=script)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += elapsed_ms

        self._raw_observations.extend(observations)
        self._fusion.update(observations, current_frame=resolved.frame_number)
        return observations

    def finalize(self) -> OCRPipelineResult:
        """Close all temporal fusion clusters and compile canonical Evidence records."""
        fused_texts = self._fusion.finalize()
        evidence_records: list[Evidence] = []

        for fused in fused_texts:
            payload = {
                "text": fused.text,
                "normalized_text": fused.normalized_text,
                "language": fused.language,
                "script": fused.script,
                "first_seen_frame": fused.first_seen_frame,
                "last_seen_frame": fused.last_seen_frame,
                "first_seen_timestamp_seconds": fused.first_seen_timestamp_seconds,
                "last_seen_timestamp_seconds": fused.last_seen_timestamp_seconds,
                "supporting_frames": fused.supporting_frames,
                "confidence_summary": fused.confidence_summary,
                "provider": fused.provider,
                "frame_timestamp": {
                    "pts_seconds": fused.first_seen_timestamp.pts_seconds,
                    "frame_index": fused.first_seen_timestamp.frame_index,
                    "timestamp_source": fused.first_seen_timestamp.timestamp_source.value,
                    "is_repaired": fused.first_seen_timestamp.is_repaired,
                },
            }

            ev = Evidence(
                evidence_type=EvidenceType.OCR,
                source_module=fused.provider,
                video_id=fused.video_id,
                frame_id=None,
                timestamp_seconds=fused.first_seen_timestamp_seconds,
                bbox=fused.latest_bbox,
                confidence=fused.mean_confidence,
                description=(
                    f"Recognized text '{fused.text}' across frames "
                    f"{fused.first_seen_frame}..{fused.last_seen_frame}"
                ),
                raw_payload=payload,
                supporting_observation_ids=fused.supporting_observation_ids,
            )
            evidence_records.append(ev)

        assert self._video_id is not None or fused_texts == []

        return OCRPipelineResult(
            video_id=self._video_id or (fused_texts[0].video_id if fused_texts else None),  # type: ignore[arg-type]
            raw_observations=list(self._raw_observations),
            fused_observations=fused_texts,
            evidence_records=evidence_records,
            frames_evaluated=self._frames_evaluated,
            frames_processed=self._frames_processed,
            total_latency_ms=self._total_latency_ms,
        )
