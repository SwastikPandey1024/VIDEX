"""Audio intelligence pipeline: extraction, ASR transcription, fusion, evidence generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from videx.audio.base import ASRProvider, MockASRConfig
from videx.audio.extraction import NoAudioStreamError, extract_audio_stream
from videx.audio.fusion import TemporalTranscriptFusion, TemporalTranscriptFusionConfig
from videx.audio.mock import MockASRProvider
from videx.audio.whisper import FasterWhisperASRProvider
from videx.domain.schemas import (
    AudioMetadata,
    Evidence,
    EvidenceType,
    SoundObservation,
    TranscriptSegment,
    Video,
)

logger = logging.getLogger(__name__)

__all__ = ["AudioPipeline", "AudioPipelineResult"]


@dataclass(frozen=True)
class AudioPipelineResult:
    """Result payload returned by AudioPipeline execution."""

    video_id: UUID
    audio_metadata: AudioMetadata | None
    raw_segments: list[TranscriptSegment]
    fused_segments: list[TranscriptSegment]
    evidence: list[Evidence]
    sound_observations: list[SoundObservation] = field(default_factory=list)
    duration_seconds: float = 0.0
    language: str = "en"
    processing_time_seconds: float = 0.0


class AudioPipeline:
    """End-to-end Audio Intelligence and Timestamped ASR Pipeline.

    Integrates:
    - Audio track extraction and format normalization (PyAV).
    - Speech recognition (faster-whisper or mock provider).
    - Conservative Unicode transcript normalization.
    - Temporal transcript deduplication and fusion.
    - Structured Evidence generation with temporal provenance.
    """

    def __init__(
        self,
        asr_provider: ASRProvider | None = None,
        fusion: TemporalTranscriptFusion | None = None,
    ) -> None:
        self.asr_provider = asr_provider or FasterWhisperASRProvider()
        self.fusion = fusion or TemporalTranscriptFusion()

    @classmethod
    def create_mock(
        cls,
        config: MockASRConfig | None = None,
        fusion_config: TemporalTranscriptFusionConfig | None = None,
    ) -> AudioPipeline:
        """Create a pipeline configured with MockASRProvider for testing."""
        return cls(
            asr_provider=MockASRProvider(config),
            fusion=TemporalTranscriptFusion(fusion_config),
        )

    def warmup(self) -> None:
        """Warm up underlying ASR model and runtime context."""
        self.asr_provider.warmup()

    def process_video(
        self,
        video_source: str | Path | Video,
        language: str | None = None,
        video_id: UUID | None = None,
    ) -> AudioPipelineResult:
        """Process video audio track and return full transcription and evidence.

        Args:
            video_source: Video file path or Video domain model.
            language: Optional language ISO code hint ('en', 'hi').
            video_id: Optional parent video UUID override.

        Returns:
            AudioPipelineResult with segments and Evidence records.

        Raises:
            NoAudioStreamError: If the video has no audio stream.
            AudioExtractionError: If audio extraction fails.
        """
        start_time = time.perf_counter()

        if isinstance(video_source, Video):
            path_str = video_source.source_path
            v_id = video_id or video_source.video_id
        else:
            path_str = str(video_source)
            v_id = video_id or uuid4()

        # Step 1: Extract audio stream from video container
        try:
            audio_arr, meta = extract_audio_stream(path_str, target_sample_rate=16000)
            duration_sec = meta.duration_seconds
        except NoAudioStreamError:
            logger.info("Video %s contains no audio stream. Returning empty audio result.", v_id)
            return AudioPipelineResult(
                video_id=v_id,
                audio_metadata=None,
                raw_segments=[],
                fused_segments=[],
                evidence=[],
                sound_observations=[],
                duration_seconds=0.0,
                language=language or "en",
                processing_time_seconds=time.perf_counter() - start_time,
            )

        # Step 2: Transcribe using ASR provider
        raw_segments = self.asr_provider.transcribe(
            audio_input=audio_arr,
            video_id=v_id,
            language=language,
        )

        # Step 3: Fuse segments across time
        fused_segments = self.fusion.fuse(raw_segments)

        # Step 4: Generate Evidence records
        evidence_list = self._generate_evidence(fused_segments, v_id)

        detected_lang = raw_segments[0].language if raw_segments else (language or "en")
        elapsed = time.perf_counter() - start_time

        return AudioPipelineResult(
            video_id=v_id,
            audio_metadata=meta,
            raw_segments=raw_segments,
            fused_segments=fused_segments,
            evidence=evidence_list,
            duration_seconds=duration_sec,
            language=detected_lang,
            processing_time_seconds=elapsed,
        )

    def process_audio(
        self,
        audio_input: Any,  # noqa: ANN401
        video_id: UUID | None = None,
        language: str | None = None,
    ) -> AudioPipelineResult:
        """Process pre-extracted audio input directly (array, path, or bytes).

        Args:
            audio_input: Pre-extracted audio.
            video_id: Optional video UUID.
            language: Optional language code.

        Returns:
            AudioPipelineResult.
        """
        start_time = time.perf_counter()
        v_id = video_id or uuid4()

        raw_segments = self.asr_provider.transcribe(
            audio_input=audio_input,
            video_id=v_id,
            language=language,
        )
        fused_segments = self.fusion.fuse(raw_segments)
        evidence_list = self._generate_evidence(fused_segments, v_id)

        detected_lang = raw_segments[0].language if raw_segments else (language or "en")
        elapsed = time.perf_counter() - start_time

        duration_sec = 0.0
        if raw_segments:
            duration_sec = max(s.end_timestamp_seconds for s in raw_segments)

        return AudioPipelineResult(
            video_id=v_id,
            audio_metadata=None,
            raw_segments=raw_segments,
            fused_segments=fused_segments,
            evidence=evidence_list,
            duration_seconds=duration_sec,
            language=detected_lang,
            processing_time_seconds=elapsed,
        )

    def _generate_evidence(
        self,
        segments: list[TranscriptSegment],
        video_id: UUID,
    ) -> list[Evidence]:
        """Convert transcript segments to standard pipeline Evidence records."""
        evidence_records: list[Evidence] = []

        for seg in segments:
            conf = seg.confidence if seg.confidence is not None else 1.0
            conf = max(0.0, min(1.0, conf))

            raw_payload = {
                "raw_text": seg.raw_text,
                "normalized_text": seg.normalized_text,
                "start_timestamp_seconds": seg.start_timestamp_seconds,
                "end_timestamp_seconds": seg.end_timestamp_seconds,
                "duration_seconds": seg.duration_seconds,
                "language": seg.language,
                "provider": seg.provider,
                "speaker_id": seg.speaker_id,
                "words": [
                    {
                        "word": w.word,
                        "start_timestamp_seconds": w.start_timestamp_seconds,
                        "end_timestamp_seconds": w.end_timestamp_seconds,
                        "confidence": w.confidence,
                    }
                    for w in seg.words
                ],
                "attributes": seg.attributes,
            }

            supporting_ids = [seg.segment_id]
            if "supporting_segment_ids" in seg.attributes:
                for sid in seg.attributes["supporting_segment_ids"]:
                    try:
                        u = UUID(str(sid))
                        if u not in supporting_ids:
                            supporting_ids.append(u)
                    except ValueError:
                        pass

            ev = Evidence(
                evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                source_module=f"asr_{seg.provider}",
                video_id=video_id,
                frame_id=None,
                timestamp_seconds=seg.start_timestamp_seconds,
                confidence=conf,
                description=f"Audio transcript [{seg.language}]: {seg.normalized_text}",
                raw_payload=raw_payload,
                supporting_observation_ids=supporting_ids,
                tags=["audio", "asr", seg.language, seg.provider],
                metadata={"end_timestamp_seconds": seg.end_timestamp_seconds},
            )
            evidence_records.append(ev)

        return evidence_records
