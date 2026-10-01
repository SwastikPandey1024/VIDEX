"""Deterministic audio speech event detector.

Consumes TranscriptSegment records from the AudioPipeline to emit
SPEECH_STARTED, SPEECH_ENDED, and SPEECH_DETECTED events with authoritative audio timestamps.
SoundEventProvider is preserved as an extension boundary for future acoustic event models.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from videx.domain.schemas import EvidenceType, TranscriptSegment
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
    EventParticipant,
)
from videx.events.types import EventSeverity, EventStatus, EventType


class AudioEventDetector:
    """Detects deterministic speech events from time-aligned transcript segments."""

    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()

    def detect_events(
        self,
        transcript_segments: Sequence[TranscriptSegment],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Convert transcript segments into grounded speech events.

        Args:
            transcript_segments: Time-aligned speech segments from ASR pipeline.
            video_id: Video UUID override.

        Returns:
            List of evidence-backed audio Event records.
        """
        events: list[Event] = []

        for seg in transcript_segments:
            v_id = video_id or seg.video_id
            participant = EventParticipant(
                participant_id=seg.segment_id,
                participant_type="speaker",
                role="speaker",
                label=seg.language or "speech",
                metadata={
                    "language": seg.language,
                    "provider": seg.provider,
                    "word_count": len(seg.words),
                },
            )

            # ── 1. SPEECH_STARTED (Instantaneous) ─────────────────────────
            start_ev = EventEvidence(
                timestamp_seconds=seg.start_timestamp_seconds,
                evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                role="trigger",
                transcript_segment_id=seg.segment_id,
                provenance={
                    "language": seg.language,
                    "provider": seg.provider,
                },
            )

            ev_start = Event(
                video_id=v_id,
                event_type=EventType.SPEECH_STARTED,
                start_timestamp_seconds=seg.start_timestamp_seconds,
                end_timestamp_seconds=seg.start_timestamp_seconds,
                confidence=seg.confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[start_ev],
                source_module="audio_event_detector",
                description=(
                    f"Speech started at {seg.start_timestamp_seconds:.2f}s "
                    f"[{seg.language or 'unknown'}]"
                ),
                attributes={
                    "language": seg.language,
                    "provider": seg.provider,
                },
            )
            events.append(ev_start)

            # ── 2. SPEECH_DETECTED (Interval) ─────────────────────────────
            detect_ev = EventEvidence(
                timestamp_seconds=seg.start_timestamp_seconds,
                evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                role="supporting",
                transcript_segment_id=seg.segment_id,
                provenance={
                    "raw_text": seg.raw_text,
                    "normalized_text": seg.normalized_text,
                    "duration_seconds": seg.duration_seconds,
                    "word_count": len(seg.words),
                },
            )

            ev_detected = Event(
                video_id=v_id,
                event_type=EventType.SPEECH_DETECTED,
                start_timestamp_seconds=seg.start_timestamp_seconds,
                end_timestamp_seconds=seg.end_timestamp_seconds,
                confidence=seg.confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[detect_ev],
                source_module="audio_event_detector",
                description=(
                    f"Spoken utterance: '{seg.raw_text}' "
                    f"({seg.start_timestamp_seconds:.2f}s - {seg.end_timestamp_seconds:.2f}s)"
                ),
                attributes={
                    "raw_text": seg.raw_text,
                    "normalized_text": seg.normalized_text,
                    "language": seg.language,
                    "provider": seg.provider,
                    "duration_seconds": round(seg.duration_seconds, 2),
                    "word_count": len(seg.words),
                },
            )
            events.append(ev_detected)

            # ── 3. SPEECH_ENDED (Instantaneous) ───────────────────────────
            end_ev = EventEvidence(
                timestamp_seconds=seg.end_timestamp_seconds,
                evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                role="trigger",
                transcript_segment_id=seg.segment_id,
                provenance={
                    "language": seg.language,
                    "provider": seg.provider,
                },
            )

            ev_end = Event(
                video_id=v_id,
                event_type=EventType.SPEECH_ENDED,
                start_timestamp_seconds=seg.end_timestamp_seconds,
                end_timestamp_seconds=seg.end_timestamp_seconds,
                confidence=seg.confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[end_ev],
                source_module="audio_event_detector",
                description=(
                    f"Speech ended at {seg.end_timestamp_seconds:.2f}s "
                    f"[{seg.language or 'unknown'}]"
                ),
                attributes={
                    "language": seg.language,
                    "provider": seg.provider,
                },
            )
            events.append(ev_end)

        return events
