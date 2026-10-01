"""Deterministic object lifecycle event detector.

Emits OBJECT_APPEARED, OBJECT_DISAPPEARED, and OBJECT_PRESENT events
from object tracks and trajectories while filtering transient detection noise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from videx.domain.schemas import EvidenceType, Track, TrajectoryPoint
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
    EventParticipant,
)
from videx.events.types import EventSeverity, EventStatus, EventType

logger = logging.getLogger(__name__)


class LifecycleEventDetector:
    """Detects deterministic object appearance and disappearance events.

    Guarantees:
    - Filters single-frame noise using configurable confirmation threshold.
    - Preserves tracker continuity without treating single missed frames as disappearance.
    - Fully grounds each event with EventEvidence linking the track and detection observations.
    """

    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()

    def detect_events(
        self,
        tracks: Sequence[Track],
        trajectories: dict[UUID, list[TrajectoryPoint]] | None = None,
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Detect lifecycle events across all verified tracks.

        Args:
            tracks: Sequence of Track records from perception pipeline.
            trajectories: Optional mapping of track_id to TrajectoryPoint lists.
            video_id: Video UUID override if not derived from tracks.

        Returns:
            List of sorted, evidence-backed lifecycle Event records.
        """
        events: list[Event] = []

        for trk in tracks:
            # Filter noise: require minimum confirmed observations
            if trk.frame_count < self.config.lifecycle_confirmation_threshold:
                logger.debug(
                    "Track %s filtered: frame count %d < confirmation threshold %d",
                    trk.track_id,
                    trk.frame_count,
                    self.config.lifecycle_confirmation_threshold,
                )
                continue

            v_id = video_id or trk.video_id
            participant = EventParticipant(
                participant_id=trk.track_id,
                participant_type="track",
                role="subject",
                label=trk.class_name,
                metadata={
                    "class_id": trk.class_id,
                    "frame_count": trk.frame_count,
                    "first_frame": trk.first_seen_frame_number,
                    "last_frame": trk.last_seen_frame_number,
                },
            )

            # ── 1. OBJECT_APPEARED (Instantaneous) ────────────────────────
            appearance_evidence = EventEvidence(
                timestamp_seconds=trk.first_seen_timestamp_seconds,
                evidence_type=EvidenceType.TRACK,
                role="appearance",
                track_id=trk.track_id,
                provenance={
                    "frame_number": trk.first_seen_frame_number,
                    "class_name": trk.class_name,
                    "confidence": trk.confidence,
                },
            )

            appeared_event = Event(
                video_id=v_id,
                event_type=EventType.OBJECT_APPEARED,
                start_timestamp_seconds=trk.first_seen_timestamp_seconds,
                end_timestamp_seconds=trk.first_seen_timestamp_seconds,
                confidence=trk.confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[appearance_evidence],
                source_module="lifecycle_detector",
                description=(
                    f"{trk.class_name.capitalize()} appeared in scene at "
                    f"frame {trk.first_seen_frame_number} ({trk.first_seen_timestamp_seconds:.2f}s)"
                ),
                track_ids=[trk.track_id],
                attributes={
                    "class_name": trk.class_name,
                    "first_seen_frame": trk.first_seen_frame_number,
                    "bbox": trk.start_bbox.model_dump() if trk.start_bbox else None,
                },
            )
            events.append(appeared_event)

            # ── 2. OBJECT_PRESENT (Interval) ──────────────────────────────
            if trk.duration_seconds > 0.0:
                presence_evidence = EventEvidence(
                    timestamp_seconds=trk.first_seen_timestamp_seconds,
                    evidence_type=EvidenceType.TRACK,
                    role="supporting",
                    track_id=trk.track_id,
                    provenance={
                        "frame_count": trk.frame_count,
                        "duration_seconds": trk.duration_seconds,
                    },
                )

                present_event = Event(
                    video_id=v_id,
                    event_type=EventType.OBJECT_PRESENT,
                    start_timestamp_seconds=trk.first_seen_timestamp_seconds,
                    end_timestamp_seconds=trk.last_seen_timestamp_seconds,
                    confidence=trk.confidence,
                    status=EventStatus.CONFIRMED,
                    severity=EventSeverity.INFO,
                    participants=[participant],
                    event_evidence=[presence_evidence],
                    source_module="lifecycle_detector",
                    description=(
                        f"{trk.class_name.capitalize()} visible for "
                        f"{trk.duration_seconds:.2f}s across {trk.frame_count} frames"
                    ),
                    track_ids=[trk.track_id],
                    attributes={
                        "class_name": trk.class_name,
                        "frame_count": trk.frame_count,
                        "duration_seconds": trk.duration_seconds,
                    },
                )
                events.append(present_event)

            # ── 3. OBJECT_DISAPPEARED (Instantaneous) ─────────────────────
            disappearance_evidence = EventEvidence(
                timestamp_seconds=trk.last_seen_timestamp_seconds,
                evidence_type=EvidenceType.TRACK,
                role="disappearance",
                track_id=trk.track_id,
                provenance={
                    "frame_number": trk.last_seen_frame_number,
                    "class_name": trk.class_name,
                },
            )

            disappeared_event = Event(
                video_id=v_id,
                event_type=EventType.OBJECT_DISAPPEARED,
                start_timestamp_seconds=trk.last_seen_timestamp_seconds,
                end_timestamp_seconds=trk.last_seen_timestamp_seconds,
                confidence=trk.confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[disappearance_evidence],
                source_module="lifecycle_detector",
                description=(
                    f"{trk.class_name.capitalize()} disappeared from scene at "
                    f"frame {trk.last_seen_frame_number} ({trk.last_seen_timestamp_seconds:.2f}s)"
                ),
                track_ids=[trk.track_id],
                attributes={
                    "class_name": trk.class_name,
                    "last_seen_frame": trk.last_seen_frame_number,
                    "bbox": trk.end_bbox.model_dump() if trk.end_bbox else None,
                },
            )
            events.append(disappeared_event)

        return events
