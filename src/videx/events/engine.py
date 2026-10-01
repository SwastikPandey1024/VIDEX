"""Temporal Event Intelligence Engine coordinating multimodal event detection.

Consumes perception, OCR, and audio evidence outputs to construct a chronologically
ordered, evidence-backed EventTimeline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from videx.audio.pipeline import AudioPipelineResult
from videx.domain.schemas import Evidence, Track, TrajectoryPoint
from videx.events.audio import AudioEventDetector
from videx.events.evidence import EvidenceLinker
from videx.events.lifecycle import LifecycleEventDetector
from videx.events.movement import MovementEventDetector
from videx.events.ocr import OCREventDetector
from videx.events.schemas import Event, EventEngineConfig
from videx.events.spatial import SpatialEventDetector, SpatialZone
from videx.events.temporal import TemporalRelationEngine
from videx.events.types import EventType, TemporalRelation
from videx.ocr.pipeline import OCRPipelineResult
from videx.perception.pipeline import PerceptionResult

logger = logging.getLogger(__name__)


class EventTimeline:
    """A chronologically ordered, searchable collection of evidence-backed events."""

    def __init__(self, events: list[Event], video_id: UUID | None = None) -> None:
        self.video_id = video_id or (events[0].video_id if events else UUID(int=0))
        # Deterministic sorting: start_ts, end_ts, event_type, event_id
        self._events: list[Event] = sorted(
            events,
            key=lambda e: (
                e.start_timestamp_seconds,
                e.end_timestamp,
                e.event_type.value,
                str(e.event_id),
            ),
        )
        self._event_map: dict[UUID, Event] = {e.event_id: e for e in self._events}
        self._evidence: list[Evidence] = EvidenceLinker.compile_all_evidence(self._events)
        self._temporal_engine = TemporalRelationEngine()

    @property
    def events(self) -> list[Event]:
        """Return all sorted events."""
        return list(self._events)

    @property
    def evidence_records(self) -> list[Evidence]:
        """Return canonical Evidence records corresponding to all events."""
        return list(self._evidence)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Any:  # noqa: ANN401
        return iter(self._events)

    def events_between(self, start_seconds: float, end_seconds: float) -> list[Event]:
        """Query events active within a specified time interval [start, end].

        An event is included if its active interval intersects [start_seconds, end_seconds].
        """
        results: list[Event] = []
        for e in self._events:
            ev_start = e.start_timestamp_seconds
            ev_end = e.end_timestamp
            if ev_start <= end_seconds and ev_end >= start_seconds:
                results.append(e)
        return results

    def filter_by_type(self, event_type: EventType | Sequence[EventType]) -> list[Event]:
        """Filter events by one or more event types."""
        types_set = {event_type} if isinstance(event_type, EventType) else set(event_type)
        return [e for e in self._events if e.event_type in types_set]

    def filter_by_participant(self, participant_id: UUID | str) -> list[Event]:
        """Find all events in which a specific participant was involved."""
        target_str = str(participant_id)
        return [
            e
            for e in self._events
            if any(str(p.participant_id) == target_str for p in e.participants)
        ]

    def filter_by_track(self, track_id: UUID) -> list[Event]:
        """Find all events referencing a specific Track ID."""
        return [e for e in self._events if track_id in e.track_ids]

    def explain_event(self, event_id: UUID) -> dict[str, Any]:
        """Retrieve the 6-part evidence explanation for an event by UUID."""
        ev = self._event_map.get(event_id)
        if not ev:
            raise KeyError(f"Event {event_id} not found in timeline")
        return EvidenceLinker.explain_event(ev)

    def find_relationships(
        self,
        target_event: Event,
        filter_relations: Sequence[TemporalRelation] | None = None,
    ) -> list[tuple[Event, list[TemporalRelation]]]:
        """Find temporal relationships between target event and all other events in timeline."""
        return self._temporal_engine.find_related_events(
            target_event,
            self._events,
            filter_relations=filter_relations,
        )

    def summary(self) -> dict[str, Any]:
        """Summarize timeline statistics."""
        type_counts: dict[str, int] = {}
        for e in self._events:
            key = e.event_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        first_ts = self._events[0].start_timestamp_seconds if self._events else 0.0
        last_ts = max((e.end_timestamp for e in self._events), default=0.0)

        return {
            "video_id": str(self.video_id),
            "total_events": len(self._events),
            "total_evidence_records": len(self._evidence),
            "time_span_seconds": round(max(0.0, last_ts - first_ts), 2),
            "first_event_timestamp": round(first_ts, 2),
            "last_event_timestamp": round(last_ts, 2),
            "event_type_distribution": type_counts,
        }


class EventEngine:
    """The central orchestrator for the VIDEX Temporal Event Intelligence Engine.

    Coordinates:
    - LifecycleEventDetector
    - MovementEventDetector
    - SpatialEventDetector
    - OCREventDetector
    - AudioEventDetector
    - TemporalRelationEngine
    - EvidenceLinker
    """

    def __init__(
        self,
        config: EventEngineConfig | None = None,
        zones: Sequence[SpatialZone] | None = None,
    ) -> None:
        self.config = config or EventEngineConfig()
        self.lifecycle_detector = LifecycleEventDetector(self.config)
        self.movement_detector = MovementEventDetector(self.config)
        self.spatial_detector = SpatialEventDetector(zones=zones, config=self.config)
        self.ocr_detector = OCREventDetector(self.config)
        self.audio_detector = AudioEventDetector(self.config)
        self.temporal_engine = TemporalRelationEngine(self.config.temporal_near_interval_seconds)

    def add_zone(self, zone: SpatialZone) -> None:
        """Register a spatial zone for boundary tracking."""
        self.spatial_detector.add_zone(zone)

    def process_multimodal(
        self,
        video_id: UUID,
        perception_result: PerceptionResult | None = None,
        ocr_result: OCRPipelineResult | None = None,
        audio_result: AudioPipelineResult | None = None,
        zones: Sequence[SpatialZone] | None = None,
    ) -> EventTimeline:
        """Process outputs from perception, OCR, and audio pipelines into an EventTimeline.

        Args:
            video_id: Unique video identifier.
            perception_result: Accumulated tracks, detections, and trajectories.
            ocr_result: Fused OCR text observations and evidence records.
            audio_result: Audio transcript segments and evidence records.
            zones: Optional runtime spatial zones.

        Returns:
            An EventTimeline populated with sorted, grounded, evidence-backed events.
        """
        all_events: list[Event] = []

        if zones:
            for z in zones:
                self.spatial_detector.add_zone(z)

        # ── 1. Perception: Lifecycle & Movement Events ────────────────────
        if perception_result:
            p_events = self.process_perception(
                tracks=perception_result.tracks,
                trajectories=perception_result.trajectories,
                video_id=video_id,
            )
            all_events.extend(p_events)

        # ── 2. OCR Text Events ────────────────────────────────────────────
        if ocr_result:
            ocr_events = self.ocr_detector.detect_events(
                fused_observations=ocr_result.fused_observations,
                video_id=video_id,
            )
            all_events.extend(ocr_events)

        # ── 3. Audio Speech Events ────────────────────────────────────────
        if audio_result:
            audio_events = self.audio_detector.detect_events(
                transcript_segments=audio_result.fused_segments or audio_result.raw_segments,
                video_id=video_id,
            )
            all_events.extend(audio_events)

        return EventTimeline(events=all_events, video_id=video_id)

    def process_perception(
        self,
        tracks: Sequence[Track],
        trajectories: dict[UUID, list[TrajectoryPoint]],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Run lifecycle, movement, and spatial detectors over tracking and trajectories."""
        events: list[Event] = []
        events.extend(
            self.lifecycle_detector.detect_events(
                tracks=tracks,
                trajectories=trajectories,
                video_id=video_id,
            )
        )
        events.extend(
            self.movement_detector.detect_events(
                tracks=tracks,
                trajectories=trajectories,
                video_id=video_id,
            )
        )
        events.extend(
            self.spatial_detector.detect_events(
                tracks=tracks,
                trajectories=trajectories,
                video_id=video_id,
            )
        )
        return events
