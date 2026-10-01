"""Deterministic compound event rules correlating cross-modal observations.

Architectural Boundary (ADR-005 - Layer 3):
This engine evaluates deterministic spatiotemporal relationships (e.g. topological containment
and temporal co-occurrence) across multiple modalities. It emits candidate correlation events
with strict evidence provenance.

IMPORTANT: This engine must NOT be used for high-level semantic interpretation, subjective
intent classification, or heuristic narrative guessing (e.g. 'theft', 'altercation').
Such semantic reasoning belongs strictly in Layer 5 (Phase 6 VLM / Semantic Router).
"""

from __future__ import annotations

from uuid import UUID

from videx.domain.schemas import EvidenceType
from videx.events.schemas import Event, EventEvidence, EventParticipant
from videx.events.temporal import TemporalRelationEngine
from videx.events.types import EventSeverity, EventStatus, EventType


class CrossModalRuleEngine:
    """Evaluates declarative spatiotemporal co-occurrence rules to produce deterministic compound relations."""

    def __init__(self, temporal_tolerance_seconds: float = 2.0) -> None:
        self.temporal_tolerance_seconds = temporal_tolerance_seconds
        self.temporal_engine = TemporalRelationEngine(temporal_tolerance_seconds)

    def detect_speech_during_zone_presence(
        self,
        events: list[Event],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Detect when speech occurs while an object is inside a spatial zone."""
        compound_events: list[Event] = []

        zone_events = [
            e
            for e in events
            if e.event_type in (EventType.OBJECT_ENTERED_ZONE, EventType.OBJECT_IN_ZONE)
        ]
        speech_events = [e for e in events if e.event_type == EventType.SPEECH_DETECTED]

        for z_ev in zone_events:
            for s_ev in speech_events:
                # Check if speech occurs near or during zone event
                if self.temporal_engine.is_overlaps(
                    z_ev, s_ev
                ) or self.temporal_engine.is_near_in_time(
                    z_ev, s_ev, max_delta=self.temporal_tolerance_seconds
                ):
                    v_id = video_id or z_ev.video_id
                    start_ts = min(z_ev.start_timestamp_seconds, s_ev.start_timestamp_seconds)
                    end_ts = max(z_ev.end_timestamp, s_ev.end_timestamp)

                    # Compound evidence linking both supporting events
                    ev1 = EventEvidence(
                        timestamp_seconds=z_ev.start_timestamp_seconds,
                        evidence_type=EvidenceType.EVENT,
                        role="context",
                        provenance={
                            "sub_event": z_ev.event_type.value,
                            "zone": z_ev.zone_name,
                        },
                    )
                    ev2 = EventEvidence(
                        timestamp_seconds=s_ev.start_timestamp_seconds,
                        evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                        role="trigger",
                        provenance={"transcript": s_ev.attributes.get("raw_text")},
                    )

                    all_participants: list[EventParticipant] = []
                    all_participants.extend(z_ev.participants)
                    for p in s_ev.participants:
                        if not any(
                            str(ep.participant_id) == str(p.participant_id)
                            for ep in all_participants
                        ):
                            all_participants.append(p)

                    speech_text = s_ev.attributes.get("raw_text", "")
                    zone_label = z_ev.zone_name or "zone"

                    compound_ev = Event(
                        video_id=v_id,
                        event_type=EventType.CUSTOM,
                        start_timestamp_seconds=start_ts,
                        end_timestamp_seconds=end_ts,
                        confidence=round(min(z_ev.confidence, s_ev.confidence), 3),
                        status=EventStatus.CONFIRMED,
                        severity=EventSeverity.MEDIUM,
                        participants=all_participants,
                        event_evidence=[ev1, ev2],
                        source_module="cross_modal_rule_engine",
                        description=(
                            f"Speech utterance ('{speech_text}') co-occurred with "
                            f"presence in '{zone_label}' at {start_ts:.2f}s - {end_ts:.2f}s"
                        ),
                        zone_name=z_ev.zone_name,
                        attributes={
                            "rule_name": "speech_in_zone",
                            "speech_text": speech_text,
                            "zone_name": zone_label,
                        },
                    )
                    compound_events.append(compound_ev)

        return compound_events
