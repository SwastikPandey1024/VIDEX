"""Evidence linking and explanation module for the Temporal Event Intelligence Engine.

Every event is converted into a canonical Evidence record (EvidenceType.EVENT)
and maintains an explainable provenance chain answering:
- WHAT happened?
- WHEN did it happen?
- WHICH object/entity participated?
- WHAT evidence supports it?
- WHICH frame/timestamp supports it?
- WHICH subsystem generated it?
"""

from __future__ import annotations

from typing import Any

from videx.domain.schemas import Evidence, EvidenceType
from videx.events.schemas import Event


class EvidenceLinker:
    """Bridges domain Events to canonical Evidence records and builds audit explanations."""

    @staticmethod
    def event_to_evidence(event: Event) -> Evidence:
        """Convert a detected Event into a canonical pipeline Evidence record.

        Args:
            event: The grounded Event domain model.

        Returns:
            A canonical Evidence record of type EvidenceType.EVENT.
        """
        payload: dict[str, Any] = {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "start_timestamp_seconds": event.start_timestamp_seconds,
            "end_timestamp_seconds": event.end_timestamp_seconds,
            "duration_seconds": event.duration_seconds,
            "is_instantaneous": event.is_instantaneous,
            "status": event.status.value,
            "severity": event.severity.value,
            "participants": [p.model_dump() for p in event.participants],
            "event_evidence": [ee.model_dump() for ee in event.event_evidence],
            "attributes": event.attributes,
            "zone_name": event.zone_name,
        }

        # Resolve primary track_id if available
        primary_track_id = event.track_ids[0] if event.track_ids else None

        return Evidence(
            evidence_type=EvidenceType.EVENT,
            source_module=event.source_module,
            video_id=event.video_id,
            timestamp_seconds=event.start_timestamp_seconds,
            track_id=primary_track_id,
            confidence=event.confidence,
            description=event.description,
            raw_payload=payload,
            supporting_observation_ids=list(event.evidence_ids),
            tags=[
                "event",
                event.event_type.value,
                event.status.value,
                event.severity.value,
            ]
            + ([event.zone_name] if event.zone_name else []),
        )

    @staticmethod
    def explain_event(event: Event) -> dict[str, Any]:
        """Generate a complete, human-readable and structured explanation of an event.

        Answers the 6 fundamental provenance questions:
        1. WHAT happened?
        2. WHEN did it happen?
        3. WHICH object/entity participated?
        4. WHAT evidence supports it?
        5. WHICH frame/timestamp supports it?
        6. WHICH subsystem generated it?
        """
        # Determine temporal string
        if event.is_instantaneous:
            when_str = f"At {event.start_timestamp_seconds:.2f}s (instantaneous)"
        else:
            when_str = (
                f"From {event.start_timestamp_seconds:.2f}s to "
                f"{event.end_timestamp_seconds:.2f}s ({event.duration_seconds:.2f}s duration)"
            )

        # Summarize participants
        participants_summary = [
            {
                "id": str(p.participant_id),
                "type": p.participant_type,
                "role": p.role,
                "label": p.label or "unlabeled",
            }
            for p in event.participants
        ]

        # Summarize evidence records
        evidence_summary = [
            {
                "evidence_id": str(ee.evidence_id),
                "type": ee.evidence_type.value,
                "role": ee.role,
                "timestamp_seconds": ee.timestamp_seconds,
                "track_id": str(ee.track_id) if ee.track_id else None,
                "ocr_observation_id": (
                    str(ee.ocr_observation_id) if ee.ocr_observation_id else None
                ),
                "transcript_segment_id": (
                    str(ee.transcript_segment_id) if ee.transcript_segment_id else None
                ),
                "provenance": ee.provenance,
            }
            for ee in event.event_evidence
        ]

        # Frame anchors
        frames_supported: list[int] = []
        for ee in event.event_evidence:
            if "frame_number" in ee.provenance:
                frames_supported.append(int(ee.provenance["frame_number"]))
            elif "frame" in ee.provenance:
                frames_supported.append(int(ee.provenance["frame"]))

        return {
            "what": {
                "event_type": event.event_type.value,
                "description": event.description,
                "status": event.status.value,
                "severity": event.severity.value,
                "confidence": round(event.confidence, 3),
            },
            "when": {
                "start_seconds": event.start_timestamp_seconds,
                "end_seconds": event.end_timestamp_seconds,
                "duration_seconds": round(event.duration_seconds, 3),
                "is_instantaneous": event.is_instantaneous,
                "summary": when_str,
            },
            "which_entities": participants_summary,
            "supporting_evidence": {
                "total_records": len(event.event_evidence),
                "evidence_ids": [str(eid) for eid in event.evidence_ids],
                "records": evidence_summary,
            },
            "temporal_anchors": {
                "start_seconds": event.start_timestamp_seconds,
                "end_seconds": event.end_timestamp,
                "supporting_frames": sorted(set(frames_supported)),
            },
            "source_subsystem": {
                "source_module": event.source_module,
                "attributes": event.attributes,
                "metadata": event.metadata,
            },
        }

    @classmethod
    def compile_all_evidence(cls, events: list[Event]) -> list[Evidence]:
        """Convert a list of events into a list of canonical Evidence records."""
        return [cls.event_to_evidence(ev) for ev in events]
