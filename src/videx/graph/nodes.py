"""GraphNode definitions and typed factory constructors for canonical domain entities."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from videx.domain.schemas import (
    Detection,
    Evidence,
    Frame,
    OCRObservation,
    Scene,
    Track,
    TranscriptSegment,
    Video,
)
from videx.events.schemas import Event
from videx.events.spatial import SpatialZone
from videx.graph.provenance import NodeProvenance
from videx.graph.types import GraphNodeType
from videx.semantic.schemas import SemanticEventPayload


def make_node_id(node_type: GraphNodeType, source_id: str | UUID) -> str:
    """Format canonical deterministic node identifier."""
    return f"{node_type.value}:{source_id}"


class GraphNode(BaseModel):
    """A first-class node in the VIDEX Evidence Graph."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(..., description="Deterministic global node ID (e.g. 'event:<uuid>')")
    node_type: GraphNodeType = Field(..., description="Categorization of this node")
    label: str = Field(..., description="Human-readable label or description")
    provenance: NodeProvenance = Field(..., description="Immutable audit and source provenance")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed domain attributes (BBoxes, text, class names, scores)",
    )

    @property
    def video_id(self) -> str:
        return self.provenance.video_id

    @property
    def source_id(self) -> str:
        return self.provenance.source_id

    @property
    def timestamp_start(self) -> float | None:
        return self.provenance.timestamp_start

    @property
    def timestamp_end(self) -> float | None:
        return self.provenance.timestamp_end


# ── Canonical Node Factories ────────────────────────────────────────────────


def create_video_node(video: Video) -> GraphNode:
    """Build a GraphNode from a canonical Video record."""
    source_id = str(video.video_id)
    return GraphNode(
        node_id=make_node_id(GraphNodeType.VIDEO, source_id),
        node_type=GraphNodeType.VIDEO,
        label=f"Video:{video.source_path}",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Video",
            video_id=source_id,
            timestamp_start=0.0,
            timestamp_end=video.duration_seconds,
            metadata={
                "fps": video.fps,
                "width": video.width,
                "height": video.height,
                "codec": video.video_codec,
            },
        ),
        attributes={
            "source_path": video.source_path,
            "duration_seconds": video.duration_seconds,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "total_frames": video.total_frames,
        },
    )


def create_scene_node(scene: Scene, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a Scene record."""
    source_id = str(scene.scene_id)
    t_start = scene.start_timestamp_seconds
    t_end = scene.end_timestamp_seconds
    return GraphNode(
        node_id=make_node_id(GraphNodeType.SCENE, source_id),
        node_type=GraphNodeType.SCENE,
        label=f"Scene {scene.scene_index} [{t_start:.2f}s - {t_end:.2f}s]",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Scene",
            video_id=video_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            metadata={"scene_index": scene.scene_index},
        ),
        attributes={
            "scene_index": scene.scene_index,
            "start_timestamp_seconds": t_start,
            "end_timestamp_seconds": t_end,
            "start_frame_number": scene.start_frame_number,
            "end_frame_number": scene.end_frame_number,
        },
    )


def create_frame_node(frame: Frame, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a Frame record."""
    source_id = str(frame.frame_id)
    return GraphNode(
        node_id=make_node_id(GraphNodeType.FRAME, source_id),
        node_type=GraphNodeType.FRAME,
        label=f"Frame {frame.frame_number} ({frame.timestamp_seconds:.3f}s)",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Frame",
            video_id=video_id,
            timestamp_start=frame.timestamp_seconds,
            timestamp_end=frame.timestamp_seconds,
            metadata={"frame_number": frame.frame_number},
        ),
        attributes={
            "frame_number": frame.frame_number,
            "timestamp_seconds": frame.timestamp_seconds,
            "width": frame.width,
            "height": frame.height,
            "frame_data_path": frame.frame_data_path,
        },
    )


def create_detection_node(detection: Detection, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a Detection record."""
    source_id = str(detection.detection_id)
    ts = detection.timestamp_seconds
    return GraphNode(
        node_id=make_node_id(GraphNodeType.DETECTION, source_id),
        node_type=GraphNodeType.DETECTION,
        label=f"{detection.class_name} ({detection.confidence:.2f}) @ {ts:.2f}s",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Detection",
            video_id=video_id,
            timestamp_start=ts,
            timestamp_end=ts,
            metadata={
                "frame_id": str(detection.frame_id),
                "class_name": detection.class_name,
                "provider": detection.provider,
            },
        ),
        attributes={
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "bbox": detection.bbox.model_dump(),
            "frame_id": str(detection.frame_id),
            "provider": detection.provider,
            "frame_number": detection.frame_number,
        },
    )


def create_track_node(track: Track, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a Track record."""
    source_id = str(track.track_id)
    t_start = track.first_seen_timestamp_seconds
    t_end = track.last_seen_timestamp_seconds
    return GraphNode(
        node_id=make_node_id(GraphNodeType.TRACK, source_id),
        node_type=GraphNodeType.TRACK,
        label=f"Track:{track.class_name}#{track.track_id} [{t_start:.2f}s - {t_end:.2f}s]",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Track",
            video_id=video_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            metadata={"class_name": track.class_name, "provider": track.provider},
        ),
        attributes={
            "class_name": track.class_name,
            "confidence": track.confidence,
            "first_seen_timestamp_seconds": t_start,
            "last_seen_timestamp_seconds": t_end,
            "first_seen_frame_number": track.first_seen_frame_number,
            "last_seen_frame_number": track.last_seen_frame_number,
            "provider": track.provider,
            "status": track.status.value,
            "detections_count": len(track.detection_ids),
        },
    )


def create_ocr_node(ocr: OCRObservation, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from an OCRObservation record."""
    source_id = str(ocr.observation_id)
    return GraphNode(
        node_id=make_node_id(GraphNodeType.OCR_OBSERVATION, source_id),
        node_type=GraphNodeType.OCR_OBSERVATION,
        label=f"OCR:'{ocr.text}' ({ocr.confidence:.2f})",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="OCRObservation",
            video_id=video_id,
            timestamp_start=ocr.timestamp_seconds,
            timestamp_end=ocr.timestamp_seconds,
            metadata={"text": ocr.text, "frame_id": str(ocr.frame_id)},
        ),
        attributes={
            "text": ocr.text,
            "normalized_text": ocr.normalized_text,
            "confidence": ocr.confidence,
            "timestamp_seconds": ocr.timestamp_seconds,
            "bbox": ocr.bbox.model_dump() if ocr.bbox else None,
            "frame_id": str(ocr.frame_id),
            "frame_number": ocr.frame_number,
        },
    )


def create_transcript_node(seg: TranscriptSegment, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a TranscriptSegment record."""
    source_id = str(seg.segment_id)
    t_start = seg.start_timestamp_seconds
    t_end = seg.end_timestamp_seconds
    snippet = seg.text[:25]
    return GraphNode(
        node_id=make_node_id(GraphNodeType.TRANSCRIPT_SEGMENT, source_id),
        node_type=GraphNodeType.TRANSCRIPT_SEGMENT,
        label=f"ASR:'{snippet}' [{t_start:.2f}s - {t_end:.2f}s]",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="TranscriptSegment",
            video_id=video_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            metadata={
                "language": seg.language,
                "speaker": seg.speaker_id,
                "provider": seg.provider,
            },
        ),
        attributes={
            "text": seg.text,
            "raw_text": seg.raw_text,
            "normalized_text": seg.normalized_text,
            "language": seg.language,
            "confidence": seg.confidence,
            "start_timestamp_seconds": t_start,
            "end_timestamp_seconds": t_end,
            "speaker_id": seg.speaker_id,
            "provider": seg.provider,
        },
    )


def create_zone_node(zone: SpatialZone, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a SpatialZone configuration."""
    source_id = str(zone.zone_id)
    return GraphNode(
        node_id=make_node_id(GraphNodeType.ZONE, source_id),
        node_type=GraphNodeType.ZONE,
        label=f"Zone:{zone.zone_name} ({zone.zone_id})",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="SpatialZone",
            video_id=video_id,
            metadata={"zone_name": zone.zone_name},
        ),
        attributes={
            "zone_id": zone.zone_id,
            "zone_name": zone.zone_name,
            "polygon_points": zone.polygon_points,
        },
    )


def create_event_node(event: Event, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a deterministic Event record."""
    source_id = str(event.event_id)
    ev_ids = [str(e) for e in event.evidence_ids]
    t_start = event.start_timestamp
    t_end = event.end_timestamp
    return GraphNode(
        node_id=make_node_id(GraphNodeType.EVENT, source_id),
        node_type=GraphNodeType.EVENT,
        label=f"Event:{event.event_type.value} [{t_start:.2f}s - {t_end:.2f}s]",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Event",
            video_id=video_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            evidence_ids=ev_ids,
            metadata={"event_type": event.event_type.value, "severity": event.severity.value},
        ),
        attributes={
            "event_type": event.event_type.value,
            "description": event.description,
            "confidence": event.confidence,
            "severity": event.severity.value,
            "status": event.status.value,
            "start_timestamp": t_start,
            "end_timestamp": t_end,
            "participants_count": len(event.participants),
            "evidence_count": len(event.evidence_ids),
        },
    )


def create_semantic_event_node(
    sem: SemanticEventPayload,
    video_id: str | UUID,
    semantic_event_id: str | UUID | None = None,
    candidate_id: str | UUID | None = None,
) -> GraphNode:
    """Build a GraphNode from a Layer 5 SemanticEventPayload record."""
    source_id = str(semantic_event_id or uuid4())
    ev_ids = [str(eid) for eid in sem.evidence_ids]
    t_start = sem.start_timestamp_seconds
    t_end = sem.end_timestamp_seconds
    return GraphNode(
        node_id=make_node_id(GraphNodeType.SEMANTIC_EVENT, source_id),
        node_type=GraphNodeType.SEMANTIC_EVENT,
        label=f"Semantic:{sem.semantic_event_type.value} [{t_start:.2f}s - {t_end:.2f}s]",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="SemanticEventPayload",
            video_id=video_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            evidence_ids=ev_ids,
            metadata={
                "claim": sem.claim,
                "status": sem.status.value,
            },
        ),
        attributes={
            "semantic_event_type": sem.semantic_event_type.value,
            "claim": sem.claim,
            "description": sem.description,
            "confidence": sem.confidence,
            "uncertainty": sem.uncertainty,
            "status": sem.status.value,
            "start_timestamp_seconds": t_start,
            "end_timestamp_seconds": t_end,
            "candidate_id": str(candidate_id) if candidate_id else None,
            "supporting_event_ids": [str(eid) for eid in sem.supporting_event_ids],
        },
    )


def create_evidence_node(evidence: Evidence, video_id: str | UUID) -> GraphNode:
    """Build a GraphNode from a canonical Evidence record."""
    source_id = str(evidence.evidence_id)
    return GraphNode(
        node_id=make_node_id(GraphNodeType.EVIDENCE, source_id),
        node_type=GraphNodeType.EVIDENCE,
        label=f"Evidence:{evidence.evidence_type.value} @ {evidence.timestamp_seconds or 0.0:.2f}s",
        provenance=NodeProvenance.create(
            source_id=source_id,
            source_type="Evidence",
            video_id=video_id,
            timestamp_start=evidence.timestamp_seconds,
            timestamp_end=evidence.timestamp_seconds,
            metadata={
                "evidence_type": evidence.evidence_type.value,
                "source_module": evidence.source_module,
            },
        ),
        attributes={
            "evidence_type": evidence.evidence_type.value,
            "timestamp_seconds": evidence.timestamp_seconds,
            "confidence": evidence.confidence,
            "source_module": evidence.source_module,
            "frame_id": str(evidence.frame_id) if evidence.frame_id else None,
            "track_id": str(evidence.track_id) if evidence.track_id else None,
        },
    )
