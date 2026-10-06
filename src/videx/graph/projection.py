"""Deterministic spatiotemporal Evidence Graph projection engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
from videx.graph.edges import create_edge
from videx.graph.nodes import (
    create_detection_node,
    create_event_node,
    create_evidence_node,
    create_frame_node,
    create_ocr_node,
    create_scene_node,
    create_semantic_event_node,
    create_track_node,
    create_transcript_node,
    create_video_node,
    create_zone_node,
    make_node_id,
)
from videx.graph.spatial import SpatialGraphEngine
from videx.graph.store import GraphStore, InMemoryGraphStore
from videx.graph.temporal import TemporalGraphEngine
from videx.graph.types import DerivationType, GraphEdgeType, GraphNodeType
from videx.semantic.schemas import SemanticEventPayload


@dataclass(frozen=True)
class GraphProjectionConfig:
    """Configuration governing deterministic graph projection."""

    near_threshold_seconds: float = 2.0
    iou_overlap_threshold: float = 0.1
    enable_temporal_relations: bool = True
    enable_spatial_relations: bool = True
    enable_zone_containment: bool = True


class GraphProjection:
    """Transforms canonical multimodal pipeline outputs into a typed spatiotemporal graph."""

    def __init__(self, config: GraphProjectionConfig | None = None) -> None:
        self.config = config or GraphProjectionConfig()

    def project(
        self,
        video: Video,
        scenes: Sequence[Scene] | None = None,
        frames: Sequence[Frame] | None = None,
        detections: Sequence[Detection] | None = None,
        tracks: Sequence[Track] | None = None,
        ocr_observations: Sequence[OCRObservation] | None = None,
        transcript_segments: Sequence[TranscriptSegment] | None = None,
        zones: Sequence[SpatialZone] | None = None,
        events: Sequence[Event] | None = None,
        semantic_events: Sequence[SemanticEventPayload] | None = None,
        evidence: Sequence[Evidence] | None = None,
        store: GraphStore | None = None,
    ) -> GraphStore:
        """Execute deterministic projection from canonical inputs into the graph store."""
        target_store = store if store is not None else InMemoryGraphStore()
        vid = str(video.video_id)

        # 1. Video Root Node
        video_node = create_video_node(video)
        target_store.upsert_node(video_node)

        # 2. Scene Nodes & Video -> Scene Containment
        scene_nodes: list[str] = []
        if scenes:
            for sc in sorted(scenes, key=lambda s: (s.scene_index, s.start_timestamp_seconds)):
                sc_node = create_scene_node(sc, vid)
                target_store.upsert_node(sc_node)
                scene_nodes.append(sc_node.node_id)
                target_store.upsert_edge(
                    create_edge(
                        source_node_id=video_node.node_id,
                        target_node_id=sc_node.node_id,
                        relationship=GraphEdgeType.CONTAINS,
                        derivation=DerivationType.STRUCTURAL,
                    )
                )

        # 3. Spatial Zone Nodes & Video -> Zone Containment
        zone_map: dict[str, str] = {}
        if zones:
            for z in sorted(zones, key=lambda item: item.zone_id):
                z_node = create_zone_node(z, vid)
                target_store.upsert_node(z_node)
                zone_map[z.zone_id] = z_node.node_id
                target_store.upsert_edge(
                    create_edge(
                        source_node_id=video_node.node_id,
                        target_node_id=z_node.node_id,
                        relationship=GraphEdgeType.CONTAINS,
                        derivation=DerivationType.STRUCTURAL,
                    )
                )

        # 4. Frame Nodes & Scene -> Frame Containment
        frame_node_map: dict[str, str] = {}
        if frames:
            for fr in sorted(frames, key=lambda f: (f.frame_number, f.timestamp_seconds)):
                f_node = create_frame_node(fr, vid)
                target_store.upsert_node(f_node)
                frame_node_map[str(fr.frame_id)] = f_node.node_id

                # Link to scene if frame is attributed to a scene, else link to video
                if fr.scene_id:
                    sc_nid = make_node_id(GraphNodeType.SCENE, fr.scene_id)
                    target_store.upsert_edge(
                        create_edge(
                            source_node_id=sc_nid,
                            target_node_id=f_node.node_id,
                            relationship=GraphEdgeType.CONTAINS,
                            derivation=DerivationType.STRUCTURAL,
                        )
                    )
                else:
                    target_store.upsert_edge(
                        create_edge(
                            source_node_id=video_node.node_id,
                            target_node_id=f_node.node_id,
                            relationship=GraphEdgeType.CONTAINS,
                            derivation=DerivationType.STRUCTURAL,
                        )
                    )

        # 5. Track Nodes
        track_node_map: dict[str, str] = {}
        if tracks:
            for trk in sorted(
                tracks,
                key=lambda t: (t.first_seen_timestamp_seconds, str(t.track_id)),
            ):
                t_node = create_track_node(trk, vid)
                target_store.upsert_node(t_node)
                track_node_map[str(trk.track_id)] = t_node.node_id
                target_store.upsert_edge(
                    create_edge(
                        source_node_id=video_node.node_id,
                        target_node_id=t_node.node_id,
                        relationship=GraphEdgeType.CONTAINS,
                        derivation=DerivationType.STRUCTURAL,
                    )
                )

        # 6. Detection Nodes & Associations
        if detections:
            sorted_dets = sorted(
                detections,
                key=lambda d: (d.timestamp_seconds, d.frame_number, str(d.detection_id)),
            )
            for det in sorted_dets:
                d_node = create_detection_node(det, vid)
                target_store.upsert_node(d_node)

                # Frame contains detection
                f_nid = frame_node_map.get(str(det.frame_id))
                if f_nid:
                    target_store.upsert_edge(
                        create_edge(
                            source_node_id=f_nid,
                            target_node_id=d_node.node_id,
                            relationship=GraphEdgeType.CONTAINS,
                            derivation=DerivationType.STRUCTURAL,
                        )
                    )

            # Link detection to track if assigned
            if tracks:
                for trk in tracks:
                    t_nid = track_node_map.get(str(trk.track_id))
                    if t_nid:
                        for det_id in trk.detection_ids:
                            d_nid = make_node_id(GraphNodeType.DETECTION, det_id)
                            if target_store.get_node(d_nid) is not None:
                                target_store.upsert_edge(
                                    create_edge(
                                        source_node_id=d_nid,
                                        target_node_id=t_nid,
                                        relationship=GraphEdgeType.BELONGS_TO,
                                        derivation=DerivationType.DETERMINISTIC,
                                    )
                                )

        # 7. OCR Observation Nodes
        if ocr_observations:
            for ocr in sorted(
                ocr_observations,
                key=lambda o: (o.timestamp_seconds, str(o.observation_id)),
            ):
                ocr_node = create_ocr_node(ocr, vid)
                target_store.upsert_node(ocr_node)
                f_nid = frame_node_map.get(str(ocr.frame_id))
                if f_nid:
                    target_store.upsert_edge(
                        create_edge(
                            source_node_id=f_nid,
                            target_node_id=ocr_node.node_id,
                            relationship=GraphEdgeType.CONTAINS,
                            derivation=DerivationType.STRUCTURAL,
                        )
                    )

        # 8. Transcript Segment Nodes
        if transcript_segments:
            for seg in sorted(
                transcript_segments,
                key=lambda s: (s.start_timestamp_seconds, str(s.segment_id)),
            ):
                seg_node = create_transcript_node(seg, vid)
                target_store.upsert_node(seg_node)
                target_store.upsert_edge(
                    create_edge(
                        source_node_id=video_node.node_id,
                        target_node_id=seg_node.node_id,
                        relationship=GraphEdgeType.CONTAINS,
                        derivation=DerivationType.STRUCTURAL,
                    )
                )

        # 9. Deterministic Event Nodes & Participations
        event_node_map: dict[str, str] = {}
        if events:
            sorted_events = sorted(
                events,
                key=lambda e: (e.start_timestamp, e.end_timestamp, str(e.event_id)),
            )
            for ev in sorted_events:
                ev_node = create_event_node(ev, vid)
                target_store.upsert_node(ev_node)
                event_node_map[str(ev.event_id)] = ev_node.node_id

                # Track participants
                for p in ev.participants:
                    if p.participant_type == "track":
                        t_nid = track_node_map.get(str(p.participant_id))
                        if t_nid:
                            target_store.upsert_edge(
                                create_edge(
                                    source_node_id=t_nid,
                                    target_node_id=ev_node.node_id,
                                    relationship=GraphEdgeType.PARTICIPATES_IN,
                                    derivation=DerivationType.DETERMINISTIC,
                                    attributes={"role": p.role, "label": p.label or ""},
                                )
                            )
                    elif p.participant_type == "zone":
                        z_nid = zone_map.get(str(p.participant_id))
                        if z_nid:
                            target_store.upsert_edge(
                                create_edge(
                                    source_node_id=ev_node.node_id,
                                    target_node_id=z_nid,
                                    relationship=GraphEdgeType.OCCURS_IN,
                                    derivation=DerivationType.DETERMINISTIC,
                                )
                            )

        # 10. Semantic Events (Layer 5)
        if semantic_events:
            for sem in sorted(
                semantic_events,
                key=lambda s: (s.start_timestamp_seconds, s.claim),
            ):
                sem_node = create_semantic_event_node(sem, vid)
                target_store.upsert_node(sem_node)

                # Link supporting deterministic events
                for sup_ev_id in sem.supporting_event_ids:
                    ev_nid = event_node_map.get(str(sup_ev_id))
                    if ev_nid:
                        target_store.upsert_edge(
                            create_edge(
                                source_node_id=sem_node.node_id,
                                target_node_id=ev_nid,
                                relationship=GraphEdgeType.DERIVED_FROM,
                                derivation=DerivationType.INFERRED,
                                reason=f"VLM reasoning claim: {sem.claim}",
                            )
                        )
                        target_store.upsert_edge(
                            create_edge(
                                source_node_id=ev_nid,
                                target_node_id=sem_node.node_id,
                                relationship=GraphEdgeType.SUPPORTS_SEMANTIC_EVENT,
                                derivation=DerivationType.DETERMINISTIC,
                            )
                        )

        # 11. Canonical Evidence Nodes
        if evidence:
            for ev_rec in sorted(
                evidence,
                key=lambda item: (item.timestamp_seconds or 0.0, str(item.evidence_id)),
            ):
                ev_node = create_evidence_node(ev_rec, vid)
                target_store.upsert_node(ev_node)

                # Link to frame if anchored
                if ev_rec.frame_id:
                    f_nid = frame_node_map.get(str(ev_rec.frame_id))
                    if f_nid:
                        target_store.upsert_edge(
                            create_edge(
                                source_node_id=ev_node.node_id,
                                target_node_id=f_nid,
                                relationship=GraphEdgeType.REFERENCES,
                                derivation=DerivationType.STRUCTURAL,
                            )
                        )

                # Link to track if anchored
                if ev_rec.track_id:
                    t_nid = track_node_map.get(str(ev_rec.track_id))
                    if t_nid:
                        target_store.upsert_edge(
                            create_edge(
                                source_node_id=ev_node.node_id,
                                target_node_id=t_nid,
                                relationship=GraphEdgeType.REFERENCES,
                                derivation=DerivationType.STRUCTURAL,
                            )
                        )

        # 12. Link Events to Grounding Evidence
        if events and evidence:
            evidence_node_ids = {
                str(e.evidence_id): make_node_id(GraphNodeType.EVIDENCE, e.evidence_id)
                for e in evidence
            }
            for ev in events:
                ev_nid = event_node_map.get(str(ev.event_id))
                if ev_nid:
                    for eid in ev.evidence_ids:
                        ground_nid = evidence_node_ids.get(str(eid))
                        if ground_nid and target_store.get_node(ground_nid) is not None:
                            target_store.upsert_edge(
                                create_edge(
                                    source_node_id=ev_nid,
                                    target_node_id=ground_nid,
                                    relationship=GraphEdgeType.SUPPORTED_BY,
                                    derivation=DerivationType.STRUCTURAL,
                                )
                            )

        # 13. Spatiotemporal Topology
        if self.config.enable_temporal_relations and events:
            temporal_engine = TemporalGraphEngine(
                near_threshold_seconds=self.config.near_threshold_seconds
            )
            for t_edge in temporal_engine.compute_temporal_edges(events):
                target_store.upsert_edge(t_edge)

        if self.config.enable_spatial_relations and detections:
            spatial_engine = SpatialGraphEngine(
                iou_threshold=self.config.iou_overlap_threshold
            )
            for s_edge in spatial_engine.compute_spatial_overlap_edges(detections):
                target_store.upsert_edge(s_edge)

        if self.config.enable_zone_containment and tracks and zones:
            spatial_engine = SpatialGraphEngine(
                iou_threshold=self.config.iou_overlap_threshold
            )
            for z_edge in spatial_engine.compute_zone_containment_edges(tracks, zones):
                target_store.upsert_edge(z_edge)

        return target_store
