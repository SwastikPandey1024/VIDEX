"""Mock data generator for VIDEX Evidence Graph testing and prototyping."""

from __future__ import annotations

from uuid import UUID, uuid4

from videx.graph.edges import create_edge
from videx.graph.nodes import (
    GraphNode,
    make_node_id,
)
from videx.graph.provenance import NodeProvenance
from videx.graph.store import InMemoryGraphStore
from videx.graph.types import DerivationType, GraphEdgeType, GraphNodeType


def create_mock_graph_store(video_id: str | UUID | None = None) -> InMemoryGraphStore:
    """Construct an InMemoryGraphStore populated with deterministic, verified test data."""
    store = InMemoryGraphStore()
    vid = str(video_id or uuid4())

    # 1. Video Node
    video_node_id = make_node_id(GraphNodeType.VIDEO, vid)
    store.upsert_node(
        GraphNode(
            node_id=video_node_id,
            node_type=GraphNodeType.VIDEO,
            label="Video:mock_surveillance.mp4",
            provenance=NodeProvenance.create(
                source_id=vid,
                source_type="Video",
                video_id=vid,
                timestamp_start=0.0,
                timestamp_end=30.0,
            ),
            attributes={"fps": 25.0, "width": 1280, "height": 720},
        )
    )

    # 2. Scene Nodes
    sc1_id = str(uuid4())
    sc1_node = make_node_id(GraphNodeType.SCENE, sc1_id)
    store.upsert_node(
        GraphNode(
            node_id=sc1_node,
            node_type=GraphNodeType.SCENE,
            label="Scene 0 [0.0s - 15.0s]",
            provenance=NodeProvenance.create(
                source_id=sc1_id,
                source_type="Scene",
                video_id=vid,
                timestamp_start=0.0,
                timestamp_end=15.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=video_node_id,
            target_node_id=sc1_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    sc2_id = str(uuid4())
    sc2_node = make_node_id(GraphNodeType.SCENE, sc2_id)
    store.upsert_node(
        GraphNode(
            node_id=sc2_node,
            node_type=GraphNodeType.SCENE,
            label="Scene 1 [15.0s - 30.0s]",
            provenance=NodeProvenance.create(
                source_id=sc2_id,
                source_type="Scene",
                video_id=vid,
                timestamp_start=15.0,
                timestamp_end=30.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=video_node_id,
            target_node_id=sc2_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    # 3. Spatial Zone Node
    zone_id = "zone_loading_dock"
    zone_node = make_node_id(GraphNodeType.ZONE, zone_id)
    store.upsert_node(
        GraphNode(
            node_id=zone_node,
            node_type=GraphNodeType.ZONE,
            label="Zone:Loading Dock",
            provenance=NodeProvenance.create(
                source_id=zone_id,
                source_type="SpatialZone",
                video_id=vid,
            ),
            attributes={"polygon_points": [(100, 100), (500, 100), (500, 500), (100, 500)]},
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=video_node_id,
            target_node_id=zone_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    # 4. Frames
    f1_id = str(uuid4())
    f1_node = make_node_id(GraphNodeType.FRAME, f1_id)
    store.upsert_node(
        GraphNode(
            node_id=f1_node,
            node_type=GraphNodeType.FRAME,
            label="Frame 25 @ 1.00s",
            provenance=NodeProvenance.create(
                source_id=f1_id,
                source_type="Frame",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=1.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=sc1_node,
            target_node_id=f1_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    f2_id = str(uuid4())
    f2_node = make_node_id(GraphNodeType.FRAME, f2_id)
    store.upsert_node(
        GraphNode(
            node_id=f2_node,
            node_type=GraphNodeType.FRAME,
            label="Frame 50 @ 2.00s",
            provenance=NodeProvenance.create(
                source_id=f2_id,
                source_type="Frame",
                video_id=vid,
                timestamp_start=2.0,
                timestamp_end=2.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=sc1_node,
            target_node_id=f2_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    # 5. Tracks
    trk1_id = str(uuid4())
    trk1_node = make_node_id(GraphNodeType.TRACK, trk1_id)
    store.upsert_node(
        GraphNode(
            node_id=trk1_node,
            node_type=GraphNodeType.TRACK,
            label="Track:person#1 [0.5s - 5.0s]",
            provenance=NodeProvenance.create(
                source_id=trk1_id,
                source_type="Track",
                video_id=vid,
                timestamp_start=0.5,
                timestamp_end=5.0,
            ),
            attributes={"class_name": "person", "confidence": 0.88},
        )
    )

    trk2_id = str(uuid4())
    trk2_node = make_node_id(GraphNodeType.TRACK, trk2_id)
    store.upsert_node(
        GraphNode(
            node_id=trk2_node,
            node_type=GraphNodeType.TRACK,
            label="Track:car#2 [1.0s - 8.0s]",
            provenance=NodeProvenance.create(
                source_id=trk2_id,
                source_type="Track",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=8.0,
            ),
            attributes={"class_name": "car", "confidence": 0.94},
        )
    )

    # Track in Zone
    store.upsert_edge(
        create_edge(
            source_node_id=trk1_node,
            target_node_id=zone_node,
            relationship=GraphEdgeType.OCCURS_IN,
            derivation=DerivationType.DETERMINISTIC,
            reason="Track centroid located inside loading dock polygon",
        )
    )

    # 6. Detections
    d1_id = str(uuid4())
    d1_node = make_node_id(GraphNodeType.DETECTION, d1_id)
    store.upsert_node(
        GraphNode(
            node_id=d1_node,
            node_type=GraphNodeType.DETECTION,
            label="Detection:person @ 1.00s",
            provenance=NodeProvenance.create(
                source_id=d1_id,
                source_type="Detection",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=1.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=f1_node,
            target_node_id=d1_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=d1_node,
            target_node_id=trk1_node,
            relationship=GraphEdgeType.BELONGS_TO,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    d2_id = str(uuid4())
    d2_node = make_node_id(GraphNodeType.DETECTION, d2_id)
    store.upsert_node(
        GraphNode(
            node_id=d2_node,
            node_type=GraphNodeType.DETECTION,
            label="Detection:car @ 1.00s",
            provenance=NodeProvenance.create(
                source_id=d2_id,
                source_type="Detection",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=1.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=f1_node,
            target_node_id=d2_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=d2_node,
            target_node_id=trk2_node,
            relationship=GraphEdgeType.BELONGS_TO,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    # Spatial Overlap between D1 and D2 in Frame 1
    store.upsert_edge(
        create_edge(
            source_node_id=d1_node,
            target_node_id=d2_node,
            relationship=GraphEdgeType.SPATIALLY_OVERLAPS,
            derivation=DerivationType.DETERMINISTIC,
            threshold=0.1,
            actual_value=0.25,
            reason="Bounding box IoU 0.25 >= threshold 0.10",
        )
    )

    # 7. OCR Observation
    ocr_id = str(uuid4())
    ocr_node = make_node_id(GraphNodeType.OCR_OBSERVATION, ocr_id)
    store.upsert_node(
        GraphNode(
            node_id=ocr_node,
            node_type=GraphNodeType.OCR_OBSERVATION,
            label="OCR:'DELIVERY' [1.0s - 3.0s]",
            provenance=NodeProvenance.create(
                source_id=ocr_id,
                source_type="OCRObservation",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=3.0,
            ),
            attributes={"text": "DELIVERY", "confidence": 0.96},
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=f1_node,
            target_node_id=ocr_node,
            relationship=GraphEdgeType.CONTAINS,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    # 8. Transcript Segment
    asr_id = str(uuid4())
    asr_node = make_node_id(GraphNodeType.TRANSCRIPT_SEGMENT, asr_id)
    store.upsert_node(
        GraphNode(
            node_id=asr_node,
            node_type=GraphNodeType.TRANSCRIPT_SEGMENT,
            label="ASR:'package arrived' [1.5s - 3.5s]",
            provenance=NodeProvenance.create(
                source_id=asr_id,
                source_type="TranscriptSegment",
                video_id=vid,
                timestamp_start=1.5,
                timestamp_end=3.5,
            ),
            attributes={"text": "package arrived", "language": "en"},
        )
    )

    # 9. Deterministic Events
    ev1_id = str(uuid4())
    ev1_node = make_node_id(GraphNodeType.EVENT, ev1_id)
    store.upsert_node(
        GraphNode(
            node_id=ev1_node,
            node_type=GraphNodeType.EVENT,
            label="Event:OBJECT_ENTERED_ZONE [1.0s - 2.0s]",
            provenance=NodeProvenance.create(
                source_id=ev1_id,
                source_type="Event",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=2.0,
            ),
            attributes={"event_type": "OBJECT_ENTERED_ZONE"},
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=trk1_node,
            target_node_id=ev1_node,
            relationship=GraphEdgeType.PARTICIPATES_IN,
            derivation=DerivationType.DETERMINISTIC,
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ev1_node,
            target_node_id=zone_node,
            relationship=GraphEdgeType.OCCURS_IN,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    ev2_id = str(uuid4())
    ev2_node = make_node_id(GraphNodeType.EVENT, ev2_id)
    store.upsert_node(
        GraphNode(
            node_id=ev2_node,
            node_type=GraphNodeType.EVENT,
            label="Event:TEXT_OBSERVED [1.0s - 3.0s]",
            provenance=NodeProvenance.create(
                source_id=ev2_id,
                source_type="Event",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=3.0,
            ),
            attributes={"event_type": "TEXT_OBSERVED"},
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ocr_node,
            target_node_id=ev2_node,
            relationship=GraphEdgeType.SUPPORTS,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    ev3_id = str(uuid4())
    ev3_node = make_node_id(GraphNodeType.EVENT, ev3_id)
    store.upsert_node(
        GraphNode(
            node_id=ev3_node,
            node_type=GraphNodeType.EVENT,
            label="Event:OBJECT_EXITED_ZONE [4.0s - 5.0s]",
            provenance=NodeProvenance.create(
                source_id=ev3_id,
                source_type="Event",
                video_id=vid,
                timestamp_start=4.0,
                timestamp_end=5.0,
            ),
            attributes={"event_type": "OBJECT_EXITED_ZONE"},
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=trk1_node,
            target_node_id=ev3_node,
            relationship=GraphEdgeType.PARTICIPATES_IN,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    # Temporal relations between events
    store.upsert_edge(
        create_edge(
            source_node_id=ev1_node,
            target_node_id=ev3_node,
            relationship=GraphEdgeType.PRECEDES,
            derivation=DerivationType.DETERMINISTIC,
            reason="Event 1 ends at 2.0s strictly before Event 3 begins at 4.0s",
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ev3_node,
            target_node_id=ev1_node,
            relationship=GraphEdgeType.FOLLOWS,
            derivation=DerivationType.DETERMINISTIC,
            reason="Event 3 begins at 4.0s strictly after Event 1 ends at 2.0s",
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ev1_node,
            target_node_id=ev2_node,
            relationship=GraphEdgeType.TEMPORALLY_NEAR,
            derivation=DerivationType.DETERMINISTIC,
            threshold=2.0,
            actual_value=0.0,
            reason="Events overlap temporally (gap = 0.0s <= threshold 2.0s)",
        )
    )

    # 10. Semantic Event (Layer 5)
    sem_id = str(uuid4())
    sem_node = make_node_id(GraphNodeType.SEMANTIC_EVENT, sem_id)
    store.upsert_node(
        GraphNode(
            node_id=sem_node,
            node_type=GraphNodeType.SEMANTIC_EVENT,
            label="Semantic:DELIVERY_DROPOFF [1.0s - 3.5s]",
            provenance=NodeProvenance.create(
                source_id=sem_id,
                source_type="SemanticEvent",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=3.5,
            ),
            attributes={
                "event_type": "DELIVERY_DROPOFF",
                "interpretation": "Person arrived at loading dock and unloaded delivery package.",
                "confidence": 0.91,
            },
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=sem_node,
            target_node_id=ev1_node,
            relationship=GraphEdgeType.DERIVED_FROM,
            derivation=DerivationType.INFERRED,
            reason="VLM reasoning grounded on zone entry candidate",
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ev1_node,
            target_node_id=sem_node,
            relationship=GraphEdgeType.SUPPORTS_SEMANTIC_EVENT,
            derivation=DerivationType.DETERMINISTIC,
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ocr_node,
            target_node_id=sem_node,
            relationship=GraphEdgeType.SUPPORTS_SEMANTIC_EVENT,
            derivation=DerivationType.DETERMINISTIC,
        )
    )

    # 11. Canonical Evidence Nodes
    evid1_id = str(uuid4())
    evid1_node = make_node_id(GraphNodeType.EVIDENCE, evid1_id)
    store.upsert_node(
        GraphNode(
            node_id=evid1_node,
            node_type=GraphNodeType.EVIDENCE,
            label="Evidence:DETECTION @ 1.00s",
            provenance=NodeProvenance.create(
                source_id=evid1_id,
                source_type="Evidence",
                video_id=vid,
                timestamp_start=1.0,
                timestamp_end=1.0,
            ),
        )
    )
    store.upsert_edge(
        create_edge(
            source_node_id=ev1_node,
            target_node_id=evid1_node,
            relationship=GraphEdgeType.SUPPORTED_BY,
            derivation=DerivationType.STRUCTURAL,
        )
    )

    return store
