"""Unit tests for VIDEX Evidence Graph domain models, store, and engines."""

from __future__ import annotations

from uuid import uuid4

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    Frame,
    Scene,
    Track,
    Video,
)
from videx.events.schemas import Event
from videx.events.spatial import SpatialZone
from videx.events.types import EventType
from videx.graph.edges import create_edge, make_edge_id
from videx.graph.mock import create_mock_graph_store
from videx.graph.nodes import (
    create_video_node,
    create_zone_node,
    make_node_id,
)
from videx.graph.projection import GraphProjection
from videx.graph.schemas import EdgeFilter, NodeFilter
from videx.graph.spatial import SpatialGraphEngine, compute_bbox_iou
from videx.graph.store import InMemoryGraphStore
from videx.graph.temporal import TemporalGraphEngine
from videx.graph.types import (
    DerivationType,
    Direction,
    GraphEdgeType,
    GraphNodeType,
)
from videx.graph.validator import GraphIntegrityValidator


def test_node_id_and_edge_id_formatting() -> None:
    nid = make_node_id(GraphNodeType.EVENT, "123")
    assert nid == "event:123"

    eid = make_edge_id("event:1", "event:2", GraphEdgeType.PRECEDES)
    assert eid == "PRECEDES:event:1->event:2"

    eid_qual = make_edge_id("a", "b", GraphEdgeType.CONTAINS, qualifier="part1")
    assert eid_qual == "CONTAINS:a->b#part1"


def test_in_memory_graph_store_crud_and_traversal() -> None:
    store = InMemoryGraphStore()
    vid = str(uuid4())

    n1 = create_video_node(Video(source_path="test.mp4", duration_seconds=10.0))
    n2 = create_zone_node(
        SpatialZone.from_rectangle("zone_1", "Zone 1", 0, 0, 100, 100),
        vid,
    )
    store.upsert_node(n1)
    store.upsert_node(n2)

    assert store.count_nodes() == 2
    assert store.get_node(n1.node_id) is not None
    assert store.get_node("non_existent") is None

    edge = create_edge(
        source_node_id=n1.node_id,
        target_node_id=n2.node_id,
        relationship=GraphEdgeType.CONTAINS,
        derivation=DerivationType.STRUCTURAL,
    )
    store.upsert_edge(edge)
    assert store.count_edges() == 1
    assert store.get_edge(edge.edge_id) is not None

    # Neighbors
    out_neighbors = store.neighbors(n1.node_id, direction=Direction.OUTBOUND)
    assert len(out_neighbors) == 1
    assert out_neighbors[0].node_id == n2.node_id

    in_neighbors = store.neighbors(n2.node_id, direction=Direction.INBOUND)
    assert len(in_neighbors) == 1
    assert in_neighbors[0].node_id == n1.node_id

    # Path finding
    paths = store.find_path(n1.node_id, n2.node_id, max_depth=2)
    assert len(paths) == 1
    assert paths[0].length == 1
    assert len(paths[0].nodes) == 2


def test_mock_graph_store_integrity() -> None:
    store = create_mock_graph_store()
    validator = GraphIntegrityValidator()
    report = validator.validate(store)

    assert report.is_valid, f"Mock graph must be valid: {report.errors}"
    assert report.total_nodes > 10
    assert report.total_edges > 10
    assert len(report.errors) == 0


def test_temporal_graph_engine_precedes_follows_and_near() -> None:
    vid = uuid4()
    ev1 = Event(
        event_id=uuid4(),
        video_id=vid,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=2.0,
        confidence=0.9,
        description="Entry",
    )
    ev2 = Event(
        event_id=uuid4(),
        video_id=vid,
        event_type=EventType.OBJECT_EXITED_ZONE,
        start_timestamp_seconds=3.0,
        end_timestamp_seconds=4.0,
        confidence=0.9,
        description="Exit",
    )
    ev3 = Event(
        event_id=uuid4(),
        video_id=vid,
        event_type=EventType.TEXT_APPEARED,
        start_timestamp_seconds=10.0,
        end_timestamp_seconds=11.0,
        confidence=0.9,
        description="Late Text",
    )

    engine = TemporalGraphEngine(near_threshold_seconds=1.5)
    edges = engine.compute_temporal_edges([ev1, ev2, ev3])

    rel_types = [e.relationship for e in edges]
    assert GraphEdgeType.PRECEDES in rel_types
    assert GraphEdgeType.FOLLOWS in rel_types
    assert GraphEdgeType.TEMPORALLY_NEAR in rel_types

    # ev1 and ev2 have gap of 1.0s <= threshold 1.5s -> TEMPORALLY_NEAR
    near_edges = [e for e in edges if e.relationship == GraphEdgeType.TEMPORALLY_NEAR]
    assert len(near_edges) == 1
    assert near_edges[0].source_node_id == f"event:{ev1.event_id}"
    assert near_edges[0].target_node_id == f"event:{ev2.event_id}"
    assert near_edges[0].provenance.actual_value == 1.0


def test_spatial_graph_engine_overlap_and_containment() -> None:
    vid = uuid4()
    fid = uuid4()
    b1 = BoundingBox(x=10, y=10, width=50, height=50, coordinate_type=CoordinateType.PIXEL)
    b2 = BoundingBox(x=20, y=20, width=50, height=50, coordinate_type=CoordinateType.PIXEL)
    b3 = BoundingBox(x=200, y=200, width=50, height=50, coordinate_type=CoordinateType.PIXEL)

    iou_12 = compute_bbox_iou(b1, b2)
    assert iou_12 > 0.0
    iou_13 = compute_bbox_iou(b1, b3)
    assert iou_13 == 0.0

    d1 = Detection(
        detection_id=uuid4(),
        frame_id=fid,
        video_id=vid,
        frame_number=1,
        timestamp_seconds=1.0,
        class_name="person",
        class_id=0,
        confidence=0.9,
        bbox=b1,
        provider="yolo",
    )
    d2 = Detection(
        detection_id=uuid4(),
        frame_id=fid,
        video_id=vid,
        frame_number=1,
        timestamp_seconds=1.0,
        class_name="car",
        class_id=2,
        confidence=0.85,
        bbox=b2,
        provider="yolo",
    )

    spatial_engine = SpatialGraphEngine(iou_threshold=0.1)
    overlap_edges = spatial_engine.compute_spatial_overlap_edges([d1, d2])
    assert len(overlap_edges) == 1
    assert overlap_edges[0].relationship == GraphEdgeType.SPATIALLY_OVERLAPS

    # Zone containment
    zone = SpatialZone.from_rectangle("dock", "Dock", 0, 0, 100, 100)
    trk = Track(
        track_id=uuid4(),
        video_id=vid,
        class_name="person",
        first_seen_frame_number=1,
        last_seen_frame_number=5,
        first_seen_timestamp_seconds=1.0,
        last_seen_timestamp_seconds=2.0,
        start_bbox=b1,
        provider="bytetrack",
    )
    containment_edges = spatial_engine.compute_zone_containment_edges([trk], [zone])
    assert len(containment_edges) == 1
    assert containment_edges[0].relationship == GraphEdgeType.OCCURS_IN


def test_graph_projection_determinism() -> None:
    vid = uuid4()
    video = Video(video_id=vid, source_path="deterministic.mp4", duration_seconds=10.0)
    scene = Scene(
        scene_id=uuid4(),
        video_id=vid,
        scene_index=0,
        start_frame_number=0,
        end_frame_number=100,
        start_timestamp_seconds=0.0,
        end_timestamp_seconds=10.0,
    )
    frame = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=scene.scene_id,
        frame_number=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
    )
    ev = Event(
        event_id=uuid4(),
        video_id=vid,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=2.0,
        confidence=0.95,
        description="Deterministic Event",
    )

    proj = GraphProjection()
    store1 = proj.project(video=video, scenes=[scene], frames=[frame], events=[ev])
    store2 = proj.project(video=video, scenes=[scene], frames=[frame], events=[ev])

    nodes1 = sorted(n.node_id for n in store1.query_nodes(NodeFilter()))
    nodes2 = sorted(n.node_id for n in store2.query_nodes(NodeFilter()))
    assert nodes1 == nodes2

    edges1 = sorted(e.edge_id for e in store1.query_edges(EdgeFilter()))
    edges2 = sorted(e.edge_id for e in store2.query_edges(EdgeFilter()))
    assert edges1 == edges2


def test_validator_detects_orphan_edges_and_cross_video_leak() -> None:
    store = InMemoryGraphStore()
    vid1 = str(uuid4())
    vid2 = str(uuid4())

    n1 = create_video_node(Video(video_id=vid1, source_path="v1.mp4", duration_seconds=5.0))
    n2 = create_video_node(Video(video_id=vid2, source_path="v2.mp4", duration_seconds=5.0))
    store.upsert_node(n1)
    store.upsert_node(n2)

    # Edge pointing to non-existent node
    orphan_edge = create_edge(
        source_node_id=n1.node_id,
        target_node_id="event:non_existent",
        relationship=GraphEdgeType.CONTAINS,
    )
    store.upsert_edge(orphan_edge)

    # Cross-video edge
    cross_video_edge = create_edge(
        source_node_id=n1.node_id,
        target_node_id=n2.node_id,
        relationship=GraphEdgeType.CONTAINS,
    )
    store.upsert_edge(cross_video_edge)

    validator = GraphIntegrityValidator()
    report = validator.validate(store)
    assert not report.is_valid
    assert any("Orphan edge" in err for err in report.errors)
    assert any("Cross-video isolation violation" in err for err in report.errors)
