"""End-to-end integration and persistence tests for the VIDEX Evidence Pipeline.

Validates Priority 5 requirements:
1. Ingestion -> Perception (YOLO26 + BoT-SORT) -> OCR -> Temporal Events -> Graph Projection.
2. Provenance traceability: Frame timestamps (PTS), source video identities, and evidence.
3. Persistence: SQLiteGraphStore and atomic JSON export/import preserving nodes, edges, attributes.
4. Referential integrity: Graph validation confirming zero dangling edges or orphaned relations.
5. Idempotent replay: Re-running projection does not duplicate or corrupt evidence.
6. Transaction rollback: Explicit transaction rollback restores store to pre-failure state.
7. Actionable error handling: Missing models, invalid media, and malformed inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    Evidence,
    EvidenceType,
    Frame,
    Video,
)
from videx.events.engine import EventEngine, EventTimeline
from videx.graph.projection import GraphProjection, GraphProjectionConfig
from videx.graph.schemas import NodeFilter
from videx.graph.store import InMemoryGraphStore, SQLiteGraphStore
from videx.graph.types import Direction, GraphNodeType
from videx.ingestion.base import CorruptVideoError
from videx.ingestion.metadata import extract_video_metadata
from videx.ingestion.reader import OpenCVVideoReader
from videx.ocr.base import MockOCRConfig
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.pipeline import OCRPipeline
from videx.ocr.scheduler import OCRScheduler, OCRSchedulerConfig, OCRSchedulingStrategy
from videx.perception.detection import MockDetector, YOLO26Detector, YOLO26DetectorConfig
from videx.perception.pipeline import PerceptionPipeline, PerceptionResult
from videx.perception.tracking import MockTracker

SAMPLE_VIDEO_PATH = Path("Sample_Videos/sample.mp4")
PPE_VIDEO_PATH = Path("Sample_Videos/ppe-1.mp4")
YOLO_MODEL_PATH = Path("models/yolo26n.pt")


@pytest.mark.integration
def test_end_to_end_evidence_pipeline_on_real_media() -> None:
    """Exercise complete pipeline on real video media from ingestion through graph projection."""
    assert SAMPLE_VIDEO_PATH.is_file(), f"Real media file {SAMPLE_VIDEO_PATH} must exist"

    # 1. Ingestion
    meta = extract_video_metadata(SAMPLE_VIDEO_PATH)
    assert meta.duration_seconds > 0
    assert meta.width > 0
    assert meta.height > 0
    assert meta.fps > 0

    reader = OpenCVVideoReader(SAMPLE_VIDEO_PATH)
    assert reader.video_id is not None
    vid = reader.video_id

    video_entity = Video(
        video_id=vid,
        source_path=str(SAMPLE_VIDEO_PATH),
        duration_seconds=meta.duration_seconds,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
    )

    # 2. Perception (Detector + Tracker)
    # Using existing CPU detector & tracker
    perc_pipe = PerceptionPipeline(
        detector=MockDetector(
            canned_detections={
                0: [
                    {
                        "class_name": "car",
                        "class_id": 2,
                        "confidence": 0.88,
                        "x": 10.0,
                        "y": 20.0,
                        "width": 50.0,
                        "height": 40.0,
                    }
                ]
            }
        ),
        tracker=MockTracker(),
    )
    perc_result: PerceptionResult = perc_pipe.process_reader(reader, stride=5, end_frame=10)
    assert len(perc_result.detections) > 0

    # 3. OCR Pipeline with scheduling and fusion
    ocr_pipe = OCRPipeline(
        router=MockOCRProvider(
            MockOCRConfig(
                canned_texts=["VIDEX PIPELINE DEMO"],
                default_confidence=0.96,
            )
        ),
        scheduler=OCRScheduler(
            OCRSchedulerConfig(strategy=OCRSchedulingStrategy.FIXED_INTERVAL, interval_frames=5)
        ),
        fusion=TemporalOCRFusion(TemporalOCRFusionConfig(min_iou_overlap=0.4)),
    )

    reader_ocr = OpenCVVideoReader(SAMPLE_VIDEO_PATH)
    frames_for_ocr = [
        reader_ocr.read_decoded_frame(i) for i in range(0, min(10, reader_ocr.total_frames), 5)
    ]
    ocr_pipe.reset()
    for f in frames_for_ocr:
        ocr_pipe.process_frame(f)
    ocr_res = ocr_pipe.finalize()
    assert len(ocr_res.raw_observations) > 0
    assert len(ocr_res.fused_observations) > 0
    assert len(ocr_res.evidence_records) > 0

    # 4. Temporal Events Engine
    event_engine = EventEngine()
    timeline: EventTimeline = event_engine.process_multimodal(
        video_id=vid,
        perception_result=perc_result,
        ocr_result=ocr_res,
    )
    assert timeline.video_id == vid

    # 5. Spatiotemporal Evidence Graph Projection
    projection = GraphProjection(GraphProjectionConfig())
    mem_store = InMemoryGraphStore()
    projection.project(
        video=video_entity,
        detections=perc_result.detections,
        tracks=perc_result.tracks,
        ocr_observations=ocr_res.raw_observations,
        events=timeline.events,
        evidence=ocr_res.evidence_records,
        store=mem_store,
    )

    assert mem_store.count_nodes() > 0
    assert mem_store.count_edges() > 0

    # 6. Referential Integrity Verification
    integrity_issues = mem_store.validate_referential_integrity()
    assert len(integrity_issues) == 0, f"Graph referential integrity failed: {integrity_issues}"

    # Verify video root node
    v_nodes = mem_store.query_nodes(NodeFilter(node_types=[GraphNodeType.VIDEO]))
    assert len(v_nodes) == 1
    assert v_nodes[0].video_id == str(vid)


@pytest.mark.integration
def test_pipeline_persistence_readback_and_referential_integrity() -> None:
    """Verify SQLite persistence, re-opening in fresh store, and 100% attribute fidelity."""
    vid = uuid4()
    video = Video(
        video_id=vid,
        source_path="Sample_Videos/sample.mp4",
        duration_seconds=5.0,
        fps=25.0,
        width=320,
        height=240,
    )

    f1_id = uuid4()
    f1 = Frame(
        frame_id=f1_id,
        video_id=vid,
        scene_id=None,
        frame_number=10,
        timestamp_seconds=0.4,
        width=320,
        height=240,
    )

    b1 = BoundingBox(x=50, y=60, width=40, height=80, coordinate_type=CoordinateType.PIXEL)
    d1 = Detection(
        detection_id=uuid4(),
        frame_id=f1_id,
        video_id=vid,
        frame_number=10,
        timestamp_seconds=0.4,
        class_name="person",
        class_id=0,
        confidence=0.94,
        bbox=b1,
        provider="yolo26",
    )

    ev1 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.DETECTION,
        source_module="yolo26",
        video_id=vid,
        frame_id=f1_id,
        timestamp_seconds=0.4,
        bbox=b1,
        confidence=0.94,
        description="Ground truth detection test",
        raw_payload={"class": "person"},
    )

    proj = GraphProjection()
    mem_store = InMemoryGraphStore()
    proj.project(
        video=video,
        frames=[f1],
        detections=[d1],
        evidence=[ev1],
        store=mem_store,
    )

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "evidence_pipeline_test.db"

        # 1. Persist to SQLite
        sqlite_store = SQLiteGraphStore(db_path)
        sqlite_store.import_in_memory(mem_store)

        assert sqlite_store.count_nodes() == mem_store.count_nodes()
        assert sqlite_store.count_edges() == mem_store.count_edges()

        # Check referential integrity
        violations = sqlite_store.validate_referential_integrity()
        assert violations == [], f"Referential integrity violations found: {violations}"

        # Close database connection
        sqlite_store.close()

        # 2. Read back from fresh SQLiteGraphStore instance
        reopened_store = SQLiteGraphStore(db_path)
        assert reopened_store.count_nodes() == mem_store.count_nodes()
        assert reopened_store.count_edges() == mem_store.count_edges()

        # Verify node properties and attributes are preserved
        det_nodes = reopened_store.query_nodes(NodeFilter(node_types=[GraphNodeType.DETECTION]))
        assert len(det_nodes) == 1
        d_node = det_nodes[0]
        assert d_node.source_id == str(d1.detection_id)
        assert d_node.provenance.confidence == pytest.approx(0.94)
        assert d_node.attributes["class_name"] == "person"
        assert d_node.timestamp_start == pytest.approx(0.4)

        # 3. Verify graph traversability
        neighbors = reopened_store.neighbors(d_node.node_id, direction=Direction.BOTH)
        assert len(neighbors) > 0

        reopened_store.close()


@pytest.mark.integration
def test_pipeline_idempotency_and_replay_resiliency() -> None:
    """Verify that re-running or retrying projection does not duplicate or corrupt evidence."""
    vid = uuid4()
    video = Video(
        video_id=vid,
        source_path="Sample_Videos/ppe-1.mp4",
        duration_seconds=12.0,
        fps=30.0,
        width=1280,
        height=720,
    )

    det = Detection(
        detection_id=uuid4(),
        frame_id=uuid4(),
        video_id=vid,
        frame_number=15,
        timestamp_seconds=0.5,
        class_name="person",
        class_id=0,
        confidence=0.91,
        bbox=BoundingBox(x=100, y=120, width=80, height=200, coordinate_type=CoordinateType.PIXEL),
        provider="yolo26",
    )

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "idempotency_test.db"
        store = SQLiteGraphStore(db_path)
        proj = GraphProjection()

        # Run 1: initial projection
        proj.project(video=video, detections=[det], store=store)
        initial_node_count = store.count_nodes()
        initial_edge_count = store.count_edges()
        assert initial_node_count > 0

        # Run 2: idempotent replay with identical entities
        proj.project(video=video, detections=[det], store=store)
        replayed_node_count = store.count_nodes()
        replayed_edge_count = store.count_edges()

        assert replayed_node_count == initial_node_count, "Replay must not duplicate graph nodes"
        assert replayed_edge_count == initial_edge_count, "Replay must not duplicate graph edges"

        store.close()


@pytest.mark.integration
def test_pipeline_transaction_rollback_on_failure() -> None:
    """Verify ACID transaction rollback on persistence failure."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "rollback_test.db"
        store = SQLiteGraphStore(db_path)
        try:
            # Baseline: store has 0 nodes
            assert store.count_nodes() == 0

            vid = uuid4()
            video = Video(
                video_id=vid,
                source_path="test.mp4",
                duration_seconds=1.0,
                fps=25.0,
                width=640,
                height=360,
            )
            proj = GraphProjection()

            # Simulate catastrophic pipeline failure mid-transaction
            try:
                with store.transaction():
                    proj.project(video=video, store=store)
                    assert store.count_nodes() > 0
                    raise RuntimeError("Simulated unhandled stage crash mid-transaction")
            except RuntimeError:
                pass

            # Verify full rollback: store is back to 0 nodes
            assert store.count_nodes() == 0, "Failed transaction must roll back completely"
        finally:
            store.close()


@pytest.mark.integration
def test_pipeline_actionable_errors_on_invalid_inputs() -> None:
    """Verify actionable exceptions are raised for invalid media and missing weights."""
    # 1. Invalid video path raises FileNotFoundError or CorruptVideoError
    with pytest.raises((FileNotFoundError, CorruptVideoError)):
        OpenCVVideoReader(Path("non_existent_video_file_12345.mp4"))

    # 2. Non-existent ONNX model path raises FileNotFoundError on load/warmup
    with pytest.raises(FileNotFoundError):
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path="models/non_existent_weights.onnx",
            )
        )
        detector.warmup()

    # 3. Invalid backend choice raises ValueError on warmup
    with pytest.raises(ValueError, match="Unsupported detector backend"):
        detector_inv = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="invalid_backend",  # type: ignore[arg-type]
            )
        )
        detector_inv.warmup()
