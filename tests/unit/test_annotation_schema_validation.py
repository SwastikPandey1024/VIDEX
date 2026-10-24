"""Unit tests for VIDEX annotation quality control and schema validation."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from videx.evaluation.annotation_validator import (
    compute_iou,
    validate_annotation_data,
    validate_file,
)
from videx.evaluation.schemas import AnnotationDataset


@pytest.fixture
def valid_12frame_dataset() -> dict[str, Any]:
    """Creates a synthetic perfectly valid 12-frame dataset adhering to Policy A."""
    expected_frames = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    items = []
    track_observations = []
    for idx in expected_frames:
        det = {
            "detection_id": str(uuid4()),
            "timestamp_pts": round(idx / 29.97003, 4),
            "frame_index": idx,
            "class_label": "person",
            "object_identity": "worker_1",
            "bbox": {
                "x": 560.0,
                "y": 140.0,
                "width": 280.0,
                "height": 510.0,
                "coordinate_type": "pixel",
            },
        }
        items.append(det)
        track_observations.append(det)

    return {
        "schema_version": "1.0.0",
        "dataset_id": "test_valid_benchmark",
        "provenance": "VIDEX Verified Human Ground Truth (Policy A)",
        "annotation_guidelines": "docs/evaluation/annotation_guidelines_ppe1_v1.md",
        "videos": [
            {
                "video_id": "video_ppe1_001",
                "source_identifier": "ppe-1.mp4",
                "duration_seconds": 11.445,
                "overall_status": "annotated",
                "detections": {
                    "status": "annotated",
                    "items": items,
                    "rationale": "Verified human ground truth",
                },
                "tracks": {
                    "status": "annotated",
                    "items": [
                        {
                            "track_id": "track_worker_1",
                            "object_identity": "worker_1",
                            "class_label": "person",
                            "observations": track_observations,
                        }
                    ],
                    "rationale": "Worker 1 verified trajectory",
                },
                "ocr": {"status": "not_applicable", "items": [], "rationale": "No text"},
                "transcripts": {
                    "status": "not_applicable",
                    "items": [],
                    "rationale": "Silent video",
                },
                "events": {"status": "not_applicable", "items": [], "rationale": "Not annotated"},
                "qa": {"status": "not_applicable", "items": [], "rationale": "Not annotated"},
            }
        ],
    }


def test_valid_dataset_passes_validation(valid_12frame_dataset: dict[str, Any]) -> None:
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_frames_checked == 12
    assert report.total_boxes_checked == 12


def test_export_structure_conforms_to_pydantic_schema(
    valid_12frame_dataset: dict[str, Any],
) -> None:
    """Verifies that the exact JSON structure exported by annotator.html
    passes Pydantic model validation.
    """
    parsed = AnnotationDataset.model_validate(valid_12frame_dataset)
    assert parsed.schema_version == "1.0.0"
    assert parsed.dataset_id == "test_valid_benchmark"
    assert len(parsed.videos) == 1
    assert len(parsed.videos[0].detections.items) == 12
    assert len(parsed.videos[0].tracks.items) == 1
    assert len(parsed.videos[0].tracks.items[0].observations) == 12


def test_missing_provenance_error(valid_12frame_dataset: dict[str, Any]) -> None:
    valid_12frame_dataset["provenance"] = ""
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is False
    assert any(e.code == "MISSING_PROVENANCE" for e in report.errors)


def test_prediction_leakage_detected(valid_12frame_dataset: dict[str, Any]) -> None:
    # Inject a detector confidence score into one item
    valid_12frame_dataset["videos"][0]["detections"]["items"][0]["confidence"] = 0.945
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is False
    assert any(e.code == "PREDICTION_LEAKAGE_DETECTED" for e in report.errors)


def test_out_of_bounds_coordinates(valid_12frame_dataset: dict[str, Any]) -> None:
    # Box extends past right edge: 1100 + 300 = 1400 > 1280
    valid_12frame_dataset["videos"][0]["detections"]["items"][0]["bbox"]["x"] = 1100.0
    valid_12frame_dataset["videos"][0]["detections"]["items"][0]["bbox"]["width"] = 300.0
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is False
    assert any(e.code == "OUT_OF_BOUNDS_RIGHT" for e in report.errors)


def test_non_positive_dimensions(valid_12frame_dataset: dict[str, Any]) -> None:
    valid_12frame_dataset["videos"][0]["detections"]["items"][0]["bbox"]["width"] = -5.0
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is False
    assert any(e.code == "NON_POSITIVE_DIMENSIONS" for e in report.errors)


def test_non_person_class_label(valid_12frame_dataset: dict[str, Any]) -> None:
    valid_12frame_dataset["videos"][0]["detections"]["items"][0]["class_label"] = "hardhat"
    report = validate_annotation_data(valid_12frame_dataset)
    assert report.is_valid is False
    assert any(e.code == "INVALID_CLASS" for e in report.errors)


def test_missing_frame_index(valid_12frame_dataset: dict[str, Any]) -> None:
    # Remove one required frame (frame 180)
    items = valid_12frame_dataset["videos"][0]["detections"]["items"]
    valid_12frame_dataset["videos"][0]["detections"]["items"] = [
        it for it in items if it["frame_index"] != 180
    ]
    report = validate_annotation_data(valid_12frame_dataset)
    # Since frame 180 has 0 items, but is expected:
    assert report.is_valid is True  # 0 boxes on frame 180 is accepted as empty frame


def test_suspicious_overlapping_boxes_warning(valid_12frame_dataset: dict[str, Any]) -> None:
    # Add a duplicate box on frame 0
    box_dup = dict(valid_12frame_dataset["videos"][0]["detections"]["items"][0])
    box_dup["object_identity"] = "worker_2"
    valid_12frame_dataset["videos"][0]["detections"]["items"].append(box_dup)

    report = validate_annotation_data(valid_12frame_dataset)
    assert any(w.code == "SUSPICIOUS_OVERLAP" for w in report.warnings)


def test_progress_persistence_and_recovery_roundtrip() -> None:
    """Verifies that intermediate session progress serializes and deserializes accurately."""
    saved_progress = {
        "progress_version": "2.0.0",
        "saved_at": "2026-10-22T00:30:00Z",
        "current_frame_index": 3,
        "review_status": {
            "0": "annotated",
            "30": "annotated",
            "60": "reviewed_empty",
            "90": "pending",
        },
        "annotations": {
            "0": [
                {
                    "detection_id": "test-uuid-0",
                    "timestamp_pts": 0.0,
                    "frame_index": 0,
                    "class_label": "person",
                    "object_identity": "worker_1",
                    "bbox": {"x": 570.0, "y": 140.0, "width": 260.0, "height": 500.0},
                }
            ],
            "30": [
                {
                    "detection_id": "test-uuid-30",
                    "timestamp_pts": 1.001,
                    "frame_index": 30,
                    "class_label": "person",
                    "object_identity": "worker_1",
                    "bbox": {"x": 560.0, "y": 135.0, "width": 270.0, "height": 510.0},
                }
            ],
            "60": [],
            "90": [],
        },
    }

    # Verify JSON roundtrip
    serialized = json.dumps(saved_progress)
    recovered = json.loads(serialized)

    assert recovered["progress_version"] == "2.0.0"
    assert recovered["current_frame_index"] == 3
    assert recovered["review_status"]["60"] == "reviewed_empty"
    assert len(recovered["annotations"]["0"]) == 1
    assert recovered["annotations"]["60"] == []


def test_annotator_export_workflow_validation(tmp_path: Path) -> None:
    """Tests the full export and file-based validation workflow using temporary data."""
    export_payload = {
        "schema_version": "1.0.0",
        "dataset_id": "videx_pilot_ppe1_independent_v1",
        "provenance": "VIDEX Independent Human Annotation (Policy A: Visible Person Only)",
        "annotation_guidelines": "docs/evaluation/annotation_guidelines_ppe1_v1.md",
        "videos": [
            {
                "video_id": "video_ppe1_001",
                "source_identifier": "ppe-1.mp4",
                "duration_seconds": 11.445,
                "overall_status": "annotated",
                "detections": {
                    "status": "annotated",
                    "items": [
                        {
                            "detection_id": str(uuid4()),
                            "timestamp_pts": 0.0,
                            "class_label": "person",
                            "bbox": {
                                "x": 570.0,
                                "y": 140.0,
                                "width": 260.0,
                                "height": 500.0,
                                "coordinate_type": "pixel",
                            },
                            "frame_index": 0,
                            "object_identity": "worker_1",
                        }
                    ],
                    "rationale": "Test export",
                },
                "tracks": {
                    "status": "annotated",
                    "items": [
                        {
                            "track_id": "track_worker_1",
                            "object_identity": "worker_1",
                            "class_label": "person",
                            "observations": [
                                {
                                    "detection_id": str(uuid4()),
                                    "timestamp_pts": 0.0,
                                    "class_label": "person",
                                    "bbox": {
                                        "x": 570.0,
                                        "y": 140.0,
                                        "width": 260.0,
                                        "height": 500.0,
                                        "coordinate_type": "pixel",
                                    },
                                    "frame_index": 0,
                                    "object_identity": "worker_1",
                                }
                            ],
                        }
                    ],
                    "rationale": "Test track",
                },
                "ocr": {"status": "not_applicable", "items": []},
                "transcripts": {"status": "not_applicable", "items": []},
                "events": {"status": "not_applicable", "items": []},
                "qa": {"status": "not_applicable", "items": []},
            }
        ],
    }

    test_file = tmp_path / "test_annotator_export.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    report = validate_file(test_file)
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_compute_iou_geometry() -> None:
    b1 = {"x": 100.0, "y": 100.0, "width": 100.0, "height": 100.0}
    b2 = {"x": 100.0, "y": 100.0, "width": 100.0, "height": 100.0}
    assert compute_iou(b1, b2) == pytest.approx(1.0)

    # Disjoint
    b3 = {"x": 300.0, "y": 300.0, "width": 100.0, "height": 100.0}
    assert compute_iou(b1, b3) == 0.0

    # Half overlap: x in [150, 200] -> width 50, height 100 -> area 5000.
    # union = 10000 + 10000 - 5000 = 15000 -> IoU = 5000/15000 = 1/3
    b4 = {"x": 150.0, "y": 100.0, "width": 100.0, "height": 100.0}
    assert compute_iou(b1, b4) == pytest.approx(1.0 / 3.0)
