"""Unit tests for VIDEX evaluation schemas and validation contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from videx.domain.schemas import BoundingBox, CoordinateType
from videx.evaluation.schemas import (
    AnnotationDataset,
    AnnotationStatus,
    GroundTruthDetection,
    GroundTruthEvent,
    GroundTruthOCR,
    GroundTruthQA,
    GroundTruthTrack,
    GroundTruthTranscript,
    ModalityAnnotation,
    VideoAnnotation,
)


class TestEvaluationSchemas:
    """Verifies schema validation, invariants, and edge cases."""

    def test_valid_ground_truth_detection(self) -> None:
        bbox = BoundingBox(
            x=10.0, y=20.0, width=50.0, height=80.0, coordinate_type=CoordinateType.PIXEL
        )
        det = GroundTruthDetection(
            timestamp_pts=1.5,
            class_label="person",
            bbox=bbox,
            frame_index=45,
            object_identity="worker_1",
        )
        assert det.timestamp_pts == 1.5
        assert det.class_label == "person"
        assert det.bbox.area == 4000.0

    def test_negative_timestamp_detection_rejected(self) -> None:
        bbox = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            GroundTruthDetection(
                timestamp_pts=-0.5,
                class_label="car",
                bbox=bbox,
            )

    def test_empty_class_label_rejected(self) -> None:
        bbox = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        with pytest.raises(ValidationError):
            GroundTruthDetection(
                timestamp_pts=1.0,
                class_label="",
                bbox=bbox,
            )

    def test_track_temporal_ordering_enforced(self) -> None:
        bbox = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        det1 = GroundTruthDetection(timestamp_pts=1.0, class_label="car", bbox=bbox)
        det2 = GroundTruthDetection(timestamp_pts=0.5, class_label="car", bbox=bbox)

        with pytest.raises(ValidationError, match="monotonically ordered"):
            GroundTruthTrack(
                track_id="trk_1",
                object_identity="car_1",
                class_label="car",
                observations=[det1, det2],
            )

    def test_transcript_interval_validation(self) -> None:
        with pytest.raises(ValidationError, match="cannot precede start_seconds"):
            GroundTruthTranscript(
                start_seconds=5.0,
                end_seconds=4.0,
                reference_text="Invalid backwards speech interval",
            )

    def test_event_interval_validation(self) -> None:
        with pytest.raises(ValidationError, match="cannot precede start_seconds"):
            GroundTruthEvent(
                event_type="loitering",
                start_seconds=10.0,
                end_seconds=8.0,
            )

    def test_answerable_qa_requires_reference_answer(self) -> None:
        with pytest.raises(ValidationError, match="must provide a reference_answer"):
            GroundTruthQA(
                question="What color is the hardhat?",
                is_answerable=True,
                reference_answer=None,
                acceptable_answers=[],
            )

    def test_unanswerable_qa_does_not_require_reference_answer(self) -> None:
        qa = GroundTruthQA(
            question="What is the worker's passport number?",
            is_answerable=False,
            reference_answer=None,
            acceptable_answers=[],
            annotation_rationale="Passport details are absent from surveillance footage",
        )
        assert qa.is_answerable is False
        assert qa.reference_answer is None

    def test_modality_status_consistency(self) -> None:
        bbox = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        det = GroundTruthDetection(timestamp_pts=0.0, class_label="box", bbox=bbox)

        # Items present when marked NOT_APPLICABLE or NOT_ANNOTATED must raise ValidationError
        with pytest.raises(ValidationError, match="must be empty when status is not_applicable"):
            ModalityAnnotation[GroundTruthDetection](
                status=AnnotationStatus.NOT_APPLICABLE,
                items=[det],
            )

        with pytest.raises(ValidationError, match="must be empty when status is not_annotated"):
            ModalityAnnotation[GroundTruthDetection](
                status=AnnotationStatus.NOT_ANNOTATED,
                items=[det],
            )

    def test_video_annotation_duration_exceeded_rejected(self) -> None:
        bbox = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        # Video is 5.0 seconds, detection is at 10.0 seconds
        det = GroundTruthDetection(timestamp_pts=10.0, class_label="car", bbox=bbox)

        with pytest.raises(ValidationError, match="exceeds video duration"):
            VideoAnnotation(
                video_id="vid_001",
                source_identifier="clip.mp4",
                duration_seconds=5.0,
                detections=ModalityAnnotation[GroundTruthDetection](
                    status=AnnotationStatus.ANNOTATED,
                    items=[det],
                ),
            )

    def test_duplicate_video_ids_in_dataset_rejected(self) -> None:
        v1 = VideoAnnotation(
            video_id="vid_same",
            source_identifier="clip1.mp4",
            duration_seconds=10.0,
        )
        v2 = VideoAnnotation(
            video_id="vid_same",
            source_identifier="clip2.mp4",
            duration_seconds=20.0,
        )
        with pytest.raises(ValidationError, match="Duplicate video_id 'vid_same'"):
            AnnotationDataset(
                dataset_id="test_ds",
                provenance="Unit Tests",
                videos=[v1, v2],
            )

    def test_deterministic_serialization_roundtrip(self) -> None:
        dataset = AnnotationDataset(
            schema_version="1.0.0",
            dataset_id="ds_roundtrip",
            provenance="Unit Tests",
            annotation_guidelines="docs/guidelines.md",
            videos=[
                VideoAnnotation(
                    video_id="v1",
                    source_identifier="test.mp4",
                    duration_seconds=12.0,
                    ocr=ModalityAnnotation[GroundTruthOCR](
                        status=AnnotationStatus.ANNOTATED,
                        items=[
                            GroundTruthOCR(
                                timestamp_pts=2.5,
                                expected_text="EXIT",
                                language="en",
                            )
                        ],
                    ),
                )
            ],
        )

        json_str = dataset.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"
        assert parsed["videos"][0]["video_id"] == "v1"

        reconstructed = AnnotationDataset.model_validate(parsed)
        assert reconstructed == dataset
