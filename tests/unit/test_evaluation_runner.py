"""Unit tests for VIDEX evaluation harness runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from videx.evaluation.runner import EvaluationHarness, format_cli_summary, load_annotation_dataset
from videx.evaluation.schemas import EvaluationResult, MetricStatus


class TestEvaluationRunner:
    """Verifies end-to-end fixture execution and CLI formatting."""

    def test_load_sample_fixture_dataset(self) -> None:
        fixture_path = Path("tests/fixtures/evaluation/sample_annotations.json")
        dataset = load_annotation_dataset(fixture_path)
        assert dataset.schema_version == "1.0.0"
        assert len(dataset.videos) == 1
        video = dataset.videos[0]
        assert video.video_id == "video_sample_001"
        assert len(video.detections.items) == 2

    def test_runner_execution_with_matching_predictions(self) -> None:
        fixture_path = Path("tests/fixtures/evaluation/sample_annotations.json")
        dataset = load_annotation_dataset(fixture_path)

        video = dataset.videos[0]
        qa_q1 = video.qa.items[0]
        qa_q2 = video.qa.items[1]

        predictions: dict[str, dict[str, list[Any]]] = {
            "video_sample_001": {
                "detections": [
                    {
                        "timestamp_pts": 0.0,
                        "class_label": "car",
                        "bbox": {
                            "x": 20.0,
                            "y": 50.0,
                            "width": 100.0,
                            "height": 80.0,
                            "coordinate_type": "pixel",
                        },
                    },
                    {
                        "timestamp_pts": 1.0,
                        "class_label": "car",
                        "bbox": {
                            "x": 60.0,
                            "y": 50.0,
                            "width": 100.0,
                            "height": 80.0,
                            "coordinate_type": "pixel",
                        },
                    },
                ],
                "qa": [
                    {
                        "qa_id": str(qa_q1.qa_id),
                        "predicted_answer": "Yes car visible",
                        "did_abstain": False,
                    },
                    {"qa_id": str(qa_q2.qa_id), "predicted_answer": None, "did_abstain": True},
                ],
            }
        }

        harness = EvaluationHarness(dataset=dataset, predictions_by_video=predictions)
        report: EvaluationResult = harness.run()

        assert len(report.per_video_results) == 1
        v_summary = report.per_video_results[0]

        # Detections should have perfect precision/recall
        det_prec = v_summary.metrics["detection_precision"]
        assert det_prec.status == MetricStatus.AVAILABLE
        assert det_prec.value == 1.0

        # OCR and Audio are NOT_APPLICABLE for this silent clip
        assert v_summary.metrics["ocr_cer"].status == MetricStatus.NOT_APPLICABLE
        assert v_summary.metrics["asr_wer"].status == MetricStatus.NOT_APPLICABLE

        # QA abstention should have 100% accuracy
        qa_acc = v_summary.metrics["qa_abstention_accuracy"]
        assert qa_acc.status == MetricStatus.AVAILABLE
        assert qa_acc.value == 1.0

        # Test CLI summary rendering
        summary_text = format_cli_summary(report)
        assert "VIDEX ACCURACY & INTEGRITY EVALUATION REPORT" in summary_text
        assert "sample.mp4" in summary_text

    def test_runner_on_unannotated_educational_slot(self) -> None:
        fixture_path = Path("tests/fixtures/evaluation/educational_slot.json")
        dataset = load_annotation_dataset(fixture_path)
        harness = EvaluationHarness(dataset=dataset)
        report = harness.run()

        assert len(report.per_video_results) == 1
        v_summary = report.per_video_results[0]
        # Modalities must be marked UNAVAILABLE, not 0.0
        assert v_summary.metrics["detection_precision"].status == MetricStatus.UNAVAILABLE
        assert v_summary.metrics["ocr_cer"].status == MetricStatus.UNAVAILABLE
        assert v_summary.metrics["asr_wer"].status == MetricStatus.UNAVAILABLE
        assert v_summary.metrics["event_f1"].status == MetricStatus.UNAVAILABLE
        assert v_summary.metrics["qa_abstention_accuracy"].status == MetricStatus.UNAVAILABLE
