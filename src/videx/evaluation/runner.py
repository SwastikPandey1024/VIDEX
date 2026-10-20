"""Unified, reproducible evaluation harness runner for VIDEX.

Loads gold-standard annotations, computes mathematically defined metrics,
enforces explicit missing/incomplete semantics, and outputs machine-readable
EvaluationResult JSON along with formatted CLI summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from videx.domain.schemas import BoundingBox
from videx.evaluation.metrics import (
    DetectionPrediction,
    EventPrediction,
    QAPrediction,
    compute_cer,
    compute_wer,
    evaluate_detections,
    evaluate_qa_abstention,
    evaluate_temporal_events,
)
from videx.evaluation.schemas import (
    AnnotationDataset,
    AnnotationStatus,
    EvaluationResult,
    MetricResult,
    MetricStatus,
    VideoAnnotation,
    VideoEvaluationSummary,
)

logger = logging.getLogger("videx.evaluation")


def get_git_commit() -> str | None:
    """Safely retrieves the current git commit hash if in a git repository."""
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        res = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except OSError as err:
        logger.debug("Failed to retrieve git commit: %s", err)
    return None


def load_annotation_dataset(path: Path) -> AnnotationDataset:
    """Loads and strictly validates an AnnotationDataset from a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return AnnotationDataset.model_validate(data)


class EvaluationHarness:
    """Executes evaluation across an AnnotationDataset using provided predictions."""

    def __init__(
        self,
        dataset: AnnotationDataset,
        predictions_by_video: dict[str, dict[str, list[Any]]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.predictions = predictions_by_video or {}
        self.config = config or {
            "iou_threshold": 0.5,
            "tiou_threshold": 0.5,
            "case_sensitive_ocr": False,
        }

    def evaluate_video(self, video_ann: VideoAnnotation) -> VideoEvaluationSummary:
        """Evaluates all annotated modalities for a single video."""
        vid = str(video_ann.video_id)
        video_preds = self.predictions.get(vid, {})
        video_metrics: dict[str, MetricResult] = {}
        unavailable_modalities: list[str] = []
        errors: list[str] = []

        # 1. Detection Evaluation
        if video_ann.detections.status == AnnotationStatus.ANNOTATED:
            gt_dets = video_ann.detections.items
            pred_dets_raw = video_preds.get("detections", [])
            # Map raw predictions if needed
            pred_dets: list[DetectionPrediction] = []
            for p in pred_dets_raw:
                if isinstance(p, DetectionPrediction):
                    pred_dets.append(p)
                elif isinstance(p, dict):
                    bbox_raw = p.get("bbox")
                    if isinstance(bbox_raw, BoundingBox):
                        bbox_obj = bbox_raw
                    elif isinstance(bbox_raw, dict):
                        bbox_obj = BoundingBox(**bbox_raw)
                    else:
                        raise ValueError(f"Invalid bounding box in prediction: {bbox_raw}")

                    pred_dets.append(
                        DetectionPrediction(
                            timestamp_pts=float(p["timestamp_pts"]),
                            class_label=str(p["class_label"]),
                            bbox=bbox_obj,
                            confidence=float(p.get("confidence", 1.0)),
                        )
                    )

            det_metrics = evaluate_detections(
                ground_truth=gt_dets,
                predictions=pred_dets,
                iou_threshold=self.config.get("iou_threshold", 0.5),
            )
            video_metrics.update(det_metrics)
        elif video_ann.detections.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["detection_precision"] = MetricResult(
                name="detection_precision",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=(
                    video_ann.detections.rationale or "Modality marked not applicable"
                ),
            )
        else:
            unavailable_modalities.append("detections")
            video_metrics["detection_precision"] = MetricResult(
                name="detection_precision",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"Detections status is {video_ann.detections.status.value}",
            )

        # 2. OCR Evaluation
        if video_ann.ocr.status == AnnotationStatus.ANNOTATED:
            gt_ocr = video_ann.ocr.items
            pred_ocr_texts: list[str] = video_preds.get("ocr", [])
            if not gt_ocr:
                video_metrics["ocr_cer"] = MetricResult(
                    name="ocr_cer",
                    status=MetricStatus.AVAILABLE,
                    value=0.0 if not pred_ocr_texts else 1.0,
                    sample_size=0,
                    metadata={"note": "Verified zero on-screen text in ground truth"},
                )
            else:
                # Concatenate/aggregate CER over ground-truth reference texts
                ref_text = " ".join(o.expected_text for o in gt_ocr)
                hyp_text = " ".join(pred_ocr_texts)
                cer_val, cer_meta = compute_cer(
                    ref_text,
                    hyp_text,
                    case_sensitive=self.config.get("case_sensitive_ocr", False),
                )
                video_metrics["ocr_cer"] = MetricResult(
                    name="ocr_cer",
                    status=MetricStatus.AVAILABLE,
                    value=cer_val,
                    sample_size=cer_meta["reference_chars"],
                    metadata=cer_meta,
                )
        elif video_ann.ocr.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["ocr_cer"] = MetricResult(
                name="ocr_cer",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.ocr.rationale or "OCR marked not applicable",
            )
        else:
            unavailable_modalities.append("ocr")
            video_metrics["ocr_cer"] = MetricResult(
                name="ocr_cer",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"OCR status is {video_ann.ocr.status.value}",
            )

        # 3. Audio / Transcript Evaluation
        if video_ann.transcripts.status == AnnotationStatus.ANNOTATED:
            gt_tr = video_ann.transcripts.items
            pred_tr_texts: list[str] = video_preds.get("transcripts", [])
            if not gt_tr:
                video_metrics["asr_wer"] = MetricResult(
                    name="asr_wer",
                    status=MetricStatus.AVAILABLE,
                    value=0.0 if not pred_tr_texts else 1.0,
                    sample_size=0,
                    metadata={"note": "Verified zero speech dialogue in ground truth"},
                )
            else:
                ref_text = " ".join(t.reference_text for t in gt_tr)
                hyp_text = " ".join(pred_tr_texts)
                wer_val, wer_meta = compute_wer(ref_text, hyp_text)
                video_metrics["asr_wer"] = MetricResult(
                    name="asr_wer",
                    status=MetricStatus.AVAILABLE,
                    value=wer_val,
                    sample_size=wer_meta["reference_words"],
                    metadata=wer_meta,
                )
        elif video_ann.transcripts.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["asr_wer"] = MetricResult(
                name="asr_wer",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=(
                    video_ann.transcripts.rationale or "Audio marked not applicable"
                ),
            )
        else:
            unavailable_modalities.append("transcripts")
            video_metrics["asr_wer"] = MetricResult(
                name="asr_wer",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"Transcripts status is {video_ann.transcripts.status.value}",
            )

        # 4. Temporal Events Evaluation
        if video_ann.events.status == AnnotationStatus.ANNOTATED:
            gt_events = video_ann.events.items
            pred_events_raw = video_preds.get("events", [])
            pred_events: list[EventPrediction] = []
            for ev in pred_events_raw:
                if isinstance(ev, EventPrediction):
                    pred_events.append(ev)
                elif isinstance(ev, dict):
                    pred_events.append(EventPrediction(**ev))

            ev_metrics = evaluate_temporal_events(
                ground_truth=gt_events,
                predictions=pred_events,
                tiou_threshold=self.config.get("tiou_threshold", 0.5),
            )
            video_metrics.update(ev_metrics)
        elif video_ann.events.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["event_f1"] = MetricResult(
                name="event_f1",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.events.rationale or "Events marked not applicable",
            )
        else:
            unavailable_modalities.append("events")
            video_metrics["event_f1"] = MetricResult(
                name="event_f1",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"Events status is {video_ann.events.status.value}",
            )

        # 5. QA Abstention Evaluation
        if video_ann.qa.status == AnnotationStatus.ANNOTATED:
            gt_qa = video_ann.qa.items
            pred_qa_raw = video_preds.get("qa", [])
            pred_qa: list[QAPrediction] = []
            for q in pred_qa_raw:
                if isinstance(q, QAPrediction):
                    pred_qa.append(q)
                elif isinstance(q, dict):
                    pred_qa.append(QAPrediction(**q))

            qa_metrics = evaluate_qa_abstention(ground_truth=gt_qa, predictions=pred_qa)
            video_metrics.update(qa_metrics)
        elif video_ann.qa.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["qa_abstention_accuracy"] = MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.qa.rationale or "QA marked not applicable",
            )
        else:
            unavailable_modalities.append("qa")
            video_metrics["qa_abstention_accuracy"] = MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"QA status is {video_ann.qa.status.value}",
            )

        return VideoEvaluationSummary(
            video_id=vid,
            source_identifier=video_ann.source_identifier,
            metrics=video_metrics,
            unavailable_modalities=unavailable_modalities,
            errors=errors,
        )

    def run(self) -> EvaluationResult:
        """Executes full evaluation over all videos in the dataset."""
        per_video_summaries: list[VideoEvaluationSummary] = []
        failures: list[str] = []

        for video_ann in self.dataset.videos:
            try:
                summary = self.evaluate_video(video_ann)
                per_video_summaries.append(summary)
            except Exception as ex:
                logger.exception("Error evaluating video %s", video_ann.video_id)
                failures.append(f"Video {video_ann.video_id}: {str(ex)}")

        # Aggregate across videos for available metrics
        aggregate_metrics = self._aggregate_metrics(per_video_summaries)

        return EvaluationResult(
            evaluation_id=uuid4(),
            timestamp=datetime.now(UTC),
            repo_commit=get_git_commit(),
            dataset_version=self.dataset.schema_version,
            configuration=self.config,
            provider_info={
                "runner_version": "1.0.0",
                "execution_mode": "fixture_based",
            },
            aggregate_metrics=aggregate_metrics,
            per_video_results=per_video_summaries,
            failures=failures,
        )

    def _aggregate_metrics(
        self,
        summaries: list[VideoEvaluationSummary],
    ) -> dict[str, MetricResult]:
        """Aggregates metrics across videos preserving sample sizes and uncertainty."""
        metric_groups: dict[str, list[MetricResult]] = {}
        for s in summaries:
            for name, res in s.metrics.items():
                if res.status == MetricStatus.AVAILABLE and res.value is not None:
                    metric_groups.setdefault(name, []).append(res)

        aggregates: dict[str, MetricResult] = {}
        for name, results in metric_groups.items():
            total_n = sum(r.sample_size for r in results)
            # Weighted average by sample size if available, otherwise simple mean
            if total_n > 0:
                weighted_sum = sum(r.value * r.sample_size for r in results if r.value is not None)
                agg_val = weighted_sum / total_n
            else:
                vals = [r.value for r in results if r.value is not None]
                agg_val = sum(vals) / len(vals) if vals else 0.0

            aggregates[name] = MetricResult(
                name=name,
                status=MetricStatus.AVAILABLE,
                value=round(agg_val, 4),
                sample_size=total_n,
                metadata={"videos_contributing": len(results)},
            )
        return aggregates


def format_cli_summary(report: EvaluationResult) -> str:
    """Formats an EvaluationResult into a clean, human-readable terminal table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"{'VIDEX ACCURACY & INTEGRITY EVALUATION REPORT':^90}")
    lines.append("=" * 90)
    lines.append(
        f"Evaluation ID: {report.evaluation_id}  |  Commit: {report.repo_commit or 'unknown'}"
    )
    lines.append(
        f"Dataset Version: {report.dataset_version}  |  Executed: {report.timestamp.isoformat()}"
    )
    lines.append("-" * 90)
    lines.append(f"{'Source Video':<20} | {'Metric Name':<28} | {'Status':<12} | {'Value / N'}")
    lines.append("-" * 90)

    for v in report.per_video_results:
        first = True
        for mname, mres in v.metrics.items():
            src_str = v.source_identifier if first else ""
            first = False
            if mres.status == MetricStatus.AVAILABLE and mres.value is not None:
                val_str = f"{mres.value:.4f} (N={mres.sample_size})"
                if mres.confidence_interval:
                    ci_l, ci_u = mres.confidence_interval
                    val_str += f" [CI: {ci_l:.2f}-{ci_u:.2f}]"
            elif mres.status == MetricStatus.NOT_APPLICABLE:
                val_str = f"N/A ({mres.reason_if_unavailable})"
            else:
                val_str = f"UNAVAILABLE ({mres.reason_if_unavailable})"

            lines.append(f"{src_str:<20} | {mname:<28} | {mres.status.value:<12} | {val_str}")
        lines.append("-" * 90)

    lines.append("AGGREGATE METRICS SUMMARY:")
    for mname, mres in report.aggregate_metrics.items():
        lines.append(f"  - {mname:<25}: {mres.value:.4f} (Total N={mres.sample_size})")

    if report.failures:
        lines.append("\nFAILURES / ALERTS:")
        for f in report.failures:
            lines.append(f"  [!] {f}")

    lines.append("=" * 90)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running VIDEX evaluation against annotation datasets."""
    parser = argparse.ArgumentParser(description="VIDEX Accuracy-First Evaluation Harness")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to annotation dataset JSON file",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions JSON file (optional)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Destination path for output EvaluationResult JSON",
    )
    args = parser.parse_args()

    dataset = load_annotation_dataset(args.annotations)
    predictions_data: dict[str, Any] = {}
    if args.predictions and args.predictions.is_file():
        with open(args.predictions, encoding="utf-8") as f:
            predictions_data = json.load(f)

    harness = EvaluationHarness(dataset=dataset, predictions_by_video=predictions_data)
    result = harness.run()

    print(format_cli_summary(result))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"Machine-readable results saved to {args.output_json}")


if __name__ == "__main__":
    main()
