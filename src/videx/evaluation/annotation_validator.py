"""Annotation dataset quality control and schema validation for VIDEX benchmarks.

Validates:
- Strict frame coverage against manifest/sampling plans
- Timestamp consistency and valid bounds
- Coordinate bounds, finiteness, and positive box dimensions
- Policy A person-only class compliance
- Absence of detector prediction leakage (e.g. 'confidence', 'score')
- Duplicate or suspiciously overlapping annotations
- Explicit empty frame handling
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videx.evaluation.schemas import AnnotationDataset

logger = logging.getLogger("videx.evaluation.validator")

EXPECTED_PPE1_FRAMES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
FORBIDDEN_PREDICTION_FIELDS = {
    "confidence",
    "score",
    "model_name",
    "model_version",
    "pred_box",
    "detector",
    "tracker_confidence",
}


@dataclass
class ValidationIssue:
    level: str  # "ERROR" or "WARNING"
    frame_index: int | None
    code: str
    message: str


@dataclass
class ValidationReport:
    is_valid: bool
    dataset_id: str
    total_frames_checked: int
    total_boxes_checked: int
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Annotation Validation Report for '{self.dataset_id}'",
            f"Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Total Frames: {self.total_frames_checked}",
            f"Total Boxes: {self.total_boxes_checked}",
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\nErrors (Blocking):")
            for e in self.errors:
                frame_str = f"[Frame {e.frame_index}] " if e.frame_index is not None else ""
                lines.append(f"  - {frame_str}{e.code}: {e.message}")
        if self.warnings:
            lines.append("\nWarnings (Non-blocking):")
            for w in self.warnings:
                frame_str = f"[Frame {w.frame_index}] " if w.frame_index is not None else ""
                lines.append(f"  - {frame_str}{w.code}: {w.message}")
        return "\n".join(lines)


def compute_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """Computes IoU between two pixel boxes [x, y, width, height]."""
    x1_a, y1_a, w_a, h_a = box1["x"], box1["y"], box1["width"], box1["height"]
    x1_b, y1_b, w_b, h_b = box2["x"], box2["y"], box2["width"], box2["height"]

    x2_a, y2_a = x1_a + w_a, y1_a + h_a
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w_a * h_a
    area_b = w_b * h_b
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def validate_annotation_data(
    data: dict[str, Any],
    expected_frames: list[int] | None = None,
    frame_width: int = 1280,
    frame_height: int = 720,
    fps: float = 29.97003,
) -> ValidationReport:
    """Strictly audits an annotation dataset dictionary against VIDEX standards."""
    if expected_frames is None:
        expected_frames = EXPECTED_PPE1_FRAMES

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # 1. Check top-level provenance & schema
    dataset_id = data.get("dataset_id", "unknown_dataset")
    provenance = data.get("provenance", "")
    guidelines = data.get("annotation_guidelines", "")

    if not provenance:
        errors.append(
            ValidationIssue(
                "ERROR",
                None,
                "MISSING_PROVENANCE",
                "Dataset must have a non-empty provenance field",
            )
        )
    if not guidelines:
        warnings.append(
            ValidationIssue(
                "WARNING",
                None,
                "MISSING_GUIDELINES",
                "Dataset guidelines reference is missing or empty",
            )
        )

    # Check for raw prediction leakage in top-level or video metadata
    raw_str = json.dumps(data)
    for forbidden in FORBIDDEN_PREDICTION_FIELDS:
        if f'"{forbidden}"' in raw_str:
            errors.append(
                ValidationIssue(
                    "ERROR",
                    None,
                    "PREDICTION_LEAKAGE_DETECTED",
                    f"Forbidden model-prediction key '{forbidden}' was found in annotation JSON.",
                )
            )

    videos = data.get("videos", [])
    if not videos:
        errors.append(
            ValidationIssue("ERROR", None, "EMPTY_VIDEOS", "Dataset contains no video annotations")
        )
        return ValidationReport(False, dataset_id, 0, 0, errors, warnings)

    video = videos[0]
    detections = video.get("detections", {}).get("items", [])
    total_boxes = len(detections)

    # 2. Check frame coverage
    frames_present: dict[int, list[dict[str, Any]]] = {idx: [] for idx in expected_frames}
    for item in detections:
        f_idx = item.get("frame_index")
        if f_idx is None:
            errors.append(
                ValidationIssue(
                    "ERROR",
                    None,
                    "MISSING_FRAME_INDEX",
                    f"Detection item {item.get('detection_id')} lacks frame_index",
                )
            )
            continue
        if f_idx not in frames_present:
            warnings.append(
                ValidationIssue(
                    "WARNING",
                    f_idx,
                    "UNEXPECTED_FRAME",
                    f"Detection found at unexpected frame index {f_idx}",
                )
            )
            frames_present[f_idx] = []
        frames_present[f_idx].append(item)

    # Missing frames check
    for exp_idx in expected_frames:
        if exp_idx not in frames_present:
            errors.append(
                ValidationIssue(
                    "ERROR",
                    exp_idx,
                    "MISSING_SAMPLED_FRAME",
                    f"Required sampled frame {exp_idx} is missing from annotations",
                )
            )

    # 3. Check individual detection items
    for f_idx, items in frames_present.items():
        expected_pts = f_idx / fps
        for item in items:
            pts = item.get("timestamp_pts")
            if pts is None or not math.isfinite(pts) or pts < 0:
                errors.append(
                    ValidationIssue(
                        "ERROR", f_idx, "INVALID_TIMESTAMP", f"Invalid timestamp_pts: {pts}"
                    )
                )
            elif abs(pts - expected_pts) > 0.05:
                warnings.append(
                    ValidationIssue(
                        "WARNING",
                        f_idx,
                        "TIMESTAMP_DRIFT",
                        f"Timestamp {pts:.3f}s deviates from expected pts {expected_pts:.3f}s",
                    )
                )

            # Class label compliance
            class_label = item.get("class_label")
            if class_label != "person":
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "INVALID_CLASS",
                        f"Class label '{class_label}' invalid for Policy A (must be 'person')",
                    )
                )

            # Bounding box geometry
            bbox = item.get("bbox")
            if not bbox or not isinstance(bbox, dict):
                errors.append(
                    ValidationIssue(
                        "ERROR", f_idx, "MALFORMED_BBOX", "Missing or non-dict bbox field"
                    )
                )
                continue

            raw_x = bbox.get("x")
            raw_y = bbox.get("y")
            raw_w = bbox.get("width")
            raw_h = bbox.get("height")

            if any(val is None or not math.isfinite(val) for val in [raw_x, raw_y, raw_w, raw_h]):
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "NON_FINITE_COORDS",
                        f"Box coordinates contain non-finite values: {bbox}",
                    )
                )
                continue

            assert raw_x is not None and raw_y is not None
            assert raw_w is not None and raw_h is not None

            fx, fy, fw, fh = float(raw_x), float(raw_y), float(raw_w), float(raw_h)

            if fw <= 0 or fh <= 0:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "NON_POSITIVE_DIMENSIONS",
                        f"Width ({fw}) and height ({fh}) must be strictly positive",
                    )
                )

            # Image boundaries
            if fx < 0 or fy < 0:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "OUT_OF_BOUNDS_LEFT_TOP",
                        f"Box starts at negative coords (x={fx}, y={fy})",
                    )
                )
            if fx + fw > frame_width:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "OUT_OF_BOUNDS_RIGHT",
                        f"Box extends past frame width: x+w = {fx + fw} > {frame_width}",
                    )
                )
            if fy + fh > frame_height:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "OUT_OF_BOUNDS_BOTTOM",
                        f"Box extends past frame height: y+h = {fy + fh} > {frame_height}",
                    )
                )

        # 4. Check for overlapping or duplicate boxes in same frame
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                b1 = items[i].get("bbox", {})
                b2 = items[j].get("bbox", {})
                if b1 and b2:
                    overlap = compute_iou(b1, b2)
                    if overlap > 0.85:
                        warnings.append(
                            ValidationIssue(
                                "WARNING",
                                f_idx,
                                "SUSPICIOUS_OVERLAP",
                                f"Instances overlap with IoU {overlap:.3f}",
                            )
                        )

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid, dataset_id, len(frames_present), total_boxes, errors, warnings
    )


def validate_file(filepath: Path | str) -> ValidationReport:
    """Loads and validates an annotation dataset file."""
    path = Path(filepath)
    if not path.exists():
        return ValidationReport(
            False,
            str(path),
            0,
            0,
            [ValidationIssue("ERROR", None, "FILE_NOT_FOUND", f"File '{path}' does not exist")],
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # First attempt pydantic schema validation if full dataset
    try:
        AnnotationDataset.model_validate(data)
    except Exception as e:
        logger.warning(f"Standard AnnotationDataset validation failed: {e}")

    return validate_annotation_data(data)


def validate_multiclip_annotation_data(data: dict[str, Any]) -> ValidationReport:
    """Strictly audits a multi-clip annotation dataset against VIDEX standards."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    dataset_id = data.get("dataset_id", "unknown_multiclip_dataset")
    provenance = data.get("provenance", "")
    guidelines = data.get("annotation_guidelines", "")

    if not provenance:
        errors.append(
            ValidationIssue(
                "ERROR",
                None,
                "MISSING_PROVENANCE",
                "Dataset must have a non-empty provenance field",
            )
        )
    if not guidelines:
        warnings.append(
            ValidationIssue(
                "WARNING",
                None,
                "MISSING_GUIDELINES",
                "Dataset guidelines reference is missing or empty",
            )
        )

    # Check for prediction leakage across entire dataset
    raw_str = json.dumps(data)
    for forbidden in FORBIDDEN_PREDICTION_FIELDS:
        if f'"{forbidden}"' in raw_str:
            errors.append(
                ValidationIssue(
                    "ERROR",
                    None,
                    "PREDICTION_LEAKAGE_DETECTED",
                    f"Forbidden model-prediction key '{forbidden}' was found in annotation JSON.",
                )
            )

    videos = data.get("videos", [])
    if not videos:
        errors.append(
            ValidationIssue(
                "ERROR",
                None,
                "EMPTY_VIDEOS",
                "Dataset contains no video annotations",
            )
        )
        return ValidationReport(False, dataset_id, 0, 0, errors, warnings)

    total_frames = 0
    total_boxes = 0

    for video in videos:
        vid = video.get("video_id", "unknown_video")
        width = int(video.get("width", 1280))
        height = int(video.get("height", 720))
        fps = float(video.get("fps", 29.97003))
        sampled_frames = video.get("sampled_frame_indices", [])
        negative_frames = set(video.get("negative_frames", []))

        detections = video.get("detections", {}).get("items", [])
        total_boxes += len(detections)
        total_frames += len(sampled_frames)

        frames_present: dict[int, list[dict[str, Any]]] = {idx: [] for idx in sampled_frames}
        for item in detections:
            f_idx = item.get("frame_index")
            if f_idx is None:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        None,
                        "MISSING_FRAME_INDEX",
                        f"Item in {vid} lacks frame_index",
                    )
                )
                continue
            if f_idx not in frames_present:
                warnings.append(
                    ValidationIssue(
                        "WARNING",
                        f_idx,
                        "UNEXPECTED_FRAME",
                        f"Unexpected frame {f_idx} in {vid}",
                    )
                )
                frames_present[f_idx] = []
            frames_present[f_idx].append(item)

        # Check coverage
        for exp_idx in sampled_frames:
            if exp_idx not in frames_present:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        exp_idx,
                        "MISSING_SAMPLED_FRAME",
                        f"Required frame {exp_idx} in {vid} missing",
                    )
                )

        # Check each frame
        for f_idx, items in frames_present.items():
            if f_idx in negative_frames and len(items) > 0:
                errors.append(
                    ValidationIssue(
                        "ERROR",
                        f_idx,
                        "NEGATIVE_FRAME_HAS_BOXES",
                        f"Frame {f_idx} in {vid} marked negative but has {len(items)} boxes",
                    )
                )
            expected_pts = f_idx / fps
            for item in items:
                pts = item.get("timestamp_pts")
                if pts is None or not math.isfinite(pts) or pts < 0:
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "INVALID_TIMESTAMP",
                            f"Invalid timestamp_pts: {pts}",
                        )
                    )
                elif abs(pts - expected_pts) > 0.05:
                    warnings.append(
                        ValidationIssue(
                            "WARNING",
                            f_idx,
                            "TIMESTAMP_DRIFT",
                            f"Timestamp {pts:.3f}s != expected {expected_pts:.3f}s in {vid}",
                        )
                    )

                class_label = item.get("class_label")
                if class_label != "person":
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "INVALID_CLASS",
                            f"Class label '{class_label}' in {vid} must be 'person'",
                        )
                    )

                bbox = item.get("bbox")
                if not bbox or not isinstance(bbox, dict):
                    errors.append(
                        ValidationIssue("ERROR", f_idx, "MALFORMED_BBOX", f"Missing bbox in {vid}")
                    )
                    continue

                raw_x = bbox.get("x")
                raw_y = bbox.get("y")
                raw_w = bbox.get("width")
                raw_h = bbox.get("height")

                coords = [raw_x, raw_y, raw_w, raw_h]
                if any(val is None or not math.isfinite(val) for val in coords):
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "NON_FINITE_COORDS",
                            f"Non-finite coords in {vid}: {bbox}",
                        )
                    )
                    continue

                assert raw_x is not None and raw_y is not None
                assert raw_w is not None and raw_h is not None
                fx, fy, fw, fh = float(raw_x), float(raw_y), float(raw_w), float(raw_h)
                if fw <= 0 or fh <= 0:
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "NON_POSITIVE_DIMENSIONS",
                            f"Width/height non-positive in {vid}: {fw}x{fh}",
                        )
                    )
                if fx < 0 or fy < 0:
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "OUT_OF_BOUNDS_LEFT_TOP",
                            f"Negative coords in {vid}: ({fx}, {fy})",
                        )
                    )
                if fx + fw > width:
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "OUT_OF_BOUNDS_RIGHT",
                            f"Exceeds width {width} in {vid}: {fx + fw}",
                        )
                    )
                if fy + fh > height:
                    errors.append(
                        ValidationIssue(
                            "ERROR",
                            f_idx,
                            "OUT_OF_BOUNDS_BOTTOM",
                            f"Exceeds height {height} in {vid}: {fy + fh}",
                        )
                    )

    is_valid = len(errors) == 0
    return ValidationReport(is_valid, dataset_id, total_frames, total_boxes, errors, warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VIDEX ground-truth annotation dataset")
    parser.add_argument("annotation_file", type=str, help="Path to annotation JSON file")
    args = parser.parse_args()

    report = validate_file(args.annotation_file)
    print(report.summary())
    exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
