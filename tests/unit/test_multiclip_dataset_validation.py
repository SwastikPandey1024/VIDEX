"""Unit tests for VIDEX Versioned Multi-Clip Evaluation Dataset (v4.0.0).

Validates:
  1. Top-level schema conformity, provenance, and guideline references.
  2. Multi-clip coverage (5 representative clips, 100 sampled frames, 170 boxes).
  3. Strict Policy A compliance (visible person extents, edge clipping, coordinate validity).
  4. Explicit negative control clip verification (sample.mp4 contains zero person instances).
  5. Absence of model prediction leakage (forbidden keys like confidence, score, etc.).
  6. Dataset manifest and fixture hash integrity (including preservation of historical v1, v2, v3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from videx.evaluation.annotation_validator import validate_multiclip_annotation_data

FIXTURE_PATH = Path("tests/fixtures/evaluation/multiclip_annotations_v4_independent.json")
MANIFEST_PATH = Path("tests/fixtures/evaluation/multiclip_v4_manifest.json")

HISTORICAL_FIXTURE_HASHES = {
    "v1_fixture": (
        "tests/fixtures/evaluation/ppe1_annotations.json",
        "edd062d360c42850a55b7de94c5f8febd475683ef0fa53f2d9b5018d27af934b",
    ),
    "v2_fixture": (
        "tests/fixtures/evaluation/ppe1_annotations_v2.json",
        "8fcd58897e00b500e73a89e5ab7e91b2022fc3efbc3044687385e44ab526b4b7",
    ),
    "v3_fixture": (
        "tests/fixtures/evaluation/ppe1_annotations_v3_independent.json",
        "6136e752850c67dd8769157a77bb7ade9b94a50b35e66aa43a8df217afc57779",
    ),
}


@pytest.fixture(scope="module")
def dataset_v4() -> dict[str, Any]:
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"Multi-clip v4 fixture not found at {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


class TestMultiClipDatasetIntegrity:
    def test_top_level_metadata(self, dataset_v4: dict[str, Any]) -> None:
        assert dataset_v4["schema_version"] == "1.0.0"
        assert dataset_v4["dataset_id"] == "videx_multiclip_v4_independent"
        assert dataset_v4["dataset_version"] == "4.0.0"
        assert "provenance" in dataset_v4 and len(dataset_v4["provenance"]) > 0
        assert "annotation_guidelines" in dataset_v4

    def test_dataset_coverage_counts(self, dataset_v4: dict[str, Any]) -> None:
        assert dataset_v4["total_clips"] == 5
        assert dataset_v4["total_sampled_frames"] == 100
        assert dataset_v4["total_ground_truth_boxes"] == 170

    def test_clip_level_structure(self, dataset_v4: dict[str, Any]) -> None:
        expected_clips = {
            "ppe-1.mp4": 20,
            "ppe-2.mp4": 20,
            "ppe-3.mp4": 20,
            "people.mp4": 25,
            "sample.mp4": 15,
        }
        assert len(dataset_v4["videos"]) == len(expected_clips)
        found_clips = {}
        for video in dataset_v4["videos"]:
            src = video["source_identifier"]
            found_clips[src] = video["sampled_frames_count"]
            assert len(video["sampled_frame_indices"]) == expected_clips[src]
            assert video["width"] > 0
            assert video["height"] > 0
            assert video["fps"] > 0
            assert "sha256" in video and len(video["sha256"]) == 64
        assert found_clips == expected_clips

    def test_negative_control_clip(self, dataset_v4: dict[str, Any]) -> None:
        sample_video = next(
            v for v in dataset_v4["videos"] if v["source_identifier"] == "sample.mp4"
        )
        assert len(sample_video["detections"]["items"]) == 0, "sample.mp4 must have 0 person dets"
        assert len(sample_video["negative_frames"]) == 15, "all 15 frames are negative frames"
        assert len(sample_video["tracks"]["items"]) == 0

    def test_policy_a_coordinate_validity(self, dataset_v4: dict[str, Any]) -> None:
        for video in dataset_v4["videos"]:
            max_w = video["width"]
            max_h = video["height"]
            for det in video["detections"]["items"]:
                assert det["class_label"] == "person"
                bbox = det["bbox"]
                assert bbox["coordinate_type"] == "pixel"
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                assert x >= 0.0, f"x negative: {x}"
                assert y >= 0.0, f"y negative: {y}"
                assert w > 0.0, f"width non-positive: {w}"
                assert h > 0.0, f"height non-positive: {h}"
                assert x + w <= max_w, f"x+w ({x+w}) exceeds frame width {max_w}"
                assert y + h <= max_h, f"y+h ({y+h}) exceeds frame height {max_h}"

    def test_validator_passes_zero_errors(self, dataset_v4: dict[str, Any]) -> None:
        report = validate_multiclip_annotation_data(dataset_v4)
        assert report.is_valid, f"Validation failed: {report.errors}"
        assert len(report.errors) == 0
        assert report.total_frames_checked == 100
        assert report.total_boxes_checked == 170

    def test_manifest_hash_integrity(self) -> None:
        if not MANIFEST_PATH.is_file():
            pytest.skip(f"Manifest not found: {MANIFEST_PATH}")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        raw = FIXTURE_PATH.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        assert actual_hash == manifest["sha256"], "Fixture hash does not match manifest"

    def test_historical_fixture_hashes_preserved(self) -> None:
        for name, (path_str, expected_hash) in HISTORICAL_FIXTURE_HASHES.items():
            path = Path(path_str)
            assert path.is_file(), f"Historical fixture missing: {path}"
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual_hash == expected_hash, (
                f"Historical fixture {name} modified! "
                f"Expected {expected_hash}, got {actual_hash}"
            )
