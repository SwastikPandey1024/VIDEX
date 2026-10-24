"""Unit tests for the standardized benchmarking framework (Priority 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videx.evaluation.benchmarking import (
    BenchmarkReport,
    BenchmarkRunMetadata,
    SystemEnvironmentMetadata,
    calculate_latency_statistics,
    collect_system_environment,
    sha256_file,
)


class TestLatencyStatisticsCalculation:
    """Test calculations and invariants for latency statistics."""

    def test_single_measurement(self) -> None:
        stats = calculate_latency_statistics([100.0])
        assert stats.count == 1
        assert stats.mean_ms == 100.0
        assert stats.median_ms == 100.0
        assert stats.p95_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.std_dev_ms == 0.0
        assert stats.throughput_fps == 10.0

    def test_multi_measurement_statistics(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_latency_statistics(latencies)
        assert stats.count == 5
        assert stats.mean_ms == 30.0
        assert stats.median_ms == 30.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0
        # Sample standard deviation for [10, 20, 30, 40, 50] is sqrt(250) ≈ 15.81
        assert stats.std_dev_ms == 15.81
        # p95 for 5 elements
        assert stats.p95_ms == 48.0
        assert stats.throughput_fps == round(1000.0 / 30.0, 2)

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot calculate statistics on empty latency list"):
            calculate_latency_statistics([])

    def test_negative_values_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Latency values must be non-negative"):
            calculate_latency_statistics([10.0, -5.0, 20.0])


class TestSystemEnvironmentCapture:
    """Test host environment metadata capture."""

    def test_collect_system_environment(self) -> None:
        env = collect_system_environment()
        assert isinstance(env, SystemEnvironmentMetadata)
        assert len(env.cpu_model) > 0
        assert env.logical_cores >= 1
        assert env.ram_total_gb > 0
        assert len(env.os_platform) > 0
        assert len(env.python_version) > 0
        assert env.thread_count >= 1


class TestBenchmarkReportIntegrity:
    """Test report creation, serialization, JSON/Markdown output, and file hashing."""

    @pytest.fixture
    def sample_report(self) -> BenchmarkReport:
        sys_env = SystemEnvironmentMetadata(
            cpu_model="Test CPU",
            logical_cores=8,
            physical_cores=4,
            ram_total_gb=16.0,
            os_platform="Windows",
            os_version="10.0.19045",
            python_version="3.13.5",
            torch_version="2.6.0",
            onnxruntime_version="1.20.1",
            opencv_version="4.10.0",
            thread_count=8,
        )
        meta = BenchmarkRunMetadata(
            benchmark_id="bench_test_001",
            timestamp_utc="2026-10-24T12:00:00Z",
            backend="pytorch",
            model_path="models/yolo26n.pt",
            model_hash="abc123hash",
            clip_path="Sample_Videos/ppe-1.mp4",
            clip_hash="def456hash",
            resolution=(1280, 720),
            fps=29.97,
            frame_count=343,
            conf_threshold=0.25,
            iou_threshold=0.45,
            stages_measured=["detection", "tracking"],
            cold_init_ms=450.5,
            warmup_ms=85.2,
            num_measured_runs=50,
            system_metadata=sys_env,
        )
        stage_stats = {
            "detection": calculate_latency_statistics([15.0, 16.0, 14.5, 15.5]),
            "tracking": calculate_latency_statistics([2.0, 2.5, 2.1, 2.4]),
        }
        e2e_stats = calculate_latency_statistics([17.0, 18.5, 16.6, 17.9])

        return BenchmarkReport(
            metadata=meta,
            stage_statistics=stage_stats,
            end_to_end_statistics=e2e_stats,
            error_count=0,
            skipped_frames=0,
            notes=["Unit test sample run"],
        )

    def test_json_serialization(self, sample_report: BenchmarkReport, tmp_path: Path) -> None:
        json_path = tmp_path / "test_report.json"
        sample_report.save_json(json_path)
        assert json_path.is_file()

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["metadata"]["benchmark_id"] == "bench_test_001"
        assert loaded["metadata"]["backend"] == "pytorch"
        assert "detection" in loaded["stage_statistics"]
        assert loaded["stage_statistics"]["detection"]["count"] == 4
        assert loaded["end_to_end_statistics"]["mean_ms"] > 0

    def test_markdown_rendering(self, sample_report: BenchmarkReport, tmp_path: Path) -> None:
        md_path = tmp_path / "test_report.md"
        sample_report.save_markdown(md_path)
        assert md_path.is_file()

        content = md_path.read_text(encoding="utf-8")
        assert "# Benchmark Report: bench_test_001" in content
        assert "## 1. System & Runtime Environment" in content
        assert "## 3. Performance Summary" in content
        assert "| `detection` |" in content
        assert "| **`End-to-End`** |" in content

    def test_sha256_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "sample.txt"
        test_file.write_text("VIDEX Benchmark Test\n", encoding="utf-8")
        h = sha256_file(test_file)
        assert len(h) == 64
        # Hash for non-existent file
        assert sha256_file(tmp_path / "does_not_exist.bin") == "FILE_NOT_FOUND"
