"""Standardized benchmarking and reproducibility framework for VIDEX.

Captures system hardware, runtime environment, model/clip hashes, separated
stage latencies, and descriptive statistics (mean, median, p95, stdev, throughput).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:
    psutil = None


@dataclass(frozen=True)
class SystemEnvironmentMetadata:
    """Hardware and runtime environment capture."""

    cpu_model: str
    logical_cores: int
    physical_cores: int | None
    ram_total_gb: float
    os_platform: str
    os_version: str
    python_version: str
    torch_version: str | None
    onnxruntime_version: str | None
    opencv_version: str | None
    thread_count: int


@dataclass(frozen=True)
class LatencyStatistics:
    """Descriptive statistics for a measured pipeline stage or end-to-end run."""

    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    throughput_fps: float


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    """Configuration, scope, and integrity hashes for a benchmark run."""

    benchmark_id: str
    timestamp_utc: str
    backend: str
    model_path: str
    model_hash: str
    clip_path: str
    clip_hash: str
    resolution: tuple[int, int]
    fps: float
    frame_count: int
    conf_threshold: float
    iou_threshold: float
    stages_measured: list[str]
    cold_init_ms: float
    warmup_ms: float
    num_measured_runs: int
    system_metadata: SystemEnvironmentMetadata


@dataclass
class BenchmarkReport:
    """Complete machine-readable benchmark report."""

    metadata: BenchmarkRunMetadata
    stage_statistics: dict[str, LatencyStatistics]
    end_to_end_statistics: LatencyStatistics
    error_count: int = 0
    skipped_frames: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializes report to dictionary format."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path) -> None:
        """Saves report to disk with deterministic newline formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8", newline="\n")

    def to_markdown(self) -> str:
        """Renders report as structured Markdown."""
        meta = self.metadata
        sys_m = meta.system_metadata

        lines = [
            f"# Benchmark Report: {meta.benchmark_id}",
            "",
            "## 1. System & Runtime Environment",
            f"- **CPU Model:** {sys_m.cpu_model}",
            f"- **Cores:** {sys_m.logical_cores} logical ({sys_m.physical_cores} physical)",
            f"- **Total RAM:** {sys_m.ram_total_gb:.1f} GB",
            f"- **OS:** {sys_m.os_platform} {sys_m.os_version}",
            f"- **Python:** {sys_m.python_version}",
            f"- **PyTorch:** {sys_m.torch_version or 'N/A'}",
            f"- **ONNX Runtime:** {sys_m.onnxruntime_version or 'N/A'}",
            f"- **OpenCV:** {sys_m.opencv_version or 'N/A'}",
            f"- **Execution Threads:** {sys_m.thread_count}",
            "",
            "## 2. Benchmark Configuration & Artifacts",
            f"- **Timestamp (UTC):** {meta.timestamp_utc}",
            f"- **Backend:** `{meta.backend}`",
            f"- **Model Path:** `{meta.model_path}`",
            f"- **Model SHA-256:** `{meta.model_hash}`",
            f"- **Clip Path:** `{meta.clip_path}`",
            f"- **Clip SHA-256:** `{meta.clip_hash}`",
            f"- **Resolution:** {meta.resolution[0]}x{meta.resolution[1]} @ {meta.fps:.2f} fps",
            f"- **Total Frames:** {meta.frame_count}",
            f"- **Confidence Threshold:** {meta.conf_threshold}",
            f"- **NMS/IoU Threshold:** {meta.iou_threshold}",
            f"- **Cold Init Time:** {meta.cold_init_ms:.2f} ms",
            f"- **Warmup Time:** {meta.warmup_ms:.2f} ms",
            f"- **Measured Runs:** {meta.num_measured_runs}",
            "",
            "## 3. Performance Summary",
            "| Stage | Count | Mean (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) | Std Dev (ms) | Throughput (FPS) |",  # noqa: E501
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for stage, stats in self.stage_statistics.items():
            lines.append(
                f"| `{stage}` | {stats.count} | {stats.mean_ms:.2f} | {stats.median_ms:.2f} | "
                f"{stats.p95_ms:.2f} | {stats.min_ms:.2f} | {stats.max_ms:.2f} | "
                f"{stats.std_dev_ms:.2f} | {stats.throughput_fps:.1f} |"
            )

        e2e = self.end_to_end_statistics
        lines.append(
            f"| **`End-to-End`** | {e2e.count} | **{e2e.mean_ms:.2f}** | **{e2e.median_ms:.2f}** | "
            f"**{e2e.p95_ms:.2f}** | **{e2e.min_ms:.2f}** | **{e2e.max_ms:.2f}** | "
            f"**{e2e.std_dev_ms:.2f}** | **{e2e.throughput_fps:.1f}** |"
        )

        lines.extend(
            [
                "",
                "## 4. Run Observations & Errors",
                f"- **Error Count:** {self.error_count}",
                f"- **Skipped Frames:** {self.skipped_frames}",
            ]
        )

        if self.notes:
            lines.append("- **Notes:**")
            for note in self.notes:
                lines.append(f"  - {note}")
        else:
            lines.append("- **Notes:** None")

        return "\n".join(lines)

    def save_markdown(self, path: Path) -> None:
        """Saves structured markdown report to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown() + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    """Computes SHA-256 hash of a file, returning hex digest."""
    if not path.is_file():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def calculate_latency_statistics(latencies_ms: list[float]) -> LatencyStatistics:
    """Calculates descriptive statistics for a sequence of latencies in milliseconds.

    Args:
        latencies_ms: List of non-negative latency measurements in milliseconds.

    Returns:
        LatencyStatistics containing count, mean, median, p95, min, max, stdev, throughput.

    Raises:
        ValueError: If latencies_ms is empty or contains negative values.
    """
    if not latencies_ms:
        raise ValueError("Cannot calculate statistics on empty latency list")

    for v in latencies_ms:
        if v < 0:
            raise ValueError(f"Latency values must be non-negative, got {v}")

    count = len(latencies_ms)
    arr = np.array(latencies_ms, dtype=np.float64)

    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    p95_v = float(np.percentile(arr, 95))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    std_v = float(np.std(arr, ddof=1)) if count > 1 else 0.0

    throughput = (1000.0 / mean_v) if mean_v > 0.0 else 0.0

    return LatencyStatistics(
        count=count,
        mean_ms=round(mean_v, 2),
        median_ms=round(median_v, 2),
        p95_ms=round(p95_v, 2),
        min_ms=round(min_v, 2),
        max_ms=round(max_v, 2),
        std_dev_ms=round(std_v, 2),
        throughput_fps=round(throughput, 2),
    )


def collect_system_environment() -> SystemEnvironmentMetadata:
    """Collects host machine hardware and Python runtime environment."""
    # CPU Model
    cpu_model = platform.processor() or "Unknown CPU"
    # Logical and physical cores
    logical_cores = (psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()) or 1
    physical_cores = psutil.cpu_count(logical=False) if psutil is not None else None

    # RAM in GB
    if psutil is not None:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = round(ram_bytes / (1024**3), 2)
    else:
        ram_gb = 0.0
        try:
            sysconf = getattr(os, "sysconf", None)
            sysconf_names = getattr(os, "sysconf_names", {})
            if (
                sysconf is not None
                and "SC_PAGE_SIZE" in sysconf_names
                and "SC_PHYS_PAGES" in sysconf_names
            ):
                ram_bytes = sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES")
                ram_gb = round(ram_bytes / (1024**3), 2)
        except (ValueError, OSError, AttributeError):
            ram_gb = 0.0

    # OS Platform & Version
    os_plat = platform.system()
    os_ver = platform.version()

    # Python Version
    py_ver = sys.version.split()[0]

    # Optional torch version
    torch_ver: str | None = None
    try:
        torch_mod = importlib.import_module("torch")
        torch_ver = str(getattr(torch_mod, "__version__", "")) or None
    except (ImportError, AttributeError):
        pass

    # Optional onnxruntime version
    onnx_ver: str | None = None
    try:
        ort_mod = importlib.import_module("onnxruntime")
        onnx_ver = str(getattr(ort_mod, "__version__", "")) or None
    except (ImportError, AttributeError):
        pass

    # Optional cv2 version
    cv2_ver: str | None = None
    try:
        cv2_mod = importlib.import_module("cv2")
        cv2_ver = str(getattr(cv2_mod, "__version__", "")) or None
    except (ImportError, AttributeError):
        pass

    # Effective thread count
    thread_count = os.cpu_count() or 1

    return SystemEnvironmentMetadata(
        cpu_model=cpu_model,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        ram_total_gb=ram_gb,
        os_platform=os_plat,
        os_version=os_ver,
        python_version=py_ver,
        torch_version=torch_ver,
        onnxruntime_version=onnx_ver,
        opencv_version=cv2_ver,
        thread_count=thread_count,
    )
