"""Structured telemetry and latency performance tracking for VIDEX Investigator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InvestigationMetrics:
    """Latency and resource counters captured across an investigation lifecycle."""

    investigation_id: str
    video_id: str
    planning_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    evidence_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tool_call_count: int = 0
    step_count: int = 0
    tags: dict[str, Any] = field(default_factory=dict)


class InvestigationTelemetry:
    """Structured telemetry collector for investigator spans and events."""

    def __init__(self, investigation_id: str, video_id: str) -> None:
        self.metrics = InvestigationMetrics(
            investigation_id=investigation_id,
            video_id=video_id,
        )
        self._start_time = time.perf_counter()

    def record_planning_time(self, duration_ms: float) -> None:
        """Record latency spent formulating the investigation plan."""
        self.metrics.planning_latency_ms = duration_ms

    def record_tool_call(self, tool_name: str, duration_ms: float) -> None:
        """Record individual tool execution latency and increment counters."""
        self.metrics.tool_latency_ms += duration_ms
        self.metrics.tool_call_count += 1
        if "graph" in tool_name or "neighbors" in tool_name:
            self.metrics.graph_latency_ms += duration_ms
        elif "evidence" in tool_name or "event" in tool_name or "track" in tool_name:
            self.metrics.evidence_latency_ms += duration_ms

    def record_validation_time(self, duration_ms: float) -> None:
        """Record latency spent validating evidence and claims."""
        self.metrics.validation_latency_ms += duration_ms

    def finalize(self, step_count: int) -> InvestigationMetrics:
        """Calculate total investigation latency and finalize metrics."""
        self.metrics.total_latency_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.metrics.step_count = step_count
        return self.metrics

    def log_event(
        self,
        event_name: str,
        step_id: str = "",
        tool_call_id: str = "",
        event_id: str = "",
        evidence_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Structured trace log supporting observability fields."""
        payload: dict[str, Any] = {
            "telemetry_event": event_name,
            "investigation_id": self.metrics.investigation_id,
            "video_id": self.metrics.video_id,
            "step_id": step_id,
            "tool_call_id": tool_call_id,
            "event_id": event_id,
            "evidence_id": evidence_id,
        }
        if details:
            payload.update(details)
        logger.debug("videx.investigator.trace: %s", payload)
