"""Bounded investigation execution loop for VIDEX Agentic Investigator."""

from __future__ import annotations

import json
import logging
import time

from videx.agent.registry import ToolRegistry
from videx.agent.schemas import (
    InvestigationPlan,
    InvestigationRequest,
    InvestigationStep,
    ToolCall,
    ToolResult,
)
from videx.agent.telemetry import InvestigationTelemetry
from videx.agent.types import InvestigationStatus

logger = logging.getLogger(__name__)


class InvestigationExecutor:
    """Bounded, deterministic execution engine for investigator plans.

    Enforces:
    - Strict maximum steps limit
    - Strict maximum tool calls limit
    - Hard timeout budget
    - Deterministic tool ordering
    - Duplicate-call caching / prevention
    - Comprehensive step and provenance tracing
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        """Computes a deterministic hashable signature for a tool call to detect duplicates."""
        try:
            param_str = json.dumps(call.parameters, sort_keys=True, default=str)
        except Exception:
            param_str = str(sorted(call.parameters.items()))
        return f"{call.tool_name}:{param_str}"

    def execute(
        self,
        request: InvestigationRequest,
        plan: InvestigationPlan,
        tool_calls: list[ToolCall],
        telemetry: InvestigationTelemetry | None = None,
    ) -> tuple[list[InvestigationStep], list[ToolResult], InvestigationStatus]:
        """Executes a bounded queue of tool calls under safety limits."""
        start_time = time.perf_counter()
        steps: list[InvestigationStep] = []
        all_results: list[ToolResult] = []

        cache: dict[str, ToolResult] = {}
        tool_call_count = 0
        status = InvestigationStatus.COMPLETED

        queue = list(tool_calls)

        for step_idx, call in enumerate(queue):
            # Check maximum steps
            if step_idx >= request.max_steps:
                logger.warning(
                    "Investigation %s exceeded max_steps limit (%d)",
                    request.investigation_id,
                    request.max_steps,
                )
                status = InvestigationStatus.MAX_STEPS_EXCEEDED
                break

            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > request.timeout_seconds:
                logger.warning(
                    "Investigation %s exceeded timeout budget (%.2fs > %.2fs)",
                    request.investigation_id,
                    elapsed,
                    request.timeout_seconds,
                )
                status = InvestigationStatus.TIMEOUT
                break

            # Check maximum tool calls
            if tool_call_count >= request.max_tool_calls:
                logger.warning(
                    "Investigation %s exceeded max_tool_calls limit (%d)",
                    request.investigation_id,
                    request.max_tool_calls,
                )
                status = InvestigationStatus.MAX_STEPS_EXCEEDED
                break

            sig = self._call_signature(call)
            step_start = time.perf_counter()

            if sig in cache:
                logger.debug("Reusing cached result for duplicate call: %s", sig)
                result = cache[sig]
                thought = f"Duplicate call detected for {call.tool_name}; reused cached result"
            else:
                thought = f"Execute planned tool {call.tool_name} for step {step_idx}"
                result = self._registry.execute(call.tool_name, call.parameters, call.call_id)
                cache[sig] = result
                tool_call_count += 1
                if telemetry:
                    telemetry.record_tool_call(call.tool_name, result.execution_time_ms)
                    telemetry.log_event(
                        event_name="tool_call_executed",
                        step_id=str(step_idx),
                        tool_call_id=call.call_id,
                        details={
                            "tool_name": call.tool_name,
                            "success": result.success,
                            "provenance_count": len(result.provenance),
                        },
                    )

            step_duration_ms = (time.perf_counter() - step_start) * 1000.0
            step = InvestigationStep(
                step_index=step_idx,
                thought=thought,
                action=call,
                result=result,
                duration_ms=step_duration_ms,
            )
            steps.append(step)
            all_results.append(result)

        return steps, all_results, status
