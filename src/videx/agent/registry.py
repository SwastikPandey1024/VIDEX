"""Controlled ToolRegistry ensuring read-only, audited agent tool execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from videx.agent.schemas import ToolResult

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], ToolResult]


class ToolDefinition(BaseModel):
    """Specification of an authorized agent tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Functional capability description")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema of args")
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema of returned data"
    )
    read_only: bool = Field(
        default=True,
        description="Strict enforcement that tool does not mutate state",
    )
    allowed_in_agent: bool = Field(
        default=True,
        description="Whether investigator agent can invoke this tool",
    )
    provenance_required: bool = Field(
        default=True,
        description="Whether result items must contain epistemic and source provenance",
    )


class ToolRegistry:
    """Registry managing authorized tools for the VIDEX Investigator."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        """Register a new authorized tool."""
        if not definition.read_only:
            raise ValueError(f"Tool '{definition.name}' must be read_only=True for agent use.")
        self._tools[definition.name] = (definition, handler)

    def get_definition(self, name: str | ToolDefinition) -> ToolDefinition | None:
        """Fetch specification for a tool by name or definition."""
        k = name.name if isinstance(name, ToolDefinition) else name
        pair = self._tools.get(k)
        return pair[0] if pair else None

    def has_tool(self, name: str | ToolDefinition) -> bool:
        """Check if tool is registered and authorized."""
        k = name.name if isinstance(name, ToolDefinition) else name
        pair = self._tools.get(k)
        return pair is not None and pair[0].allowed_in_agent

    def list_tools(self) -> list[str]:
        """List names of all registered and authorized tools."""
        return [pair[0].name for pair in self._tools.values() if pair[0].allowed_in_agent]

    def list_definitions(self) -> list[ToolDefinition]:
        """List all registered and authorized tool specifications."""
        return [pair[0] for pair in self._tools.values() if pair[0].allowed_in_agent]

    def execute(
        self,
        name: str,
        parameters: dict[str, Any],
        call_id: str | None = None,
    ) -> ToolResult:
        """Safely execute a registered tool with error handling and latency telemetry."""
        cid = call_id or str(uuid4())
        pair = self._tools.get(name)
        if pair is None:
            return ToolResult(
                call_id=cid,
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered in ToolRegistry",
                execution_time_ms=0.0,
            )

        definition, handler = pair
        if not definition.allowed_in_agent:
            return ToolResult(
                call_id=cid,
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is disabled for agent invocation",
                execution_time_ms=0.0,
            )

        t0 = time.perf_counter()
        try:
            result = handler(parameters)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=cid,
                tool_name=name,
                success=result.success,
                data=result.data,
                error=result.error,
                provenance=result.provenance,
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("Error executing tool '%s': %s", name, exc)
            return ToolResult(
                call_id=cid,
                tool_name=name,
                success=False,
                error=f"Unhandled tool exception: {exc}",
                execution_time_ms=elapsed_ms,
            )
