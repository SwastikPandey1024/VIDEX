"""Pydantic schemas for VIDEX Agentic Investigation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from videx.agent.types import QuestionCategory


class EvidenceReference(BaseModel):
    """Reference to a canonical evidence record grounding an investigation step."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Canonical Evidence UUID")
    evidence_type: str = Field(..., description="Categorization of evidence (e.g. 'detection')")
    video_id: str = Field(..., description="Video ID scoping this evidence")
    timestamp_seconds: float | None = Field(default=None, description="Authoritative PTS")
    source_module: str = Field(default="", description="Subsystem that generated the evidence")
    confidence: float | None = Field(default=None, description="Detection/sensor confidence")
    description: str | None = Field(default=None, description="Human-readable summary")


class ToolCall(BaseModel):
    """Structured request to execute an agent tool."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str = Field(..., description="Target registered tool name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Typed arguments")
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ToolCall:
        params = arguments if arguments is not None else (parameters or {})
        return cls(tool_name=tool_name, parameters=params)


class ToolResult(BaseModel):
    """Structured outcome of an agent tool execution."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(..., description="Matching ToolCall call_id")
    tool_name: str = Field(..., description="Tool that was executed")
    success: bool = Field(..., description="True if tool executed without uncaught error")
    data: Any = Field(default=None, description="Structured returned payload")
    error: str | None = Field(default=None, description="Error message if failed")
    provenance: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Audit provenance records preserved across returned items",
    )
    execution_time_ms: float = Field(default=0.0, description="Measured tool execution latency")


class InvestigationStep(BaseModel):
    """Single discrete step in an agent investigation."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(..., ge=0)
    thought: str = Field(..., description="Agent reasoning or intent for this step")
    action: ToolCall = Field(..., description="Tool invocation performed")
    result: ToolResult | None = Field(default=None, description="Execution result")
    duration_ms: float = Field(default=0.0, description="Step duration in milliseconds")


class InvestigationPlan(BaseModel):
    """Deterministic or LLM-formulated investigation plan."""

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    category: QuestionCategory = Field(..., description="Classified question category")
    objective: str = Field(..., description="Core investigation goal")
    planned_steps: list[str] = Field(default_factory=list, description="Sequenced step intents")
    required_tools: list[str] = Field(default_factory=list, description="Tool names required")
    video_id: str = Field(..., description="Target video ID")


class InvestigationRequest(BaseModel):
    """Entry point request payload for an investigator query."""

    model_config = ConfigDict(frozen=True)

    investigation_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Scoping video ID")
    question: str = Field(..., description="User query or hypothesis to investigate")
    max_steps: int = Field(default=10, ge=1, le=50, description="Safety limit on steps")
    max_tool_calls: int = Field(default=15, ge=1, le=100, description="Safety limit on tool calls")
    timeout_seconds: float = Field(default=30.0, ge=0.01, le=120.0, description="Timeout budget")
    allow_semantic_reasoning: bool = Field(
        default=True,
        description=(
            "Whether to permit VLM reasoning fallback if deterministic evidence is inconclusive"
        ),
    )

    @classmethod
    def create(
        cls,
        video_id: str | UUID,
        question: str,
        max_steps: int = 10,
        max_tool_calls: int = 15,
        timeout_seconds: float = 30.0,
        allow_semantic_reasoning: bool = True,
    ) -> InvestigationRequest:
        return cls(
            video_id=str(video_id),
            question=question,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            allow_semantic_reasoning=allow_semantic_reasoning,
        )


class InvestigationTrace(BaseModel):
    """Complete auditable trace of an investigation execution."""

    model_config = ConfigDict(frozen=True)

    investigation_id: str
    video_id: str
    user_question: str
    plan: InvestigationPlan
    steps: list[InvestigationStep] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    events_used: list[str] = Field(default_factory=list)
    semantic_calls: int = 0
    total_latency_ms: float = 0.0
    validation_passed: bool = True
