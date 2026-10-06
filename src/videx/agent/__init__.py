"""VIDEX Agentic Investigation & Query Reasoning (Phase 8)."""

from __future__ import annotations

from videx.agent.answer import InvestigationResult, render_answer
from videx.agent.claims import Claim, ClaimSet, ClaimValidationReport, ClaimValidator
from videx.agent.evidence_gate import EvidenceSufficiencyGate, SufficiencyCheckResult
from videx.agent.executor import InvestigationExecutor
from videx.agent.investigator import Investigator
from videx.agent.mock import MockInvestigator, create_mock_context
from videx.agent.planner import InvestigationPlanner
from videx.agent.registry import ToolDefinition, ToolRegistry
from videx.agent.schemas import (
    EvidenceReference,
    InvestigationPlan,
    InvestigationRequest,
    InvestigationStep,
    InvestigationTrace,
    ToolCall,
    ToolResult,
)
from videx.agent.telemetry import InvestigationMetrics, InvestigationTelemetry
from videx.agent.tools import AgentToolContext, register_investigation_tools
from videx.agent.types import (
    ClaimStatus,
    InvestigationStatus,
    QuestionCategory,
    ToolName,
)

__all__ = [
    "AgentToolContext",
    "Claim",
    "ClaimSet",
    "ClaimStatus",
    "ClaimValidationReport",
    "ClaimValidator",
    "EvidenceReference",
    "EvidenceSufficiencyGate",
    "InvestigationExecutor",
    "InvestigationMetrics",
    "InvestigationPlan",
    "InvestigationPlanner",
    "InvestigationRequest",
    "InvestigationResult",
    "InvestigationStatus",
    "InvestigationStep",
    "InvestigationTelemetry",
    "InvestigationTrace",
    "Investigator",
    "MockInvestigator",
    "QuestionCategory",
    "SufficiencyCheckResult",
    "ToolCall",
    "ToolDefinition",
    "ToolName",
    "ToolRegistry",
    "ToolResult",
    "create_mock_context",
    "register_investigation_tools",
    "render_answer",
]
