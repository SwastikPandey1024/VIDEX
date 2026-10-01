"""VIDEX Semantic Intelligence Foundation (Phase 6.0).

Layer 4 Semantic Router and Layer 5 VLM Provider boundaries for evidence-backed
semantic event reasoning.
"""

from videx.semantic.cache import SemanticCache
from videx.semantic.candidates import CandidateSelectionConfig, CandidateSelector
from videx.semantic.crops import CropExtractionError, CropExtractor
from videx.semantic.evidence_bundle import EvidenceBundleBuilder
from videx.semantic.mock import MockVLMProvider
from videx.semantic.policy import BudgetSnapshot, RoutingPolicy, RoutingPolicyConfig
from videx.semantic.providers import VLMProvider, build_evidence_grounded_prompt
from videx.semantic.qwen import Qwen3VLAdapter, Qwen3VLConfig
from videx.semantic.router import SemanticRouter
from videx.semantic.schemas import (
    CandidateEvent,
    CropRegion,
    EvidenceBundle,
    KeyframeMetadata,
    RoutingDecisionResult,
    RoutingRequest,
    SemanticEventPayload,
)
from videx.semantic.types import (
    RoutingDecision,
    RoutingPriority,
    SemanticEventType,
    SemanticStatus,
)
from videx.semantic.validator import EvidenceValidator, ValidationResult

__all__ = [
    "BudgetSnapshot",
    "CandidateEvent",
    "CandidateSelectionConfig",
    "CandidateSelector",
    "CropExtractionError",
    "CropExtractor",
    "CropRegion",
    "EvidenceBundle",
    "EvidenceBundleBuilder",
    "EvidenceValidator",
    "KeyframeMetadata",
    "MockVLMProvider",
    "Qwen3VLAdapter",
    "Qwen3VLConfig",
    "RoutingDecision",
    "RoutingDecisionResult",
    "RoutingPolicy",
    "RoutingPolicyConfig",
    "RoutingPriority",
    "RoutingRequest",
    "SemanticCache",
    "SemanticEventPayload",
    "SemanticEventType",
    "SemanticRouter",
    "SemanticStatus",
    "VLMProvider",
    "ValidationResult",
    "build_evidence_grounded_prompt",
]
