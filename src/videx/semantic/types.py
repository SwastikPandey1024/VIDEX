"""VIDEX semantic intelligence taxonomy, statuses, and routing enumerations."""

from enum import StrEnum


class SemanticStatus(StrEnum):
    """Lifecycle and confirmation state of a semantic reasoning result."""

    SUPPORTED = "supported"
    """The semantic claim is strongly supported by the supplied visual/audio evidence."""

    UNCERTAIN = "uncertain"
    """The model observed partial visual/audio indicators but confidence is low or ambiguous."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """The supplied evidence bundle does not contain enough context to confirm or deny the claim."""

    REJECTED = "rejected"
    """The result failed validation (e.g., hallucinated evidence IDs, temporal mismatch)."""


class SemanticEventType(StrEnum):
    """Taxonomy of high-level semantic event interpretations produced by VLM reasoning."""

    INTERACTION = "interaction"
    """An intentional physical or social interaction between two or more tracked entities."""

    OBJECT_TRANSFER = "object_transfer"
    """An object was exchanged, handed over, picked up, or deposited."""

    VEHICLE_ACTIVITY = "vehicle_activity"
    """Complex vehicle behavior (e.g. checkpoint inspection, loading, passenger boarding)."""

    SECURITY_INCIDENT = "security_incident"
    """A noteworthy security event (e.g. perimeter breach, loitering altercation)."""

    COMMERCIAL_ACTIVITY = "commercial_activity"
    """A business or retail interaction (e.g. transaction, point of sale interaction)."""

    GENERAL_ACTIVITY = "general_activity"
    """General open-world human or scene activity."""

    CUSTOM = "custom"
    """User-defined or domain-specific semantic activity."""


class RoutingPriority(StrEnum):
    """Operational priority for candidate evaluation by the Semantic Router."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingDecision(StrEnum):
    """Decision emitted by the Semantic Router for a candidate event."""

    DISPATCH_VLM = "dispatch_vlm"
    """Candidate passed saliency, budget, and policy checks; routed for VLM reasoning."""

    SERVE_FROM_CACHE = "serve_from_cache"
    """Identical candidate context found in semantic cache; skipped VLM inference."""

    SUPPRESS_LOW_SALIENCY = "suppress_low_saliency"
    """Candidate saliency fell below configured threshold; suppressed to conserve compute."""

    SUPPRESS_QUERY_MISMATCH = "suppress_query_mismatch"
    """Candidate does not match user query or active search filter."""

    SUPPRESS_BUDGET_EXCEEDED = "suppress_budget_exceeded"
    """Inference token or request budget exceeded; queued or dropped by policy."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """Candidate lacks minimum visual or temporal evidence required for meaningful reasoning."""
