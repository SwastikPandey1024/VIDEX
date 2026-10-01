"""Configurable routing policies, token estimation, and rate/budget guardrails."""

from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple

from pydantic import BaseModel, Field

from videx.semantic.schemas import CandidateEvent, EvidenceBundle
from videx.semantic.types import RoutingDecision, RoutingPriority


class BudgetSnapshot(NamedTuple):
    """Current usage snapshot for telemetry and observability."""

    requests_last_minute: int
    tokens_last_minute: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    is_budget_exceeded: bool


class RoutingPolicyConfig(BaseModel):
    """Operational parameters governing semantic routing gating and compute limits."""

    min_saliency_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum candidate saliency required to justify VLM invocation",
    )
    min_query_relevance_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum query relevance when a search query is actively specified",
    )
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum VLM invocations permitted per sliding 60-second window",
    )
    max_tokens_per_minute: int = Field(
        default=100_000,
        ge=1,
        description="Maximum estimated VLM tokens permitted per sliding 60-second window",
    )
    base_prompt_tokens: int = Field(
        default=400,
        ge=50,
        description="Fixed token overhead for system prompt instructions and schemas",
    )
    tokens_per_visual_crop: int = Field(
        default=320,
        ge=50,
        description="Estimated token cost per image patch/crop passed to VLM",
    )
    tokens_per_transcript_second: int = Field(
        default=20,
        ge=1,
        description="Estimated token cost per second of spoken dialogue context",
    )


class RoutingPolicy:
    """Evaluates candidate events against saliency, query relevance, and rate limits."""

    def __init__(self, config: RoutingPolicyConfig | None = None) -> None:
        self.config = config or RoutingPolicyConfig()
        # Sliding 60-second window timestamps: list of (timestamp, token_count)
        self._usage_history: deque[tuple[float, int]] = deque()

    def evaluate_gating(
        self,
        candidate: CandidateEvent,
        query: str | None = None,
        priority_override: RoutingPriority | None = None,
    ) -> tuple[RoutingDecision, str]:
        """Determine whether candidate qualifies for VLM reasoning.

        Returns:
            Tuple of (RoutingDecision, rationale_string).
        """
        priority = priority_override or candidate.priority

        # 1. Critical events bypass low saliency threshold
        if priority == RoutingPriority.CRITICAL:
            return (
                RoutingDecision.DISPATCH_VLM,
                "Priority is CRITICAL; bypassing standard thresholds",
            )

        # 2. Query relevance check (if query specified)
        if query and query.strip():
            if candidate.query_relevance_score < self.config.min_query_relevance_threshold:
                return (
                    RoutingDecision.SUPPRESS_QUERY_MISMATCH,
                    f"Query relevance {candidate.query_relevance_score:.2f} < "
                    f"threshold {self.config.min_query_relevance_threshold:.2f}",
                )

        # 3. Saliency threshold check
        if candidate.saliency_score < self.config.min_saliency_threshold:
            return (
                RoutingDecision.SUPPRESS_LOW_SALIENCY,
                f"Candidate saliency {candidate.saliency_score:.2f} < "
                f"threshold {self.config.min_saliency_threshold:.2f}",
            )

        return RoutingDecision.DISPATCH_VLM, "Candidate passed saliency, query, and priority gating"

    def estimate_tokens(self, bundle: EvidenceBundle) -> int:
        """Deterministically estimate prompt token consumption for an evidence bundle."""
        crop_tokens = len(bundle.crops) * self.config.tokens_per_visual_crop
        dialogue_seconds = max(0.0, bundle.end_timestamp_seconds - bundle.start_timestamp_seconds)
        transcript_tokens = int(dialogue_seconds * self.config.tokens_per_transcript_second)
        ocr_tokens = sum(len(o.get("text", "").split()) * 2 for o in bundle.ocr_observations)
        trajectory_tokens = len(bundle.trajectory_summary) * 30

        total = (
            self.config.base_prompt_tokens
            + crop_tokens
            + transcript_tokens
            + ocr_tokens
            + trajectory_tokens
        )
        return max(self.config.base_prompt_tokens, total)

    def check_budget(self, estimated_tokens: int) -> bool:
        """Check if executing a request with estimated tokens exceeds sliding-window limits."""
        now = time.monotonic()
        self._prune_expired_usage(now)

        current_requests = len(self._usage_history)
        current_tokens = sum(tokens for _, tokens in self._usage_history)

        if (current_requests + 1) > self.config.max_requests_per_minute:
            return False
        if (current_tokens + estimated_tokens) > self.config.max_tokens_per_minute:
            return False

        return True

    def record_usage(self, token_count: int) -> None:
        """Record executed request and tokens into sliding window."""
        now = time.monotonic()
        self._prune_expired_usage(now)
        self._usage_history.append((now, token_count))

    def get_budget_snapshot(self) -> BudgetSnapshot:
        """Return current rate and token usage metrics."""
        now = time.monotonic()
        self._prune_expired_usage(now)
        curr_req = len(self._usage_history)
        curr_tok = sum(t for _, t in self._usage_history)
        exceeded = (curr_req >= self.config.max_requests_per_minute) or (
            curr_tok >= self.config.max_tokens_per_minute
        )
        return BudgetSnapshot(
            requests_last_minute=curr_req,
            tokens_last_minute=curr_tok,
            max_requests_per_minute=self.config.max_requests_per_minute,
            max_tokens_per_minute=self.config.max_tokens_per_minute,
            is_budget_exceeded=exceeded,
        )

    def reset_budget(self) -> None:
        """Clear rate limit and budget tracking state."""
        self._usage_history.clear()

    def _prune_expired_usage(self, now: float) -> None:
        """Evict records older than 60 seconds."""
        cutoff = now - 60.0
        while self._usage_history and self._usage_history[0][0] < cutoff:
            self._usage_history.popleft()
