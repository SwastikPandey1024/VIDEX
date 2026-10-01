"""Deterministic, explainable candidate event selection for semantic reasoning."""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from videx.events.engine import EventTimeline
from videx.events.schemas import Event
from videx.events.types import EventSeverity, EventType
from videx.semantic.schemas import CandidateEvent
from videx.semantic.types import RoutingPriority


class CandidateSelectionConfig(BaseModel):
    """Configuration governing deterministic candidate selection and saliency scoring."""

    min_saliency_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum saliency required for a candidate to be considered for VLM routing",
    )
    temporal_cluster_window_seconds: float = Field(
        default=2.5,
        ge=0.0,
        description="Temporal window (s) for clustering proximate events into compound candidates",
    )
    cross_modal_boost: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="Saliency bonus awarded when visual and audio/OCR evidence co-occur",
    )
    ocr_presence_boost: float = Field(
        default=0.15,
        ge=0.0,
        le=0.3,
        description="Saliency bonus when text/license plate OCR is involved",
    )
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            EventSeverity.CRITICAL.value: 1.00,
            EventSeverity.HIGH.value: 0.80,
            EventSeverity.MEDIUM.value: 0.60,
            EventSeverity.LOW.value: 0.35,
            EventSeverity.INFO.value: 0.20,
        }
    )
    event_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            EventType.CUSTOM.value: 0.75,  # CrossModal compound rules
            EventType.ANOMALY.value: 0.90,
            EventType.LOITERING.value: 0.70,
            EventType.CROWD_FORMATION.value: 0.75,
            EventType.OBJECT_ENTERED_ZONE.value: 0.65,
            EventType.OBJECT_EXITED_ZONE.value: 0.50,
            EventType.OBJECT_CHANGED_DIRECTION.value: 0.55,
            EventType.OBJECT_STARTED_MOVING.value: 0.45,
            EventType.OBJECT_STOPPED_MOVING.value: 0.45,
            EventType.SPEECH_DETECTED.value: 0.50,
            EventType.TEXT_CHANGED.value: 0.60,
            EventType.TEXT_APPEARED.value: 0.40,
            EventType.OBJECT_APPEARED.value: 0.30,
            EventType.OBJECT_DISAPPEARED.value: 0.30,
        }
    )


class CandidateSelector:
    """Selects candidate events from an EventTimeline using pure deterministic rules.

    Guarantees full explainability: every candidate includes a structured reason,
    saliency breakdown, and complete provenance of source event IDs.
    """

    def __init__(self, config: CandidateSelectionConfig | None = None) -> None:
        self.config = config or CandidateSelectionConfig()

    def select_candidates(
        self,
        timeline_or_events: EventTimeline | Sequence[Event],
        query: str | None = None,
        video_id: UUID | None = None,
    ) -> list[CandidateEvent]:
        """Evaluate events and return prioritized, explainable CandidateEvents.

        Args:
            timeline_or_events: An EventTimeline or list of Event objects.
            query: Optional natural language search/alert query.
            video_id: Optional fallback video UUID.

        Returns:
            List of CandidateEvents sorted by priority and saliency.
        """
        events: list[Event] = (
            timeline_or_events.events
            if isinstance(timeline_or_events, EventTimeline)
            else list(timeline_or_events)
        )
        if not events:
            return []

        v_id = video_id or events[0].video_id

        # 1. Cluster events into candidate temporal groups
        clusters = self._cluster_events(events)

        candidates: list[CandidateEvent] = []
        for cluster in clusters:
            candidate = self._evaluate_cluster(cluster, query=query, video_id=v_id)
            if candidate is not None:
                candidates.append(candidate)

        # 2. Sort deterministically by priority, then saliency descending, then start_ts
        priority_ranks = {
            RoutingPriority.CRITICAL: 0,
            RoutingPriority.HIGH: 1,
            RoutingPriority.MEDIUM: 2,
            RoutingPriority.LOW: 3,
        }
        candidates.sort(
            key=lambda c: (
                priority_ranks.get(c.priority, 99),
                -c.saliency_score,
                -c.query_relevance_score,
                c.start_timestamp,
                str(c.candidate_id),
            )
        )
        return candidates

    def _cluster_events(self, events: list[Event]) -> list[list[Event]]:
        """Group temporally proximate and spatially related events."""
        if not events:
            return []

        # Sort events by start_timestamp
        sorted_evs = sorted(events, key=lambda e: (e.start_timestamp_seconds, str(e.event_id)))
        clusters: list[list[Event]] = []
        current_cluster: list[Event] = [sorted_evs[0]]

        window = self.config.temporal_cluster_window_seconds

        for ev in sorted_evs[1:]:
            cluster_end = max(e.end_timestamp for e in current_cluster)

            # Check temporal intersection or proximity within window
            if ev.start_timestamp_seconds <= (cluster_end + window):
                current_cluster.append(ev)
            else:
                clusters.append(current_cluster)
                current_cluster = [ev]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _evaluate_cluster(
        self,
        cluster: list[Event],
        query: str | None,
        video_id: UUID,
    ) -> CandidateEvent | None:
        """Score a cluster and construct a CandidateEvent."""
        if not cluster:
            return None

        # Temporal bounds
        start_ts = min(e.start_timestamp_seconds for e in cluster)
        end_ts = max(e.end_timestamp for e in cluster)

        # Participant collection
        participant_set: set[str] = set()
        for ev in cluster:
            for p in ev.participants:
                if p.label:
                    participant_set.add(p.label.lower())
                participant_set.add(str(p.participant_id))
            if ev.zone_name:
                participant_set.add(ev.zone_name.lower())

        # Evidence and event IDs
        source_event_ids = [e.event_id for e in cluster]
        evidence_ids: list[UUID] = []
        for e in cluster:
            for eid in e.evidence_ids:
                if eid not in evidence_ids:
                    evidence_ids.append(eid)

        # 1. Deterministic Saliency Calculation
        saliency_score, reasons = self._calculate_saliency(cluster)

        # 2. Query Relevance Calculation
        query_score, query_reason = self._calculate_query_relevance(cluster, query)

        # Filter by minimum saliency threshold (unless boosted by strong query match)
        if query is not None:
            effective_score = max(saliency_score, (saliency_score + query_score) / 2.0)
            if effective_score < self.config.min_saliency_threshold and query_score < 0.5:
                return None
        else:
            if saliency_score < self.config.min_saliency_threshold:
                return None

        # 3. Priority Determination
        priority = self._determine_priority(saliency_score, query_score, cluster, query is not None)

        # 4. Formulate explanation string
        full_reason = "; ".join(reasons)
        if query and query_reason:
            full_reason += f" | Query match: {query_reason}"

        return CandidateEvent(
            video_id=video_id,
            source_event_ids=source_event_ids,
            start_timestamp=round(start_ts, 3),
            end_timestamp=round(end_ts, 3),
            participant_ids=sorted(participant_set),
            saliency_score=round(saliency_score, 3),
            query_relevance_score=round(query_score, 3),
            reason=full_reason,
            priority=priority,
            evidence_ids=evidence_ids,
        )

    def _calculate_saliency(self, cluster: list[Event]) -> tuple[float, list[str]]:
        """Compute explainable saliency score from cluster features."""
        reasons: list[str] = []

        # Feature 1: Highest event severity
        max_sev_val = max(self.config.severity_weights.get(e.severity.value, 0.2) for e in cluster)
        dominant_sev = max(
            cluster, key=lambda e: self.config.severity_weights.get(e.severity.value, 0.2)
        ).severity
        reasons.append(f"severity={dominant_sev.value} (score={max_sev_val:.2f})")

        # Feature 2: Event type prominence
        type_scores = [
            self.config.event_type_weights.get(e.event_type.value, 0.35) for e in cluster
        ]
        max_type_score = max(type_scores)

        # Feature 3: Modality diversity (Cross-modal detection)
        event_types = {e.event_type for e in cluster}
        has_speech = any(
            t in (EventType.SPEECH_DETECTED, EventType.SPEECH_STARTED, EventType.SPEECH_ENDED)
            for t in event_types
        )
        has_spatial = any(
            t in (EventType.OBJECT_ENTERED_ZONE, EventType.OBJECT_EXITED_ZONE, EventType.LOITERING)
            for t in event_types
        )
        has_movement = any(
            t
            in (
                EventType.OBJECT_STARTED_MOVING,
                EventType.OBJECT_STOPPED_MOVING,
                EventType.OBJECT_CHANGED_DIRECTION,
            )
            for t in event_types
        )
        has_ocr = any(
            t in (EventType.TEXT_APPEARED, EventType.TEXT_CHANGED, EventType.TEXT_DISAPPEARED)
            for t in event_types
        )
        has_compound = EventType.CUSTOM in event_types or EventType.ANOMALY in event_types

        cross_modal_boost = 0.0
        if (has_speech and (has_spatial or has_movement)) or has_compound:
            cross_modal_boost = self.config.cross_modal_boost
            reasons.append(f"cross_modal_co_occurrence (+{cross_modal_boost:.2f})")

        ocr_boost = 0.0
        if has_ocr:
            ocr_boost = self.config.ocr_presence_boost
            reasons.append(f"ocr_text_presence (+{ocr_boost:.2f})")

        # Feature 4: Event density (multiple events in cluster)
        density_boost = min(0.15, (len(cluster) - 1) * 0.05)
        if density_boost > 0:
            reasons.append(f"event_density={len(cluster)} (+{density_boost:.2f})")

        # Mean confidence
        mean_conf = sum(e.confidence for e in cluster) / len(cluster)

        # Base combination
        base_saliency = (max_sev_val * 0.40) + (max_type_score * 0.30) + (mean_conf * 0.30)
        total_saliency = min(1.0, base_saliency + cross_modal_boost + ocr_boost + density_boost)

        return total_saliency, reasons

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from",
        "of", "is", "was", "are", "were", "what", "which", "who", "whom", "this", "that",
        "did", "does", "do", "happened", "occurred",
    }

    def _calculate_query_relevance(
        self,
        cluster: list[Event],
        query: str | None,
    ) -> tuple[float, str | None]:
        """Match natural language query terms against cluster attributes deterministically."""
        if not query or not query.strip():
            return 1.0, None

        # Clean query tokens (lowercase words >= 3 chars, removing stop words)
        raw_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower()))
        tokens = {t for t in raw_tokens if t not in self.STOP_WORDS}
        if not tokens:
            tokens = raw_tokens
        if not tokens:
            return 1.0, None

        # Build corpus from cluster metadata
        corpus_parts: list[str] = []
        for e in cluster:
            corpus_parts.append(e.event_type.value)
            corpus_parts.append(e.description.lower())
            if e.zone_name:
                corpus_parts.append(e.zone_name.lower())
            for p in e.participants:
                if p.label:
                    corpus_parts.append(p.label.lower())
                corpus_parts.append(p.role.lower())
            for k, v in e.attributes.items():
                corpus_parts.append(f"{k} {v}".lower())

        corpus_text = " ".join(corpus_parts)

        # Count matches
        matched_tokens = [t for t in tokens if t in corpus_text]
        match_fraction = len(matched_tokens) / len(tokens)

        # Add partial substring boost
        match_reason = (
            f"matched {len(matched_tokens)}/{len(tokens)} terms ({', '.join(matched_tokens)})"
        )
        return round(match_fraction, 3), match_reason

    def _determine_priority(
        self,
        saliency_score: float,
        query_score: float,
        cluster: list[Event],
        query_active: bool = False,
    ) -> RoutingPriority:
        """Determine operational evaluation priority."""
        has_critical = any(e.severity == EventSeverity.CRITICAL for e in cluster)
        has_high = any(e.severity == EventSeverity.HIGH for e in cluster)

        if has_critical:
            return RoutingPriority.CRITICAL
        if query_active and (saliency_score >= 0.85 and query_score >= 0.70):
            return RoutingPriority.CRITICAL
        if has_high or saliency_score >= 0.70 or (query_active and query_score >= 0.80):
            return RoutingPriority.HIGH
        if saliency_score >= 0.45 or (query_active and query_score >= 0.50):
            return RoutingPriority.MEDIUM
        return RoutingPriority.LOW
