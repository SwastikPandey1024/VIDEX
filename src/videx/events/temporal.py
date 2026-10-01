"""Deterministic temporal relationship engine for events.

Implements qualitative temporal relationships (Allen's interval algebra subset
plus NEAR_IN_TIME proximity) to analyze interactions between multimodal events.
"""

from __future__ import annotations

from collections.abc import Sequence

from videx.events.schemas import Event
from videx.events.types import TemporalRelation


class TemporalRelationEngine:
    """Evaluates qualitative temporal relationships between events."""

    def __init__(self, near_threshold_seconds: float = 2.0) -> None:
        self.near_threshold_seconds = near_threshold_seconds

    @staticmethod
    def is_before(a: Event, b: Event, tolerance: float = 0.0) -> bool:
        """True if Event A strictly ends before Event B begins."""
        return (a.end_timestamp + tolerance) < b.start_timestamp

    @staticmethod
    def is_after(a: Event, b: Event, tolerance: float = 0.0) -> bool:
        """True if Event A strictly begins after Event B ends."""
        return a.start_timestamp > (b.end_timestamp + tolerance)

    @staticmethod
    def is_overlaps(a: Event, b: Event) -> bool:
        """True if Event A and Event B share a non-zero temporal intersection."""
        intersection_start = max(a.start_timestamp, b.start_timestamp)
        intersection_end = min(a.end_timestamp, b.end_timestamp)
        return intersection_start < intersection_end

    @staticmethod
    def is_contains(a: Event, b: Event) -> bool:
        """True if Event A temporally encompasses Event B."""
        return a.start_timestamp <= b.start_timestamp and a.end_timestamp >= b.end_timestamp

    @staticmethod
    def is_during(a: Event, b: Event) -> bool:
        """True if Event A occurs completely within the span of Event B."""
        return b.start_timestamp <= a.start_timestamp and b.end_timestamp >= a.end_timestamp

    @staticmethod
    def is_equals(a: Event, b: Event, epsilon: float = 1e-4) -> bool:
        """True if Event A and Event B share identical start and end timestamps."""
        return (
            abs(a.start_timestamp - b.start_timestamp) <= epsilon
            and abs(a.end_timestamp - b.end_timestamp) <= epsilon
        )

    def is_near_in_time(self, a: Event, b: Event, max_delta: float | None = None) -> bool:
        """True if two events occur in close temporal proximity without necessarily overlapping."""
        delta = max_delta if max_delta is not None else self.near_threshold_seconds
        # If they overlap, they are already co-occurring
        if self.is_overlaps(a, b):
            return True

        if a.end_timestamp <= b.start_timestamp:
            gap = b.start_timestamp - a.end_timestamp
        else:
            gap = a.start_timestamp - b.end_timestamp

        return gap <= delta

    def determine_relations(
        self,
        a: Event,
        b: Event,
    ) -> list[TemporalRelation]:
        """Determine all applicable qualitative temporal relations between Event A and Event B."""
        relations: list[TemporalRelation] = []

        if self.is_equals(a, b):
            relations.append(TemporalRelation.EQUALS)

        if self.is_before(a, b):
            relations.append(TemporalRelation.BEFORE)
        elif self.is_after(a, b):
            relations.append(TemporalRelation.AFTER)

        if self.is_overlaps(a, b):
            relations.append(TemporalRelation.OVERLAPS)

        if self.is_contains(a, b):
            relations.append(TemporalRelation.CONTAINS)

        if self.is_during(a, b):
            relations.append(TemporalRelation.DURING)

        if self.is_near_in_time(a, b):
            relations.append(TemporalRelation.NEAR_IN_TIME)

        return relations

    def find_related_events(
        self,
        target: Event,
        candidates: Sequence[Event],
        filter_relations: Sequence[TemporalRelation] | None = None,
    ) -> list[tuple[Event, list[TemporalRelation]]]:
        """Find candidate events satisfying temporal relations with the target event.

        Args:
            target: The focal Event.
            candidates: Candidate Event records.
            filter_relations: Optional subset of relations to filter by.

        Returns:
            List of (candidate_event, matching_relations) tuples.
        """
        results: list[tuple[Event, list[TemporalRelation]]] = []
        target_set = set(filter_relations) if filter_relations is not None else None

        for cand in candidates:
            if cand.event_id == target.event_id:
                continue

            rels = self.determine_relations(target, cand)
            if not rels:
                continue

            if target_set is not None:
                matched = [r for r in rels if r in target_set]
                if matched:
                    results.append((cand, matched))
            else:
                results.append((cand, rels))

        return results
