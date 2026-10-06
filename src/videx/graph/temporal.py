"""Deterministic temporal relationship engine for the Evidence Graph."""

from __future__ import annotations

from collections.abc import Sequence

from videx.events.schemas import Event
from videx.graph.edges import GraphEdge, create_edge
from videx.graph.nodes import GraphNode, make_node_id
from videx.graph.types import DerivationType, GraphEdgeType, GraphNodeType


class TemporalGraphEngine:
    """Computes deterministic temporal edges (PRECEDES, FOLLOWS, TEMPORALLY_NEAR) over events."""

    def __init__(self, near_threshold_seconds: float = 2.0) -> None:
        self.near_threshold_seconds = near_threshold_seconds

    def compute_temporal_edges(
        self,
        events: Sequence[Event | GraphNode],
    ) -> list[GraphEdge]:
        """Generate pairwise deterministic temporal relationship edges using authoritative PTS."""
        edges: list[GraphEdge] = []

        # Normalize to (node_id, start_pts, end_pts)
        event_tuples: list[tuple[str, float, float]] = []
        for ev in events:
            if isinstance(ev, Event):
                nid = make_node_id(GraphNodeType.EVENT, ev.event_id)
                start_ts = ev.start_timestamp
                end_ts = ev.end_timestamp
            else:
                nid = ev.node_id
                start_ts = ev.timestamp_start or 0.0
                end_ts = ev.timestamp_end or start_ts
            event_tuples.append((nid, start_ts, end_ts))

        # Sort by start_timestamp for deterministic evaluation
        sorted_tuples = sorted(event_tuples, key=lambda t: (t[1], t[2], t[0]))

        n = len(sorted_tuples)
        for i in range(n):
            a_id, a_start, a_end = sorted_tuples[i]
            for j in range(i + 1, n):
                b_id, b_start, b_end = sorted_tuples[j]

                # 1. Strictly PRECEDES / FOLLOWS
                if a_end < b_start:
                    gap = b_start - a_end
                    edges.append(
                        create_edge(
                            source_node_id=a_id,
                            target_node_id=b_id,
                            relationship=GraphEdgeType.PRECEDES,
                            derivation=DerivationType.DETERMINISTIC,
                            reason=(
                                f"Event {a_id} strictly ends at {a_end:.3f}s "
                                f"before {b_id} begins at {b_start:.3f}s"
                            ),
                            actual_value=gap,
                        )
                    )
                    edges.append(
                        create_edge(
                            source_node_id=b_id,
                            target_node_id=a_id,
                            relationship=GraphEdgeType.FOLLOWS,
                            derivation=DerivationType.DETERMINISTIC,
                            reason=(
                                f"Event {b_id} strictly begins at {b_start:.3f}s "
                                f"after {a_id} ends at {a_end:.3f}s"
                            ),
                            actual_value=gap,
                        )
                    )

                # 2. TEMPORALLY_NEAR
                # Check overlap or proximity within near_threshold_seconds
                is_overlapping = max(a_start, b_start) <= min(a_end, b_end)
                if is_overlapping:
                    gap = 0.0
                else:
                    gap = max(0.0, max(a_start - b_end, b_start - a_end))

                if gap <= self.near_threshold_seconds:
                    edges.append(
                        create_edge(
                            source_node_id=a_id,
                            target_node_id=b_id,
                            relationship=GraphEdgeType.TEMPORALLY_NEAR,
                            derivation=DerivationType.DETERMINISTIC,
                            threshold=self.near_threshold_seconds,
                            actual_value=gap,
                            reason=(
                                f"Temporal gap {gap:.3f}s <= "
                                f"threshold {self.near_threshold_seconds:.2f}s"
                            ),
                        )
                    )

        return edges
