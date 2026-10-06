"""Typed query service for the Evidence Graph supporting Phase 8 & 9 operations."""

from __future__ import annotations

from uuid import UUID

from videx.graph.nodes import GraphNode, make_node_id
from videx.graph.schemas import EdgeFilter, GraphPath, GraphQueryResult, NodeFilter
from videx.graph.store import GraphStore
from videx.graph.types import Direction, GraphEdgeType, GraphNodeType


class GraphQueryService:
    """Provides high-level typed query and provenance traversal operations over GraphStore."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    @property
    def store(self) -> GraphStore:
        return self._store

    def query(
        self,
        node_filter: NodeFilter | None = None,
        edge_filter: EdgeFilter | None = None,
    ) -> GraphQueryResult:
        """Execute composite graph query returning matched nodes, edges, and execution metrics."""
        import time

        t0 = time.perf_counter()
        nodes = self._store.query_nodes(node_filter or NodeFilter())
        edges = self._store.query_edges(edge_filter or EdgeFilter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return GraphQueryResult(
            nodes=nodes,
            edges=edges,
            execution_time_ms=elapsed_ms,
            total_nodes_scanned=self._store.count_nodes(),
            total_edges_scanned=self._store.count_edges(),
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch node by global ID."""
        return self._store.get_node(node_id)

    def query_nodes(self, node_filter: NodeFilter) -> list[GraphNode]:
        """Query nodes matching multi-criteria filter."""
        return self._store.query_nodes(node_filter)

    def neighbors(
        self,
        node_id: str,
        direction: Direction = Direction.BOTH,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        """Fetch adjacent nodes."""
        return self._store.neighbors(node_id, direction, relationship)

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[GraphPath]:
        """Traverse shortest paths between source and target nodes."""
        return self._store.find_path(source_id, target_id, max_depth)

    def events_between(
        self,
        video_id: str | UUID,
        start_sec: float,
        end_sec: float,
    ) -> list[GraphNode]:
        """Query chronological events within a temporal window."""
        return self._store.query_nodes(
            NodeFilter(
                video_id=str(video_id),
                node_types=(GraphNodeType.EVENT, GraphNodeType.SEMANTIC_EVENT),
                time_start=start_sec,
                time_end=end_sec,
            )
        )

    def events_involving_track(self, track_id: str | UUID) -> list[GraphNode]:
        """Query all events that a tracked entity participated in."""
        t_nid = make_node_id(GraphNodeType.TRACK, track_id)
        # Track -(PARTICIPATES_IN)-> Event
        return self._store.neighbors(
            t_nid,
            direction=Direction.OUTBOUND,
            relationship=GraphEdgeType.PARTICIPATES_IN,
        )

    def evidence_for_event(self, event_id: str | UUID) -> list[GraphNode]:
        """Fetch all canonical Evidence records grounding a given event."""
        ev_nid = make_node_id(GraphNodeType.EVENT, event_id)
        # Event -(SUPPORTED_BY)-> Evidence
        direct_evidence = self._store.neighbors(
            ev_nid,
            direction=Direction.OUTBOUND,
            relationship=GraphEdgeType.SUPPORTED_BY,
        )
        return [n for n in direct_evidence if n.node_type == GraphNodeType.EVIDENCE]

    def related_events(
        self,
        event_id: str | UUID,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        """Query events connected directly to this event."""
        ev_nid = make_node_id(GraphNodeType.EVENT, event_id)
        candidates = self._store.neighbors(
            ev_nid,
            direction=Direction.BOTH,
            relationship=relationship,
        )
        return [
            n
            for n in candidates
            if n.node_type in (GraphNodeType.EVENT, GraphNodeType.SEMANTIC_EVENT)
        ]

    def temporal_neighbors(
        self,
        event_id: str | UUID,
        window_sec: float | None = None,
    ) -> list[GraphNode]:
        """Retrieve events in close temporal proximity."""
        ev_nid = make_node_id(GraphNodeType.EVENT, event_id)
        node = self._store.get_node(ev_nid)
        if node is None or node.timestamp_start is None:
            return []

        if window_sec is not None:
            w_start = max(0.0, node.timestamp_start - window_sec)
            w_end = (node.timestamp_end or node.timestamp_start) + window_sec
            in_window = self._store.query_nodes(
                NodeFilter(
                    video_id=node.video_id,
                    node_types=(GraphNodeType.EVENT, GraphNodeType.SEMANTIC_EVENT),
                    time_start=w_start,
                    time_end=w_end,
                )
            )
            return [n for n in in_window if n.node_id != ev_nid]

        # Use explicit TEMPORALLY_NEAR edges
        near_nodes = self._store.neighbors(
            ev_nid,
            direction=Direction.BOTH,
            relationship=GraphEdgeType.TEMPORALLY_NEAR,
        )
        return [
            n
            for n in near_nodes
            if n.node_type in (GraphNodeType.EVENT, GraphNodeType.SEMANTIC_EVENT)
        ]

    def spatial_neighbors(self, node_id_or_source_id: str | UUID) -> list[GraphNode]:
        """Query co-temporal or overlapping spatial neighbors."""
        str_id = str(node_id_or_source_id)
        if ":" in str_id:
            nid = str_id
        else:
            nid = make_node_id(GraphNodeType.DETECTION, str_id)

        return self._store.neighbors(
            nid,
            direction=Direction.BOTH,
            relationship=GraphEdgeType.SPATIALLY_OVERLAPS,
        )

    def trace_evidence_chain(self, semantic_event_id: str | UUID) -> list[GraphNode]:
        """Trace complete causal chain: SemanticEvent -> Events -> Evidence -> Frames."""
        sem_nid = make_node_id(GraphNodeType.SEMANTIC_EVENT, semantic_event_id)
        chain: list[GraphNode] = []
        sem_node = self._store.get_node(sem_nid)
        if sem_node is None:
            return []

        chain.append(sem_node)

        # Supporting deterministic events
        sup_events = self._store.neighbors(
            sem_nid,
            direction=Direction.OUTBOUND,
            relationship=GraphEdgeType.DERIVED_FROM,
        )
        for ev in sup_events:
            chain.append(ev)
            # Supporting evidence for event
            ev_evidences = self._store.neighbors(
                ev.node_id,
                direction=Direction.OUTBOUND,
                relationship=GraphEdgeType.SUPPORTED_BY,
            )
            for evid in ev_evidences:
                if evid not in chain:
                    chain.append(evid)
                # Frame referenced by evidence
                frames = self._store.neighbors(
                    evid.node_id,
                    direction=Direction.OUTBOUND,
                    relationship=GraphEdgeType.REFERENCES,
                )
                for f in frames:
                    if f not in chain:
                        chain.append(f)

        return chain
