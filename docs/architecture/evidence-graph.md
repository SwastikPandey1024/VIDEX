# VIDEX Architecture: Evidence Graph & Spatiotemporal Memory (Phase 7.0)

## 1. Executive Summary & Epistemological Placement

The **Evidence Graph** (`src/videx/graph/`) constitutes the relational, topological, and associative memory layer of VIDEX. It bridges low-level multimodal perception and deterministic event detection (Phases 1–5) and semantic VLM reasoning (Phase 6/6.1) with downstream consumers:
- **Phase 8 Agentic Video Investigation:** Autonomous multi-step hypothesis verification, cross-event reasoning, and chain-of-evidence traversals.
- **Phase 9 Interactive UI:** Multi-track spatiotemporal timeline browsing, entity relationship graphs, and evidence-audited drilldowns.

```text
┌───────────────────────────────────────────────────────────────┐
│ Layer 1: Canonical Evidence Store (PostgreSQL / SQLite)       │
│ Videos, Frames, Detections, Tracks, OCR, Audio, Events        │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                │ Deterministic Projection
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 2: Evidence Graph Engine (videx.graph)                  │
│  - Typed Spatiotemporal Nodes & Edges                         │
│  - GraphStore Abstraction (InMemoryGraphStore)                │
│  - Deterministic Temporal & Spatial Edge Derivations          │
│  - Full Provenance & Audit Trail                              │
│  - GraphIntegrityValidator                                    │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────────┐ ┌─────────────────────────────┐
│ Phase 8: Agentic Query API    │ │ Phase 9: UI Integration     │
│ (Path finding, evidence chains│ │ (Timeline, entities, graph  │
│  hypothesis testing)          │ │  visualizer, inspector)     │
└───────────────────────────────┘ └─────────────────────────────┘
```

### Core Axioms
1. **System of Record Invariance:** PostgreSQL (or local SQLite) is the immutable, authoritative system of record. The Evidence Graph is a **derived spatiotemporal relationship projection** over canonical records.
2. **Deterministic Projection:** Projecting identical canonical evidence inputs produces bit-for-bit identical graph node and edge topologies.
3. **Strict Cross-Video Isolation:** Graph state is strictly scoped to `video_id`. No cross-video leakage or state contamination can occur.
4. **Complete Provenance:** Every graph node and edge preserves its canonical source identifiers, exact presentation timestamps (PTS), and explicit derivation reasons.

---

## 2. Graph Domain Model

### 2.1 Node Hierarchy (`GraphNodeType`)

| Node Type | Canonical Source Entity | Primary Role | Provenance Retained |
|---|---|---|---|
| `VIDEO` | `videx.domain.schemas.Video` | Root context for media streams | `video_id`, filename, duration, fps |
| `SCENE` | `videx.domain.schemas.Scene` | Temporal visual boundary segment | `scene_id`, start/end PTS, scene index |
| `FRAME` | `videx.domain.schemas.Frame` | Discrete visual observation point | `frame_id`, frame index, PTS timestamp |
| `DETECTION` | `videx.domain.schemas.Detection` | Single-frame object bounding box | `detection_id`, class name, confidence, BBox |
| `TRACK` | `videx.domain.schemas.Track` | Persistent spatiotemporal entity | `track_id`, class name, first/last seen PTS |
| `OBJECT` | Canonical Track / Entity Concept | Domain-level tracked physical entity | `object_id`, class name, tracking span |
| `OCR_OBSERVATION` | `videx.domain.schemas.OCRObservation` | Observed text snippet in visual frame | `ocr_id`, text, confidence, bbox, PTS |
| `TRANSCRIPT_SEGMENT`| `videx.domain.schemas.TranscriptSegment`| Transcribed speech audio segment | `segment_id`, text, language, start/end PTS |
| `ZONE` | `videx.events.spatial.SpatialZone` | User-defined spatial region of interest | `zone_id`, name, polygon vertices |
| `EVENT` | `videx.events.schemas.Event` | Deterministic state transition / rule event| `event_id`, event type, start/end PTS |
| `SEMANTIC_EVENT` | `videx.semantic.schemas.SemanticEvent` | Layer 5 VLM-grounded interpretation | `semantic_event_id`, VLM reasoning, decision |
| `EVIDENCE` | `videx.domain.schemas.Evidence` | Canonical grounding record | `evidence_id`, evidence type, timestamp |

### 2.2 Relationship Hierarchy (`GraphEdgeType`)

| Edge Type | Semantics & Direction | Derivation Rule |
|---|---|---|
| `CONTAINS` | Parent structural container $\to$ Child entity | Deterministic structural hierarchy (e.g. Video $\to$ Frame, Video $\to$ Zone) |
| `SUPPORTS` | Evidence / Observation $\to$ Target entity | Grounding relation (e.g. Evidence $\to$ Event, OCR $\to$ Event) |
| `DESCRIBES` | Text / Transcript $\to$ Target entity | Semantic attribution (e.g. Transcript $\to$ AudioSegment) |
| `BELONGS_TO` | Detection $\to$ Track | Multi-object tracking identity association |
| `REPRESENTS` | Track $\to$ Object | Track identity to physical object abstraction |
| `PARTICIPATES_IN` | Track / Participant $\to$ Event | Kinematic or spatial involvement in event |
| `SUPPORTED_BY` | Event / Track $\to$ Evidence | Inverse audit reference to canonical evidence |
| `PRECEDES` | Event $A$ $\to$ Event $B$ | Deterministic temporal order: $A.\text{end} < B.\text{start}$ |
| `FOLLOWS` | Event $B$ $\to$ Event $A$ | Deterministic temporal order: $B.\text{start} > A.\text{end}$ |
| `TEMPORALLY_NEAR`| Event $A$ $\to$ Event $B$ | Deterministic proximity: $\text{gap} \le \Delta_{\text{thresh}}$ with recorded $\text{actual\_gap}$ |
| `SPATIALLY_NEAR`| Detection / Track $A$ $\to$ $B$ | Deterministic centroid distance $\le \Delta_{\text{dist}}$ |
| `SPATIALLY_OVERLAPS`| Detection $A$ $\to$ Detection $B$ | Co-temporal BBox intersection $\text{IoU} \ge \tau$ |
| `OCCURS_IN` | Track / Event $\to$ Zone | Ray-casting point-in-polygon containment |
| `DERIVED_FROM` | SemanticEvent $\to$ Event | VLM interpretation grounded in deterministic candidate event |
| `SUPPORTS_SEMANTIC_EVENT`| Event / Evidence $\to$ SemanticEvent | Upward propagation from deterministic to semantic layer |

---

## 3. Spatiotemporal Memory Engines

### 3.1 Temporal Graph Engine (`videx.graph.temporal`)
- Operates strictly on presentation timestamps (PTS in seconds).
- Generates `PRECEDES` / `FOLLOWS` edges between temporally ordered events.
- Evaluates `TEMPORALLY_NEAR` edges with configurable threshold (`near_threshold_seconds = 2.0s`).
- Edge metadata records `derivation="deterministic"`, `threshold_seconds`, and `actual_gap_seconds`.

### 3.2 Spatial Graph Engine (`videx.graph.spatial`)
- Operates on exact bounding boxes and polygon coordinate geometries.
- Generates `SPATIALLY_OVERLAPS` edges between concurrent detections sharing a frame when $\text{IoU} \ge \tau_{\text{iou}}$.
- Generates `OCCURS_IN` edges when track positions fall inside configured `SpatialZone` polygons.
- Strictly adheres to non-hallucinatory boundaries: does not synthesize subjective spatial intents (e.g. "following", "loitering"), which are handled in Layer 5 VLM reasoning.

---

## 4. Graph Store Abstraction (`GraphStore`)

The storage interface defines a minimal, typed, high-performance contract:
```python
class GraphStore(Protocol):
    def upsert_node(self, node: GraphNode) -> None: ...
    def upsert_edge(self, edge: GraphEdge) -> None: ...
    def get_node(self, node_id: str) -> GraphNode | None: ...
    def get_edge(self, edge_id: str) -> GraphEdge | None: ...
    def neighbors(self, node_id: str, direction: Direction, relationship: GraphEdgeType | None) -> list[GraphNode]: ...
    def find_path(self, source_id: str, target_id: str, max_depth: int) -> list[list[str]]: ...
    def query_nodes(self, filter: NodeFilter) -> list[GraphNode]: ...
    def query_edges(self, filter: EdgeFilter) -> list[GraphEdge]: ...
```

`InMemoryGraphStore` implements this protocol using indexed hash maps:
- `_nodes: dict[str, GraphNode]`
- `_edges: dict[str, GraphEdge]`
- `_out_edges: dict[str, set[str]]` (node_id -> set of edge_ids)
- `_in_edges: dict[str, set[str]]` (node_id -> set of edge_ids)
- `_video_index: dict[str, set[str]]` (video_id -> set of node_ids)
- `_type_index: dict[GraphNodeType, set[str]]` (type -> set of node_ids)

---

## 5. Downstream Integration Contracts

### 5.1 Phase 8: Agentic Video Investigation
The Phase 8 Agent interacts with the Evidence Graph via `videx.graph.query.GraphQueryService`:
1. `events_between(video_id, start_sec, end_sec)`: Finds chronological events within an investigation time window.
2. `events_involving_track(track_id)`: Traces all events where a specific tracked entity was a participant.
3. `evidence_for_event(event_id)`: Fetches the immutable evidence chain (frames, crops, audio, OCR) supporting an event.
4. `temporal_neighbors(event_id, window_sec)`: Retrieves upstream and downstream events for causal chain hypothesis testing.
5. `find_path(source_node_id, target_node_id)`: Discovers multi-hop relational pathways between disparate entities (e.g., Track $A \to$ Event $E \to$ Zone $Z \leftarrow$ Track $B$).

### 5.2 Phase 9: Interactive UI
The frontend renders:
1. **Interactive Spatiotemporal Timeline:** Chronological tracks, audio waveforms, OCR appearances, and events projected from graph temporal nodes.
2. **Evidence Inspector:** Clicking any node surfaces its complete provenance: source video, frame image URI, bounding box coordinates, and model confidence.
3. **Sub-graph Visualizer:** Localized graph view centered on selected events showing participating tracks, zones, and supporting evidence.
