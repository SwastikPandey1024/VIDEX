# VIDEX Agentic Investigator Architecture & UI Contract

## 1. Overview and Core Principle

The VIDEX Investigator is an evidence-grounded agentic query reasoning engine designed to answer complex user questions over processed video data. It is **not** a generic chatbot; it is a deterministic, auditable system that plans and executes typed read-only tools over canonical spatiotemporal evidence, events, tracks, OCR, transcripts, and the evidence graph.

### Core Investigation Loop

```text
User Question
    ↓
Intent Understanding & Classification (TEMPORAL, TRACK, OCR, AUDIO, SPATIAL, RELATIONAL, SEMANTIC, EVENT, OBJECT)
    ↓
Investigation Plan (bounded steps, required tools)
    ↓
Typed Tool Calls (read-only, provenance-preserving)
    ↓
Evidence Retrieval & Graph Traversal
    ↓
Evidence Sufficiency Gate (evidence grounding and verification check)
    ↓
Optional Bounded Semantic Reasoning (via VLMProvider boundary)
    ↓
Claim Validation (epistemic tagging & video namespace isolation)
    ↓
Auditable Answer & Structured Investigation Result
```

### Grounding Design Principle

**All emitted claims must cite supporting evidence IDs verified within the target video namespace.** If canonical evidence or graph associations are insufficient to corroborate an answer, the system enters an abstention state, returns `INSUFFICIENT_EVIDENCE`, and specifies what evidence was lacking.

---

## 2. Agent Boundary and Security Constraints

### Allowed Capabilities (Read-Only)

- Query canonical `Evidence`, `Event`, `Track`, `Frame`, `Scene`, `OCRObservation`, and `TranscriptSegment` records.
- Query and traverse `InMemoryGraphStore` via `GraphQueryService` (temporal neighbors, spatial neighbors, related events, evidence chains).
- Seek video frames by time offset or frame index.
- Compare frames across distinct points in time.
- Selectively invoke semantic reasoning through the existing `VLMProvider` and `EvidenceBundle` boundary when deterministic evidence is incomplete.

### Forbidden Capabilities

- **NO mutation:** The agent cannot write or mutate canonical Evidence, Events, or Graph nodes/edges.
- **NO arbitrary SQL or shell commands:** The agent has no shell or raw SQL execution permissions.
- **NO unrestricted network or filesystem access:** Only designated in-memory and video asset paths.
- **NO unrestricted raw video ingestion:** The VLM receives selected, targeted evidence crops and keyframes, never unbounded video streams.
- **NO invented evidence IDs:** Every claim must cite actual, verifiable evidence identifiers existing within the requested video namespace.

---

## 3. Epistemic State Handling

The system preserves strict epistemic separation across all investigation phases. Claims and evidence results are tagged with one of four explicit epistemic statuses:

| Epistemic Status | Description | Typical Source |
| --- | --- | --- |
| `deterministically observed` | Ground truth derived directly from perception or detectors (tracks, timestamps, OCR, transcripts, canonical events). | ByteTrack, YOLO, PaddleOCR, Whisper, temporal event engine |
| `heuristically associated` | Association inferred via spatial IoU overlap, proximity heuristics, or temporal co-occurrence windows. | Evidence Graph spatial/temporal neighbor traversals |
| `VLM inferred` | Higher-level semantic interpretation produced by the vision-language model boundary. | VLMProvider / SemanticRouter |
| `insufficient evidence` | Explicit marker denoting that available evidence fails to corroborate a claim. | Truthful abstention gate |

Responses and UI components must never flatten these into equivalent facts.

---

## 4. Tool Registry and Contracts

The system exposes 16 typed read-only tools through a controlled `ToolRegistry`:

1. **`search_events`**: Queries temporal events within an optional time window or involving specific tracks.
2. **`get_event`**: Retrieves canonical Event details by ID with participants and supporting evidence.
3. **`get_track`**: Retrieves track kinematics, bounding boxes, and duration by track ID.
4. **`get_object`**: Retrieves persistent domain objects or detections by class name.
5. **`get_frames`**: Retrieves discrete video frames by ID with PTS timestamps and storage paths.
6. **`get_evidence`**: Retrieves canonical Evidence grounding records by evidence UUID or event ID.
7. **`get_ocr`**: Queries text snippets recognized by OCR within temporal or spatial filters.
8. **`get_transcript`**: Queries transcribed speech audio segments within a temporal window.
9. **`get_scene`**: Retrieves scene shot boundaries and keyframe intervals.
10. **`query_graph`**: Executes filtered node queries over the Evidence Graph.
11. **`find_related_events`**: Finds topologically or causally connected events in the Evidence Graph.
12. **`find_temporal_neighbors`**: Retrieves co-occurring events within `window_sec` of an event.
13. **`find_spatial_neighbors`**: Retrieves nodes sharing bounding box overlap or zone containment.
14. **`seek_video`**: Seeks video metadata and keyframe references at a specific timestamp.
15. **`compare_frames`**: Compares frame metadata across two distinct temporal points.
16. **`reason_semantic`**: Invokes bounded VLM multimodal analysis over a targeted evidence query.

Every tool returns a structured `ToolResult` containing:

- `tool_name`: Name of executed tool
- `success`: Boolean execution status
- `data`: Typed payload dictionary
- `provenance`: List of `NodeProvenance` records capturing source IDs, derivation types, and confidence scores

---

## 5. Execution Safeguards and Boundaries

The `InvestigationExecutor` enforces strict operational limits to guarantee bounded execution:

- **Maximum Steps Ceiling**: Defaults to 10 steps per investigation.
- **Maximum Tool Calls Ceiling**: Defaults to 20 tool calls per investigation.
- **Timeout Budget**: Default 10.0 seconds total per investigation.
- **Deduplication Cache**: Exact duplicate tool calls (same tool name and arguments) are detected and returned from cache without re-querying.
- **Deterministic Step Ordering**: Ensures reproducible execution traces for auditing.

---

## 6. Evidence Grounding and Claim Auditing Gates

Before an investigation completes, all formulated claims must pass two verification gates:

### EvidenceSufficiencyGate

Verifies that:

- Supporting evidence records exist in canonical stores or the Evidence Graph.
- Video isolation is strictly enforced (`video_id` must match target video).
- Timestamps are non-negative and valid (`t_start <= t_end`).
- Referenced participants and events exist.
- Epistemic status is non-empty.

### ClaimValidator

Audits candidate claims against retrieved evidence context:

- Rejects any `SUPPORTED` claim that lacks cited `evidence_ids`.
- Rejects any claim citing unknown or unretrieved evidence IDs.
- Validates confidence bounds (`0.0 <= confidence <= 1.0`).
- Flags uncorroborated assertions as `INSUFFICIENT_EVIDENCE` or `REJECTED`.

---

## 7. Phase 9 UI Consumption Contract

Phase 9 (Web / Desktop UI) will consume the following data structures emitted by the Investigator:

### 1. `InvestigationResult`

The top-level response payload:

```typescript
interface InvestigationResult {
  investigation_id: string;        // UUIDv4
  video_id: string;                // Target video UUID
  question: string;                // Original user query
  status: InvestigationStatus;     // "completed" | "insufficient_evidence" | "timeout" | "failed"
  answer: string;                  // Grounded, human-readable markdown response
  claims: Claim[];                 // Individual auditable claims
  evidence_references: EvidenceReference[]; // Citable evidence objects
  events_used: string[];           // Event IDs contributing to the answer
  tools_executed: string[];        // Sequence of tools invoked
  trace: InvestigationTrace;       // Complete execution and audit trace
  metrics: {
    planning_latency_ms: number;
    tool_latency_ms: number;
    graph_latency_ms: number;
    evidence_latency_ms: number;
    validation_latency_ms: number;
    total_latency_ms: number;
    step_count: number;
    tool_call_count: number;
  };
}
```

### 2. `Claim`

Structured factual assertions for interactive verification widgets:

```typescript
interface Claim {
  claim_id: string;                // UUID
  claim_text: string;              // Specific statement
  confidence: number;              // Float in [0.0, 1.0]
  epistemic_status: string;        // "deterministically observed" | "heuristically associated" | "VLM inferred"
  status: ClaimStatus;             // "supported" | "uncertain" | "insufficient_evidence" | "rejected"
  evidence_ids: string[];          // Clickable evidence chips in UI
  event_ids: string[];             // Related timeline event markers
  timestamps: [number, number] | null; // [start_seconds, end_seconds] for video seeking
  participant_ids: string[];       // Track IDs for bounding box overlay
  video_id: string;
}
```

### 3. `EvidenceReference`

Interactive evidence citation cards in the UI:

```typescript
interface EvidenceReference {
  evidence_id: string;
  evidence_type: string;           // "detection" | "track" | "ocr" | "audio_transcript" | "frame"
  video_id: string;
  timestamp_seconds: number | null;
  confidence: number;
  description: string;
  frame_id?: string;
  bounding_box?: { x: number; y: number; width: number; height: number };
}
```

### 4. `InvestigationTrace`

Explainability panel data structure:

```typescript
interface InvestigationTrace {
  investigation_id: string;
  video_id: string;
  user_question: string;
  plan: {
    category: string;             // e.g. "TEMPORAL", "TRACK", "SEMANTIC"
    objective: string;
    planned_steps: string[];
    required_tools: string[];
  };
  steps: {
    step_id: string;
    step_number: number;
    action: { tool_name: string; parameters: Record<string, any> };
    result: { success: boolean; data: Record<string, any>; provenance: any[] };
    latency_ms: number;
  }[];
  evidence_ids: string[];
  events_used: string[];
  semantic_calls: number;
  total_latency_ms: number;
  validation_passed: boolean;
}
```

### 5. `TimelineReference`

For jumping the video player scrubber directly to relevant segments:

```typescript
interface TimelineReference {
  event_id: string;
  start_pts: number;
  end_pts: number;
  label: string;
  evidence_ids: string[];
}
```

---

## 8. Observed Performance Benchmarks

Measured on the 8-video VIDEX corpus during the Phase 8 validation run:

| Metric | Measured Value |
| --- | --- |
| Average Planning Latency | ~0.10 ms |
| Average Tool Execution Latency | ~0.15 ms |
| Average Claim Validation Latency | ~0.18 ms |
| Total End-to-End Investigation Latency | ~0.71 ms |
| Total Acceptance Test Suite Duration | 1.12 seconds (10 tests) |
| Total Test Suite Pass Rate | 336 passed, 1 skipped (0 failures) |

*(Note: These figures reflect the in-memory graph store and deterministic/mock VLM provider. Live Qwen3-VL neural inference will add GPU forward pass latency in production).*
