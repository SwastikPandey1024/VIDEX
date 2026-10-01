# VIDEX Phase 6.0 — Semantic Intelligence Foundation Architecture

## Executive Summary

Phase 6.0 operationalizes the epistemological hierarchy formalized in **ADR-005**. It establishes **Layer 4 (The Semantic Router)** and **Layer 5 (The VLM Provider Boundary)** without prematurely executing unconstrained full-video inference or adding heavy GPU runtime dependencies.

```text
EventTimeline (Phase 5)
    │
    ▼
Candidate Semantic Event Selection (Layer 4)
    │  - Deterministic event clustering (temporal window: 2.5s)
    │  - Explainable saliency scoring (severity, modality, OCR, density)
    │  - Structured query token matching
    │  - Candidate prioritization & gating
    │
    ▼
Evidence Bundle Construction (Layer 4)
    │  - Temporal keyframe selection (leading, midpoint, trailing, instantaneous)
    │  - Visual crop extraction (15% context margin, frame clamping, geometry validation)
    │  - Cross-modal context aggregation (spatial context, OCR, transcripts)
    │  - Explicit evidence ID preservation
    │
    ▼
Semantic Router Gating & Guardrails (Layer 4)
    │  - Deterministic SHA-256 request hashing
    │  - Local TTL in-process cache (candidate deduplication)
    │  - Sliding-window token & rate budget enforcement
    │
    ▼
VLM Provider Boundary (Layer 5)
    │  - VLMProvider abstract protocol
    │  - MockVLMProvider for offline, deterministic testing
    │  - Qwen3VLAdapter (disabled by default, pluggable for local/API backends)
    │  - Strict epistemic prompt formulation (Observed Evidence vs Interpretation)
    │
    ▼
Evidence Validator (Layer 5)
    │  - Rejects hallucinated evidence IDs
    │  - Rejects hallucinated supporting event IDs
    │  - Enforces temporal bounds consistency
    │  - Enforces valid confidence interval [0, 1]
    │  - Enforces explicit abstention (INSUFFICIENT_EVIDENCE / UNCERTAIN)
    │
    ▼
Evidence-Backed Canonical Semantic Event
```

---

## 1. Architectural Invariants Enforced

1. **Never send unconstrained full video to a VLM:**
   Only bounded, minimal `EvidenceBundle` instances containing targeted keyframe crops, OCR text observations, and speech transcript segments are routed.
2. **Never rerun detection, tracking, OCR, or ASR unnecessarily:**
   All candidate generation and evidence packaging leverage pre-existing outputs from Phases 1–5 stored in `EventTimeline` and canonical `Evidence` records.
3. **Every semantic conclusion must reference supplied `evidence_ids`:**
   `EvidenceValidator` checks that every ID in `SemanticEventPayload.evidence_ids` is an exact subset of `EvidenceBundle.evidence_ids`. Any hallucination causes instant rejection.
4. **Semantic reasoning is distinct from deterministic events:**
   High-level interpretations are typed as `SemanticEventType` and stored in canonical `Event` instances with `layer="semantic_reasoning"` and `source_module="semantic_router:provider"`.
5. **Unsupported conclusions are rejected or marked abstained:**
   Statuses explicitly support `SUPPORTED`, `UNCERTAIN`, `INSUFFICIENT_EVIDENCE`, and `REJECTED`. Abstained results cannot become falsely confirmed events.
6. **Preserve authoritative PTS/timestamp semantics:**
   `FrameTimestamp` seconds and PTS from container demuxing are maintained throughout keyframe selection, crop extraction, and temporal intervals.
7. **Keep VLM implementation behind a provider interface:**
   `VLMProvider` protocol decouples router logic from concrete models. `MockVLMProvider` provides complete offline CPU verification, while `Qwen3VLAdapter` is isolated behind execution modes (`disabled`, `mock`, `api`, `local_cpu`, `local_gpu`).
8. **No graph databases or heavy external infrastructure:**
   Lightweight in-process caching, deterministic hashing, and structured Pydantic domain models are used without Neo4j, Redis, or PostgreSQL migrations.
9. **No premature agentic workflows or frontend overhead:**
   Clean functional pipelines with explainable telemetry outputs.

---

## 2. Layer 4: Semantic Router

### 2.1 Candidate Selection & Explainable Saliency (`CandidateSelector`)
- **Clustering:** Groups temporally proximate events within a sliding window (default 2.5 seconds).
- **Explainable Saliency:** Saliency is deterministically computed from:
  - Base event severity weights (`critical=1.00`, `high=0.80`, `medium=0.60`, `low=0.35`, `info=0.20`)
  - Event type weights (`anomaly=0.90`, `custom/compound=0.75`, `entered_zone=0.65`, etc.)
  - Modality diversity bonus (`+0.25` for audio + spatial/motion co-occurrence)
  - Text presence bonus (`+0.15` when OCR observations are present)
  - Event density boost (`min(0.15, (count - 1) * 0.05)`)
- **Query Relevance:** Tokenizes natural language queries, eliminates stop words, and computes deterministic lexical overlap against candidate metadata, zone names, and participant labels.
- **Priority Tiering:** Classifies candidates into `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`.

### 2.2 Evidence Bundle Construction (`EvidenceBundleBuilder`)
- **Keyframe Selection:**
  - Interval events: selects `leading` (start), `midpoint` (center), and `trailing` (end) frames.
  - Instantaneous events: selects closest valid frame timestamp.
- **Crop Extraction (`CropExtractor`):**
  - Expands bounding boxes by a configurable context margin (default 15%).
  - Clamps coordinates to `[0, frame_width]` and `[0, frame_height]`.
  - Rejects degenerate boxes (< 1.0 px width or height).
  - Preserves frame index, timestamp seconds, and source `evidence_id`.
- **Multimodal Context:** Aggregates time-bounded OCR text and speech transcript segments.

### 2.3 Semantic Cache & Budget Management (`SemanticCache`, `RoutingPolicy`)
- **Request Hashing:** Computes deterministic SHA-256 hash across candidate ID, temporal bounds, cited evidence IDs, query string, and provider name.
- **Budget Guardrails:** Tracks request and token consumption across sliding 60-second windows. If limits are reached, candidates are cleanly suppressed (`SUPPRESS_BUDGET_EXCEEDED`) with structured telemetry.

---

## 3. Layer 5: VLM Provider Boundary & Validation

### 3.1 VLM Provider Interface & Qwen3-VL Adapter
- `VLMProvider` protocol defines:
  ```python
  def analyze(self, bundle: EvidenceBundle, request: RoutingRequest) -> SemanticEventPayload: ...
  ```
- `MockVLMProvider`: Allows setting deterministic test scenarios (confidence, status, synthetic hallucinations, artificial latencies).
- `Qwen3VLAdapter`: Implements `VLMProvider` with lazy module checks (`importlib.util.find_spec`) so the repository can run on any environment without `torch` or `transformers` installed.

### 3.2 Epistemic Prompt Contract
The prompt strictly bifurcates:
```text
=== OBSERVED EVIDENCE (CANONICAL GROUND TRUTH) ===
- Temporal Interval: [t_start, t_end]
- Visual Evidence IDs: ...
- Audio Transcript Segments: ...
- Text / OCR Observations: ...

=== MODEL INTERPRETATION (REASONING OBJECTIVE) ===
- Only reason from the supplied evidence above.
- Do NOT hallucinate objects, actions, or timestamps.
- Cite evidence_ids directly from the supplied list.
- Abstain if evidence is ambiguous or incomplete.
```

### 3.3 Evidence Validator (`EvidenceValidator`)
Validates every `SemanticEventPayload` before canonical conversion:
1. `validate_evidence_ids`: Ensures all cited IDs exist in `bundle.evidence_ids`.
2. `validate_event_ids`: Ensures all cited event IDs exist in `bundle.supporting_event_ids`.
3. `validate_temporal_bounds`: Ensures semantic interval falls within bundle bounds (with 0.5s tolerance).
4. `validate_confidence_and_status`: Checks valid float bounds `[0.0, 1.0]` and enforces status consistency.

### 3.4 Phase 6.1: Pluggable VLM Runtime Configuration & Factory

Phase 6.1 introduces a centralized provider factory `create_vlm_provider()` and formalizes environment variable mappings for multi-environment VLM execution without introducing hard runtime dependencies:

- **Environment Configuration:**
  - `VIDEX_SEMANTIC_PROVIDER` (or `VIDEX_VLM_PROVIDER`): `mock` (default), `qwen` / `qwen3_vl`.
  - `VIDEX_SEMANTIC_EXECUTION_MODE` (or `VIDEX_VLM_EXECUTION_MODE`): `disabled`, `mock`, `api`, `local_cpu`, `local_gpu`.
  - `VIDEX_SEMANTIC_MODEL` (or `VIDEX_VLM_MODEL`): `Qwen/Qwen2.5-VL-7B-Instruct` (default) or any compatible endpoint model.
  - `VIDEX_SEMANTIC_API_BASE_URL` (or `VIDEX_VLM_API_BASE_URL`): OpenAI-compatible or vLLM server endpoint.
  - `VIDEX_SEMANTIC_API_KEY` (or `VIDEX_VLM_API_KEY`): Bearer token for remote VLM API access (never committed to repository).

- **Execution Mode Support:**
  1. `mock`: Fully offline deterministic simulation with zero external dependencies.
  2. `api`: Dispatches OpenAI/vLLM-compatible Chat Completions API requests over HTTP.
  3. `local_cpu` / `local_gpu`: Leverages HuggingFace `transformers` and `torch` with automatic device placement when installed.
  4. `disabled`: Provider reports `is_available() == False` and abstains gracefully.

---

## 4. Verification & Quality Gates

- **Unit Tests:** 20 passing unit tests in `tests/unit/test_semantic.py`.
- **Acceptance Tests:** 4 passing end-to-end integration tests in `tests/integration/test_semantic_acceptance.py`.
- **Corpus Evidence Integrity Tests:** 4 passing tests in `tests/integration/test_corpus_evidence_integrity.py`.
- **Regression Suite:** 296 passed, 1 skipped across the entire VIDEX repository.
- **Type Safety & Linting:** 100% clean `ruff check .` and `mypy src` (0 errors across 68 source files).
- **Smoke Tests:** Validated all 4 subsystem smoke scripts (`semantic`, `events`, `ocr`, `audio`).
- **Sample Corpus Validation:** Evaluated all 8 videos in `Sample_Videos/` via `scripts/evaluate_sample_corpus.py` with 100% evidence integrity pass rate.
