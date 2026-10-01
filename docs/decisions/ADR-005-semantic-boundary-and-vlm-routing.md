# ADR-005 — Epistemological Hierarchy, Deterministic Rule Boundaries, and Phase 6 Semantic Router with Qwen3-VL

**Date:** 2026-10-01  
**Status:** Approved Architectural Decision  
**Deciders:** Swastik Pandey  

---

## Context & Motivation

Phase 5 introduced the **Temporal Event Intelligence Engine** to VIDEX, establishing a pure CPU-based, deterministic foundation for event extraction across object lifecycles, kinematics, spatial zones, OCR, and speech audio. As part of Phase 5, the `CrossModalRuleEngine` was added to correlate multi-modal observations, such as `speech during zone presence`.

While co-occurrence detection is necessary and useful, it represents the exact junction where video analytics systems are vulnerable to **architectural decay through rule creep**:

1. **The Hardcoded Heuristic Trap:** When engineering teams want high-level insights without an inference pipeline, they inevitably start hardcoding pseudo-semantic rules (e.g., `if speech_contains("help") and in_zone("vault") -> "attempted_robbery"`). This creates brittle, combinatorial rule explosions that break on minor linguistic or visual variations and lack contextual nuance.
2. **The Naive VLM Trap:** Conversely, feeding raw, unsegmented video streams directly into a large Vision-Language Model (VLM) like Qwen3-VL is economically prohibitive, computationally slow, and prone to hallucinations when not grounded in verified spatiotemporal evidence.

To preserve VIDEX's core mission of verifiable, auditable intelligence, we must establish a **strict epistemological boundary** before embarking on Phase 6 (Semantic Reasoning).

---

## The 5-Layer Epistemological Hierarchy

Going forward into Phase 6 and beyond, VIDEX enforces a strict unidirectional information pipeline:

```text
┌────────────────────────────────────────────────────────┐
│ Layer 1: RAW EVIDENCE                                 │
│ (Frame pixels, BBoxes, Trajectories, OCR, Audio PCM)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 2: DETERMINISTIC EVENTS                          │
│ (State transitions: EnteredZone, SpeechStarted, Moved) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 3: DETERMINISTIC RELATIONSHIPS                   │
│ (Allen's interval algebra, Spatial containment,        │
│  Cross-modal co-occurrence: Speech during presence)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 4: CANDIDATE SEMANTIC EVENTS & SEMANTIC ROUTER   │
│ (Saliency filtering, query triage, bounding crop pack) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 5: VLM / SEMANTIC REASONING (Qwen3-VL)           │
│ (Evidence-grounded open-world reasoning & explanation) │
└────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Responsibilities & Boundaries

| Layer | Component | Subsystem Responsibility | Explicit Restrictions |
|---|---|---|---|
| **1. Raw Evidence** | Ingestion, YOLO26, BoT-SORT, PaddleOCR, Faster-Whisper | Capture physical signals and derive localized observations (`Detection`, `TrajectoryPoint`, `TextObservation`, `TranscriptSegment`). | Zero interpretation. Must be immutable and anchored to Presentation Timestamps (PTS). |
| **2. Deterministic Events** | `LifecycleEventDetector`, `MovementEventDetector`, `SpatialEventDetector`, `OCREventDetector`, `AudioEventDetector` | Detect discrete state transitions of a single entity/modality (e.g., `OBJECT_ENTERED_ZONE`, `SPEECH_STARTED`). | Pure mathematical & boolean algorithms (Jordan curve, velocity thresholds, debouncing). Zero semantic inference. |
| **3. Deterministic Relationships** | `TemporalRelationEngine`, `CrossModalRuleEngine` | Compute topological containment and Allen's temporal algebra (e.g., `OVERLAPS`, `DURING`, `NEAR_IN_TIME`). Emit structural co-occurrence events. | **STRICT BOUNDARY:** Must only report mathematical/topological co-occurrence (e.g., "Speech utterance co-occurred with presence in Zone A"). Must **never** assign intent, sentiment, or subjective narrative meaning. |
| **4. Candidate Semantic Events** | `SemanticRouter` | Assemble spatiotemporally proximate events and evidence into structured **interrogation packets**. Filter, score saliency, enforce token budgets, and extract visual crops. | Does not generate final semantic answers; prepares minimal visual/textual context payloads for VLM inspection. |
| **5. VLM / Semantic Reasoning** | `Qwen3-VL`, Semantic Agents | Execute open-world visual reasoning over cropped keyframes and transcript context. Answer complex queries ("Why did the vehicle stop?", "What item was exchanged?"). | Must link all synthesized conclusions back to the 6-part evidence chain (Layer 1–3). Ungrounded generation is rejected. |

---

## Scrutiny & Reframing of `CrossModalRuleEngine`

The `CrossModalRuleEngine` introduced in Phase 5 is hereby formalized as a **Layer 3 Deterministic Relationship Engine**, not a semantic interpreter:

- **What it is:** A high-throughput, deterministic correlation engine that links disjoint evidence streams using precise temporal windows and spatial predicates.
- **What it outputs:** Structural correlation events (`speech_in_zone`, `co_located_motion`) containing references to both supporting evidence records (`EventEvidence` with roles `"context"` and `"trigger"`).
- **What it is forbidden from doing:** It must never synthesize human action categories, behavioral assessments, or intent classifications (e.g. `trespassing_intent`, `commercial_transaction`, `hostile_altercation`). Such labels belong exclusively to Layer 5.

---

## Phase 6 Architecture: The Semantic Router & Qwen3-VL

### 1. Data Flow

```text
     Video Stream
      │        │
    Frames   Audio
      │        │
   Perception OCR/ASR
      │        │
      ▼        ▼
  Canonical Evidence
         │
         ▼
  Temporal Event Engine (Phase 5)
  (Lifecycle, Kinematics, Spatial, OCR, Audio, Relations)
         │
         ▼
    EventTimeline
         │
         ▼
 ┌───────────────────────────────────────────────┐
 │ Phase 6: Semantic Router                      │
 │                                               │
 │  1. Saliency Evaluation (Query relevance)     │
 │  2. Token Budget & Rate Limiting Guardrails   │
 │  3. Temporal Clustered Framing                │
 │  4. Visual Crop Extractor (BBox + Margin)     │
 │  5. Multimodal Prompt Assembler               │
 └──────────────────────┬────────────────────────┘
                        │
                        ▼
 ┌───────────────────────────────────────────────┐
 │ Qwen3-VL (Vision-Language Reasoner)           │
 │                                               │
 │  - Targeted image crops (not raw full video)  │
 │  - Grounded prompt with transcript & telemetry│
 │  - Structured JSON response schema            │
 └──────────────────────┬────────────────────────┘
                        │
                        ▼
           Evidence-Backed Semantic Event
           (With 6-Part Explainability Chain)
```

### 2. Semantic Router Key Responsibilities

1. **Selective Invocation (Token Optimization):**
   - In a 1-hour 30 FPS video (~108,000 frames), the Event Engine might emit 500 deterministic events.
   - The Semantic Router evaluates user search queries, active alerts, and saliency heuristics to select only the top 1–5% of events that genuinely require visual reasoning.
   - Saves >95% of VLM compute and operational cost.

2. **Precision Crop Extraction:**
   - Rather than sending full 4K or 1080p frames, the Router leverages the tracked bounding boxes from Layer 1 evidence.
   - Extracts cropped keyframes centered around the target participant with a 15% context margin, maximizing VLM spatial attention on the relevant interaction.

3. **Grounded Context Assembly:**
   - Packages the candidate packet with:
     - Exact timestamp interval $[t_{\text{start}}, t_{\text{end}}]$.
     - Participant tracking history and kinematic vectors.
     - Synchronized ASR speech transcript segments and OCR text.
     - Spatial zone context.

4. **Structured Output Enforcement:**
   - Forces Qwen3-VL to respond in a strict Pydantic-validated JSON schema (`SemanticEventPayload`).
   - Ensures any generated interpretation cites the specific `evidence_ids` passed in the context.

---

## Decision Drivers & Trade-Offs

| Decision Driver | Justification |
|---|---|
| **Auditability & Provenance** | Ensuring every semantic statement can be traced back to raw sensor frames and timestamps. |
| **Computational Efficiency** | Pure CPU processing through Layers 1–3; GPU/VLM compute is activated strictly on-demand via Layer 4. |
| **Maintainability** | Prevents the codebase from accumulating thousands of lines of fragile, ad-hoc `if/else` heuristic rules. |
| **Model Agnosticism** | `Qwen3-VL` is wrapped behind the provider interface (`VLMProvider`), allowing seamless swapping with future multimodal backends. |

---

## Negative Invariants (Enforced Rules)

1. **No Pseudo-Semantic Event Types in Deterministic Detectors:** Detectors must never emit subjective event types like `SUSPICIOUS_BEHAVIOR` or `VIOLENT_INTERACTION`.
2. **No Unconstrained Video Ingestion by VLM:** Qwen3-VL must never be used as a primary frame-by-frame object detector or motion tracker.
3. **No Ungrounded Semantic Claims:** Every semantic event generated in Phase 6 must retain the 6-part explainability chain linking to Layer 1 canonical evidence records.
