# VIDEX — System Design

**Version:** 0.2 (Phase 0 Review & Architecture Alignment)  
**Last Updated:** 2026-09-28  
**Status:** Approved Specification — Phase 0 baseline verified, models proposed pending benchmarks  

---

## 1. Architectural Overview & Philosophy

VIDEX (Video Intelligence & Evidence Extraction) is an open-source, model-agnostic system designed to transform raw, unconstrained video streams and files into timestamped, structured, and auditable multimodal intelligence records.

### Core Architectural Principles

1. **Strict Model & Provider Agnosticism:** Every AI/ML model operates behind an abstract Python protocol (`src/videx/providers/base.py`). Concrete backends can be swapped via configuration without touching orchestration or API logic.
2. **Explicit Candidate Status:** No model is marked "Accepted" or "benchmark-proven" until reproducible empirical evaluation has taken place on domain-specific footage. All model selections remain **Proposed** during Phase 0.
3. **Decoupled Perception and Temporal Intelligence:** Low-level detectors and trackers feed an independent **Temporal Event Engine** that synthesizes trajectory, lifecycle, and rule-based events *before* heavy semantic reasoning is engaged.
4. **Distinction between Evidence Store and Evidence Graph:** Structured observations (detections, tracks, audio segments, frames) are durably persisted in the **Evidence Store** (PostgreSQL primary, SQLite local/dev option, S3/object storage for media files). The **Evidence Graph** is an associative graph-like relationship model over persisted evidence without introducing a dedicated graph database.
5. **Conditional Heavy Inference:** Resource-intensive models (YOLOE-26, SAM 3.1, Action Recognition, InternVideo3) are **explicitly conditional** and never run indiscriminately on every frame.
6. **Bilingual & Vernacular First:** Built-in routing for Hindi (Devanagari script), English, and code-switched conversational Hinglish.

---

## 2. Canonical Architecture Diagrams

The definitive visual models for VIDEX are maintained as PlantUML diagrams within `docs/architecture/`:

| Diagram | Source File | Description |
|---|---|---|
| **01 System Architecture** | [01-system-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/01-system-architecture.puml) | High-level component interactions from ingestion to UI |
| **02 Processing Pipeline** | [02-processing-pipeline.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/02-processing-pipeline.puml) | End-to-end dataflow across video, audio, temporal, and fusion stages |
| **03 Technology Architecture** | [03-technology-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/03-technology-architecture.puml) | Model selections, conditional components, and routing topology |
| **04 Evidence Graph** | [04-evidence-graph.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/04-evidence-graph.puml) | Domain schema, entity attributes, and relational graph links |
| **05 Deployment Architecture** | [05-deployment-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/05-deployment-architecture.puml) | Cloud/hybrid infrastructure, storage, GPU workers, and API hosting |
| **06 Processing Sequence** | [06-processing-sequence.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/06-processing-sequence.puml) | Runtime event and message lifecycle across asynchronous services |

---

## 3. End-to-End Processing Pipeline

The canonical flow follows a linear data pipeline that cascades from raw physical ingestion to graph-indexed intelligence:

```
Video Input (File / Stream / URL)
       │
       ▼
1. INGESTION & VALIDATION (FFmpeg, OpenCV, PySceneDetect)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
2. VIDEO PIPELINE                 3. AUDIO PIPELINE
   • Scene / Shot Boundaries         • Audio Extraction
   • Frame Router                    • faster-whisper ASR
   • Object Detection (YOLO26)       • Speech Timestamps
   • Tracking (BoT-SORT / ByteTrack) • Sound Event Detection (Conditional)
   • OCR Router (PP-OCRv6 / Indic)
   • Temporal OCR Fusion
       │                                 │
       └────────────────┬────────────────┘
                        ▼
4. TEMPORAL EVENT ENGINE
   • Trajectory & Velocity Analysis
   • Lifecycle Tracking (Appear, Dwell, Disappear)
   • Spatial Boundaries (Zone Entry / Exit)
   • Deterministic Event Generation
                        │
                        ▼
5. EVIDENCE STORE (Durable Persistence)
   • PostgreSQL: Primary structured metadata & evidence store
   • SQLite: Local development / testing option
   • S3 / Object Storage: Video & media artifact storage (not a database)
   • Persists: Video, Frames, Detections, Tracks, OCR, Audio, Raw Evidence
                        │
                        ▼
6. SEMANTIC ROUTER
   • Decision router for visual-language reasoning
   ├── Route A (Default Candidate): Qwen3-VL (Short / Medium Horizon)
   └── Route B (Conditional Route): InternVideo3 (Long-Horizon Research)
                        │
                        ▼
7. EVENT FUSION LAYER
   • Correlates: Frames ↔ Tracks ↔ OCR ↔ Audio ↔ Semantic Explanations
   • Resolves cross-modal temporal alignment and deduplication
                        │
                        ▼
8. EVIDENCE GRAPH (Relationship Model over Persisted Evidence)
   • Graph-like relational model (no dedicated graph database)
   • Structured entity-relationship graph connecting all observations
   • Enables multi-hop queries, causal chains, and evidence verification
                        │
                        ▼
9. SEARCH & SUMMARY GENERATOR
   • Structured Video Reports
   • Hybrid Lexical / Vector Search (when justified)
                        │
                        ▼
10. FASTAPI BACKEND (REST API)
                        │
                        ▼
11. REACT + TYPESCRIPT INTELLIGENCE UI
```

---

## 4. Subsystem Specifications & Model Candidates

### 4.1 Ingestion & Scene Segmentation
- **Responsibilities:** Validates input codecs, container headers, FPS, and audio presence. Segments video into logical scenes/shots using content-aware thresholding (`PySceneDetect`).
- **Frame Router:** Drops redundant frames; samples keyframes dynamically based on motion intensity and scene cut events.

### 4.2 Multimodal Perception
- **Object Detection (ADR-001):**
  - *Baseline Candidate:* **YOLO26**. Chosen for optimal real-time throughput and standard class accuracy.
  - *Conditional Candidate:* **YOLOE-26**. Activated *only* when the fixed class vocabulary becomes a measurable limitation for domain queries (open-vocabulary zero-shot).
  - *Status:* **Proposed** (Awaiting benchmark).
- **Multi-Object Tracking (ADR-002):**
  - *Default Candidate:* **BoT-SORT**. Kalman filtering + camera motion compensation.
  - *Alternative Candidate:* **ByteTrack**. High-speed motion-only association without appearance modeling.
  - *Conditional ReID:* Deep appearance embeddings are enabled *only* if benchmarks prove unacceptable ID switching in crowded scenarios.
  - *Status:* **Proposed** (Awaiting benchmark).
- **Optical Character Recognition (ADR-003):**
  - *Architecture:* **OCR Router**.
  - *Supported Scripts Candidate:* **PP-OCRv6** for Latin, numbers, and standard video overlays.
  - *Specialized Route:* **Hindi / Indic OCR** for complex Devanagari ligatures and vernacular scene text.
  - *Temporal OCR Fusion:* Aggregates, tracks, and votes on text strings across consecutive frames to eliminate transient OCR flicker and misrecognitions.
  - *Status:* **Proposed** (Awaiting benchmark).
- **Audio & Speech Recognition (ADR-004):**
  - *Pipeline Components:*
    - **Audio Extraction:** High-fidelity demuxing and conversion via FFmpeg.
    - **faster-whisper ASR (Baseline Candidate):** Efficient CTranslate2 transcription in Hindi and English.
    - **Speech Timestamps:** Segment-level and word-level temporal alignment against video clock.
    - **Sound Event Detection (Conditional):** Non-speech acoustic event classification (e.g., sirens, alarms, crashes) when relevant to domain.
  - *Specialized Hindi Route:* **IndicWhisper / AI4Bharat** is conditional, evaluated only if `faster-whisper` exhibits high WER on regional accents.
  - *Status:* **Proposed** (Awaiting benchmark).

### 4.3 Advanced Perception (Explicitly Conditional)
These components are **not mandatory** and are never run on the full video stream:
- **SAM 3.1 (Segment Anything Model):** Invoked conditionally on selected keyframes where pixel-exact segmentation masks or fine-grained boundary evidence is required.
- **Action Recognition (VideoMAE / MMAction2):** Invoked conditionally over targeted temporal intervals flagged by the Temporal Event Engine where complex human actions require spatio-temporal classification.

### 4.4 Temporal Event Engine
- Placed **before** semantic reasoning to keep processing efficient and deterministic.
- Analyzes trajectory coordinate streams, bounding box deltas, velocity, directional vectors, and dwell times.
- Detects foundational operational events:
  - Object entrance, presence, dwell, and disappearance.
  - Zone boundary crossings (polygon intersection).
  - Rapid movement or abnormal acceleration.
- Emits candidate events with associated timestamps and track references to prune what needs deep semantic evaluation.

### 4.5 Evidence Store vs. Evidence Graph (Core Distinction)

A critical distinction exists between the **Evidence Store** and the **Evidence Graph**:

```
┌────────────────────────────────────────────────────────┐
│                    EVIDENCE STORE                      │
│  • Structured persistence & media artifact storage     │
│  • PostgreSQL: Primary structured metadata & evidence  │
│    persistence                                         │
│  • SQLite: Local development & testing option          │
│  • S3 / Object Storage: Video, media & artifact store  │
│    (binary/media storage, NOT a database)              │
│  • Persists: Video, Frames, Detections, Tracks,        │
│    OCRObservation, AudioSegment, Evidence items        │
│  • Immutable audit trail and physical storage log      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (Event Fusion & Graph Model Construction)
┌────────────────────────────────────────────────────────┐
│                    EVIDENCE GRAPH                      │
│  • Graph-like relationship model over persisted        │
│    evidence (NOT a dedicated graph database)           │
│  • Entities: Video, Scene, Track, OCR, Audio, Event    │
│  • Edges: participates_in, supports, contextualizes,   │
│    temporally_aligned, spatial_proximity               │
│  • Enables multi-hop reasoning, timeline correlation,  │
│    causal tracing, and evidence-backed reporting       │
└────────────────────────────────────────────────────────┘
```

- **Evidence Store:** Focuses on *persistence, integrity, and retrieval of raw observational data and media*:
  - **PostgreSQL:** Primary production storage for structured metadata, detections, tracks, OCR observations, audio transcripts, events, and evidence records.
  - **SQLite:** Supported for local development, rapid prototyping, and automated testing where appropriate without requiring an external database service.
  - **S3 / Object Storage:** Dedicated binary storage for raw uploaded videos, decoded keyframes, visual crops, and generated media artifacts. S3/object storage is used strictly for media and file storage, **not** as a database.
- **Evidence Graph:** Focuses on *relationships, causality, and contextual meaning*:
  - An associative, graph-like relationship model constructed directly over persisted evidence records.
  - **No dedicated graph database** (such as Neo4j) is introduced; the graph model is implemented through relational foreign keys, associative adjacency structures, and query-time traversals over the persisted Evidence Store.
  - Models how an audio snippet correlates with an OCR text overlay, which vehicle participated in an incident event, and which specific video frame serves as ground-truth proof. Detailed schema defined in [04-evidence-graph.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/04-evidence-graph.puml).

### 4.6 Semantic Reasoning Router
- **Semantic Router:** Evaluates candidate event intervals and user queries to route tasks to the appropriate multimodal reasoning engine:
  - **Qwen3-VL (Default / Practical Candidate):** Primary visual-language model for short-to-medium horizon comprehension, frame question-answering, scene captioning, and structured entity attribute extraction.
  - **InternVideo3 (Conditional Long-Horizon Route):** Dedicated research layer for long-context temporal video reasoning. Activated *only* when multi-minute video continuity or holistic narrative reasoning exceeds the practical context window of Qwen3-VL.

### 4.7 Event Fusion Layer
- Synthesizes deterministic events from the Temporal Event Engine with high-level descriptions from the Semantic Router.
- Fuses cross-modal evidence (e.g., matching a detected license plate OCR observation with a tracked vehicle, or correlating a spoken warning in audio with a sudden departure event).
- Populates the **Evidence Graph** relationship model with verified links between events and supporting evidence.

---

## 5. Domain Model & Protocol Layer

All domain models are defined in `src/videx/domain/schemas.py` and enforce strict validation via Pydantic v2:

| Entity | Purpose | Storage Destination |
|---|---|---|
| `Video` | Video registration, duration, codec, resolution metadata | Evidence Store (PostgreSQL / SQLite) |
| `Scene` | Scene change boundaries and shot segments | Evidence Store (PostgreSQL / SQLite) |
| `Frame` | Decoded keyframe with timestamp and storage URI | Evidence Store (PostgreSQL / SQLite + S3) |
| `BoundingBox` | Normalized coordinates (`[0.0, 1.0]`) and pixel helpers | Embedded in observations |
| `Detection` | Single-frame object observation with class and confidence | Evidence Store (PostgreSQL / SQLite) |
| `Track` | Associated multi-frame object trajectory with lifecycle | Evidence Store (PostgreSQL / SQLite) |
| `TrajectoryPoint` | Specific spatial coordinate at a timestamp | Evidence Store (PostgreSQL / SQLite) |
| `OCRObservation` | Text observation with bounding box and temporal lifespan | Evidence Store (PostgreSQL / SQLite) |
| `AudioSegment` | Timestamped audio transcript and speech metrics | Evidence Store (PostgreSQL / SQLite) |
| `Event` | Verified incident (e.g., zone crossing, interaction) | Evidence Store (PostgreSQL / SQLite) |
| `Evidence` | Explicit reference linking an event to frame/track/audio proof | Evidence Store (PostgreSQL / SQLite) |

All perception engines implement standard `typing.Protocol` interfaces in `src/videx/providers/base.py`:
- `DetectionProvider`
- `TrackingProvider`
- `OCRProvider`
- `ASRProvider`
- `VideoReasoningProvider`
- `EventDetector`

---

## 6. Deployment Architecture

As modeled in [05-deployment-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/05-deployment-architecture.puml):
- **Development Profile:** 100% CPU-runnable using mock providers or lightweight quantizations (`faster-whisper` int8, CPU-friendly detection/tracking) with local **SQLite** and local filesystem storage.
- **Production Profile:**
  - FastAPI stateless backend on CPU instances (e.g., AWS EC2 or container service).
  - Asynchronous background job queue (e.g., Celery/SQS).
  - Isolated GPU inference workers running heavy model containers (YOLO26, BoT-SORT, Qwen3-VL).
  - **Storage Architecture:**
    - **PostgreSQL (+ pgvector):** Primary structured metadata and evidence persistence, hosting relational tables and the Evidence Graph relationship model.
    - **S3 / Object Storage:** Media and artifact storage for raw video assets, keyframe images, and visual crops (strictly object storage, not a database).

---

## 7. Open Decisions Before Phase 1 Promotion

1. **Benchmarking Execution:** Run the formal benchmarking protocols defined in ADR-001 through ADR-004 on domain-specific Hindi video footage.
2. **Promote Baseline Candidates:** Upgrade ADR status from `Proposed` to `Accepted` only upon verified latency and accuracy criteria.
3. **Threshold Tuning for Conditional Models:** Define quantitative performance triggers for activating YOLOE-26, SAM 3.1, Action Recognition, and InternVideo3.
