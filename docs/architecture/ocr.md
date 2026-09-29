# VIDEX Architecture: Optical Character Recognition (OCR Pipeline Foundation)

## 1. Overview & Architectural Boundary

Phase 3.0 establishes the Optical Character Recognition (OCR) evidence pipeline for VIDEX, connecting deterministic video ingestion to multilingual text extraction, language/script routing, multi-frame temporal fusion, and structured evidence production.

```text
                  VIDEO
                    │
                    ▼
         ┌─────────────────────┐
         │   Video Ingestion   │
         │ (OpenCVVideoReader) │
         └──────────┬──────────┘
                    │  DecodedFrame (frame_array, frame_index, FrameTimestamp)
                    ▼
         ┌─────────────────────┐
         │    OCRScheduler     │ (ALL_FRAMES / FIXED_INTERVAL / TEMPORAL_INTERVAL / SCENE_BOUNDARY / ROI Crop)
         └──────────┬──────────┘
                    │  Scheduled DecodedFrame / Region Crop
                    ▼
         ┌─────────────────────┐
         │      OCRRouter      │ (Language & Script-aware dispatch)
         └───────┬─────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌──────────────┐      ┌──────────────┐
│PaddleGeneral │      │ PaddleIndic  │
│ (PP-OCRv6)   │      │(Devanagari)  │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │  Raw OCRObservation[] (text, norm_text, conf, bbox, authoritative FrameTimestamp)
                  ▼
       ┌─────────────────────┐
       │  TemporalOCRFusion  │ (Clusters by text identity, spatial IoU/distance, and temporal continuity)
       └──────────┬──────────┘
                  │  Fused TextObservation[] (first_seen, last_seen, supporting_obs_ids, confidence_summary)
                  ▼
       ┌─────────────────────┐
       │   Evidence Layer    │ (EvidenceType.OCR records linked to upstream raw observation IDs)
       └─────────────────────┘
```

---

## 2. Core Architectural Principles

### 2.1 Authoritative FrameTimestamp Provenance (Zero Clock Drift)
The OCR layer must **never invent or calculate its own timestamps** from `frame_number / fps` or system clocks.
- Every `DecodedFrame` or `Frame` ingested by the OCR pipeline carries an authoritative, immutable `FrameTimestamp` object with:
  - `pts_seconds`: Authoritative Presentation Timestamp in seconds.
  - `frame_index`: Sequential zero-based frame index.
  - `timestamp_source`: Explicit provenance (`TimestampSource.CONTAINER` vs `TimestampSource.DERIVED`).
  - `is_repaired`: Boolean flag indicating whether container timestamp jitter or reordering was sanitized.
  - `repair_reason`: Explanation of timestamp repair (e.g., `backward_pts`, `missing_pts`, `duplicate_pts`, `interpolated`).
  - `original_pts_seconds`: Raw un-repaired container PTS when repaired.
- **OCRObservation Timestamp Provenance**:
  - `OCRObservation.frame_timestamp`: Stores the authoritative `FrameTimestamp` object directly as a first-class field.
  - `OCRObservation.timestamp_seconds`: Synchronized numeric projection of `frame_timestamp.pts_seconds`.
- **TextObservation Timestamp Provenance**:
  - `TextObservation.first_seen_timestamp` and `TextObservation.last_seen_timestamp`: Preserve the exact authoritative `FrameTimestamp` objects of the first and last supporting observations.

### 2.2 Raw vs. Normalized Text Separation
The Evidence layer must always preserve exactly what the OCR engine returned while supporting clean semantic clustering:
- `text`: The raw, unadorned string returned by the OCR model.
- `normalized_text`: A conservatively normalized variant used for temporal matching.
- **Normalization Rules**:
  - Unicode NFKC normalization.
  - Whitespace collapsing (internal whitespace collapsed to single space, leading/trailing whitespace stripped).
  - Explicit preservation of Indic Devanagari Zero-Width Joiner (ZWJ, `\u200D`) and Zero-Width Non-Joiner (ZWNJ, `\u200C`) when adjacent to Devanagari characters, ensuring valid conjunct formation (e.g. Hindi "क्या", "पक्का").
  - Stripping of invisible control characters such as Byte Order Marks (`\uFEFF`) and zero-width spaces (`\u200B`).
  - No aggressive dictionary rewriting or fuzzy spell-checking is applied in the normalization pass.

---

## 3. Provider Abstraction Contract

The OCR subsystem enforces a strict provider-driven interface via the `OCRProvider` protocol:

```python
class OCRProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def supported_languages(self) -> tuple[str, ...] | list[str]: ...

    def warmup(self) -> None: ...

    def detect_text(
        self,
        frame: DecodedFrame | Frame,
    ) -> list[OCRObservation]: ...

    def detect_text_batch(
        self,
        frames: list[DecodedFrame | Frame],
    ) -> list[list[OCRObservation]]: ...
```

### Implementations:
1. `MockOCRProvider`: A hermetic, deterministic mock provider for unit tests, CI pipelines, and benchmarking without GPU or network dependencies.
2. `PaddleOCRProvider`: Base adapter wrapping the PaddleOCR engine with dynamic runtime import isolation.
3. `PaddleGeneralOCRProvider`: Specialization configured for Latin/English and multilingual Latin-script recognition.
4. `PaddleIndicOCRProvider`: Dedicated route configured explicitly for Hindi and Devanagari script recognition.

---

## 4. Multilingual Routing & The PP-OCRv6 Boundary

### 4.1 The Model Family Boundary
PaddleOCR upstream documentation designates:
- **PP-OCRv6**: The current general 50-language unified model centered on Latin, English, Chinese, and Japanese alphabets. It is **not** optimized or designated as the primary Hindi/Indic OCR model.
- **devanagari_PP-OCRv5_mobile_rec**: The dedicated Devanagari recognition model supporting Hindi, Marathi, Nepali, Sanskrit, and related Devanagari script languages.

To prevent silent recognition degradation:
- General Latin, English, and Western European languages route to `PaddleGeneralOCRProvider` (configured for `PP-OCRv6`).
- Hindi (`hi`), Marathi (`mr`), Sanskrit (`sa`), and any request specifying `script="Devanagari"` route explicitly to `PaddleIndicOCRProvider` using `devanagari_PP-OCRv5_mobile_rec`.

### 4.2 OCRRouter
The `OCRRouter` selects providers based on configurable mappings:
1. **Language Map**: Maps ISO 639-1 language codes (e.g. `'en' -> 'general'`, `'hi' -> 'indic'`).
2. **Script Map**: Maps script names (e.g. `'Latin' -> 'general'`, `'Devanagari' -> 'indic'`).
3. **Provider Capability Match**: Inspects `provider.supported_languages` if explicit routes are not mapped.
4. **Default Route Fallback**: Falls back to the configured default provider (typically `'general'`).

---

## 5. Temporal OCR Fusion

A primary flaw in naive video OCR is emitting independent semantic facts for every single video frame (e.g. 30 identical records per second for a stationary sign).

`TemporalOCRFusion` aggregates repeated per-frame text observations into durable, continuous `TextObservation` objects.

### 5.1 Clustering Algorithm
For each incoming frame:
1. **Stale Cluster Retirement**: Any active cluster whose unobserved gap exceeds `max_frame_gap` (default 5 frames) relative to the incoming frame index is finalized and closed.
2. **Compatibility Matching**: An observation matches an active cluster if:
   - **Text Identity**: `obs.normalized_text == cluster.normalized_text` (or case-insensitive if configured).
   - **Spatial Compatibility**:
     - Either Intersection over Union (IoU) of the bounding box with the cluster's most recent box $\ge \text{min\_iou\_overlap}$ (default 0.20), OR
     - Centroid Euclidean distance $\le \text{max\_centroid\_distance\_px}$ (default 60 px, handling panning and motion).
   - **Temporal Continuity**: Elapsed frame gap $(f_\text{current} - f_\text{cluster.last}) \le \text{max\_frame\_gap}$.
3. **Cluster Incorporation**: The observation's frame index, timestamp, bounding box, confidence, and observation ID are appended to the cluster. The representative text string is updated if the new observation has higher confidence.
4. **New Cluster Creation**: Any observation that does not match an existing active cluster spawns a new `_ActiveTextCluster`.
5. **Finalization**: When video ingestion concludes, `fusion.finalize()` flushes all remaining active clusters into completed `TextObservation` domain models.

### 5.2 Preservation of Raw Evidence
Temporal fusion **does not destroy** individual per-frame OCR observations.
- Raw `OCRObservation` instances remain intact in memory.
- Every `TextObservation` retains an explicit list of `supporting_observation_ids` pointing directly to the underlying raw OCR observation records.

---

## 6. OCR Frame Selection & Scheduling

Full-frame OCR on every frame of 4K or 60fps video is computationally prohibitive. `OCRScheduler` provides deterministic sampling abstractions:

| Strategy | Behavior | Use Case |
| :--- | :--- | :--- |
| `ALL_FRAMES` | Processes every decoded frame | Benchmark evaluation, short clips |
| `FIXED_INTERVAL` | Samples 1 frame every $N$ frames (e.g. every 5 frames) | High-speed dashcam / surveillance |
| `TEMPORAL_INTERVAL` | Samples 1 frame every $T$ seconds (e.g. every 0.5s) | Variable frame rate (VFR) streams |
| `SCENE_BOUNDARY` | Samples keyframes at detected scene boundaries | Cinematic or edited video sequences |
| `DETECTION_GUIDED` | Evaluates crops around perception bounding boxes | License plate and text ROI reading |

### Detection-Guided Crop Hook
The scheduler provides `OCRScheduler.create_cropped_frame(original, bbox, margin_px)` which extracts a spatial ROI from a `DecodedFrame` while strictly copying and preserving the authoritative `FrameTimestamp` provenance from the source frame.

---

## 7. Evidence Graph Integration

Every OCR result is integrated into the VIDEX Evidence domain graph:

```text
TextObservation (fused temporal text)
    ├── supports ──► OCRObservation 1 (frame 10, ts=0.40s)
    ├── supports ──► OCRObservation 2 (frame 11, ts=0.44s)
    └── supports ──► OCRObservation 3 (frame 12, ts=0.48s)
                         │
                         └── belongs to ──► DecodedFrame (frame_index=12, FrameTimestamp)
                                               │
                                               └── belongs to ──► Video
```

The resulting `Evidence` record has:
- `evidence_type = EvidenceType.OCR`
- `source_module = provider_name`
- `video_id = parent_video_id`
- `timestamp_seconds = first_seen_timestamp_seconds`
- `confidence = mean_confidence`
- `supporting_observation_ids = [obs_id_1, obs_id_2, ...]`
- `raw_payload = {text, normalized_text, first_seen_frame, last_seen_frame, confidence_summary, frame_timestamp}`

---

## 8. Quality Signals & Scoring Foundation

Each `OCRObservation` records objective quality signals:
- `confidence`: Primary model confidence score $\in [0, 1]$.
- `recognition_confidence`: Specific confidence from text recognition head.
- `detection_confidence`: Specific confidence from text bounding box detector.
- `orientation`: Text rotation angle in degrees.
- `crop_quality`: Spatial crop clarity metric.
- `preprocessing_applied`: Pipeline transforms (e.g., dewarp, contrast adjustment).
- `confidence_summary`: In `TextObservation`, captures `mean`, `min`, `max`, and `count` across supporting frames.

*Note: No subjective or uncalibrated "composite accuracy" scores are generated without empirical benchmark evaluation.*

---

## 9. Future Extensions (Roadmap)
1. **Indic Script Expansion**: Add dedicated routes for Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Gujarati (`gu`), and Punjabi (`pa`).
2. **Perception-Guided Pipeline Hook**: Directly route YOLO-detected vehicle license plates and street signs into high-resolution crop OCR passes.
3. **Spatial IoU Tracking with Velocity Compensation**: Incorporate bounding box velocity prediction during rapid camera pan.
