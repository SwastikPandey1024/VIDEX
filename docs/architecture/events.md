# VIDEX Architecture: Temporal Event Intelligence Engine (Phase 5.0)

## 1. Overview & Architectural Boundary

Phase 5.0 introduces the **Temporal Event Intelligence Engine** to VIDEX. The engine transforms perception, OCR, and audio evidence into structured, timestamped, evidence-backed events using pure deterministic reasoning.

```text
  Perception Pipeline          OCR Pipeline           Audio Pipeline
 (Tracks & Trajectories)   (Fused Text Clusters)   (Transcript Segments)
           │                         │                       │
           └─────────────────────────┼───────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │            EventEngine            │
                   │  - LifecycleEventDetector         │
                   │  - MovementEventDetector          │
                   │  - SpatialEventDetector           │
                   │  - OCREventDetector               │
                   │  - AudioEventDetector             │
                   │  - TemporalRelationEngine         │
                   │  - CrossModalRuleEngine           │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │           EventTimeline           │
                   │  - Chronological Sorting          │
                   │  - Interval Queries               │
                   │  - 6-Part Evidence Explainability │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                     Downstream Evidence Graph & Audit
```

### The 5-Layer Epistemological Hierarchy

To prevent the rule engine from degenerating into a hard-coded heuristic reasoning system, VIDEX enforces a strict 5-layer epistemological boundary:

```text
RAW EVIDENCE
     ↓  (Tracks, Detections, Trajectories, OCR Text, ASR Transcripts)
DETERMINISTIC EVENTS
     ↓  (Single-modality primitive state transitions: EnteredZone, SpeechStarted)
DETERMINISTIC RELATIONSHIPS
     ↓  (Interval algebra, Spatial containment, Multi-modal co-occurrence)
CANDIDATE SEMANTIC EVENTS
     ↓  (Saliency filtering, Query triage, Targeted visual crop packaging)
VLM / REASONING (Qwen3-VL)
        (Evidence-backed semantic interpretation & open-world reasoning)
```

### Core Architecture Axioms
1. **Traceable to Source Evidence**: Every event references concrete, immutable perception, OCR, or audio evidence identifiers (`evidence_ids` and `EventEvidence`). No ungrounded textual claims are permitted.
2. **First-Class Temporal Provenance**: Events strictly inherit presentation timestamps (PTS) from `FrameTimestamp` and normalized audio seconds. No FPS-derived timestamps are introduced.
3. **Deterministic Before Semantic**: Fast, deterministic CPU algorithms characterize state transitions and topological relationships before calling any expensive language or vision-language models.
4. **Relational Co-occurrence vs Semantic Interpretation**: The `CrossModalRuleEngine` computes deterministic spatiotemporal relationships (Layer 3). It must **never** hardcode subjective semantic interpretations (e.g. human intent or sentiment), which strictly belong in Layer 5 (Phase 6 VLM / Reasoning).
5. **Decoupled Architecture**: Detectors consume domain-level data structures (`Track`, `Trajectory`, `TextObservation`, `TranscriptSegment`) rather than specific hardware, framework, or vendor models (YOLO26, BoT-SORT, PaddleOCR, Faster-Whisper).

---

## 2. Event Domain Model & Taxonomy

The event model separates instantaneous boundary events from continuous state interval events.

```text
Instantaneous Event (start == end)       Interval Event (start < end)
       PTS: t0                                   PTS: t0 ───► t1
          ▼                                            ▼       ▼
   [ObjectAppeared]                              [ObjectPresent]
```

### 2.1 Event Schema Attributes
Every `Event` record adheres to the following immutable contract:
- `event_id`: Unique UUID identifier.
- `event_type`: Categorical `EventType` enum member.
- `start_timestamp_seconds`: Start Presentation Timestamp (inclusive).
- `end_timestamp_seconds`: End Presentation Timestamp (inclusive).
- `duration_seconds`: `max(0.0, end_timestamp_seconds - start_timestamp_seconds)`.
- `confidence`: Confidence score in `[0.0, 1.0]`, derived deterministically from supporting observations.
- `status`: Current status (`CONFIRMED`, `TENTATIVE`, `REJECTED`, `SUPERSEDED`).
- `severity`: Operational severity level (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- `participants`: List of `EventParticipant` tracking identities, text tokens, or audio channels.
- `event_evidence`: Typed lightweight `EventEvidence` linkage descriptors.
- `evidence_ids`: Quick-lookup list of UUIDs pointing to supporting evidence items.
- `source_module`: Subsystem originating the event (e.g. `events.lifecycle`, `events.spatial`).
- `attributes`: Flexible typed metadata (e.g. velocity, zone ID, text payload, bearing).
- `created_at`: UTC ISO-8601 generation timestamp.

### 2.2 Implemented Event Taxonomy

| Family | Event Type | Mode | Description |
|---|---|---|---|
| **Lifecycle** | `OBJECT_APPEARED` | Instantaneous | First confirmed observation of a track. |
| | `OBJECT_PRESENT` | Interval | Continuous span of confirmed track visibility. |
| | `OBJECT_DISAPPEARED` | Instantaneous | Final observation of a terminated or expired track. |
| **Movement** | `OBJECT_STARTED_MOVING` | Instantaneous | Transition from stationary to pixel displacement exceeding threshold. |
| | `OBJECT_STOPPED_MOVING` | Instantaneous | Transition from moving to stationary. |
| | `OBJECT_CHANGED_DIRECTION` | Instantaneous | Trajectory bearing deviation exceeding angular threshold. |
| **Spatial** | `OBJECT_ENTERED_ZONE` | Instantaneous | Track centroid crosses boundary from outside to inside a defined polygon/rectangle. |
| | `OBJECT_EXITED_ZONE` | Instantaneous | Track centroid crosses boundary from inside to outside a defined zone. |
| **OCR** | `TEXT_APPEARED` | Instantaneous | First observation of a stable text cluster. |
| | `TEXT_CHANGED` | Instantaneous / Interval | Shift in recognized string content for an identical track/spatial location. |
| | `TEXT_DISAPPEARED` | Instantaneous | Final observation of a text cluster. |
| **Audio** | `SPEECH_STARTED` | Instantaneous | Leading edge of an ASR transcript segment. |
| | `SPEECH_ENDED` | Instantaneous | Trailing edge of an ASR transcript segment. |
| | `SPEECH_DETECTED` | Interval | Full duration of recognized spoken speech. |
| **Compound** | `CROSS_MODAL_CORRELATION` | Interval | Rule-evaluated multi-modal correlation (e.g. speech inside zone). |

---

## 3. Evidence-Backed Model & 6-Part Explainability Chain

To prevent "hallucinated intelligence," VIDEX enforces an explicit relation:
```text
Event ────► EventEvidence ────► Source Observation (Track / Point / Text / Audio)
```

Lightweight `EventEvidence` records preserve:
- `evidence_id`: UUID pointing to the underlying evidence.
- `evidence_type`: `EvidenceType` enum (e.g. `TRACK`, `VISUAL_DETECTION`, `OCR`, `TRANSCRIPT`).
- `role`: Role played in detection (e.g. `trigger`, `leading_edge`, `trailing_edge`, `context`).
- `timestamp_seconds`: PTS of the specific observation.
- `frame_index`: Optional video frame sequence number.
- `track_id`: Optional associated track identifier.
- `observation_id`: Optional underlying subsystem observation ID.
- `metadata`: Provenance dictionary.

### The 6 Questions of Explainability
Every event can be audited without rerunning detectors via `timeline.explain_event(event_id)`:
1. **WHAT happened?** Event type, description, severity, and status.
2. **WHEN did it happen?** Start PTS, end PTS, duration, and instantaneous status.
3. **WHICH entities participated?** Roles and IDs of participants (e.g. track 17, zone 'gate_a').
4. **WHAT evidence supports it?** Linked evidence UUIDs, evidence types, and operational roles.
5. **WHICH frame/timestamp anchors it?** Exact video frame numbers and container Presentation Timestamps.
6. **WHICH subsystem generated it?** Source module, detector version, and algorithmic attributes.

---

## 4. Deterministic Detectors

### 4.1 Lifecycle Detector (`LifecycleEventDetector`)
- **Track Continuity**: Uses tracker state (`TrackState.ACTIVE` vs `TERMINATED`) and trajectory points.
- **Noise Filtering**: Requires a track to reach at least `min_confirmation_frames` (default: 3 frames) before emitting `OBJECT_APPEARED`. Tracks with only 1 or 2 frames are suppressed as detection noise.
- **Disappearance Logic**: When track ends or state transitions to `TERMINATED`, `OBJECT_DISAPPEARED` is emitted at the final trajectory point PTS.

### 4.2 Movement Detector (`MovementEventDetector`)
- **Pixel-Space Kinematics**: Computes displacement $\Delta p = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ and instantaneous velocity $v_{px} = \frac{\Delta p}{\Delta t}$.
- **Calibration Safety**: All velocities are explicitly denominated in `px/s`. The engine makes **zero physical speed claims** (such as km/h or mph) without calibrated homography matrices.
- **Direction Changes**: Direction angle $\theta = \operatorname{atan2}(\Delta y, \Delta x) \times \frac{180}{\pi}$. Bearing deltas exceeding `direction_change_degrees_threshold` (default: 45.0°) generate `OBJECT_CHANGED_DIRECTION`.

### 4.3 Spatial Zone Detector (`SpatialEventDetector`)
- **Geometric Primitives**: Supports `polygon` and `rectangle` zones.
- **Ray-Casting Algorithm**: Employs deterministic Jordan curve ray casting to determine if a point $(x, y)$ resides inside a polygon.
- **Hysteresis & Debouncing**:
  - `boundary_tolerance_px` (default: 5.0 px): Suppresses jitter on exact boundary pixels.
  - `debounce_interval_seconds` (default: 0.5 s): Rapid in-out-in crossings within the debounce window are coalesced, eliminating oscillatory false alarms.

### 4.4 OCR Detector (`OCREventDetector`)
- **Fused Clusters as Truth**: Consumes `TextObservation` clusters emitted by `TemporalOCRFusion`. Never reruns OCR.
- **Text Changes**: Tracks string variations across consecutive frames within the same cluster. When normalized text changes, `TEXT_CHANGED` is emitted referencing both observations.

### 4.5 Audio Detector (`AudioEventDetector`)
- **ASR Segments**: Consumes `TranscriptSegment` objects produced by Faster-Whisper.
- **Speech Events**: Emits `SPEECH_STARTED` at segment start PTS, `SPEECH_ENDED` at segment end PTS, and `SPEECH_DETECTED` spanning the speech interval.
- **Sound Event Provider Hook**: Preserves forward compatibility with non-speech audio events (`SoundEventProvider`) without fabricating synthetic sounds.

---

## 5. Temporal Semantics & Timeline Engine

### 5.1 Temporal Relation Engine
The `TemporalRelationEngine` implements deterministic interval algebra:
- `BEFORE`: Event A ends before Event B starts (`end_a < start_b`).
- `AFTER`: Event A starts after Event B ends (`start_a > end_b`).
- `OVERLAPS`: Event A starts before Event B, but ends after Event B starts (`start_a < start_b < end_a < end_b`).
- `CONTAINS`: Event A completely encompasses Event B (`start_a <= start_b` and `end_a >= end_b`).
- `DURING`: Event A is completely encompassed by Event B (`start_b <= start_a` and `end_b >= end_a`).
- `NEAR_IN_TIME`: The temporal distance between intervals is within `event_temporal_near_interval_seconds` (default: 2.0 s).
- `EQUALS`: Start and end timestamps are identical within a small floating-point $\epsilon$ ($10^{-3}$ s).

### 5.2 Event Timeline
- **Canonical Sorting Key**:
  1. `start_timestamp_seconds` (ascending)
  2. `end_timestamp_seconds` (ascending)
  3. `event_type.value` (lexicographical)
  4. `event_id` (lexicographical for deterministic stability)
- **Interval Querying**: `timeline.events_between(start_sec, end_sec)` locates all events active during a time window using interval overlap logic.

---

## 6. Configuration & Thresholds

All thresholds are centralized in `videx.config.Settings` and `EventEngineConfig`:

| Configuration Setting | Default | Rationale |
|---|---|---|
| `event_lifecycle_confirmation_threshold` | 3 frames | Prevents transient single-frame detector hallucinations from emitting lifecycle events. |
| `event_disappearance_threshold_seconds` | 1.0 s | Tolerates brief tracker occlusion gaps before declaring true disappearance. |
| `event_movement_velocity_threshold_px_s` | 15.0 px/s | Separates sensor noise / bounding box jitter from intentional object motion. |
| `event_direction_change_degrees_threshold` | 45.0° | Detects significant turns while ignoring subtle steering adjustments. |
| `event_spatial_boundary_tolerance_px` | 5.0 px | Creates an unambiguous boundary zone to prevent oscillation. |
| `event_spatial_debounce_interval_seconds` | 0.5 s | Temporal refractory period to debounce boundary jitter. |
| `event_temporal_near_interval_seconds` | 2.0 s | Temporal window for correlating proximate multimodal events. |

---

## 7. Performance & Benchmarks

Phase 5.0 introduces **zero ML inference overhead**. All algorithms are pure CPU operations operating over existing perception and audio memory buffers.

Benchmark results on representative test video streams:
- **Event Engine Latency**: ~5.5 ms for 100-frame multimodal window.
- **Event Processing Throughput**: > 2,800 events / second.
- **Memory Footprint**: Lightweight reference structures; zero duplication of raw frame pixels or audio PCM buffers.

---

## 8. Known Limitations

1. **Pixel-Space Kinematics**: Velocities and displacements are strictly measured in camera sensor pixels. World-space speeds require camera intrinsic/extrinsic calibration and ground-plane homography.
2. **Acoustic Event Classification**: Non-speech audio (e.g. glass breaks, gunshots, engine revs) is supported via provider interfaces, but classifier models are deferred to Phase 8+.
3. **Complex Semantic Reasoning**: Phase 5.0 detects *what happened geometrically and temporally*, not *why it matters in a broader narrative*.

---

## 9. Future VLM Boundary (Phase 6+)

To maintain strict architectural discipline, **Qwen3-VL is NOT introduced in Phase 5.0**.

The Event Intelligence Engine establishes the clean deterministic foundation for future VLM reasoning. As formalized in [ADR-005](../decisions/ADR-005-semantic-boundary-and-vlm-routing.md), VIDEX transitions from temporal events to semantic reasoning through a dedicated **Semantic Router**:

```text
Video Stream
 │
 ├── Frames
 │    ├── Detection (YOLO26)
 │    ├── Tracking (BoT-SORT)
 │    ├── Trajectory
 │    └── OCR (PaddleOCR)
 │
 └── Audio
      └── ASR (Faster-Whisper)
          │
          ▼
     Canonical Evidence (Layer 1)
          │
          ▼
   Temporal Event Engine (Phase 5 - Layer 2 & 3)
          │
          ├── Lifecycle
          ├── Movement
          ├── Spatial
          ├── OCR
          ├── Audio
          └── Relations (TemporalRelationEngine & CrossModalRuleEngine)
          │
          ▼
     Event Timeline
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │ Phase 6: Semantic Router (Layer 4)           │
   │  - Saliency Filtering & Query Triage         │
   │  - Token Budget & Throttling Guardrails      │
   │  - Spatial Bounding Crop Extractor (+Margin) │
   │  - Synchronized Transcript/OCR Framing       │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ Qwen3-VL (Layer 5)                           │
   │  - Evidence-grounded visual-linguistic model │
   │  - Targeted keyframe crops (no full video)   │
   │  - Structured JSON semantic output           │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
             Evidence-backed Semantic Events
          (Complete 6-Part Provenance Chain)
```

By guaranteeing that all candidates arrive with frame indices, bounding boxes, and timestamp provenance, future VLM agents will receive targeted visual crops rather than processing redundant raw video frames, reducing inference tokens and cloud costs by over 95%.

