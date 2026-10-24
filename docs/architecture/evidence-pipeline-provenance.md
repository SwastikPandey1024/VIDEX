# VIDEX Evidence Pipeline — Provenance, Data Flow & Failure Propagation

## 1. Overview & Architecture

The VIDEX Evidence Pipeline orchestrates multimodal video perception, temporal fusion, associative memory projection, and persistent graph storage. Every analytical output is strictly grounded in immutable source video provenance with presentation timestamp (PTS) traceability.

```mermaid
flowchart TD
    A["Video Source (.mp4)"] --> B["Ingestion Engine\n(OpenCVVideoReader)"]
    B --> C["DecodedFrame\nFrameTimestamp (PTS)"]
    
    C --> D1["Perception Pipeline\n(YOLO26 Detector + ByteTracker)"]
    C --> D2["OCR Pipeline\n(OCRScheduler + PaddleOCR + Fusion)"]
    A --> D3["Audio Pipeline\n(AudioExtractor + Whisper ASR)"]
    
    D1 --> E1["Detections & Tracks\n(BoundingBox + Trajectory)"]
    D2 --> E2["Text Observations & Evidence\n(OCRObservation + Temporal Fusion)"]
    D3 --> E3["Transcript Segments & Audio Evidence\n(TranscriptSegment + AudioEvidence)"]
    
    E1 --> F["Event Engine\n(Movement, Spatial Zones, Lifecycle)"]
    E2 --> F
    E3 --> F
    
    F --> G["Event Timeline & Evidence Bundles"]
    
    G --> H["Graph Projection Engine\n(GraphProjection)"]
    H --> I["Evidence Graph Store\n(InMemoryGraphStore / SQLiteGraphStore)"]
    
    I --> J["Associative Traversal & Persistence\n(Read-Back, Paths, ACID Transactions)"]
```

---

## 2. Provenance Guarantees & Tracing

1. **Source Traceability:**
   - Every `DecodedFrame`, `Detection`, `Track`, `OCRObservation`, and `Evidence` record carries `video_id`, `frame_index`, and `timestamp_seconds` (authoritative container PTS).
   - Timestamp repair mechanisms preserve `original_pts_seconds`, `is_repaired`, and `repair_reason` without corrupting temporal monotonicity.

2. **Perception Grounding:**
   - Bounding boxes conform to Policy A (pixel extents clamped to frame dimensions $[0, W] \times [0, H]$).
   - Track trajectories record historical centroid positions and frame associations.

3. **Multimodal Evidence Grounding:**
   - Temporal OCR fusion clusters text observations across consecutive frames, recording `first_seen_frame`, `last_seen_frame`, `supporting_frames`, and `supporting_observation_ids`.
   - Audio segments preserve start and end PTS timestamps, confidence metrics, and language detection metadata.
   - High-level `Event` records link directly to supporting evidence IDs via `event.evidence_ids`.

4. **Referential Graph Consistency:**
   - `GraphNode` records represent domain entities (`video`, `scene`, `frame`, `detection`, `track`, `ocr`, `transcript`, `event`, `evidence`).
   - `GraphEdge` records establish typed, directed spatiotemporal relationships (`CONTAINS`, `CO_OCCURS`, `APPEARS_IN`, `TEMPORALLY_PRECEDES`, `SPATIALLY_NEAR`).
   - Referential integrity checks verify that no dangling edge endpoints exist in the store.

---

## 3. Persistence & Transaction Safety

1. **Storage Backends:**
   - `InMemoryGraphStore`: Adjacency-indexed in-memory store for high-throughput pipeline execution, sub-millisecond local queries, and atomic JSON export/import.
   - `SQLiteGraphStore`: ACID-compliant relational store with foreign keys, secondary indices, and transactional guarantees.

2. **Idempotency & Replay:**
   - Node and edge identifiers use deterministic hashing / deterministic string keys (`make_node_id`, `make_edge_id`).
   - Re-running graph projection or retrying failed pipeline batches performs idempotent upserts (`INSERT OR REPLACE` / `ON CONFLICT DO UPDATE`), preventing duplicate records.

3. **Failure Propagation & Rollbacks:**
   - Unhandled exceptions during batch persistence trigger explicit `ROLLBACK` via `SQLiteGraphStore.transaction()`, preventing partial or corrupted graph states.
   - Invalid media paths or missing model files raise actionable exceptions (`FileNotFoundError`, `RuntimeError`) rather than returning misleading empty successes.

---

## 4. Test Environments, Mocks & Benchmark Scope Clarifications

1. **Mock vs. Live Model Testing:**
   - In `test_end_to_end_evidence_pipeline_on_real_media`, detector and tracker stages utilize `MockDetector` and `MockTracker` over real decoded OpenCV video frames. This hermetically isolates ingestion, temporal indexing (PTS), event engine routing, and SQLite ACID persistence without incurring weight download overhead or long CPU inference times during default test runs.
   - Live neural models (YOLO26, ByteTrack, PaddleOCR) are validated in dedicated acceptance and benchmarking suites (`test_ocr_acceptance.py`, `scripts/run_standardized_benchmark.py`).

2. **OCR Video Preprocessor Optimization Scope:**
   - Disabling paper document preprocessors (`use_doc_preprocessor=False` and `use_textline_orientation=False`) bypasses document orientation classification and paper unwarping on upright video camera frames.
   - The reported 225x figure (24.5s -> 108ms) is strictly an isolated **synthetic-frame microbenchmark** on a printed text card (`"VIDEX PIPELINE DEMO"`). It is not a generalized video-level performance guarantee, and zero recall degradation cannot be extrapolated across natural video scenes with motion blur, perspective distortion, or dynamic lighting.

3. **Backend Parity & Production Posture:**
   - While solitary-worker tracking parity on `ppe-1.mp4` demonstrated high bounding box agreement (median IoU 0.9955), ONNX Runtime under CPU execution exhibits higher variance and p95 tail latency (180.82 ms vs. PyTorch 88.70 ms in standardized single-pass tracking runs).
   - Multi-person scenes with occlusion and identity switching have not yet been evaluated for ONNX parity.
   - PyTorch FP32 remains the authoritative production default backend.
