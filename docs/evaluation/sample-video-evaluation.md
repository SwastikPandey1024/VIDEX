# VIDEX Sample Video Corpus Evaluation Contract

## 1. Corpus Purpose & Evaluation Posture

The `Sample_Videos` dataset is designed as a **local verification, integration, and demonstration corpus**. 

> [!IMPORTANT]
> This corpus is NOT a formal academic benchmark with labeled ground-truth datasets (such as MOT17, COCO, or ActivityNet). It is a pragmatic, evidence-based integration evaluation suite used to test multimodal pipeline hand-offs, timestamp preservation, candidate extraction, and semantic routing across heterogeneous video domains.

### Explicit Evaluation Categories:
- **A. Capabilities Demonstrated by Video:** Observable directly through pipeline execution (e.g., PTS monotonicity, frame decoding, multi-object tracking, spatial containment, candidate formation).
- **B. Capabilities Requiring Human Ground Truth:** Metrics requiring external annotations to compute Precision/Recall/mAP (e.g., exact true positive detector count, MOTA/IDF1 tracking scores).
- **C. Capabilities Not Meaningfully Evaluated by Video:** Subsystems whose domain cues are absent in that clip (e.g., ASR on silent videos, OCR in scenes with zero signage or license plates).

---

## 2. Per-Video Evaluation Specification

### 2.1 `bikes.mp4` (Urban Bicycle Flow)
- **Technical Profile:** 1280x720, 50.0 FPS (High frame rate), 9.14s, 458 frames, H.264 CFR, AAC Audio.
- **A. Capabilities Demonstrated:**
  - Ingestion: High frame rate (50 FPS) decoding and strictly monotonic PTS indexing.
  - Perception: `bicycle` and `person` object detection; BoT-SORT high-frequency trajectory tracking.
  - Kinematics: Fast linear motion vectors and velocity estimation in pixel space.
  - Events: `object_appeared`, `object_started_moving`, `object_stopped_moving`.
  - Routing: Candidate clustering over cyclist cluster arrivals.
- **B. Requires Ground Truth:** Cyclist count precision, ID switch rate across occlusions.
- **C. Not Meaningfully Evaluated:** Speech transcription (audio stream contains traffic/wind noise, not human dialogue); text OCR (no prominent text plates).

### 2.2 `cars.mp4` (Vehicle Traffic & Turning)
- **Technical Profile:** 1280x720, 30.0 FPS, 12.23s, 368 frames, H.264 CFR, AAC Audio.
- **A. Capabilities Demonstrated:**
  - Perception: `car`, `truck`, `bus` detection and track persistence.
  - Spatial: Zone intersection (e.g., roadway lane or intersection boundary).
  - Kinematics: Multi-angle vehicle turning (`object_changed_direction` angular deflection).
  - Events: Spatial entry/exit, motion state transitions.
  - Semantic Routing: Saliency calculation on compound vehicle turning and zone dwell.
- **B. Requires Ground Truth:** Exact vehicle counts, vehicle classification fine-tuning.
- **C. Not Meaningfully Evaluated:** Dense speech recognition (engine noise dominant).

### 2.3 `motorbikes.mp4` (Two-Wheeler Traffic Stream)
- **Technical Profile:** 1366x720, 25.0 FPS, 27.40s, 686 frames, H.264 CFR, Silent.
- **A. Capabilities Demonstrated:**
  - Ingestion: Non-standard 1366x720 resolution handling, long clip trajectory buffering.
  - Perception: Small-object detection (`motorcycle`, `person`), trajectory tracking across dense lane filtering.
  - Events: Longitudinal motion tracking, velocity differentiation.
  - Routing: Temporal candidate generation across multiple overlapping tracks.
- **B. Requires Ground Truth:** Rider vs vehicle association accuracy.
- **C. Not Meaningfully Evaluated:** Audio/ASR (no audio stream present in container).

### 2.4 `people.mp4` (Pedestrian Density & Loitering)
- **Technical Profile:** 1280x720, 30.0 FPS, 21.03s, 632 frames, H.264 CFR, AAC Audio.
- **A. Capabilities Demonstrated:**
  - Perception: Dense multi-person detection and tracking, identity maintenance under partial occlusion.
  - Kinematics: Stationary dwell vs active walking trajectories.
  - Spatial: Loitering event detection within designated observation zones.
  - Events: `object_entered_zone`, `loitering`, `crowd_formation`.
  - Semantic Routing: High-saliency candidate formation from multi-person interactions.
- **B. Requires Ground Truth:** Re-identification accuracy across occlusions, crowd count.
- **C. Not Meaningfully Evaluated:** High-confidence OCR (incidental background text only).

### 2.5 `ppe-1.mp4` (Workplace Compliance — Single Worker)
- **Technical Profile:** 1280x720, 29.97 FPS (NTSC), 11.41s, 343 frames, H.264 CFR, Silent.
- **A. Capabilities Demonstrated:**
  - Ingestion: Fractional NTSC framerate (30000/1001) timestamp precision.
  - Perception: Person detection, steady tracking.
  - Semantic Routing: Candidate nomination for safety gear / PPE inspection.
  - VLM Reasoning: Visual inspection of vest/helmet in cropped keyframes.
- **B. Requires Ground Truth:** True compliance classification (hardhat present/absent).
- **C. Not Meaningfully Evaluated:** Audio/ASR (silent).

### 2.6 `ppe-2.mp4` (Workplace Compliance — Multi-Worker Interaction)
- **Technical Profile:** 1280x720, 25.0 FPS, 8.20s, 206 frames, H.264 CFR, Silent.
- **A. Capabilities Demonstrated:**
  - Perception: Multi-worker track formation, proximity tracking.
  - Events: Spatial co-presence, interaction candidates.
  - Semantic Routing: Explainable candidate priority tiering (HIGH / CRITICAL based on safety proximity).
- **B. Requires Ground Truth:** Multi-worker compliance ground truth.
- **C. Not Meaningfully Evaluated:** Audio/ASR (silent).

### 2.7 `ppe-3.mp4` (Workplace Compliance with Audio)
- **Technical Profile:** 1280x720, 30.0 FPS, 11.60s, 349 frames, H.264 CFR, AAC Audio.
- **A. Capabilities Demonstrated:**
  - Ingestion & Demuxing: Synchronized audio and video streams.
  - Audio: Environmental audio capture and ASR processing.
  - Cross-modal: Temporal alignment between worker visual presence and acoustic cues.
  - Semantic Routing: Cross-modal saliency bonus (`cross_modal_boost=0.25`).
- **B. Requires Ground Truth:** Speech transcript accuracy in industrial noise.
- **C. Not Meaningfully Evaluated:** License plate OCR.

### 2.8 `sample.mp4` (Lightweight Test Stream)
- **Technical Profile:** 320x240, 25.0 FPS, 2.96s, 75 frames, MPEG-4 CFR, Silent.
- **A. Capabilities Demonstrated:**
  - Ultra-fast end-to-end smoke testing (< 50ms execution).
  - MPEG-4 container handling and legacy fourcc decoding.
  - Minimal candidate gating and pipeline validation.
- **B. Requires Ground Truth:** None.
- **C. Not Meaningfully Evaluated:** High-resolution perception, ASR, OCR.

---

## 3. Evaluation Acceptance Criteria

A full evaluation run across the corpus is considered **successful** if and only if:
1. **Zero Unhandled Exceptions:** All 8 videos decode and process without unhandled crashes.
2. **Strict PTS Monotonicity:** Every generated frame, observation, and event has an exact, non-decreasing presentation timestamp ($t_0 \le t_1 \le \dots \le t_N$).
3. **Evidence Grounding:** Every emitted Event references valid `evidence_ids` created during perception/OCR/audio.
4. **Candidate Explainability:** Every candidate event produced has a non-empty, explainable reason string and a saliency score in $[0.0, 1.0]$.
5. **No Hallucinated Persistence:** Unsupported or rejected candidate interpretations never enter the canonical event stream.

---

## 4. Model Cache & Offline Reproducibility Contract

- **Deterministic Offline Gating**: CI environments and clean developer workstations must never trigger hidden, multi-gigabyte network downloads during video evaluation runs.
- **Whisper ASR Model Resolution**:
  - Preferred Local Cache: `models/faster-whisper-tiny` (`model.bin`, `config.json`).
  - Hugging Face Model Identifier: `Systran/faster-whisper-tiny`.
  - Offline Behavior: If the model cache is absent, the evaluation script logs an explicit `[DEPENDENCY UNAVAILABLE]` warning and falls back to deterministic `MockASRProvider`.
  - Provisioning:
    ```bash
    uv run huggingface-cli download Systran/faster-whisper-tiny --local-dir models/faster-whisper-tiny
    ```
- **VLM Reasoning Runtime Boundary**:
  - Canonical Architecture Target: `Qwen3-VL` (Layer 5).
  - Default Model Identifier: `Qwen/Qwen3-VL-8B-Instruct` (configurable via `VIDEX_SEMANTIC_MODEL`).
  - Safe Default Mode: `VIDEX_SEMANTIC_PROVIDER=mock` (or `execution_mode=disabled`).
  - Real VLM execution requires explicitly setting `VIDEX_SEMANTIC_PROVIDER=qwen` and configuring either a remote `api` endpoint or installing `torch` / `transformers` for `local_gpu` / `local_cpu`. In this evaluation suite, `REAL VLM RUNTIME = NOT EXECUTED` is strictly reported.
