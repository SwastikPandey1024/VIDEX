# Research Note: Efficient Video OCR & Speech Transcription (ASR)

**Domain:** Multimodal Video Text Extraction & Acoustic Speech Intelligence  
**Target Components:** `src/videx/providers/ocr.py` & `src/videx/providers/asr.py`  
**Status:** In Progress / Active Reference  
**Relevant ADRs:** [ADR-003 — OCR Architecture & Engine Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-003-ocr.md) · [ADR-004 — Audio & Hindi ASR Engine Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-004-asr.md)

---

## 1. Problem and Engineering Question

Video multimodal understanding requires extracting on-screen textual semantics (signboards, safety placards, slides, chyrons) and audio speech narratives. Unlike scanned flatbed documents or high-fidelity studio recordings, video presents unique computational and perceptual bottlenecks:
1. **The OCR Latency Barrier:** Running deep neural text detection (e.g. DBNet) and text recognition (e.g. SVTR) across full high-resolution video frames ($1080\text{p}/720\text{p}$) is orders of magnitude more expensive than object detection. In raw CPU benchmarks, full-frame OCR takes tens of seconds per frame, rendering per-frame execution physically impossible.
2. **Audio Hallucination & Acoustic Noise:** Industrial video environments feature ambient machinery noise, long intervals of silence, and multilingual / code-switched speech (Hindi-English / Hinglish). Naive Transformer ASR models hallucinate repetitive loops during silence and consume CPU cycles transcribing acoustic static.

The core engineering questions are:
- How can VIDEX schedule OCR execution (scene-aware gating, motion thresholds, text-region cropping) to extract vital text evidence without collapsing pipeline throughput?
- How should Faster-Whisper be tuned on CPU (INT8 quantization, Voice Activity Detection, beam search vs. greedy decoding) to produce synchronized word-level timestamps with minimal real-time factor ($\text{RTF} < 0.25$)?

---

## 2. Current VIDEX Implementation and Evidence

### OCR Subsystem (`src/videx/providers/ocr.py`)
- **Engine:** PaddleOCR (PP-OCRv6 English/Devanagari model suite).
- **Architecture:** Two-stage detector-recognizer (DBNet text detector + SVTR text recognizer).
- **Execution Policy:** Keyframe-only extraction (1 Hz interval, yielding 12 keyframes on an 11.44s clip).

### Empirical Profiling Findings (`Sample_Videos/ppe-1.mp4`, 12 Keyframes)
From `outputs/profiling/cpu_baseline_report.md`:
```
PaddleOCR (PP-OCRv6 CPU Steady-State):
├── Cold-Start Engine Warmup:  15,981.5 ms  (~16.0 s)
└── Keyframe Steady-State (12 keyframes):
    ├── Mean Latency:         53,596.9 ms  (~53.6 s per keyframe)
    ├── Median Latency:       49,162.4 ms
    ├── P90 Latency:          89,213.8 ms
    ├── P99 Latency:          93,434.9 ms
    ├── Min Latency:          26,361.6 ms
    ├── Max Latency:          93,770.4 ms
    └── Cumulative Wall Time: 643.17 s     (~10.7 minutes!)
```
*Crucial Finding:* Running OCR on just 12 keyframes consumed **$643\text{ seconds}$** of CPU time—over $6.5\times$ longer than detecting objects across all 343 frames ($96\text{ seconds}$). Unconditional full-frame OCR is mathematically incompatible with scalable video processing.

### ASR Subsystem (`src/videx/providers/asr.py`)
- **Engine:** `faster-whisper-tiny` backed by CTranslate2.
- **Model Loading Time:** $2,972.7\text{ ms}$ cold start.
- **Timestamp Strategy:** Word-level timestamps aligned to the video master clock.
- **Status on Baseline Video:** Bypassed on `Sample_Videos/ppe-1.mp4` due to absence of an active audio stream in the test clip.

---

## 3. Official Documentation and Primary Papers

### Primary Documentation
1. **PaddleOCR (PaddlePaddle Team):**
   - [Official Documentation & Architecture](https://www.paddleocr.ai/latest/en/index.html)
   - [PaddleOCR GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)
   - *Key Algorithms:* DBNet (Real-time Scene Text Detection with Differentiable Binarization, Liao et al., AAAI 2020) and SVTR (Single Visual Model for Scene Text Recognition, Du et al., IJCAI 2022).
2. **OpenAI Whisper & Faster-Whisper:**
   - *Radford, A., et al. (2022).* "Robust Speech Recognition via Large-Scale Weak Supervision." [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
   - *Faster-Whisper (SYSTRAN):* [GitHub Repository](https://github.com/SYSTRAN/faster-whisper). Re-engineers Whisper in CTranslate2, delivering up to $4\times$ speedup on CPU with 8-bit quantization.
3. **Voice Activity Detection (VAD):**
   - *Silero VAD:* Deep learning-based enterprise-grade Voice Activity Detector operating at $<1\text{ ms}$ per audio chunk. Integrated natively into Faster-Whisper to filter out non-speech silence.

---

## 4. Candidate Approaches and Hardware Requirements

### OCR Optimization Matrix

| Approach | Scheduling Rule | Resolution / Input | Estimated Latency | Accuracy / Utility |
| :--- | :--- | :--- | :---: | :--- |
| **Un-gated Frame OCR (Naive)** | Every frame (30 FPS) | Full frame ($1280\times 720$) | ~53,000 ms / frame (Prohibitive) | Redundant duplicate text |
| **Periodic Keyframe OCR (Current)** | Fixed 1 Hz (1 frame/sec) | Full frame ($1280\times 720$) | ~53,000 ms / keyframe | High latency overhead |
| **Motion/Scene-Gated OCR** | Run only when scene cut or high motion is detected | Full frame or downscaled | ~5,000 ms amortized | Eliminates static frame re-computation |
| **Crop-Gated Targeted OCR (Proposed)** | Triggered by detector on text-bearing objects (signs, screens) | Isolated bounding-box crop | **$200\text{–}600\text{ ms / crop}$** | Concentrates compute only where text exists |

### ASR Tuning Matrix
- **Model Size:** `tiny` or `base` for CPU development; `medium` or `large-v3` for GPU production.
- **Quantization:** `compute_type="int8"` on CPU via CTranslate2.
- **VAD Pre-Filtering:** Enforce `vad_filter=True` with `min_silence_duration_ms=500` to bypass empty audio frames completely.

---

## 5. Expected Trade-offs and Failure Modes

1. **OCR Temporal Duplication & Flickering:**
   - *Failure Mode:* A static sign visible for 10 seconds generates 10 slightly differing raw OCR string variations due to minor pixel noise and compression artifacts.
   - *Mitigation:* Implement Levenshtein-distance string clustering and IoU spatial overlap matching across sequential frames to emit a single consolidated text event.
2. **Low-Resolution Video Degradation:**
   - *Failure Mode:* Text smaller than 16 pixels height suffers severe Character Error Rate (CER) inflation in standard recognizers.
   - *Mitigation:* Local bicubic upsampling on cropped text candidates prior to passing into the recognition network.
3. **Whisper Repetition Loops in Noise:**
   - *Failure Mode:* Ambient hum or background engine rumble triggers Whisper decoder to repeat phrases indefinitely (e.g., "Thank you. Thank you. Thank you.").
   - *Mitigation:* Configure `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`, and `condition_on_previous_text=False`.

---

## 6. Proposed Experiment and Acceptance Criteria

### Experiment 1: Crop-Gated OCR Acceleration
- **Baseline:** Full-frame PaddleOCR on 12 keyframes ($53.6\text{s/call}$).
- **Intervention:** Run YOLO26n to detect object classes with high text probability (e.g. `signboard`, `whiteboard`, `screen`, `nameplate`), crop those regions, and pass only the crops to PaddleOCR recognition (`rec_only=True`).
- **Target:** Amortized OCR latency $\le 800\text{ ms}$ per processed keyframe with zero loss in target text recovery.

### Experiment 2: Faster-Whisper INT8 CPU Latency & Parity
- **Dataset:** 3 benchmark audio clips (English workplace safety briefing, Hindi narration, and mixed Hinglish dialogue).
- **Metrics:** Real-Time Factor ($\text{RTF} = \text{latency} / \text{duration}$), Word Error Rate (WER), memory consumption.
- **Target:** $\text{RTF} \le 0.25$ on 4 CPU threads (transcribes a 60-second clip in under 15 seconds) with $\text{WER} \le 18\%$.

---

## 7. Actual Experiment Results (Kept Separate from Hypotheses)

### PaddleOCR Empirical Baseline (Recorded 2026-10-21)
- **Model:** PP-OCRv6 CPU.
- **Engine Warm-up:** $15.98\text{ s}$ single-call cold initialization.
- **12-Keyframe Execution:** $643.17\text{ s}$ total execution time.
- **Distribution:** Mean $53.6\text{ s}$, P90 $89.2\text{ s}$, Max $93.8\text{ s}$.
- **Conclusion:** Confirmed that raw, un-gated full-frame OCR represents the single largest latency bottleneck in VIDEX by a factor of over $100\times$. Architectural gating is an absolute prerequisite.

### Faster-Whisper Tiny Baseline
- **Model Checkpoint:** `models/faster-whisper-tiny`.
- **Initialization Time:** $2,972.7\text{ ms}$ cold load on CPU.

---

## 8. Decision, Rationale, and Link to Relevant ADRs

- **OCR Decision:** Adopt the **Routed, Keyframe-Gated OCR** architecture. Restrict full-frame OCR to coarse text region localization, and prioritize high-speed crop recognition for detected objects.
  - *Relevant ADR:* [ADR-003 — OCR Architecture & Engine Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-003-ocr.md).
- **ASR Decision:** Adopt **faster-whisper (CTranslate2 INT8)** with native Silero VAD gating as the baseline speech transcription engine.
  - *Relevant ADR:* [ADR-004 — Audio & Hindi ASR Engine Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-004-asr.md).
