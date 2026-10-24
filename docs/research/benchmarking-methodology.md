# Research Note: Benchmarking Methodology & Evaluation Science

**Domain:** Measurement Discipline, Evaluation Science & Statistical Rigor  
**Target Components:** `scripts/profile_cpu_baseline.py` & `tests/unit/test_benchmark_integrity.py`  
**Status:** In Progress / Active Reference  
**Relevant Documents:** [Roadmap](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/VIDEX_Optimization_and_Accuracy_Roadmap.md) · [Annotation Guidelines](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/annotation-guidelines.md) · [Evaluation Audit](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/evaluation-vs-accuracy-audit.md)

---

## 1. Problem and Engineering Question

In complex computer vision and multimodal intelligence systems, performance claims and accuracy metrics are only as trustworthy as the measurement harness that produces them. Common anti-patterns in video pipeline engineering include:
1. **Conflating Parity with Accuracy:** Assuming that because an ONNX or quantized model reproduces PyTorch outputs within $\Delta < 1\%$, the system is inherently "accurate." Parity tests verify *runtime consistency*; only independent human ground-truth measures *real-world accuracy*.
2. **Timing Scope Ambiguity:** Comparing timing numbers across different execution boundaries (e.g. comparing an isolated single-stage loop against an end-to-end streaming pipeline), leading to phantom "regressions" or false claims of acceleration.
3. **Sampling Bias & Spatial Autocorrelation:** Evaluating detectors on consecutive video frames (e.g., 30 frames from a single second of footage) and claiming high sample diversity, when adjacent frames are virtually identical in lighting, pose, and background.
4. **Statistical Naivety:** Reporting only mean latency without recording cold starts, percentiles ($P_{50}, P_{90}, P_{99}$), outliers, memory RSS growth, or CPU cache warm-up state.

The core engineering question is:
- **How do we establish a mathematically sound, reproducible benchmarking standard for VIDEX that enforces rigorous statistical timing scopes, separates parity from accuracy, and prevents annotation contamination?**

---

## 2. Current VIDEX Implementation and Evidence

### Measurement Infrastructure
- **Profiling Script:** `scripts/profile_cpu_baseline.py` recording structured metrics in `outputs/profiling/cpu_baseline_ppe1.json`.
- **Integrity Test Harness:** `tests/unit/test_benchmark_integrity.py` asserting:
  - Metric schema completeness and structural consistency.
  - Distribution invariant rules: $\text{Min} \le \text{Median} \le P_{90} \le P_{99} \le \text{Max}$.
  - Percentage arithmetic summing to $100.0\% \pm 0.5\%$.
  - Separation of cold-start latencies from steady-state inference loops.
  - Audit trail SHA-256 hash preservation.

### The 96s vs. 42s Timing Scope Case Study
A critical demonstration of timing scope discipline arose during the baseline CPU profiling of `Sample_Videos/ppe-1.mp4`:
- **Isolated Stage Loop (Section 3 of Profiling Report):**
  - YOLO26n inference was executed in an isolated loop over 343 sequentially decoded frames.
  - Total wall time: **$96.12\text{ s}$** ($280.2\text{ ms/frame}$).
- **End-to-End Pipeline Multi-Run (Section 4 of Profiling Report):**
  - Run 1 (Cold Start): **$42.30\text{ s}$** ($8.11\text{ FPS}$).
  - Run 2 (Warm Steady-State): **$46.15\text{ s}$** ($7.43\text{ FPS}$).
- **The Physical Explanation:**
  The end-to-end pipeline run measured the active streaming pipeline under its native batching and frame-router sampling conditions, whereas the isolated benchmark forced synchronous, single-frame eager evaluation of every single frame without downstream pipelining. Benchmarks must always explicitly state their measurement scope.

---

## 3. Official Documentation and Primary Papers

### Primary Literature & Standards
1. **COCO Object Detection Evaluation:**
   - [COCO Dataset Detection Evaluation Resource](https://cocodataset.org/dataset/detection-eval.htm): Defines 101-point interpolated AP, strict IoU thresholds ($0.50:0.05:0.95$), scale division, and max detection limits ($100$ per image).
2. **TrackEval Benchmarking Standard:**
   - *Luiten, J., et al. (2020).* "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking." [GitHub TrackEval](https://github.com/JonathonLuiten/TrackEval). Establishes standardized MOT metric computation decoupled from tracker internal heuristics.
3. **PyTorch Profiler & Benchmarking Best Practices:**
   - [PyTorch Benchmark Documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html): Emphasizes warm-up iterations, timer synchronization, timer overhead subtraction, and multi-thread contention management.
4. **Scientific Reproducibility in Empirical Machine Learning:**
   - *Pineau, J., et al. (2021).* "Improving Reproducibility in Machine Learning Research." *Journal of Machine Learning Research*. Mandates publishing code, environment specifications, random seeds, hardware topology, and exact variance statistics.

---

## 4. Methodological Framework: Parity vs. Accuracy

VIDEX establishes a strict two-pillar evaluation framework:

```
                          VIDEX EVALUATION FRAMEWORK
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 ┌──────────────────────┐                            ┌──────────────────────┐
 │   PILLAR 1: PARITY   │                            │  PILLAR 2: ACCURACY  │
 ├──────────────────────┤                            ├──────────────────────┤
 │ Scope: Engine vs.    │                            │ Scope: Prediction    │
 │        Engine        │                            │        vs. Reality   │
 │ Target: PyTorch vs.  │                            │ Target: Prediction   │
 │         ONNX/OV/INT8 │                            │         vs. Human GT │
 │ Metric: IoU Parity,  │                            │ Metric: Precision,   │
 │         Score Delta  │                            │         Recall, mAP, │
 │ Goal: Zero compile   │                            │         HOTA, WER    │
 │       regressions    │                            │ Goal: High real-world│
 └──────────────────────┘                            │       fidelity       │
                                                     └──────────────────────┘
```

### Pillar 1: Runtime Parity Verification
- **Purpose:** Verifies that exporting or quantizing a model (e.g. converting `yolo26n.pt` to `yolo26n.onnx` or applying OpenVINO INT8) does not corrupt numerical behavior.
- **Protocol:** Run both engines on identical preprocessed input tensors.
- **Pass Criteria:**
  - Spatial IoU between matching boxes: $\text{IoU} \ge 0.98$.
  - Confidence score divergence: $|\Delta \text{conf}| \le 0.02$.
  - Detection count match: $\pm 1$ box across full clip.

### Pillar 2: Ground-Truth Accuracy Measurement
- **Purpose:** Measures the real-world perceptual competence of the system against physical ground truth.
- **Protocol:** Compare model predictions against human-verified annotations under explicit guidelines (e.g., Policy A: full anatomy including hard hat and limbs).
- **Pass Criteria:** Strictly enforced 1-to-1 matching (no duplicate credit), $\text{mAP}_{50} \ge 0.85$.

---

## 5. Statistical Rigor & Metric Protocols

To ensure reproducible measurements, every VIDEX benchmark must adhere to these statistical rules:

1. **Warm-Up Isolation:**
   - Always execute a minimum of 5 warm-up iterations before recording timings. Never include weight loading or initial JIT trace compiles in steady-state latency calculations.
2. **Percentiles Over Means:**
   - Because OS scheduling, CPU thermal throttling, and garbage collection introduce latency outliers, report:
     - **Median ($P_{50}$):** Robust indicator of typical latency.
     - **P90 & P99:** Measures worst-case jitter for real-time streaming SLAs.
     - **Min & Max:** Bounds of the physical system.
3. **Memory Instrumentation:**
   - Report both internal framework heap (`tracemalloc` peak memory) and operating system Resident Set Size (`psutil.Process().memory_info().rss`).
   - Measure RSS start, RSS end, and Delta across multi-run executions to catch unmanaged memory growth.
4. **Deterministic Frame Addressing:**
   - Video decoding must address frames by explicit sequential indices, not floating-point timestamps, to eliminate decoder seek jitter.

---

## 6. Proposed Experiment and Acceptance Criteria

### Continuous Benchmark Integrity CI Gate
1. **Automated Assertion Suite:**
   - Every profiling run automatically exports a standardized JSON schema.
   - `pytest tests/unit/test_benchmark_integrity.py` executes in CI to enforce:
     - All timing distributions obey non-decreasing order ($\text{Min} \le \text{Median} \le P_{90} \le P_{99} \le \text{Max}$).
     - Subsystem breakdown percentages sum to $100.0\% \pm 0.5\%$.
     - Audit trail JSON files match recorded SHA-256 digests.

### Acceptance Criteria
- Zero manual "eyeballing" of numbers: all statistical invariants tested programmatically.
- Standard deviation across 3 repeated runs on identical hardware must satisfy $\sigma / \mu < 0.05$ ($<5\%$ run-to-run variance).

---

## 7. Actual Experiment Results (Kept Separate from Hypotheses)

### Integrity Suite Validation (Executed 2026-10-22)
- Test Suite: `tests/unit/test_benchmark_integrity.py`.
- **Status:** **18 Passed, 0 Failed, 9 Skipped** (ONNX parity tests safely skipped pending export).
- Verified Invariants:
  - Validated that `cpu_baseline_ppe1.json` satisfies distribution monotonicity across all stages (Decoding, YOLO26n, BoT-SORT, PaddleOCR).
  - Validated that perception stage percentages sum to $100.0\%$.
  - Validated that the audit trail SHA-256 hash matches the original human export fixtures, ensuring complete provenance protection.

---

## 8. Decision, Rationale, and Link to Relevant Architecture

- **Decision:** Formalize this benchmarking methodology as the mandatory standard for all performance and accuracy claims in VIDEX.
- **Rationale:** Without strict timing scope boundaries and automated integrity assertions, optimization efforts risk chasing measurement artifacts rather than genuine system speedups.
- **Relevant Documents:**
  - [Roadmap](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/VIDEX_Optimization_and_Accuracy_Roadmap.md)
  - [Annotation Guidelines](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/annotation-guidelines.md)
  - [Evaluation Audit](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/evaluation-vs-accuracy-audit.md)
