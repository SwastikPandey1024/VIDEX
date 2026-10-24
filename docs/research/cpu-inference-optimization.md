# Research Note: CPU Inference Optimization

**Domain:** Inference Acceleration & Runtime Engineering  
**Target Component:** `src/videx/providers/detector.py` & Graph Runtime Wrappers  
**Status:** In Progress / Active Reference  
**Relevant ADR:** [ADR-001 — Object Detector Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-001-detector.md)

---

## 1. Problem and Engineering Question

In client surveillance and edge intelligence deployments, dedicated GPU infrastructure is frequently unavailable, cost-prohibitive, or architecturally decoupled. VIDEX must deliver actionable video perception directly on multi-core server/desktop x86 CPUs.

Current PyTorch eager-mode execution of YOLO26n runs at **$280.2\text{ ms/frame}$** ($\approx 3.57\text{ FPS}$), consuming over $95\%$ of the perception pipeline's wall-clock time. This prevents real-time processing of standard 30 FPS surveillance streams.

The core engineering questions are:
1. **Graph Compilation & Runtime Optimization:** How much latency reduction can graph-level fusions (operator merging, constant folding, dead code elimination) and runtime engines (ONNX Runtime, OpenVINO) achieve over raw PyTorch CPU eager mode?
2. **Quantization & Operator Precision:** Can Post-Training Quantization (PTQ) to INT8 achieve a $\ge 2\times$ throughput speedup on modern vector extensions (AVX-512 VNNI / AMX) without regressing object detection mAP below the acceptable threshold ($\Delta \text{mAP} < 1.5\%$)?
3. **Threading & Concurrency Discipline:** How should intra-op and inter-op threads be configured across multi-core CPUs to prevent thread thrashing and memory bus saturation during concurrent frame decoding?

---

## 2. Current VIDEX Implementation and Evidence

### Current State
- **Runtime:** `torch.device("cpu")` with PyTorch `2.14.0+cpu` running `models/yolo26n.pt`.
- **Hardware Profile:** Intel Core (12 physical cores, 16 logical threads), Windows 11.
- **Profiling Script:** `scripts/profile_cpu_baseline.py` producing `outputs/profiling/cpu_baseline_ppe1.json`.
- **Benchmark Suite:** `tests/unit/test_benchmark_integrity.py` asserting timing-scope consistency and arithmetic invariants.

### Measured Latency Breakdown (`Sample_Videos/ppe-1.mp4`, 343 frames)
```
Isolated YOLO26n PyTorch CPU:
├── Cold Start Load: 9,893.3 ms
├── 1st Frame Warmup: 445.4 ms
└── Steady-State Inference (342 frames):
    ├── Mean:   280.22 ms  (3.57 FPS)
    ├── Median: 225.72 ms
    ├── P90:    407.79 ms
    ├── P99:   1117.48 ms
    ├── Min:     82.46 ms
    └── Max:   1961.45 ms
```

### Key Observation: Eager Mode Inefficiency
PyTorch eager mode introduces substantial Python runtime overhead per operator call, creates temporary intermediate tensor allocations that exhaust CPU L2/L3 cache, and lacks kernel fusion for Conv+BatchNorm+SiLU blocks.

---

## 3. Official Documentation and Primary Papers

### Primary Documentation
1. **ONNX Runtime (Microsoft):**
   - [Performance Tuning Guide](https://onnxruntime.ai/docs/performance/tune-performance/): Details session options, graph optimization levels (`ORT_ENABLE_ALL`), memory pattern optimization, and execution providers.
   - [Model Quantization Guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): Explains Static vs. Dynamic PTQ, symmetric vs. asymmetric quantization, calibration datasets, and operator exclusions.
2. **OpenVINO (Intel):**
   - [OpenVINO Architecture & Performance Tuning](https://docs.openvino.ai/): Explains Intel Advanced Vector Extensions (AVX2, AVX-512, AMX), precision modes (FP32, FP16, INT8), latency vs. throughput execution hints (`PERFORMANCE_HINT`), and CPU stream pooling.
3. **PyTorch Profiler:**
   - [PyTorch Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html) & [Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): Operator-level profiling, trace export to Chrome tracing (`chrome://tracing`), memory timeline recording, and CPU worker thread instrumentation.
4. **Ultralytics Model Export:**
   - [YOLO Model Export Documentation](https://docs.ultralytics.com/modes/export/): Mechanics of exporting PyTorch weights to ONNX (`format="onnx"`), OpenVINO (`format="openvino"`), dynamic batching, and simplification (`simplify=True`).

---

## 4. Candidate Approaches and Hardware Requirements

| Candidate Runtime | Precision | Optimization Mechanism | Target Architecture | Projected FPS | Engineering Complexity |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **PyTorch Eager (Current)** | FP32 | Unoptimized operator-by-operator execution | General CPU | ~3.6 FPS | Baseline (Low) |
| **ONNX Runtime (ORT)** | FP32 | Graph fusion, constant folding, memory planning | Any x86/ARM CPU | 8–14 FPS | Medium |
| **OpenVINO CPU Engine** | FP32 | Intel-tailored kernel scheduling & fused MHA/Conv | Intel Core / Xeon | 10–18 FPS | Medium |
| **ONNX Runtime (Quantized)** | INT8 | Post-training static quantization (VNNI instructions) | AVX-512 / VNNI CPUs | 18–28 FPS | High (Calibration required) |
| **OpenVINO (Quantized)** | INT8 | NNCF (Neural Network Compression Framework) | Intel AVX-512 / AMX | 20–32 FPS | High |

### Threading Architecture & Hardware Bounds
- **Thread Count:** Setting thread count equal to total logical cores (16) often causes CPU cache thrashing due to OS scheduling contention. Benchmarks must sweep `threads ∈ [1, 2, 4, 8, 12]` to identify the inflection point where memory bandwidth saturates.
- **Affinity:** Binding inference worker threads to physical performance cores (P-cores) while delegating video decode (`cv2`) to efficiency cores (E-cores) prevents latency jitter.

---

## 5. Expected Trade-offs and Failure Modes

1. **Parity Drift vs. PyTorch Baseline:**
   - *Failure Mode:* Graph simplifiers (e.g. `onnxslim`) or precision casting slightly modify box coordinates, resulting in dropped detections at strict confidence thresholds ($0.25$).
   - *Mitigation:* Automated parity test suite verifying $\text{IoU} \ge 0.98$ and $|\Delta \text{score}| \le 0.02$ across a 100-frame validation sequence.
2. **INT8 Quantization Outlier Corruption:**
   - *Failure Mode:* Uniform 8-bit quantization scales saturate when layer activations have long heavy-tailed distributions, degrading small-object recall.
   - *Mitigation:* Symmetric per-channel weight quantization and KL-divergence calibration using 200 representative surveillance video frames.
3. **Memory Footprint & Leakage in Long Videos:**
   - *Failure Mode:* Unmanaged session memory buffers accumulating across thousands of frames in high-throughput video processing.
   - *Mitigation:* Enforce explicit session cleanup, verify zero RSS drift over 1,000 continuous frames, and track memory with `tracemalloc`.

---

## 6. Proposed Experiment and Acceptance Criteria

### Experiment Protocol
1. **Export Pipeline Validation:**
   - Export `models/yolo26n.pt` to `models/yolo26n.onnx` with fixed input resolution $640\times 640$, dynamic batching disabled for deterministic edge memory profiling.
   - Validate ONNX model validity using `onnx.checker.check_model`.
2. **Isolated Latency & Parity Benchmark (`scripts/benchmark_onnx_fp32.py`):**
   - Run 343 frames of `Sample_Videos/ppe-1.mp4` under identical preprocessing (letterbox scaling, RGB conversion, $1/255$ normalization).
   - Test thread configurations: $\text{intra\_op\_num\_threads} \in \{1, 4, 8, 12\}$.
   - Record per-frame distribution (Mean, Median, P90, P99, Min, Max).
3. **Parity Acceptance Verification:**
   - Compute coordinate-wise IoU and confidence delta between PyTorch and ONNX outputs for every frame.

### Acceptance Criteria
- **Latency Target:** Mean inference latency $\le 100\text{ ms/frame}$ on x86 CPU ($\ge 10\text{ FPS}$).
- **Parity Gate:** Mean IoU $\ge 0.98$ across all matching bounding boxes; total detection count matching PyTorch within $\pm 1.0\%$.
- **Stability:** Zero crashes or unhandled C++ runtime exceptions across multi-run executions.

---

## 7. Actual Experiment Results (Kept Separate from Hypotheses)

### PyTorch FP32 Baseline (Recorded 2026-10-21)
- **Mean Frame Latency:** $280.23\text{ ms}$
- **Throughput:** $3.57\text{ FPS}$
- **P90 Latency:** $407.79\text{ ms}$
- **P99 Latency:** $1117.48\text{ ms}$
- **Peak Traced Memory:** $11.66\text{ MB}$ (PyTorch internal), RSS Delta $+161.0\text{ MB}$ (process total).

### ONNX Runtime FP32 Implementation Status
- `scripts/benchmark_onnx_fp32.py` authored with full timing instrumentation, output tensor decoding, letterbox transformation, and parity metrics.
- Benchmark validation tests integrated into `tests/unit/test_benchmark_integrity.py` (9 conditional parity assertion tests awaiting exported weights artifact).

---

## 8. Decision, Rationale, and Link to Relevant ADR

- **Decision:** Advance **ONNX Runtime FP32** as the primary near-term CPU execution engine for YOLO26n, followed by an evaluation of **OpenVINO FP32/INT8** for Intel-specific deployments.
- **Rationale:** ONNX Runtime provides a universal, hardware-portable runtime that decouples VIDEX from PyTorch's heavy Python runtime on target deployment environments, while preserving numerical parity with the trained model.
- **Relevant ADR:** [ADR-001 — Object Detector Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-001-detector.md).
