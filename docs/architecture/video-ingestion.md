# VIDEX — Video Ingestion & Temporal Foundation

**Status:** Implemented & Hardened (Phase 1.1 / 1.2 Packet-Accurate Timing)  
**Package:** `src/videx/ingestion`  
**Tests:** `tests/unit/test_ingestion.py`  
**Entrypoint:** `python -m videx.ingestion <video_path> [--json] [--no-scenes]`  

---

## 1. Purpose & Architectural Context

Video ingestion is the entry gateway for VIDEX. Downstream perception engines (object detection, multi-object tracking, OCR routing, and speech transcription) require reliable, deterministic access to video frames and temporal coordinate systems.

Raw video files are inherently noisy:
- Containers may specify nominal frame rates (e.g., 29.97 fps) that diverge from true presentation timestamps (PTS).
- Variable frame rate (VFR) encoding and container stream discrepancies can lead to desynchronization between audio transcripts and visual bounding boxes.
- Corrupted frames or broken container indices can crash inference workers mid-stream.
- Eager image encoding (e.g. compressing every frame to JPEG) creates heavy, redundant CPU bottlenecks for computer vision models that consume raw arrays.

Phase 1/1.1 provides a hardened, decoupled video ingestion subsystem that validates input video files, extracts rich stream and container metadata, detects VFR discrepancies, maps timestamps deterministically, provides raw NumPy frame access, optimizes sequential frame reading, detects shot/scene boundaries, and emits a validated, immutable `VideoManifest`.

---

## 2. Ingestion Flow

```
Input Video Path / Stream
           │
           ▼
┌───────────────────────────────────────┐
│ 1. Validation & Guarding              │  • Existence & regular file check
│    (validation.py)                    │  • Non-zero file size check
│                                       │  • Container extension filter
│                                       │  • Header readability probe
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2. Metadata & VFR Detection           │  • ffprobe: streams, codecs, audio, r_fps, avg_fps
│    (metadata.py)                      │  • OpenCV VideoCapture (fallback): dims, fps
│                                       │  • VFR classification heuristic (|r - avg| / avg > 0.5%)
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 3. Scene Boundary Segmentation        │  • PySceneDetectDetector: Content changes
│    (scenes.py)                        │  • SingleSceneDetector: Fallback
│                                       │  • Non-overlapping Scene models
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 4. Manifest Construction              │  • VideoManifest domain model (with is_vfr)
│    (manifest.py)                      │  • Format summary & JSON export
│                                       │  • manifest.to_video() conversion
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 5. Deterministic Frame Access         │  • OpenCVVideoReader implementation
│    (reader.py, timing.py)             │  • TimestampIndex (CFR / Explicit PTS)
│                                       │  • DecodedFrame (raw NumPy arrays)
│                                       │  • Sequential seek optimization
└───────────────────────────────────────┘
```

---

## 3. Core Abstractions

### 3.1 `VideoReader` Protocol (`src/videx/ingestion/base.py`)

All frame decoding and random-access seeking is decoupled behind the `VideoReader` protocol:

```python
class VideoReader(Protocol):
    @property
    def is_opened(self) -> bool: ...
    @property
    def total_frames(self) -> int: ...
    @property
    def fps(self) -> float: ...
    @property
    def duration_seconds(self) -> float: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

    def open(self) -> None: ...
    def close(self) -> None: ...
    def read_decoded_frame(self, frame_index: int) -> DecodedFrame: ...
    def read_decoded_frame_at_timestamp(self, timestamp_seconds: float) -> DecodedFrame: ...
    def read_frame(self, frame_index: int) -> tuple[Frame, bytes]: ...
    def read_frame_at_timestamp(self, timestamp_seconds: float) -> tuple[Frame, bytes]: ...
    def get_timestamp_for_frame(self, frame_index: int) -> float: ...
    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int: ...
```

#### Why This Abstraction Exists
1. **Decoder Decoupling:** Downstream perception providers (detectors, trackers, OCR) never interact directly with underlying decoding libraries (`cv2.VideoCapture`, `PyAV`, or `FFmpeg`).
2. **Type-Safe Stream Inspection:** Standard stream properties (`total_frames`, `fps`, `duration_seconds`, `width`, `height`) are declared on the protocol itself.
3. **Deterministic Outputs:** Every frame access returns a typed `Frame` or `DecodedFrame` domain entity, ensuring that calling `read_decoded_frame(i)` multiple times returns identical pixel payloads.
4. **Resource Safety:** Implements Python's context manager protocol (`with service.create_reader(manifest) as reader:`) to guarantee operating system file descriptors and hardware decoders are closed immediately upon completion.

### 3.2 Raw Frame Access (`DecodedFrame`)

Computer-vision models (YOLO, BoT-SORT, OCR) require in-memory image matrices (NumPy arrays or PyTorch tensors) for batch inference. Forcing eager JPEG re-encoding during frame reads causes an unnecessary compression cycle (~5–15 ms per frame) followed immediately by a re-decoding step by the model.

`DecodedFrame` (`src/videx/ingestion/base.py`) solves this:
- **`frame_array`:** Direct in-memory BGR NumPy `ndarray` of shape `(height, width, channels)`.
- **`to_jpeg(quality=95)`:** Explicit, on-demand JPEG serialization used only when frames are persisted to disk or transmitted over HTTP.
- **`to_domain_frame()`:** Converts to the standard `Frame` domain schema for evidence indexing.

### 3.3 Sequential Read Seek Optimization

In OpenCV, invoking `cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))` triggers an internal demuxer seek and keyframe decode up to frame `i`. When reading thousands of frames sequentially, redundant seeking turns linear decoding ($O(N)$) into seek thrashing ($O(N^2)$).

`OpenCVVideoReader` tracks the current decoder cursor (`self._current_frame_pos`). When `read_decoded_frame(i)` matches `self._current_frame_pos`, no seek is performed—the frame is decoded directly from the active hardware stream. Seeking is triggered only upon non-sequential jumps.

---

## 4. Temporal Model & Timestamp Semantics

Converting between continuous time ($t \in \mathbb{R}_{\ge 0}$) and discrete frame indices ($i \in \mathbb{N}_0$) is central to multi-modal evidence correlation (e.g., aligning ASR speech timestamps with visual object appearances).

```
                      ┌────────────────────────────────────────┐
                      │            TimestampIndex              │
                      │  (Protocol: frame_index <-> seconds)   │
                      └──────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       ┌─────────────────────────┐               ┌─────────────────────────┐
       │   CFRTimestampIndex     │               │   PyAVTimestampIndex    │
       │   (O(1) Math Mapping)   │               │   (Packet-Accurate PTS) │
       └─────────────────────────┘               └─────────────────────────┘
         • FAST timing mode policy                 • EXACT timing mode policy (default)
         • t = i / fps                             • Scans container presentation timestamps
         • Explicitly configured only              • Strict monotonic repair & provenance
                                                   • Authoritative variance classification
```

### 4.1 Timing Policy: EXACT vs FAST

To prevent subtle VFR footage from being misclassified as CFR due to inaccurate multiplexer headers, VIDEX enforces an explicit timing policy via `TimingMode` (`src/videx/ingestion/base.py`):

1. **`TimingMode.EXACT` (Default for Evidence Processing):**
   - Scans actual presentation timestamps from the container using PyAV.
   - Authoritative VFR status is established from true PTS variance, eliminating false-negative metadata heuristic risks.
   - Preserves source timestamp provenance (`CONTAINER` vs `DERIVED`).
2. **`TimingMode.FAST`:**
   - Allowed only when explicitly configured and preliminary metadata indicates CFR (`is_vfr_hint = False`).
   - Uses the $O(1)$ formula ($t = i / \text{fps}$ and $i = \min(\text{round}(t \times \text{fps}), N - 1)$) with zero container scanning overhead.

### 4.2 Preliminary Metadata Hint vs Authoritative Timing

- **Preliminary Metadata Timing Hint (`is_vfr_hint`):**
  Extracted from ffprobe stream headers by comparing `r_frame_rate` against `avg_frame_rate`. Many video encoders (e.g., in surveillance systems or smartphones) write identical frame rates in container headers even when frame presentation timestamps vary dynamically. This hint is retained on `VideoManifest.is_vfr_hint` for auditability but is **never** used as the sole basis for timing calculations in `EXACT` mode.
- **Authoritative Timing (`is_vfr`):**
  Determined by `detect_vfr_from_timestamps` during the PyAV container scan. If the relative variance of inter-frame presentation intervals ($\Delta t_i = t_i - t_{i-1}$) exceeds 2% of the median interval, the video is authoritatively classified as VFR (`manifest.is_vfr = True`).

### 4.3 Timestamp Provenance: Source vs Derived

VIDEX maintains an explicit provenance model for every frame timestamp via `FrameTimestamp` (`src/videx/ingestion/base.py`):
- **`timestamp_source = CONTAINER`:** The presentation timestamp (PTS) was extracted directly from decoded frame headers in the container bitstream in presentation order and was strictly monotonic.
- **`timestamp_source = DERIVED`:** The timestamp was synthesized, interpolated, or defensively adjusted.
  - `is_repaired = True`: Indicates defensive repair.
  - `repair_reason`: Reason for adjustment (`"missing_pts"`, `"duplicate_pts"`, `"non_monotonic_pts"`, or `"boundary_repair"`).
  - `original_pts_seconds`: Preserves the original raw container PTS for forensic traceability.

Valid container presentation timestamps are preserved exactly without synthetic drift. `FrameTimestamp` implements numeric comparison and arithmetic dunder methods, allowing seamless consumer access as float seconds while preserving full forensic metadata on `reader.get_frame_timestamp(i)` and `decoded_frame.frame_timestamp`.

### 4.4 Codec Frame Reordering (B-Frames: PTS vs DTS)

In modern codecs (H.264, HEVC) with bidirectional predictive frames (B-frames), packets are multiplexed in **Decode Timestamp (DTS)** order (decoding order) rather than **Presentation Timestamp (PTS)** order (display order):
- `container.demux()` yields packets in packet arrival / DTS order (e.g., packet PTS values `[0, 80, 40, 120, 160]`).
- `container.decode()` reorders frames internally into monotonic presentation display order (PTS) (`[0, 40, 80, 120, 160]`).

The VIDEX evidence pipeline requires presentation timestamps (display timing) rather than packet decoding timestamps. `PyAVTimestampIndex` demuxes and decodes frames at the C level in presentation order without allocating uncompressed NumPy pixel arrays in Python. This guarantees exact parity with OpenCV's sequential decode order and eliminates frame-index and timestamp drift across GOP boundaries.

### 4.5 OpenCV Frame Seeking Limitations
`OpenCVVideoReader` relies on `cv2.VideoCapture`. Several intrinsic limitations of OpenCV timing must be accounted for:
1. **Post-Read Decoder Drift:** Querying `cv2.CAP_PROP_POS_MSEC` immediately after reading frame $N$ frequently reports the timestamp of frame $N+1$ or intermediate decoder drift.
2. **VFR Seeking Approximations:** In VFR containers, calling `cap.set(cv2.CAP_PROP_POS_FRAMES, n)` seeks by estimated packet offsets rather than presentation timestamps, which can introduce off-by-one frame navigation errors.
3. **Frame Count Discrepancies:** OpenCV's `CAP_PROP_FRAME_COUNT` estimation can disagree with actual demuxed frame counts in VFR containers.

**Architectural Solution:**
- `OpenCVVideoReader` binds directly to an authoritative `TimestampIndex` (`CFRTimestampIndex` for CFR in FAST mode, `PyAVTimestampIndex` for EXACT mode).
- Frame count and duration on the reader are synchronized with the authoritative index, ensuring that frame extraction and timestamp extraction cannot silently disagree about frame numbering.
- All presentation timestamps are derived from the index rather than post-read OpenCV queries.

---

## 5. Architectural Scoping Decisions

### 5.1 Why S3 / Object Storage Is Intentionally Deferred
- **Seekability Requirement:** Video decoding (via OpenCV, FFmpeg, or hardware decoders) requires seekable byte-range access to parse container atoms (MOOV atom in MP4, EBML in MKV) and navigate I-frames. Streaming decoders directly over raw HTTP/S3 GET requests introduces high latency and buffering overhead.
- **Clean Ingestion Boundary:** Cloud-based inference workers standardly stage video assets from S3 to local ephemeral scratch storage (`/tmp`) before decoding. Introducing an upstream `VideoSourceResolver` in Phase 6 will handle remote staging without altering any internal ingestion contracts.

### 5.2 Why ML Models Are Not Introduced in Phase 1
- **Subsystem Isolation:** Introducing YOLO, BoT-SORT, or Whisper in Phase 1 would violate architectural modularity by coupling foundational ingestion to specific model weights, GPU dependencies, and CUDA runtime environments.
- **Deterministic Baseline:** Phase 1 guarantees deterministic frame access, timing integrity, and scene segmentation as a prerequisite so that perception models (Phase 2) can be benchmarked against verified ground truth.

---

## 6. Failure Modes & Exception Hierarchy

VIDEX rejects invalid or unreadable video files upfront before compute resources are scheduled:

```
IngestionError
  ├── VideoValidationError
  │     ├── VideoNotFoundError (missing path, directory passed)
  │     ├── UnsupportedContainerError (unsupported extension)
  │     └── CorruptVideoError (0-byte file, unreadable container)
  └── FrameAccessError
        ├── FrameOutOfBoundsError (frame_index < 0 or >= total_frames)
        └── TimestampOutOfBoundsError (timestamp < 0.0 or > duration)
```

| Failure Mode | Detected By | Raised Exception | Action Taken |
|---|---|---|---|
| Non-existent path | `validation.py` | `VideoNotFoundError` | Immediate rejection with path context |
| Directory passed as video | `validation.py` | `VideoNotFoundError` | Rejection (not a regular file) |
| Empty (0-byte) file | `validation.py` | `CorruptVideoError` | Rejection |
| Unsupported extension | `validation.py` | `UnsupportedContainerError` | Rejection with allowed extensions list |
| Corrupt container headers | `validation.py` / `metadata.py` | `CorruptVideoError` | Rejection |
| Frame index out of bounds | `reader.py` / `timing.py` | `FrameOutOfBoundsError` | Bounds validation, prevents decoder crash |
| Timestamp out of bounds | `reader.py` / `timing.py` | `TimestampOutOfBoundsError` | Bounds validation against duration + epsilon |

---

## 7. The `VideoManifest` Domain Contract

Upon successful ingestion, the subsystem emits a validated `VideoManifest` (`src/videx/domain/schemas.py`).

### Schema Structure
```json
{
  "video_id": "71c4f9a6-4728-4c87-a2d5-97c7076cde2b",
  "source_path": "/path/to/sample.mp4",
  "filename": "sample.mp4",
  "filesize_bytes": 55509,
  "mime_type": "video/mp4",
  "duration_seconds": 3.0,
  "fps": 25.0,
  "total_frames": 75,
  "width": 320,
  "height": 240,
  "video_codec": "mpeg4",
  "is_vfr": false,
  "has_audio": false,
  "audio_codec": null,
  "scenes": [
    {
      "scene_id": "6162d6ec-8aca-44cb-9292-164d9d3dc115",
      "video_id": "71c4f9a6-4728-4c87-a2d5-97c7076cde2b",
      "scene_index": 0,
      "start_frame_number": 0,
      "end_frame_number": 74,
      "start_timestamp_seconds": 0.0,
      "end_timestamp_seconds": 3.0
    }
  ],
  "ingested_at": "2026-09-28T09:16:16.443877Z",
  "metadata": {
    "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
    "r_frame_rate": 25.0,
    "avg_frame_rate": 25.0,
    "ffprobe_streams": 1,
    "bitrate": "148024",
    "pix_fmt": "yuv420p"
  }
}
```

### Persistence Conversion
Calling `manifest.to_video()` produces a standard `Video` domain entity ready for relational persistence in the Evidence Store with canonical `video_codec`.

---

## 8. CLI Usage

The ingestion subsystem can be run directly from the command line:

```bash
# Print human-readable summary (including CFR/VFR mode)
python -m videx.ingestion path/to/video.mp4

# Emit machine-readable JSON
python -m videx.ingestion path/to/video.mp4 --json

# Ingest without content-aware scene detection (single scene)
python -m videx.ingestion path/to/video.mp4 --no-scenes
```

---

## 9. Technical Debt & Upstream Dependency Tracking

- **Starlette TestClient / HTTPX Deprecation (`StarletteDeprecationWarning`):**
  Starlette currently issues a deprecation warning when creating a `TestClient` instance wrapping `httpx`. While filtered via `pyproject.toml` (`filterwarnings = ["ignore:.*Using `httpx` with `starlette.testclient` is deprecated.*:UserWarning"]`) to keep test runs clean, this is actively tracked as technical debt. The test client harness will be modernized when transitioning to async test clients or updated FastAPI/Starlette major releases in future milestones.
