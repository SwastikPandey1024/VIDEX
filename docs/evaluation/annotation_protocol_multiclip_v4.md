# Multi-Clip Evaluation Dataset (v4.0.0) — Annotation Protocol

## 1. Overview & Purpose
This document defines the formal annotation protocol, ground truth bounding box standards, and validation constraints for the **VIDEX Multi-Clip Evaluation Dataset (v4.0.0)**.
Dataset Version: `v4.0.0`
Protocol Standard: **Policy A (Visible Human Extents Only)**

The objective of v4.0.0 is to expand evaluation coverage from single-clip verification (`ppe-1.mp4`) to a comprehensive multi-clip evaluation benchmark consisting of 100 independently verified frames across 5 distinct video sources, covering:
- Single-person industrial inspection (`ppe-1.mp4`)
- Two-person industrial interaction (`ppe-2.mp4`)
- Three-person dense industrial occlusions & scale variations (`ppe-3.mp4`)
- Complex multi-pedestrian urban transit scenes with dynamic crossings (`people.mp4`)
- Negative control non-person vehicle transit scene (`sample.mp4`)

---

## 2. Policy A Annotation Standard

### 2.1 Visible Boundary Rule
- **Extents Only**: Annotations strictly enclose the **visible** portions of a human body.
- **No Hallucination**: Do not hallucinate, extrapolate, or infer occluded anatomy (e.g., legs behind crates, torso behind desks).
- **Coordinate Clamping**: Coordinates $[x, y, w, h]$ are absolute pixel coordinates with origin $(0, 0)$ at the top-left corner.
  - $0 \le x < \text{width}$
  - $0 \le y < \text{height}$
  - $1 \le w \le \text{width} - x$
  - $1 \le h \le \text{height} - y$
  - $x + w \le \text{width}$ and $y + h \le \text{height}$
- **Partial Extents**: If only an upper body, head, or limb is visible due to boundary cutoff or occlusion, annotate the tight visible box enclosing that person provided they are identifiable as a person.

### 2.2 Temporal Continuity & Track IDs
- Track identifiers (`person_1`, `person_2`, etc.) must remain temporally consistent across frames within the same clip.
- If a person leaves the camera FOV and re-enters with ambiguous identity, a new identifier is assigned.

### 2.3 Negative Control Frames
- To measure false discovery rates (false positives), the dataset includes explicit negative frames containing **0** human occupants (`person_count: 0`, `annotations: []`).
- In v4.0.0, `sample.mp4` serves as the negative control clip across 15 sampled frames.

---

## 3. Sampling Protocol

### 3.1 Frame Selection Strategy
Sampling uses deterministic, stratified frame intervals across each video to avoid temporal over-correlation:
- `ppe-1.mp4`: 20 frames sampled at interval 15 (Frames 0 to 285)
- `ppe-2.mp4`: 20 frames sampled at interval 10 (Frames 0 to 190)
- `ppe-3.mp4`: 20 frames sampled at interval 15 (Frames 0 to 285)
- `people.mp4`: 25 frames sampled at interval 25 (Frames 0 to 600)
- `sample.mp4`: 15 frames sampled at interval 5 (Frames 0 to 70)

### 3.2 Dataset Distribution
| Video Clip | Scenario | Duration / Total Frames | Sampled Frames | Ground Truth Boxes | Track IDs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ppe-1.mp4` | Single worker, industrial | 343 frames @ 29.97 fps | 20 | 20 | `person_1` |
| `ppe-2.mp4` | 2 workers, co-presence | 206 frames @ 25.00 fps | 20 | 40 | `person_1`, `person_2` |
| `ppe-3.mp4` | 3 workers, occlusions | 349 frames @ 30.00 fps | 20 | 60 | `person_1`, `person_2`, `person_3` |
| `people.mp4` | Multi-pedestrian crossing | 632 frames @ 30.00 fps | 25 | 50 | `ped_1`, `ped_2`, `ped_3`, `ped_4` |
| `sample.mp4` | Negative control (vehicles) | 75 frames @ 25.00 fps | 15 | 0 | None (Empty controls) |
| **Total** | **5 Clips** | **1,605 frames** | **100 frames** | **170 boxes** | **10 distinct tracks** |

---

## 4. Integrity, Hashes & Verification

Every v4.0.0 artifact is cryptographically hashed with SHA-256.
- Fixture path: `tests/fixtures/evaluation/multiclip_annotations_v4_independent.json`
- Manifest path: `tests/fixtures/evaluation/multiclip_v4_manifest.json`

Validation rules enforced by `src.videx.evaluation.annotation_validator.validate_multiclip_annotation_data`:
1. Metadata schema conformity (`version == "v4.0.0"`, `policy == "Policy A"`).
2. Clip metadata presence (resolution, fps, frame count).
3. Frame indices strictly monotonic and within `[0, total_frames - 1]`.
4. Pixel coordinate boundaries strictly non-negative and bounded by video width and height.
5. Negative frames strictly validated with `person_count == 0` and empty annotations list.
6. Track ID consistency and label formatting.
