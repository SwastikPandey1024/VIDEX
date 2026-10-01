# VIDEX Sample Evaluation & Demo Video Corpus

This directory documents the canonical local evaluation and demonstration video corpus located at `Sample_Videos/`.

## 1. Corpus Purpose & Scope

The `Sample_Videos` dataset serves as the **local runtime evaluation and demonstration suite** for the end-to-end VIDEX multimodal video intelligence system (Layers 1 through 5).

> [!NOTE]
> This corpus is an evaluation, verification, and regression demo dataset, **not a formal academic benchmark**. It provides real-world media streams across traffic, pedestrian density, personal protective equipment (PPE), and audio interactions to validate pipeline robustness under diverse codecs, resolutions, and multimodal conditions.

---

## 2. Media Inventory & Manifest

The full machine-readable metadata manifest is maintained at [datasets/manifests/sample_videos.json](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/datasets/manifests/sample_videos.json).

| Filename | Size (MB) | Duration | Resolution | FPS | Video Codec | Audio Stream | Domain / Key Features |
|---|---|---|---|---|---|---|---|
| `bikes.mp4` | 2.99 MB | 9.14 s | 1280x720 | 50.0 | H.264 (CFR) | AAC 48kHz Stereo | Urban bicycle movement, high frame rate (50 fps) |
| `cars.mp4` | 9.07 MB | 12.23 s | 1280x720 | 30.0 | H.264 (CFR) | AAC 48kHz Stereo | Traffic flow, vehicles, turn kinematics |
| `motorbikes.mp4` | 8.25 MB | 27.40 s | 1366x720 | 25.0 | H.264 (CFR) | None | Two-wheeler traffic, lane movement, longer duration |
| `people.mp4` | 11.06 MB | 21.03 s | 1280x720 | 30.0 | H.264 (CFR) | AAC 48kHz Stereo | Pedestrian crowding, loitering, multi-person tracking |
| `ppe-1.mp4` | 1.45 MB | 11.41 s | 1280x720 | 29.97 | H.264 (CFR) | None | Industrial safety, hardhat/vest detection |
| `ppe-2.mp4` | 2.53 MB | 8.20 s | 1280x720 | 25.0 | H.264 (CFR) | None | Construction site compliance, worker interactions |
| `ppe-3.mp4` | 3.21 MB | 11.60 s | 1280x720 | 30.0 | H.264 (CFR) | AAC 48kHz Stereo | PPE compliance with background ambient audio |
| `sample.mp4` | 0.08 MB | 2.96 s | 320x240 | 25.0 | MPEG-4 (CFR) | None | Minimal lightweight smoke/unit test stream |

**Total Corpus Size:** ~38.64 MB across 8 video assets.

---

## 3. Git Retention & Storage Policy

- All 8 sample video files are small (each < 12 MB, total 38.6 MB).
- Because the total corpus size is well below typical repository limits (< 50 MB total) and contains essential evaluation assets for testing, the binaries are committed directly under `Sample_Videos/`.
- No Git LFS overhead or external blob store download is required for running evaluations.
- If larger test files (> 50 MB) are introduced in future phases, they must be placed under `data/videos/` (which is `.gitignore`d) or registered via an external dataset download script.

---

## 4. Evaluation Workflow

To run the complete automated evaluation against this corpus:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/evaluate_sample_corpus.ps1
```

Or via direct python runner:

```bash
uv run python scripts/evaluate_sample_corpus.py
```
