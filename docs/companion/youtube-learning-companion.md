# VIDEX YouTube Learning Companion

## 1. Overview & Architecture

The **VIDEX YouTube Learning Companion** (Phase 9.2) extends VIDEX to transform YouTube educational videos and entire course playlists into accurate, evidence-grounded, timestamp-linked study notes.

Built specifically for rigorous technical disciplines and competitive exams (such as **GATE Data Science & AI** and **GATE Computer Science**), it enforces a strict zero-hallucination policy: **every substantive mathematical assertion, definition, theorem, formula, worked example, and exam trap is linked to authoritative timestamped source evidence**.

```
                           ┌──────────────────────────────┐
                           │   YouTube Video / Playlist   │
                           └──────────────┬───────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             [YouTubeConnector]                [Compliant Transcript Layer]
           Official API v3 / Web               - FileTranscriptProvider (JSON/VTT/SRT)
         Pagination & Deduplication            - Local VIDEX ASR (Whisper)
                        │                      - Official Caption Provider
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                             [StudyNoteSynthesizer]
                             - Bounded Window Chunking
                             - LaTeX Equation Extraction
                             - Evidence Grounding Gate
                             - Epistemic Uncertainty Audit
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
              [MarkdownExporter]               [PlaylistJobOrchestrator]
            - Clickable Timestamps            - Resumable Manifest
            - Obsidian Callouts               - 00_Playlist_Index.md
            - LaTeX Math Blocks               - 00_Topic_Index.md
            - Wiki-links                      - Quality Gate Evaluation
```

---

## 2. Configuration & Credentials

All configuration is loaded via `videx.config.Settings` (read from environment variables or `.env`).

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `YOUTUBE_API_KEY` | `""` (empty) | Google Cloud YouTube Data API v3 key (optional) |
| `YOUTUBE_API_BASE_URL` | `https://www.googleapis.com/youtube/v3` | Endpoint base URL |
| `YOUTUBE_REQUEST_TIMEOUT_SECONDS` | `15.0` | Timeout in seconds for HTTP requests |
| `YOUTUBE_MAX_RESULTS_PER_PAGE` | `50` | Pagination batch size (1–50) |
| `NOTES_EXPORT_DIR` | `data/notes` | Target directory for generated notes & manifests |

### Setting Up Official YouTube API Credentials

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. In **APIs & Services**, enable the **YouTube Data API v3**.
4. In **Credentials**, click **Create Credentials** -> **API Key**.
5. Copy the key and add it to your `.env` file:
   ```bash
   YOUTUBE_API_KEY=AIzaSy...
   ```
6. **Graceful Fallback:** If `YOUTUBE_API_KEY` is not provided, VIDEX automatically operates in public web discovery mode or local fixture mode, emitting clear actionable guidance.

---

## 3. Compliant Access Layer & Policy Boundaries

VIDEX strictly respects YouTube terms of service, copyright, access controls, and API policies:
- **No Circumvention:** VIDEX does not attempt unauthorized decryption, scraping of private streams, or circumvention of bot safeguards.
- **No Hallucination on Missing Transcripts:** If a video's transcript is unavailable, VIDEX **never** invents lecture content or substitutes hallucinated explanations. Instead, it generates a structured note with status `AWAITING_TRANSCRIPT`, an explicit uncertainty warning, and actionable instructions detailing how to supply a transcript or authorized media file.
- **Supported Transcript Ingestion Modes:**
  1. `FileTranscriptProvider`: Parses user-supplied `.json`, `.vtt` (WebVTT), `.srt` (SubRip), or timestamped `.txt` files.
  2. `VIDEXASRTranscriptProvider`: Transcribes authorized local audio/video media files using the existing VIDEX Whisper pipeline.
  3. `YouTubeCaptionProvider`: Accesses captions via official YouTube APIs when credentials and permissions allow.
  4. `CompositeTranscriptProvider`: Chains multiple providers in prioritized sequence.

---

## 4. Structured Study Note Contract & GATE Profile

Per-video study notes adhere to a mathematically rigorous output profile tailored for GATE Linear Algebra revision:

1. **Frontmatter & Metadata:** Video ID, canonical source URL, duration, transcript coverage percentage, lecturer, and tags.
2. **Learning Objectives:** Concise, outcome-focused bullet points.
3. **Executive Summary:** High-level conceptual overview.
4. **Thematic Sections with Clickable Timestamps:**
   - Clickable YouTube timestamp links: `https://youtube.com/watch?v={video_id}&t={seconds}s`
   - Formatted labels: `[MM:SS]` or `[HH:MM:SS]`
   - Definitions: Term, formal definition, notation, and assumptions.
   - Theorems: Name, conditions/hypotheses, and mathematical conclusion.
   - Formulas: Balanced LaTeX expressions (e.g. `$A x = \lambda x$`), variable definitions, and applicability conditions.
   - Worked Examples: Step-by-step problem statements, calculations, and key insights.
   - GATE Exam Traps: Common pitfalls, mathematically correct concepts, and exam trap explanations.
5. **High-Yield Revision Takeaways:** Core points ranked by importance (`essential`, `high`, `revision`).
6. **Source-Grounded Practice Questions:** Self-test questions with collapsible answers/hints.
7. **Epistemic Uncertainties:** Explicit flags for audio ambiguities, corrupted formulas, or omitted steps.

---

## 5. Playlist Orchestration & Resumability

The `PlaylistJobOrchestrator` provides scalable batch processing for entire courses:
- **Preserves Sequence:** Maintains natural course progression order ($1, 2, \dots, N$).
- **Resumable State:** Persists execution progress in `processing_manifest.json`. If interrupted, restarts from the last unfinished video without re-running completed notes.
- **Retry Control:** Supports selective retrying of failed or partial videos (`retry_failed=True`).
- **Comprehensive Indexing:**
  - `00_Playlist_Index.md`: Course syllabus table linking to every study note and YouTube video.
  - `00_Topic_Index.md`: Cross-video index cross-referencing definitions, theorems, and formulas across all lectures.

---

## 6. Benchmark Playlist Audit: GATE Linear Algebra

- **Playlist URL:** `https://youtube.com/playlist?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC`
- **Course Title:** *Linear Algebra for GATE DA | Complete Course* by Sachin Mittal (GO Classes).
- **Discovered Video Count:** 79 videos (verified dynamically via `playlistVideoListRenderer` without hardcoding).
- **Benchmark Pilot Lecture:** `DrCeIbpfuzE` (*1a. Why Study Linear Algebra | The Importance of Linear Algebra in Machine Learning*).
  - Duration: 1,331 seconds (~22 minutes).
  - Coverage: 100% transcript grounding.
  - Evaluation Gate: Passed with 0 unsupported claims, 100% valid LaTeX equations, and 100% valid timestamp intervals.

---

## 7. Python API Sample Usage

```python
from pathlib import Path
from videx.companion import (
    YouTubeConnector,
    FileTranscriptProvider,
    StudyNoteSynthesizer,
    MarkdownExporter,
    PlaylistJobOrchestrator,
    evaluate_study_note,
)

# 1. Single Lecture Note Generation
connector = YouTubeConnector()
video_meta = connector.get_video_metadata("https://www.youtube.com/watch?v=DrCeIbpfuzE")

transcript_provider = FileTranscriptProvider(
    file_map={"DrCeIbpfuzE": "tests/fixtures/companion/DrCeIbpfuzE.json"}
)
transcript = transcript_provider.get_transcript("DrCeIbpfuzE")

synthesizer = StudyNoteSynthesizer()
note = synthesizer.synthesize(video_meta, transcript)

# 2. Markdown Export
exporter = MarkdownExporter(enable_obsidian=True)
exporter.write_note_file(note, output_dir="data/notes")

# 3. Grounding Audit & Quality Gate
eval_result = evaluate_study_note(note, reference_concepts=["vector", "matrix", "linear transformation"])
print(f"Grounding ratio: {1.0 - eval_result.unsupported_claim_rate:.1%}")
print(f"Quality gate passed: {eval_result.is_acceptable}")

# 4. Resumable Playlist Orchestration
orchestrator = PlaylistJobOrchestrator(
    connector=connector,
    transcript_provider=transcript_provider,
    output_dir="data/notes",
)
manifest = orchestrator.process_playlist(
    "https://youtube.com/playlist?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC",
    limit=5,
)
print(f"Processed: {manifest.processed_count}, Awaiting: {manifest.awaiting_transcript_count}")
```
