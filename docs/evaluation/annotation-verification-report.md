# VIDEX Phase 9.1 — Pilot Annotation Verification and Provenance Report

## 1. Overview & Verification Methodology

In Phase 9.1, all pilot ground-truth video records were audited directly against physical video frames using `cv2.VideoCapture` and `OpenCVVideoReader.read_decoded_frame()`.

Each annotated example was audited against six criteria:
1. **Source Visibility/Audibility**: Is the annotated entity or speech physically present in the video container?
2. **Temporal Alignment**: Do presentation timestamps (PTS) match the observable media?
3. **Coordinate Conventions**: Do bounding boxes adhere to top-left origin pixel coordinates within frame resolution boundaries?
4. **Trajectory Identity**: Are multi-frame tracks consistent across observed frames?
5. **Event Intervals**: Are temporal boundary intervals supported by physical motion?
6. **QA Answerability**: Are answerable queries substantiated by evidence, and are unanswerable queries genuinely unobservable?

---

## 2. Video Audit Findings

### 2.1 Video 1: `sample.mp4`

| Property | Value |
|---|---|
| **Source File** | `Sample_Videos/sample.mp4` |
| **Resolution** | $320 \times 240$ pixels |
| **FPS / Frames** | $25.0$ fps, $75$ frames ($3.000$ seconds) |
| **Audio Stream** | None (silent MP4 container) |
| **Visual Content** | Synthetic scene detection test clip: Scene 1 (frames 0–49, solid green background), Scene 2 (frames 50–74, solid red background) |

#### Audit Findings & Corrections:
1. **Detections & Tracks (Historical Fabrication Detected)**:
   - *Previous Annotation*: Annotated a `car` object moving from $(20, 50, 100, 80)$ at frame 0 to $(60, 50, 100, 80)$ at frame 25.
   - *Physical Reality*: Physical inspection of every decoded frame revealed that **no vehicle, car, or street scene exists in `sample.mp4`**.
   - *Correction*: Retained the record strictly as a historical synthetic test fixture, but explicitly changed status to `AnnotationStatus.ANNOTATION_INCOMPLETE` with documented provenance.
2. **On-Screen Text (OCR)**:
   - *Previous Annotation*: Marked `not_applicable` with rationale *"No legible on-screen text"*.
   - *Physical Reality*: Inverted text inspection confirmed that `sample.mp4` contains white rendered text on every frame:
     - Frame 0 (pts 0.0s): `"Scene 1 - Frame 0"` at $(x=20, y=104, w=225, h=18)$
     - Frame 25 (pts 1.0s): `"Scene 1 - Frame 25"` at $(x=20, y=104, w=239, h=18)$
     - Frame 50 (pts 2.0s): `"Scene 2 - Frame 50"` at $(x=20, y=104, w=239, h=18)$
   - *Correction*: Updated OCR modality status to `AnnotationStatus.ANNOTATED` with verified ground-truth text observations.
3. **Question-Answering (QA)**:
   - *Previous QA Item 1*: *"Is there a vehicle visible in the clip?"* with reference answer *"Yes, a car is visible..."*.
   - *Physical Reality*: Factual error; no vehicle is visible.
   - *Correction*: Reference answer updated to: *"No, there is no vehicle visible in sample.mp4; the video contains synthetic text on solid color backgrounds."*
   - *QA Item 2*: *"What is the spoken dialogue in the clip?"* (`is_answerable=False`): Verified; clip is silent, correct abstention expected.

---

### 2.2 Video 2: `ppe-1.mp4`

| Property | Value |
|---|---|
| **Source File** | `Sample_Videos/ppe-1.mp4` |
| **Resolution** | $1280 \times 720$ pixels |
| **FPS / Frames** | $29.97$ fps, $343$ frames ($11.445$ seconds) |
| **Audio Stream** | None (silent MP4 container) |
| **Visual Content** | Industrial workplace surveillance camera recording a worker in high-visibility vest and hardhat |

#### Audit Findings & Corrections:
1. **Detections & Tracks**:
   - Frame 0 (pts 0.0s): Worker bounding box at $(x=450, y=120, w=320, h=550)$. Verified: Worker is standing in this bounding region.
   - Frame 90 (pts 3.0s): Worker bounding box at $(x=460, y=125, w=315, h=545)$. Verified: Worker is inside this region.
   - Track `track_worker_1`: Monotonic timestamps, single identity across frames. Verified.
2. **OCR & Audio**:
   - `ocr`: No legible signage or text in workplace scene. Marked `not_applicable` with verified rationale.
   - `transcripts`: Silent video. Marked `not_applicable` with verified rationale.
3. **Temporal Events**:
   - Event `worker_active` ($0.0$s to $11.0$s): Worker is continuously present and active in scene. Verified.
4. **Question-Answering (QA)**:
   - QA 1: *"Is a worker present in the scene?"* (`is_answerable=True`): Verified.
   - QA 2: *"What is the worker's employee badge number?"* (`is_answerable=False`): Verified; badge number is unobservable in footage, model must abstain.

---

## 3. Domain Scope & Educational Baseline Blocker

> [!WARNING]
> **Surveillance Footage is NOT an Educational Evaluation Set.**
>
> Neither `sample.mp4` nor `ppe-1.mp4` represent educational content (lectures, slides, pedagogical speech). The Learning Companion educational baseline remains **strictly blocked** until:
> 1. An authorized, permitted academic lecture video clip is acquired.
> 2. Multimodal ground-truth annotations (slide OCR, spoken terminology transcriptions, pedagogical concept transitions, and lecture QA) are manually annotated and peer-reviewed.

No mock or synthetic evaluation will be substituted for the educational baseline.
