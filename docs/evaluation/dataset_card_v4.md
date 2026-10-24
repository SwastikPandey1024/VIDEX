# Dataset Card: VIDEX Multi-Clip Ground Truth Dataset v4.0.0

## Dataset Summary
- **Dataset Name:** VIDEX Multi-Clip Ground Truth Dataset v4.0.0
- **Version:** `v4.0.0`
- **Release Date:** October 2026
- **Policy Standard:** Policy A (Strict Visible Person Bounding Boxes, clamped to frame bounds)
- **Total Clips:** 5 real-world video sources
- **Total Annotated Frames:** 100 frames
- **Total Ground Truth Detections:** 170 bounding boxes
- **Negative Control Frames:** 15 frames (0 human detections)

---

## Intended Use
This dataset is designed for standardized, multi-scenario evaluation of:
1. Object detection recall, precision, and false discovery rate (FDR) on human bodies across diverse visual contexts (industrial workwear, street clothing, long distance, close up).
2. Multi-object tracker ID consistency, fragmentation, and ID-switch resilience.
3. False positive rate verification against negative controls (scenes with traffic/vehicles but no pedestrians).

---

## Clip Breakdown
| Clip Identifier | Source File | Dimensions | FPS | Total Frames | Sampled | GT Boxes | Scenarios & Visual Challenges |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `video_ppe1_001` | `ppe-1.mp4` | 1280x720 | 29.97 | 343 | 20 | 20 | Industrial indoor, solitary worker, high visibility vest, stationary and walking motion. |
| `video_ppe2_001` | `ppe-2.mp4` | 1280x720 | 25.00 | 206 | 20 | 40 | Industrial warehouse, dual workers, parallel motion, equipment handling, partial body occlusions. |
| `video_ppe3_001` | `ppe-3.mp4` | 1280x720 | 30.00 | 349 | 20 | 60 | Industrial yard, 3 workers, heavy background clutter, person crossing, scale differences (foreground vs background). |
| `video_people_001`| `people.mp4`| 1280x720 | 30.00 | 632 | 25 | 50 | Urban pedestrian crossing, varying pedestrian speeds, overlapping trajectories, natural lighting. |
| `video_sample_001`| `sample.mp4`| 320x240  | 25.00 | 75  | 15 | 0  | **Negative Control**: Moving cars, road environment, zero pedestrians. Tests false positive suppression. |

---

## Provenance and Review
- **Origin:** Real video clips stored under `Sample_Videos/`.
- **Generation Method:** Stratified sampling across temporal progression; ground truth bounding boxes verified using Policy A visual extents.
- **Independence:** Generated and validated independently of model inference weights. Bounding coordinates represent verified human presence, not YOLO model output.
- **Integrity:** Validated against `src.videx.evaluation.annotation_validator.validate_multiclip_annotation_data()`.

---

## Limitations
- **Resolution Distribution:** Four clips are 720p (`1280x720`) and one negative control is 240p (`320x240`). Future iterations should incorporate 1080p and 4K footage.
- **Lighting Conditions:** Predominantly daylight and well-lit indoor industrial environments; night-vision and low-light scenarios remain future evaluation targets.
- **Negative Control Scope:** Negative control contains vehicles and road scenes; future versions should add indoor negative frames (e.g. empty factories or conference rooms).
