"""Phase 1 Manual Smoke Test Script.

Generates a sample.mp4 video, runs the ingestion service, demonstrates:
1. VideoManifest generation & summary output
2. Deterministic frame retrieval at timestamp
3. Scene boundary identification
4. JSON export
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videx.ingestion.manifest import format_manifest_summary
from videx.ingestion.service import VideoIngestionService


def create_sample_video(path: Path) -> None:
    """Create a 3-second sample video with 2 scenes (75 frames at 25 fps)."""
    width, height = 320, 240
    fps = 25.0
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    try:
        # Scene 1: Green frames (frames 0..39, 1.6s)
        for i in range(40):
            frame = np.full((height, width, 3), (0, 180, 0), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Scene 1 - Frame {i}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            out.write(frame)

        # Scene 2: Red frames (frames 40..74, 1.4s)
        for i in range(40, 75):
            frame = np.full((height, width, 3), (0, 0, 200), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Scene 2 - Frame {i}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            out.write(frame)
    finally:
        out.release()


def main() -> None:
    sample_path = Path("sample.mp4")
    print(f"Creating sample video: {sample_path.resolve()} ...")
    create_sample_video(sample_path)

    print("\n--- Running VideoIngestionService ---")
    service = VideoIngestionService(detect_scenes=True)
    manifest = service.ingest(sample_path)

    print("\n" + format_manifest_summary(manifest))

    print("\n--- Demonstrating Deterministic Frame Access ---")
    with service.create_reader(manifest) as reader:
        test_ts = 1.0  # inside Scene 1
        frame_meta, frame_bytes = reader.read_frame_at_timestamp(test_ts)
        print(
            f"Frame at timestamp {test_ts:.2f}s:\n"
            f"  Frame Number: {frame_meta.frame_number}\n"
            f"  Reported PTS: {frame_meta.timestamp_seconds:.3f}s\n"
            f"  Dimensions:   {frame_meta.width}x{frame_meta.height}\n"
            f"  JPEG Bytes:   {len(frame_bytes):,} bytes"
        )

        test_ts2 = 2.0  # inside Scene 2
        frame_meta2, frame_bytes2 = reader.read_frame_at_timestamp(test_ts2)
        print(
            f"\nFrame at timestamp {test_ts2:.2f}s:\n"
            f"  Frame Number: {frame_meta2.frame_number}\n"
            f"  Reported PTS: {frame_meta2.timestamp_seconds:.3f}s\n"
            f"  Dimensions:   {frame_meta2.width}x{frame_meta2.height}\n"
            f"  JPEG Bytes:   {len(frame_bytes2):,} bytes"
        )

    print("\n--- Demonstrating Scene Boundaries ---")
    print(f"Total Scenes Detected: {len(manifest.scenes)}")
    for sc in manifest.scenes:
        print(
            f"  Scene [{sc.scene_index}]: "
            f"Time {sc.start_timestamp_seconds:.2f}s -> {sc.end_timestamp_seconds:.2f}s "
            f"(Frames {sc.start_frame_number}..{sc.end_frame_number})"
        )

    print("\n--- Smoke Test Succeeded ---")


if __name__ == "__main__":
    main()
