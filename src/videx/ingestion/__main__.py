"""CLI entrypoint for video ingestion.

Usage::

    python -m videx.ingestion <path_to_video> [--json] [--no-scenes]
"""

from __future__ import annotations

import argparse
import sys

from videx.ingestion.base import IngestionError
from videx.ingestion.manifest import format_manifest_summary
from videx.ingestion.service import VideoIngestionService


def main() -> int:
    """CLI entrypoint for running ingestion on a video file."""
    parser = argparse.ArgumentParser(
        prog="python -m videx.ingestion",
        description=(
            "VIDEX Video Ingestion — Validate, extract metadata, and produce a VideoManifest"
        ),
    )
    parser.add_argument("video_path", help="Path to the video file to ingest")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full VideoManifest as formatted JSON instead of text summary",
    )
    parser.add_argument(
        "--no-scenes",
        action="store_true",
        help="Disable content-aware scene detection (treat as single scene)",
    )

    args = parser.parse_args()

    try:
        service = VideoIngestionService(detect_scenes=not args.no_scenes)
        manifest = service.ingest(args.video_path)

        if args.json:
            print(manifest.model_dump_json(indent=2))
        else:
            print(format_manifest_summary(manifest))

        return 0
    except IngestionError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error during ingestion: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
