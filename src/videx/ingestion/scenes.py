"""Scene detection abstractions and implementations.

Provides content-aware scene boundary segmentation using PySceneDetect, with clean
fallback options for single-scene videos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from videx.domain.schemas import Scene
from videx.ingestion.base import SceneDetector

logger = logging.getLogger(__name__)


class SingleSceneDetector(SceneDetector):
    """Fallback scene detector that treats the entire video as a single scene.

    Useful for short clips, testing, or when scene detection is explicitly disabled.
    """

    @property
    def detector_name(self) -> str:
        return "single_scene"

    def detect_scenes(
        self,
        video_path: Path | str,
        video_id: UUID,
        fps: float,
        total_frames: int | None = None,
        duration_seconds: float | None = None,
    ) -> list[Scene]:
        frames = total_frames if total_frames is not None and total_frames > 0 else 1
        if duration_seconds is not None and duration_seconds > 0.0:
            duration = duration_seconds
        else:
            duration = frames / fps if fps > 0 else 0.0

        return [
            Scene(
                video_id=video_id,
                scene_index=0,
                start_frame_number=0,
                end_frame_number=max(0, frames - 1),
                start_timestamp_seconds=0.0,
                end_timestamp_seconds=duration,
            )
        ]


class PySceneDetectDetector(SceneDetector):
    """Content-aware scene detector powered by PySceneDetect.

    Identifies shot transitions and scene changes using differences in frame content.
    """

    def __init__(self, threshold: float = 27.0, min_scene_len_frames: int = 15) -> None:
        """Initialize detector with sensitivity threshold.

        Args:
            threshold: Content change threshold (lower is more sensitive, default 27.0).
            min_scene_len_frames: Minimum frame length of a detected scene.
        """
        self.threshold = threshold
        self.min_scene_len_frames = min_scene_len_frames

    @property
    def detector_name(self) -> str:
        return "pyscenedetect_content"

    def detect_scenes(
        self,
        video_path: Path | str,
        video_id: UUID,
        fps: float,
        total_frames: int | None = None,
        duration_seconds: float | None = None,
    ) -> list[Scene]:
        resolved_path = Path(video_path).resolve()

        try:
            from scenedetect import ContentDetector, detect  # type: ignore[import-untyped]

            detector = ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len_frames,
            )
            scene_list = detect(str(resolved_path), detector)

            if not scene_list:
                return SingleSceneDetector().detect_scenes(
                    video_path=resolved_path,
                    video_id=video_id,
                    fps=fps,
                    total_frames=total_frames,
                    duration_seconds=duration_seconds,
                )

            result: list[Scene] = []
            for idx, (start_tc, end_tc) in enumerate(scene_list):
                if hasattr(start_tc, "frame_num"):
                    start_f = int(start_tc.frame_num)
                else:
                    start_f = int(start_tc.get_frames())

                if hasattr(end_tc, "frame_num"):
                    end_f_raw = int(end_tc.frame_num)
                else:
                    end_f_raw = int(end_tc.get_frames())
                end_f = max(start_f, end_f_raw - 1)

                if hasattr(start_tc, "seconds"):
                    start_sec = float(start_tc.seconds)
                else:
                    start_sec = float(start_tc.get_seconds())

                if hasattr(end_tc, "seconds"):
                    end_sec = float(end_tc.seconds)
                else:
                    end_sec = float(end_tc.get_seconds())

                result.append(
                    Scene(
                        video_id=video_id,
                        scene_index=idx,
                        start_frame_number=start_f,
                        end_frame_number=end_f,
                        start_timestamp_seconds=start_sec,
                        end_timestamp_seconds=end_sec,
                    )
                )

            return result

        except Exception as err:
            logger.warning("PySceneDetect failed (%s); falling back to SingleSceneDetector", err)
            return SingleSceneDetector().detect_scenes(
                video_path=resolved_path,
                video_id=video_id,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration_seconds,
            )
