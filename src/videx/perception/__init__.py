"""Perception engine: Object detection and multi-object tracking.

Provides concrete implementations of DetectionProvider (YOLO26, MockDetector)
and TrackingProvider (BoT-SORT, MockTracker), pipeline coordination, and
trajectory computation.
"""

from __future__ import annotations

from videx.perception.detection import MockDetector, YOLO26Detector, YOLO26DetectorConfig
from videx.perception.pipeline import PerceptionPipeline, PerceptionResult
from videx.perception.tracking import BoTSORTConfig, BoTSORTTracker, MockTracker
from videx.perception.trajectory import TrajectoryAnalyzer

__all__ = [
    "BoTSORTConfig",
    "BoTSORTTracker",
    "MockDetector",
    "MockTracker",
    "PerceptionPipeline",
    "PerceptionResult",
    "TrajectoryAnalyzer",
    "YOLO26Detector",
    "YOLO26DetectorConfig",
]
