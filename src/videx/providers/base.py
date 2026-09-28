"""VIDEX provider interfaces.

Defines ``Protocol`` classes for every AI/ML component in the pipeline.
Using ``Protocol`` (structural subtyping) means:

- Providers do NOT need to inherit from a base class.
- Any class that implements the required methods is automatically a valid provider.
- Tests can use lightweight stub/mock implementations without importing ML libraries.
- Concrete providers can be swapped by changing configuration alone.

Design rules:
- Interfaces accept and return domain types from ``videx.domain.schemas``.
- ``bytes`` is used for raw frame pixel data to avoid a compile-time numpy dependency
  in this module (implementations convert internally as needed).
- Every method has a clear docstring specifying contract pre/postconditions.
- No implementation logic belongs in this file.

Usage::

    from videx.providers.base import DetectionProvider

    class MyDetector:
        # Implement the DetectionProvider protocol
        ...

    def run_pipeline(detector: DetectionProvider, frame_data: bytes, frame: Frame) -> ...:
        detections = detector.detect(frame_data, frame)
        ...
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from videx.domain.schemas import (
    AudioSegment,
    Detection,
    Event,
    Evidence,
    Frame,
    OCRObservation,
    Track,
    TrajectoryPoint,
    Video,
)

# ``bytes`` represents encoded frame image data (e.g., JPEG or PNG bytes).
# Concrete providers decode this to their preferred array format internally.
FrameBytes = bytes


# ── Detection ──────────────────────────────────────────────────────────────


@runtime_checkable
class DetectionProvider(Protocol):
    """Object detection provider.

    Accepts a raw video frame and returns a list of detected objects.
    Implementations wrap a specific model backend (e.g., YOLOv8, PP-YOLOE+).
    """

    @property
    def provider_name(self) -> str:
        """Unique string key identifying this provider (e.g., 'yolov8l').

        Used as ``Detection.provider`` in output records.
        """
        ...

    def detect(self, frame_data: FrameBytes, frame_meta: Frame) -> list[Detection]:
        """Run inference on a single frame.

        Args:
            frame_data: Encoded frame bytes (JPEG / PNG).
            frame_meta: Frame metadata (video_id, frame_id, timestamp, dimensions).

        Returns:
            List of Detection records. May be empty if nothing is detected.
            Detection IDs and frame/video IDs must be correctly set.

        Raises:
            RuntimeError: If the provider is not initialised or inference fails.
        """
        ...

    def detect_batch(
        self,
        batch: list[tuple[FrameBytes, Frame]],
    ) -> list[list[Detection]]:
        """Run inference on a batch of frames.

        The returned list has the same length as ``batch``, in the same order.
        Default implementations may call ``detect`` in a loop; GPU providers
        should override for true batch inference.

        Args:
            batch: List of (frame_data, frame_meta) pairs.

        Returns:
            List of detection lists, one per input frame.
        """
        ...

    def warmup(self) -> None:
        """Optional: Run a dummy forward pass to initialise the model.

        Call once before processing begins to avoid cold-start latency.
        Providers that do not need warmup should implement as a no-op.
        """
        ...


# ── Tracking ───────────────────────────────────────────────────────────────


@runtime_checkable
class TrackingProvider(Protocol):
    """Multi-object tracking provider.

    Maintains state across frames and associates detections into Tracks.
    The tracker is stateful — it must be called once per frame, in order.
    """

    @property
    def provider_name(self) -> str:
        """Unique string key identifying this tracker (e.g., 'bytetrack')."""
        ...

    def update(
        self,
        detections: list[Detection],
        frame_meta: Frame,
    ) -> list[TrajectoryPoint]:
        """Update the tracker with detections from a new frame.

        This is called once per frame, in chronological order. The tracker
        internally manages track lifecycle (new / active / lost / deleted).

        Args:
            detections: All detections from the current frame.
            frame_meta: Current frame metadata.

        Returns:
            List of TrajectoryPoints for all currently active tracks.
            Each point records the track's position in this frame.
        """
        ...

    def active_tracks(self) -> list[Track]:
        """Return all currently active Track records.

        A Track becomes active when confirmed and remains until deleted.
        This reflects the tracker's internal state after the last ``update`` call.
        """
        ...

    def reset(self) -> None:
        """Reset the tracker state.

        Call this before processing a new video to discard all track state.
        """
        ...


# ── OCR ────────────────────────────────────────────────────────────────────


@runtime_checkable
class OCRProvider(Protocol):
    """Optical character recognition provider.

    Detects and recognises text in a video frame, returning structured
    OCR observations with bounding boxes and confidence scores.
    """

    @property
    def provider_name(self) -> str:
        """Unique string key identifying this OCR provider (e.g., 'easyocr_hi')."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """ISO 639-1 language codes this provider supports (e.g., ['hi', 'en'])."""
        ...

    def recognise(self, frame_data: FrameBytes, frame_meta: Frame) -> list[OCRObservation]:
        """Detect and recognise all text regions in a frame.

        Args:
            frame_data: Encoded frame bytes.
            frame_meta: Frame metadata (provides frame_id, video_id, timestamp).

        Returns:
            List of OCRObservation records. Empty if no text found.
            Each observation includes the detected text, confidence, and bbox.
        """
        ...


# ── ASR ────────────────────────────────────────────────────────────────────


@runtime_checkable
class ASRProvider(Protocol):
    """Automatic speech recognition provider.

    Transcribes the audio track of a video, returning time-aligned
    transcript segments.
    """

    @property
    def provider_name(self) -> str:
        """Unique string key identifying this ASR provider (e.g., 'faster_whisper_hi')."""
        ...

    @property
    def primary_language(self) -> str:
        """Primary language this provider is configured for (ISO 639-1, e.g., 'hi')."""
        ...

    def transcribe(self, audio_data: bytes, video: Video) -> list[AudioSegment]:
        """Transcribe audio and return timestamped segments.

        Args:
            audio_data: Raw audio bytes (WAV, 16kHz mono recommended).
            video: The parent Video record (for video_id association).

        Returns:
            List of AudioSegment records ordered by ``start_timestamp_seconds``.
            Each segment covers a continuous speech utterance or pause.
        """
        ...


# ── VLM / Semantic Reasoning ───────────────────────────────────────────────


@runtime_checkable
class VideoReasoningProvider(Protocol):
    """Vision-language model provider for semantic scene understanding.

    Accepts frame images and optional context, returns natural language
    descriptions or answers to questions about scene content.
    """

    @property
    def provider_name(self) -> str:
        """Unique string key (e.g., 'gpt4o', 'llava_13b')."""
        ...

    def caption_frame(
        self,
        frame_data: FrameBytes,
        frame_meta: Frame,
        context: str | None = None,
    ) -> Evidence:
        """Generate a natural language caption for a single frame.

        Args:
            frame_data: Encoded frame bytes.
            frame_meta: Frame metadata.
            context: Optional context string (e.g., prior captions, detections).

        Returns:
            An Evidence record of type ``EvidenceType.VLM_CAPTION``.
        """
        ...

    def answer_question(
        self,
        question: str,
        frame_data: FrameBytes,
        frame_meta: Frame,
        context: str | None = None,
    ) -> str:
        """Answer a natural language question about a frame.

        Args:
            question: The question to answer.
            frame_data: Encoded frame bytes.
            frame_meta: Frame metadata.
            context: Optional additional context.

        Returns:
            Natural language answer string.
        """
        ...


# ── Event Detection ────────────────────────────────────────────────────────


@runtime_checkable
class EventDetector(Protocol):
    """Event detection engine.

    Analyses pipeline outputs (tracks, trajectory points, evidence records)
    and emits structured Event records when defined conditions are met.
    """

    @property
    def detector_name(self) -> str:
        """Unique string key identifying this detector (e.g., 'rule_based_v1')."""
        ...

    def process_frame(
        self,
        frame_meta: Frame,
        trajectory_points: list[TrajectoryPoint],
        ocr_observations: list[OCRObservation],
        audio_segments: list[AudioSegment],
        additional_context: dict[str, Any] | None = None,
    ) -> list[Event]:
        """Process pipeline outputs for one frame and emit any triggered events.

        This method is called once per frame, in chronological order.
        The event detector is stateful (it tracks dwell times, zone crossings, etc.)

        Args:
            frame_meta: The current frame.
            trajectory_points: Track positions updated in this frame.
            ocr_observations: OCR results from this frame.
            audio_segments: ASR segments active at this frame's timestamp.
            additional_context: Arbitrary extra data (zone configs, thresholds, etc.).

        Returns:
            List of newly triggered Event records. Empty if no events occurred.
        """
        ...

    def finalise(self) -> list[Event]:
        """Emit any remaining events at end-of-video.

        Call once after the last frame has been processed to flush events
        that require end-of-track or end-of-video signals (e.g., loitering
        events that were still active when the video ended).

        Returns:
            List of any remaining Event records.
        """
        ...

    def reset(self) -> None:
        """Reset all event detector state. Call before processing a new video."""
        ...
