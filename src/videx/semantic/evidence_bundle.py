"""Deterministic EvidenceBundle construction and temporal keyframe selection."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from videx.audio.pipeline import AudioPipelineResult
from videx.domain.schemas import BoundingBox, FrameTimestamp, TimestampSource
from videx.events.engine import EventTimeline
from videx.events.schemas import Event
from videx.ingestion.base import DecodedFrame, VideoReader
from videx.ocr.pipeline import OCRPipelineResult
from videx.perception.pipeline import PerceptionResult
from videx.semantic.crops import CropExtractor
from videx.semantic.schemas import CandidateEvent, CropRegion, EvidenceBundle, KeyframeMetadata

logger = logging.getLogger(__name__)


class EvidenceBundleBuilder:
    """Builds a minimal, self-contained EvidenceBundle for a CandidateEvent.

    Applies deterministic, timestamp-driven keyframe selection (leading, midpoint,
    trailing) and extracts targeted bounding-box crops without duplicating raw video.
    """

    def __init__(
        self,
        crop_extractor: CropExtractor | None = None,
        max_crops_per_bundle: int = 6,
        extract_crops_with_bytes: bool = True,
    ) -> None:
        self.crop_extractor = crop_extractor or CropExtractor()
        self.max_crops_per_bundle = max_crops_per_bundle
        self.extract_crops_with_bytes = extract_crops_with_bytes

    def build_bundle(
        self,
        candidate: CandidateEvent,
        timeline: EventTimeline | list[Event],
        video_reader: VideoReader | None = None,
        perception_result: PerceptionResult | None = None,
        ocr_result: OCRPipelineResult | None = None,
        audio_result: AudioPipelineResult | None = None,
    ) -> EvidenceBundle:
        """Construct an EvidenceBundle for the given candidate.

        Args:
            candidate: The CandidateEvent to package evidence for.
            timeline: Full EventTimeline or event collection for cross-referencing.
            video_reader: Optional VideoReader for reading frame pixels and keyframes.
            perception_result: Optional PerceptionResult containing tracks and trajectories.
            ocr_result: Optional OCRPipelineResult containing fused text observations.
            audio_result: Optional AudioPipelineResult containing speech transcripts.

        Returns:
            EvidenceBundle with keyframes, crops, and synchronized multimodal context.
        """
        all_events: list[Event] = (
            timeline.events if isinstance(timeline, EventTimeline) else list(timeline)
        )
        event_map: dict[UUID, Event] = {e.event_id: e for e in all_events}

        # 1. Retrieve supporting events for this candidate
        supporting_events = [
            event_map[ev_id] for ev_id in candidate.source_event_ids if ev_id in event_map
        ]

        # 2. Select temporal keyframes (timestamp-driven)
        keyframes = self._select_keyframes(
            start_ts=candidate.start_timestamp,
            end_ts=candidate.end_timestamp,
            video_reader=video_reader,
        )

        # 3. Extract targeted crops from keyframes or supporting visual observations
        crops = self._extract_candidate_crops(
            candidate=candidate,
            supporting_events=supporting_events,
            keyframes=keyframes,
            video_reader=video_reader,
            perception_result=perception_result,
        )

        # 4. Gather trajectory summaries
        trajectories = self._summarize_trajectories(
            candidate=candidate,
            perception_result=perception_result,
        )

        # 5. Gather OCR text observations active in candidate window
        ocr_texts = self._gather_ocr_context(
            candidate=candidate,
            ocr_result=ocr_result,
            supporting_events=supporting_events,
        )

        # 6. Gather speech transcript segments active in candidate window
        transcripts = self._gather_transcript_context(
            candidate=candidate,
            audio_result=audio_result,
            supporting_events=supporting_events,
        )

        # 7. Collect spatial zone context
        zones = self._gather_spatial_context(supporting_events)

        # 8. Compile consolidated evidence IDs
        all_evidence_ids: list[UUID] = list(candidate.evidence_ids)
        for ev in supporting_events:
            for eid in ev.evidence_ids:
                if eid not in all_evidence_ids:
                    all_evidence_ids.append(eid)
        for crop in crops:
            if crop.source_evidence_id not in all_evidence_ids:
                all_evidence_ids.append(crop.source_evidence_id)

        return EvidenceBundle(
            candidate_id=candidate.candidate_id,
            video_id=candidate.video_id,
            start_timestamp_seconds=candidate.start_timestamp,
            end_timestamp_seconds=candidate.end_timestamp,
            keyframes=keyframes,
            crops=crops,
            trajectory_summary=trajectories,
            ocr_observations=ocr_texts,
            transcript_segments=transcripts,
            spatial_context=zones,
            supporting_event_ids=candidate.source_event_ids,
            evidence_ids=all_evidence_ids,
            provenance_metadata={
                "builder_version": "1.0",
                "keyframes_count": len(keyframes),
                "crops_count": len(crops),
                "ocr_count": len(ocr_texts),
                "transcripts_count": len(transcripts),
            },
        )

    def _select_keyframes(
        self,
        start_ts: float,
        end_ts: float,
        video_reader: VideoReader | None,
    ) -> list[KeyframeMetadata]:
        """Select representative keyframes: leading, midpoint, and trailing.

        For instantaneous events (duration <= 0.1s), selects a single frame.
        """
        duration = max(0.0, end_ts - start_ts)
        targets: list[tuple[float, str]] = []

        if duration <= 0.10:
            targets.append((start_ts, "instantaneous"))
        else:
            mid_ts = round((start_ts + end_ts) / 2.0, 3)
            targets.append((start_ts, "leading"))
            targets.append((mid_ts, "midpoint"))
            targets.append((end_ts, "trailing"))

        keyframes: list[KeyframeMetadata] = []
        for ts, role in targets:
            frame_idx = 0
            frame_ts_meta: FrameTimestamp | None = None

            if video_reader is not None:
                # Clamp timestamp to valid video duration
                dur = getattr(video_reader, "duration_seconds", None)
                clamped_ts = max(0.0, min(ts, (dur - 0.001) if dur and dur > 0 else ts))

                # Query index from reader
                ts_idx = getattr(video_reader, "timestamp_index", None) or getattr(
                    video_reader, "_timing_index", None
                )
                if ts_idx is not None and hasattr(ts_idx, "get_frame_for_timestamp"):
                    try:
                        frame_idx = ts_idx.get_frame_for_timestamp(clamped_ts)
                        if hasattr(ts_idx, "get_frame_timestamp"):
                            frame_ts_meta = ts_idx.get_frame_timestamp(frame_idx)
                    except Exception:
                        frame_idx = 0
                elif hasattr(video_reader, "read_frame_at_timestamp"):
                    try:
                        frame_tuple = video_reader.read_frame_at_timestamp(clamped_ts)
                        if frame_tuple is not None:
                            frame_obj, _ = frame_tuple
                            frame_idx = frame_obj.frame_number
                            frame_ts_meta = frame_obj.frame_timestamp
                    except Exception as exc:
                        logger.debug("Could not read frame at timestamp %.3fs: %s", clamped_ts, exc)
                        frame_idx = 0
            else:
                # Fallback index calculation for testing without active video file
                frame_idx = int(round(ts * 30.0))
                frame_ts_meta = FrameTimestamp(
                    frame_index=frame_idx,
                    pts_seconds=ts,
                    timestamp_source=TimestampSource.DERIVED,
                )

            keyframes.append(
                KeyframeMetadata(
                    frame_index=frame_idx,
                    timestamp_seconds=ts,
                    role=role,
                    frame_timestamp=frame_ts_meta,
                )
            )

        return keyframes

    def _extract_candidate_crops(
        self,
        candidate: CandidateEvent,
        supporting_events: list[Event],
        keyframes: list[KeyframeMetadata],
        video_reader: VideoReader | None,
        perception_result: PerceptionResult | None,
    ) -> list[CropRegion]:
        """Extract bounding-box crops of key participants across keyframe moments."""
        crops: list[CropRegion] = []
        if video_reader is None or perception_result is None:
            return crops

        # Map trajectory points by track_id and frame_index
        points_by_track: dict[UUID, dict[int, BoundingBox]] = {}
        for trk in perception_result.tracks:
            # Match trajectory points
            t_points = perception_result.trajectories.get(trk.track_id, [])
            points_by_track[trk.track_id] = {tp.frame_number: tp.bbox for tp in t_points}

        # Candidate participants
        track_uuids = [UUID(p) for p in candidate.participant_ids if self._is_uuid(p)]

        for kf in keyframes:
            if len(crops) >= self.max_crops_per_bundle:
                break

            decoded: DecodedFrame | None = None
            try:
                decoded = video_reader.read_decoded_frame(kf.frame_index)
            except Exception as e:
                logger.debug(f"Could not read frame {kf.frame_index} for crop: {e}")
                continue

            if decoded is None:
                continue

            # Look for active track bounding boxes at this frame
            for trk_id in track_uuids:
                if len(crops) >= self.max_crops_per_bundle:
                    break
                trk_bboxes = points_by_track.get(trk_id, {})
                box = trk_bboxes.get(kf.frame_index)
                if box is not None:
                    # Find associated evidence ID
                    ev_id = (
                        candidate.evidence_ids[0]
                        if candidate.evidence_ids
                        else candidate.candidate_id
                    )
                    try:
                        crop = self.crop_extractor.extract_crop(
                            frame=decoded,
                            bbox=box,
                            source_evidence_id=ev_id,
                            frame_index=kf.frame_index,
                            timestamp_seconds=kf.timestamp_seconds,
                            encode_jpeg=self.extract_crops_with_bytes,
                        )
                        crops.append(crop)
                    except Exception as err:
                        logger.debug(f"Crop extraction skipped: {err}")

        return crops

    def _summarize_trajectories(
        self,
        candidate: CandidateEvent,
        perception_result: PerceptionResult | None,
    ) -> list[dict[str, Any]]:
        """Summarize motion direction, velocity, and distance for active tracks."""
        summaries: list[dict[str, Any]] = []
        if perception_result is None:
            return summaries

        track_uuids = {UUID(p) for p in candidate.participant_ids if self._is_uuid(p)}

        for trk in perception_result.tracks:
            if trk.track_id in track_uuids:
                pts = [
                    p
                    for p in perception_result.trajectories.get(trk.track_id, [])
                    if candidate.start_timestamp <= p.timestamp_seconds <= candidate.end_timestamp
                ]
                if pts:
                    first_p = pts[0]
                    last_p = pts[-1]
                    dx = last_p.bbox.center_x - first_p.bbox.center_x
                    dy = last_p.bbox.center_y - first_p.bbox.center_y
                    dt = max(0.001, last_p.timestamp_seconds - first_p.timestamp_seconds)
                    summaries.append(
                        {
                            "track_id": str(trk.track_id),
                            "class_name": trk.class_name,
                            "point_count": len(pts),
                            "net_displacement_px": round((dx**2 + dy**2) ** 0.5, 2),
                            "mean_velocity_px_s": round(((dx**2 + dy**2) ** 0.5) / dt, 2),
                        }
                    )
        return summaries

    def _gather_ocr_context(
        self,
        candidate: CandidateEvent,
        ocr_result: OCRPipelineResult | None,
        supporting_events: list[Event],
    ) -> list[dict[str, Any]]:
        """Extract text strings recognized within or near the candidate interval."""
        results: list[dict[str, Any]] = []

        if ocr_result is not None:
            for obs in ocr_result.fused_observations:
                # Intersect intervals
                if (
                    obs.first_seen_timestamp_seconds <= candidate.end_timestamp
                    and obs.last_seen_timestamp_seconds >= candidate.start_timestamp
                ):
                    results.append(
                        {
                            "observation_id": str(obs.observation_id),
                            "text": obs.text,
                            "confidence": round(obs.mean_confidence, 3),
                            "start_ts": obs.first_seen_timestamp_seconds,
                            "end_ts": obs.last_seen_timestamp_seconds,
                        }
                    )

        # Also collect text from event attributes if not in ocr_result
        for ev in supporting_events:
            if "text" in ev.attributes and not any(
                r["text"] == ev.attributes["text"] for r in results
            ):
                results.append(
                    {
                        "observation_id": str(ev.event_id),
                        "text": str(ev.attributes["text"]),
                        "confidence": ev.confidence,
                        "start_ts": ev.start_timestamp_seconds,
                        "end_ts": ev.end_timestamp,
                    }
                )

        return results

    def _gather_transcript_context(
        self,
        candidate: CandidateEvent,
        audio_result: AudioPipelineResult | None,
        supporting_events: list[Event],
    ) -> list[dict[str, Any]]:
        """Extract spoken speech segments active during the candidate interval."""
        transcripts: list[dict[str, Any]] = []

        if audio_result is not None:
            active_segments = audio_result.fused_segments or audio_result.raw_segments
            for seg in active_segments:
                if (
                    seg.start_timestamp_seconds <= candidate.end_timestamp
                    and seg.end_timestamp_seconds >= candidate.start_timestamp
                ):
                    transcripts.append(
                        {
                            "segment_id": str(seg.segment_id),
                            "text": seg.raw_text,
                            "language": seg.language,
                            "confidence": seg.confidence,
                            "start_ts": seg.start_timestamp_seconds,
                            "end_ts": seg.end_timestamp_seconds,
                        }
                    )

        # Also collect speech from event attributes
        for ev in supporting_events:
            speech_txt = ev.attributes.get("speech_text") or ev.attributes.get("raw_text")
            if speech_txt and not any(t["text"] == speech_txt for t in transcripts):
                transcripts.append(
                    {
                        "segment_id": str(ev.event_id),
                        "text": str(speech_txt),
                        "language": ev.attributes.get("language", "en"),
                        "confidence": ev.confidence,
                        "start_ts": ev.start_timestamp_seconds,
                        "end_ts": ev.end_timestamp,
                    }
                )

        return transcripts

    def _gather_spatial_context(self, supporting_events: list[Event]) -> list[str]:
        """Extract unique zone names active in supporting events."""
        zones: set[str] = set()
        for ev in supporting_events:
            if ev.zone_name:
                zones.add(ev.zone_name)
        return sorted(zones)

    @staticmethod
    def _is_uuid(val: str) -> bool:
        try:
            UUID(val)
            return True
        except (ValueError, AttributeError):
            return False
