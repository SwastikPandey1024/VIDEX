"""Deterministic OCR text event detector.

Consumes temporally fused TextObservation records to emit TEXT_APPEARED,
TEXT_DISAPPEARED, and TEXT_CHANGED events without re-running OCR.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from videx.domain.schemas import BoundingBox, EvidenceType, TextObservation
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
    EventParticipant,
)
from videx.events.types import EventSeverity, EventStatus, EventType


def _calculate_iou(b1: BoundingBox | None, b2: BoundingBox | None) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    if b1 is None or b2 is None:
        return 0.0

    x_left = max(b1.x, b2.x)
    y_top = max(b1.y, b2.y)
    x_right = min(b1.x + b1.width, b2.x + b2.width)
    y_bottom = min(b1.y + b1.height, b2.y + b2.height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = b1.width * b1.height
    area2 = b2.width * b2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0.0 else 0.0


class OCREventDetector:
    """Detects text appearance, disappearance, and text change events from fused observations."""

    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()

    def detect_events(
        self,
        fused_observations: Sequence[TextObservation],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Process fused OCR observations and emit structured text events.

        Args:
            fused_observations: Sequence of temporally fused TextObservation records.
            video_id: Video UUID override.

        Returns:
            List of evidence-backed OCR text Event records.
        """
        events: list[Event] = []
        obs_list = sorted(fused_observations, key=lambda o: o.first_seen_timestamp_seconds)

        for obs in obs_list:
            v_id = video_id or obs.video_id
            participant = EventParticipant(
                participant_id=obs.observation_id,
                participant_type="text",
                role="subject",
                label=obs.normalized_text,
                metadata={
                    "raw_text": obs.text,
                    "language": obs.language,
                    "first_frame": obs.first_seen_frame,
                    "last_frame": obs.last_seen_frame,
                },
            )

            # ── 1. TEXT_APPEARED ──────────────────────────────────────────
            appear_ev = EventEvidence(
                timestamp_seconds=obs.first_seen_timestamp_seconds,
                evidence_type=EvidenceType.OCR,
                role="appearance",
                ocr_observation_id=obs.observation_id,
                provenance={
                    "frame_number": obs.first_seen_frame,
                    "text": obs.text,
                    "normalized_text": obs.normalized_text,
                    "provider": obs.provider,
                },
            )

            ev_appear = Event(
                video_id=v_id,
                event_type=EventType.TEXT_APPEARED,
                start_timestamp_seconds=obs.first_seen_timestamp_seconds,
                end_timestamp_seconds=obs.first_seen_timestamp_seconds,
                confidence=obs.mean_confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[appear_ev],
                source_module="ocr_event_detector",
                description=(
                    f"Text '{obs.text}' appeared in scene at frame {obs.first_seen_frame} "
                    f"({obs.first_seen_timestamp_seconds:.2f}s)"
                ),
                attributes={
                    "text": obs.text,
                    "normalized_text": obs.normalized_text,
                    "language": obs.language,
                    "frame_number": obs.first_seen_frame,
                    "bbox": obs.latest_bbox.model_dump() if obs.latest_bbox else None,
                },
            )
            events.append(ev_appear)

            # ── 2. TEXT_DISAPPEARED ───────────────────────────────────────
            disappear_ev = EventEvidence(
                timestamp_seconds=obs.last_seen_timestamp_seconds,
                evidence_type=EvidenceType.OCR,
                role="disappearance",
                ocr_observation_id=obs.observation_id,
                provenance={
                    "frame_number": obs.last_seen_frame,
                    "text": obs.text,
                    "normalized_text": obs.normalized_text,
                },
            )

            ev_disappear = Event(
                video_id=v_id,
                event_type=EventType.TEXT_DISAPPEARED,
                start_timestamp_seconds=obs.last_seen_timestamp_seconds,
                end_timestamp_seconds=obs.last_seen_timestamp_seconds,
                confidence=obs.mean_confidence,
                status=EventStatus.CONFIRMED,
                severity=EventSeverity.INFO,
                participants=[participant],
                event_evidence=[disappear_ev],
                source_module="ocr_event_detector",
                description=(
                    f"Text '{obs.text}' ceased to be visible at frame {obs.last_seen_frame} "
                    f"({obs.last_seen_timestamp_seconds:.2f}s)"
                ),
                attributes={
                    "text": obs.text,
                    "normalized_text": obs.normalized_text,
                    "frame_number": obs.last_seen_frame,
                    "bbox": obs.latest_bbox.model_dump() if obs.latest_bbox else None,
                },
            )
            events.append(ev_disappear)

        # ── 3. TEXT_CHANGED ───────────────────────────────────────────────
        # Pairwise comparison: if two observations occupy overlapping spatial bounding box
        # in successive/nearby time and text differs, emit TEXT_CHANGED.
        for i in range(len(obs_list)):
            obs1 = obs_list[i]
            for j in range(i + 1, len(obs_list)):
                obs2 = obs_list[j]

                # Temporal check: obs2 should begin near or right after obs1 ends
                time_gap = obs2.first_seen_timestamp_seconds - obs1.last_seen_timestamp_seconds
                if time_gap > 3.0:  # Beyond reasonable temporal continuity for same region
                    continue

                # Spatial check: bounding boxes overlap significantly
                iou = _calculate_iou(obs1.latest_bbox, obs2.latest_bbox)
                if iou >= self.config.ocr_spatial_iou_threshold:
                    if obs1.normalized_text != obs2.normalized_text:
                        change_ts = obs2.first_seen_timestamp_seconds
                        v_id = video_id or obs1.video_id

                        ev1 = EventEvidence(
                            timestamp_seconds=obs1.last_seen_timestamp_seconds,
                            evidence_type=EvidenceType.OCR,
                            role="pre_change",
                            ocr_observation_id=obs1.observation_id,
                            provenance={"text": obs1.text, "frame": obs1.last_seen_frame},
                        )
                        ev2 = EventEvidence(
                            timestamp_seconds=obs2.first_seen_timestamp_seconds,
                            evidence_type=EvidenceType.OCR,
                            role="post_change",
                            ocr_observation_id=obs2.observation_id,
                            provenance={"text": obs2.text, "frame": obs2.first_seen_frame},
                        )

                        part1 = EventParticipant(
                            participant_id=obs1.observation_id,
                            participant_type="text",
                            role="source",
                            label=obs1.normalized_text,
                        )
                        part2 = EventParticipant(
                            participant_id=obs2.observation_id,
                            participant_type="text",
                            role="target",
                            label=obs2.normalized_text,
                        )

                        ev_changed = Event(
                            video_id=v_id,
                            event_type=EventType.TEXT_CHANGED,
                            start_timestamp_seconds=change_ts,
                            end_timestamp_seconds=change_ts,
                            confidence=min(obs1.mean_confidence, obs2.mean_confidence),
                            status=EventStatus.CONFIRMED,
                            severity=EventSeverity.MEDIUM,
                            participants=[part1, part2],
                            event_evidence=[ev1, ev2],
                            source_module="ocr_event_detector",
                            description=(
                                f"Text changed from '{obs1.text}' to '{obs2.text}' in same region "
                                f"(IoU {iou:.2f}) at {change_ts:.2f}s"
                            ),
                            attributes={
                                "previous_text": obs1.text,
                                "previous_normalized_text": obs1.normalized_text,
                                "new_text": obs2.text,
                                "new_normalized_text": obs2.normalized_text,
                                "spatial_iou": round(iou, 2),
                                "change_frame": obs2.first_seen_frame,
                            },
                        )
                        events.append(ev_changed)

        return events
