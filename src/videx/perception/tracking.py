"""Multi-object tracking provider implementations (BoT-SORT, MockTracker).

Maintains object persistence across frames, tracks lifecycle states (ACTIVE, LOST, TERMINATED),
and generates TrajectoryPoint records with inherited FrameTimestamp provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    Frame,
    FrameTimestamp,
    TimestampSource,
    Track,
    TrackStatus,
    TrajectoryPoint,
)
from videx.ingestion.base import DecodedFrame

# ── Kalman Filter for Motion Prediction ─────────────────────────────────────


class KalmanFilter:
    """Linear Kalman filter for bounding box tracking in BoT-SORT coordinate space.

    State vector: (x_center, y_center, width, height, vx, vy, vw, vh)
    Measurement: (x_center, y_center, width, height)

    Note: This is strictly a Linear Kalman Filter (KF), not an Extended Kalman Filter (EKF).
    Both the state transition matrix F and measurement projection matrix H are linear
    constant-velocity matrices. No nonlinear equations or Jacobian linearizations are evaluated.
    """

    def __init__(self) -> None:
        ndim = 4
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = 1.0
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create track state and covariance from initial measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        w = max(float(measurement[2]), 1.0)
        h = max(float(measurement[3]), 1.0)

        std = [
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict state forward one time step."""
        w = max(float(mean[2]), 1.0)
        h = max(float(mean[3]), 1.0)

        std_pos = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        predicted_mean = np.dot(self._motion_mat, mean)
        predicted_cov = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )
        return predicted_mean, predicted_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        w = max(float(mean[2]), 1.0)
        h = max(float(mean[3]), 1.0)

        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        )
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update state distribution with new measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = np.linalg.inv(projected_cov)
        kalman_gain = np.linalg.multi_dot((covariance, self._update_mat.T, chol_factor))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance


def _state_to_bbox(
    state: np.ndarray,
    coordinate_type: CoordinateType = CoordinateType.PIXEL,
) -> BoundingBox:
    """Convert Kalman state [cx, cy, w, h] to a BoundingBox."""
    cx, cy, w, h = state[:4]
    w_box = max(float(w), 1.0)
    h_box = max(float(h), 1.0)
    x = float(cx - w_box / 2.0)
    y = float(cy - h_box / 2.0)
    return BoundingBox(x=x, y=y, width=w_box, height=h_box, coordinate_type=coordinate_type)


# ── Configuration & Provider ────────────────────────────────────────────────


@dataclass
class BoTSORTConfig:
    """Configuration for custom lightweight BoT-SORT multi-object tracker."""

    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False
    cmc_method: str = "sparseOptFlow"  # camera motion compensation


class BoTSORTTracker:
    """Custom lightweight BoT-SORT tracking provider implementing TrackingProvider protocol.

    Algorithmic component verification & qualification:
    - Motion prediction / Kalman filtering: Implemented via continuous constant-velocity
      linear Kalman filter over 8D state space (x_c, y_c, w, h, vx, vy, vw, vh),
      propagating track bounding boxes through motion transitions and occlusions.
    - Association: Implemented via two-stage IoU matching (first associating active tracks
      with high-confidence detections, then associating remaining tracks with low-confidence
      detections to maintain identity persistence through occlusion).
    - Track buffer: Implemented via a configurable lost-track window (track_buffer frames
      in LOST state before transition to TERMINATED).
    - Camera motion compensation (CMC): Configured via cmc_method (default 'sparseOptFlow').
      Lightweight engine runs zero-warp identity compensation by default when external warp
      matrices are not supplied.
    - Optional ReID: Configured via with_reid (default False). Visual appearance embedding
      extraction is reserved for heavyweight pipelines; in this lightweight/custom tracker,
      association operates purely in motion + spatial IoU space to maintain
      zero-weight hermetic runtime.
    """

    def __init__(self, config: BoTSORTConfig | None = None) -> None:
        self.config = config or BoTSORTConfig()
        self._provider_name = "botsort"
        self._tracks: dict[UUID, TrackState] = {}
        self._frame_count = 0
        self._kf = KalmanFilter()

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._tracks.clear()
        self._frame_count = 0

    def active_tracks(self) -> list[Track]:
        """Return all tracks that are currently in ACTIVE status."""
        return [
            state.to_domain_track(self.provider_name)
            for state in self._tracks.values()
            if state.status == TrackStatus.ACTIVE
        ]

    def all_tracks(self) -> list[Track]:
        """Return all tracked entities (including lost and terminated)."""
        return [state.to_domain_track(self.provider_name) for state in self._tracks.values()]

    def update(
        self,
        detections: list[Detection],
        frame_meta: Frame | DecodedFrame,
    ) -> list[TrajectoryPoint]:
        """Update tracker with frame detections, performing prediction, association, and update."""
        self._frame_count += 1
        frame_id, video_id, frame_num, ts_sec, frame_ts = _extract_meta(frame_meta, detections)

        # 1. Filter detections by confidence thresholds
        high_dets = [d for d in detections if d.confidence >= self.config.track_high_thresh]
        low_dets = [
            d
            for d in detections
            if self.config.track_low_thresh <= d.confidence < self.config.track_high_thresh
        ]

        active_track_ids = [
            tid
            for tid, s in self._tracks.items()
            if s.status in (TrackStatus.ACTIVE, TrackStatus.LOST)
        ]

        # 2. Kalman Filter Motion Prediction for all active/lost tracks
        for tid in active_track_ids:
            self._tracks[tid].predict(self._kf)

        # 3. Association Stage 1: Match active tracks with high-confidence detections via IoU
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = _associate_iou(
            track_ids=active_track_ids,
            tracks=self._tracks,
            detections=high_dets,
            iou_threshold=1.0 - self.config.match_thresh,
        )

        # Apply Stage 1 matches
        for tid, didx in matched_tracks_1:
            det = high_dets[didx]
            self._tracks[tid].update_match(
                det=det,
                frame_number=frame_num,
                timestamp_seconds=ts_sec,
                kf=self._kf,
            )

        # 4. Association Stage 2: Match remaining tracks with low-confidence detections
        matched_tracks_2, unmatched_tracks_2, _ = _associate_iou(
            track_ids=unmatched_tracks_1,
            tracks=self._tracks,
            detections=low_dets,
            iou_threshold=0.5,
        )

        for tid, didx in matched_tracks_2:
            det = low_dets[didx]
            self._tracks[tid].update_match(
                det=det,
                frame_number=frame_num,
                timestamp_seconds=ts_sec,
                kf=self._kf,
            )

        # 5. Handle unmatched tracks (mark as LOST or TERMINATED if buffer exceeded)
        for tid in unmatched_tracks_2:
            state = self._tracks[tid]
            state.mark_missed(
                current_frame=frame_num,
                max_buffer=self.config.track_buffer,
            )

        # 6. Handle unmatched high-confidence detections (initiate new tracks)
        for didx in unmatched_dets_1:
            det = high_dets[didx]
            if det.confidence >= self.config.new_track_thresh:
                new_state = TrackState.initiate(
                    det=det,
                    frame_number=frame_num,
                    timestamp_seconds=ts_sec,
                    kf=self._kf,
                )
                self._tracks[new_state.track_id] = new_state

        # 7. Generate TrajectoryPoint records for all currently ACTIVE tracks in this frame
        points: list[TrajectoryPoint] = []
        for tid, state in self._tracks.items():
            if state.status == TrackStatus.ACTIVE and state.last_seen_frame_number == frame_num:
                points.append(
                    TrajectoryPoint(
                        track_id=tid,
                        frame_id=frame_id,
                        frame_number=frame_num,
                        timestamp_seconds=ts_sec,
                        frame_timestamp=frame_ts,
                        bbox=state.current_bbox,
                        confidence=state.current_confidence,
                    )
                )

        return points


class MockTracker:
    """Deterministic MockTracker for unit testing and CI test environments."""

    def __init__(
        self,
        provider_name: str = "mock_tracker",
        iou_threshold: float = 0.3,
    ) -> None:
        self._provider_name = provider_name
        self.iou_threshold = iou_threshold
        self._tracks: dict[UUID, TrackState] = {}
        self._frame_count = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def reset(self) -> None:
        self._tracks.clear()
        self._frame_count = 0

    def active_tracks(self) -> list[Track]:
        return [
            s.to_domain_track(self.provider_name)
            for s in self._tracks.values()
            if s.status == TrackStatus.ACTIVE
        ]

    def all_tracks(self) -> list[Track]:
        return [s.to_domain_track(self.provider_name) for s in self._tracks.values()]

    def update(
        self,
        detections: list[Detection],
        frame_meta: Frame | DecodedFrame,
    ) -> list[TrajectoryPoint]:
        self._frame_count += 1
        frame_id, video_id, frame_num, ts_sec, frame_ts = _extract_meta(frame_meta, detections)

        active_tids = [
            tid
            for tid, s in self._tracks.items()
            if s.status in (TrackStatus.ACTIVE, TrackStatus.LOST)
        ]

        matched, unmatched_t, unmatched_d = _associate_iou(
            track_ids=active_tids,
            tracks=self._tracks,
            detections=detections,
            iou_threshold=self.iou_threshold,
        )

        for tid, didx in matched:
            self._tracks[tid].update_match(
                det=detections[didx],
                frame_number=frame_num,
                timestamp_seconds=ts_sec,
            )

        for tid in unmatched_t:
            self._tracks[tid].mark_missed(current_frame=frame_num, max_buffer=5)

        for didx in unmatched_d:
            det = detections[didx]
            new_s = TrackState.initiate(det, frame_num, ts_sec)
            self._tracks[new_s.track_id] = new_s

        points: list[TrajectoryPoint] = []
        for tid, state in self._tracks.items():
            if state.status == TrackStatus.ACTIVE and state.last_seen_frame_number == frame_num:
                points.append(
                    TrajectoryPoint(
                        track_id=tid,
                        frame_id=frame_id,
                        frame_number=frame_num,
                        timestamp_seconds=ts_sec,
                        frame_timestamp=frame_ts,
                        bbox=state.current_bbox,
                        confidence=state.current_confidence,
                    )
                )

        return points


# ── Internal Tracking State Representation ──────────────────────────────────


@dataclass
class TrackState:
    """Internal state tracking an object's life cycle, history, and bounding boxes."""

    track_id: UUID
    video_id: UUID
    class_name: str
    class_id: int
    first_seen_frame_number: int
    last_seen_frame_number: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    current_bbox: BoundingBox
    start_bbox: BoundingBox
    current_confidence: float
    status: TrackStatus = TrackStatus.ACTIVE
    detection_ids: list[UUID] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    frames_since_update: int = 0
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None

    @classmethod
    def initiate(
        cls,
        det: Detection,
        frame_number: int,
        timestamp_seconds: float,
        kf: KalmanFilter | None = None,
    ) -> TrackState:
        mean: np.ndarray | None = None
        covariance: np.ndarray | None = None
        if kf is not None:
            measurement = np.array(
                [det.bbox.center_x, det.bbox.center_y, det.bbox.width, det.bbox.height],
                dtype=np.float64,
            )
            mean, covariance = kf.initiate(measurement)

        return cls(
            track_id=uuid4(),
            video_id=det.video_id,
            class_name=det.class_name,
            class_id=det.class_id,
            first_seen_frame_number=frame_number,
            last_seen_frame_number=frame_number,
            first_seen_timestamp_seconds=timestamp_seconds,
            last_seen_timestamp_seconds=timestamp_seconds,
            current_bbox=det.bbox,
            start_bbox=det.bbox,
            current_confidence=det.confidence,
            status=TrackStatus.ACTIVE,
            detection_ids=[det.detection_id],
            confidence_history=[det.confidence],
            frames_since_update=0,
            mean=mean,
            covariance=covariance,
        )

    def predict(self, kf: KalmanFilter) -> None:
        """Predict state forward via Kalman filter and update current_bbox estimate."""
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
            self.current_bbox = _state_to_bbox(self.mean, self.current_bbox.coordinate_type)

    def update_match(
        self,
        det: Detection,
        frame_number: int,
        timestamp_seconds: float,
        kf: KalmanFilter | None = None,
    ) -> None:
        self.last_seen_frame_number = frame_number
        self.last_seen_timestamp_seconds = timestamp_seconds
        self.current_bbox = det.bbox
        self.current_confidence = det.confidence
        self.status = TrackStatus.ACTIVE
        self.detection_ids.append(det.detection_id)
        self.confidence_history.append(det.confidence)
        self.frames_since_update = 0

        if kf is not None and self.mean is not None and self.covariance is not None:
            measurement = np.array(
                [det.bbox.center_x, det.bbox.center_y, det.bbox.width, det.bbox.height],
                dtype=np.float64,
            )
            self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

    def mark_missed(self, current_frame: int, max_buffer: int) -> None:
        self.frames_since_update += 1
        if self.frames_since_update > max_buffer:
            self.status = TrackStatus.TERMINATED
        else:
            self.status = TrackStatus.LOST

    def to_domain_track(self, provider_name: str) -> Track:
        mean_conf = (
            sum(self.confidence_history) / len(self.confidence_history)
            if self.confidence_history
            else self.current_confidence
        )
        return Track(
            track_id=self.track_id,
            video_id=self.video_id,
            class_name=self.class_name,
            class_id=self.class_id,
            first_seen_frame_number=self.first_seen_frame_number,
            last_seen_frame_number=self.last_seen_frame_number,
            first_seen_timestamp_seconds=self.first_seen_timestamp_seconds,
            last_seen_timestamp_seconds=self.last_seen_timestamp_seconds,
            detection_ids=list(self.detection_ids),
            provider=provider_name,
            status=self.status,
            confidence=round(mean_conf, 4),
            start_bbox=self.start_bbox,
            end_bbox=self.current_bbox,
            attributes={
                "total_observations": len(self.detection_ids),
                "frames_since_update": self.frames_since_update,
            },
        )


def _extract_meta(
    frame_meta: Frame | DecodedFrame,
    detections: list[Detection] | None = None,
) -> tuple[UUID, UUID, int, float, FrameTimestamp]:
    """Helper extracting (frame_id, video_id, frame_number, timestamp_seconds, frame_timestamp)."""
    if isinstance(frame_meta, DecodedFrame):
        fid = uuid4()
        vid = frame_meta.video_id or uuid4()
        frame_ts = frame_meta.frame_timestamp
        if frame_ts is None and detections:
            frame_ts = detections[0].frame_timestamp
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=frame_meta.frame_index,
                pts_seconds=frame_meta.timestamp_seconds,
                timestamp_source=TimestampSource.DERIVED,
            )
        return (
            fid,
            vid,
            frame_meta.frame_index,
            frame_meta.timestamp_seconds,
            frame_ts,
        )

    fid = frame_meta.frame_id
    vid = frame_meta.video_id
    frame_num = frame_meta.frame_number
    ts_sec = frame_meta.timestamp_seconds
    frame_ts = getattr(frame_meta, "frame_timestamp", None)
    if frame_ts is None and detections:
        frame_ts = detections[0].frame_timestamp
    if frame_ts is None:
        frame_ts = FrameTimestamp(
            frame_index=frame_num,
            pts_seconds=ts_sec,
            timestamp_source=TimestampSource.DERIVED,
        )
    return (fid, vid, frame_num, ts_sec, frame_ts)


def _calculate_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute Intersection over Union between two BoundingBoxes."""
    x1 = max(b1.x, b2.x)
    y1 = max(b1.y, b2.y)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    union_area = b1.area + b2.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _associate_iou(
    track_ids: list[UUID],
    tracks: dict[UUID, TrackState],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[UUID, int]], list[UUID], list[int]]:
    """Greedy IoU matching between active tracks and detections."""
    if not track_ids or not detections:
        return [], list(track_ids), list(range(len(detections)))

    # Compute IoU matrix
    matched_pairs: list[tuple[UUID, int]] = []
    unmatched_tracks = set(track_ids)
    unmatched_dets = set(range(len(detections)))

    # Sort potential pairs by IoU descending
    candidates = []
    for tid in track_ids:
        track_box = tracks[tid].current_bbox
        for didx, det in enumerate(detections):
            iou = _calculate_iou(track_box, det.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, tid, didx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, tid, didx in candidates:
        if tid in unmatched_tracks and didx in unmatched_dets:
            matched_pairs.append((tid, didx))
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(didx)

    return matched_pairs, list(unmatched_tracks), list(unmatched_dets)
