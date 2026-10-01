"""VIDEX event taxonomy, status, and severity enumerations."""

from enum import StrEnum


class EventType(StrEnum):
    """Taxonomy of detectable video, perception, audio, and multimodal events."""

    # ── Object Lifecycle Events ──────────────────────────────────────────
    OBJECT_APPEARED = "object_appeared"
    """A tracked object was first reliably observed in the video."""

    OBJECT_DISAPPEARED = "object_disappeared"
    """A tracked object ended or left the camera view."""

    OBJECT_PRESENT = "object_present"
    """An interval event spanning the duration a tracked object was visible."""

    # ── Movement & Kinematic Events ──────────────────────────────────────
    OBJECT_STARTED_MOVING = "object_started_moving"
    """A tracked object transitioned from stationary to moving in pixel space."""

    OBJECT_STOPPED_MOVING = "object_stopped_moving"
    """A tracked object transitioned from moving to stationary in pixel space."""

    OBJECT_CHANGED_DIRECTION = "object_changed_direction"
    """A tracked object altered its displacement trajectory beyond angular threshold."""

    OBJECT_MOVING = "object_moving"
    """Interval event denoting continuous motion of a tracked object."""

    OBJECT_STATIONARY = "object_stationary"
    """Interval event denoting a stationary dwelling object."""

    # ── Spatial & Zone Events ────────────────────────────────────────────
    OBJECT_ENTERED_ZONE = "object_entered_zone"
    """A tracked object centroid or bounding box crossed into a spatial zone."""

    OBJECT_EXITED_ZONE = "object_exited_zone"
    """A tracked object centroid or bounding box crossed out of a spatial zone."""

    OBJECT_LEFT_ZONE = "object_left_zone"
    """Alias for OBJECT_EXITED_ZONE preserved for backward compatibility."""

    OBJECT_IN_ZONE = "object_in_zone"
    """Interval event denoting an object residing within a spatial zone."""

    LOITERING = "loitering"
    """An object dwelled within a spatial zone longer than configured threshold."""

    CROWD_FORMATION = "crowd_formation"
    """Multiple tracked objects converged in spatial proximity."""

    # ── OCR & Text Events ────────────────────────────────────────────────
    TEXT_APPEARED = "text_appeared"
    """A distinct text observation or license plate was first observed."""

    TEXT_DISAPPEARED = "text_disappeared"
    """A text observation ceased to be visible on screen."""

    TEXT_CHANGED = "text_changed"
    """Text observed in the same spatial region underwent content modification."""

    # ── Audio & Speech Events ────────────────────────────────────────────
    SPEECH_STARTED = "speech_started"
    """A speech utterance began at an authoritative audio timestamp."""

    SPEECH_ENDED = "speech_ended"
    """A speech utterance concluded at an authoritative audio timestamp."""

    SPEECH_DETECTED = "speech_detected"
    """Interval event spanning a transcribed speech segment."""

    SOUND_EVENT_DETECTED = "sound_event_detected"
    """Acoustic sound event detected by a SoundEventProvider (future phase hook)."""

    # ── General / Custom ────────────────────────────────────────────────
    ANOMALY = "anomaly"
    """A statistically or rule-defined unusual pattern was observed."""

    CUSTOM = "custom"
    """User-defined or plugin-defined event type."""


class EventStatus(StrEnum):
    """Lifecycle and confirmation state of an Event."""

    DETECTED = "detected"
    """Event pattern matched initial detector trigger criteria."""

    CONFIRMED = "confirmed"
    """Event has satisfied minimum confirmation thresholds (e.g. frame count)."""

    IN_PROGRESS = "in_progress"
    """Interval event currently actively unfolding across video frames."""

    ENDED = "ended"
    """Interval event has concluded."""

    REJECTED = "rejected"
    """Preliminary event invalidated by subsequent evidence (e.g. tracking glitch)."""


class EventSeverity(StrEnum):
    """Operational severity or importance level of an Event."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TemporalRelation(StrEnum):
    """Qualitative temporal relationship between two events (Allen's interval algebra subset)."""

    BEFORE = "before"
    """Event A strictly precedes Event B in time (A.end < B.start)."""

    AFTER = "after"
    """Event A strictly follows Event B in time (A.start > B.end)."""

    OVERLAPS = "overlaps"
    """Event A and Event B share a non-empty temporal intersection."""

    CONTAINS = "contains"
    """Event A temporally encompasses Event B (A.start <= B.start and A.end >= B.end)."""

    DURING = "during"
    """Event A is temporally encompassed by Event B (B.start <= A.start and B.end >= A.end)."""

    NEAR_IN_TIME = "near_in_time"
    """Event A and Event B occur within a maximum configured temporal distance."""

    EQUALS = "equals"
    """Event A and Event B have identical start and end timestamps."""
