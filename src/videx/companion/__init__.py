"""VIDEX YouTube Learning Companion.

Transforms educational YouTube videos and playlists into evidence-grounded,
timestamp-linked study notes for technical learning and competitive exams (GATE).
"""

from videx.companion.connector import (
    YouTubeConnector,
    YouTubeConnectorError,
    YouTubeQuotaExceededError,
    YouTubeRateLimitError,
    YouTubeURLParser,
)
from videx.companion.evaluation import (
    CompanionEvaluationResult,
    evaluate_study_note,
)
from videx.companion.markdown import MarkdownExporter
from videx.companion.orchestrator import PlaylistJobOrchestrator
from videx.companion.schemas import (
    DefinitionItem,
    EvidenceReference,
    EvidenceSourceType,
    ExamTrapItem,
    FormulaItem,
    NoteGenerationStatus,
    NoteItemType,
    NoteValidationResult,
    PlaylistAccessStatus,
    PlaylistProcessingManifest,
    PracticeQuestionItem,
    StudyNote,
    StudyNoteSection,
    TakeawayItem,
    TheoremItem,
    TranscriptAcquisitionMethod,
    TranscriptDocument,
    TranscriptSegment,
    UncertaintyItem,
    UncertaintyType,
    VideoAccessStatus,
    VideoProcessingRecord,
    WorkedExample,
    YouTubePlaylistMetadata,
    YouTubeVideoMetadata,
)
from videx.companion.synthesizer import StudyNoteSynthesizer
from videx.companion.transcript import (
    BaseTranscriptProvider,
    CompositeTranscriptProvider,
    FileTranscriptProvider,
    VIDEXASRTranscriptProvider,
    YouTubeCaptionProvider,
)

__all__ = [
    "BaseTranscriptProvider",
    "CompanionEvaluationResult",
    "CompositeTranscriptProvider",
    "DefinitionItem",
    "EvidenceReference",
    "EvidenceSourceType",
    "ExamTrapItem",
    "FileTranscriptProvider",
    "FormulaItem",
    "MarkdownExporter",
    "NoteGenerationStatus",
    "NoteItemType",
    "NoteValidationResult",
    "PlaylistAccessStatus",
    "PlaylistJobOrchestrator",
    "PlaylistProcessingManifest",
    "PracticeQuestionItem",
    "StudyNote",
    "StudyNoteSection",
    "StudyNoteSynthesizer",
    "TakeawayItem",
    "TheoremItem",
    "TranscriptAcquisitionMethod",
    "TranscriptDocument",
    "TranscriptSegment",
    "UncertaintyItem",
    "UncertaintyType",
    "VIDEXASRTranscriptProvider",
    "VideoAccessStatus",
    "VideoProcessingRecord",
    "WorkedExample",
    "YouTubeCaptionProvider",
    "YouTubeConnector",
    "YouTubeConnectorError",
    "YouTubePlaylistMetadata",
    "YouTubeQuotaExceededError",
    "YouTubeRateLimitError",
    "YouTubeURLParser",
    "YouTubeVideoMetadata",
    "evaluate_study_note",
]
