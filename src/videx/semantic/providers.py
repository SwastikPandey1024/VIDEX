"""VLM provider interface and evidence-grounded prompt contract."""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from videx.semantic.schemas import EvidenceBundle, RoutingRequest, SemanticEventPayload


def build_evidence_grounded_prompt(
    bundle: EvidenceBundle,
    query: str | None = None,
) -> str:
    """Construct an evidence-grounded prompt adhering to the strict epistemic contract.

    Enforces that the model only reasons from provided evidence, never fabricates IDs
    or timestamps, and explicitly abstains when context is insufficient.
    """
    lines: list[str] = [
        "You are an expert, objective video intelligence analyst operating inside VIDEX.",
        "Your task is to analyze the supplied multimodal evidence bundle and produce a "
        "verifiable, structured semantic interpretation.",
        "",
        "=================================================================",
        "=== STRICT EPISTEMIC RULES & GROUNDING CONSTRAINTS ===",
        "=================================================================",
        "1. GROUNDING MANDATE: Reason ONLY from the evidence provided below. "
        "Do NOT invent, assume, or hallucinate entities, actions, or objects.",
        "2. PROVENANCE INTEGRITY: You MUST cite only the exact evidence_ids provided in the "
        "evidence bundle. Never invent or hallucinate evidence UUIDs.",
        "3. TEMPORAL BOUNDS: Your event start and end timestamps MUST fall within the provided "
        "interval [start_timestamp_seconds, end_timestamp_seconds].",
        "4. ABSTENTION RULE: If the visual or audio evidence is insufficient, blurry, ambiguous, "
        "or lacks definitive indicators, set status to 'insufficient_evidence' or 'uncertain'.",
        "5. SEPARATION: Clearly separate OBSERVED EVIDENCE from INTERPRETATION in description.",
        "",
        "=================================================================",
        "=== SUPPLIED OBSERVED EVIDENCE ===",
        "=================================================================",
        f"Video ID: {bundle.video_id}",
        (
            f"Candidate Time Window: {bundle.start_timestamp_seconds:.3f}s - "
            f"{bundle.end_timestamp_seconds:.3f}s"
        ),
        (
            f"Spatial Zones Active: "
            f"{', '.join(bundle.spatial_context) if bundle.spatial_context else 'None'}"
        ),
        f"Supplied Canonical Evidence IDs: {[str(eid) for eid in bundle.evidence_ids]}",
        f"Supporting Temporal Event IDs: {[str(evid) for evid in bundle.supporting_event_ids]}",
        f"Keyframe Count: {len(bundle.keyframes)}",
        f"Visual Crops Count: {len(bundle.crops)}",
    ]

    # Include OCR observations
    if bundle.ocr_observations:
        lines.append("")
        lines.append("--- Observed On-Screen Text (OCR) ---")
        for obs in bundle.ocr_observations:
            txt = obs.get("text", "")
            st = obs.get("start_ts", 0.0)
            conf = obs.get("confidence", 0.0)
            lines.append(f"  • Text: '{txt}' at {st:.2f}s (conf: {conf:.2f})")

    # Include Audio transcripts
    if bundle.transcript_segments:
        lines.append("")
        lines.append("--- Observed Spoken Audio (ASR) ---")
        for seg in bundle.transcript_segments:
            stxt = seg.get("text", "")
            lang = seg.get("language", "en")
            st = seg.get("start_ts", 0.0)
            et = seg.get("end_ts", 0.0)
            lines.append(f"  • Speech: '{stxt}' [{lang}] at {st:.2f}s - {et:.2f}s")

    # Include Trajectory summaries
    if bundle.trajectory_summary:
        lines.append("")
        lines.append("--- Observed Entity Kinematics ---")
        for traj in bundle.trajectory_summary:
            cname = traj.get("class_name")
            tid = traj.get("track_id")
            disp = traj.get("net_displacement_px")
            vel = traj.get("mean_velocity_px_s")
            lines.append(f"  • Entity {cname} (Track {tid}): Disp={disp}px, Vel={vel}px/s")

    if query:
        lines.append("")
        lines.append("=================================================================")
        lines.append(f"=== OPERATIONAL INQUIRY / USER QUERY: {query} ===")
        lines.append("=================================================================")

    lines.append("")
    lines.append("=================================================================")
    lines.append("=== REQUIRED OUTPUT FORMAT ===")
    lines.append("=================================================================")
    lines.append("Respond ONLY with valid JSON conforming to the following structure:")
    sample_json = {
        "semantic_event_type": "interaction | vehicle_activity | security_incident | custom",
        "claim": "Concise factual summary statement",
        "description": "Evidence-grounded description distinguishing observed facts from reasoning",
        "confidence": 0.85,
        "uncertainty": 0.15,
        "status": "supported | uncertain | insufficient_evidence | rejected",
        "start_timestamp_seconds": bundle.start_timestamp_seconds,
        "end_timestamp_seconds": bundle.end_timestamp_seconds,
        "evidence_ids": [str(eid) for eid in bundle.evidence_ids],
        "supporting_event_ids": [str(evid) for evid in bundle.supporting_event_ids],
        "attributes": {"primary_action": "...", "intent": "..."},
    }
    lines.append(json.dumps(sample_json, indent=2))
    return "\n".join(lines)


@runtime_checkable
class VLMProvider(Protocol):
    """Abstract vision-language model provider protocol for semantic reasoning.

    Accepts structured EvidenceBundles and returns validated SemanticEventPayloads.
    Decoupled from specific backends (e.g. Qwen3-VL, local checkpoints, or cloud APIs).
    """

    @property
    def provider_name(self) -> str:
        """Unique string identifier for this VLM provider."""
        ...

    def warmup(self) -> None:
        """Warm up model weights, tokenizers, or accelerator runtime."""
        ...

    def is_available(self) -> bool:
        """Check if provider weights/runtime are initialized and available for inference."""
        ...

    def analyze(
        self,
        bundle: EvidenceBundle,
        request: RoutingRequest,
    ) -> SemanticEventPayload:
        """Execute evidence-grounded semantic reasoning over an EvidenceBundle.

        Args:
            bundle: Self-contained evidence context with keyframes, crops, and audio/OCR.
            request: The parent routing request containing query and configuration.

        Returns:
            Validated SemanticEventPayload citing only provided evidence IDs.
        """
        ...
