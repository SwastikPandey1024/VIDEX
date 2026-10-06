"""Deterministic question classifier and investigation planner for VIDEX."""

from __future__ import annotations

import re
from typing import Any

from videx.agent.schemas import InvestigationPlan, InvestigationRequest, ToolCall
from videx.agent.types import QuestionCategory, ToolName


class InvestigationPlanner:
    """Classifies user questions and formulates structured, deterministic investigation plans."""

    _TEMPORAL_REGEX = re.compile(
        r"(?:between|from)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?",
        re.IGNORECASE,
    )
    _TRACK_REGEX = re.compile(
        r"\btrack(?:_id)?\b\s*[:#]?\s*([a-zA-Z0-9_\-]+)\b",
        re.IGNORECASE,
    )
    _EVENT_REGEX = re.compile(
        r"\bevent(?:_id)?\b\s*[:#]?\s*([a-zA-Z0-9_\-]+)\b",
        re.IGNORECASE,
    )

    def classify_question(self, question: str) -> QuestionCategory:
        """Classifies a user query into a known question category."""
        q = question.lower()

        # Specific modals / keyword checks
        if any(w in q for w in ("ocr", "text", "license plate", "sign", "words appeared")):
            return QuestionCategory.OCR
        if any(w in q for w in ("transcript", "said", "speech", "spoken", "audio", "hear")):
            return QuestionCategory.AUDIO
        if any(
            w in q
            for w in (
                "interact",
                "interaction",
                "why",
                "intent",
                "suspicious",
                "fighting",
                "stealing",
                "weapon",
            )
        ):
            return QuestionCategory.SEMANTIC
        if self._TRACK_REGEX.search(q) or "track " in q:
            return QuestionCategory.TRACK
        if self._TEMPORAL_REGEX.search(q) or "seconds" in q or "timeline" in q:
            return QuestionCategory.TEMPORAL
        if any(w in q for w in ("spatial", "near", "next to", "overlap", "zone", "vicinity")):
            return QuestionCategory.SPATIAL
        if any(
            w in q for w in ("related", "neighbor", "follow", "precede", "before event", "after")
        ):
            return QuestionCategory.RELATIONAL
        if "event" in q or self._EVENT_REGEX.search(q):
            return QuestionCategory.EVENT
        if any(w in q for w in ("car", "vehicle", "person", "dog", "object")):
            return QuestionCategory.OBJECT

        return QuestionCategory.GENERAL

    def create_plan(self, request: InvestigationRequest) -> InvestigationPlan:
        """Constructs an InvestigationPlan for the given user request."""
        category = self.classify_question(request.question)
        required_tools: list[str] = []
        planned_steps: list[str] = []

        if category == QuestionCategory.TEMPORAL:
            required_tools = [ToolName.SEARCH_EVENTS.value, ToolName.QUERY_GRAPH.value]
            planned_steps = [
                f"Query events within specified temporal window for video {request.video_id}",
                "Retrieve supporting frame/track evidence for bounded interval",
            ]
        elif category == QuestionCategory.TRACK:
            required_tools = [
                ToolName.GET_TRACK.value,
                ToolName.SEARCH_EVENTS.value,
                ToolName.QUERY_GRAPH.value,
            ]
            planned_steps = [
                "Retrieve track trajectory and bounding boxes",
                "Find events involving target track",
                "Trace graph associations for the track",
            ]
        elif category == QuestionCategory.OCR:
            required_tools = [ToolName.GET_OCR.value]
            planned_steps = [
                f"Retrieve canonical OCR text observations for video {request.video_id}"
            ]
        elif category == QuestionCategory.AUDIO:
            required_tools = [ToolName.GET_TRANSCRIPT.value]
            planned_steps = [
                f"Retrieve canonical speech transcripts and audio for video {request.video_id}"
            ]
        elif category == QuestionCategory.SPATIAL:
            required_tools = [ToolName.FIND_SPATIAL_NEIGHBORS.value, ToolName.QUERY_GRAPH.value]
            planned_steps = [
                "Identify focal nodes or bounding boxes",
                "Query graph spatial neighbors sharing IoU overlap or zone containment",
            ]
        elif category == QuestionCategory.RELATIONAL:
            required_tools = [
                ToolName.FIND_RELATED_EVENTS.value,
                ToolName.FIND_TEMPORAL_NEIGHBORS.value,
            ]
            planned_steps = [
                "Find related events and temporal neighbors in the evidence graph",
                "Trace evidence provenance chains",
            ]
        elif category == QuestionCategory.SEMANTIC:
            required_tools = [
                ToolName.SEARCH_EVENTS.value,
                ToolName.GET_EVIDENCE.value,
            ]
            if request.allow_semantic_reasoning:
                required_tools.append(ToolName.REASON_SEMANTIC.value)
            planned_steps = [
                "Gather deterministic events and evidence first",
                "Check deterministic evidence sufficiency",
                "If inconclusive and permitted, invoke bounded VLM semantic reasoning",
            ]
        elif category == QuestionCategory.EVENT:
            required_tools = [ToolName.GET_EVENT.value, ToolName.GET_EVIDENCE.value]
            planned_steps = [
                "Lookup target event details and participants",
                "Retrieve canonical evidence supporting target event",
            ]
        else:
            required_tools = [ToolName.SEARCH_EVENTS.value, ToolName.GET_SCENE.value]
            planned_steps = [
                "Perform broad event search across video",
                "Inspect canonical scene boundaries and context",
            ]

        return InvestigationPlan(
            category=category,
            objective=f"Investigate question: '{request.question}'",
            planned_steps=planned_steps,
            required_tools=required_tools,
            video_id=request.video_id,
        )

    def generate_initial_tool_calls(
        self, request: InvestigationRequest, plan: InvestigationPlan
    ) -> list[ToolCall]:
        """Generates initial typed tool calls derived from the question and plan."""
        q = request.question
        calls: list[ToolCall] = []

        if plan.category == QuestionCategory.TEMPORAL:
            m = self._TEMPORAL_REGEX.search(q)
            t_start: float | None = float(m.group(1)) if m else None
            t_end: float | None = float(m.group(2)) if m else None
            params: dict[str, Any] = {"video_id": request.video_id}
            if t_start is not None:
                params["time_start"] = t_start
            if t_end is not None:
                params["time_end"] = t_end
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.SEARCH_EVENTS.value,
                    arguments=params,
                )
            )

        elif plan.category == QuestionCategory.TRACK:
            m = self._TRACK_REGEX.search(q)
            track_id = m.group(1) if m else "1"
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.GET_TRACK.value,
                    arguments={"track_id": track_id, "video_id": request.video_id},
                )
            )
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.SEARCH_EVENTS.value,
                    arguments={"track_id": track_id, "video_id": request.video_id},
                )
            )

        elif plan.category == QuestionCategory.OCR:
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.GET_OCR.value,
                    arguments={"video_id": request.video_id},
                )
            )

        elif plan.category == QuestionCategory.AUDIO:
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.GET_TRANSCRIPT.value,
                    arguments={"video_id": request.video_id},
                )
            )

        elif plan.category == QuestionCategory.EVENT:
            m = self._EVENT_REGEX.search(q)
            event_id = m.group(1) if m else ""
            if event_id:
                calls.append(
                    ToolCall.create(
                        tool_name=ToolName.GET_EVENT.value,
                        arguments={"event_id": event_id, "video_id": request.video_id},
                    )
                )
                calls.append(
                    ToolCall.create(
                        tool_name=ToolName.GET_EVIDENCE.value,
                        arguments={"event_id": event_id, "video_id": request.video_id},
                    )
                )
            else:
                calls.append(
                    ToolCall.create(
                        tool_name=ToolName.SEARCH_EVENTS.value,
                        arguments={"video_id": request.video_id},
                    )
                )

        elif plan.category == QuestionCategory.RELATIONAL:
            m = self._EVENT_REGEX.search(q)
            ev_id = m.group(1) if m else "ev-1"
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.FIND_RELATED_EVENTS.value,
                    arguments={"event_id": ev_id},
                )
            )
            calls.append(
                ToolCall.create(
                    tool_name=ToolName.FIND_TEMPORAL_NEIGHBORS.value,
                    arguments={"event_id": ev_id, "window_seconds": 5.0},
                )
            )

        elif plan.category == QuestionCategory.SPATIAL:
            # Query graph for tracks or events first to locate spatial neighbors
            calls.append(
                ToolCall.create(
                    ToolName.QUERY_GRAPH.value,
                    arguments={"video_id": request.video_id, "node_types": ["track", "event"]},
                )
            )

        elif plan.category == QuestionCategory.SEMANTIC:
            # First retrieve deterministic events
            calls.append(
                ToolCall.create(
                    ToolName.SEARCH_EVENTS.value,
                    arguments={"video_id": request.video_id},
                )
            )
            # If permitted, also schedule semantic reasoning candidate
            if request.allow_semantic_reasoning:
                calls.append(
                    ToolCall.create(
                        ToolName.REASON_SEMANTIC.value,
                        arguments={"query": request.question},
                    )
                )

        else:
            calls.append(
                ToolCall.create(
                    ToolName.SEARCH_EVENTS.value,
                    arguments={"video_id": request.video_id},
                )
            )

        return calls
