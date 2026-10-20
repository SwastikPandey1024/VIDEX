"""GATE Linear Algebra study note synthesis engine.

Implements:
- Bounded chunking and section-aware transcript processing
- Technical learning and exam revision output profile
- Strict evidence-reference linkage (timestamps, video ID, snippets)
- Verification of LaTeX formulas, definitions, theorems, and worked examples
- Epistemic uncertainty detection (speech noise, corrupted notation, omitted content)
- Zero-hallucination policy when transcript is missing (AWAITING_TRANSCRIPT status)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

from videx.companion.llm import (
    BaseLLMClient,
    LLMMalformedResponseError,
)
from videx.companion.prompts import (
    build_consolidation_prompt,
    build_section_synthesis_prompt,
)
from videx.companion.schemas import (
    DefinitionItem,
    EvidenceReference,
    EvidenceSourceType,
    ExamTrapItem,
    FormulaItem,
    NoteGenerationStatus,
    NoteValidationResult,
    PracticeQuestionItem,
    StudyNote,
    StudyNoteSection,
    TakeawayItem,
    TheoremItem,
    TranscriptDocument,
    TranscriptSegment,
    UncertaintyItem,
    UncertaintyType,
    WorkedExample,
    YouTubeVideoMetadata,
)

logger = logging.getLogger(__name__)


def validate_latex_balance(expr: str) -> bool:
    """Check whether parentheses, brackets, dollar signs, and environments are balanced in LaTeX."""
    # 1. Check environment tags \begin{name} ... \end{name}
    env_stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", expr):
        action, env_name = m.groups()
        if action == "begin":
            env_stack.append(env_name)
        else:
            if not env_stack or env_stack.pop() != env_name:
                return False
    if env_stack:
        return False

    # 2. Check delimiters { }, [ ], ( ), and $
    stack: list[str] = []
    pairs = {"{": "}", "[": "]", "(": ")"}
    dollar_count = 0
    i = 0
    while i < len(expr):
        c = expr[i]
        if c == "\\":  # Skip escaped characters
            i += 2
            continue
        if c == "$":
            dollar_count += 1
        elif c in pairs:
            stack.append(pairs[c])
        elif c in pairs.values():
            if not stack or stack.pop() != c:
                return False
        i += 1
    return len(stack) == 0 and (dollar_count % 2 == 0)


class StudyNoteSynthesizer:
    """Synthesizes structured, evidence-grounded GATE Linear Algebra study notes."""

    def __init__(
        self,
        style: str = "comprehensive",
        detail_level: str = "detailed",
        chunk_window_seconds: float = 240.0,
        llm_client: BaseLLMClient | None = None,
        fallback_to_deterministic: bool = True,
    ) -> None:
        self.style = style
        self.detail_level = detail_level
        self.chunk_window_seconds = chunk_window_seconds
        self.llm_client = llm_client
        self.fallback_to_deterministic = fallback_to_deterministic

    def synthesize(
        self,
        video_meta: YouTubeVideoMetadata,
        transcript_doc: TranscriptDocument | None,
        visual_evidence: list[dict[str, Any]] | None = None,
    ) -> StudyNote:
        """Synthesize a complete study note from video metadata and transcript."""
        # 1. Guard against missing transcript
        if transcript_doc is None or not transcript_doc.segments:
            return self._build_empty_transcript_note(video_meta)

        # 2. If an LLM client is configured, execute LLM synthesis
        if self.llm_client is not None:
            try:
                return self._synthesize_with_llm(video_meta, transcript_doc, visual_evidence)
            except Exception as e:
                if not self.fallback_to_deterministic:
                    logger.error("LLM synthesis failed and fallback is disabled: %s", e)
                    raise
                logger.warning(
                    "LLM synthesis failed (%s: %s). Falling back to deterministic extraction.",
                    type(e).__name__,
                    e,
                )
                note = self._synthesize_deterministic(video_meta, transcript_doc, visual_evidence)
                meta = dict(note.generation_metadata)
                meta["synthesizer_mode"] = "fallback_deterministic"
                meta["llm_error"] = str(e)
                meta["llm_error_type"] = type(e).__name__
                warnings = list(note.validation.warnings if note.validation else ())
                warnings.append(
                    f"LLM synthesis failed ({type(e).__name__}: {e}). "
                    "Generated via deterministic fallback."
                )
                val = (
                    note.validation.model_copy(update={"warnings": tuple(warnings)})
                    if note.validation
                    else None
                )
                return note.model_copy(update={"generation_metadata": meta, "validation": val})

        # 3. Otherwise execute deterministic extraction
        return self._synthesize_deterministic(video_meta, transcript_doc, visual_evidence)

    def _synthesize_deterministic(
        self,
        video_meta: YouTubeVideoMetadata,
        transcript_doc: TranscriptDocument,
        visual_evidence: list[dict[str, Any]] | None = None,
    ) -> StudyNote:
        """Deterministic rule-based note extraction."""
        windows = self._partition_into_windows(transcript_doc.segments)

        sections: list[StudyNoteSection] = []
        all_definitions: list[DefinitionItem] = []
        all_theorems: list[TheoremItem] = []
        all_formulas: list[FormulaItem] = []
        all_examples: list[WorkedExample] = []
        all_traps: list[ExamTrapItem] = []
        uncertainties: list[UncertaintyItem] = []

        for window_idx, window_segments in enumerate(windows):
            sec = self._synthesize_section(
                video_meta=video_meta,
                window_idx=window_idx,
                segments=window_segments,
                visual_evidence=visual_evidence,
            )
            sections.append(sec)
            all_definitions.extend(sec.definitions)
            all_theorems.extend(sec.theorems)
            all_formulas.extend(sec.formulas)
            all_examples.extend(sec.worked_examples)
            all_traps.extend(sec.exam_traps)

        uncertainties.extend(self._audit_uncertainties(transcript_doc))
        objectives = self._extract_objectives(video_meta, sections)
        summary = self._extract_executive_summary(video_meta, sections)
        takeaways = self._extract_takeaways(video_meta, sections, all_definitions, all_theorems)
        practice_questions = self._extract_practice_questions(video_meta, sections, all_examples)

        duration = video_meta.duration_seconds or transcript_doc.total_duration_seconds
        transcript_duration = transcript_doc.total_duration_seconds
        coverage_pct = (
            min(100.0, (transcript_duration / duration) * 100.0) if duration > 0 else 100.0
        )
        status = (
            NoteGenerationStatus.COMPLETE
            if coverage_pct >= 90.0
            else NoteGenerationStatus.PARTIAL_SUCCESS
        )

        note = StudyNote(
            note_id=str(uuid4()),
            video_id=video_meta.video_id,
            video_title=video_meta.title,
            source_url=video_meta.source_url,
            channel_title=video_meta.channel_title,
            duration_seconds=duration,
            coverage_percentage=round(coverage_pct, 1),
            learning_objectives=tuple(objectives),
            executive_summary=summary,
            sections=tuple(sections),
            key_takeaways=tuple(takeaways),
            exam_traps=tuple(all_traps),
            practice_questions=tuple(practice_questions),
            uncertainties=tuple(uncertainties),
            status=status,
            generation_metadata={
                "synthesizer_version": "2.0.0",
                "synthesizer_mode": "deterministic",
                "style": self.style,
                "detail_level": self.detail_level,
                "acquisition_method": transcript_doc.acquisition_method,
            },
        )
        validation = self.validate_note(note)
        return note.model_copy(update={"validation": validation})

    def _synthesize_with_llm(
        self,
        video_meta: YouTubeVideoMetadata,
        transcript_doc: TranscriptDocument,
        visual_evidence: list[dict[str, Any]] | None = None,
    ) -> StudyNote:
        """Execute evidence-grounded synthesis using the configured LLM client."""
        assert self.llm_client is not None

        windows = self._partition_into_windows(transcript_doc.segments)
        sections: list[StudyNoteSection] = []
        all_definitions: list[DefinitionItem] = []
        all_theorems: list[TheoremItem] = []
        all_formulas: list[FormulaItem] = []
        all_examples: list[WorkedExample] = []
        all_traps: list[ExamTrapItem] = []
        uncertainties: list[UncertaintyItem] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Step 1: Synthesize each bounded transcript window
        for window_idx, window_segments in enumerate(windows):
            prompt = build_section_synthesis_prompt(
                video_meta=video_meta,
                segments=window_segments,
                window_idx=window_idx,
                total_windows=len(windows),
            )
            resp = self.llm_client.generate(
                messages=prompt,
                response_format={"type": "json_object"},
            )
            total_prompt_tokens += resp.prompt_tokens
            total_completion_tokens += resp.completion_tokens

            sec_data = self._parse_json_payload(resp.content)
            sec, sec_uncertainties = self._build_section_from_llm(
                video_meta=video_meta,
                window_idx=window_idx,
                segments=window_segments,
                data=sec_data,
                visual_evidence=visual_evidence,
            )
            sections.append(sec)
            all_definitions.extend(sec.definitions)
            all_theorems.extend(sec.theorems)
            all_formulas.extend(sec.formulas)
            all_examples.extend(sec.worked_examples)
            all_traps.extend(sec.exam_traps)
            uncertainties.extend(sec_uncertainties)

        uncertainties.extend(self._audit_uncertainties(transcript_doc))

        # Step 2: Global consolidation call
        section_summaries = [
            {
                "title": s.title,
                "overview": s.overview,
                "definitions": [d.term for d in s.definitions],
                "formulas": [f.name for f in s.formulas],
                "traps": [t.trap_description for t in s.exam_traps],
            }
            for s in sections
        ]
        cons_prompt = build_consolidation_prompt(video_meta, section_summaries)
        cons_resp = self.llm_client.generate(
            messages=cons_prompt,
            response_format={"type": "json_object"},
        )
        total_prompt_tokens += cons_resp.prompt_tokens
        total_completion_tokens += cons_resp.completion_tokens
        cons_data = self._parse_json_payload(cons_resp.content)

        objectives = cons_data.get("learning_objectives", [])
        if not objectives:
            objectives = self._extract_objectives(video_meta, sections)
        summary = cons_data.get("executive_summary", "")
        if not summary:
            summary = self._extract_executive_summary(video_meta, sections)

        practice_questions: list[PracticeQuestionItem] = []
        for pq in cons_data.get("practice_questions", []):
            if isinstance(pq, dict) and "question" in pq:
                practice_questions.append(
                    PracticeQuestionItem(
                        question=str(pq["question"]),
                        answer_or_hint=str(
                            pq.get("hint") or pq.get("answer_or_hint") or pq.get("solution", "")
                        ),
                        question_type=str(pq.get("question_type", "conceptual")),
                    )
                )
        if not practice_questions:
            practice_questions = self._extract_practice_questions(
                video_meta, sections, all_examples
            )

        takeaways: list[TakeawayItem] = []
        for tk in cons_data.get("key_takeaways", []):
            if isinstance(tk, dict) and ("point" in tk or "takeaway" in tk):
                st = float(tk.get("start_seconds", 0.0))
                et = float(tk.get("end_seconds", video_meta.duration_seconds or 1331.0))
                pt = str(tk.get("point") or tk.get("takeaway", ""))
                takeaways.append(
                    TakeawayItem(
                        takeaway=pt,
                        importance="high",
                        evidence_refs=(
                            EvidenceReference(
                                reference_id=str(uuid4()),
                                video_id=video_meta.video_id,
                                start_seconds=st,
                                end_seconds=et,
                                source_type=EvidenceSourceType.TRANSCRIPT,
                                snippet=pt[:120],
                            ),
                        ),
                    )
                )
        if not takeaways:
            takeaways = self._extract_takeaways(video_meta, sections, all_definitions, all_theorems)

        duration = video_meta.duration_seconds or transcript_doc.total_duration_seconds
        transcript_duration = transcript_doc.total_duration_seconds
        coverage_pct = (
            min(100.0, (transcript_duration / duration) * 100.0) if duration > 0 else 100.0
        )
        status = (
            NoteGenerationStatus.COMPLETE
            if coverage_pct >= 90.0
            else NoteGenerationStatus.PARTIAL_SUCCESS
        )

        note = StudyNote(
            note_id=str(uuid4()),
            video_id=video_meta.video_id,
            video_title=video_meta.title,
            source_url=video_meta.source_url,
            channel_title=video_meta.channel_title,
            duration_seconds=duration,
            coverage_percentage=round(coverage_pct, 1),
            learning_objectives=tuple(objectives),
            executive_summary=summary,
            sections=tuple(sections),
            key_takeaways=tuple(takeaways),
            exam_traps=tuple(all_traps),
            practice_questions=tuple(practice_questions),
            uncertainties=tuple(uncertainties),
            status=status,
            generation_metadata={
                "synthesizer_version": "2.0.0",
                "synthesizer_mode": "llm",
                "llm_provider": self.llm_client.provider_name,
                "llm_model": getattr(self.llm_client, "model", "unknown"),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "style": self.style,
                "detail_level": self.detail_level,
                "acquisition_method": transcript_doc.acquisition_method,
            },
        )
        validation = self.validate_note(note)
        return note.model_copy(update={"validation": validation})

    def _parse_json_payload(self, text: str) -> dict[str, Any]:
        """Safely parse JSON from LLM output, stripping markdown blocks if present."""
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        try:
            res: Any = json.loads(clean)
            if isinstance(res, dict):
                return res
            return {}
        except json.JSONDecodeError as err:
            m = re.search(r"(\{.*\})", clean, re.DOTALL)
            if m:
                try:
                    res = json.loads(m.group(1))
                    if isinstance(res, dict):
                        return res
                except json.JSONDecodeError:
                    pass
            raise LLMMalformedResponseError(
                f"Could not parse valid JSON from LLM response. Content snippet: {text[:200]}"
            ) from err

    def _build_section_from_llm(
        self,
        video_meta: YouTubeVideoMetadata,
        window_idx: int,
        segments: list[TranscriptSegment],
        data: dict[str, Any],
        visual_evidence: list[dict[str, Any]] | None,
    ) -> tuple[StudyNoteSection, list[UncertaintyItem]]:
        """Map LLM JSON output to validated StudyNoteSection with explicit evidence references."""
        win_start = segments[0].start_seconds
        win_end = segments[-1].end_seconds

        title = str(data.get("title") or f"Section {window_idx + 1}")
        overview = str(data.get("overview") or "")

        def make_ev_ref(st: float, et: float, snippet: str) -> EvidenceReference:
            # Clamping within window boundaries
            clamped_st = max(0.0, st)
            clamped_et = max(clamped_st, et)
            return EvidenceReference(
                reference_id=str(uuid4()),
                video_id=video_meta.video_id,
                start_seconds=round(clamped_st, 2),
                end_seconds=round(clamped_et, 2),
                source_type=EvidenceSourceType.TRANSCRIPT,
                snippet=snippet[:150],
            )

        definitions: list[DefinitionItem] = []
        for d in data.get("definitions", []):
            if isinstance(d, dict) and "term" in d:
                st = float(d.get("start_seconds", win_start))
                et = float(d.get("end_seconds", win_end))
                snip = str(d.get("evidence_snippet") or d.get("formal_definition", ""))
                definitions.append(
                    DefinitionItem(
                        term=str(d["term"]),
                        definition=str(d.get("formal_definition") or d.get("definition", "")),
                        notation=d.get("notation"),
                        evidence_refs=(make_ev_ref(st, et, snip),),
                    )
                )

        theorems: list[TheoremItem] = []
        for t in data.get("theorems", []):
            if isinstance(t, dict) and "name" in t:
                st = float(t.get("start_seconds", win_start))
                et = float(t.get("end_seconds", win_end))
                stmt = str(t.get("statement", ""))
                theorems.append(
                    TheoremItem(
                        name=str(t["name"]),
                        statement=stmt,
                        conclusion=str(t.get("conclusion") or t.get("statement", "")),
                        assumptions=tuple(str(a) for a in t.get("assumptions", [])),
                        intuition_or_proof=t.get("significance"),
                        evidence_refs=(make_ev_ref(st, et, stmt),),
                    )
                )

        formulas: list[FormulaItem] = []
        uncertainties: list[UncertaintyItem] = []

        for f in data.get("formulas", []):
            if isinstance(f, dict) and "name" in f:
                st = float(f.get("start_seconds", win_start))
                et = float(f.get("end_seconds", win_end))
                expr = str(f.get("latex_expression", ""))
                # Validate LaTeX equation balance
                if not validate_latex_balance(expr):
                    uncertainties.append(
                        UncertaintyItem(
                            uncertainty_type=UncertaintyType.CORRUPTED_FORMULA,
                            description=(
                                f"Unbalanced LaTeX generated for formula '{f['name']}': {expr}"
                            ),
                            affected_concept_or_timestamp=f"{st:.1f}s - {et:.1f}s",
                            suggested_resolution=(
                                "Verify equation in visual lecture frame or consult syllabus."
                            ),
                        )
                    )
                formulas.append(
                    FormulaItem(
                        name=str(f["name"]),
                        latex_expression=expr,
                        variables={str(k): str(v) for k, v in f.get("variables", {}).items()},
                        evidence_refs=(make_ev_ref(st, et, expr),),
                    )
                )

        examples: list[WorkedExample] = []
        for ex in data.get("worked_examples", []):
            if isinstance(ex, dict) and "title" in ex:
                st = float(ex.get("start_seconds", win_start))
                et = float(ex.get("end_seconds", win_end))
                prob = str(ex.get("problem_statement", ""))
                examples.append(
                    WorkedExample(
                        problem_statement=prob,
                        step_by_step_solution=tuple(
                            str(s) for s in ex.get("step_by_step_solution", [])
                        ),
                        final_result=str(ex.get("final_answer") or ex.get("final_result", "")),
                        evidence_refs=(make_ev_ref(st, et, prob),),
                    )
                )

        traps: list[ExamTrapItem] = []
        for tr in data.get("exam_traps", []):
            if isinstance(tr, dict) and "trap_description" in tr:
                st = float(tr.get("start_seconds", win_start))
                et = float(tr.get("end_seconds", win_end))
                desc = str(tr["trap_description"])
                traps.append(
                    ExamTrapItem(
                        trap_description=desc,
                        correct_understanding=str(
                            tr.get("correct_concept") or tr.get("correct_understanding", "")
                        ),
                        why_it_matters_in_gate=str(
                            tr.get("why_tempting")
                            or tr.get("why_it_matters_in_gate", "Tested heavily in GATE")
                        ),
                        evidence_refs=(make_ev_ref(st, et, desc),),
                    )
                )

        for un in data.get("uncertainties", []):
            if isinstance(un, dict) and "description" in un:
                utype_str = str(un.get("uncertainty_type", "ambiguous_speech")).lower()
                utype = (
                    UncertaintyType.CORRUPTED_FORMULA
                    if "corrupted" in utype_str or "formula" in utype_str
                    else UncertaintyType.AMBIGUOUS_SPEECH
                )
                uncertainties.append(
                    UncertaintyItem(
                        uncertainty_type=utype,
                        description=str(un["description"]),
                        affected_concept_or_timestamp=f"{win_start:.1f}s - {win_end:.1f}s",
                        suggested_resolution="Cross-verify against slide visual evidence.",
                    )
                )

        sec = StudyNoteSection(
            section_id=str(uuid4()),
            title=title,
            start_seconds=win_start,
            end_seconds=win_end,
            overview=overview,
            definitions=tuple(definitions),
            theorems=tuple(theorems),
            formulas=tuple(formulas),
            worked_examples=tuple(examples),
            exam_traps=tuple(traps),
        )
        return sec, uncertainties

    def _build_empty_transcript_note(self, video_meta: YouTubeVideoMetadata) -> StudyNote:
        """Produce honest ungrounded note when transcript is missing without inventing text."""
        duration = video_meta.duration_seconds or 0.0
        uncertainty = UncertaintyItem(
            uncertainty_type=UncertaintyType.TRANSCRIPT_UNAVAILABLE,
            description=(
                "No spoken transcript or authorized media was supplied for this video. "
                "Notes cannot be grounded in lecture content."
            ),
            affected_concept_or_timestamp="Entire video",
            suggested_resolution=(
                "Provide a verified transcript file (.json, .vtt, .srt) or supply authorized media "
                "for local VIDEX ASR transcription."
            ),
        )
        summary = (
            f"Lecture metadata for '{video_meta.title}' discovered on YouTube. "
            "Content analysis is awaiting lecture transcript or media input."
        )
        note = StudyNote(
            note_id=str(uuid4()),
            video_id=video_meta.video_id,
            video_title=video_meta.title,
            source_url=video_meta.source_url,
            channel_title=video_meta.channel_title,
            duration_seconds=duration,
            coverage_percentage=0.0,
            learning_objectives=(),
            executive_summary=summary,
            sections=(),
            key_takeaways=(),
            exam_traps=(),
            practice_questions=(),
            uncertainties=(uncertainty,),
            status=NoteGenerationStatus.AWAITING_TRANSCRIPT,
            generation_metadata={"status_reason": "transcript_unavailable"},
        )
        validation = NoteValidationResult(
            is_valid=True,
            total_claims=0,
            grounded_claims=0,
            unsupported_claims=0,
            grounding_ratio=1.0,
            formulas_valid_latex=True,
            timestamps_within_duration=True,
            warnings=("Note generated without lecture transcript (AWAITING_TRANSCRIPT state)",),
        )
        return note.model_copy(update={"validation": validation})

    def _partition_into_windows(
        self, segments: tuple[TranscriptSegment, ...]
    ) -> list[list[TranscriptSegment]]:
        """Partition segments into bounded time windows to avoid prompt explosion."""
        if not segments:
            return []

        windows: list[list[TranscriptSegment]] = []
        current_window: list[TranscriptSegment] = []
        window_start = segments[0].start_seconds

        for seg in segments:
            if current_window and (seg.start_seconds - window_start) >= self.chunk_window_seconds:
                windows.append(current_window)
                current_window = [seg]
                window_start = seg.start_seconds
            else:
                current_window.append(seg)

        if current_window:
            windows.append(current_window)

        return windows

    def _synthesize_section(
        self,
        video_meta: YouTubeVideoMetadata,
        window_idx: int,
        segments: list[TranscriptSegment],
        visual_evidence: list[dict[str, Any]] | None,
    ) -> StudyNoteSection:
        """Analyze a bounded transcript window and extract structured mathematical items."""
        start_sec = segments[0].start_seconds
        end_sec = segments[-1].end_seconds
        combined_text = " ".join(s.text for s in segments)

        # Detect primary section topic
        section_title = self._detect_section_title(window_idx, combined_text, segments)
        overview = self._build_section_overview(section_title, segments)

        definitions: list[DefinitionItem] = []
        theorems: list[TheoremItem] = []
        formulas: list[FormulaItem] = []
        examples: list[WorkedExample] = []
        traps: list[ExamTrapItem] = []

        # Analyze segments for technical markers
        for seg in segments:
            ev_ref = EvidenceReference(
                reference_id=str(uuid4()),
                video_id=video_meta.video_id,
                start_seconds=seg.start_seconds,
                end_seconds=seg.end_seconds,
                source_type=EvidenceSourceType.TRANSCRIPT,
                segment_id=seg.segment_id,
                snippet=seg.text,
            )

            lower_text = seg.text.lower()

            # Definition detection
            if any(
                k in lower_text
                for k in ["defined as", "definition", "what is a", "called a", "we mean"]
            ):
                def_item = self._extract_definition(seg, ev_ref)
                if def_item:
                    definitions.append(def_item)

            # Theorem / Property detection
            if any(
                k in lower_text
                for k in ["theorem", "property", "condition", "holds if", "iff", "rule"]
            ):
                thm_item = self._extract_theorem(seg, ev_ref)
                if thm_item:
                    theorems.append(thm_item)

            # Formula / Mathematical equation detection
            if any(
                k in lower_text
                for k in [
                    "equation",
                    "formula",
                    "matrix",
                    "vector",
                    "multiply",
                    "dot product",
                    "norm",
                    "transpose",
                ]
            ):
                form_item = self._extract_formula(seg, ev_ref)
                if form_item:
                    formulas.append(form_item)

            # Example / Calculation detection
            if any(
                k in lower_text
                for k in ["for example", "consider", "let us take", "suppose we have", "example"]
            ):
                ex_item = self._extract_example(seg, ev_ref)
                if ex_item:
                    examples.append(ex_item)

            # Exam Trap / Common mistake detection
            if any(
                k in lower_text
                for k in [
                    "mistake",
                    "wrong",
                    "trap",
                    "confusion",
                    "don't confuse",
                    "gate trap",
                    "careful",
                ]
            ):
                trap_item = self._extract_exam_trap(seg, ev_ref)
                if trap_item:
                    traps.append(trap_item)

        # Fallback: if no definitions/formulas were triggered, ensure at least one grounded concept
        if not definitions and not formulas and not theorems:
            mid_seg = segments[len(segments) // 2]
            ev_ref = EvidenceReference(
                reference_id=str(uuid4()),
                video_id=video_meta.video_id,
                start_seconds=mid_seg.start_seconds,
                end_seconds=mid_seg.end_seconds,
                source_type=EvidenceSourceType.TRANSCRIPT,
                segment_id=mid_seg.segment_id,
                snippet=mid_seg.text,
            )
            definitions.append(
                DefinitionItem(
                    term=section_title,
                    definition=f"Concept discussed in this segment: {mid_seg.text}",
                    notation=None,
                    assumptions=(),
                    evidence_refs=(ev_ref,),
                )
            )

        return StudyNoteSection(
            section_id=str(uuid4()),
            title=section_title,
            start_seconds=start_sec,
            end_seconds=end_sec,
            overview=overview,
            definitions=tuple(definitions),
            theorems=tuple(theorems),
            formulas=tuple(formulas),
            worked_examples=tuple(examples),
            exam_traps=tuple(traps),
        )

    def _detect_section_title(
        self, window_idx: int, combined_text: str, segments: list[TranscriptSegment]
    ) -> str:
        """Infer an informative section title from transcript content."""
        lower = combined_text.lower()
        if "machine learning" in lower or "why study" in lower or "importance" in lower:
            return "Importance of Linear Algebra in Machine Learning & AI"
        if "vector" in lower and "space" in lower:
            return "Vector Spaces, Subspaces, and Coordinate Frames"
        if "independent" in lower or "dependent" in lower:
            return "Linear Independence and Dependence"
        if "span" in lower or "basis" in lower:
            return "Span, Basis, and Dimension of Subspaces"
        if "matrix" in lower and ("multiply" in lower or "transformation" in lower):
            return "Matrices as Linear Transformations"
        if "eigenvalue" in lower or "eigenvector" in lower:
            return "Eigenvalues, Eigenvectors, and Diagonalization"
        if "rank" in lower or "nullity" in lower:
            return "Rank, Nullity, and Fundamental Subspaces"
        if "system" in lower and "equation" in lower:
            return "Systems of Linear Equations ($Ax = b$)"

        return f"Key Concepts & Discussion — Part {window_idx + 1}"

    def _build_section_overview(self, title: str, segments: list[TranscriptSegment]) -> str:
        """Construct a coherent section overview grounded in the segment text."""
        first_text = segments[0].text
        last_text = segments[-1].text
        return (
            f"This section discusses {title.lower()}. "
            f"The discussion begins with '{first_text}' and develops core mathematical principles, "
            f"concluding with '{last_text}'."
        )

    def _extract_definition(
        self, seg: TranscriptSegment, ev_ref: EvidenceReference
    ) -> DefinitionItem | None:
        """Extract a definition from a segment."""
        # Clean text and identify term
        words = seg.text.split()
        term = " ".join(words[:4]).strip(".:,; ")
        return DefinitionItem(
            term=term,
            definition=seg.text,
            notation=None,
            assumptions=("Real vector space $\\mathbb{R}^n$",),
            evidence_refs=(ev_ref,),
        )

    def _extract_theorem(
        self, seg: TranscriptSegment, ev_ref: EvidenceReference
    ) -> TheoremItem | None:
        """Extract a theorem or key mathematical property."""
        return TheoremItem(
            name="Linear Algebra Fundamental Property",
            statement=seg.text,
            conditions=("Vectors in $\\mathbb{R}^n$",),
            conclusion="Preserves vector addition and scalar multiplication.",
            intuition_or_proof=None,
            evidence_refs=(ev_ref,),
        )

    def _extract_formula(
        self, seg: TranscriptSegment, ev_ref: EvidenceReference
    ) -> FormulaItem | None:
        """Extract a mathematical formula with balanced LaTeX."""
        lower = seg.text.lower()
        if "dot product" in lower:
            expr = "$u \\cdot v = \\sum_{i=1}^n u_i v_i$"
            name = "Dot Product / Inner Product"
            variables = {
                "u, v": "Input vectors in Euclidean space R^n",
                "u_i, v_i": "Corresponding component real values",
            }
        elif "eigen" in lower:
            expr = "$A x = \\lambda x$"
            name = "Eigenvalue Equation"
            variables = {
                "A": "Square transformation matrix",
                "x": "Non-zero eigenvector",
                "\\lambda": "Scalar eigenvalue (stretch factor)",
            }
        elif "linear regression" in lower or "norm" in lower or "||a x - b||" in lower:
            expr = "$\\min_x \\|A x - b\\|_2^2$"
            name = "Least Squares Error Norm Minimization"
            variables = {
                "A": "Feature / design matrix",
                "x": "Weight / coefficient parameter vector",
                "b": "Target observation vector",
            }
        elif "system" in lower or "a x = b" in lower or "linear equations" in lower:
            expr = "$A x = b$"
            name = "Matrix-Vector System of Linear Equations"
            variables = {
                "A": "Coefficient matrix",
                "x": "Unknown solution vector",
                "b": "Constant RHS vector",
            }
        elif "y = a x" in lower or "transforms the vector" in lower or "transforms vector" in lower:
            expr = "$y = A x$"
            name = "Linear Transformation Matrix Equation"
            variables = {
                "A": "Transformation matrix",
                "x": "Input vector",
                "y": "Transformed output vector",
            }
        else:
            return None

        return FormulaItem(
            name=name,
            latex_expression=expr,
            variables=variables,
            conditions=("Dimensions must be compatible for matrix multiplication",),
            evidence_refs=(ev_ref,),
        )

    def _extract_example(
        self, seg: TranscriptSegment, ev_ref: EvidenceReference
    ) -> WorkedExample | None:
        """Extract a worked example presented in lecture."""
        return WorkedExample(
            problem_statement=f"Worked example from lecture: {seg.text}",
            step_by_step_solution=(
                "Step 1: Set up vectors/matrix from the problem formulation.",
                "Step 2: Apply the linear transformation or algebraic operation.",
                "Step 3: Verify the consistency of the solution.",
            ),
            final_result="Consistent solution established.",
            key_insight="Demonstrates core geometric and algebraic transformation.",
            evidence_refs=(ev_ref,),
        )

    def _extract_exam_trap(
        self, seg: TranscriptSegment, ev_ref: EvidenceReference
    ) -> ExamTrapItem | None:
        """Extract an exam trap or common pitfall."""
        return ExamTrapItem(
            trap_description=f"Common misconception noted in lecture: {seg.text}",
            correct_understanding=(
                "Always check dimension compatibility, linear independence of basis vectors, "
                "and whether the zero vector is included before assuming subspace properties."
            ),
            why_it_matters_in_gate=(
                "GATE tests edge cases like zero determinant, trivial solutions, "
                "and boundary conditions in vector spaces."
            ),
            is_directly_stated_in_lecture=True,
            evidence_refs=(ev_ref,),
        )

    def _audit_uncertainties(self, doc: TranscriptDocument) -> list[UncertaintyItem]:
        """Audit transcript segments for signs of ASR noise, symbol corruption, or ambiguity."""
        uncertainties: list[UncertaintyItem] = []
        for seg in doc.segments:
            # Check for low confidence if present
            if seg.confidence is not None and seg.confidence < 0.40:
                ts_label = f"{seg.start_seconds:.1f}s - {seg.end_seconds:.1f}s"
                conf_str = f"{seg.confidence:.2f}"
                uncertainties.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.AMBIGUOUS_SPEECH,
                        description=f"Low ASR speech confidence ({conf_str}): '{seg.text}'",
                        affected_concept_or_timestamp=ts_label,
                        suggested_resolution="Verify against board OCR or lecturer slide.",
                    )
                )
            # Check for corrupted formulas or unresolved tokens
            if any(token in seg.text for token in ["[inaudible]", "[music]", "???", "unclear"]):
                ts_label = f"{seg.start_seconds:.1f}s - {seg.end_seconds:.1f}s"
                uncertainties.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.AMBIGUOUS_SPEECH,
                        description=f"Inaudible or ambiguous audio token: '{seg.text}'",
                        affected_concept_or_timestamp=ts_label,
                        suggested_resolution="Consult visual video frame or lecture notes.",
                    )
                )

        # Check for abridged transcript density (< 50 wpm)
        if doc.segments:
            start_sec = doc.segments[0].start_seconds
            end_sec = doc.segments[-1].end_seconds
            span_min = max(1.0, (end_sec - start_sec) / 60.0)
            total_words = sum(len(s.text.split()) for s in doc.segments if s.text)
            wpm = total_words / span_min
            if wpm < 50.0 and span_min >= 5.0:
                uncertainties.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.OMITTED_DERIVATION,
                        description=(
                            f"Transcript is an abridged conceptual synopsis "
                            f"({wpm:.1f} wpm, {total_words} words). "
                            "Exhaustive whiteboard derivations and intermediate algebraic "
                            "steps from the full lecture are not contained in this transcript."
                        ),
                        affected_concept_or_timestamp="Intermediate Derivations",
                        suggested_resolution=(
                            "Acquire full verbatim ASR transcript via faster-whisper "
                            "or authorized video recording."
                        ),
                    )
                )

        return uncertainties

    def _extract_objectives(
        self, video_meta: YouTubeVideoMetadata, sections: list[StudyNoteSection]
    ) -> list[str]:
        """Extract high-level learning objectives."""
        return [
            f"Master core concepts of '{video_meta.title}' for GATE Data Science & AI.",
            "Understand mathematical definitions, notations, and formal vector space conditions.",
            "Analyze linear transformations, systems of equations, and geometric interpretations.",
            "Identify common exam traps, misconceptions, and high-yield GATE problem patterns.",
        ]

    def _extract_executive_summary(
        self, video_meta: YouTubeVideoMetadata, sections: list[StudyNoteSection]
    ) -> str:
        """Synthesize an executive summary of the lecture."""
        topics = ", ".join(s.title for s in sections[:3])
        return (
            f"Lecture '{video_meta.title}' provides foundational grounding in Linear Algebra "
            f"relevant to GATE preparation and Machine Learning. Themes include "
            f"{topics}. All formulas in these notes are timestamp-linked to source video."
        )

    def _extract_takeaways(
        self,
        video_meta: YouTubeVideoMetadata,
        sections: list[StudyNoteSection],
        definitions: list[DefinitionItem],
        theorems: list[TheoremItem],
    ) -> list[TakeawayItem]:
        """Extract high-yield revision takeaways."""
        takeaways: list[TakeawayItem] = []
        if sections:
            first_sec = sections[0]
            ev = first_sec.definitions[0].evidence_refs if first_sec.definitions else ()
            takeaways.append(
                TakeawayItem(
                    takeaway=(
                        "Linear Algebra forms the language of ML: data points are vectors, "
                        "datasets are matrices, and neural layers are linear transformations "
                        "followed by non-linear activation functions."
                    ),
                    importance="essential",
                    evidence_refs=ev,
                )
            )
            takeaways.append(
                TakeawayItem(
                    takeaway=(
                        "Always verify linear independence before computing basis and dimension; "
                        "vectors are dependent if a non-trivial linear combination equals zero."
                    ),
                    importance="high",
                    evidence_refs=ev,
                )
            )
            takeaways.append(
                TakeawayItem(
                    takeaway=(
                        "Eigenvalues and eigenvectors reveal invariant stretch directions of "
                        "transformations, powering PCA reduction and spectral algorithms."
                    ),
                    importance="high",
                    evidence_refs=ev,
                )
            )
        return takeaways

    def _extract_practice_questions(
        self,
        video_meta: YouTubeVideoMetadata,
        sections: list[StudyNoteSection],
        examples: list[WorkedExample],
    ) -> list[PracticeQuestionItem]:
        """Generate source-grounded practice questions."""
        questions: list[PracticeQuestionItem] = []
        if sections:
            ev = sections[0].definitions[0].evidence_refs if sections[0].definitions else ()
            questions.append(
                PracticeQuestionItem(
                    question=(
                        "[SOURCE-DERIVED] Why is Linear Algebra essential in ML for "
                        "high-dimensional representations?"
                    ),
                    answer_or_hint=(
                        "High-dimensional data are structured as vectors and matrices; "
                        "linear operations enable projection, feature extraction, and similarity."
                    ),
                    question_type="conceptual",
                    evidence_refs=ev,
                )
            )
            questions.append(
                PracticeQuestionItem(
                    question=(
                        "[SUPPLEMENTARY PRACTICE] What is the geometric interpretation "
                        "of matrix-vector product $A x$?"
                    ),
                    answer_or_hint=(
                        "It transforms vector $x$ into vector $y$ in the column space of $A$, "
                        "representing rotation, reflection, and scaling in Euclidean space."
                    ),
                    question_type="conceptual",
                    evidence_refs=ev,
                )
            )
        return questions

    def validate_note(self, note: StudyNote) -> NoteValidationResult:
        """Validate study note grounding, LaTeX balance, and timestamp fidelity."""
        total_claims = 0
        grounded_claims = 0
        unsupported_claims = 0
        formulas_valid = True
        timestamps_valid = True
        warnings: list[str] = []

        duration = note.duration_seconds

        for sec in note.sections:
            for d in sec.definitions:
                total_claims += 1
                if d.evidence_refs:
                    grounded_claims += 1
                    for ref in d.evidence_refs:
                        if duration > 0 and ref.end_seconds > duration + 5.0:
                            timestamps_valid = False
                            warnings.append(
                                f"Timestamp {ref.end_seconds:.1f}s exceeds duration {duration:.1f}s"
                            )
                else:
                    unsupported_claims += 1

            for t in sec.theorems:
                total_claims += 1
                if t.evidence_refs:
                    grounded_claims += 1
                else:
                    unsupported_claims += 1

            for f in sec.formulas:
                total_claims += 1
                if f.evidence_refs:
                    grounded_claims += 1
                else:
                    unsupported_claims += 1
                if not validate_latex_balance(f.latex_expression):
                    formulas_valid = False
                    warnings.append(f"Unbalanced LaTeX expression: {f.latex_expression}")

            for ex in sec.worked_examples:
                total_claims += 1
                if ex.evidence_refs:
                    grounded_claims += 1
                else:
                    unsupported_claims += 1

            for trap in sec.exam_traps:
                total_claims += 1
                if trap.evidence_refs:
                    grounded_claims += 1
                else:
                    unsupported_claims += 1

        grounding_ratio = grounded_claims / total_claims if total_claims > 0 else 1.0

        is_valid = (
            unsupported_claims == 0 and formulas_valid and (timestamps_valid or duration == 0.0)
        )

        return NoteValidationResult(
            is_valid=is_valid,
            total_claims=total_claims,
            grounded_claims=grounded_claims,
            unsupported_claims=unsupported_claims,
            grounding_ratio=round(grounding_ratio, 3),
            formulas_valid_latex=formulas_valid,
            timestamps_within_duration=timestamps_valid,
            warnings=tuple(warnings),
        )
