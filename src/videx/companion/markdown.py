"""Markdown and Obsidian-compatible exporter for VIDEX study notes.

Generates:
- Publication-ready, structured Markdown with clickable YouTube timestamp links
- Obsidian YAML frontmatter, wiki-links, and callouts
- Resumable playlist index (00_Playlist_Index.md)
- Global topic index (00_Topic_Index.md) cross-referencing mathematical concepts across videos
"""

from __future__ import annotations

import re
from pathlib import Path

from videx.companion.schemas import (
    PlaylistProcessingManifest,
    StudyNote,
)


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    total_sec = int(seconds)
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    s = re.sub(r'[\\/*?:"<>|\[\]{}]', "", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:120]


class MarkdownExporter:
    """Exports StudyNote and Playlist manifests to publication-ready Markdown."""

    def __init__(self, enable_obsidian: bool = True) -> None:
        self.enable_obsidian = enable_obsidian

    def export_note_to_markdown(self, note: StudyNote) -> str:
        """Convert a StudyNote into clean, evidence-grounded Markdown."""
        lines: list[str] = []

        # 1. Frontmatter
        if self.enable_obsidian:
            clean_title = note.video_title.replace('"', '\\"')
            lines.append("---")
            lines.append(f'title: "{clean_title}"')
            lines.append(f'video_id: "{note.video_id}"')
            lines.append(f'source_url: "{note.source_url}"')
            if note.playlist_id:
                lines.append(f'playlist_id: "{note.playlist_id}"')
            if note.channel_title:
                clean_lecturer = note.channel_title.replace('"', '\\"')
                lines.append(f'lecturer: "{clean_lecturer}"')
            lines.append(f'duration: "{format_seconds(note.duration_seconds)}"')
            lines.append(f'coverage: "{note.coverage_percentage}%"')
            lines.append(f'status: "{note.status.value}"')
            lines.append("tags:")
            lines.append("  - gate-data-science")
            lines.append("  - linear-algebra")
            lines.append("  - videx-study-note")
            lines.append("---\n")

        # 2. Header
        lines.append(f"# {note.video_title}\n")
        lines.append(
            f"**Source Video:** [{note.source_url}]({note.source_url}) | "
            f"**Duration:** {format_seconds(note.duration_seconds)} | "
            f"**Transcript Coverage:** {note.coverage_percentage}%\n"
        )

        # Status badge if incomplete
        if note.status.value != "complete":
            lines.append(
                f"> [!warning] Processing Status: {note.status.value.upper()}\n"
                f"> {note.executive_summary}\n"
            )

        # 3. Learning Objectives
        if note.learning_objectives:
            lines.append("## 🎯 Learning Objectives\n")
            for obj in note.learning_objectives:
                lines.append(f"- {obj}")
            lines.append("")

        # 4. Executive Summary
        lines.append("## 📌 Executive Summary\n")
        lines.append(f"{note.executive_summary}\n")

        # 5. Thematic Sections with Clickable Timestamps
        if note.sections:
            lines.append("## 📖 Lecture Notes & Concepts\n")
            for idx, sec in enumerate(note.sections, start=1):
                start_label = format_seconds(sec.start_seconds)
                click_url = (
                    f"https://youtube.com/watch?v={note.video_id}&t={int(sec.start_seconds)}s"
                )
                lines.append(f"### {idx}. {sec.title} [{start_label}]({click_url})\n")
                lines.append(f"{sec.overview}\n")

                # Definitions
                for d in sec.definitions:
                    ts_link = ""
                    if d.evidence_refs:
                        ref = d.evidence_refs[0]
                        ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                    lines.append(f"> [!note] Definition: **{d.term}**{ts_link}")
                    lines.append(f"> {d.definition}")
                    if d.notation:
                        lines.append(f"> \n> **Notation:** {d.notation}")
                    if d.assumptions:
                        lines.append(f"> \n> **Assumptions:** {', '.join(d.assumptions)}")
                    lines.append("")

                # Theorems
                for thm in sec.theorems:
                    ts_link = ""
                    if thm.evidence_refs:
                        ref = thm.evidence_refs[0]
                        ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                    lines.append(f"> [!abstract] Theorem: **{thm.name}**{ts_link}")
                    lines.append(f"> **Statement:** {thm.statement}")
                    if thm.conditions:
                        lines.append(f"> \n> **Conditions:** {', '.join(thm.conditions)}")
                    lines.append(f"> \n> **Conclusion:** {thm.conclusion}")
                    lines.append("")

                # Formulas
                for f in sec.formulas:
                    ts_link = ""
                    if f.evidence_refs:
                        ref = f.evidence_refs[0]
                        ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                    lines.append(f"#### 📐 Formula: {f.name}{ts_link}\n")
                    lines.append(f"$$\n{f.latex_expression.strip('$')}\n$$\n")
                    if f.variables:
                        lines.append("**Variables:**")
                        for var_k, var_desc in f.variables.items():
                            lines.append(f"- `${var_k}$`: {var_desc}")
                        lines.append("")

                # Worked Examples
                for ex in sec.worked_examples:
                    ts_link = ""
                    if ex.evidence_refs:
                        ref = ex.evidence_refs[0]
                        ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                    lines.append(f"> [!example] Worked Example{ts_link}")
                    lines.append(f"> **Problem:** {ex.problem_statement}")
                    if ex.step_by_step_solution:
                        lines.append("> \n> **Solution:**")
                        for step in ex.step_by_step_solution:
                            lines.append(f"> - {step}")
                    lines.append(f"> \n> **Result:** {ex.final_result}")
                    if ex.key_insight:
                        lines.append(f"> \n> **Insight:** {ex.key_insight}")
                    lines.append("")

                # Exam Traps
                for trap in sec.exam_traps:
                    ts_link = ""
                    if trap.evidence_refs:
                        ref = trap.evidence_refs[0]
                        ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                    origin = (
                        "Direct from lecture"
                        if trap.is_directly_stated_in_lecture
                        else "Conceptually inferred"
                    )
                    lines.append(f"> [!caution] GATE Exam Trap ({origin}){ts_link}")
                    lines.append(f"> **Mistake / Pitfall:** {trap.trap_description}")
                    lines.append(f"> \n> **Correct Concept:** {trap.correct_understanding}")
                    lines.append(f"> \n> **Why Tested in GATE:** {trap.why_it_matters_in_gate}")
                    lines.append("")

        # 6. Key Takeaways
        if note.key_takeaways:
            lines.append("## 💡 High-Yield Revision Takeaways\n")
            for t in note.key_takeaways:
                ts_link = ""
                if t.evidence_refs:
                    ref = t.evidence_refs[0]
                    ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                lines.append(f"- **[{t.importance.upper()}]** {t.takeaway}{ts_link}")
            lines.append("")

        # 7. Practice Questions
        if note.practice_questions:
            lines.append("## ❓ Source-Grounded Practice Questions\n")
            for idx, q in enumerate(note.practice_questions, start=1):
                ts_link = ""
                if q.evidence_refs:
                    ref = q.evidence_refs[0]
                    ts_link = f" ([{ref.timestamp_label}]({ref.clickable_url}))"
                lines.append(f"**Q{idx} ({q.question_type.title()}):** {q.question}{ts_link}")
                lines.append("<details><summary>Click to view Answer / Hint</summary>")
                lines.append(f"\n> {q.answer_or_hint}\n")
                lines.append("</details>\n")

        # 8. Uncertainties and Missing Inputs
        if note.uncertainties:
            lines.append("## ⚠️ Epistemic Uncertainties & Content Notes\n")
            for u in note.uncertainties:
                lines.append(f"> [!warning] {u.uncertainty_type.value.replace('_', ' ').title()}")
                lines.append(f"> **Scope:** {u.affected_concept_or_timestamp}")
                lines.append(f"> \n> **Detail:** {u.description}")
                lines.append(f"> \n> **Action Needed:** {u.suggested_resolution}")
                lines.append("")

        # 9. Validation Audit Summary Footer
        if note.validation:
            v = note.validation
            lines.append("---\n")
            lines.append(
                f"*(VIDEX Grounding Audit: {v.grounded_claims}/{v.total_claims} claims grounded "
                f"({v.grounding_ratio * 100:.1f}%), LaTeX valid: {v.formulas_valid_latex}, "
                f"Status: {note.status.value})*\n"
            )

        return "\n".join(lines)

    def write_note_file(self, note: StudyNote, output_dir: Path | str) -> Path:
        """Write note to a markdown file in output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"{sanitize_filename(note.video_title)}.md"
        file_path = out_path / fname
        content = self.export_note_to_markdown(note)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def generate_playlist_index(
        self, manifest: PlaylistProcessingManifest, notes: list[StudyNote]
    ) -> str:
        """Generate a complete 00_Playlist_Index.md linking to all notes and source videos."""
        lines: list[str] = []
        lines.append(f"# 📚 {manifest.playlist_title}\n")
        lines.append(
            f"**Playlist URL:** [{manifest.source_url}]({manifest.source_url}) | "
            f"**Total Discovered:** {manifest.total_videos_discovered} videos | "
            f"**Generated Notes:** {manifest.processed_count} | "
            f"**Awaiting Transcripts:** {manifest.awaiting_transcript_count}\n"
        )
        lines.append("## 📑 Course Syllabus & Video Directory\n")
        lines.append("| # | Video Title | Duration | Status | Study Note | Source Link |")
        lines.append("|---|---|---|---|---|---|")

        notes_by_id = {n.video_id: n for n in notes}

        for r in manifest.records:
            note = notes_by_id.get(r.video_id)
            dur_str = format_seconds(note.duration_seconds) if note else "-"
            status_badge = f"`{r.status.value}`"
            note_link = f"[[{sanitize_filename(r.title)}]]" if r.note_file_path else "*Awaiting*"
            source_link = f"[Watch Video]({r.source_url})"
            row = (
                f"| {r.position + 1} | {r.title} | {dur_str} | {status_badge} | "
                f"{note_link} | {source_link} |"
            )
            lines.append(row)

        lines.append("\n---\n*Generated by VIDEX YouTube Learning Companion*")
        return "\n".join(lines)

    def generate_topic_index(self, notes: list[StudyNote]) -> str:
        """Generate 00_Topic_Index.md cross-linking concepts across multiple lectures."""
        lines: list[str] = []
        lines.append("# 🔍 GATE Linear Algebra — Topic & Concept Index\n")
        lines.append(
            "Cross-video index linking definitions, theorems, and key formulas across the course.\n"
        )

        topics_map: dict[str, list[tuple[str, str, str]]] = {}

        for note in notes:
            clean_title = sanitize_filename(note.video_title)
            for sec in note.sections:
                for d in sec.definitions:
                    topics_map.setdefault(d.term, []).append(
                        (note.video_title, clean_title, "Definition")
                    )
                for t in sec.theorems:
                    topics_map.setdefault(t.name, []).append(
                        (note.video_title, clean_title, "Theorem")
                    )
                for f in sec.formulas:
                    topics_map.setdefault(f.name, []).append(
                        (note.video_title, clean_title, "Formula")
                    )

        if not topics_map:
            lines.append("*No concepts indexed yet.*")
            return "\n".join(lines)

        for concept in sorted(topics_map.keys()):
            occurrences = topics_map[concept]
            lines.append(f"### {concept}\n")
            for vid_title, fname, item_type in occurrences:
                lines.append(f"- **{item_type}** in [[{fname}|{vid_title}]]")
            lines.append("")

        return "\n".join(lines)
