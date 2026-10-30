"""Markdown review view of one pre-annotated skill: plain text, no UI.

Every evidence quote is printed verbatim, with the cited lines of the skill numbered
so a reviewer can find them.
"""

from __future__ import annotations

from pathlib import Path

from skillalign.llm.templates import with_line_numbers
from skillalign.models import Annotation, Evidence, Span


def _cited_lines(skill_dir: Path, span: Span) -> str:
    path = skill_dir / span.file
    if not path.is_file():
        return f"(file not found: {span.file})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = "\n".join(lines[span.line_start - 1:span.line_end])
    return with_line_numbers(chunk, start=span.line_start)


def _evidence(skill_dir: Path, spans: list) -> list[str]:
    """Each cited span: its location, the quote (when it has one), and the numbered lines."""
    out: list[str] = []
    for ev in spans:
        out.append(f"`{ev.file}:{ev.line_start}-{ev.line_end}`")
        out.append("")
        if isinstance(ev, Evidence):  # findings cite bare spans, which have no quote
            out += ["Quote:", "", "```text", ev.quote, "```", ""]
        out += ["Lines:", "", "```text", _cited_lines(skill_dir, ev), "```", ""]
    return out


def render_markdown(ann: Annotation, skill_dir: Path) -> str:
    prov = ann.provenance
    lines = [f"# {ann.skill_id}", "",
             f"- label: **{ann.skill_label.value}**",
             f"- split: {ann.split.value if ann.split else '-'}",
             f"- prefilled: {ann.prefilled}, adjudicated: {ann.adjudicated}"]
    if prov is not None:
        lines += [f"- provenance: {prov.system}, model {prov.model or '-'}, "
                  f"prompt {prov.prompt_hash or '-'}, seed {prov.seed}",
                  f"- notes: {prov.notes or '-'}"]
    lines += ["", "## Intents", ""]
    for i in ann.intents:
        lines += [f"### {i.id}: {i.action} {i.object}",
                  f"polarity {i.polarity.value}, modality {i.modality.value}"
                  + (f", constraints: {'; '.join(i.constraints)}" if i.constraints else ""),
                  ""]
        lines += _evidence(skill_dir, i.sources)
        lines.append("")
    lines += ["## Behaviors", ""]
    for b in ann.behaviors:
        target = f" -> {b.target}" if b.target else ""
        dest = f", destination {b.destination}" if b.destination else ""
        lines += [f"### {b.id}: {b.capability.value} {b.action}{target}",
                  f"origin {b.origin.value}, reachability {b.reachability.value}{dest}", ""]
        lines += _evidence(skill_dir, b.evidence)
        lines.append("")
    lines += ["## Alignment", ""]
    lines += [f"- {a.intent}: {a.status.value} (behaviors: {', '.join(a.behaviors) or '-'})"
              for a in ann.alignment] or ["- (none)"]
    lines += ["", "## Findings", ""]
    for f in ann.findings:
        lines += [f"### {f.id}: {f.type.value} (risk {f.risk.value})",
                  f"intent {f.intent or '-'}, behaviors {', '.join(f.behaviors) or '-'}",
                  "", f.rationale or "", ""]
        lines += _evidence(skill_dir, f.evidence)
        lines.append("")
    if not ann.findings:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"
