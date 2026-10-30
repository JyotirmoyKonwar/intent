"""Evidence cards: templated summaries of behaviors, the only view of code the judge gets.

A card is a one-line summary (capability, action, target, destination, origin, reachability),
the composite parts and call chain when there are any, and a short numbered snippet of the
cited lines. The text comes from a template; no LLM writes it. With `mode="raw"` (ablation
`--cards raw`) the card carries a longer snippet and no summary line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from skillalign.llm.templates import with_line_numbers
from skillalign.models import Behavior, Card, Span
from skillalign.normalize.loader import LoadedSkill

SNIPPET_MAX_LINES = 10
RAW_MAX_LINES = 40
Mode = Literal["cards", "raw"]


def _arrow(b: Behavior) -> str:
    parts = [b.action]
    if b.target:
        parts.append(f"target={b.target}")
    if b.destination:
        parts.append(f"to={b.destination}")
    return "  ".join(parts)


def card_text(b: Behavior, by_id: dict[str, Behavior]) -> str:
    lines = [f"[{b.id}] {b.capability.value}  {_arrow(b)}  "
             f"origin={b.origin.value}  reach={b.reachability.value}"]
    if b.composite_of:
        parts = ", ".join(f"{p} ({by_id[p].capability.value})" if p in by_id else p
                          for p in b.composite_of)
        lines.append(f"     composite of: {parts}")
    if b.conditions:
        lines.append(f"     only when: {'; '.join(b.conditions)}")
    if b.call_chain:
        lines.append(f"     chain: {' | '.join(b.call_chain)}")
    return "\n".join(lines)


def _snippet(loaded: LoadedSkill, span: Span, max_lines: int) -> str:
    text = loaded.texts.get(span.file)
    if text is None:
        return ""
    lines = text.splitlines()
    end = min(span.line_end, span.line_start + max_lines - 1, len(lines))
    body = "\n".join(lines[span.line_start - 1:end])
    shown = with_line_numbers(body, start=span.line_start)
    return shown + ("\n..." if end < span.line_end else "")


def build_card(loaded: LoadedSkill, b: Behavior, by_id: dict[str, Behavior], *,
               mode: Mode = "cards") -> Card:
    ev = b.evidence[0]
    span = Span(file=ev.file, line_start=min(e.line_start for e in b.evidence
                                             if e.file == ev.file),
                line_end=max(e.line_end for e in b.evidence if e.file == ev.file))
    limit = SNIPPET_MAX_LINES if mode == "cards" else RAW_MAX_LINES
    snippet = _snippet(loaded, span, limit)
    others = [e for e in b.evidence if e.file != ev.file][:2]
    for e in others:  # a second file's lines, short
        snippet += f"\n[{e.file}]\n" + _snippet(loaded, Span(file=e.file,
                                                             line_start=e.line_start,
                                                             line_end=e.line_end), 4)
    head = f"[{b.id}] ({span.file}:{span.line_start}-{span.line_end})"
    text = card_text(b, by_id) if mode == "cards" else head
    return Card(behavior_id=b.id, text=text, snippet=snippet.strip("\n"), span=span,
                chain=list(b.call_chain))


def build_cards(loaded: LoadedSkill, behaviors: Sequence[Behavior], *,
                mode: Mode = "cards") -> dict[str, Card]:
    by_id = {b.id: b for b in behaviors}
    return {b.id: build_card(loaded, b, by_id, mode=mode) for b in behaviors}


def render_card(card: Card) -> str:
    """The card as the judge reads it."""
    if not card.snippet:
        return card.text
    body = "\n".join("        " + ln for ln in card.snippet.splitlines())
    where = f"{card.span.file}:{card.span.line_start}-{card.span.line_end}"
    return f"{card.text}\n     {where}:\n{body}"
