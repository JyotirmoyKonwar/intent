"""Check that evidence points at real lines and quotes real text in the skill directory."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from skillalign.models import Annotation, Evidence, Span

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


@cache
def _lines(path: Path) -> tuple[str, ...]:
    return tuple(path.read_text(encoding="utf-8", errors="replace").splitlines())


def check_span(span: Span, root: Path) -> str | None:
    """Return an error message, or None if the span is valid under `root`."""
    path = (root / span.file).resolve()
    if not path.is_relative_to(root.resolve()):
        return f"{span.file}: outside the skill root"
    if not path.is_file():
        return f"{span.file}: file not found"
    n = len(_lines(path))
    if span.line_end > n:
        return f"{span.file}:{span.line_start}-{span.line_end}: file has only {n} lines"
    if isinstance(span, Evidence):
        text = "\n".join(_lines(path)[span.line_start - 1 : span.line_end])
        if _norm(span.quote) not in _norm(text):
            return (f"{span.file}:{span.line_start}-{span.line_end}: "
                    f"quote not found in those lines: {span.quote[:60]!r}")
    return None


def check_annotation(ann: Annotation, root: Path) -> list[str]:
    """All span and quote errors in an annotation. Empty list means every span is valid."""
    errors: list[str] = []

    def visit(owner: str, spans: list) -> None:
        for s in spans:
            if (err := check_span(s, root)) is not None:
                errors.append(f"{owner}: {err}")

    for i in ann.intents:
        visit(i.id, i.sources)
    for b in ann.behaviors:
        visit(b.id, b.evidence)
    for f in ann.findings:
        visit(f.id, f.evidence)
    return errors
