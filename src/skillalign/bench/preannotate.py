"""Pre-annotation: a model reads a whole skill and drafts one `Annotation` for human checking.

Outputs go to `<out_dir>/<stem>.json`. The run is resumable: a skill whose output exists is
skipped and never overwritten. A skill whose reply never validates gets
`<stem>.failed.json` and the batch goes on. Span errors (checked against the files) are
recorded in the annotation's provenance notes, and the skill also gets a `.failed.json`
so they show up in the failure log. Span-invalid annotations are kept for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import openai
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skillalign.intents.extract import file_stem
from skillalign.llm import LLMClient, StructuredOutputError
from skillalign.llm.templates import load_prompt, with_line_numbers
from skillalign.models import (
    Alignment,
    Annotation,
    Behavior,
    Evidence,
    FileRole,
    Finding,
    Intent,
    Provenance,
    SkillLabel,
    Span,
)
from skillalign.normalize.loader import LoadedSkill, load_skill
from skillalign.spans import check_annotation

PREANNOTATE_MAX_TOKENS = 4096
#: Longest skill text sent in one prompt, in estimated tokens. Files that do not fit are left
#: out and named in the provenance notes.
MAX_SKILL_TOKENS = 10_500  # 16384 context - 4096 output - template and line numbers


class DraftSpan(Span):
    """A cited line range. The model gives no quote; the code copies it from the files,
    because Qwen3-32B could not copy text exactly (it joined wrapped lines, swapped quote
    marks and shortened with "...")."""

    model_config = ConfigDict(extra="ignore")


class DraftIntent(Intent):
    sources: list[DraftSpan] = Field(min_length=1)  # type: ignore[assignment]


class DraftBehavior(Behavior):
    evidence: list[DraftSpan] = Field(min_length=1)  # type: ignore[assignment]


class Draft(BaseModel):
    """What the model is asked for: an `Annotation` without the bookkeeping fields.

    Every part is required. In `Annotation` they default to empty lists, and constrained
    decoding then lets the model stop after `skill_label` and invent a `provenance`.
    """

    model_config = ConfigDict(extra="ignore")

    skill_label: SkillLabel
    intents: list[DraftIntent]
    behaviors: list[DraftBehavior]
    alignment: list[Alignment]
    findings: list[Finding]

    @model_validator(mode="after")
    def _cross_references(self) -> Draft:
        # Run the id and reference checks of `Annotation`, so a bad draft is retried.
        to_annotation(self, "draft", lambda span: "x")
        return self


#: Longest quote copied from the files, in characters.
QUOTE_CHARS = 300
UNRESOLVED = "[unresolved span]"


def span_quote(span: Span, root: Path) -> str:
    """The text of the cited lines, copied from the file. `UNRESOLVED` if there is none, so
    `check_annotation` reports the span."""
    path = (root / span.file).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return UNRESOLVED
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines[span.line_start - 1 : span.line_end]).strip()
    return text[:QUOTE_CHARS] or UNRESOLVED


def canonical_files(draft: Draft, loaded: LoadedSkill) -> None:
    """Fix the case of cited file names: a skill with `SKILL.MD` gets cited as `SKILL.md`.
    Only an unambiguous case-insensitive match is rewritten; any other name stays and is
    reported by the span check."""
    by_lower: dict[str, list[str]] = {}
    for path in loaded.texts:
        by_lower.setdefault(path.lower(), []).append(path)
    spans = [sp for i in draft.intents for sp in i.sources]
    spans += [sp for b in draft.behaviors for sp in b.evidence]
    spans += [sp for f in draft.findings for sp in f.evidence]
    for sp in spans:
        found = by_lower.get(sp.file.lower(), [])
        if sp.file not in loaded.texts and len(found) == 1:
            sp.file = found[0]


def to_annotation(draft: Draft, skill_id: str, quote_of) -> Annotation:
    def evidence(spans: list[DraftSpan]) -> list[Evidence]:
        return [Evidence(**dict(sp), quote=quote_of(sp)) for sp in spans]

    return Annotation(
        skill_id=skill_id, skill_label=draft.skill_label,
        intents=[Intent(**{**dict(i), "sources": evidence(i.sources)}) for i in draft.intents],
        behaviors=[Behavior(**{**dict(b), "evidence": evidence(b.evidence)})
                   for b in draft.behaviors],
        alignment=draft.alignment, findings=draft.findings)


@dataclass
class Outcome:
    skill_id: str
    status: str  # "done", "failed", or "skipped"
    path: Path | None = None
    error: str | None = None


def est_tokens(text: str) -> int:
    """A cautious token estimate: ASCII at 3.5 characters a token, anything else at 1.5 tokens a
    character (CJK text tokenizes far worse than the 4-chars rule)."""
    ascii_chars = sum(c < "\x80" for c in text)
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars) * 1.5) + 1


def skill_text(loaded: LoadedSkill) -> tuple[str, list[str]]:
    """All real text files, numbered, SKILL.md first. Returns (text, files left out)."""
    files = sorted(loaded.skill.files, key=lambda a: (a.role is not FileRole.SKILL_MD, a.path))
    parts: list[str] = []
    left_out: list[str] = []
    used = 0
    for art in files:
        if art.role is FileRole.VIRTUAL or art.path not in loaded.texts:
            continue
        block = f"=== FILE {art.path} ===\n{with_line_numbers(loaded.texts[art.path])}\n"
        if used + est_tokens(block) > MAX_SKILL_TOKENS:
            if not parts:  # the first file alone is too long: send its top, mark it cut
                keep = block.splitlines()
                while est_tokens("\n".join(keep)) > MAX_SKILL_TOKENS:
                    keep = keep[: int(len(keep) * 0.8)]
                parts.append("\n".join(keep) + "\n")
                used = MAX_SKILL_TOKENS
                left_out.append(f"{art.path} (cut)")
                continue
            left_out.append(art.path)
            continue
        parts.append(block)
        used += est_tokens(block)
    return "".join(parts), left_out


def preannotate_skill(client: LLMClient, skill_dir: Path, skill_id: str, out_dir: Path,
                      ) -> Outcome:
    stem = file_stem(skill_id)
    out = out_dir / f"{stem}.json"
    failed = out_dir / f"{stem}.failed.json"
    if out.exists():
        return Outcome(skill_id, "skipped", out)

    loaded = load_skill(skill_dir, skill_id=skill_id)
    text, left_out = skill_text(loaded)
    prompt = load_prompt("preannotate")
    message = prompt.render(skill_id=skill_id, skill_text=text)
    try:
        [draft] = client.structured([{"role": "user", "content": message}], Draft, n=1,
                                    max_tokens=PREANNOTATE_MAX_TOKENS)
    except (StructuredOutputError, openai.APIError) as err:  # a timeout must not end the batch
        out_dir.mkdir(parents=True, exist_ok=True)
        failed.write_text(json.dumps({"skill_id": skill_id, "error": str(err),
                                      "prompt_hash": prompt.sha256}, indent=2) + "\n")
        return Outcome(skill_id, "failed", error=str(err))

    canonical_files(draft, loaded)
    ann = to_annotation(draft, skill_id, lambda span: span_quote(span, loaded.root))
    span_errors = check_annotation(ann, loaded.root)
    notes = [f"date {date.today().isoformat()}"]
    if left_out:
        notes.append(f"files left out of prompt: {', '.join(left_out)}")
    if span_errors:
        notes.append(f"span errors ({len(span_errors)}): " + " | ".join(span_errors))
    provenance = Provenance(system="preannotate", model=client.model,
                            prompt_hash=prompt.sha256, seed=client.settings.seed,
                            notes="; ".join(notes))
    ann = ann.model_copy(update={"prefilled": True, "provenance": provenance})

    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(ann.model_dump_json(indent=2) + "\n")
    if span_errors:
        failed.write_text(json.dumps({"skill_id": skill_id, "error": "span errors",
                                      "span_errors": span_errors}, indent=2) + "\n")
    elif failed.exists():
        failed.unlink()
    return Outcome(skill_id, "done", out, error="; ".join(span_errors) or None)


def run_batch(client: LLMClient, sample: list[dict], corpus_dir: Path, out_dir: Path,
              *, limit: int | None = None) -> list[Outcome]:
    """Pre-annotate the first `limit` entries of a sample (`bench/sample.py` output)."""
    entries = sample[:limit] if limit is not None else sample
    return [preannotate_skill(client, corpus_dir / e["path"], e["skill_id"], out_dir)
            for e in entries]


def load_sample(path: Path) -> list[dict]:
    return json.loads(path.read_text())
