"""B1 claim extraction: one chunk of skill documentation at a time, quotes checked on disk.

For each doc chunk the model lists claims with a verbatim quote and line range. The code
checks each quote with `spans.check_span` and drops claims whose quote is not in the cited
lines. Quotes are never repaired. Other chunks see the frontmatter description as
"already known", so the model adds only what the description leaves out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from skillalign.llm import LLMClient, StructuredOutputError
from skillalign.llm.templates import load_prompt, with_line_numbers
from skillalign.models import Evidence, Intent, Polarity
from skillalign.normalize.loader import load_skill
from skillalign.spans import check_span


class ClaimDraft(BaseModel):
    action: str = Field(min_length=1)
    object: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    polarity: Polarity = Polarity.POSITIVE
    quote: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class ClaimReply(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)


@dataclass
class ExtractStats:
    chunks: int = 0
    failed_chunks: int = 0  # the model gave no valid reply for this chunk
    claims_raw: int = 0
    dropped_bad_span: int = 0  # quote not found in the cited lines (or lines invalid)
    relocated: int = 0  # quote was verbatim in the chunk, but the model cited the wrong lines
    claims_kept: int = 0


@dataclass
class ExtractResult:
    skill_id: str
    intents: list[Intent]
    stats: ExtractStats = field(default_factory=ExtractStats)


def extract_claims(client: LLMClient, skill_dir: Path, *, skill_id: str | None = None,
                   ) -> ExtractResult:
    """Extract intents (B1) from one skill folder. Makes one structured call per chunk."""
    loaded = load_skill(skill_dir, skill_id=skill_id)
    prompt = load_prompt("claims_b1")
    description = loaded.description or ""
    known = description or "(none)"
    stats = ExtractStats()
    intents: list[Intent] = []

    for chunk in loaded.chunks:
        stats.chunks += 1
        lines = loaded.texts[chunk.file].splitlines()[chunk.line_start - 1:chunk.line_end]
        section = with_line_numbers("\n".join(lines), start=chunk.line_start)
        # The description chunk is the source of the seed, so it is not told about itself.
        seed_text = "(none)" if chunk.kind == "frontmatter-description" else known
        message = prompt.render(known=seed_text, file=chunk.file, section=section)
        try:
            [reply] = client.structured([{"role": "user", "content": message}], ClaimReply,
                                        n=1)
        except StructuredOutputError:
            stats.failed_chunks += 1
            continue
        for draft in reply.claims:
            stats.claims_raw += 1
            intent = _checked_intent(draft, chunk.file, loaded.root, len(intents) + 1)
            if intent is None:
                intent = _relocated_intent(draft, chunk, lines, loaded.root, len(intents) + 1)
                stats.relocated += intent is not None
            if intent is None:
                stats.dropped_bad_span += 1
                continue
            intents.append(intent)
            stats.claims_kept += 1

    return ExtractResult(skill_id=skill_id or loaded.skill.skill_id, intents=intents,
                         stats=stats)


def file_stem(skill_id: str) -> str:
    """A skill id as a file name: `github/owner/repo/@sha` -> `github__owner__repo__@sha`."""
    return skill_id.replace("/", "__")


def save_claims(result: ExtractResult, out_dir: Path) -> Path:
    """Write `<stem>.json` (a list of Intent) and `<stem>.stats.json` under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = file_stem(result.skill_id)
    out = out_dir / f"{stem}.json"
    out.write_text(json.dumps([i.model_dump(mode="json") for i in result.intents],
                              indent=2) + "\n")
    (out_dir / f"{stem}.stats.json").write_text(
        json.dumps(asdict(result.stats), indent=2) + "\n")
    return out


MAX_QUOTE_LINES = 4


def _relocated_intent(draft: ClaimDraft, chunk, lines: list[str], root: Path, number: int,
                      ) -> Intent | None:
    """Models often quote the right text but number the line wrong. Look for the quote in the
    chunk's own lines and cite where it really is. The quote is still checked against the
    file by `check_span`, so nothing is repaired: the quote must be there word for word."""
    for width in range(1, MAX_QUOTE_LINES + 1):
        for offset in range(len(lines) - width + 1):
            start = chunk.line_start + offset
            moved = draft.model_copy(update={"line_start": start, "line_end": start + width - 1})
            intent = _checked_intent(moved, chunk.file, root, number)
            if intent is not None:
                return intent
    return None


def _checked_intent(draft: ClaimDraft, file: str, root: Path, number: int) -> Intent | None:
    """An Intent for `draft` if its quote is in the cited lines of `file`, else None."""
    try:
        ev = Evidence(file=file, line_start=draft.line_start, line_end=draft.line_end,
                      quote=draft.quote)
    except ValidationError:
        return None
    if check_span(ev, root) is not None:
        return None
    return Intent(id=f"I{number}", action=draft.action, object=draft.object,
                  constraints=draft.constraints, polarity=draft.polarity, sources=[ev])
