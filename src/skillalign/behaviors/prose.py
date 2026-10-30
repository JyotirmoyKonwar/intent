"""C4: behaviors that only exist as instructions in prose (the detectors cannot see them).

The model reads each documentation section and lists the instructions it gives the agent:
"run `git diff --staged`", "show the message to the user", "do not tell the user". The code
checks each quote against the cited lines, as in B1, and drops what does not match. A prose
behavior has `origin=agent-instruction` and `reachability=entry`. A behavior the detectors
already found at the same lines (a fenced block, an HTML comment) is not added twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from skillalign.llm import LLMClient, StructuredOutputError
from skillalign.llm.templates import load_prompt, with_line_numbers
from skillalign.models import Behavior, Capability, Evidence, Origin, Reachability, Span
from skillalign.normalize.loader import LoadedSkill
from skillalign.spans import check_span

PROSE_CAPABILITIES = [
    "FS_READ", "FS_WRITE", "FS_DELETE", "ENV_READ", "CREDENTIAL_ACCESS", "NET_OUTBOUND",
    "DOWNLOAD_EXEC", "PROC_EXEC", "PKG_INSTALL", "PERSISTENCE", "SYS_CONFIG_MODIFY",
    "USER_INTERACTION", "AGENT_INSTRUCTION",
]
Cap = Literal[
    "FS_READ", "FS_WRITE", "FS_DELETE", "ENV_READ", "CREDENTIAL_ACCESS", "NET_OUTBOUND",
    "DOWNLOAD_EXEC", "PROC_EXEC", "PKG_INSTALL", "PERSISTENCE", "SYS_CONFIG_MODIFY",
    "USER_INTERACTION", "AGENT_INSTRUCTION",
]
PROSE_MAX_TOKENS = 1024
MIN_SECTION_CHARS = 20


class ProseDraft(BaseModel):
    capability: Cap
    action: str = Field(min_length=1)
    target: str | None = None
    destination: str | None = None
    quote: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class ProseReply(BaseModel):
    behaviors: list[ProseDraft] = Field(default_factory=list)


@dataclass
class ProseStats:
    chunks: int = 0
    failed_chunks: int = 0
    raw: int = 0
    dropped_bad_span: int = 0
    dropped_duplicate: int = 0
    dropped_own_script: int = 0
    kept: int = 0


def _covered(loaded: LoadedSkill, existing: Sequence[Behavior]) -> list[tuple[Capability, Span]]:
    out = []
    for b in existing:
        for ev in b.evidence:
            try:
                out.append((b.capability, loaded.real_span(ev)))
            except StopIteration:
                out.append((b.capability, ev))
    return out


def _runs_own_script(d: ProseDraft, scripts: set[str]) -> bool:
    """"Run `python scripts/x.py`" starts the skill's own entry point. The script's behaviors
    are found in the code, so the instruction to run it is not a behavior of its own."""
    return d.capability == "PROC_EXEC" and any(
        name in f"{d.target or ''} {d.quote}" for name in scripts)


def _overlaps(a: Span, b: Span) -> bool:
    return a.file == b.file and a.line_start <= b.line_end and b.line_start <= a.line_end


def extract_prose_behaviors(client: LLMClient, loaded: LoadedSkill,
                            existing: Sequence[Behavior] = (), *, first_id: int | None = None,
                            ) -> tuple[list[Behavior], ProseStats]:
    """C4 over every documentation section. New ids continue after `existing`."""
    prompt = load_prompt("c4_prose")
    covered = _covered(loaded, existing)
    number = first_id if first_id is not None else len(existing) + 1
    stats = ProseStats()
    scripts = {PurePosixPath(a.path).name for a in loaded.code_files() if "#" not in a.path}
    out: list[Behavior] = []
    for chunk in loaded.chunks:
        if chunk.kind != "section" or len(chunk.text.strip()) < MIN_SECTION_CHARS:
            continue
        stats.chunks += 1
        lines = loaded.texts[chunk.file].splitlines()[chunk.line_start - 1:chunk.line_end]
        message = prompt.render(file=chunk.file, capabilities=", ".join(PROSE_CAPABILITIES),
                                section=with_line_numbers("\n".join(lines), start=chunk.line_start))
        try:
            [reply] = client.structured([{"role": "user", "content": message}], ProseReply,
                                        n=1, max_tokens=PROSE_MAX_TOKENS)
        except StructuredOutputError:
            stats.failed_chunks += 1
            continue
        for d in reply.behaviors:
            stats.raw += 1
            try:
                ev = Evidence(file=chunk.file, line_start=d.line_start, line_end=d.line_end,
                              quote=d.quote)
            except ValidationError:
                stats.dropped_bad_span += 1
                continue
            if check_span(ev, loaded.root) is not None:
                stats.dropped_bad_span += 1
                continue
            if _runs_own_script(d, scripts):
                stats.dropped_own_script += 1
                continue
            cap = Capability(d.capability)
            # a detector already read these lines (a fenced block, an HTML comment)
            if any(_overlaps(span, ev) for _, span in covered):
                stats.dropped_duplicate += 1
                continue
            out.append(Behavior(id=f"B{number}", capability=cap, action=d.action,
                                target=d.target, destination=d.destination,
                                origin=Origin.AGENT_INSTRUCTION,
                                reachability=Reachability.ENTRY, evidence=[ev]))
            covered.append((cap, ev))
            number += 1
            stats.kept += 1
    return out, stats
