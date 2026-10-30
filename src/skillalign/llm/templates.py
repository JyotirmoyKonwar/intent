"""Prompt templates and the line-numbering helper.

Templates live in `prompts/*.j2` as plain text with `$name` placeholders (`string.Template`),
so no template engine is needed. Each loaded template carries a sha256 of its text. Every
annotation records that hash in `Provenance.prompt_hash`, so a prompt edit shows in the output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    sha256: str

    def render(self, **values: str) -> str:
        return Template(self.text).substitute(values)


@cache
def load_prompt(name: str) -> PromptTemplate:
    text = (PROMPT_DIR / f"{name}.j2").read_text(encoding="utf-8")
    return PromptTemplate(name, text, hashlib.sha256(text.encode()).hexdigest())


def with_line_numbers(text: str, start: int = 1) -> str:
    """Prefix every line with its 1-based number, the way `spans.py` counts them: `12| text`."""
    return "\n".join(f"{n}| {line}" for n, line in enumerate(text.splitlines(), start=start))
