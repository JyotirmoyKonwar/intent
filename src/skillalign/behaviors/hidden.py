"""Hidden-content scan: text a reader does not see but an agent reads."""

from __future__ import annotations

import re
from dataclasses import dataclass

ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
BIDI = re.compile("[‪-‮⁦-⁩]")
TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")
BASE64_BLOB = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/=])")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
INVISIBLE_HTML = re.compile(
    r"<[a-z][^>]*(?:style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|"
    r"font-size\s*:\s*0(?![.\d])|opacity\s*:\s*0(?![.\d])|color\s*:\s*(?:white|#fff))"
    r"|\bhidden(?:\s|>|=))", re.I)
AGENT_WORDS = re.compile(r"\b(ignore (?:all |previous |the )|you (?:must|should|are)|"
                         r"silently|do not (?:tell|mention|show)|without (?:telling|asking)|"
                         r"secretly|always run|run:|execute|curl|wget|sudo|system prompt|"
                         r"before (?:responding|answering)|after (?:translating|responding)|"
                         r"send|upload|exfiltrate)\b", re.I)


@dataclass
class HiddenItem:
    kind: str
    file: str
    line_start: int
    line_end: int
    snippet: str


def scan_text(path: str, text: str) -> list[HiddenItem]:
    out: list[HiddenItem] = []
    for n, line in enumerate(text.splitlines(), 1):
        for kind, rx in (("zero-width-chars", ZERO_WIDTH), ("bidi-control-chars", BIDI),
                         ("unicode-tag-chars", TAG_CHARS), ("ansi-escape", ANSI)):
            if rx.search(line):
                out.append(HiddenItem(kind, path, n, n, repr(line.strip()[:80])))
        m = BASE64_BLOB.search(line)
        if m and "://" not in line and not m.group(0).startswith(("/", ".")):
            out.append(HiddenItem("encoded-blob", path, n, n, m.group(0)[:60]))
        if INVISIBLE_HTML.search(line):
            out.append(HiddenItem("invisible-markup", path, n, n, line.strip()[:80]))
    return out


def comment_items(path: str, comments) -> list[HiddenItem]:
    """HTML comments in Markdown that talk to an agent or hold a command."""
    out: list[HiddenItem] = []
    for c in comments:
        body = c.text.strip()
        if AGENT_WORDS.search(body) or len(body) > 200:
            out.append(HiddenItem("comment-instruction", path, c.line_start, c.line_end,
                                  body[:120]))
    return out
