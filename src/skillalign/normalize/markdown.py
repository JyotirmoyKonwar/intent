"""Markdown structure with exact line numbers: frontmatter, sections, fenced code, comments.

All line numbers are 1-based and inclusive, matching `Span`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml
from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark")


@dataclass
class Frontmatter:
    data: dict[str, Any] | None  # None when absent or unparsable
    line_start: int = 0  # line of the opening '---' (0 = no frontmatter)
    line_end: int = 0  # line of the closing '---'
    error: str | None = None

    @property
    def present(self) -> bool:
        return self.line_end > 0

    def get_str(self, key: str) -> str | None:
        v = (self.data or {}).get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    def key_line(self, key: str) -> int | None:
        """Line where `key:` is defined, for evidence spans."""
        return self._key_lines.get(key)

    _key_lines: dict[str, int] = field(default_factory=dict)


@dataclass
class Section:
    heading: str  # "" for text before the first heading
    level: int  # 0 for the preamble
    line_start: int  # the heading line (or first body line)
    line_end: int


@dataclass
class CodeBlock:
    lang: str  # info string's first word, lower-case ("" if none)
    line_start: int  # first content line (after the opening fence)
    line_end: int  # last content line; < line_start for an empty block
    text: str


@dataclass
class HtmlComment:
    line_start: int
    line_end: int
    text: str


@dataclass
class MarkdownDoc:
    lines: list[str]
    frontmatter: Frontmatter
    sections: list[Section]
    code_blocks: list[CodeBlock]
    comments: list[HtmlComment]

    def text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])


def parse_frontmatter(lines: list[str]) -> Frontmatter:
    if not lines or lines[0].strip() != "---":
        return Frontmatter(data=None)
    for i in range(1, min(len(lines), 400)):
        if lines[i].strip() in ("---", "..."):
            raw = "\n".join(lines[1:i])
            key_lines = {m.group(1): n + 2 for n, line in enumerate(lines[1:i])
                         if (m := re.match(r"^([A-Za-z0-9_-]+)\s*:", line))}
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as err:
                return Frontmatter(None, 1, i + 1, error=f"yaml: {str(err)[:200]}",
                                   _key_lines=key_lines)
            if not isinstance(data, dict):
                return Frontmatter(None, 1, i + 1, error="frontmatter is not a mapping",
                                   _key_lines=key_lines)
            return Frontmatter(data, 1, i + 1, _key_lines=key_lines)
    return Frontmatter(None, error="unclosed frontmatter")


_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)


def parse_markdown(text: str) -> MarkdownDoc:
    lines = text.splitlines()
    fm = parse_frontmatter(lines)
    offset = fm.line_end  # body starts after the frontmatter
    body = "\n".join(lines[offset:])
    tokens = _md.parse(body)

    headings: list[tuple[int, int, str]] = []  # (line, level, text)
    blocks: list[CodeBlock] = []
    for i, tok in enumerate(tokens):
        if tok.map is None:
            continue
        start, end = tok.map[0] + offset + 1, tok.map[1] + offset  # 1-based inclusive
        if tok.type == "heading_open":
            title = tokens[i + 1].content.strip() if i + 1 < len(tokens) else ""
            headings.append((start, int(tok.tag[1]), title))
        elif tok.type == "fence":
            closed = end > start and lines[end - 1].strip().startswith(tok.markup[0] * 3)
            content_end = end - 1 if closed else end
            blocks.append(CodeBlock(lang=(tok.info.split() or [""])[0].lower(),
                                    line_start=start + 1, line_end=content_end,
                                    text=tok.content.rstrip("\n")))
        elif tok.type == "code_block":  # indented code
            blocks.append(CodeBlock(lang="", line_start=start, line_end=end,
                                    text=tok.content.rstrip("\n")))

    sections: list[Section] = []
    first_body = offset + 1
    n = len(lines)
    if headings and headings[0][0] > first_body and any(
            lines[k].strip() for k in range(first_body - 1, headings[0][0] - 1)):
        sections.append(Section("", 0, first_body, headings[0][0] - 1))
    elif not headings and n >= first_body:
        sections.append(Section("", 0, first_body, n))
    for j, (line, level, title) in enumerate(headings):
        end = headings[j + 1][0] - 1 if j + 1 < len(headings) else n
        sections.append(Section(title, level, line, max(line, end)))

    comments = []
    for m in _COMMENT.finditer(text):
        if text[m.start() - 1 : m.start()] == "`" and text[m.end() : m.end() + 1] == "`":
            continue  # `<!-- x -->` in inline code is documentation, not a hidden comment
        s = text.count("\n", 0, m.start()) + 1
        e = text.count("\n", 0, m.end()) + 1
        comments.append(HtmlComment(s, e, m.group(1).strip()))

    return MarkdownDoc(lines, fm, sections, blocks, comments)
