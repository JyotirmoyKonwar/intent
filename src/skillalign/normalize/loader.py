"""Stage A: load a skill folder into one structured view (PLAN.md §6.2).

- every text file with its role (skill_md / doc / code / config / data) and language
- SKILL.md frontmatter (name, description, allowed-tools, ...)
- doc chunks for intent extraction: the frontmatter description and each Markdown section
- fenced code blocks as *virtual files*, so detectors treat them like scripts; their
  spans map back to real lines with `real_span()`
- HTML comments in Markdown (a common hiding place for agent instructions)

Files are only read as text. Nothing is imported or executed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from skillalign.collect.store import IGNORED_DIRS, MAX_FILE_BYTES, is_binary
from skillalign.models import FileArtifact, FileRole, Skill, Span
from skillalign.normalize.languages import (
    CODE_LANGS,
    DEEP_LANGS,
    FENCE_LANG,
    detect_language,
    file_role,
    sniff_untagged_fence,
)
from skillalign.normalize.markdown import HtmlComment, MarkdownDoc, parse_markdown

_FENCE_EXT = {"python": "py", "shell": "sh", "javascript": "js", "typescript": "ts",
              "powershell": "ps1", "json": "json", "yaml": "yaml", "toml": "toml"}


@dataclass
class DocChunk:
    """One piece of declarative text for claim extraction (B1)."""

    file: str
    kind: str  # "frontmatter-description", "section", "frontmatter-field"
    heading: str
    line_start: int
    line_end: int
    text: str

    @property
    def span(self) -> Span:
        return Span(file=self.file, line_start=self.line_start, line_end=self.line_end)


@dataclass
class LoadedSkill:
    root: Path
    skill: Skill
    skill_md: str | None  # path of the root SKILL.md
    frontmatter: dict[str, Any] | None
    frontmatter_error: str | None
    texts: dict[str, str]  # path (real or virtual) → text
    markdown: dict[str, MarkdownDoc]  # real Markdown files
    chunks: list[DocChunk]
    comments: list[tuple[str, HtmlComment]]  # (file, comment)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)

    # -- convenience -------------------------------------------------------

    @property
    def name(self) -> str | None:
        v = (self.frontmatter or {}).get("name")
        return v if isinstance(v, str) else None

    @property
    def description(self) -> str | None:
        v = (self.frontmatter or {}).get("description")
        return v if isinstance(v, str) else None

    @property
    def allowed_tools(self) -> list[str]:
        v = (self.frontmatter or {}).get("allowed-tools")
        if isinstance(v, str):
            return [t.strip() for t in v.replace(",", " ").split() if t.strip()]
        return [str(t) for t in v] if isinstance(v, list) else []

    def files(self, *roles: FileRole) -> list[FileArtifact]:
        return [f for f in self.skill.files if not roles or f.role in roles]

    def artifact(self, path: str) -> FileArtifact:
        return next(f for f in self.skill.files if f.path == path)

    def code_files(self, deep_only: bool = False) -> list[FileArtifact]:
        """Real code files plus virtual files from fenced code, by language."""
        langs = DEEP_LANGS if deep_only else CODE_LANGS
        return [f for f in self.skill.files
                if f.role in (FileRole.CODE, FileRole.VIRTUAL) and f.language in langs]

    def real_span(self, span: Span) -> Span:
        """Map a span in a virtual file back to its Markdown parent."""
        art = self.artifact(span.file)
        if art.role is not FileRole.VIRTUAL or art.parent is None:
            return span
        return Span(file=art.parent, line_start=span.line_start + art.line_offset,
                    line_end=span.line_end + art.line_offset)

    @property
    def languages(self) -> dict[str, int]:
        """Lines of code per language, over real and virtual code files."""
        out: dict[str, int] = {}
        for f in self.code_files():
            out[f.language] = out.get(f.language, 0) + len(self.texts[f.path].splitlines())
        return out


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            yield Path(dirpath) / f


def load_skill(root: Path, *, skill_id: str | None = None, platform: str | None = None,
               source_url: str | None = None, commit: str | None = None,
               license: str | None = None) -> LoadedSkill:
    root = root.resolve()
    artifacts: list[FileArtifact] = []
    texts: dict[str, str] = {}
    markdown: dict[str, MarkdownDoc] = {}
    chunks: list[DocChunk] = []
    comments: list[tuple[str, HtmlComment]] = []
    skipped: list[tuple[str, str]] = []
    skill_md = None

    for p in _walk(root):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            skipped.append((rel, "symlink"))
            continue
        data = p.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            skipped.append((rel, "too-large"))
            continue
        if is_binary(data):
            skipped.append((rel, "binary"))
            continue
        text = data.decode("utf-8", errors="replace")
        lang = detect_language(rel, text.split("\n", 1)[0])
        is_root_md = rel.lower() == "skill.md"
        if is_root_md:
            skill_md = rel
        role = file_role(rel, lang, is_root_skill_md=is_root_md)
        artifacts.append(FileArtifact(path=rel, role=role, language=lang,
                                      sha256=hashlib.sha256(data).hexdigest()))
        texts[rel] = text
        if lang == "markdown":
            markdown[rel] = parse_markdown(text)

    frontmatter = fm_error = None
    for path, doc in markdown.items():
        # Virtual files for fenced code.
        for block in doc.code_blocks:
            if block.fenced and not block.lang:
                lang = sniff_untagged_fence(block.text)
            else:
                lang = FENCE_LANG.get(block.lang)
            if lang is None or block.line_end < block.line_start:
                continue
            vpath = f"{path}#L{block.line_start}-L{block.line_end}.{_FENCE_EXT.get(lang, 'txt')}"
            artifacts.append(FileArtifact(path=vpath, role=FileRole.VIRTUAL, language=lang,
                                          parent=path, line_offset=block.line_start - 1))
            texts[vpath] = block.text
        comments += [(path, c) for c in doc.comments]

        # Doc chunks: frontmatter description, then each section with text in it.
        fm = doc.frontmatter
        if path == skill_md:
            frontmatter, fm_error = fm.data, fm.error
            if fm.data:
                for key in ("description", "when_to_use", "when-to-use"):
                    if (val := fm.get_str(key)) and (line := fm.key_line(key)):
                        end = _value_end(doc, line, fm.line_end)
                        chunks.append(DocChunk(path, "frontmatter-description" if key ==
                                               "description" else "frontmatter-field",
                                               key, line, end, val))
        for sec in doc.sections:
            body = doc.text(sec.line_start, sec.line_end).strip()
            if body:
                chunks.append(DocChunk(path, "section", sec.heading, sec.line_start,
                                       sec.line_end, body))

    skill = Skill(skill_id=skill_id or root.name, platform=platform, source_url=source_url,
                  commit=commit, license=license, files=artifacts)
    return LoadedSkill(root, skill, skill_md, frontmatter, fm_error, texts, markdown, chunks,
                       comments, skipped)


def _value_end(doc: MarkdownDoc, key_line: int, fm_end: int) -> int:
    """Last line of a (possibly multi-line) YAML value that starts on `key_line`."""
    end = key_line
    for n in range(key_line + 1, fm_end):
        line = doc.lines[n - 1]
        if line.strip() and not line[:1].isspace():
            break
        end = n
    return end


def is_skill_dir(path: Path) -> bool:
    return any(PurePosixPath(p.name).name.lower() == "skill.md" for p in path.iterdir()
               if p.is_file())
