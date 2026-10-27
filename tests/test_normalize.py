import pytest

from skillalign.models import FileRole, Span
from skillalign.normalize.languages import detect_language, file_role
from skillalign.normalize.loader import load_skill
from skillalign.normalize.markdown import parse_markdown

from .conftest import SKILLS

DOC = """---
name: demo
description: >
  Does a thing
  over two lines.
allowed-tools: Bash Read
---

Intro text.

# Title

Use `<!-- toc -->` markers.

<!-- hidden
instruction -->

```python
print("hi")
x = 1
```

## Part two

    indented code
"""


def test_frontmatter_and_key_lines():
    doc = parse_markdown(DOC)
    fm = doc.frontmatter
    assert fm.present and (fm.line_start, fm.line_end) == (1, 7)
    assert fm.data["name"] == "demo"
    assert fm.get_str("description") == "Does a thing over two lines."
    assert fm.key_line("description") == 3


def test_sections_code_and_comments():
    doc = parse_markdown(DOC)
    assert [(s.heading, s.level, s.line_start, s.line_end) for s in doc.sections] == [
        ("", 0, 8, 10), ("Title", 1, 11, 22), ("Part two", 2, 23, 25)]
    py = doc.code_blocks[0]
    assert (py.lang, py.line_start, py.line_end) == ("python", 19, 20)
    assert doc.text(py.line_start, py.line_end) == 'print("hi")\nx = 1'
    assert doc.code_blocks[1].lang == ""  # indented block
    assert [(c.line_start, c.line_end, c.text) for c in doc.comments] == [
        (15, 16, "hidden\ninstruction")]


@pytest.mark.parametrize("text, error", [
    ("# no frontmatter\n", None),
    ("---\nname: [unclosed\n---\n", "yaml"),
    ("---\n- a list\n---\n", "not a mapping"),
    ("---\nname: x\n", "unclosed"),
])
def test_bad_frontmatter(text, error):
    fm = parse_markdown(text).frontmatter
    assert fm.data is None
    assert (fm.error is None) if error is None else (error in fm.error)


def test_languages_and_roles():
    assert detect_language("x/run", "#!/usr/bin/env python3") == "python"
    assert detect_language("a.sh") == "shell"
    assert file_role("package.json", "json") is FileRole.CONFIG
    req = "requirements-dev.txt"
    assert file_role(req, detect_language(req)) is FileRole.CONFIG
    assert file_role("docs/usage.md", "markdown") is FileRole.DOC
    assert file_role("SKILL.md", "markdown", is_root_skill_md=True) is FileRole.SKILL_MD


def test_load_inline_code_as_virtual_file():
    s = load_skill(SKILLS / "17-ub-inline-code-exfil")
    [v] = [f for f in s.skill.files if f.role is FileRole.VIRTUAL]
    assert v.path == "SKILL.md#L12-L14.sh" and v.language == "shell" and v.parent == "SKILL.md"
    assert s.texts[v.path].splitlines()[2].startswith("curl -s -X POST")
    assert s.real_span(Span(file=v.path, line_start=3, line_end=3)) == Span(
        file="SKILL.md", line_start=14, line_end=14)
    assert s.code_files(deep_only=True) == [v]


def test_load_roles_comments_and_chunks():
    s = load_skill(SKILLS / "10-ub-npm-postinstall")
    roles = {f.path: f.role for f in s.skill.files}
    assert roles == {"SKILL.md": FileRole.SKILL_MD, "package.json": FileRole.CONFIG,
                     "scripts/setup.js": FileRole.CODE, "scripts/toc.js": FileRole.CODE}
    assert s.comments == []  # the toc markers are inside backticks
    assert s.chunks[0].kind == "frontmatter-description" and s.chunks[0].line_start == 3

    hidden = load_skill(SKILLS / "08-ub-hidden-instruction")
    assert [(f, c.line_start) for f, c in hidden.comments] == [("SKILL.md", 11)]


@pytest.mark.parametrize("skill", sorted(p.name for p in SKILLS.iterdir()))
def test_every_fixture_loads(skill):
    s = load_skill(SKILLS / skill)
    assert s.skill_md == "SKILL.md" and s.name and s.description
    assert s.frontmatter_error is None and not s.skipped
    assert s.chunks and all(c.text for c in s.chunks)
