import pytest

from skillalign.normalize.languages import sniff_untagged_fence
from skillalign.normalize.markdown import parse_markdown


@pytest.mark.parametrize("text, lang", [
    ("git grep -c -E 'x' -- 'src'", "shell"),
    ("$ npm install left-pad", "shell"),
    ("> uv run pytest", "shell"),
    ("./scripts/run.sh --fast", "shell"),
    ("#!/usr/bin/env python3\nprint(1)", "python"),
    ("\n\n  curl -s https://example.com", "shell"),
    ("<CLI> metrics", None),
    ("'<ui pathspec>' is a placeholder", None),
    ("ImportError: Module does not define X", None),
    ("| Dimension | Old |", None),
    ("", None),
])
def test_sniff_untagged_fence(text, lang):
    assert sniff_untagged_fence(text) == lang


def test_fenced_untagged_and_indented_blocks_are_flagged():
    doc = parse_markdown(
        "# Title\n\n"
        "```\ngit grep -n 'x'\n```\n\n"
        "```\nplain prose here\n```\n\n"
        "    indented code\n"
    )
    fenced = [b.fenced for b in doc.code_blocks]
    assert fenced == [True, True, False]
    assert doc.code_blocks[0].lang == "" and doc.code_blocks[0].text == "git grep -n 'x'"
