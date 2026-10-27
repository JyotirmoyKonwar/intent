import pytest

from skillalign.models import Annotation, Evidence, Span
from skillalign.spans import check_annotation, check_span

from .conftest import GOLD_FILES, SKILLS

ROOT = SKILLS / "11-sd-destination"


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_gold_spans_match_files(path):
    ann = Annotation.model_validate_json(path.read_text())
    assert check_annotation(ann, SKILLS / path.stem) == []


def test_valid_evidence():
    ev = Evidence(file="scripts/report.py", line_start=23, line_end=23,
                  quote="requests.post(SHARE_URL")
    assert check_span(ev, ROOT) is None


def test_quote_across_lines_ignores_whitespace():
    ev = Evidence(file="SKILL.md", line_start=8, line_end=9,
                  quote="saves the report to   `./reports/weekly.md`.")
    assert check_span(ev, ROOT) is None


@pytest.mark.parametrize("span, message", [
    (Evidence(file="scripts/report.py", line_start=22, line_end=22,
              quote="requests.post(SHARE_URL"), "quote not found"),
    (Span(file="scripts/report.py", line_start=20, line_end=99), "only 27 lines"),
    (Span(file="scripts/missing.py", line_start=1, line_end=1), "file not found"),
])
def test_bad_spans(span, message):
    err = check_span(span, ROOT)
    assert err is not None and message in err
