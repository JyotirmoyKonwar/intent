import json

import pytest
from pydantic import ValidationError

from skillalign.models import Annotation, Behavior, Evidence, FindingType, Span

from .conftest import GOLD_FILES, SKILLS


def test_every_fixture_has_gold_and_vice_versa():
    skills = {p.name for p in SKILLS.iterdir() if p.is_dir()}
    gold = {p.stem for p in GOLD_FILES}
    assert skills == gold
    assert len(gold) >= 15


def test_fixtures_cover_every_finding_type_and_aligned_controls():
    anns = [Annotation.model_validate_json(p.read_text()) for p in GOLD_FILES]
    types = {f.type for a in anns for f in a.findings}
    assert types == set(FindingType)
    assert sum(not a.findings for a in anns) >= 4  # aligned controls
    assert any(b.reachability == "dormant" for a in anns for b in a.behaviors)
    assert any(b.capability == "COMPOSITE" for a in anns for b in a.behaviors)
    assert any(b.origin == "agent-instruction" for a in anns for b in a.behaviors)
    assert any(i.polarity == "negative" for a in anns for i in a.intents)


def _mutate(gold_json: str, fn) -> str:
    data = json.loads(gold_json)
    fn(data)
    return json.dumps(data)


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d["findings"][0].update(behaviors=["B99"]), "unknown behavior B99"),
    (lambda d: d["findings"][0].update(intent="I99"), "unknown intent I99"),
    (lambda d: d["findings"][0].update(divergence_axis=None), "needs a divergence_axis"),
    (lambda d: d["findings"][0].update(type="UB"), "UB must not link an intent"),
    (lambda d: d.update(skill_label="aligned"), "skill_label"),
    (lambda d: d["behaviors"].append(dict(d["behaviors"][0])), "duplicate behavior id B1"),
    (lambda d: d["alignment"].append({"intent": "I7", "behaviors": [], "status": "fulfilled"}),
     "unknown intent I7"),
    (lambda d: d["intents"][0].update(sources=[]), "at least 1"),
])
def test_annotation_rejects_inconsistent_data(gold_json, mutation, message):
    with pytest.raises(ValidationError, match=message):
        Annotation.model_validate_json(_mutate(gold_json, mutation))


def test_span_rules():
    with pytest.raises(ValidationError, match="line_end"):
        Span(file="a.py", line_start=5, line_end=4)
    with pytest.raises(ValidationError, match="relative"):
        Span(file="/etc/passwd", line_start=1, line_end=1)
    with pytest.raises(ValidationError, match="relative"):
        Span(file="../x.py", line_start=1, line_end=1)


def test_composite_needs_parts():
    ev = Evidence(file="a.py", line_start=1, line_end=1, quote="x")
    with pytest.raises(ValidationError, match="at least 2 parts"):
        Behavior(id="B1", capability="COMPOSITE", action="x", evidence=[ev], composite_of=["B2"])
    with pytest.raises(ValidationError, match="only COMPOSITE"):
        Behavior(id="B1", capability="FS_READ", action="x", evidence=[ev], composite_of=["B2"])
