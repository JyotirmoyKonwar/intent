"""`pytest -m fixtures -s`: the full pipeline on the 17 fixtures against the real server.

Prints skill-level and per-type scores. Skipped when no server answers. Not part of the
default run, which must work without a GPU.
"""

import httpx
import pytest

from skillalign.align.pipeline import DetectOptions, detect_skill
from skillalign.config import get_settings
from skillalign.eval.metrics import evaluate
from skillalign.llm import LLMClient
from skillalign.models import Annotation

from .conftest import GOLD_FILES, SKILLS

pytestmark = pytest.mark.fixtures


def server_up() -> bool:
    try:
        return httpx.get(get_settings().llm_base_url.rstrip("/") + "/models", timeout=3
                         ).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not server_up(), reason="no LLM server reachable")
def test_pipeline_on_fixtures_prints_per_type_f1(capsys):
    golds, preds = {}, {}
    client = LLMClient(thinking=False)
    for path in GOLD_FILES:
        gold = Annotation.model_validate_json(path.read_text())
        golds[gold.skill_id] = gold
        res = detect_skill(client, SKILLS / path.stem, DetectOptions(), skill_id=gold.skill_id)
        preds[gold.skill_id] = res.annotation
    for lenient in (False, True):
        ev = evaluate(golds, preds, lenient=lenient)
        with capsys.disabled():
            c = ev.skill_level
            print(f"\n{'lenient' if lenient else 'strict'}: skill P {c.precision:.2f} "
                  f"R {c.recall:.2f} F1 {c.f1:.2f}; macro F1 {ev.macro_f1():.2f}")
            for t, k in ev.per_type().items():
                print(f"  {t:10} P {k.precision:.2f} R {k.recall:.2f} F1 {k.f1:.2f} "
                      f"(tp {k.tp} fp {k.fp} fn {k.fn})")
    assert preds.keys() == golds.keys()
