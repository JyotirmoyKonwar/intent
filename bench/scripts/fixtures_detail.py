"""Pipeline on the fixtures with a per-skill diff against gold (cached calls are free)."""
import json, sys
from pathlib import Path
from skillalign.align.pipeline import DetectOptions, detect_skill
from skillalign.llm import LLMClient
from skillalign.models import Annotation
root = Path("tests/fixtures"); client = LLMClient(thinking=False)
for gp in sorted((root / "gold").glob("*.json")):
    g = Annotation.model_validate_json(gp.read_text())
    r = detect_skill(client, root / "skills" / gp.stem, DetectOptions(), skill_id=g.skill_id)
    a = r.annotation
    print(f"{gp.stem[:34]:34} gold {[f.type.value for f in g.findings]} pred {[f.type.value for f in a.findings]}")
    for f in a.findings:
        if f.type.value in ("UI", "UI-partial") and f.intent:
            i = a.intent(f.intent); j = r.judged[f.intent]
            print(f"      {f.type.value} {i.action} | {i.object[:50]} cands={j.candidates} ans={j.answer.achieves_action if j.answer else None}")
