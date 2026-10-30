"""L1 on the 17 fixtures with the served model; writes predictions and prints the scores."""
import json, sys
from pathlib import Path
from skillalign.baselines.l1 import run_l1
from skillalign.eval.metrics import evaluate
from skillalign.llm import LLMClient
from skillalign.models import Annotation

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-32B"
root = Path("tests/fixtures"); out = Path("data/bench/l1_fixtures") / model.replace("/", "_")
out.mkdir(parents=True, exist_ok=True)
client = LLMClient(model=model, thinking=False)
golds, preds = {}, {}
for gp in sorted((root / "gold").glob("*.json")):
    g = Annotation.model_validate_json(gp.read_text()); golds[g.skill_id] = g
    p = run_l1(client, root / "skills" / gp.stem, g.skill_id); preds[g.skill_id] = p
    (out / gp.name).write_text(p.model_dump_json(indent=1))
    print(gp.stem, p.skill_label.value, [f.type.value for f in p.findings], flush=True)
for lenient in (False, True):
    ev = evaluate(golds, preds, lenient=lenient); c = ev.skill_level
    print("lenient" if lenient else "strict", f"skill P {c.precision:.2f} R {c.recall:.2f} F1 {c.f1:.2f} macro {ev.macro_f1():.2f}",
          {t: round(k.f1, 2) for t, k in ev.per_type().items()})
print(client.usage)
