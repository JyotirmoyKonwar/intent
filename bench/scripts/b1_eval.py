"""B1 vs gold intents on the fixtures (32B): recall of gold intent spans, extra claims, drops."""
import sys
from pathlib import Path
from skillalign.intents.extract import extract_claims
from skillalign.llm import LLMClient
from skillalign.llm.templates import load_prompt
from skillalign.models import Annotation

root = Path("tests/fixtures"); client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
print("claims_b1 hash", load_prompt("claims_b1").sha256[:12])
hit = tot = kept = extra = drop = 0
def ov(a, b): return a.file == b.file and a.line_start <= b.line_end and b.line_start <= a.line_end
for gp in sorted((root / "gold").glob("*.json")):
    g = Annotation.model_validate_json(gp.read_text())
    r = extract_claims(client, root / "skills" / gp.stem)
    srcs = [s for i in r.intents for s in i.sources]
    h = sum(any(ov(s, x) for x in srcs for s in i.sources) for i in g.intents)
    # claims not overlapping any gold intent
    x = sum(not any(ov(s, y) for i in g.intents for y in i.sources) for s in srcs)
    hit += h; tot += len(g.intents); kept += len(r.intents); extra += x; drop += r.stats.dropped_bad_span
    print(f"{gp.stem[:30]:30} gold {len(g.intents)} hit {h} kept {len(r.intents)} extra {x} dropped {r.stats.dropped_bad_span}")
print(f"RECALL {hit}/{tot}  kept {kept} extra {extra} dropped {drop} (drop rate {drop/(kept+drop):.0%})")
