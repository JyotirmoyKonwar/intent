"""C4 on all fixtures (32B): prose behaviors found vs gold agent-instruction behaviors."""
import json, sys
from pathlib import Path
from skillalign.behaviors import detect_behaviors
from skillalign.behaviors.prose import extract_prose_behaviors
from skillalign.llm import LLMClient
from skillalign.llm.templates import load_prompt
from skillalign.models import Annotation, Origin
from skillalign.normalize.loader import load_skill

root = Path("tests/fixtures")
client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
print("c4_prose hash", load_prompt("c4_prose").sha256[:12])
tp = fp = fn = 0
for gp in sorted((root / "gold").glob("*.json")):
    gold = Annotation.model_validate_json(gp.read_text())
    loaded = load_skill(root / "skills" / gp.stem)
    det = detect_behaviors(loaded).behaviors
    prose, st = extract_prose_behaviors(client, loaded, det)
    def covered(b):
        f = b.evidence[0]
        for d in det:
            for e in d.evidence:
                r = loaded.real_span(e)
                if r.file == f.file and r.line_start <= f.line_end and f.line_start <= r.line_end:
                    return True
        return False
    g = [b for b in gold.behaviors if b.origin is Origin.AGENT_INSTRUCTION and not covered(b)]
    def hit(p, b):
        e, f = p.evidence[0], b.evidence[0]
        return e.file == f.file and e.line_start <= f.line_end and f.line_start <= e.line_end
    matched_g = [b for b in g if any(hit(p, b) for p in prose)]
    extra = [p for p in prose if not any(hit(p, b) for b in g)]
    tp += len(matched_g); fn += len(g) - len(matched_g); fp += len(extra)
    print(gp.stem[:32], "gold", len(g), "found", len(matched_g), "extra", len(extra), "dropped", st.dropped_bad_span, "dup", st.dropped_duplicate)
    for p in extra: print("     extra:", p.capability.value, p.action[:50], "|", p.evidence[0].quote[:70])
print("TOTAL tp", tp, "fn", fn, "fp(extra)", fp, client.usage)
