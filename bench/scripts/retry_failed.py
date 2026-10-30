"""Retry pre-annotations that failed with a cut-off reply: smaller skill budget, 6000 output
tokens, temperature 0.3 (a repeating loop at temperature 0 repeats forever). Seed 21."""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import skillalign.bench.preannotate as pa
from skillalign.config import get_settings
from skillalign.intents.extract import file_stem
from skillalign.llm import LLMClient

pa.PREANNOTATE_MAX_TOKENS = 6000
pa.MAX_SKILL_TOKENS = 8000
s = get_settings(); bench = s.data_dir / "bench"; pre = bench / "preannotated"
old = Path("/tmp/claude-1000/-home-jyo-Desktop-Projects-ICSE-intent/06e95e87-e659-477c-848f-fe5e0620263d/scratchpad/failed_old")
old.mkdir(exist_ok=True)
sample = json.loads((bench / "sample.json").read_text())
todo = [e for e in sample if (pre / f"{file_stem(e['skill_id'])}.failed.json").exists()
        and not (pre / f"{file_stem(e['skill_id'])}.json").exists()]


def one(e):
    f = pre / f"{file_stem(e['skill_id'])}.failed.json"
    shutil.copy(f, old / f.name)
    client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
    orig = client.structured
    client.structured = lambda m, sch, **kw: orig(m, sch, temperature=0.3, seed=21, **kw)
    f.unlink()
    o = pa.preannotate_skill(client, s.data_dir / "corpus" / e["path"], e["skill_id"], pre)
    print(o.status, e["skill_id"][-45:], (o.error or "")[:100], flush=True)


with ThreadPoolExecutor(4) as pool:
    list(pool.map(one, todo))
