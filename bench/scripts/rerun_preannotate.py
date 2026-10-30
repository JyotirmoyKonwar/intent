import json, sys
from pathlib import Path
from skillalign.config import get_settings
from skillalign.bench.preannotate import preannotate_skill, load_sample
from skillalign.llm import LLMClient

temp, seed = float(sys.argv[1]), int(sys.argv[2])
s = get_settings(); bench = s.data_dir / "bench"
sample = {e["skill_id"]: e for e in load_sample(bench / "sample.json")}
ids = [k for k in sample if k.endswith(tuple(sys.argv[4:]))]
client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
orig = client.structured
client.structured = lambda m, sch, **kw: orig(m, sch, temperature=temp, seed=seed, **kw)
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
for sid in ids:
    o = preannotate_skill(client, s.data_dir / "corpus" / sample[sid]["path"], sid, out)
    print(sid[-40:], o.status, o.error, flush=True)
