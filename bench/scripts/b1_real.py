"""B1 on the first N sampled real skills (Qwen3-32B, thinking off). Resumable via the cache."""
import json, sys, time
from pathlib import Path
from skillalign.config import get_settings
from skillalign.intents.extract import extract_claims, save_claims, file_stem
from skillalign.llm import LLMClient

n = int(sys.argv[1])
s = get_settings()
bench = s.data_dir / "bench"
sample = json.loads((bench / "sample.json").read_text())[:n]
client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
rows = []
for e in sample:
    t = time.time()
    out = bench / "claims_32b"
    try:
        r = extract_claims(client, s.data_dir / "corpus" / e["path"], skill_id=e["skill_id"])
    except Exception as err:  # noqa: BLE001
        rows.append({"skill_id": e["skill_id"], "error": repr(err)[:200]})
        print("ERR", e["skill_id"], repr(err)[:120], flush=True)
        continue
    save_claims(r, out)
    st = r.stats
    rows.append({"skill_id": e["skill_id"], "secs": round(time.time() - t, 1), **st.__dict__})
    print(e["skill_id"][-60:], st.chunks, st.claims_kept, st.dropped_bad_span, st.failed_chunks,
          f"{time.time() - t:.0f}s", flush=True)
(bench / "claims_32b_summary.json").write_text(json.dumps(rows, indent=1))
tot = lambda k: sum(r.get(k, 0) for r in rows)
print("TOTAL chunks", tot("chunks"), "kept", tot("claims_kept"), "dropped", tot("dropped_bad_span"),
      "failed_chunks", tot("failed_chunks"), "usage", client.usage, flush=True)
