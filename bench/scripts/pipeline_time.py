"""Time the full detect pipeline on N sampled skills (Qwen3-32B, thinking off, 5 votes)."""
import json, sys, time
from skillalign.align.pipeline import DetectOptions, detect_skill
from skillalign.align.report import render_report, report_json
from skillalign.config import get_settings
from skillalign.intents.extract import file_stem
from skillalign.llm import LLMClient

n = int(sys.argv[1]); s = get_settings(); bench = s.data_dir / "bench"
out = bench / "detect_32b"; out.mkdir(parents=True, exist_ok=True)
sample = json.loads((bench / "sample.json").read_text())[:n]
rows = []
for e in sample:
    stem = file_stem(e["skill_id"])
    if (out / f"{stem}.json").exists():
        continue
    client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
    t = time.time()
    try:
        res = detect_skill(client, s.data_dir / "corpus" / e["path"], DetectOptions(), skill_id=e["skill_id"])
    except Exception as err:  # noqa: BLE001
        row = {"skill_id": e["skill_id"], "error": repr(err)[:300], "secs": round(time.time() - t, 1)}
        rows.append(row); print("ERR", row, flush=True); continue
    (out / f"{stem}.json").write_text(json.dumps(report_json(res), indent=1, ensure_ascii=False))
    (out / f"{stem}.md").write_text(render_report(res))
    u = client.usage
    a = res.annotation
    row = {"skill_id": e["skill_id"], "secs": round(time.time() - t, 1), "calls": u.calls, "cache_hits": u.cache_hits,
           "model_latency_s": round(u.latency_s, 1), "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
           "invalid": u.invalid_outputs, "truncated": u.truncated, "intents": len(a.intents), "behaviors": len(a.behaviors),
           "findings": [f.type.value for f in a.findings], "label": a.skill_label.value}
    rows.append(row); print(json.dumps(row), flush=True)
    (bench / "detect_32b_summary.json").write_text(json.dumps(rows, indent=1))
done = [r for r in rows if "error" not in r]
print("DONE", len(done), "ok", len(rows) - len(done), "errors; mean secs", sum(r["secs"] for r in done) / max(len(done), 1))
