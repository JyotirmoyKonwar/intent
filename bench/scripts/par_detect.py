"""Run `detect` on sampled skills with several workers; resumable; writes <out>/<stem>.json|.md.

Usage: par_detect.py WORKERS OUT_DIR START END [flags: --no-instructions --no-composite
       --raw --holistic --no-implied --model M --thinking --votes N]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from skillalign.align.pipeline import DetectOptions, detect_skill
from skillalign.align.report import render_report, report_json
from skillalign.bench.preannotate import load_sample
from skillalign.config import get_settings
from skillalign.intents.extract import file_stem
from skillalign.llm import LLMClient

a = sys.argv[1:]
workers, outname, lo, hi = int(a[0]), a[1], int(a[2]), int(a[3])
flags = set(a[4:])
model = a[a.index("--model") + 1] if "--model" in a else "Qwen/Qwen3-32B"
votes = int(a[a.index("--votes") + 1]) if "--votes" in a else 5
opts = DetectOptions(instructions="--no-instructions" not in flags,
                     composite="--no-composite" not in flags,
                     cards="raw" if "--raw" in flags else "cards",
                     judge="holistic" if "--holistic" in flags else "checklist",
                     implied_table="--no-implied" not in flags, votes=votes)
s = get_settings(); bench = s.data_dir / "bench"
out = bench / outname; out.mkdir(parents=True, exist_ok=True)
sample = load_sample(bench / "sample.json")[lo:hi]
if "--reverse" in flags:
    sample = sample[::-1]
rows = []


def one(e):
    stem = file_stem(e["skill_id"])
    if (out / f"{stem}.json").exists():
        return None
    client = LLMClient(model=model, thinking="--thinking" in flags)
    t = time.time()
    try:
        res = detect_skill(client, s.data_dir / "corpus" / e["path"], opts, skill_id=e["skill_id"])
    except Exception as err:  # noqa: BLE001
        row = {"skill_id": e["skill_id"], "error": repr(err)[:300], "secs": round(time.time() - t, 1)}
        print("ERR", row, flush=True); rows.append(row); return row
    (out / f"{stem}.json").write_text(json.dumps(report_json(res), indent=1, ensure_ascii=False))
    (out / f"{stem}.md").write_text(render_report(res))
    u = client.usage; an = res.annotation
    row = {"skill_id": e["skill_id"], "secs": round(time.time() - t, 1), "calls": u.calls,
           "cache_hits": u.cache_hits, "prompt_tokens": u.prompt_tokens,
           "completion_tokens": u.completion_tokens, "invalid": u.invalid_outputs,
           "truncated": u.truncated, "label": an.skill_label.value,
           "findings": [f.type.value for f in an.findings]}
    print(json.dumps(row), flush=True); rows.append(row)
    (bench / f"{outname}_summary.json").write_text(json.dumps(rows, indent=1))
    return row


with ThreadPoolExecutor(workers) as pool:
    list(pool.map(one, sample))
ok = [r for r in rows if "error" not in r]
print("DONE", len(ok), "ok", len(rows) - len(ok), "errors")
