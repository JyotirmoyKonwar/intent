"""Pre-annotate the sample with several workers in parallel (vLLM batches the requests).

Resumable like `bench preannotate`: skills with an output or a .failed.json are skipped.
Usage: uv run python bench/scripts/par_preannotate.py WORKERS [START END]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from skillalign.bench.preannotate import load_sample, preannotate_skill
from skillalign.config import get_settings
from skillalign.llm import LLMClient

workers = int(sys.argv[1])
s = get_settings(); bench = s.data_dir / "bench"
sample = load_sample(bench / "sample.json")
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, len(sample))


def one(e):
    client = LLMClient(model="Qwen/Qwen3-32B", thinking=False)
    t = time.time()
    try:
        o = preannotate_skill(client, s.data_dir / "corpus" / e["path"], e["skill_id"],
                              bench / "preannotated")
    except Exception as err:  # noqa: BLE001
        print("ERR", e["skill_id"], repr(err)[:200], flush=True)
        return None
    print(f"{o.status:7} {time.time() - t:6.0f}s {e['skill_id'].split('/')[-1][:50]} {(o.error or '')[:120]}",
          flush=True)
    return o


with ThreadPoolExecutor(workers) as pool:
    outs = list(pool.map(one, sample[lo:hi]))
from collections import Counter
print("DONE", Counter(o.status for o in outs if o))
