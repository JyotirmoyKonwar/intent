# Server-LLM progress log

## 2026-10-29: C1/C2 (Qwen3-8B, GPU 0, port 8020 via tunnel to laptop :8000)

Server (college-gpu, `/scratch/user4/intent/run/serve_qwen3_8b_gpu0.sh`, copy of serve_qwen3_8b.sh with CUDA_VISIBLE_DEVICES=0):
`vllm.entrypoints.cli.main serve /scratch/user4/intent/models/Qwen3-8B --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8020 --max-model-len 16384 --gpu-memory-utilization 0.40 --enable-prefix-caching --reasoning-parser qwen3 --structured-outputs-config '{"backend": "xgrammar"}'`
Tunnel: `ssh -N -L 8000:localhost:8020 college-gpu`. Port 8000 on the box is JupyterHub's proxy.

C1 llm-smoke: Paris, agreement 1.00, invalid_outputs 0, truncated 0. PASS.

C2 extract claims, 17 fixtures: 32 kept, 6 dropped (quote not in cited lines), 0 chunks with no valid reply.
Drop rate 6/38 = 15.8%. `spans.py` not loosened.

| fixture | kept | dropped |
|---|---|---|
| 01-aligned-csv-to-json | 1 | 1 |
| 02-aligned-weather-api | 1 | 0 |
| 03-aligned-instruction-commit-msg | 3 | 2 |
| 04-ui-partial-cloud-upload | 3 | 0 |
| 05-ui-pdf-encrypt | 6 | 0 |
| 06-ub-credential-copy | 1 | 0 |
| 07-ub-exfil-crossfile | 1 | 0 |
| 08-ub-hidden-instruction | 1 | 0 |
| 09-ub-shell-persistence | 0 | 0 |
| 10-ub-npm-postinstall | 1 | 1 |
| 11-sd-destination | 2 | 1 |
| 12-sd-negative-claim | 2 | 0 |
| 13-sd-scope | 1 | 0 |
| 14-sd-condition | 2 | 1 |
| 15-aligned-dormant-code | 2 | 0 |
| 16-aligned-implied-download | 2 | 0 |
| 17-ub-inline-code-exfil | 3 | 0 |

Note: 09-ub-shell-persistence produced 0 claims (no drops, so the model returned none).
Time: 3-9 s per skill.

## 2026-10-29: C3 (Qwen3-32B, GPU 0, port 8020)

Server: `/scratch/user4/intent/run/serve_qwen3_32b_gpu0.sh`: same as the 8B command but model `/scratch/user4/intent/models/Qwen3-32B`, `--served-model-name Qwen/Qwen3-32B`, `--gpu-memory-utilization 0.85`. Weights downloaded with `hf download Qwen/Qwen3-32B` (62 GB). 8B server stopped first (`run/stop_qwen3_8b.sh`). Tunnel unchanged.

llm-smoke --model Qwen/Qwen3-32B: Paris, agreement 0.67, invalid 0, truncated 0. PASS (at threshold).

`bench preannotate --limit 3`: done 3, failed 0, 25.7 s wall (22.8 s model latency), about 7.6 s per skill, 348 completion tokens total.

**FAIL (C3 span check, vacuous):** all 3 outputs are empty annotations: `intents`, `behaviors`, `alignment`, `findings` are all `[]` and `skill_label` is "aligned". Raw cached replies contain only `skill_id`, `skill_label` and an invented `provenance` object (e.g. system "Claude 3.5 Sonnet", fake hash); the model skipped intents/behaviors entirely (93–136 completion tokens each, finish_reason stop). There are no spans to verify, so "valid spans" is true only trivially. Stopped here; C4 not run.

### C3 rerun after prompt/schema fix (same server)

Fix: `Draft` schema with five required keys (`bench/preannotate.py`); prompt `preannotate.j2` asks for exactly those keys, tells the model to list every claim and behavior, and carries a worked example. Empty outputs deleted before the rerun.

`bench preannotate --limit 3`: done 3, failed 0 (batch level), 4 calls, 1 invalid output, 1 truncated reply (retried), 8712 completion tokens, 633 s model time, 10m36s wall: **about 211 s per skill** at about 13 tokens/s (shared GPU). The earlier 7.6 s per skill was for empty replies and is not valid.

Outputs are now filled:
| skill | intents | behaviors | span errors |
|---|---|---|---|
| gtow-trainer-drill | 4 | 6 | 5 (I1, I2, I4, B1, B3) |
| implement-human-handoff | 10 | 9 | 0 |
| invalid-name-chars | 1 | 0 | 0 |

**C3 span check FAILS for gtow-trainer-drill.** The five bad quotes are model errors, not checker strictness: I1, I2 and I4 quote text that wraps over two physical lines but cite one line (e.g. I2 cites line 5, the phrase starts on line 5 and ends on line 6); B1 changes `"` to `'` and `\n` to `n`; B3 changes `"` to `'`. `spans.py` not loosened. Annotation kept for review with errors in provenance notes and a `.failed.json`.
Stopped here; C4 not run.

### C3 rerun 2 (prompt: quotes inside one line, character for character)

Only gtow-trainer-drill redone (other two kept; they carry the previous prompt hash). 1 call, 1958 tokens, 140 s, 0 invalid, 0 truncated.
**Span errors went up, 5 to 9** (I2, I3, I5, B1, B2, B4, B5, B6, B9). Typical errors: I2 quotes "build precise GTOW Trainer drill deep-links", which spans the line 4/5 wrap; I5 still writes "..." for omitted text; B1/B2 cite lines 81-84 with a quote that changes `"` to `'`; B4/B5/B6/B9 quotes are paraphrases of the text. The prompt rule did not hold for Qwen3-32B without thinking. `spans.py` not loosened. C4 not run.

### C3 rerun 3: model gives line ranges only, code copies the quotes

Change: `Draft` spans carry `file`, `line_start`, `line_end`; `bench/preannotate.py` fills each quote from the cited lines (up to 300 chars). Prompt no longer asks for quotes. Both models of failure seen before (joined wrapped lines, swapped quote marks, "..." and paraphrase) are gone by design. All 3 outputs deleted and redone under one prompt hash.
`--limit 3`: 3 done, 0 failed, 0 span errors, 407 s model time (about 136 s per skill), 5653 completion tokens. Intents/behaviors: 5/18, 10/9, 2/0.

## 2026-10-29: C4 (Qwen3-32B, `--limit 25`)

- First run: 21 annotations in 51m45s, then the 22nd request hit the 300 s client timeout (`APITimeoutError`) and aborted the batch. Fixed: `preannotate_skill` now catches `openai.APIError`, writes `.failed.json`, and the batch goes on (test added). Resumed with `SKILLALIGN_LLM_TIMEOUT_S=900`.
- Resumed run: 4 done, 21 skipped, 0 failed; 9 calls, 5 invalid outputs, 5 truncated replies (the repetition loops from the README, retried with shorter output), 32127 completion tokens, 42m04s wall (about 630 s per skill for these four).
- **Result: 25 of 25 skills annotated, 0 lost.** 147 intents, 180 behaviors in total.
- **Span errors: 2 of 25** (`Kernel8901/.../quality-controller`, `vallenmulia99/.../design-md`), 1 each, kept in the annotation and logged in `.failed.json`. Below the 3-of-25 stop threshold.
- Speed: about 4 minutes per skill wall-clock overall (about 100 minutes for 25 skills including one wasted timeout), 12-13 tokens/s on a GPU shared with other jobs. Skills that hit repetition loops cost 10 minutes. The 25 skills on disk in `data/bench/preannotated/`.
- Server command: see the C3 section (`serve_qwen3_32b_gpu0.sh`, GPU 0, port 8020, util 0.85). Commit state: nothing committed; working tree dirty.

## C5 (2026-10-29)

`uv run pytest`: 132 passed. `uv run ruff check src tests`: clean.

## 2026-10-29 (late): Day 4 + 5 engineering, M3 detectors (offline)

New package `src/skillalign/behaviors/` (`facts`, `patterns`, `pyflow`, `shell`, `js`, `config`, `hidden`, `detect`) and CLI `skillalign behaviors <skill-dir> [--json]`. Tests: `tests/test_behaviors.py`.
- Python: `ast` detector with constant resolution, call graph and entry points, def-use taint with per-function summaries across files (exfiltration, credential copy, download-and-execute composites).
- Shell: pipelines, redirections, substitutions, persistence, `curl | sh`, upload of private files. JS: regex level (no data flow). Config: npm install hooks, URL dependencies, `.mcp.json`, agent hooks, broad `allowed-tools`.
- Hidden scan: zero-width, bidi and tag characters, encoded blobs, invisible HTML, HTML comments that address an agent (commands inside them become behaviors).
- Dormant rule: a Python function nothing reaches is `dormant`; a fenced block is `entry` only when the text before it tells the agent to run it (shell blocks with no commentary count as runnable), otherwise `dormant`.
- Gold check on the 17 fixtures: 49 of 51 gold behaviors found with matching capability and evidence lines. The 2 misses are fixture 03, whose behaviors exist only in prose (C4 LLM instruction behaviors, Day 6). 5 extra detections, all true operations the gold does not list (e.g. `grep ~/.bashrc`, `mkdir`).
- Corpus smoke test: 300 skills, 0 crashes, 9 s total. Noise cut after a first pass (shell function names and bash syntax no longer reported as PROC_EXEC; an API key in an HTTP header no longer makes an exfiltration composite). Remaining known noise: `ssh $TOKEN@host` reads as an exfiltration composite; PROC_EXEC is still the most common capability (494 of about 1,000 behaviors in 300 skills).
- Not done: JS data flow, download-then-chmod-then-run across statements, C4 prose behaviors, cards. `pytest -m fixtures` per-type F1 needs the judge (M4).
Suite: 167 passed, 1 skipped; ruff clean. Nothing committed.

## 2026-10-30: server work on Qwen3-32B (port 8000 tunnel, `SKILLALIGN_LLM_THINKING=0`) and offline M4/M6/eval build

Scripts for everything below are in `bench/scripts/` (run with `uv run python bench/scripts/<name>.py`; paths in them assume the repo root).

### Pre-annotation pool
- Another session's `bench preannotate --limit 80` (pid started before this one) was already running, so I did not start a second writer. At the last check 58 of the 80 skills had annotations (the rest still running).
- **Span errors: 5 skills, all cleared.** `Kernel8901/.../quality-controller` and `vallenmulia99/.../design-md` (the two named earlier), plus `separateweb-capture`, `facial-and-lipsync` and `aw2exec-got-plt` from the newer batch. A rerun at temperature 0 returns the cached (bad) reply, so the first four were redone at temperature 0.3, seed 21 (provenance records seed 21 and says why); all four came back with 0 span errors. `aw2exec-got-plt` was a different bug: the file is `SKILL.MD`, the model cited `SKILL.md`. Fixed in code (`canonical_files` in `bench/preannotate.py`, unambiguous case-insensitive match only; test added) and the cached reply now validates. Old files are kept outside the repo tree for comparison.
- `check_preannotated.py` over the sample: 58 annotations, 0 with span errors.

### Prompt hashes (sha256 prefix; log a new line whenever a prompt changes)
| prompt | hash | note |
|---|---|---|
| claims_b1 v1 | 3642797875c2 | recall 20/23 gold intents on the fixtures, 5 extra claims, quote-drop 2% (32B) |
| claims_b1 v2 | 0896e3247576 | in use. Recall 22/23, 9 extra claims, drop 5% (32B). v1 returned `{"claims": []}` for plain description lines like "Get the current weather for a city" |
| c4_prose v1 | 255296225d38 | recall 7/7 gold prose behaviors but 19 extras: "Run `python scripts/x.py`" on every skill, and third-person descriptions ("Reads logs…") read as instructions |
| c4_prose v2 | 1d5881d4d185 | in use, with a code filter for "run the skill's own script" and USER_INTERACTION treated as implied. Targets (fixture 03) 2/2, extras 4 (one is a hidden-comment line the detector already covers, now deduplicated by span) |
| preannotate | see C3/C4 sections above | not changed today |

### B1 on real skills
- Fixtures with 32B: quote-drop 2% (v1) and 5% (v2), against 15.8% with the 8B model on 2026-10-29. So the 16% figure was a property of the 8B model, not of the checker.
- Real text is worse: the first real skills drop 2/6, 5/9, 7/40 claims. Looking at the drops (`maintenance-cleanup`): the model quotes the right text with the wrong line number, or strips Markdown emphasis (`**Telegram** cache` quoted as `Telegram cache`). Added `relocated` handling in `intents/extract.py`: if the quote is verbatim elsewhere in the same chunk, cite where it is (the checker still verifies the quote; `spans.py` not loosened). Test added. The Markdown-emphasis drops stay dropped.
- A 30-skill run with the pre-relocation code and prompt v1 was started in the background (`data/bench/claims_32b/`, summary at `data/bench/claims_32b_summary.json` when it finishes). That run is the "does 16% hold" measurement; rerun with v2 + relocation afterwards for the comparison.

### Judge smoke test (fixture 11, 32B, 5 votes)
`skillalign detect tests/fixtures/skills/11-sd-destination`: 53 s, 18 calls, 1301 completion tokens, 0 invalid. Result: UB for the paste-service upload instead of the gold SD(destination). Cause is upstream: B1 attached "save it locally" to the wrong intent, so the intent that the judge saw had no "locally" constraint. The code path itself works (card ids checked, votes agree 1.00). Before this C4 also added three false prose behaviors on this fixture; fixed by c4_prose v2.

### Throughput (50 skills)
Started in the background (`bench/scripts/pipeline_time.py`, results in `data/bench/detect_32b_summary.json` and reports in `data/bench/detect_32b/`). The GPU is shared with the 80-skill batch, so these numbers are pessimistic. First skills: 2.6 s to 92 s per skill for small skills (several calls cached from the B1 run). Final numbers to be filled in when the run finishes.

### Offline build (no server needed)
- `eval/` (match, metrics, stats): strict and lenient matching, per-type and skill-level P/R/F1, confusion matrix, control FPR, line IoU, SD-axis accuracy, bootstrap CIs, exact McNemar. Gold against itself scores 1.0 on all 17 fixtures.
- `intents/vocab.py`: 65-verb action vocabulary with expected capabilities (B2) and a keyword category classifier (v1).
- `align/`: cards (cards and raw modes), D1 candidates, D2 checklist judge with card-id check and voting, holistic variant, D3 implied table, D4 rules, risk, JSON and Markdown report, pipeline with all ablation flags. CLI: `skillalign detect|baseline|eval`, `skillalign bench inject`.
- `behaviors/prose.py`: the C4 prose-behavior extractor (found both fixture-03 misses with 32B).
- `baselines/`: R2 (rules) and M1 (similarity, lexical embedder, `tune` for dev thresholds). Both output `Annotation`.
- `bench/inject.py`: DROP_IMPL, DROP_VARIANT, ADD_SINK_SPLIT, ADD_INSTRUCTION, RETARGET, NEGATE_CLAIM and the VAGUE_DOC control, with train and test templates, canary hosts only. Every output gold validates against the mutated files, and the detectors find the injected sinks (train: urllib + base64 across three files; test: requests + hex across a subpackage).
- Known detector gap found while writing the test templates: `http.client.HTTPSConnection(host).request("POST", ...)` loses the host (the first argument of `.request` is read as a URL, giving destination "POST"). Left as is; the test template uses `requests.post` instead.
- Suite: 241 tests (`uv run pytest`), `ruff check` clean. `pytest -m fixtures -s` runs the full pipeline on the 17 fixtures against the server (not part of the default run).

### Update (2026-10-30, later)
- **B1 on 30 real skills (Qwen3-32B, prompt v1, no relocation): 501 kept, 77 dropped = 13.3% drop rate** (540 chunks, 0 chunks with no valid reply, 0 invalid, 5074 s model time). So the 16% from the 8B fixture run holds roughly on real text, and the 32B fixture figure (2%) does not carry over. Outputs in `data/bench/claims_32b/`.
- **Pipeline bug found by the 26-skill timing run:** 272 UI findings, 22 of 26 skills misaligned. Cause: claims that no detector can verify (guidance prose, DATA_TRANSFORM/USER_INTERACTION-only verbs, skills with no code) had no candidate behaviors and so became UI by default. Fixed in `align/pipeline.py` (`verifiable`: UI needs code in the skill and an expected capability the detectors can report; test added). Old results kept in `data/bench/detect_32b_v1/`. The 50-skill run was restarted with the fix (`data/bench/detect_32b/`); its numbers replace the first-run timing (mean 181 s per skill, median 77 s, max 1017 s, on a GPU shared with the 80-skill batch).
- Preannotated skills on disk: 72.

## 2026-10-30 night: unattended server run (Qwen3-32B, thinking off), final numbers

**Pre-annotation pool:** 77 of 80 skills annotated. The 8 earlier failures were 5 prompts over the 16k context (CJK-heavy skills; fixed with a token estimate, `MAX_SKILL_TOKENS` in `bench/preannotate.py`, test added) and 3 outputs cut at 4096 tokens. After the retry 5 were done and 3 still fail with truncated output (`manim-video`, `blender-kiln/plugin`, and one more; see the `.failed.json` files): they need a hand pass or a split prompt. One finished with a span error (`impeccable`, I14, an unresolved line), logged in its `.failed.json`.

**B1 on 30 real skills, three prompt versions (same skills, 32B):**
| prompt | hash | kept | dropped | relocated | drop rate |
|---|---|---|---|---|---|
| v1 | 3642797875c2 | 501 | 77 | 0 (not built yet) | 13.3% |
| v2 | 0896e3247576 | 1294 | 312 | 70 | 19.4% |
| v3 (in use) | a4b4fc40927b | 1526 | 167 | 84 | 9.9% |
v3 adds "keep Markdown symbols exactly, quote only the few words that hold the claim". On fixtures v3 keeps gold-intent recall at 22/23 with 1 drop. v2/v3 list far more claims than v1 (about 50 per skill for v3), which is a recall gain but also what drives the UI count below.

**Pipeline throughput, 50 skills (`detect`, 32B, 5 votes, GPU shared with other jobs):** 0 errors; mean 409 s per skill, median 230 s, p90 1082 s, max 2077 s; 5,901 calls and 493k completion tokens in total (19,423 s of model time). With the GPU to itself this would be lower; plan the corpus run (Day 12) around a few hundred seconds per skill per 32B, or use 8B for extraction. Findings over the 50 skills (before the vocabulary fix below): UI 304, UB 58, SD 31, UI-partial 20; 31 skills misaligned.
**Known precision problem:** of the 304 UI findings, 260 were "judge said no" on candidates that exist and 44 had no candidate. Many come from reference-style skills (for example a three.js doc skill) where B1 turns library documentation into claims ("creates SkeletonHelper"). Vocabulary fixes made today (file verbs need a file-like object; "run `python scripts/x.py`" is not a claim) cut the fixture false positives, but the 50-skill numbers above were not re-run. Needs your look at a sample of UI findings in `data/bench/detect_32b/*.md`.

**Full pipeline on the 17 fixtures (`pytest -m fixtures -s`, after the fixes):** skill-level P 0.92 R 0.92 F1 0.92. Strict per type: UI P 0.33 R 1.00, UI-partial 0, UB P 0.67 R 1.00 F1 0.80, SD P 0.50 R 0.25; macro F1 0.41. Lenient: macro 0.54, UB F1 0.92, SD 0.75. Before the fixes: skill F1 0.88, macro 0.35. Remaining misses: fixtures 11 and 12 (SD reported as UB because the judge does not cite the offending card with an axis), 14 (SD condition seen as UI-partial), 04 (UI-partial missed), 03 (a false SD).

**L1-32B on the fixtures (the pre-annotator as a system):** skill-level F1 1.00; strict macro F1 0.49 (UI 0.67, UI-partial 0.29, UB 0.60, SD 0.40), lenient macro 0.57. So on 17 easy fixtures the single-prompt 32B is on par with the full pipeline at the finding level; the fixtures are too small and too easy to separate them. Predictions in `data/bench/l1_fixtures/`.

Suite: 245 tests pass, ruff clean. Nothing committed.

**vLLM stopped on college-gpu at the end of the run** (Qwen3-32B, pid 17799; GPU0 memory went from 89.6 GB to a few GB). Restart with `/scratch/user4/intent/run/serve_qwen3_32b_gpu0.sh` and the tunnel from the C3 section. Note `run/stop.sh` does not match this server's command line; it was stopped with `pkill -f 'vllm.entrypoints.cli.main serve .../Qwen3-32B'`.

## 2026-10-30 afternoon/evening: last server day (Qwen3-32B restarted 12:12, GPU 0 only)

**Why this day:** the GPU box is unavailable for about 7-10 days from 2026-10-31, so everything that needs the model was run today and cached in `data/cache/llm.sqlite`.

- **Sample widened from 80 to 220 skills** (100 natural-random, 120 enriched). `extend_sample` in `bench/sample.py` appends and leaves the first 80 entries untouched (checked). This is the pool the hand checks draw from (PLAN.md §5.7: 180-300 skills).
- **Pre-annotation: 220 of 220 annotated** (Qwen3-32B, thinking off, 8 workers in parallel via `bench/scripts/par_preannotate.py`, about 110-210 s per skill, 3.5x faster in wall time than one at a time). 12 skills first failed with replies cut at 4096 tokens; `bench/scripts/retry_failed.py` re-ran them with `MAX_SKILL_TOKENS` 8000, 6000 output tokens, temperature 0.3, seed 21 (recorded in provenance); all 12 succeeded. **10 annotations still cite a line that does not exist** (`[unresolved span]`, plus one skill with two missing script files); `check_preannotated.py` lists them. The reviewer fixes those by hand.
- **Full `detect` pipeline (32B, 5 votes, default options) on the sample: reports in `data/bench/detect_32b_v2/`** (`.json` and `.md`, one per skill; `par_detect.py`, up to 34 workers at once). **216 of 220 done.** The 4 without a report are the largest skills; each ran for hours and was stopped at 22:15: `Research-Agent/.../impeccable`, `elithril/blender-kiln/plugin`, `vallenatrix/.../popular-web-designs`, `zuwasi/wazatator/.squad`. They need a smaller claim budget or a hand pass.
- **Not run (decided with the user): the five ablations.** They stay a Day 11 job and need the server back.
- **Not run: L1 with Qwen3-8B, the 4B/8B/32B grid, the test-set evaluations, the large corpus run.** Reasons: the test set does not exist yet (frozen Day 13), nothing is hand-labelled so there is no dev set to score against, and the 8B/4B runs need the 32B server swapped out. L1-32B on the sample is the pre-annotation itself (`baselines/l1.py` reuses it).

**Fix needed before the next server session (does not need the server):** the pipeline reports far too many findings on real skills. Over the first 212 reports: 166 skills misaligned, 14.1 findings per skill, UI 2052, UI-partial 147, UB 483, SD 307. That is the known precision problem (B1 v3 lists about 50 claims per skill; each one the judge doubts becomes a UI). Judge answers are cached, so changing only the decision rules (a vote threshold, dropping claims that no detector can verify, limiting UI to claims with a high-confidence expected capability) can be re-scored with no model calls. The first hand-checked skills will show which rule to use.

**Cache locking:** with 30+ workers at once SQLite raised `database is locked` (6 skills lost, re-run later). `llm/cache.py` now uses WAL, a 120 s busy timeout and a retry on lock; 20 LLM-client tests pass.

**Final `detect` numbers (216 skills, default options):** 46 aligned, 170 misaligned. Findings: UI 2845, UB 561, SD 432, UI-partial 175 (18.6 per skill on average, median 6). The tail of very large skills drives the mean. Treat these as the unfiltered output: the decision rules above still need tightening before any precision number means anything.

**Closed down at 22:20:** `detect` processes stopped, vLLM on college-gpu stopped, tunnel closed. `pytest` 245 passed, 1 skipped; `ruff check src tests` clean. Nothing committed. To resume: `/scratch/user4/intent/run/serve_qwen3_32b_gpu0.sh` and `ssh -N -L 8000:localhost:8020 college-gpu`. All responses from today are in `data/cache/llm.sqlite`, so re-running `par_detect.py` for the 216 finished skills costs no model calls.
