# Server-hosted LLM work: plan for 2026-10-30

**Goal for tomorrow (Fri 2026-10-30, Day 4):** every step that needs the GPU box's LLM is done. That means the vLLM smoke test passes, the B1 claim extraction runs on the fixtures, and the first pre-annotation batch (about 25 skills, Qwen3-32B) is cached on disk. The offline code those steps depend on is written and tested first, so GPU time is not wasted.

**Source of truth:** [STEP_BY_STEP.md](STEP_BY_STEP.md) (Day 3 and Day 4), [IMPLEMENTATION.md](IMPLEMENTATION.md) §4 M0/M2/M5 and §6, [PLAN.md](PLAN.md). If this file disagrees with them, they win.

**Hard rules (from the project):**
- Every LLM response is cached in `data/cache/llm.sqlite`. A rerun costs nothing.
- Tests must run without a GPU. `uv run pytest` must pass at every step. It passes today: 114 tests.
- Crawled skills are untrusted. Parse them as text only. Never execute them.
- Do not touch the test set. It is frozen on Day 13 and nothing here uses it.
- Do not commit or push unless the user asks. Leave the working tree dirty for review.
- Do not copy code from Code Steward (`~/Desktop/Projects/cs_2`). Write from the design notes in IMPLEMENTATION.md §9.

---

## Current state (checked 2026-10-29)

| Item | State |
|---|---|
| `llm/client.py`, `llm/cache.py`, `tests/test_llm_client.py` | Implemented, uncommitted diff. 114 tests pass. |
| Constrained JSON, `structured()`, `vote()`, retries, truncation handling | Done. |
| vLLM on the GPU box (`localhost:8000` via SSH tunnel) | **Not reachable from the dev laptop.** `/v1/models` returns nothing on :8000. |
| ollama on the laptop (`:11434`) | Running, but **no models pulled**. Not a substitute for the server work. |
| `llm-smoke` CLI | Exists. Never run against the real server. |
| `bench/` package, `intents/`, `behaviors/`, `align/`, `llm/prompts/` | **Not created.** Day 3 items. |
| B1 claim extraction | Not written. |
| `bench/sample.py`, `bench/preannotate.py`, `bench/render.py` | Not written. |
| Status records | `STEP_BY_STEP.md` marks the GPU smoke test done. `IMPLEMENTATION.md` §4 M0 still has it open. Fix after the test passes. |

---

## Part A: User (GPU box and tunnel). Do first, in parallel with Part B

These need the user's access to the GPU box. The agent cannot do them.

1. Start the 8B server on the GPU box, with the command from [README.md](../README.md):

   ```bash
   vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384 --enable-prefix-caching --reasoning-parser qwen3 \
     --structured-outputs-config '{"backend": "xgrammar"}'
   ```

   Do **not** add `"disable_any_whitespace": true` (see README).

2. From the laptop, open the tunnel and keep it open:

   ```bash
   ssh -N -L 8000:localhost:8000 <gpu-box>
   ```

3. Check it:

   ```bash
   curl -s http://localhost:8000/v1/models
   ```

4. When the 32B model is needed (Part C), stop the 8B server and start 32B on the same port. The 96 GB card fits one 32B model in bf16 (about 64 GB). It cannot hold 32B next to the 8B unless 32B is AWQ. Note the server command in your run log. Do not guess flags. Use the README pattern and check that `/v1/models` lists the model.

Tell the agent when the tunnel is up. Parts B and C can start without it, but Part C cannot.

---

## Part B: Offline code (agent). Start now; no server needed

Each step ends with `uv run pytest` and `uv run ruff check src tests`. Tests for LLM steps use a fake transport. The client takes `transport=` for this. See `tests/test_llm_client.py` for the pattern.

### B1. Test baseline and review
- Run `uv run pytest` and `uv run ruff check`. Expect 114 passing. Stop and report if not.
- Read the uncommitted diff in `llm/client.py` and `llm/cache.py`. Confirm the truncated-reply handling in `_send` and `_repair` is as described in README. Do not change behavior.

### B2. Benchmark sample (`src/skillalign/bench/sample.py`)
- **Input:** `data/corpus/dedup.json` (canonical skills, 1,988 entries; see `data/corpus/summary.json`).
- **Output:** `data/bench/sample.json`. A list of skill IDs and paths, with a `stratum` field (`random` or `enriched`).
- **Random sample:** `random.Random(SEED)` with `SEED` from `Settings.seed` (default 13). Take 60 skills for the pilot pool, drawn from the natural-random stratum.
- **Enriched:** skills that have at least one detector hit from the corpus profile (`data/corpus/profile.jsonl`). Take 20.
- Deterministic: same seed, same output. Add a test on a small fake corpus.
- **Done when:** `skillalign bench sample` writes the file, and the test passes.

### B3. Prompt templates (`src/skillalign/llm/prompts/`)
- Add a small loader that reads `*.j2` files, returns the text, and records a version and sha256 of the template. Every `Annotation.provenance` records the prompt hash (IMPLEMENTATION.md §7).
- Add a line-numbering helper: prefix every line of a file with its 1-based number, as `spans.py` expects. Reuse the idea from Code Steward's `with_line_numbers`. Write new code.
- Templates to add:
  - `claims_b1.j2`: B1 per-chunk claim extraction. Input: one doc chunk with line numbers. Output: a JSON list of claims, each with a verbatim quote and line range.
  - `preannotate.j2`: whole skill, line-numbered, output: one `Annotation` (see `models.py`). Read `models.py` for the exact fields before writing.
- Rule from IMPLEMENTATION.md §3: the model must cite spans. The code checks them, not the model.

### B4. B1 claim extraction (`src/skillalign/intents/extract.py`)
- Input: a skill directory. Use the normalizer (`normalize/loader.py`) to get doc chunks with line ranges.
- Seed each chunk with the frontmatter `description`. Give the model the seed as "already known", so it adds only what is missing (IMPLEMENTATION.md §9). Keep prompts short for 8B.
- Call `LLMClient.structured()` with a schema for claims. Use `n=1`, temperature 0.
- **Verbatim check:** for each claim, call `spans.check_span` on its evidence. Drop claims whose quote does not match. Count the drops. Do not repair quotes.
- Output: a list of `Intent` objects (see `models.py`) for the skill, saved as JSON under `data/bench/claims/<skill-id>.json`.
- Tests with the fake transport: a good reply, a reply with a bad quote (dropped), a truncated reply (retried, not cached), and a fenced reply (parsed).
- **Done when:** the fake-transport tests pass. Real-model run is in Part C.

### B5. Pre-annotation (`src/skillalign/bench/preannotate.py`)
- Input: skill IDs from `data/bench/sample.json`. Output: `data/bench/preannotated/<skill-id>.json`, one `Annotation` per skill, with `provenance` set (model ID, prompt hash, seed, date, system name).
- **Resumable:** skip any skill whose output file already exists. Never overwrite.
- **Failure handling:** if `StructuredOutputError` is raised, write `data/bench/preannotated/<skill-id>.failed.json` with the error text. Keep going. Do not stop the batch.
- Model: pass `model=` to `LLMClient`, default `Qwen/Qwen3-32B`. The 32B ID is the default for pre-annotation (IMPLEMENTATION.md §4 M5).
- Thinking off (`SKILLALIGN_LLM_THINKING=0`), temperature 0, `max_tokens` 4096 (Annotation is larger than the default 2048).
- Validate each result with `models.py` validators and with `spans.check_annotation`. Record any span errors in the output and in the failure log. Do not drop the whole annotation for one bad span.
- Test with the fake transport: one skill passes, one fails and is logged, rerun skips the done skill.
- **Done when:** the fake-transport tests pass, and `skillalign preannotate --limit 3` runs against a fake transport in a test.

### B6. Render view (`src/skillalign/bench/render.py`)
- Markdown view of one pre-annotated skill, with line-numbered evidence, for the human check. Plain Markdown; no UI.
- **Done when:** a test renders one fixture annotation and checks that every evidence quote appears.

### B7. CLI wiring (`src/skillalign/cli.py`)
- Add commands: `bench sample`, `extract claims <skill-dir>`, `bench preannotate [--limit N] [--model M]`, `bench render <id>`. Keep the same style as the existing commands.
- **Done when:** `uv run skillalign --help` lists them, and `pytest tests/test_cli.py` passes.

### B8. Offline checkpoint
- `uv run pytest` and `uv run ruff check` pass.
- Report to the user: what was built, the list of new files, and the test count. Wait for the tunnel before Part C.

---

## Part C: Server runs (needs the tunnel from Part A)

Run these in order. Stop at the first failure and report it. Do not work around a failure silently.

### C1. Smoke test (8B)
```bash
uv run skillalign llm-smoke
```
- **Pass:** the answer is Paris, with agreement 1.00 or at least 0.67, and no error.
- **If it fails:** check the README whitespace note. Check that the server lists the model. Do not change the client to hide a parse error.
- Then update `IMPLEMENTATION.md` §4 M0 to tick the GPU item. Leave `STEP_BY_STEP.md` as is, since it already ticks it.

### C2. B1 on the fixtures (8B)
- Run `uv run skillalign extract claims tests/fixtures/skills/<each>` for all 17 fixture skills.
- **Pass:** every claim's quote is verified; invalid claims are counted; no skill crashes. Write the counts to `bench/progress.md` (create the file if missing).
- **If many quotes fail:** report the drop rate. Do not loosen `spans.py`. The loosest allowed is the existing whitespace normalization.

### C3. Pre-annotation smoke test (32B)
- Switch to the 32B server (Part A, step 4). Run `llm-smoke` again with `--model Qwen/Qwen3-32B`. Pass if it answers.
- Run `uv run skillalign bench preannotate --limit 3`. Check the three outputs by reading them. Each must have valid spans.
- Time the three skills. Record the seconds per skill in `bench/progress.md`. This feeds the throughput plan on Day 7.

### C4. First batch (Day 4 target: about 25 skills)
- Run `uv run skillalign bench preannotate --limit 25`.
- Check the failure count. If more than 3 of 25 fail, stop and report. The guideline or prompt may need work before the pilot on Day 5.
- Log in `bench/progress.md`: skills done, failures, seconds per skill, the server command used, and the commit state.

### C5. Hand-off
- Run `uv run pytest` once more on the laptop. Pass.
- Report to the user: C1–C4 results, the failure list, and the seconds per skill. Do not commit. Offer to commit when the user says so.

---

## Definition of done for tomorrow

- [ ] Part A tunnel is up, and `llm-smoke` passes on 8B (C1).
- [ ] B1 claims run on all 17 fixtures with quote checks counted (C2).
- [ ] 32B smoke test passes and 3 pre-annotated skills have valid spans (C3).
- [ ] First batch of about 25 pre-annotated skills is cached on disk, with failures logged (C4).
- [ ] `uv run pytest` and `uv run ruff check` pass (B8, C5).
- [ ] `bench/progress.md` has today's numbers.

## Out of scope for tomorrow

- Human checking of pre-annotated skills (Day 5 onward).
- The pilot revision of the guideline (Day 5).
- C4 instruction behaviors and the judge (Day 6+). They need the server too, but they are later.
- Colab training and the corpus extraction run.
- Any change to the test set.

## Risks

| Risk | What to do |
|---|---|
| 8B repetition loops on big schemas (README note) | Truncated replies are never cached and are retried with a shorter request. Count them in `usage`. Report if frequent. |
| 32B does not fit next to 8B | Run one model at a time. Use Part A step 4. |
| Annotation output too large for 4096 tokens | Log the truncations. Split the schema by intents and behaviors before increasing the limit. |
| Quote drop rate high on B1 | Report the rate. Do not loosen the checker. |
| Pre-annotation slower than planned | Record seconds per skill. Cut the batch to what finishes, and keep resumability. |
