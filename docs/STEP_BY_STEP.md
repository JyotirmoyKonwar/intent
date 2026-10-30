# SkillAlign: Day-by-Day Step-by-Step

A working checklist for the 17 days to the solution paper deadline (**Fri 2026-11-13 AoE**). It expands the schedule in [IMPLEMENTATION.md §6](IMPLEMENTATION.md) into concrete steps, with the command or file each step touches and what "done" looks like. The design is in [PLAN.md](PLAN.md) and the milestones are in [IMPLEMENTATION.md](IMPLEMENTATION.md). If this file and those two disagree, they win; update this file to match.

Status key: `[x]` done, `[ ]` open, `(stretch)` only if the day has slack. Cut order is in IMPLEMENTATION.md §6.

---

## Day 1: Tue 2026-10-27. Foundations (M0)

Goal: contracts, LLM client, and fixtures work with no GPU.

- [x] Set up the repo: `uv sync`, `ruff`, `pytest`, package layout under `src/skillalign/`.
- [x] Freeze data contracts (`models.py`) and config (`config.py`).
- [x] Span checker (`spans.py`): every quote and line range must match the file on disk.
- [x] LLM client (`llm/client.py`): OpenAI-compatible, JSON-schema output, retries, majority vote over `n` samples, sqlite cache.
- [x] Fixture skills with gold labels in `tests/fixtures/`.
- [x] CLI: `schema`, `validate`, `llm-smoke`.
- [x] On the GPU box: start vLLM with Qwen3-8B and run `uv run skillalign llm-smoke`.
- [x] Commit the working state (the tree has uncommitted changes in `.gitignore`, `README.md`, `config.py`, `test_collect.py`, and `.env.example`).

**Done when:** `uv run pytest` passes without a GPU, and the smoke test passes on the GPU box.

---

## Day 2: Wed 2026-10-28. Crawl, normalizer, guideline

**Engineering**

- [x] Run the GitHub crawl: `uv run skillalign collect github --max-repos 1500` (needs `GITHUB_TOKEN` in `.env`). Target 2–5k canonical skills. On disk: 8,455 GitHub and 98 official skill folders. After filter and dedup (`uv run skillalign corpus`): 1,988 canonical skills (1,891 GitHub, 97 official), kept as is, just under the 2k floor. Numbers are in `data/corpus/summary.json` and `data/corpus/dedup.json`.
- [x] Official repos are collected (98 skills, 97 after dedup).
- [x] Normalizer: `uv run skillalign normalize <skill-dir>`.
- [x] Check the normalizer on 5 real skills. Read the output by hand. (Done on 7 skills; see `docs/guideline.md` §5 for the two normalizer gaps found.)

**Benchmark**

- [x] Write the first draft of `docs/guideline.md`: definitions, decision procedure, implied-behavior table, edge cases.

**Done when:** the crawl has finished or its target is set, and the guideline draft exists.

---

## Day 3: Thu 2026-10-29. Claims, vocabulary, sampling

**Engineering**

- [x] B1: per-chunk claim extraction from SKILL.md and README, with a verbatim quote check using `spans.py`.
- [x] B2: action vocabulary (about 40 verbs mapped to expected capabilities).
- [x] Category classifier (v1).

**Benchmark**

- [x] `bench/sample.py`: random sample plus an enriched sample.
- [x] `bench/preannotate.py`: Qwen3-32B reads a whole skill with line numbers and outputs an `Annotation`.
- [x] `bench/render.py`: Markdown view for review.

**Done when:** B1 runs on the fixtures and the three bench scripts run on 3 skills.

---

## Day 4: Fri 2026-10-30. Detectors, first pre-annotation batch

**Engineering**

- [x] M3 detectors for Python (`ast`), shell, and config files (dependencies, postinstall, hooks, `.mcp.json`, `allowed-tools`).
- [x] Hidden-content scan.
- [x] Check the normalizer's dormant-code decision (IMPLEMENTATION.md §4, M2): a fenced block is reachable only when the text tells the agent to run it.

**Benchmark**

- [x] Pre-annotate the first batch on the GPU box with `bench/preannotate.py`. Cache every response.

**Done when:** detectors pass their fixtures, and the first batch is pre-annotated.

---

## Day 5: Sat 2026-10-31. Call graph, data flow, pilot

**Engineering**

- [x] Import and call graph, and reachability from entry points.
- [x] Def-use data flow inside functions, plus call-argument flow.
- [x] Composite patterns: exfiltration, download-and-execute, persistence, credential harvesting.

**Benchmark**

- [ ] **Pilot on 20 skills.** Revise the guideline and the pre-annotation prompt from what the pilot shows.
- [ ] Anchoring set: label 20 skills by hand from scratch before reading any LLM output.

**Done when:** the composite fixtures pass, and the guideline has a v2.

---

## Day 6: Sun 2026-11-01. Cards, judge, check pace starts

**Engineering**

- [x] C4 instruction behaviors (LLM) and cards.
- [x] M4 judge (code and fake-LLM tests; not yet run on the server): D1 candidates, D2 checklist judge (n=5 votes, card IDs required), D3 implied table, D4 rules, risk, and JSON plus Markdown report.

**Benchmark**

- [ ] Start checking pre-annotated skills at about 25 per day. Log the count in `bench/progress.md` each day.

**Done when:** the judge returns a report for one fixture skill.

---

## Day 7: Mon 2026-11-02. End-to-end and injection

**Engineering**

- [ ] **M4 end to end on the fixtures.** `skillalign detect` exists and passes on scripted-judge tests; still to run on the server for all 17 fixtures.
- [ ] Throughput check: time the whole pipeline on 50 skills. This sets the corpus plan.
- [x] Add the ablation flags from the start: `--no-instructions`, `--no-composite`, `--cards raw`, `--judge holistic`, `--no-implied-table`, `--votes`, `--model`.

**Benchmark**

- [x] `bench/inject.py`: the six operators from PLAN.md §5.3 plus `VAGUE_DOC`. Separate templates for train and test. Canary hosts only (`canary.local`).

**Done when:** `pytest -m fixtures` prints per-type F1, and the throughput number is written down.

---

## Day 8: Tue 2026-11-03. Baseline R2 and M1, eval harness

**Engineering**

- [x] M6 baseline R2 (rules).
- [x] M6 baseline M1 (similarity). Tune the threshold on dev only.
- [x] `eval/match.py`: strict and lenient matching (PLAN.md §8.1).

**Benchmark**

- [ ] Check today's pre-annotated skills (target about 25).

**Done when:** R2 and M1 output `Annotation` objects that the eval harness can score.

---

## Day 9: Wed 2026-11-04. L1 baselines, first dev results

**Engineering**

- [ ] M6 baseline L1 (zero-shot Qwen3-8B) and L1-32B.
- [ ] First dev-set results: per-type metrics, bootstrap CIs, McNemar tests. (Harness `eval/` is built and tested on the gold labels; no dev results yet.)

**Benchmark**

- [ ] Check today's skills.

**Done when:** the first dev results table exists, even if rough.

---

## Day 10: Thu 2026-11-05. Fix the worst errors

**Engineering**

- [ ] Look at the dev errors. Fix the worst error classes, starting with the most frequent. Do not add new features.

**Benchmark**

- [ ] Check today's skills.

**Done when:** the dev F1 has moved, or the errors are written down with a reason they stay.

---

## Day 11: Fri 2026-11-06. Ablations on dev

**Engineering**

- [ ] Run the five must-ship ablations on dev: `--no-instructions`, `--no-composite`, `--cards raw`, `--judge holistic`, `--no-implied-table`.
- [ ] (stretch) LoRA judge on Colab A100. Needs the injected-train data from day 7. Drop it if the day has no slack.

**Benchmark**

- [ ] Finish the remaining checks.

**Done when:** the ablation table on dev is complete.

---

## Day 12: Sat 2026-11-07. Corpus run and final checks

**Engineering**

- [ ] Corpus run (RQ1): a few thousand skills, with 50 findings checked by hand.
- [ ] Prevalence from the natural-random sample only.

**Benchmark**

- [ ] Finish all checks. The annotation pace is the biggest schedule risk: if the checks are behind, use the cut order in IMPLEMENTATION.md §6.

**Done when:** the corpus prevalence numbers and the 50 hand-checked findings are recorded.

---

## Day 13: Sun 2026-11-08. Freeze the test set, run all test evaluations

- [ ] **Freeze the test set** (repo-level split, about 30% dev and 70% test). After this, no changes to the test set.
- [ ] Run all test evaluations: main table (RQ2), ablations (RQ3), the 4B/8B/32B × thinking grid (RQ4), and cost per skill (RQ5).
- [ ] Compute the bootstrap CIs and McNemar tests on the test results.
- [ ] Make the figures and tables (PLAN.md §9.4).
- [ ] Start the paper: method and benchmark sections.

**Done when:** all test numbers are in `results/`, and the frozen split is recorded with its hash.

---

## Day 14: Mon 2026-11-09. Fill gaps

- [ ] Fill any missing numbers or ablation cells.
- [ ] Paper: evaluation section.

**Done when:** every table cell in the paper has a number.

---

## Day 15: Tue 2026-11-10. Paper text

- [ ] Paper: introduction, related work, threats to validity.

**Done when:** the paper has a complete draft with no placeholders.

---

## Day 16: Wed 2026-11-11. Artifact and full draft

- [ ] Anonymize the artifact repo: remove names, paths, and tokens. Keep the neutralized benchmark, the guideline, the prompts, and the cached responses.
- [ ] Full draft, then revise. Check the length: 5–10 pages IEEE plus 2 reference pages.
- [ ] Double-anonymous check: no author names or identifying URLs in the paper.

**Done when:** the artifact runs from a clean clone, and the paper is within the page limit.

---

## Day 17: Thu 2026-11-12. Polish and submit

- [ ] Polish the paper and check the references.
- [ ] Submit to HotCRP before **Fri 2026-11-13 AoE**. Do not leave the submission to the last hour.

**Done when:** the HotCRP submission is confirmed.

---

## Rules that apply every day

- Log the day's progress in one line: skills checked, milestone steps done, blockers.
- Cache every LLM response. A rerun must cost nothing.
- Crawled content is untrusted: parse it as text and ASTs only, and never run it.
- Tests must run without a GPU.
- Do not change the test set after day 13.
- If a day slips, use the cut order before adding time to the next day.
