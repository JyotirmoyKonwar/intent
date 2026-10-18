# SkillAlign: Implementation Plan

This document covers how we build what [PLAN.md](PLAN.md) describes: the stack, repository layout, data contracts, milestones, budgets and timeline. Section numbers such as §6.3 point to PLAN.md.

**Assumptions:** about 12 weeks (deadline around 2027-01-11, to be confirmed), 3–4 people, a tight API budget, and small local models (Qwen3 4B–8B) as the main engine.

---

## 1. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language and environment | Python ≥3.13, `uv` | Already set up in this repo |
| Data models | `pydantic` v2 | One schema for extractor output, annotations and reports |
| CLI | `typer` | `skillalign crawl / normalize / extract / detect / eval / train` |
| Parsing | `tree-sitter` + `tree-sitter-language-pack`, `python-frontmatter`, `markdown-it-py`, `bashlex` | Works across languages, gives exact line numbers |
| Call graph and data flow | Our own tree-sitter resolver; `semgrep` taint mode as an adapter; `bandit` | No single tool covers every language a skill can contain |
| Rules | Rules from Cisco skill-scanner (YARA/YAML), Semgrep registry packs | Reuse existing rules instead of writing them from scratch |
| Graphs | `networkx` | Behavior graph, path enumeration |
| **Local LLM serving** | **`vllm`** (OpenAI-compatible server) with **guided JSON decoding** (xgrammar); fallback `llama.cpp`/`ollama` for laptops | Constrained output is necessary for 4–8B models |
| Models | Qwen3-8B, Qwen3-4B, Qwen3-Embedding-0.6B | §6.4 |
| Fine-tuning | `peft` LoRA + `trl` SFT (or `unsloth` to save memory) | Trains SkillAlign-Judge on one GPU |
| API (sparing use) | One OpenAI-compatible or Anthropic client behind the same interface | For silver labels and the L3 reference only |
| Caching | `diskcache`, keyed by hash(prompt, model, params, seed) | Reruns are reproducible and cost nothing |
| Sandbox (optional) | Docker, `mitmproxy`, `strace`, `inotifywait` | C6 |
| Annotation UI | `streamlit` | Linking intents to behaviors needs a custom UI |
| Experiments | `pandas`, `scikit-learn`, `scipy` (bootstrap, McNemar), `matplotlib` | |
| Quality | `pytest`, `ruff`, `pyright`, GitHub Actions | |

### Compute
- **Inference:** Qwen3-8B in bf16 needs about 16 GB of VRAM, so one 24 GB GPU is enough. An AWQ/GPTQ 4-bit build runs on 12 GB. Qwen3-4B runs on 8–12 GB.
- **LoRA fine-tuning of 8B:** one 24 GB GPU in 4-bit QLoRA, or a 40–80 GB GPU in bf16. Expect a few hours for each run of 2–5k examples.
- **Corpus run (RQ1):** on the order of 10–20k skills with batched vLLM. **Measure throughput on 50 skills in week 6** before committing to running on the full corpus. Most of the work is symbolic, and the LLM only sees cards.

> **Action item:** list the GPUs the team actually has (model, VRAM, and whether they are shared) before week 1 ends. This decides between bf16 and quantized models and whether QLoRA is needed.

### API budget plan
| Use | Size | When |
|---|---|---|
| Prompt and schema prototyping (before switching to local) | Small, a few dozen skills | Weeks 2–4 |
| **Silver labels** for judge training | About 1–2k (intent, cards) checklist items from *non-benchmark* corpus skills; cards are short, so tokens per item are small | One batch in week 7 |
| **L3 frontier reference** baseline | Test set only, about 280 real skills plus a subset of the injected test set | One run in week 9 |

**Pilot first:** run 20 items, measure the tokens, and extrapolate before each batch. Cache every response, because we cannot afford to rerun. If the budget is very tight, drop L3 on the injected set first, then reduce the silver labels to about 500 items and rely more on injected training data.

---

## 2. Repository layout

```
intent/
├── pyproject.toml
├── docs/                      # challenge PDF, PLAN.md, IMPLEMENTATION.md, guideline.md, schema/
├── src/skillalign/
│   ├── models.py              # pydantic: Skill, FileArtifact, Span, Intent, Behavior, Card, Finding, Report
│   ├── config.py              # paths, model endpoints, thresholds, seeds
│   ├── llm/
│   │   ├── client.py          # call(prompt, schema, n) → validated objects; vLLM guided-JSON / API
│   │   ├── cache.py
│   │   └── prompts/           # versioned *.j2 templates (extract_claims, merge_intents,
│   │                          #   instructions, checklist_judge, implied_check, holistic_baseline)
│   ├── collect/               # official.py github.py clawhub.py skillssh.py dedup.py history.py
│   ├── normalize/             # loader.py markdown.py languages.py
│   ├── intents/               # extract.py canonicalize.py (action vocab) consolidate.py conflicts.py
│   ├── behaviors/
│   │   ├── taxonomy.py        # capability enum + API→capability maps
│   │   ├── detectors/         # python.py js.py shell.py config.py hidden_content.py
│   │   ├── callgraph.py  reachability.py  dataflow.py
│   │   ├── composite.py       # pattern library
│   │   ├── instructions.py    # C4
│   │   └── dynamic/           # optional C6
│   ├── cards.py               # Behavior → evidence card text (templates)
│   ├── align/                 # candidates.py judge.py implied.py decide.py risk.py
│   ├── report/                # json + markdown/html
│   ├── train/                 # build_dataset.py (injected + silver → SFT jsonl), sft.py, eval_judge.py
│   ├── baselines/             # cisco.py rules.py static.py similarity.py llm_zeroshot.py multiagent.py
│   ├── bench/                 # schema.py sample.py triage.py inject.py (operators) annotate_app.py agreement.py
│   ├── eval/                  # match.py metrics.py stats.py run.py tables.py figures.py
│   └── cli.py
├── data/                      # gitignored: raw/ corpus/ bench/ runs/ train/ cache/
├── tests/fixtures/skills/     # 20–30 tiny hand-written skills with gold labels
└── scripts/
```

---

## 3. Data contracts (frozen in week 1)

Every stage reads and writes JSON that is validated by `models.py`, so people can work on stages in parallel against fixtures.

- **`Span`**: `file, line_start, line_end`. All evidence anywhere in the system is a `Span`. **Objects without valid spans are rejected.**
- **`Skill`, `FileArtifact`** (role, language, sha256; virtual files keep a parent path and line offset).
- **`Intent`, `Behavior`**: PLAN.md §4.2 and §4.3.
- **`Card`**: `behavior_id, text, snippet, span, chain`. This is the only view of code the LLM gets.
- **`ChecklistAnswer`**: `achieves_action, constraints[], axis_mismatch, contradicts_negative, cited_cards[]`. This is the judge's output schema, the SFT target, and what annotators fill in.
- **`Finding`, `Report`**: PLAN.md §5.5. Gold annotations and predictions use the **same schema**.

Export the JSON Schema to `docs/schema/`. That file doubles as the vLLM guided-decoding grammar.

---

## 4. Milestones and tasks

### M0: Foundations (week 1)
- [ ] Add dependencies; set up ruff, pyright, pytest and CI.
- [ ] Write `models.py`, `config.py` and the JSON Schema export.
- [ ] Write the LLM client: a vLLM endpoint with guided JSON, `n` samples with majority vote, retries on validation failure, disk cache, token and GPU-time logging. The API backend sits behind the same interface.
- [ ] Bring up vLLM with Qwen3-8B and Qwen3-4B on the team GPU, and smoke-test guided JSON.
- [ ] Hand-write 20–30 **fixture skills** with gold labels: one per type, per language, per composite pattern; a contradicted negative claim; an instruction-only skill; hidden HTML-comment instructions; aligned controls.
- [ ] Write the CLI skeleton.

**Exit criterion:** CI is green, the fixtures validate, and a guided-JSON call to the local 8B model works.

### M1: Collection and corpus (weeks 1–3)
- [ ] `collect/official.py`, `github.py` (code search, rate limits, keep clones for history), `clawhub.py`, `skillssh.py`. Check each site's ToS and robots.txt first.
- [ ] Import released samples from MalSkillBench and MaliciousSkillBench if their licenses allow.
- [ ] Filter, dedup (MinHash), profile, and write a manifest for each skill.
- [ ] `collect/history.py`: find drift candidates where code changed and docs did not, filtered by a capability diff from M3's detectors. A first version can use regexes until M3 lands.

**Exit criterion:** a corpus statistics table and at least 1,000 drift candidates.

> **Safety:** crawled content is untrusted. Parse it as text and ASTs only, store it with no execute permissions, and never run it outside the sandbox.

### M2: Normalization and intent extraction (weeks 2–4)
- [ ] `normalize/*`: roles, languages, frontmatter, fenced code saved as virtual files with line mapping.
- [ ] `intents/extract.py`: per-chunk claim extraction with guided JSON and a verbatim quote check.
- [ ] `intents/canonicalize.py`: the action vocabulary (about 40 verbs → expected capabilities) and the category classifier (Qwen3-4B).
- [ ] `intents/consolidate.py`: embedding clusters plus an LLM merge. `conflicts.py` finds contradictions between docs.
- [ ] **Exit criterion:** intent recall ≥ 0.9 on the fixtures with Qwen3-8B. Record the 4B result as well.

### M3: Symbolic behavior extraction (weeks 3–6, the critical path)
- [ ] `taxonomy.py`: API maps seeded from Cisco, Semgrep and Bandit rules.
- [ ] `detectors/python.py`, `js.py`, `shell.py`: tree-sitter queries with constant resolution for URLs, paths, env names and command strings.
- [ ] `detectors/config.py`: dependencies with a typosquat check, postinstall hooks, Claude Code hooks, `.mcp.json`, `allowed-tools`.
- [ ] `detectors/hidden_content.py`: HTML comments, zero-width characters, base64 blobs, very long lines.
- [ ] `callgraph.py` with cross-language edges; `reachability.py` from the entry points.
- [ ] `dataflow.py`: the Semgrep taint adapter plus a light def-use pass.
- [ ] `composite.py`: the pattern library from §6.3.
- [ ] `instructions.py` (C4, local LLM).
- [ ] `cards.py`: card templates. Unit-test that cards fit within about 300 tokens each.
- [ ] **Exit criterion:** every fixture behavior and composite is recovered with the correct chain, and the pipeline runs on 100 random corpus skills without crashing.

### M4: Alignment, decisions, reports (weeks 5–7)
- [ ] `align/candidates.py`: capability compatibility ∪ top-k by embedding.
- [ ] `align/judge.py`: the checklist judge (guided JSON, n=5 votes, required card IDs).
- [ ] `align/implied.py`: the implied table plus a fallback narrow question to the LLM.
- [ ] `align/decide.py`, `risk.py`: deterministic rules.
- [ ] `report/`: JSON and Markdown/HTML output.
- [ ] Ablation flags wired in from the start: `--sources`, `--no-instructions`, `--no-composite`, `--no-crossfile`, `--cards raw`, `--judge {checklist,holistic,similarity}`, `--votes`, `--no-implied-table`, `--model`.
- [ ] **Exit criterion:** zero-shot Qwen3-8B reaches F1 ≥ 0.75 per type on the fixtures, with end-to-end `skillalign detect <dir>`.

### M5: Benchmark construction (weeks 2–8, runs in parallel)
- [ ] **Week 2:** draft `docs/guideline.md`: definitions, the decision procedure, the implied table, edge cases, 3 or more worked examples per type.
- [ ] **Week 3:** `bench/annotate_app.py`: file tree, code with line numbers, pre-filled intents and behaviors (no pre-filled findings), a linking UI, a timer, JSON output. `bench/agreement.py`: κ and span overlap.
- [ ] **Week 3:** `bench/sample.py` (random and enriched) and `bench/triage.py` (all baselines ∪ Qwen3-4B screen).
- [ ] **Week 3–4:** pilot on 30 shared skills, then revise the guideline and the implied table.
- [ ] **Weeks 4–8:** main annotation. Test splits are double-annotated and adjudicated; dev skills get one annotator plus an LLM check. Run the 40-skill **anchoring check**.
- [ ] **Weeks 5–6:** `bench/inject.py`: the 10 operators from §5.3, with **separate template sets for train and test**. Canary payloads point only at `canary.local`.
- [ ] **Week 8: freeze the test set** (repo-level split). It is never edited after this point.

Annotation pace to track weekly: about 230 person-hours in total, roughly 8 h per person per week. If the team falls behind, shrink the enriched split first.

### M6: Baselines (weeks 5–8)
- [ ] R1 `cisco.py`, R2 `rules.py`, S1 `static.py`, M1 `similarity.py` (threshold tuned on dev).
- [ ] L1 `llm_zeroshot.py` (local 8B, whole skill truncated to fit), L2 `multiagent.py` (SkillProbe-style, local 8B), L3 the same prompt on the frontier API (week 9, test set only).
- [ ] Map every output into the `Report` schema.

### M7: SkillAlign-Judge fine-tuning (weeks 6–9)
- [ ] `train/build_dataset.py`:
  1. **Injected-train**: checklist targets derived exactly from operator metadata.
  2. **Silver**: an API model fills in checklists for about 1–2k items from corpus skills *not in the benchmark* (week 7 batch). Keep only items where 2 of 3 API samples agree, or spot-check them.
  3. Hold out 10% for validation.
- [ ] `train/sft.py`: LoRA (or QLoRA) on Qwen3-8B, plus a 4B variant if time allows. Log every hyperparameter.
- [ ] `train/eval_judge.py`: checklist accuracy on the dev split.
- [ ] **Exit criterion:** the LoRA judge beats the zero-shot 8B judge on dev. If it does not, report that honestly and keep zero-shot with thinking mode as the main system.

### M8: Dynamic stage (weeks 7–9, optional, the first thing to cut)
- [ ] A sandbox image with canary secrets, a fake HOME, a mitmproxy egress proxy, strace or inotify.
- [ ] Discover entry points and run them with arguments taken from the SKILL.md examples.
- [ ] Turn traces into behaviors marked `observed_dynamically=True`.

### M9: Evaluation and study (weeks 8–11)
- [ ] `eval/match.py` (strict and lenient, PLAN.md §8.1), `metrics.py`, `stats.py` (bootstrap CIs, McNemar).
- [ ] `eval/run.py`: every system × every split, with outputs in `data/runs/<system>/<date>/`.
- [ ] RQ2 main table; RQ3 ablations; RQ4 grid (4B/8B × zero-shot/thinking/LoRA, plus the L3 reference); cost logging.
- [ ] RQ1: run on the full corpus with local models, manually verify 100 sampled findings, compute prevalence from the natural-random split.
- [ ] RQ5: 50 sampled findings, 3 raters.
- [ ] Disclose confirmed malicious skills to the platforms.

### M10: Paper and artifact (weeks 10–12)
- [ ] Figures and tables as listed in PLAN.md §9.4.
- [ ] Artifact: code, the neutralized benchmark, the guideline, prompts, LoRA weights, cached LLM responses, Docker image. Archive on Zenodo.
- [ ] Threats to validity.

---

## 5. Work split

| Person | Owns | Also |
|---|---|---|
| **A: Data** | M1 collection, dedup, drift mining; RQ1 corpus run | Annotation |
| **B: Static analysis** | M3 detectors, call graph, data flow, composites, cards; M8 dynamic | Annotation |
| **C: LLM** | LLM client and vLLM setup, M2 intents, C4 instructions, M4 judge, M7 fine-tuning, RQ4 | Annotation |
| **D: Benchmark and evaluation** | M5 guideline, app, sampling, injection, agreement; M6 baselines; M9 evaluation harness | Adjudication lead |

With only 3 people, merge D into A (data and benchmark), and have C take the baselines.

**Weekly sync:** annotation hours versus target, fixture F1 per stage, open guideline questions, GPU and API spend.

---

## 6. Timeline (12 weeks)

| Week | Engineering | Benchmark |
|---|---|---|
| 1 | M0; vLLM up; fixtures | Crawlers start |
| 2 | M2 starts | Guideline draft; crawl continues |
| 3 | M2; M3 starts | Annotation app; sampling and triage; **pilot (30 skills)** |
| 4 | M2 done; M3 | Guideline revised; **main annotation starts** |
| 5 | M3; M4 starts; M6 starts | Annotation; injection operators |
| 6 | M3 done; M4; M7 dataset; throughput measurement | Annotation; injected set built |
| 7 | M4 done (end-to-end); **silver-label batch**; M7 training; M8 (optional) | Annotation; anchoring check |
| 8 | M6 done; M9 starts on dev | **Test set frozen** |
| 9 | M7 done; M8 done or dropped; **L3 API run**; main results | Adjudication wrap-up |
| 10 | Ablations, RQ4 grid, RQ1 corpus run | κ / agreement report |
| 11 | RQ5 ratings; disclosures; writing | |
| 12 | Artifact; final paper | |

**Critical path:** contracts (M0) → symbolic behaviors (M3) → judge (M4) → evaluation (M9). **The biggest schedule risk is annotation (M5).** The pilot must happen in week 3.

**What to cut, in order, if time runs out:** M8 dynamic → the L2 multi-agent baseline → 4B fine-tuning → the size of the enriched split → the RQ5 user rating (replace with author assessment).

---

## 7. Engineering rules
- **Untrusted data.** Crawled skills are never imported or executed on the host. Anything that runs goes through the sandbox.
- **Every claim needs a span.** Objects without a valid `Span` and card IDs are rejected at the model layer.
- **Prompts are code.** Templates are versioned, and every report records the prompt hash, model ID, quantization and seed.
- **Determinism.** Temperature 0 for single calls; fixed seeds for voting; all responses cached and released.
- **No tuning on test.** Prompts, thresholds, patterns and the implied table are tuned only on dev, fixtures and injected-train.
- **Fair comparison.** All LLM baselines use the same local backbone and the same context budget as SkillAlign.

---

## 8. Decisions to make before implementation starts

These come out of the planning phase. Settle each one before week 1.

1. **Official deadline and page limit** from the ICSE industry challenge CFP. They may change §6 and §9.4 of PLAN.md.
2. **GPU inventory.** Model, VRAM, and whether the GPUs are shared. This decides bf16 vs. quantized models and LoRA vs. QLoRA.
3. **API provider and dollar cap** for silver labels and L3.
4. **Registries to crawl.** Confirm the ToS for ClawHub and skills.sh. Decide whether to request the MalSkillBench and MaliciousSkillBench data from their authors.
5. **Annotation logistics.** Who annotates, weekly hours, the adjudicator.
6. **Whether dynamic analysis (M8) is in scope** or explicitly future work.
7. **Assign owners** for the A/B/C/D roles in §5.

---

## 9. Reuse from Code Steward (`~/Desktop/Projects/cs_2`): design reference only

Code Steward is under the **Siemens Inner Source License v1.5**, which restricts sharing to Siemens affiliates. Its code therefore **must not be copied** into SkillAlign, because the SkillAlign artifact will be public. We reuse its *ideas and lessons* and **re-implement from scratch** in this repo. Rules for doing that:
- Do not copy and paste any file, function, prompt text or test.
- Write new code from the design notes below, not side by side with the old source.
- If anyone is unsure about a piece, check with the Code Steward owner or Siemens before release.

| Code Steward piece | What we take (concept) | Where it goes in SkillAlign |
|---|---|---|
| `inference/spec_agent.py`: `SpecificationGenerationSource` protocol plus an augmentation agent that registers several sources | A **pluggable multi-source extractor** pattern: each declarative source (frontmatter, SKILL.md sections, README, docstrings) is one source class that returns intent candidates, and the outputs are merged afterward | `intents/extract.py`, `intents/consolidate.py` (Stage B) |
| `spec_agent.py`: "seed properties" passed to the LLM with "avoid repeating" | Give the LLM the **symbolic findings first**, so it only adds what is missing. This makes the prompts shorter, which matters for 4–8B models. | B1 claim extraction (seed from frontmatter `description`); C4 instruction extraction (seed from regex hits) |
| `spec_agent.py`: line-numbered source in prompts (`with_line_numbers`) | Prefix every line with its number so the LLM can cite **spans**. We verify each citation against the file. | All LLM prompts that cite evidence (B1, C4, D2) |
| `spec_agent.py`: robust JSON parsing (stripping markdown fences and thinking text; a bare array treated as a list of items), confidence clamping, dedup | Defensive output parsing. **With vLLM guided decoding this becomes the fallback path**, still needed for the API backend and for Qwen3 thinking mode. | `llm/client.py` |
| `spec_extractor.py`: deterministic extractors (docstrings, type hints, assertions, NLP regex patterns) run before the LLM | **Symbolic-first, LLM-second** ordering, and keeping track of which extractor produced each claim | Stage B (regex claim patterns for "never…", "only…", "supports…") and Stage C overall |
| `analysis/impact.py` + `integration/gitnexus_index.py` / `mcp_client.py`: GitNexus code graph, with an AST fallback when it is unavailable | Use **GitNexus** (third-party, [abhigyanpatwari/GitNexus](https://github.com/abhigyanpatwari/GitNexus)) as an *optional* cross-file call graph and execution-flow ("process") provider. Our own tree-sitter resolver is the default and the fallback. Code Steward's lessons apply: index freshness checks, timeouts, and test-only blast radius. | `behaviors/callgraph.py` (adapter); evaluated as an optional backend. **Check GitNexus's license** before relying on it in the artifact. |
| `llm_models/azure.py`, `aws.py`: providers adapted to one OpenAI-style `chat.completions.create` interface, retry on auth failure | A **single provider interface**. In our case: vLLM (OpenAI-compatible, local) as the default, and one API provider for silver labels and L3 | `llm/client.py` |
| `observability/langfuse.py`, `tracing.py`: Langfuse prompt registry and traces, non-fatal wrappers | **Prompt versioning and token/latency logging**. We can self-host Langfuse, or keep it lighter with versioned `*.j2` files plus the disk cache's logs. Observability must never break a run. | `llm/prompts/`, `llm/client.py`; decide in M0 whether Langfuse is worth the setup |
| `llm_evaluation/deepeval.py`: LLM-as-judge scoring of generated artifacts | An **automatic proxy for explanation quality** (G-Eval-style rubric scoring of each finding's rationale), used alongside the human ratings for RQ5. It is not a substitute for them. | `eval/` (RQ5, optional) |
| `agents/graph.py`: LangGraph pipeline with a validation→regeneration loop | The **refinement loop** idea: re-ask the judge when its cited card IDs are invalid or its votes split evenly. LangGraph itself is overkill for our linear pipeline, but it suits the **L2 multi-agent (SkillProbe-style) baseline**. | `align/judge.py` (retry rule); `baselines/multiagent.py` |
| `validation/orchestrator.py` (unit → property → formal layers) | **Layered confirmation**: static evidence → optional dynamic run → (future) generated tests that check whether an intent is fulfilled, CASCADE-style | M8 dynamic stage; future work |
| `verification/report_builder.py`, `certification_report.py` (Markdown/PDF reports) | Layout of human-readable reports: summary first, then findings with evidence and code excerpts | `report/` (Stage E) |
| Project setup: `uv`, ruff, mypy, pre-commit, CI, mkdocs, a `Makefile`, config precedence (explicit → project → user → env) | The same engineering conventions, rewritten for this repo | M0 |

**Not reusable:** fix generation, git apply, the pytest plugin and the FastAPI UI. They solve a different problem. For annotation, Streamlit is simpler than the FastAPI UI.

**Lessons to carry over:**
- Every LLM-dependent step needs a deterministic fallback so that runs never fail outright.
- Keep observability non-fatal.
- Reasoning models leak "thinking" text into outputs. Strip it, or use constrained decoding.
