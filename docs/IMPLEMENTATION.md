# SkillAlign: Implementation Plan

This document covers how we build what [PLAN.md](PLAN.md) describes: the stack, repository layout, data contracts, milestones and the day-by-day schedule. Section numbers such as §6.3 point to PLAN.md.

**Assumptions (re-scoped 2026-10-27):** 17 days to the deadline (**2026-11-13 AoE**), one person (possibly 3–4 more later), a tight API budget, an RTX PRO 6000 (96 GB) for inference, a Colab A100 (40 GB) for training only, and small local models (Qwen3 4B–8B) as the main engine.

---

## 1. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language and environment | Python ≥3.13, `uv` | Already set up in this repo |
| Data models | `pydantic` v2 | One schema for extractor output, annotations and reports |
| CLI | `typer` | `skillalign schema / validate / llm-smoke`, later `collect / extract / detect / eval` |
| Parsing | `tree-sitter` + `tree-sitter-language-pack`, `python-frontmatter`, `markdown-it-py`, `bashlex` | Exact line numbers; Python's own `ast` is fine for Python files |
| Call graph and data flow | Our own light resolver (imports + calls + def-use within a function) | Enough for the composite patterns in §6.3; Semgrep only if it fits the schedule |
| Graphs | `networkx` | Behavior graph, path enumeration |
| **LLM access** | `openai` client against **any OpenAI-compatible server**; `response_format=json_schema` for constrained output | The same code talks to vLLM on the GPU box and to ollama on a laptop |
| **Serving (GPU box, not a project dependency)** | `vllm` serving Qwen3-8B, Qwen3-4B and Qwen3-32B | 96 GB fits 32B in bf16 (~64 GB) or 8B + 32B-AWQ side by side |
| Embeddings | Qwen3-Embedding-0.6B via the same server (`/v1/embeddings`) | Recall only |
| Caching | stdlib `sqlite3`, keyed by hash(model, messages, schema, params, seed, sample index) | Reruns are reproducible and free; the cache is part of the artifact |
| Fine-tuning (stretch) | `peft` QLoRA + `trl` SFT on Colab A100 | Only if days 10–12 have slack |
| Annotation review | Rendered Markdown view + hand-edited JSON, validated by `skillalign validate` | No time for a UI; the validator catches broken spans and references |
| Experiments | `pandas`, `scikit-learn`, `scipy` (bootstrap, McNemar), `matplotlib` | |
| Quality | `pytest`, `ruff` | CI is optional for one person |

### Compute
- **Inference (RTX PRO 6000, 96 GB):** run one vLLM server per model or one at a time. Qwen3-8B bf16 ≈ 16 GB, Qwen3-32B bf16 ≈ 64 GB; with `--gpu-memory-utilization` split, 8B and 32B-AWQ fit together. Use prefix caching, because the judge prompts share long prefixes.
- **Laptop (MX330, 2 GB):** too small for real runs. Use it for code and tests only; the tests use a fake LLM transport. Point `SKILLALIGN_LLM_BASE_URL` at the GPU box (SSH tunnel) for real calls.
- **Colab A100 (40 GB):** QLoRA of the 8B judge if we get to it. Not used for inference.
- **Throughput check on day 6:** time the full pipeline on 50 skills before planning the corpus run.

### API budget
Optional. The local 32B replaces the API for pre-annotation and silver labels. If money is available, spend it on one L3 frontier reference run on the test set (day 12–13). Pilot 20 items first and extrapolate; cache every response.

---

## 2. Repository layout

```
intent/
├── pyproject.toml
├── docs/                      # challenge PDF, PLAN.md, IMPLEMENTATION.md, guideline.md, schema/
├── src/skillalign/
│   ├── models.py              # pydantic data contracts (M0, done)
│   ├── config.py              # env-based settings (M0, done)
│   ├── spans.py               # check spans/quotes against files on disk (M0, done)
│   ├── llm/
│   │   ├── client.py          # structured(messages, schema, n) → validated objects; voting (M0, done)
│   │   ├── cache.py           # sqlite response cache (M0, done)
│   │   └── prompts/           # versioned *.j2 templates
│   ├── collect/               # official.py github.py clawhub.py dedup.py
│   ├── normalize/             # loader.py markdown.py languages.py
│   ├── intents/               # extract.py canonicalize.py consolidate.py
│   ├── behaviors/             # taxonomy.py detectors/ callgraph.py dataflow.py composite.py instructions.py
│   ├── cards.py
│   ├── align/                 # candidates.py judge.py implied.py decide.py risk.py
│   ├── report/
│   ├── baselines/             # rules.py similarity.py llm_zeroshot.py (+ cisco.py static.py if time)
│   ├── bench/                 # sample.py preannotate.py render.py inject.py agreement.py
│   ├── eval/                  # match.py metrics.py stats.py run.py tables.py
│   └── cli.py
├── data/                      # gitignored: raw/ corpus/ bench/ runs/ cache/
├── tests/fixtures/skills/     # hand-written skills with gold.json (M0, done)
└── scripts/
```

---

## 3. Data contracts (frozen on day 1)

Every stage reads and writes JSON validated by `src/skillalign/models.py`. Run `skillalign schema` to export the JSON Schema to `docs/schema/`.

- **`Span`**: `file, line_start, line_end` (1-based, inclusive, relative to the skill root). **`Evidence`** = `Span` + `quote` (the exact text at those lines, or a substring of it). `spans.py` checks both against the files; objects with broken evidence are rejected.
- **`FileArtifact`, `Skill`**: file role, language, sha256; virtual files (fenced code inside Markdown) keep a parent path and a line offset.
- **`Intent`, `Behavior`**: PLAN.md §4.2 and §4.3.
- **`Card`**: `behavior_id, text, snippet, span, chain`. The only view of code the judge sees.
- **`ChecklistAnswer`**: `achieves_action, constraints[], axis_mismatch, contradicts_negative, cited_cards[]`. The judge's output schema and the fine-tuning target.
- **`Finding`, `Annotation`**: PLAN.md §5.5. Gold labels and system predictions use the **same `Annotation` model**; `provenance` records which system, model and prompt produced it. Model validators check that every reference (intent IDs, behavior IDs, `composite_of`, alignment links) resolves, that each finding type has the fields it needs, and that `skill_label` matches the findings.

---

## 4. Milestones and tasks

Each milestone has a **must** part (needed for the paper) and a **stretch** part.

### M0: Foundations (day 1) — done
- [x] Dependencies, package layout, `ruff`, `pytest`.
- [x] `models.py`, `config.py`, `spans.py`, JSON Schema export.
- [x] LLM client: OpenAI-compatible endpoint, JSON-schema constrained output, parse fallback (strips `<think>` and code fences), retries on validation failure, `n` samples with majority vote, sqlite cache, token and latency logging.
- [x] Fixture skills with gold labels covering every type and the main edge cases.
- [x] CLI: `schema`, `validate`, `llm-smoke`.
- [x] **On the GPU box:** start vLLM with Qwen3-8B and run `skillalign llm-smoke`.

### M1: Collection (days 2–3)
- [x] Official repos (`skillalign collect official`): anthropics/skills, openai/skills, vercel-labs/agent-skills, huggingface/skills → 98 skills, 97 after dedup (2026-10-27).
- [x] GitHub code-search collector (`skillalign collect github`), size-sharded past the 1,000-result cap, resumable, hits logged. **Needs `GITHUB_TOKEN`.**
- [x] Snapshot store with per-skill manifests and licenses; filter; exact + MinHash dedup; corpus profile (`skillalign corpus`).
- [x] Run the GitHub crawl. Target: 2–5k skills. (1,988 canonical skills after dedup; see `data/corpus/summary.json`.)
- **Must:** `anthropics/skills` and other official repos; a GitHub code-search sample of `SKILL.md` files (keep the whole skill directory). Filter, MinHash dedup, manifest per skill. Target: 2–5k skills.
- **Stretch:** ClawHub / skills.sh (check ToS and robots.txt first); MalSkillBench / MaliciousSkillBench samples if the license allows.

> **Safety:** crawled content is untrusted. Parse it as text and ASTs only, store it without execute permissions, never run it.

### M2: Normalization and intents (days 2–3)
- [x] Normalizer (`skillalign normalize <dir>`): roles, languages, frontmatter, doc chunks with line ranges, fenced code as virtual files mapped back to real lines, HTML comments.
- [x] **Decided for M3 (`behaviors/detect.py`):** most fenced code in SKILL.md is an *example*, not an instruction to run (the official pdf skill has 12 Python examples). Treat a virtual file as reachable only when the surrounding text tells the agent to run it; otherwise mark it `dormant`, or UB precision will suffer.
- **Must:** roles, languages, frontmatter, fenced code as virtual files; B1 per-chunk claim extraction with a verbatim quote check; B2 action vocabulary (~40 verbs → expected capabilities); category classifier.
- **Stretch:** B3 embedding clusters + LLM merge (v1 can merge by exact action/object).

### M3: Behaviors (days 4–6, critical path)
- **Must:** Python (`ast`) and shell detectors with constant resolution; config detector (deps, postinstall, hooks, `.mcp.json`, `allowed-tools`); hidden-content scan; import/call graph and reachability from entry points; def-use data flow inside functions plus call-argument flow; composite patterns (exfiltration, download-and-execute, persistence, credential harvesting); C4 instruction behaviors (LLM); cards.
- **Stretch:** JS/TS beyond regex-level patterns; cross-language edges beyond "shell/SKILL.md names a script".

### M4: Alignment and report (days 6–7)
- **Must:** D1 candidates, D2 checklist judge (n=5 votes, card IDs required), D3 implied table + narrow question, D4 rules, risk, JSON + Markdown report. Ablation flags from the start: `--no-instructions`, `--no-composite`, `--cards raw`, `--judge holistic`, `--no-implied-table`, `--votes`, `--model`.
- **Exit:** `skillalign detect` runs on all fixtures; per-type F1 on fixtures is printed by `pytest -m fixtures`.

### M5: Benchmark (days 2–13, in parallel)
- **Day 2:** `docs/guideline.md`: definitions, decision procedure, implied table, edge cases.
- **Day 3:** `bench/sample.py` (random + enriched), `bench/preannotate.py` (Qwen3-32B, whole skill with line numbers → `Annotation`), `bench/render.py` (Markdown view for review).
- **Days 4–5:** pilot on 20 skills → revise guideline and pre-annotation prompt.
- **Days 5–12:** check ~25 skills per day (≈ 4–5 h). Do the 20-skill from-scratch anchoring check early (before reading the LLM version).
- **Day 7:** `bench/inject.py` with the six operators in PLAN.md §5.3 + `VAGUE_DOC`; separate templates for train and test; canary hosts only (`canary.local`).
- **Day 13 (Sun 2026-11-08): freeze the test set** (repo-level split, ~30% dev / 70% test).

### M6: Baselines (days 7–9)
- **Must:** R2 rules, M1 similarity (threshold tuned on dev), L1 zero-shot 8B, L1-32B. All mapped to `Annotation`.
- **Stretch:** R1 Cisco skill-scanner, S1 Semgrep/Bandit, L3 frontier API.

### M7: Evaluation (days 8–13)
- **Must:** `eval/match.py` (strict + lenient, PLAN.md §8.1), metrics per type and split, bootstrap CIs, McNemar; main table (RQ2), the five must-ship ablations (RQ3), the 4B/8B/32B × thinking grid (RQ4), cost per skill (RQ5), prevalence on natural-random + corpus run on a few thousand skills with 50 findings checked (RQ1).
- **Stretch:** LoRA judge on Colab (needs injected-train data from day 7).

### M8: Paper and artifact (days 13–17)
- Figures and tables (PLAN.md §9.4), threats to validity, anonymous artifact repo with code, neutralized benchmark, guideline, prompts, cached responses.

---

## 5. Work split

**Now (one person):** follow the schedule in §6 in order; annotation runs in the afternoon slot on days 5–12.

**If helpers join**, hand off in this order (each handoff removes the most hours from the critical path):
1. **Annotation checker** (from day 5): reviews pre-annotated skills, double-checks 40 test skills for κ. Biggest win.
2. **Data and baselines**: collection, dedup, R2/M1/L1, then R1/S1.
3. **Evaluation and figures**: eval harness, stats, tables.

---

## 6. Schedule (17 days)

| Day | Date | Engineering | Benchmark / paper |
|---|---|---|---|
| 1 | Tue 10-27 | **M0** (contracts, LLM client, fixtures) | — |
| 2 | Wed 10-28 | M1 crawl; M2 normalizer | Guideline draft |
| 3 | Thu 10-29 | M2 claims + vocabulary | Sampling; pre-annotation + render scripts |
| 4 | Fri 10-30 | M3 Python/shell/config detectors | Pre-annotate the first batch on the GPU box |
| 5 | Sat 10-31 | M3 call graph, data flow, composites | **Pilot (20 skills)**; anchoring set |
| 6 | Sun 11-01 | M3 C4 + cards; **M4 judge** | Check ~25/day from here |
| 7 | Mon 11-02 | **M4 end to end on fixtures**; throughput check | Injection operators |
| 8 | Tue 11-03 | M6 R2, M1; eval harness | Check |
| 9 | Wed 11-04 | M6 L1, L1-32B; first dev results | Check |
| 10 | Thu 11-05 | Fix the worst error classes on dev | Check |
| 11 | Fri 11-06 | Ablation runs on dev; (stretch) LoRA | Check |
| 12 | Sat 11-07 | Corpus run (RQ1) | Finish checks |
| 13 | Sun 11-08 | **Test set frozen**; all test runs, stats, figures | Paper: method + benchmark sections |
| 14 | Mon 11-09 | Fill gaps | Paper: evaluation |
| 15 | Tue 11-10 | — | Paper: intro, related work, threats |
| 16 | Wed 11-11 | Artifact anonymization | Full draft, revise |
| 17 | Thu 11-12 | — | Polish; **submit by Fri 11-13 AoE** |

**Critical path:** contracts → behaviors (M3) → judge (M4) → test runs (day 13). **The biggest schedule risk is annotation pace**: track skills checked per day against the target of ~25.

**Cut order if behind:** stretch items → enriched split down to 60 → injected test down to 100 → thinking-mode variants → L1-32B baseline → corpus run (keep prevalence from natural-random only).

---

## 7. Engineering rules
- **Untrusted data.** Crawled skills are never imported or executed. No sandbox in v1.
- **Every claim needs evidence.** `Evidence` must point at real lines and quote real text; `skillalign validate --root` enforces it.
- **Prompts are code.** Templates are versioned; every `Annotation` records the system, model ID, prompt hash and seed in `provenance`.
- **Determinism.** Temperature 0 for single calls; fixed seeds for voting; all responses cached and released.
- **No tuning on test.** Prompts, thresholds, patterns and the implied table are tuned only on dev, fixtures and injected-train.
- **Fair comparison.** L1 uses the same 8B backbone and context budget as SkillAlign.

---

## 8. Decisions

Settled on 2026-10-27:
1. **Deadline and format:** 2026-11-13 AoE; 5–10 pages IEEE + 2 pages of references.
2. **GPU:** RTX PRO 6000 96 GB for inference; Colab A100 40 GB for training only.
3. **Annotation:** 180–300 skills, pre-annotated by Qwen3-32B and checked by a person.
4. **Dynamic analysis, drift mining, L2:** out of scope for v1.

Still open:
- API provider and dollar cap, if any (only affects the optional L3 run).
- Registries beyond GitHub and official repos (ClawHub/skills.sh ToS).
- Whether to recruit an annotation checker (recommended by day 5).

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
