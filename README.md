# SkillAlign

Detects mismatches between what an agent skill *claims* (SKILL.md, README, docs) and what it *does* (code, scripts, config, agent instructions). Plan: [docs/PLAN.md](docs/PLAN.md). Schedule and milestones: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Setup

```bash
uv sync
uv run pytest
```

## LLM endpoint

Every LLM call goes through one OpenAI-compatible endpoint, set with environment variables:

| Variable | Default |
|---|---|
| `SKILLALIGN_LLM_BASE_URL` | `http://localhost:8000/v1` |
| `SKILLALIGN_LLM_MODEL` | `Qwen/Qwen3-8B` |
| `SKILLALIGN_LLM_THINKING` | `0` (set `1` for Qwen3 thinking mode) |
| `SKILLALIGN_DATA_DIR` | `./data` (the response cache lives in `data/cache/llm.sqlite`) |

On the GPU box (RTX PRO 6000):

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384 --enable-prefix-caching --reasoning-parser qwen3 \
  --structured-outputs-config '{"backend": "xgrammar"}'
```

Do not add `"disable_any_whitespace": true`. It bans all whitespace in constrained JSON, which pushes Qwen3 off its usual output format. Sampled answers (temperature > 0, as used for voting) then become nonsense: 0 of 32 calls named Paris as the capital of France. Tested on vLLM 0.33 with xgrammar 0.2.8.

Known issue: with the default (unlimited) whitespace, large schemas such as the full `Annotation` sometimes loop on blank lines until `max_tokens`. In testing, this happened in 1 of 4 runs without thinking and 4 of 4 with thinking. The client marks such replies as truncated, never caches them, and retries with a request for shorter output.

From a laptop, tunnel to it (`ssh -N -L 8000:localhost:8000 <gpu-box>`), then check it:

```bash
uv run skillalign llm-smoke
```

ollama works too: `SKILLALIGN_LLM_BASE_URL=http://localhost:11434/v1 SKILLALIGN_LLM_MODEL=qwen3:8b`.

## Commands

- `skillalign schema` exports the JSON Schema of the data contracts to `docs/schema/`.
- `skillalign validate gold/*.json --root skills/` checks annotations: schema, internal references, and that every span and quote matches the files.
- `skillalign llm-smoke` makes one constrained-JSON call to the endpoint.
- `skillalign detect <skill-dir> [--out-dir D]` runs the full pipeline (claims, behaviors, judge, rules) and prints or writes the report. Ablation flags: `--no-instructions`, `--no-composite`, `--cards raw`, `--judge holistic`, `--no-implied-table`, `--votes N`, `--model M`.
- `skillalign baseline r2|m1 <skill-dir>` runs a baseline. `skillalign eval <gold-dir> <pred-dir> [--lenient]` scores predictions with bootstrap CIs.
- `skillalign bench inject <skills-dir> <gold-dir> <out> --templates train|test` builds the injected split from aligned skills (canary hosts only).
- `uv run pytest -m fixtures -s` runs the whole pipeline on the 17 fixtures against the server and prints per-type F1.

## Building the corpus

Everything goes under `data/` (gitignored). Crawled skills are stored as plain text and never run.

```bash
uv run skillalign collect official
```

GitHub code search needs a token. Create a fine-grained token with read-only access to public repositories, put it in a `.env` file at the repo root (gitignored; copy `.env.example`) as `GITHUB_TOKEN=...`, then:

```bash
uv run skillalign collect github --max-repos 1500
```

Filter, deduplicate and profile what was collected (writes `data/corpus/dedup.json`, `profile.jsonl`, `summary.json`):

```bash
uv run skillalign corpus
```

To see how one skill is parsed: `uv run skillalign normalize <skill-dir>`.

## Fixtures

`tests/fixtures/skills/` has 17 small hand-written skills, and `tests/fixtures/gold/` has their labels in the same `Annotation` format the system outputs. They cover every finding type (UI, UI-partial, UB, SD with four axes), composites across files, hidden and inline agent instructions, an npm install hook, dormant code, implied behavior, and aligned controls. All network targets use `canary.local`.
