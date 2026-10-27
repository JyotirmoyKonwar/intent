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
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384 --enable-prefix-caching --reasoning-parser qwen3
```

From a laptop, tunnel to it (`ssh -N -L 8000:localhost:8000 <gpu-box>`), then check it:

```bash
uv run skillalign llm-smoke
```

ollama works too: `SKILLALIGN_LLM_BASE_URL=http://localhost:11434/v1 SKILLALIGN_LLM_MODEL=qwen3:8b`.

## Commands

- `skillalign schema` exports the JSON Schema of the data contracts to `docs/schema/`.
- `skillalign validate gold/*.json --root skills/` checks annotations: schema, internal references, and that every span and quote matches the files.
- `skillalign llm-smoke` makes one constrained-JSON call to the endpoint.

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
