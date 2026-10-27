"""Command-line entry point: `skillalign --help`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, ValidationError

from skillalign.config import REPO_ROOT, get_settings
from skillalign.models import SCHEMA_MODELS, Annotation
from skillalign.spans import check_annotation

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def schema(out: Path = REPO_ROOT / "docs" / "schema") -> None:
    """Export the JSON Schema of the data contracts."""
    out.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n")
    typer.echo(f"wrote {len(SCHEMA_MODELS)} schemas to {out}")


@app.command()
def validate(
    files: Annotated[list[Path], typer.Argument(help="Annotation JSON files")],
    root: Annotated[Path | None, typer.Option(
        help="Skill directory for span checks, or a folder of skill directories named like "
             "the JSON files (gold/01-x.json -> <root>/01-x/). Default: the JSON's folder.")
    ] = None,
) -> None:
    """Validate annotation files: schema, internal references, and spans against the files."""
    bad = 0
    for f in files:
        try:
            ann = Annotation.model_validate_json(f.read_text())
        except ValidationError as err:
            bad += 1
            typer.echo(f"FAIL {f}\n{err}")
            continue
        skill_dir = root or f.parent
        if (skill_dir / f.stem).is_dir():
            skill_dir = skill_dir / f.stem
        errors = check_annotation(ann, skill_dir)
        if errors:
            bad += 1
            typer.echo(f"FAIL {f}")
            for e in errors:
                typer.echo(f"  - {e}")
        else:
            typer.echo(f"ok   {f}  ({len(ann.intents)} intents, {len(ann.behaviors)} "
                       f"behaviors, {len(ann.findings)} findings)")
    raise typer.Exit(1 if bad else 0)


class _Smoke(BaseModel):
    capital: str
    population_millions: float = Field(ge=0)


@app.command("llm-smoke")
def llm_smoke(model: str | None = None, n: int = 3) -> None:
    """Check the LLM endpoint: one constrained-JSON call with `n` samples."""
    from skillalign.llm import LLMClient

    client = LLMClient(model=model)
    typer.echo(f"endpoint {get_settings().llm_base_url}  model {client.model}")
    msgs = [{"role": "user", "content":
             "What is the capital of France and its population in millions? Answer in JSON."}]
    answer, agreement = client.vote(msgs, _Smoke, n=n)
    typer.echo(f"answer {answer.model_dump()}  agreement {agreement:.2f}")
    typer.echo(f"usage {client.usage}")


# ---------------------------------------------------------------- collection

collect_app = typer.Typer(no_args_is_help=True, help="Snapshot skills into data/corpus/.")
app.add_typer(collect_app, name="collect")


def _store():
    from skillalign.collect.store import CorpusStore

    return CorpusStore(get_settings().data_dir)


@collect_app.command("official")
def collect_official_cmd() -> None:
    """Shallow-clone the official skill repos and snapshot every skill in them."""
    from skillalign.collect.repos import collect_official

    ms = collect_official(_store(), get_settings().data_dir / "raw" / "clones")
    typer.echo(f"{len(ms)} official skills")


@collect_app.command("repo")
def collect_repo_cmd(repo: Annotated[str, typer.Argument(help="owner/name on GitHub")],
                     platform: str = "github") -> None:
    """Shallow-clone one GitHub repo and snapshot every skill in it."""
    from skillalign.collect.repos import clone, collect_repo

    dest = get_settings().data_dir / "raw" / "clones" / repo.replace("/", "__")
    commit = clone(f"https://github.com/{repo}", dest)
    ms = collect_repo(_store(), dest, platform=platform, repo=repo, commit=commit)
    typer.echo(f"{len(ms)} skills from {repo}@{commit[:12]}")


@collect_app.command("github")
def collect_github_cmd(max_repos: int = 1000, query: str = "filename:SKILL.md") -> None:
    """Find skills with GitHub code search (needs GITHUB_TOKEN) and snapshot them."""
    import logging

    from skillalign.collect.github import GitHubClient, collect_github

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = GitHubClient()
    ms = collect_github(client, _store(), get_settings().data_dir / "raw" / "github",
                        query=query, max_repos=max_repos)
    typer.echo(f"{len(ms)} new skills, {client.requests} requests")


@app.command("corpus")
def corpus_cmd() -> None:
    """Filter, deduplicate and profile the collected skills."""
    from skillalign.collect.dedup import dedup
    from skillalign.collect.profile import profile

    store = _store()
    corpus_dir = get_settings().data_dir / "corpus"
    result = dedup(store)
    result.save(corpus_dir / "dedup.json")
    aliases = sum(len(c["aliases"]) for c in result.clusters)
    typer.echo(f"filtered out {len(result.filtered)}; {aliases} duplicates in "
               f"{len(result.clusters)} clusters; {len(result.canonical)} canonical skills")
    summary = profile(store, result.canonical, corpus_dir / "profile.jsonl")
    (corpus_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(json.dumps(summary, indent=2))


@app.command("normalize")
def normalize_cmd(skill_dir: Path) -> None:
    """Show how a skill folder is parsed: files, roles, doc chunks, code blocks, comments."""
    from skillalign.normalize.loader import load_skill

    s = load_skill(skill_dir)
    typer.echo(f"name: {s.name}\ndescription: {s.description}\n"
               f"allowed-tools: {s.allowed_tools}\nfrontmatter error: {s.frontmatter_error}")
    typer.echo("\nfiles:")
    for f in s.skill.files:
        typer.echo(f"  {f.role.value:9} {f.language or '-':11} {f.path}")
    typer.echo("\ndoc chunks:")
    for c in s.chunks:
        typer.echo(f"  {c.file}:{c.line_start}-{c.line_end}  {c.kind}  {c.heading!r}")
    for f, c in s.comments:
        typer.echo(f"\ncomment {f}:{c.line_start}: {c.text[:100]!r}")
    for path, reason in s.skipped:
        typer.echo(f"skipped {path}: {reason}")


if __name__ == "__main__":
    app()
