"""Command-line entry point: `skillalign --help`."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, ValidationError

from skillalign.config import REPO_ROOT, get_settings
from skillalign.models import SCHEMA_MODELS, Annotation
from skillalign.spans import check_annotation

app = typer.Typer(no_args_is_help=True, add_completion=False)

#: Pre-annotation model (IMPLEMENTATION.md §4 M5). The 8B model is for smoke tests only.
PREANNOTATE_MODEL = "Qwen/Qwen3-32B"


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
def collect_github_cmd(max_repos: int = 1000, query: str = "filename:SKILL.md",
                       max_skills_per_repo: int = 50) -> None:
    """Find skills with GitHub code search (needs GITHUB_TOKEN) and snapshot them."""
    import logging

    from skillalign.collect.github import GitHubClient, collect_github

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = GitHubClient()
    ms = collect_github(client, _store(), get_settings().data_dir / "raw" / "github",
                        query=query, max_repos=max_repos,
                        max_skills_per_repo=max_skills_per_repo)
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



@app.command("behaviors")
def behaviors_cmd(
    skill_dir: Path,
    as_json: bool = typer.Option(False, "--json", help="Print Behavior objects as JSON"),
) -> None:
    """List what a skill's code, config and hidden text can do (M3). Nothing is run."""
    from skillalign.behaviors import detect_behaviors
    from skillalign.normalize.loader import load_skill

    found = detect_behaviors(load_skill(skill_dir))
    if as_json:
        typer.echo(json.dumps([b.model_dump(mode="json") for b in found.behaviors], indent=2))
        return
    for b in found.behaviors:
        ev = b.evidence[0]
        typer.echo(f"{b.id:4} {b.capability.value:17} {b.reachability.value:9} "
                   f"{ev.file}:{ev.line_start}  {b.action}"
                   + (f"  -> {b.destination}" if b.destination else "")
                   + (f"  [{', '.join(b.composite_of)}]" if b.composite_of else ""))
    for h in found.hidden:
        typer.echo(f"hidden {h.kind}: {h.file}:{h.line_start}  {h.snippet}")
    for n in found.notes:
        typer.echo(f"note: {n}")


@app.command("detect")
def detect_cmd(
    skill_dir: Path,
    model: Annotated[str | None, typer.Option(help="Model ID; default SKILLALIGN_LLM_MODEL")]
    = None,
    out_dir: Annotated[Path | None, typer.Option(help="Write <id>.json and <id>.md here")]
    = None,
    votes: Annotated[int, typer.Option(help="Judge samples per question")] = 5,
    no_instructions: Annotated[bool, typer.Option(help="Ablation: no agent-instruction "
                                                       "behaviors")] = False,
    no_composite: Annotated[bool, typer.Option(help="Ablation: no composite behaviors")] = False,
    cards: Annotated[str, typer.Option(help="cards (default) or raw code for the judge")
                     ] = "cards",
    judge: Annotated[str, typer.Option(help="checklist (default) or holistic")] = "checklist",
    no_implied_table: Annotated[bool, typer.Option(help="Ablation: every leftover behavior "
                                                        "is UB")] = False,
) -> None:
    """Find misalignments between what a skill says and what it does (M4)."""
    from skillalign.align.pipeline import DetectOptions, detect_skill
    from skillalign.align.report import render_report, report_json
    from skillalign.intents.extract import file_stem
    from skillalign.llm import LLMClient

    if cards not in ("cards", "raw") or judge not in ("checklist", "holistic"):
        raise typer.BadParameter("--cards is cards|raw and --judge is checklist|holistic")
    opts = DetectOptions(instructions=not no_instructions, composite=not no_composite,
                         cards=cards, judge=judge,  # type: ignore[arg-type]
                         implied_table=not no_implied_table, votes=votes)
    client = LLMClient(model=model)
    res = detect_skill(client, skill_dir, opts)
    text = render_report(res)
    if out_dir is None:
        typer.echo(text)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem(res.annotation.skill_id)
        (out_dir / f"{stem}.json").write_text(json.dumps(report_json(res), indent=2) + "\n")
        (out_dir / f"{stem}.md").write_text(text)
        typer.echo(f"{out_dir / stem}.json  {res.annotation.skill_label.value}, "
                   f"{len(res.annotation.findings)} findings")
    typer.echo(f"usage {client.usage}", err=True)


@app.command("baseline")
def baseline_cmd(
    name: Annotated[str, typer.Argument(help="r2 or m1")],
    skill_dir: Path,
    out: Annotated[Path | None, typer.Option(help="Write the Annotation here")] = None,
    low: Annotated[float, typer.Option(help="M1: below this similarity a claim is unmatched")
                   ] = 0.15,
    match: Annotated[float, typer.Option(help="M1: at or above this similarity it is a match")
                     ] = 0.4,
) -> None:
    """Run a baseline (R2 rules or M1 similarity) on one skill and print its findings."""
    from skillalign.baselines.m1 import M1Config, run_m1
    from skillalign.baselines.r2 import run_r2

    if name == "r2":
        ann = run_r2(skill_dir)
    elif name == "m1":
        ann = run_m1(skill_dir, M1Config(low, match))
    else:
        raise typer.BadParameter("baseline is r2 or m1")
    for f in ann.findings:
        typer.echo(f"{f.id} {f.type.value:10} {f.rationale}")
    typer.echo(f"{ann.skill_label.value}, {len(ann.findings)} findings")
    if out:
        out.write_text(json.dumps(ann.model_dump(mode="json"), indent=2) + "\n")


def _load_annotations(folder: Path) -> dict[str, Annotation]:
    out = {}
    for f in sorted(folder.glob("*.json")):
        if f.name.endswith((".failed.json", ".stats.json")):
            continue
        ann = Annotation.model_validate_json(f.read_text())
        out[ann.skill_id] = ann
    return out


@app.command("eval")
def eval_cmd(
    gold: Annotated[Path, typer.Argument(help="Folder of gold Annotation JSON files")],
    pred: Annotated[Path, typer.Argument(help="Folder of predicted Annotation JSON files")],
    lenient: Annotated[bool, typer.Option(help="Also accept UI/UI-partial and SD/UB "
                                               "confusions")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
    boot: Annotated[int, typer.Option(help="Bootstrap resamples for the 95% CIs")] = 1000,
) -> None:
    """Score predictions against gold: per-type and skill-level P/R/F1 with bootstrap CIs."""
    from skillalign.eval.metrics import confusion_matrix, control_fpr, evaluate

    golds, preds = _load_annotations(gold), _load_annotations(pred)
    ev = evaluate(golds, preds, lenient=lenient)
    summary = ev.summary()
    summary["control_fpr"] = control_fpr(golds, preds)
    summary["missing_predictions"] = sorted(set(golds) - set(preds))
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return
    f1, lo, hi = ev.skill_f1_ci(n_boot=boot)
    c = ev.skill_level
    typer.echo(f"skills {len(golds)} ({'lenient' if lenient else 'strict'} matching)\n"
               f"skill level  P {c.precision:.2f}  R {c.recall:.2f}  F1 {f1:.2f} "
               f"[{lo:.2f}, {hi:.2f}]  control FPR {summary['control_fpr']:.2f}")
    for t, k in ev.per_type().items():
        _, tlo, thi = ev.finding_f1_ci(t, n_boot=boot)
        typer.echo(f"{t:10} P {k.precision:.2f}  R {k.recall:.2f}  F1 {k.f1:.2f} "
                   f"[{tlo:.2f}, {thi:.2f}]  tp {k.tp} fp {k.fp} fn {k.fn}")
    typer.echo(f"macro F1 {ev.macro_f1():.2f}  mean IoU {ev.mean_iou():.2f}")
    for g, row in confusion_matrix(golds, preds).items():
        if row:
            typer.echo(f"  gold {g:10} -> " + ", ".join(f"{p} {n}" for p, n in row.items()))
    if summary["missing_predictions"]:
        typer.echo(f"no prediction for {len(summary['missing_predictions'])} skills")


# ---------------------------------------------------------------- intents and benchmark

extract_app = typer.Typer(no_args_is_help=True, help="Pull claims out of skill documents.")
app.add_typer(extract_app, name="extract")


@extract_app.command("claims")
def extract_claims_cmd(
    skill_dir: Path,
    model: Annotated[str | None, typer.Option(help="Model ID; default SKILLALIGN_LLM_MODEL")]
    = None,
    out_dir: Annotated[Path | None, typer.Option(help="Default: data/bench/claims")] = None,
) -> None:
    """Extract intents (B1) from one skill folder. Quotes are checked against the files."""
    from skillalign.intents.extract import extract_claims, save_claims
    from skillalign.llm import LLMClient

    settings = get_settings()
    result = extract_claims(LLMClient(model=model), skill_dir)
    path = save_claims(result, out_dir or settings.data_dir / "bench" / "claims")
    s = result.stats
    typer.echo(f"{path}\n{s.claims_kept} kept, {s.dropped_bad_span} dropped (quote not in the "
               f"cited lines), {s.failed_chunks} chunks with no valid reply, "
               f"{s.chunks} chunks")


bench_app = typer.Typer(no_args_is_help=True, help="Benchmark sample, pre-annotation, review.")
app.add_typer(bench_app, name="bench")


@bench_app.command("sample")
def bench_sample_cmd() -> None:
    """Draw the pilot sample from data/corpus into data/bench/sample.json."""
    from skillalign.bench.sample import write_sample

    settings = get_settings()
    entries = write_sample(settings.data_dir / "corpus",
                           settings.data_dir / "bench" / "sample.json", settings.seed)
    by_stratum = Counter(e["stratum"] for e in entries)
    typer.echo(f"{len(entries)} skills: {by_stratum['random']} random, "
               f"{by_stratum['enriched']} enriched")


@bench_app.command("preannotate")
def bench_preannotate_cmd(
    limit: Annotated[int | None, typer.Option(help="Only the first N sampled skills")] = None,
    model: Annotated[str, typer.Option(help="Model ID for pre-annotation")] = PREANNOTATE_MODEL,
) -> None:
    """Draft one Annotation per sampled skill into data/bench/preannotated/. Resumable."""
    from skillalign.bench.preannotate import load_sample, run_batch
    from skillalign.llm import LLMClient

    settings = get_settings()
    bench = settings.data_dir / "bench"
    client = LLMClient(model=model, thinking=False)
    outcomes = run_batch(client, load_sample(bench / "sample.json"),
                         settings.data_dir / "corpus", bench / "preannotated", limit=limit)
    counts = Counter(o.status for o in outcomes)
    typer.echo(f"done {counts['done']}, failed {counts['failed']}, "
               f"skipped {counts['skipped']}\nusage {client.usage}")
    for o in outcomes:
        if o.status == "failed" or o.error:
            typer.echo(f"  {o.status:6} {o.skill_id}: {o.error}")


@bench_app.command("render")
def bench_render_cmd(skill_id: str) -> None:
    """Write a Markdown review view of one pre-annotated skill to data/bench/rendered/."""
    from skillalign.bench.preannotate import load_sample
    from skillalign.bench.render import render_markdown
    from skillalign.intents.extract import file_stem

    settings = get_settings()
    bench = settings.data_dir / "bench"
    entry = next((e for e in load_sample(bench / "sample.json") if e["skill_id"] == skill_id),
                 None)
    if entry is None:
        raise typer.BadParameter(f"{skill_id} is not in the sample")
    stem = file_stem(skill_id)
    ann = Annotation.model_validate_json((bench / "preannotated" / f"{stem}.json").read_text())
    text = render_markdown(ann, settings.data_dir / "corpus" / entry["path"])
    out = bench / "rendered" / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    typer.echo(str(out))


@bench_app.command("inject")
def bench_inject_cmd(
    base_skills: Annotated[Path, typer.Argument(help="Folder of aligned skill directories")],
    base_gold: Annotated[Path, typer.Argument(help="Folder of their gold JSON files, named "
                                                  "like the skill directories")],
    out: Annotated[Path, typer.Argument(help="Output folder: skills/, gold/, manifest.json")],
    templates: Annotated[str, typer.Option(help="train or test templates")] = "train",
    operator: Annotated[list[str] | None, typer.Option(help="Only these operators")] = None,
) -> None:
    """Build the injected split: mutate aligned skills with known-effect operators."""
    from skillalign.bench.inject import OPERATORS, build_set

    if templates not in ("train", "test"):
        raise typer.BadParameter("--templates is train or test")
    bad = [o for o in operator or [] if o not in OPERATORS]
    if bad:
        raise typer.BadParameter(f"unknown operator {bad}; choose from {sorted(OPERATORS)}")
    bases = []
    for gold_file in sorted(base_gold.glob("*.json")):
        ann = Annotation.model_validate_json(gold_file.read_text())
        if ann.findings or not (base_skills / gold_file.stem).is_dir():
            continue  # only aligned skills with a directory can be mutated
        bases.append((base_skills / gold_file.stem, ann))
    rows = build_set(bases, out, seed=get_settings().seed,
                     templates=templates, operators=operator)  # type: ignore[arg-type]
    counts = Counter(r["status"] for r in rows)
    typer.echo(f"{len(bases)} bases, {counts['ok']} injected, {counts['skipped']} skipped "
               f"-> {out}")


if __name__ == "__main__":
    app()
