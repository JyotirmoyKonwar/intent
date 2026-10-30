"""Run the detectors over one loaded skill, decide reachability, build composites."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from skillalign.behaviors.config import analyze_allowed_tools, analyze_config
from skillalign.behaviors.facts import ENCODE, NET_IN, SENSITIVE, Fact, FactSet
from skillalign.behaviors.hidden import HiddenItem, comment_items, scan_text
from skillalign.behaviors.js import analyze_js
from skillalign.behaviors.pyflow import PyProject
from skillalign.behaviors.shell import analyze_command, analyze_shell
from skillalign.models import Behavior, Evidence, FileRole, Origin, Reachability, Span
from skillalign.models import Capability as C
from skillalign.normalize.loader import LoadedSkill

QUOTE_CHARS = 300
_RUN = re.compile(r"\b(run|runs|running|execute|invoke|launch|start|install|call|command|"
                  r"commands|then|next|first|step|before|after|setup|usage|perform|type)\b", re.I)
_EXAMPLE = re.compile(r"\b(example|examples|e\.g\.|for instance|sample|snippet|output|returns?|"
                      r"result|looks like|template|schema|syntax|format|pseudo|illustrat|demo|"
                      r"reference|signature|alternatively|you can also|optional|such as|"
                      r"basic|advanced)\b", re.I)
_COMMENT_CMD = re.compile(r"\b(?:curl|wget|bash|sh|zsh|python3?|node|rm|cat|nc|ncat|ssh|scp|"
                          r"sudo|pip|npm|eval|base64|chmod|git|crontab)\b[^\n]*")


@dataclass
class Detection:
    behaviors: list[Behavior]
    hidden: list[HiddenItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)


def detect_behaviors(loaded: LoadedSkill) -> Detection:
    facts = FactSet()
    proj = PyProject(facts)
    notes: list[str] = []
    hidden: list[HiddenItem] = []
    hooks: list[tuple[str, str]] = []
    runnable: dict[str, bool] = {}
    arts = loaded.skill.files

    for art in arts:
        text = loaded.texts.get(art.path)
        if text is None:
            continue
        virtual = art.role is FileRole.VIRTUAL
        origin = Origin.AGENT_INSTRUCTION if virtual else Origin.CODE
        if virtual:
            runnable[art.path] = _block_runnable(loaded, art.path, art.language)
        lang = art.language
        if art.role in (FileRole.CODE, FileRole.VIRTUAL):
            if lang == "python":
                proj.add(art.path, text, origin)
            elif lang == "shell":
                analyze_shell(text, art.path, facts, origin=origin)
            elif lang in ("javascript", "typescript"):
                analyze_js(text, art.path, facts, origin=origin)
            elif lang in ("powershell", "batch"):
                analyze_shell(text, art.path, facts, origin=origin)
                notes.append(f"{art.path}: {lang} read with the shell rules only")
        elif art.role is FileRole.CONFIG:
            hooks += analyze_config(art.path, text, facts)
        if art.role in (FileRole.SKILL_MD, FileRole.DOC, FileRole.CODE, FileRole.CONFIG):
            hidden += scan_text(art.path, text)
    proj.analyze()
    notes += proj.notes

    for path, comment in loaded.comments:
        hidden += comment_items(path, [comment])
        if any(h.kind == "comment-instruction" and h.line_start == comment.line_start
               for h in hidden):
            for m in _COMMENT_CMD.finditer(comment.text):
                cmd = re.sub(r"\s*-->.*$", "", m.group(0)).strip()
                line = comment.line_start + comment.text[: m.start()].count("\n")
                analyze_command(cmd, path, line, facts, origin=Origin.AGENT_INSTRUCTION)

    if loaded.skill_md and loaded.allowed_tools:
        md = loaded.texts[loaded.skill_md].splitlines()
        line = next((i + 1 for i, ln in enumerate(md) if ln.startswith("allowed-tools")), 1)
        analyze_allowed_tools(loaded.skill_md, loaded.allowed_tools, line, facts)

    referenced = _referenced_files(loaded, hooks)
    entries = proj.entry_modules(referenced) | {
        p for p, ok in runnable.items() if ok and p in proj.modules}
    entries = {p for p in entries if "#" not in p or runnable.get(p, False)}
    reach = proj.reach(entries)
    has_entry = bool(entries) or any(
        f.file not in proj.modules for f in facts if runnable.get(f.file, True))

    def reachability(f: Fact) -> Reachability:
        if f.file in runnable and not runnable[f.file]:
            return Reachability.DORMANT
        if f.file in proj.modules and (f.file, f.scope) not in reach:
            return Reachability.DORMANT
        return Reachability.ENTRY if has_entry else Reachability.REACHABLE

    composites = _composites(facts, proj, hooks, loaded)
    behaviors = _build(loaded, facts, composites, reachability)
    return Detection(behaviors, hidden, notes, list(facts))


# ---------------------------------------------------------------- reachability helpers


def _referenced_files(loaded: LoadedSkill, hooks: list[tuple[str, str]]) -> set[str]:
    """Code files that the documents or an install hook name."""
    docs = "\n".join(t for p, t in loaded.texts.items() if "#" not in p and (
        loaded.artifact(p).role in (FileRole.SKILL_MD, FileRole.DOC)))
    docs += "\n" + "\n".join(cmd for cmd, _ in hooks)
    out: set[str] = set()
    for art in loaded.skill.files:
        if art.role is not FileRole.CODE:
            continue
        name = PurePosixPath(art.path).name
        if re.search(r"(?<![\w.-])(?:[\w./-]*/)?" + re.escape(name) + r"(?![\w])", docs):
            out.add(art.path)
    return out


def _block_runnable(loaded: LoadedSkill, vpath: str, lang: str | None) -> bool:
    """A fenced block is something the agent should run only if the text around it says so."""
    m = re.match(r"^(.*)#L(\d+)-L\d+\.", vpath)
    if not m or m.group(1) not in loaded.markdown:
        return True
    doc, start = loaded.markdown[m.group(1)], int(m.group(2))
    before: list[str] = []
    n = start - 2  # 0-based index of the opening fence
    while n > 0 and len(before) < 3:
        n -= 1
        text = doc.lines[n].strip()
        if not text:
            if before:
                break
            continue
        if text.startswith("#"):
            break
        before.append(text)
    heading = next((s.heading for s in doc.sections
                    if s.line_start <= start and s.heading), "")
    near = " ".join(before)
    if _RUN.search(near):
        return not (_EXAMPLE.search(near) and not re.search(r"\b(run|execute)\b", near, re.I))
    if _EXAMPLE.search(near) or _EXAMPLE.search(heading):
        return False
    return lang == "shell"


# ---------------------------------------------------------------- composites


@dataclass
class _Composite:
    kind: str
    action: str
    parts: list[Fact]  # components in order; the last is the sink
    anchor: tuple[str, int, int] | None = None


def _composites(facts: FactSet, proj: PyProject, hooks: list[tuple[str, str]],
                loaded: LoadedSkill) -> list[_Composite]:
    out: dict[frozenset, _Composite] = {}

    def add(kind: str, action: str, parts: list[Fact], anchor=None) -> None:
        key = frozenset(f.idx for f in parts)
        if len(key) >= 2 and key not in out:
            out[key] = _Composite(kind, action, parts, anchor)

    for use in proj.sinks.values():
        src = [facts.items[t] for t in sorted(t for t in use.tokens if isinstance(t, int))
               if facts.items[t] is not use.sink]
        sens = [f for f in src if SENSITIVE in f.tags]
        steps = [f for f in src if ENCODE in f.tags]
        dest = use.sink.destination or use.sink.target or "a remote host"
        if use.kind == "net" and sens:
            what = sens[0].target or "sensitive data"
            add("exfiltration", f"send {what} to {dest}", [*sens, *steps, use.sink], use.anchor)
        elif use.kind == "write" and sens and use.sink.cap not in (C.PERSISTENCE,):
            what = sens[0].target or "sensitive data"
            add("credential-copy", f"copy {what} to {use.sink.target or 'another file'}",
                [*sens, *steps, use.sink], use.anchor)
        elif use.kind == "exec":
            nets = [f for f in src if NET_IN in f.tags]
            if nets:
                add("download-exec", f"download from {nets[0].destination or 'a remote host'}"
                    " and run it", [*nets, use.sink], use.anchor)

    by_line: dict[tuple[str, int], list[Fact]] = {}
    for f in facts:
        if f.file in proj.modules:
            continue
        for ln in range(f.l0, f.l1 + 1):
            by_line.setdefault((f.file, ln), []).append(f)
    for group in by_line.values():
        sens = [f for f in group if SENSITIVE in f.tags and f.cap != C.NET_OUTBOUND]
        nets = [f for f in group if f.cap == C.NET_OUTBOUND and NET_IN not in f.tags]
        if sens and nets:
            add("exfiltration", f"send {sens[0].target} to {nets[0].destination or 'a host'}",
                [*sens, nets[0]])

    hook_files = {p for p in _referenced_files(loaded, hooks)
                  if any(PurePosixPath(p).name in cmd for cmd, _ in hooks)}
    for path in {f.file for f in facts if f.file not in proj.modules}:
        nets = [f for f in facts if f.file == path and f.cap == C.NET_OUTBOUND
                and NET_IN in f.tags]
        evals = [f for f in facts if f.file == path and f.cap == C.CODE_EVAL
                 and f.l0 >= (nets[0].l0 if nets else 10**9)]
        if nets and evals:
            lead = [f for f in facts if f.cap == C.PROC_EXEC and f.origin == Origin.CONFIG
                    and PurePosixPath(path).name in (f.target or "")] if path in hook_files \
                else []
            add("download-exec", f"download from {nets[0].destination or 'a remote host'} and "
                "run it" + (" on npm install" if lead else ""), [*lead, nets[0], evals[0]])
    return list(out.values())


# ---------------------------------------------------------------- output


def _quote(loaded: LoadedSkill, span: Span) -> str:
    lines = loaded.texts.get(span.file, "").splitlines()
    text = "\n".join(lines[span.line_start - 1 : span.line_end]).strip()
    return text[:QUOTE_CHARS] or "[blank]"


def _evidence(loaded: LoadedSkill, f: Fact, extra: bool = True) -> list[Evidence]:
    spans = [(f.file, f.l0, f.l1)] + (f.also if extra else [])
    out: list[Evidence] = []
    for file, a, b in spans:
        try:
            real = loaded.real_span(Span(file=file, line_start=a, line_end=b))
        except StopIteration:
            continue
        ev = Evidence(**real.model_dump(), quote=_quote(loaded, real))
        if all((e.file, e.line_start, e.line_end) != (ev.file, ev.line_start, ev.line_end)
               for e in out):
            out.append(ev)
    return out


def _build(loaded: LoadedSkill, facts: FactSet, composites: list[_Composite],
           reach) -> list[Behavior]:
    order = {a.path: i for i, a in enumerate(loaded.skill.files)}
    singles = sorted(facts, key=lambda f: (f.origin != Origin.CODE and f.origin != Origin.CONFIG,
                                           order.get(f.file, 99), f.l0, f.idx))
    ids: dict[int, str] = {}
    behaviors: list[Behavior] = []
    for n, f in enumerate(singles, 1):
        ids[f.idx] = f"B{n}"
        behaviors.append(Behavior(
            id=f"B{n}", capability=f.cap, action=f.action[:120], target=f.target,
            destination=f.destination, conditions=f.conditions, origin=f.origin,
            reachability=reach(f), evidence=_evidence(loaded, f), call_chain=f.chain))
    n = len(behaviors)
    for comp in composites:
        part_ids = list(dict.fromkeys(ids[f.idx] for f in comp.parts))
        if len(part_ids) < 2:
            continue
        n += 1
        sink = comp.parts[-1]
        ev: list[Evidence] = []
        if comp.anchor:
            real = loaded.real_span(Span(file=comp.anchor[0], line_start=comp.anchor[1],
                                         line_end=comp.anchor[2]))
            ev.append(Evidence(**real.model_dump(), quote=_quote(loaded, real)))
        for f in comp.parts:
            for e in _evidence(loaded, f, extra=False):
                if all((e.file, e.line_start) != (x.file, x.line_start) for x in ev):
                    ev.append(e)
        behaviors.append(Behavior(
            id=f"B{n}", capability=C.COMPOSITE, action=comp.action[:120],
            target=comp.parts[0].target, destination=sink.destination or sink.target,
            origin=sink.origin, reachability=reach(sink), evidence=ev,
            composite_of=part_ids))
    return behaviors
