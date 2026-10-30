"""R2: capability rules plus keyword matching against the docs (PLAN.md §7).

- A reachable behavior whose capability has no matching keyword anywhere in the docs is UB.
- A verb in the docs (from the B2 vocabulary) whose expected capabilities no reachable
  behavior has is UI.
It has no judge, no implied table and no SD, so it is the floor the other systems must beat.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from skillalign.baselines.common import FindingBuilder, annotation
from skillalign.behaviors import detect_behaviors
from skillalign.intents.vocab import ACTION_VOCAB, canonical_verb
from skillalign.models import Capability as C
from skillalign.models import Evidence, FileRole, Intent, Reachability
from skillalign.normalize.loader import LoadedSkill, load_skill

#: Words in the docs that declare a capability.
KEYWORDS: dict[C, re.Pattern] = {c: re.compile(p, re.I) for c, p in {
    C.FS_READ: r"\b(read|reads|load|open|parse|input|file|files|scan)\b",
    C.FS_WRITE: r"\b(write|writes|save|saves|output|export|create|generate|append|file)\b",
    C.FS_DELETE: r"\b(delete|remove|clean|purge|erase)\b",
    C.FS_ENUMERATE: r"\b(list|find|search|scan|walk|glob|directory|folder|files)\b",
    C.ENV_READ: r"\b(env|environment|variable|variables|config|api[_ -]?key)\b",
    C.CREDENTIAL_ACCESS: r"\b(credential|credentials|token|password|secret|login|auth|"
                         r"authenticate|key)\b",
    C.NET_OUTBOUND: r"\b(http|https|url|api|download|fetch|upload|send|post|request|network|"
                    r"web|online|server|cloud|sync|publish|notify)\b",
    C.NET_INBOUND: r"\b(serve|server|listen|port|inbound)\b",
    C.DOWNLOAD_EXEC: r"\b(download|install|run|execute|script)\b",
    C.PROC_EXEC: r"\b(run|runs|execute|invoke|launch|command|commands|shell|cli|git|build|"
                 r"compile|test|lint|script)\b",
    C.CODE_EVAL: r"\b(eval|evaluate|exec|execute|dynamic)\b",
    C.PERSISTENCE: r"\b(schedule|cron|startup|autostart|persist|daemon|service)\b",
    C.SYS_CONFIG_MODIFY: r"\b(config|configure|settings|setup|profile)\b",
    C.PKG_INSTALL: r"\b(install|pip|npm|package|packages|dependency|dependencies)\b",
    C.ENCODE_OBFUSCATE: r"\b(encode|encoding|base64|encrypt|encrypts|compress|obfuscate)\b",
    C.DATA_TRANSFORM: r"\b(convert|transform|format|parse|summarize|extract|merge|split|"
                      r"analyze|validate)\b",
    C.AGENT_INSTRUCTION: r"\b(instruct|instruction|agent|always|never|ignore)\b",
    C.TOOL_PERMISSION: r"\b(tool|tools|permission|permissions|allowed)\b",
    C.EXTERNAL_API: r"\b(api|service|endpoint|sdk)\b",
    C.USER_INTERACTION: r"\b(ask|prompt|confirm|show|display|tell)\b",
    C.DNS: r"\b(dns|domain|resolve)\b",
    C.PRIV_ESCALATION: r"\b(sudo|root|admin|privilege)\b",
}.items()}
#: Capabilities too ordinary to report as undeclared by themselves.
IGNORED = {C.DATA_TRANSFORM, C.USER_INTERACTION}
#: Capabilities the detectors never report, so a doc verb that needs only these is not UI.
UNDETECTED = {C.DATA_TRANSFORM, C.USER_INTERACTION}


def doc_text(loaded: LoadedSkill) -> str:
    return "\n".join(t for p, t in loaded.texts.items() if "#" not in p
                     and loaded.artifact(p).role in (FileRole.SKILL_MD, FileRole.DOC))


def doc_intents(loaded: LoadedSkill) -> list[Intent]:
    """One intent per distinct vocabulary verb at the start of a documentation sentence,
    sourced from the first line that has it."""
    seen: set[str] = set()
    out: list[Intent] = []
    for chunk in loaded.chunks:
        lines = loaded.texts[chunk.file].splitlines()
        for n in range(chunk.line_start, chunk.line_end + 1):
            line = lines[n - 1].strip()
            text = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)
            m = re.match(r"([A-Za-z]+)\b\s*(.*)", text)
            if not m:
                continue
            verb = canonical_verb(m.group(1))
            if verb is None or verb in seen or text.lower().startswith(("do not", "don't")):
                continue
            seen.add(verb)
            out.append(Intent(
                id=f"I{len(out) + 1}", action=verb, object=m.group(2)[:80] or "it",
                expected_capabilities=list(ACTION_VOCAB[verb]),
                sources=[Evidence(file=chunk.file, line_start=n, line_end=n, quote=text[:200])]))
    return out


def run_r2(skill_dir: Path, *, skill_id: str | None = None):
    loaded = load_skill(skill_dir, skill_id=skill_id)
    behaviors = [b for b in detect_behaviors(loaded).behaviors
                 if b.reachability is not Reachability.DORMANT]
    docs = doc_text(loaded)
    intents = doc_intents(loaded)
    out = FindingBuilder()
    have = {b.capability for b in behaviors}
    has_code = bool(loaded.code_files())
    own_scripts = {PurePosixPath(a.path).name for a in loaded.code_files()}
    for intent in intents:
        want = set(intent.expected_capabilities)
        if want <= UNDETECTED:
            if has_code or not want:
                continue  # no detector reports these, so any code could be doing it
        elif C.PROC_EXEC in want and any(n in intent.sources[0].quote for n in own_scripts):
            continue  # "run scripts/x.py" is the skill's own entry point, not a new behavior
        if want and not have & want:
            out.ui(intent, f"The docs say to {intent.action}, but no behavior has "
                           f"{[c.value for c in intent.expected_capabilities]}.")
    for b in behaviors:
        if b.capability in IGNORED or b.composite_of:
            continue
        if not KEYWORDS[b.capability].search(docs):
            out.ub(b, f"{b.capability.value} behavior with no related word in the docs.")
    composites = [b for b in behaviors if b.composite_of]
    parts = {p for b in composites for p in b.composite_of}
    for b in composites:
        out.ub(b, "A composite source-to-sink path is never declared by keywords.")
    # a part already covered by a composite is not reported twice
    out.findings[:] = [f for f in out.findings
                       if not (f.type.value == "UB" and f.behaviors[0] in parts)]
    for n, f in enumerate(out.findings, 1):
        f.id = f"F{n}"
    return annotation(loaded.skill.skill_id if skill_id is None else skill_id, "baseline:R2",
                      intents, behaviors, out.findings)

