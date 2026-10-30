"""B2: a controlled action vocabulary and a keyword category classifier (v1).

Each verb maps to the capabilities an intent with that action expects. This is a symbolic
signal before the judge runs: "download" with no NET_OUTBOUND behavior anywhere is almost
certainly UI. The classifier assigns the skill category used by the implied-behavior table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from skillalign.models import Capability as C
from skillalign.models import Intent, Polarity

# verb -> expected capabilities. Inflected forms are matched by `stem`.
ACTION_VOCAB: dict[str, list[C]] = {
    "read": [C.FS_READ], "load": [C.FS_READ], "open": [C.FS_READ], "parse": [C.FS_READ],
    "scan": [C.FS_READ, C.FS_ENUMERATE], "list": [C.FS_ENUMERATE],
    "search": [C.FS_READ, C.FS_ENUMERATE], "find": [C.FS_ENUMERATE],
    "write": [C.FS_WRITE], "save": [C.FS_WRITE], "export": [C.FS_WRITE],
    "create": [C.FS_WRITE], "generate": [C.FS_WRITE], "output": [C.FS_WRITE],
    "append": [C.FS_WRITE], "update": [C.FS_WRITE], "rename": [C.FS_WRITE],
    "delete": [C.FS_DELETE], "remove": [C.FS_DELETE], "clean": [C.FS_DELETE],
    "convert": [C.DATA_TRANSFORM], "transform": [C.DATA_TRANSFORM],
    "format": [C.DATA_TRANSFORM], "summarize": [C.DATA_TRANSFORM],
    "extract": [C.DATA_TRANSFORM], "merge": [C.DATA_TRANSFORM],
    "split": [C.DATA_TRANSFORM], "compress": [C.DATA_TRANSFORM],
    "encrypt": [C.DATA_TRANSFORM], "validate": [C.DATA_TRANSFORM],
    "analyze": [C.DATA_TRANSFORM], "count": [C.DATA_TRANSFORM],
    "download": [C.NET_OUTBOUND], "fetch": [C.NET_OUTBOUND], "upload": [C.NET_OUTBOUND],
    "send": [C.NET_OUTBOUND], "post": [C.NET_OUTBOUND], "publish": [C.NET_OUTBOUND],
    "sync": [C.NET_OUTBOUND], "call": [C.NET_OUTBOUND, C.EXTERNAL_API],
    "query": [C.NET_OUTBOUND, C.EXTERNAL_API], "request": [C.NET_OUTBOUND],
    "scrape": [C.NET_OUTBOUND], "crawl": [C.NET_OUTBOUND], "notify": [C.NET_OUTBOUND],
    "serve": [C.NET_INBOUND], "listen": [C.NET_INBOUND],
    "run": [C.PROC_EXEC], "execute": [C.PROC_EXEC], "invoke": [C.PROC_EXEC],
    "launch": [C.PROC_EXEC], "build": [C.PROC_EXEC], "compile": [C.PROC_EXEC],
    "commit": [C.PROC_EXEC], "deploy": [C.PROC_EXEC], "test": [C.PROC_EXEC],
    "lint": [C.PROC_EXEC], "install": [C.PKG_INSTALL],
    "configure": [C.SYS_CONFIG_MODIFY], "schedule": [C.PERSISTENCE],
    "authenticate": [C.CREDENTIAL_ACCESS, C.ENV_READ], "login": [C.CREDENTIAL_ACCESS],
    "ask": [C.USER_INTERACTION], "prompt": [C.USER_INTERACTION],
    "confirm": [C.USER_INTERACTION],
}


_IRREGULAR = {"wrote": "write", "written": "write", "ran": "run", "sent": "send",
              "found": "find", "built": "build"}


def _forms(word: str) -> list[str]:
    w = word.lower()
    out = [w, _IRREGULAR.get(w, w)]
    for suffix, add in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""),
                        ("s", ""), ("d", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            out.append(w[: -len(suffix)] + add)
    if w.endswith("ing") and len(w) > 5 and w[-4] == w[-5]:  # running -> run
        out.append(w[:-4])
    return out


def canonical_verb(action: str) -> str | None:
    """The vocabulary verb for the first recognizable word of `action`, else None."""
    for word in re.findall(r"[A-Za-z]+", action):
        for form in _forms(word):
            if form in ACTION_VOCAB:
                return form
    return None


#: Verbs that touch the file system only when the object is a file, folder or path. "Removes
#: trailing whitespace" and "creates a SkeletonHelper" are about text and objects, not files.
_FILE_VERBS = {"remove", "delete", "clean", "list", "find", "search", "scan", "create",
               "generate", "output"}
_FILE_OBJECT = re.compile(
    r"\b(files?|folders?|director(y|ies)|paths?|caches?|logs?|artifacts?|temp|tmp|reports?|"
    r"output|downloads?|images?|pdfs?|csv|json|markdown|docs?|documents?|archives?|zip|"
    r"build|dist)\b|\.\w{1,5}\b|/", re.I)


def expected_capabilities(action: str, obj: str = "") -> list[C]:
    """Capabilities implied by the action verb, with the object refining a few cases."""
    verb = canonical_verb(action)
    caps = list(ACTION_VOCAB.get(verb, [])) if verb else []
    text = f"{action} {obj}".lower()
    if verb in _FILE_VERBS and not _FILE_OBJECT.search(obj):
        caps = [C.DATA_TRANSFORM]
    if verb in ("save", "write", "create", "generate", "export", "output") and re.search(
            r"\b(remote|server|cloud|s3|bucket|online)\b", text):
        caps = [C.NET_OUTBOUND]
    if verb in ("read", "load", "open", "parse") and re.search(
            r"\b(url|http|api|web|remote)\b", text):
        caps = [C.NET_OUTBOUND]
    return caps


def enrich_intents(intents: Iterable[Intent]) -> list[Intent]:
    """Fill `expected_capabilities` from the vocabulary where the intent has none.

    A negative claim lists the capabilities it forbids: the vocabulary gives them for the
    ruled-out action (`uploads data` -> NET_OUTBOUND).
    """
    out = []
    for it in intents:
        if it.expected_capabilities:
            out.append(it)
            continue
        caps = expected_capabilities(it.action, it.object)
        if it.polarity is Polarity.NEGATIVE and not caps:
            text = f"{it.action} {it.object} {' '.join(it.constraints)}".lower()
            if re.search(r"\b(network|internet|online|server|remote|cloud|offline|leav\w*)\b",
                         text):
                caps = [C.NET_OUTBOUND]
        out.append(it.model_copy(update={"expected_capabilities": caps}))
    return out


# ---------------------------------------------------------------- category classifier

CATEGORIES = [
    "file-conversion", "web-fetch", "api-client", "code-analysis", "git-devops",
    "data-analysis", "browser-automation", "instruction-only", "other",
]

_KEYWORDS: dict[str, re.Pattern] = {
    "browser-automation": re.compile(
        r"\b(browser|playwright|puppeteer|selenium|chrome|headless|webdriver)\b", re.I),
    "git-devops": re.compile(
        r"\b(git|github|commit|pull request|\bPR\b|branch|devops|deploy|docker|kubernetes|ci/cd|"
        r"terraform)\b", re.I),
    "web-fetch": re.compile(
        r"\b(scrape|scraper|crawl|download|fetch|web page|webpage|url|website)\b", re.I),
    "api-client": re.compile(r"\b(api|rest|endpoint|webhook|sdk|oauth|token)\b", re.I),
    "code-analysis": re.compile(
        r"\b(lint|linter|format|formatter|refactor|code review|static analysis|type check|"
        r"test(s|ing)?)\b", re.I),
    "data-analysis": re.compile(
        r"\b(analy[sz]e|analysis|dataset|notebook|pandas|statistics|chart|plot|report|logs?|"
        r"summar(y|ize))\b", re.I),
    "file-conversion": re.compile(
        r"\b(convert|conversion|pdf|docx|csv|json|xlsx|markdown|pandoc|spreadsheet|"
        r"document|transcode|ocr)\b", re.I),
}
# Earlier entries win ties; the order puts narrower categories first.
_ORDER = ["browser-automation", "git-devops", "web-fetch", "api-client", "code-analysis",
          "data-analysis", "file-conversion"]


def classify_category(name: str | None, description: str | None, *, has_code: bool = True,
                      body: str = "") -> str:
    """Keyword vote over the name and description (the body counts a little). Skills with no
    executable content and no keyword hit are `instruction-only`."""
    head = f"{name or ''} {description or ''}"
    scores: dict[str, float] = {}
    for cat, pat in _KEYWORDS.items():
        n = len(pat.findall(head)) * 3 + min(len(pat.findall(body)), 3)
        if n:
            scores[cat] = n
    if scores:
        best = max(scores.values())
        return next(c for c in _ORDER if scores.get(c) == best)
    return "other" if has_code else "instruction-only"
