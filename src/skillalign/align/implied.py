"""D3: is an unlinked behavior implied by the skill's category? (guideline §3).

The table is symbolic. A behavior that the table does not settle gets one narrow question to
the model: would a user who read only the description expect this? Without a model the
answer is "not implied", so the behavior is reported as UB.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache

from pydantic import BaseModel

from skillalign.llm import LLMClient, StructuredOutputError
from skillalign.llm.client import majority
from skillalign.llm.templates import load_prompt
from skillalign.models import Behavior, Origin
from skillalign.models import Capability as C

Rule = Callable[[Behavior], bool]

_TMP = re.compile(r"(^|/)(tmp|temp)(/|$)|tempfile|mkdtemp|%TEMP%|\$TMPDIR|\bTemporaryDirectory",
                  re.I)
_SECRET_PATH = re.compile(r"\.ssh|\.aws|\.gnupg|\.netrc|\.npmrc|\.pypirc|credentials|id_rsa|"
                          r"\.env\b|token|secret|passw|keychain|cookies|login data", re.I)
_KEY_VAR = re.compile(r"KEY|TOKEN|API|SECRET|AUTH", re.I)
_OUTSIDE = re.compile(r"^(~|/|\.\./|\$HOME|\$\{?HOME)", re.I)


def _inside_project(b: Behavior) -> bool:
    return not (b.target and (_OUTSIDE.match(b.target) or _SECRET_PATH.search(b.target)))


def _own(b: Behavior) -> bool:
    return b.capability is C.FS_READ and _inside_project(b)


def _always(b: Behavior) -> bool:
    """Rows that apply to every category: temp files, reading one's own files, stdout, and
    showing something to the user or asking them."""
    if b.capability is C.USER_INTERACTION:
        return True
    if b.composite_of or b.origin is not Origin.CODE:
        return False
    if b.capability in (C.FS_WRITE, C.FS_DELETE) and b.target and _TMP.search(b.target):
        return True
    return _own(b) and not (b.target and _SECRET_PATH.search(b.target))


def _caps(*caps: C, inside: bool = False, tool: re.Pattern | None = None) -> Rule:
    def rule(b: Behavior) -> bool:
        if b.composite_of or b.capability not in caps:
            return False
        if inside and not _inside_project(b):
            return False
        return not (tool and not tool.search(f"{b.action} {b.target or ''}"))
    return rule


def _api_key_env(b: Behavior) -> bool:
    return b.capability is C.ENV_READ and bool(b.target and _KEY_VAR.search(b.target)
                                               ) and not b.composite_of


def _user_url(b: Behavior) -> bool:
    """A request to a URL the user gives: no fixed host was resolved."""
    return b.capability is C.NET_OUTBOUND and not b.destination and not b.composite_of


def _declared_tool(*names: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(names) + r")\b", re.I)


IMPLIED: dict[str, list[Rule]] = {
    "file-conversion": [
        _caps(C.FS_READ, C.FS_WRITE, C.DATA_TRANSFORM, inside=True),
        _caps(C.PROC_EXEC, tool=_declared_tool("pandoc", "libreoffice", "soffice", "ffmpeg",
                                               "convert", "magick", "pdftotext", "qpdf"))],
    "web-fetch": [_user_url, _caps(C.FS_WRITE, C.DATA_TRANSFORM, inside=True)],
    "api-client": [_api_key_env, _caps(C.DATA_TRANSFORM, C.EXTERNAL_API)],
    "code-analysis": [
        _caps(C.FS_READ, C.FS_WRITE, C.FS_ENUMERATE, C.DATA_TRANSFORM, inside=True),
        _caps(C.PROC_EXEC, tool=_declared_tool("ruff", "eslint", "flake8", "black", "prettier",
                                               "mypy", "pytest", "tsc", "gofmt"))],
    "git-devops": [
        _caps(C.PROC_EXEC, tool=_declared_tool("git", "gh", "docker", "kubectl", "make")),
        _caps(C.FS_READ, C.FS_WRITE, C.FS_ENUMERATE, inside=True)],
    "data-analysis": [
        _caps(C.FS_READ, C.FS_WRITE, C.FS_ENUMERATE, C.DATA_TRANSFORM, C.PKG_INSTALL,
              inside=True)],
    "browser-automation": [_caps(C.NET_OUTBOUND, C.PROC_EXEC, C.DATA_TRANSFORM),
                           _caps(C.FS_WRITE, C.FS_READ, inside=True)],
    "instruction-only": [],
}


def table_says(category: str | None, b: Behavior) -> bool:
    """True if the table implies `b` for `category`. False means 'not covered by the table'."""
    if b.composite_of:
        return False  # a source -> sink path is never implied
    return _always(b) or any(rule(b) for rule in IMPLIED.get(category or "", []))


class ExpectedAnswer(BaseModel):
    expected: bool


@cache
def _prompt():
    return load_prompt("implied_d3")


def would_user_expect(client: LLMClient, description: str, card: str, *, votes: int = 3) -> bool:
    """The narrow D3 question. Majority over `votes` samples; any failure means 'no'."""
    message = _prompt().render(description=description or "(none)", card=card)
    try:
        samples = client.structured([{"role": "user", "content": message}], ExpectedAnswer,
                                    n=votes, max_tokens=64)
    except StructuredOutputError:
        return False
    winner, _ = majority(samples, key=lambda a: a.expected)
    return winner.expected
