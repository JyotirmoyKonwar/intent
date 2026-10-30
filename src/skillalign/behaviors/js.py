"""JavaScript and TypeScript detector. Regex level: no parser, so no data flow."""

from __future__ import annotations

import re

from skillalign.behaviors.facts import DYNAMIC, NET_IN, SENSITIVE, Fact, FactSet
from skillalign.behaviors.patterns import SECRET_NAME, host_of, is_sysconfig, path_class
from skillalign.models import Capability as C
from skillalign.models import Origin

_URL = re.compile(r"https?://[^\s\"'`)]+")
_ARGV = re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*process\.argv\[(\d+)\]")
_ASSIGN = re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(['\"`])([^'\"`]*)\2")
_CP = re.compile(r"\b(?:exec|execSync|spawn|spawnSync|execFile|execFileSync|fork)\s*\(")
_NET = re.compile(r"\b(?:https?|axios|got|superagent|request)\.(?:get|post|put|request|patch|"
                  r"delete|head)\s*\(|\bfetch\s*\(|\bnew XMLHttpRequest\b|\bnew WebSocket\s*\(|"
                  r"\bnet\.(?:connect|createConnection)\s*\(|\baxios\s*\(|\bgot\s*\(")
_EVAL = re.compile(r"\beval\s*\(|\bnew Function\s*\(|\bvm\.(?:runIn\w+|Script)\s*\(|"
                   r"\bFunction\s*\(")
_READ = re.compile(r"\b(?:readFileSync|readFile|createReadStream|readdirSync|readdir|"
                   r"readlinkSync)\s*\(")
_WRITE = re.compile(r"\b(?:writeFileSync|writeFile|appendFileSync|appendFile|"
                    r"createWriteStream|copyFileSync|copyFile|renameSync|rename|mkdirSync|"
                    r"mkdir)\s*\(")
_DELETE = re.compile(r"\b(?:unlinkSync|unlink|rmSync|rmdirSync|rmdir|rm)\s*\(")
_ENV = re.compile(r"process\.env(?:\.(\w+)|\[['\"](\w+)['\"]\])?")
_ENC = re.compile(r"\bBuffer\.from\([^)]*['\"]base64['\"]|\batob\s*\(|\bbtoa\s*\(|"
                  r"toString\(['\"]base64['\"]\)")


def analyze_js(text: str, file: str, facts: FactSet, *, origin: Origin = Origin.CODE,
               scope: str = "<module>") -> None:
    lines = text.splitlines()
    names: dict[str, str] = {}
    for line in lines:
        for m in _ARGV.finditer(line):
            names[m.group(1)] = f"argv[{m.group(2)}]"
        for m in _ASSIGN.finditer(line):
            names.setdefault(m.group(1), m.group(3))
    has_cp = bool(re.search(r"child_process|\bcp\.|execa|shelljs", text))

    def add(cap: C, action: str, n: int, *, target=None, destination=None, tags=()):
        return facts.add(Fact(cap, action, file, n, n, target=target, destination=destination,
                              origin=origin, scope=scope, tags=set(tags)))

    def first_arg(line: str, m: re.Match) -> str | None:
        rest = line[m.end():]
        lit = re.match(r"\s*(['\"`])([^'\"`]*)\1", rest)
        if lit:
            return lit.group(2)
        ident = re.match(r"\s*(process\.argv\[\d+\]|\w+)", rest)
        if ident:
            tok = ident.group(1)
            if tok.startswith("process.argv"):
                return "argv[" + tok[tok.index("[") + 1 : -1] + "]"
            return names.get(tok)
        return None

    for n, line in enumerate(lines, 1):
        code = line.split("//", 1)[0] if "http" not in line.split("//", 1)[0] else line
        for m in _NET.finditer(code):
            url = _URL.search(code)
            if url is None:
                lit = first_arg(code, m)
                url = re.match(r"https?://\S+", lit) if lit else None
            u = url.group(0) if url else None
            add(C.NET_OUTBOUND, f"request {host_of(u) or 'a remote host'}", n, target=u,
                destination=host_of(u) or u, tags={NET_IN})
        if has_cp:
            for m in _CP.finditer(code):
                cmd = first_arg(code, m)
                add(C.PROC_EXEC, f"run {cmd or 'a command'}"[:90], n, target=cmd,
                    tags=() if cmd else {DYNAMIC})
                if cmd:
                    from skillalign.behaviors.shell import analyze_command
                    analyze_command(cmd, file, n, facts, origin=origin, scope=scope)
        for m in _EVAL.finditer(code):
            arg = first_arg(code, m)
            add(C.CODE_EVAL, "evaluate a string as code", n, target=arg,
                tags=() if arg else {DYNAMIC})
        for m in _READ.finditer(code):
            path = first_arg(code, m)
            kind = path_class(path)
            if "readdir" in m.group(0):
                add(C.FS_ENUMERATE, f"list files under {path or 'a directory'}", n,
                    target=path)
            elif kind == "credential":
                add(C.CREDENTIAL_ACCESS, f"read {path}", n, target=path, tags={SENSITIVE})
            else:
                add(C.FS_READ, f"read {path or 'a file'}", n, target=path,
                    tags={SENSITIVE} if kind == "sensitive" else ())
        for m in _WRITE.finditer(code):
            path = first_arg(code, m)
            kind = path_class(path)
            if kind == "persistence":
                add(C.PERSISTENCE, f"write {path}", n, target=path)
            elif is_sysconfig(path):
                add(C.SYS_CONFIG_MODIFY, f"write {path}", n, target=path)
            else:
                add(C.FS_WRITE, f"write {path or 'a file'}", n, target=path)
        for m in _DELETE.finditer(code):
            if "fs." in code or "Sync" in m.group(0) or "unlink" in m.group(0):
                path = first_arg(code, m)
                add(C.FS_DELETE, f"delete {path or 'a path'}", n, target=path)
        for m in _ENV.finditer(code):
            key = m.group(1) or m.group(2)
            add(C.ENV_READ, f"read {key or 'all environment variables'}", n,
                target=f"${key}" if key else "process.env (all)",
                tags={SENSITIVE} if not key or SECRET_NAME.search(key) else ())
        if _ENC.search(code):
            add(C.ENCODE_OBFUSCATE, "base64 encode or decode", n, target="base64")
