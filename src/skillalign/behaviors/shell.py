"""Shell detector: commands, pipelines, redirections and substitutions, from text only.

`analyze_shell` reads a script or a fenced shell block. `analyze_command` reads one command
line that a Python or JavaScript call hands to a shell.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable

from skillalign.behaviors.facts import DYNAMIC, NET_IN, SENSITIVE, Fact, FactSet
from skillalign.behaviors.patterns import SECRET_NAME, host_of, is_sysconfig, path_class
from skillalign.models import Capability as C
from skillalign.models import Origin

INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "python", "python3", "python2",
                "node", "nodejs", "perl", "ruby", "php", "pwsh", "powershell", "deno", "bun"}
FETCHERS = {"curl", "wget", "http", "https", "aria2c", "fetch", "lwp-download"}
READERS = {"cat", "head", "tail", "less", "more", "wc", "tac", "nl", "bat", "jq", "xxd",
           "hexdump", "strings", "file", "stat", "md5sum", "sha1sum", "sha256sum", "shasum",
           "cksum", "od", "sort", "uniq", "cut", "diff", "cmp", "paste", "column", "fold",
           "rev", "base64"}
PATTERN_FIRST = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "sed", "awk", "gawk"}
DELETERS = {"rm", "rmdir", "unlink", "shred", "trash", "trash-put"}
ENUMERATORS = {"find", "ls", "tree", "locate", "fd", "du", "dir", "mlocate"}
WRITERS_LAST = {"cp", "mv", "install", "ln", "rsync", "scp"}
WRITERS_ALL = {"touch", "mkdir", "mkfifo", "truncate"}
INSTALLERS = {"pip", "pip3", "pipx", "uv", "poetry", "conda", "mamba", "npm", "yarn", "pnpm",
              "bun", "apt", "apt-get", "yum", "dnf", "brew", "apk", "gem", "cargo", "go",
              "pacman", "zypper", "choco", "winget", "snap", "composer", "nuget"}
INSTALL_VERBS = {"install", "add", "i", "ci", "get", "-s", "update", "upgrade"}
PRIVILEGE = {"sudo", "su", "doas", "pkexec", "runas", "setcap", "visudo"}
PERSIST = {"crontab", "at", "schtasks", "update-rc.d", "chkconfig", "launchctl", "systemctl"}
NETWORK = {"ssh", "sftp", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "rsync", "scp"}
DNS_TOOLS = {"nslookup", "dig", "host", "resolvectl"}
ENCODERS = {"base64", "uuencode", "uudecode"}
NOOP = {"echo", "printf", "cd", "set", "export", "unset", "local", "declare", "readonly", "shift",
        "return", "exit", "true", "false", "test", "[", "[[", ":", "trap", "source", ".",
        "alias", "type", "command", "hash", "wait", "sleep", "date", "pwd", "basename",
        "dirname", "realpath", "readlink", "hostname", "uname", "id", "whoami", "tr", "xargs",
        "read", "getopts", "let", "which", "whereis", "tput", "clear", "yes", "seq", "expr",
        "bc", "sed", "awk", "gawk", "grep", "egrep", "fgrep", "rg", "ag", "tee", "head", "tail",
        "wc", "sort", "uniq", "cut", "cat", "mktemp", "time", "then", "else", "fi", "do",
        "done", "esac", "case", "in", "function", "select", "until", "while", "for", "if",
        "{", "}", "(", ")", "!", "env", "printenv", "fg", "bg", "jobs", "kill", "ps", "nohup",
        "shopt", "pushd", "popd", "ulimit", "umask", "break", "continue", "eval",
        "mapfile", "readarray", "compgen", "complete", "bind", "enable", "help", "history",
        "caller", "dirs", "disown", "fc", "logout", "suspend", "times", "coproc"}
_COMMAND_WORD = re.compile(r"^[A-Za-z0-9_./~$][\w./+@-]*$")
_FUNC_DEF = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)", re.M)
KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "for", "in",
            "case", "esac", "!", "{", "}", "time", "function", "select"}
SKIP_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/stdin", "-", ""}

_DOWNLOAD_EXEC = re.compile(
    r"(?:\b(?:ba|z|da|k)?sh|\bsource|\.)\s+<\(\s*(?:curl|wget)|"
    r"\beval\b[^|;]*\$\(\s*(?:curl|wget)|"
    r"\b(?:ba|z)?sh\s+-c\s+[\"']?\$\(\s*(?:curl|wget)|"
    r"\b(?:python3?|node|perl|ruby)\s+-c?e?\s*[\"']?\$\(\s*(?:curl|wget)")
_SUBST = re.compile(r"\$\(|`")
_HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][\w]*)['\"]?")
_ENV_REF = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})")


def _norm(path: str) -> str:
    p = path.replace("${HOME}", "~").replace("$HOME", "~").replace("$USERPROFILE", "~")
    return p.replace("%USERPROFILE%", "~").strip()


def _logical_lines(text: str) -> list[tuple[int, int, str]]:
    """(first line, last line, text) with backslash continuations joined and heredocs cut."""
    out: list[tuple[int, int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start, buf = i, lines[i]
        while buf.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            buf = buf.rstrip()[:-1] + " " + lines[i].strip()
        end = i
        stripped = buf.strip()
        if stripped and not (stripped.startswith("#") and not stripped.startswith("#!")):
            out.append((start + 1, end + 1, buf))
        m = _HEREDOC.search(buf)
        if m and "<<<" not in buf:
            delim = m.group(1)
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
        i += 1
    return out


def _substitutions(line: str) -> tuple[str, list[str]]:
    """Replace `$(...)` and backticks with a placeholder; return the pieces."""
    subs: list[str] = []
    out: list[str] = []
    i = 0
    while i < len(line):
        if line.startswith("$(", i) and not line.startswith("$((", i):
            depth, j = 1, i + 2
            while j < len(line) and depth:
                depth += {"(": 1, ")": -1}.get(line[j], 0)
                j += 1
            subs.append(line[i + 2 : j - 1])
            out.append("__SUB__")
            i = j
        elif line[i] == "`":
            j = line.find("`", i + 1)
            if j < 0:
                out.append(line[i])
                i += 1
            else:
                subs.append(line[i + 1 : j])
                out.append("__SUB__")
                i = j + 1
        else:
            out.append(line[i])
            i += 1
    return "".join(out), subs


def _tokens(line: str) -> list[str]:
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = "#"
        return list(lex)
    except ValueError:
        return line.split()


def _stages(tokens: list[str]) -> list[list[list[str]]]:
    """Command groups, each a list of pipe stages, each a list of tokens."""
    groups: list[list[list[str]]] = [[[]]]
    for tok in tokens:
        if tok in (";", "&&", "||", "&", ";;", "(", ")"):
            if groups[-1][-1] or len(groups[-1]) > 1:
                groups.append([[]])
        elif tok in ("|", "|&"):
            groups[-1].append([])
        else:
            groups[-1][-1].append(tok)
    return [g for g in groups if any(g)]


class _Ctx:
    def __init__(self, facts: FactSet, file: str, origin: Origin, scope: str,
                 functions: frozenset[str] = frozenset()) -> None:
        self.facts, self.file, self.origin, self.scope = facts, file, origin, scope
        self.functions = functions  # names the script defines itself: not programs

    def add(self, cap: C, action: str, l0: int, l1: int, *, target: str | None = None,
            destination: str | None = None, tags: Iterable[str] = ()) -> Fact:
        return self.facts.add(Fact(cap, action, self.file, l0, l1, target=target,
                                   destination=destination, origin=self.origin,
                                   scope=self.scope, tags=set(tags)))

    def read(self, path: str, l0: int, l1: int) -> Fact:
        path = _norm(path)
        kind = path_class(path)
        if kind == "credential":
            return self.add(C.CREDENTIAL_ACCESS, f"read {path}", l0, l1, target=path,
                            tags={SENSITIVE})
        return self.add(C.FS_READ, f"read {path}", l0, l1, target=path,
                        tags={SENSITIVE} if kind == "sensitive" else ())

    def write(self, path: str, l0: int, l1: int, verb: str = "write") -> Fact:
        path = _norm(path)
        kind = path_class(path)
        if kind == "persistence":
            return self.add(C.PERSISTENCE, f"{verb} {path}", l0, l1, target=path)
        if is_sysconfig(path) and kind != "sensitive":
            return self.add(C.SYS_CONFIG_MODIFY, f"{verb} {path}", l0, l1, target=path)
        return self.add(C.FS_WRITE, f"{verb} {path}", l0, l1, target=path)


def analyze_shell(text: str, file: str, facts: FactSet, *, origin: Origin = Origin.CODE,
                  scope: str = "<module>") -> None:
    ctx = _Ctx(facts, file, origin, scope, frozenset(_FUNC_DEF.findall(text)))
    for l0, l1, line in _logical_lines(text):
        _line(line, l0, l1, ctx)


def analyze_command(command: str, file: str, line: int, facts: FactSet, *,
                    origin: Origin = Origin.CODE, scope: str = "<module>") -> None:
    """One command line given to a shell from another language, at `line` of `file`."""
    _line(command, line, line, _Ctx(facts, file, origin, scope))


def _line(line: str, l0: int, l1: int, ctx: _Ctx, depth: int = 0) -> None:
    if depth > 3:
        return
    if _DOWNLOAD_EXEC.search(line):
        url = next(iter(re.findall(r"https?://[^\s\"')|]+", line)), None)
        ctx.add(C.DOWNLOAD_EXEC, "download a script and run it", l0, l1, target="shell",
                destination=url if isinstance(url, str) else None, tags={DYNAMIC})
        return
    flat, subs = _substitutions(line)
    for sub in subs:
        _line(sub, l0, l1, ctx, depth + 1)
    tokens = _tokens(flat)
    if tokens and tokens[0].startswith("#!"):
        return
    for group in _stages(tokens):
        _group(group, l0, l1, ctx, depth)


def _words(stage: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Command words and (operator, target) redirections of one stage."""
    words: list[str] = []
    redirs: list[tuple[str, str]] = []
    i = 0
    while i < len(stage):
        tok = stage[i]
        op = tok.lstrip("0123456789") if tok[:1].isdigit() and not tok.isdigit() else tok
        if op in (">", ">>", ">|", "&>", "&>>", "<") and i + 1 < len(stage):
            redirs.append((op, stage[i + 1]))
            i += 2
        elif tok.isdigit() and i + 1 < len(stage) and stage[i + 1] in (">", ">>", "<", ">&"):
            i += 1
        elif tok in (">&", "<<", "<<<", "<<-") and i + 1 < len(stage):
            i += 2
        else:
            words.append(tok)
            i += 1
    while words and (words[0] in KEYWORDS or re.match(r"^[A-Za-z_]\w*=", words[0])):
        words.pop(0)
    return words, redirs


def _group(group: list[list[str]], l0: int, l1: int, ctx: _Ctx, depth: int) -> None:
    prev: list[str] | None = None
    parsed = [_words(stage) for stage in group]
    for i, (words, redirs) in enumerate(parsed):
        if words and _base(words[0]) in FETCHERS and i + 1 < len(parsed) and parsed[i + 1][0] \
                and _base(parsed[i + 1][0][0]) in INTERPRETERS | {"sudo"}:
            prev = words  # the next stage runs this download: one DOWNLOAD_EXEC covers both
            continue
        if prev is not None and words and prev[0] in FETCHERS and \
                _base(words[0]) in INTERPRETERS | {"sudo"}:
            url = next((w for w in prev if "://" in w), None)
            ctx.add(C.DOWNLOAD_EXEC, "download a script and run it", l0, l1,
                    target=_base(words[0]), destination=url, tags={DYNAMIC})
            prev = words
            continue
        _stage(words, redirs, l0, l1, ctx, depth, piped_in=prev is not None)
        prev = words or prev


def _data_args(args: list[str]) -> str:
    """The arguments of a curl or wget call that carry the request body."""
    out: list[str] = []
    for i, a in enumerate(args):
        if re.match(r"^(-d|--data[\w-]*|-F|--form|--json|-T|--upload-file|--post-data|"
                    r"--body-data|--post-file)$", a) and i + 1 < len(args):
            out.append(args[i + 1])
        elif re.match(r"^(--data[\w-]*|--form|--json|--post-data)=", a):
            out.append(a.split("=", 1)[1])
        elif a.startswith("-d") and len(a) > 2 and not a.startswith("--"):
            out.append(a[2:])
    return " ".join(out)


def _base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _paths(args: list[str], *, skip_first: bool = False) -> list[str]:
    plain = [a for a in args if not a.startswith("-") and a != "__SUB__"]
    if skip_first and plain:
        plain = plain[1:]
    return [a for a in plain if "/" in a or "." in a or a.startswith("~") or a.startswith("$")]


def _stage(words: list[str], redirs: list[tuple[str, str]], l0: int, l1: int, ctx: _Ctx,
           depth: int, *, piped_in: bool) -> None:
    for op, target in redirs:
        target = _norm(target)
        if target in SKIP_TARGETS or target == "__SUB__":
            continue
        if op == "<":
            ctx.read(target, l0, l1)
        else:
            ctx.write(target, l0, l1, "append to" if op.endswith(">>") else "write")
    if not words:
        return
    cmd, args = _base(words[0]), words[1:]
    joined = " ".join(words)

    carried = _data_args(args) if cmd in FETCHERS else joined
    for name in dict.fromkeys(_ENV_REF.findall(joined)):
        if SECRET_NAME.search(name):
            # A secret in a header, a user name or the URL authenticates the call. Only a
            # secret in the sent data (or outside a download tool) is a leak.
            leaked = f"${name}" in carried or f"${{{name}}}" in carried
            ctx.add(C.ENV_READ, f"read ${name}", l0, l1, target=f"${name}",
                    tags={SENSITIVE} if leaked else ())

    if cmd in PRIVILEGE:
        ctx.add(C.PRIV_ESCALATION, f"run with elevated rights ({cmd})", l0, l1, target=cmd)
        if args and cmd in ("sudo", "doas", "pkexec"):
            _stage(*_words(args), l0, l1, ctx, depth, piped_in=piped_in)
        return
    if cmd in ("bash", "sh", "zsh", "dash") and "-c" in args and args.index("-c") + 1 < len(args):
        _line(args[args.index("-c") + 1], l0, l1, ctx, depth + 1)
        return
    if cmd in ("nohup", "xargs", "time", "exec", "command", "env") and args:
        rest = [a for a in args if not a.startswith("-") and "=" not in a.split("/")[0]]
        if cmd == "env" and not rest:
            ctx.add(C.ENV_READ, "read all environment variables", l0, l1, target="env (all)",
                    tags={SENSITIVE})
            return
        if rest:
            _stage(*_words(rest), l0, l1, ctx, depth, piped_in=piped_in)
        return

    if cmd in FETCHERS:
        _fetch(cmd, args, l0, l1, ctx)
    elif cmd in READERS or cmd in PATTERN_FIRST:
        sed_inplace = cmd == "sed" and any(a.startswith("-i") or a == "--in-place"
                                             for a in args)
        for p in _paths(args, skip_first=cmd in PATTERN_FIRST and not piped_in):
            (ctx.write(p, l0, l1, "edit") if sed_inplace else ctx.read(p, l0, l1))
        if cmd == "base64":
            ctx.add(C.ENCODE_OBFUSCATE, "base64 encode or decode", l0, l1, target=cmd)
    elif cmd in DELETERS:
        for p in _paths(args) or [a for a in args if not a.startswith("-")]:
            ctx.add(C.FS_DELETE, f"delete {_norm(p)}", l0, l1, target=_norm(p))
    elif cmd in ENUMERATORS:
        found = _paths(args) or [a for a in args if not a.startswith("-")][:1] or ["."]
        ctx.add(C.FS_ENUMERATE, f"list files under {_norm(found[0])}", l0, l1,
                target=_norm(found[0]))
        if cmd == "find":
            if "-delete" in args:
                ctx.add(C.FS_DELETE, "delete the files found", l0, l1, target=_norm(found[0]))
            if "-exec" in args:
                tail = args[args.index("-exec") + 1 :]
                end = next((k for k, a in enumerate(tail) if a in (";", "\\;", "+")), len(tail))
                _stage(*_words(tail[:end]), l0, l1, ctx, depth, piped_in=True)
    elif cmd in WRITERS_LAST and cmd not in ("scp", "rsync"):
        paths = [a for a in args if not a.startswith("-")]
        if paths:
            ctx.write(paths[-1], l0, l1, {"cp": "copy to", "mv": "move to"}.get(cmd, "write"))
            for src in paths[:-1]:
                ctx.read(src, l0, l1) if cmd == "cp" else None
    elif cmd in ("scp", "rsync", "sftp"):
        remote = [a for a in args if re.match(r"^[\w.-]+@[\w.-]+:|^[\w.-]+:[/~]", a)]
        ctx.add(C.NET_OUTBOUND, f"{cmd} transfer", l0, l1, target=" ".join(args)[:80],
                destination=remote[0].split(":")[0].split("@")[-1] if remote else None)
    elif cmd in WRITERS_ALL:
        for p in [a for a in args if not a.startswith("-")]:
            ctx.write(p, l0, l1, "create")
    elif cmd == "tee":
        for p in [a for a in args if not a.startswith("-")]:
            ctx.write(p, l0, l1, "append to" if "-a" in args else "write")
    elif cmd == "dd":
        for a in args:
            if a.startswith("of="):
                ctx.write(a[3:], l0, l1)
            elif a.startswith("if="):
                ctx.read(a[3:], l0, l1)
    elif cmd == "chmod":
        mode = next((a for a in args if not a.startswith("-")), "")
        if re.search(r"[ugoa]*\+s|^[0-7]*[4-7][0-7]{3}$|777", mode):
            ctx.add(C.PRIV_ESCALATION, f"change permissions ({mode})", l0, l1, target=mode)
        else:
            tgt = _paths(args[1:]) or args[1:2]
            ctx.add(C.FS_WRITE, f"change permissions ({mode})", l0, l1,
                    target=tgt[0] if tgt else None)
    elif cmd in INSTALLERS:
        verb = next((a for a in args if not a.startswith("-")), "")
        pkg_verb = verb in INSTALL_VERBS or (cmd == "uv" and verb in ("pip", "add", "tool"))
        if pkg_verb or (cmd in ("pip", "pip3", "pipx") and verb):
            pkgs = [a for a in args if not a.startswith("-") and a != verb][:6]
            ctx.add(C.PKG_INSTALL, f"{cmd} {verb} {' '.join(pkgs)}".strip(), l0, l1,
                    target=" ".join(pkgs) or "project dependencies")
    elif cmd in PERSIST:
        if cmd in ("crontab", "at", "schtasks", "update-rc.d", "chkconfig") or \
                any(a in ("enable", "load", "bootstrap") for a in args):
            ctx.add(C.PERSISTENCE, f"{cmd} {' '.join(args)}".strip()[:80], l0, l1,
                    target=cmd)
    elif cmd in NETWORK:
        host = next((a.split("@")[-1].split(":")[0] for a in args
                     if not a.startswith("-") and "." in a or "@" in a), None)
        ctx.add(C.NET_OUTBOUND, f"{cmd} connection", l0, l1, target=cmd, destination=host)
    elif cmd in DNS_TOOLS:
        ctx.add(C.DNS, f"{cmd} lookup", l0, l1, target=" ".join(args)[:60])
    elif cmd in ("env", "printenv"):
        named = [a for a in args if not a.startswith("-")]
        ctx.add(C.ENV_READ, "read environment variables" if not named else f"read {named[0]}",
                l0, l1, target=named[0] if named else "env (all)",
                tags={SENSITIVE} if not named or SECRET_NAME.search(named[0]) else ())
    elif cmd == "eval":
        ctx.add(C.CODE_EVAL, "eval a string", l0, l1, target=" ".join(args)[:60],
                tags={DYNAMIC})
    elif cmd == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        ctx.add(C.PROC_EXEC, f"run git {sub}".strip(), l0, l1, target=" ".join(words)[:80])
        if sub == "config" and ("--global" in args or "--system" in args):
            ctx.add(C.SYS_CONFIG_MODIFY, "change the global git config", l0, l1,
                    target=" ".join(args)[:60])
        if sub in ("clone", "pull", "push", "fetch", "ls-remote", "submodule"):
            url = next((a for a in args if "://" in a or re.match(r"^[\w.-]+@[\w.-]+:", a)),
                       None)
            ctx.add(C.NET_OUTBOUND, f"git {sub}", l0, l1, target=url,
                    destination=host_of(url) or (url.split("@")[-1].split(":")[0] if url
                                                 else None),
                    tags={NET_IN} if sub in ("clone", "pull", "fetch") else ())
    elif cmd in ("defaults", "gsettings", "sysctl", "reg", "setx", "netsh", "dconf", "nmcli"):
        ctx.add(C.SYS_CONFIG_MODIFY, f"{cmd} {' '.join(args)}".strip()[:80], l0, l1, target=cmd)
    elif cmd in ("npm", "pip", "pip3") and "config" in args:
        ctx.add(C.SYS_CONFIG_MODIFY, f"{cmd} config", l0, l1, target=cmd)
    elif (cmd not in NOOP and cmd not in ctx.functions and _COMMAND_WORD.match(words[0])
          and not cmd.startswith(("$", "__SUB__"))):
        ctx.add(C.PROC_EXEC, f"run {cmd}", l0, l1, target=" ".join(words)[:80])


def _fetch(cmd: str, args: list[str], l0: int, l1: int, ctx: _Ctx) -> None:
    url = next((a for a in args if "://" in a), None)
    host = host_of(url) if url else None
    flags = [a for a in args if a.startswith("-")]
    upload = any(re.match(r"^(-d|--data.*|-F|--form.*|-T|--upload-file|--post-data|--post-file|"
                          r"--json)$", f) or f.startswith("-d") and len(f) > 2 and f[2] != "-"
                 for f in flags) or any(
        a in ("POST", "PUT", "PATCH") for a in args)
    tags = set() if upload else {NET_IN}
    for a in args:
        m = re.match(r"^(?:--data[\w-]*=|--form=)?[^=@]*@(.+)$", a) if "@" in a else None
        if m and m.group(1) not in ("-", ""):
            ctx.read(m.group(1), l0, l1)
    out = None
    for i, a in enumerate(args):
        if a in ("-o", "-O", "--output", "--output-document") and i + 1 < len(args):
            out = args[i + 1]
        elif a == "-O" and cmd == "curl":
            out = "(remote name)"
    if cmd == "wget" and out is None and not upload:
        out = "(remote name)"
    ctx.add(C.NET_OUTBOUND, "upload data" if upload else "download", l0, l1,
            target=url, destination=host or url, tags=tags)
    if out and out != "-":
        ctx.write(out, l0, l1, "save download to")
