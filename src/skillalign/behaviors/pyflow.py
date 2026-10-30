"""Python detector: operations, constants, call edges and data flow, from the AST only.

The flow analysis is small on purpose. A value's *taint* is the set of facts that produced it
(a file read, an environment read, a network response, an encoding step). Taint moves through
assignments, calls and returns. Each function gets a summary (what its return value carries,
and which sinks its parameters reach), so a flow across functions and files is found.
Paths are not tracked: a branch is treated as if it ran.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from skillalign.behaviors import patterns as P
from skillalign.behaviors.facts import DYNAMIC, ENCODE, NET_IN, SENSITIVE, Fact, FactSet
from skillalign.behaviors.shell import analyze_command
from skillalign.models import Capability as C
from skillalign.models import Origin

Taint = frozenset  # of int (a fact index) or str ("p:<param>")
EMPTY: Taint = frozenset()
CLIENT_FACTORIES = {
    "requests.Session", "httpx.Client", "httpx.AsyncClient", "boto3.client", "boto3.resource",
    "boto3.Session", "storage.Client", "google.cloud.storage.Client", "aiohttp.ClientSession",
    "http.client.HTTPSConnection", "http.client.HTTPConnection", "smtplib.SMTP",
    "smtplib.SMTP_SSL", "ftplib.FTP", "paramiko.SSHClient", "urllib3.PoolManager",
    "azure.storage.blob.BlobServiceClient", "BlobServiceClient", "slack_sdk.WebClient",
}
WRITE_CAPS = (C.FS_WRITE, C.PERSISTENCE, C.SYS_CONFIG_MODIFY)
PATH_FACTORIES = {"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "Path"}
SKIP_METHODS = {"close", "raise_for_status", "json", "text", "content", "get_bucket", "Bucket",
                "set_debuglevel", "login", "starttls", "quit", "ehlo", "mount", "headers",
                "set_missing_host_key_policy", "load_system_host_keys", "close_all"}


@dataclass
class SinkUse:
    kind: str  # "net", "write", "exec"
    sink: Fact
    tokens: Taint
    anchor: tuple[str, int, int] | None = None


@dataclass
class FuncInfo:
    mod: str
    qual: str
    node: ast.AST | None
    params: list[str] = field(default_factory=list)
    ret: Taint = EMPTY
    sinks: dict[tuple, SinkUse] = field(default_factory=dict)
    calls: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class PyModule:
    path: str
    tree: ast.Module
    origin: Origin
    imports: dict[str, str] = field(default_factory=dict)
    exprs: dict[str, tuple[ast.AST, int]] = field(default_factory=dict)  # module constants
    taints: dict[str, Taint] = field(default_factory=dict)
    funcs: dict[str, FuncInfo] = field(default_factory=dict)  # includes "<module>", "<main>"
    classes: dict[str, list[str]] = field(default_factory=dict)  # class → method quals
    has_main: bool = False
    imported_paths: set[str] = field(default_factory=set)


def _line_span(node: ast.AST) -> tuple[int, int]:
    return getattr(node, "lineno", 1), getattr(node, "end_lineno", None) or getattr(
        node, "lineno", 1)


class PyProject:
    def __init__(self, facts: FactSet) -> None:
        self.facts = facts
        self.modules: dict[str, PyModule] = {}
        self.sinks: dict[tuple, SinkUse] = {}
        self.notes: list[str] = []

    # ------------------------------------------------------------------ load

    def add(self, path: str, text: str, origin: Origin = Origin.CODE) -> None:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError) as err:
            self.notes.append(f"{path}: not parsed as Python ({type(err).__name__})")
            return
        mod = PyModule(path, tree, origin)
        self.modules[path] = mod
        mod.funcs["<module>"] = FuncInfo(path, "<module>", None)
        mod.funcs["<main>"] = FuncInfo(path, "<main>", None)
        for node in tree.body:
            self._collect(mod, node, "")
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_main_guard(node.test):
                mod.has_main = True

    def _collect(self, mod: PyModule, node: ast.stmt, prefix: str) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                mod.imports[(a.asname or a.name).split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            for a in node.names:
                mod.imports[a.asname or a.name] = f"{base}.{a.name}".lstrip(".") \
                    if node.level == 0 else f"{base}.{a.name}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual = f"{prefix}{node.name}"
            params = [a.arg for a in node.args.posonlyargs + node.args.args
                      + node.args.kwonlyargs]
            if node.args.vararg:
                params.append(node.args.vararg.arg)
            if node.args.kwarg:
                params.append(node.args.kwarg.arg)
            mod.funcs[qual] = FuncInfo(mod.path, qual, node, params)
        elif isinstance(node, ast.ClassDef):
            mod.classes[node.name] = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._collect(mod, sub, f"{node.name}.")
                    mod.classes[node.name].append(f"{node.name}.{sub.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and value is not None:
                    mod.exprs[t.id] = (value, node.lineno)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, ast.stmt):
                    self._collect(mod, sub, prefix)

    # ------------------------------------------------------------------ resolve modules

    def find_module(self, dotted: str, from_path: str) -> PyModule | None:
        """The skill file that `import dotted` names, if there is one."""
        level = len(dotted) - len(dotted.lstrip("."))
        name = dotted.lstrip(".")
        base = PurePosixPath(from_path).parent
        if level:
            for _ in range(level - 1):
                base = base.parent
        rel = PurePosixPath(*name.split(".")) if name else PurePosixPath(".")
        wanted = [f"{rel}.py", f"{rel}/__init__.py"] if name else []
        best = None
        for path, mod in self.modules.items():
            p = PurePosixPath(path)
            if "#" in path:
                continue
            for w in wanted:
                if str(p) == w or str(p).endswith("/" + w):
                    here = (base / w) == p if not level else (base / w) == p
                    if best is None or here:
                        best = mod
        return best

    def resolve_import(self, mod: PyModule, qualified: str) -> tuple[PyModule, str] | None:
        """For `a.b.func` or `.util.func`: (module a.b, "func") if a.b is a skill file."""
        if "." not in qualified.lstrip("."):
            return None
        head, _, tail = qualified.rpartition(".")
        target = self.find_module(head, mod.path)
        if target is not None:
            return target, tail
        return None

    # ------------------------------------------------------------------ analysis

    def analyze(self, rounds: int = 3) -> None:
        for mod in self.modules.values():
            for q in mod.imports.values():
                head = q.rpartition(".")[0] if "." in q else q
                for cand in (q, head):
                    target = self.find_module(cand, mod.path)
                    if target is not None and target is not mod:
                        mod.imported_paths.add(target.path)
        for _ in range(rounds):
            for mod in self.modules.values():
                _Walker(self, mod).run()

    def reach(self, entries: set[str]) -> set[tuple[str, str]]:
        """(file, scope) pairs reachable from the entry modules."""
        seen: set[tuple[str, str]] = set()
        stack = [(p, s) for p in entries for s in ("<module>", "<main>") if p in self.modules]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            mod = self.modules.get(node[0])
            if mod is None or node[1] not in mod.funcs:
                continue
            stack.extend(mod.funcs[node[1]].calls)
            if node[1] == "<module>":
                stack.extend((p, "<module>") for p in mod.imported_paths)
        return seen

    def entry_modules(self, referenced: set[str]) -> set[str]:
        imported = {p for m in self.modules.values() for p in m.imported_paths}
        return {p for p, m in self.modules.items()
                if "#" not in p and (m.has_main or p in referenced or p not in imported)}


def _is_main_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__")


class _Walker:
    """Runs the module-level code, the `__main__` block and every function of one module."""

    def __init__(self, proj: PyProject, mod: PyModule) -> None:
        self.proj, self.mod, self.facts = proj, mod, proj.facts
        self.func: FuncInfo = mod.funcs["<module>"]
        self.scope = "<module>"
        self.taints: dict[str, Taint] = {}
        self.exprs: dict[str, tuple[ast.AST, int]] = {}
        self.kinds: dict[str, tuple] = {}
        self.params: set[str] = set()
        self.conds: list[str] = []

    # ------------------------------------------------------------------ drivers

    def run(self) -> None:
        mod = self.mod
        self.enter(mod.funcs["<module>"], {}, set())
        main_body: list[ast.stmt] = []
        for node in mod.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and _is_main_guard(node.test):
                main_body += node.body
                continue
            self.stmt(node)
        mod.taints = dict(self.taints)
        if main_body:
            self.enter(mod.funcs["<main>"], dict(self.taints), set(), exprs=self.exprs)
            self.stmts(main_body)
        for fn in mod.funcs.values():
            if fn.node is None:
                continue
            self.enter(fn, dict(mod.taints), set(fn.params))
            for p in fn.params:
                self.taints[p] = frozenset({f"p:{p}"})
            args = fn.node.args  # type: ignore[attr-defined]
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                if a.annotation is not None and "Path" in _short(a.annotation):
                    self.kinds[a.arg] = ("path",)
            body = fn.node.body  # type: ignore[attr-defined]
            fn.ret = EMPTY
            self.stmts(body)

    def enter(self, fn: FuncInfo, taints: dict, params: set, exprs: dict | None = None) -> None:
        self.func, self.scope = fn, fn.qual
        self.taints = taints
        self.exprs = dict(exprs if exprs is not None else {})
        self.kinds = {}
        self.params = params
        self.conds = []

    # ------------------------------------------------------------------ statements

    def stmts(self, body: list[ast.stmt]) -> None:
        for s in body:
            self.stmt(s)

    def stmt(self, s: ast.stmt) -> None:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if s.name in self.mod.funcs and self.func.node is not None:
                return  # a nested def is a function of its own only at top level
            self.stmts(s.body)
        elif isinstance(s, ast.ClassDef):
            return
        elif isinstance(s, ast.Assign):
            t = self.expr(s.value)
            for target in s.targets:
                self.bind(target, t, s.value, s.lineno)
        elif isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self.bind(s.target, self.expr(s.value), s.value, s.lineno)
        elif isinstance(s, ast.AugAssign):
            t = self.expr(s.value) | self.expr(s.target)
            self.bind(s.target, t, None, s.lineno)
        elif isinstance(s, ast.Return):
            if s.value is not None:
                self.func.ret = self.func.ret | self.expr(s.value)
        elif isinstance(s, ast.If):
            self.expr(s.test)
            cond = _short(s.test)
            if _is_main_guard(s.test):
                self.stmts(s.body)
                self.stmts(s.orelse)
                return
            self.conds.append(cond)
            self.stmts(s.body)
            self.conds.pop()
            self.conds.append(f"not ({cond})")
            self.stmts(s.orelse)
            self.conds.pop()
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            t = self.expr(s.iter)
            self.bind(s.target, t, None, s.lineno)
            self.stmts(s.body)
            self.stmts(s.orelse)
        elif isinstance(s, ast.While):
            self.expr(s.test)
            self.conds.append(_short(s.test))
            self.stmts(s.body)
            self.conds.pop()
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                t = self.expr(item.context_expr)
                if item.optional_vars is not None:
                    self.bind(item.optional_vars, t, item.context_expr, s.lineno)
            self.stmts(s.body)
        elif isinstance(s, ast.Try):
            self.stmts(s.body)
            for h in s.handlers:
                self.stmts(h.body)
            self.stmts(s.orelse)
            self.stmts(s.finalbody)
        elif isinstance(s, ast.Expr):
            self.expr(s.value)
        elif isinstance(s, (ast.Raise, ast.Assert, ast.Delete)):
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.expr):
                    self.expr(child)
        elif hasattr(ast, "Match") and isinstance(s, ast.Match):
            self.expr(s.subject)
            for case in s.cases:
                self.stmts(case.body)

    def bind(self, target: ast.AST, taint: Taint, value: ast.AST | None, line: int) -> None:
        if isinstance(target, ast.Name):
            self.taints[target.id] = taint
            if value is not None:
                self.exprs[target.id] = (value, line)
                kind = self.kind_of(value)
                if kind:
                    self.kinds[target.id] = kind
                else:
                    self.kinds.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for el in target.elts:
                self.bind(el, taint, None, line)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            base = target
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name):
                self.taints[base.id] = self.taints.get(base.id, EMPTY) | taint

    # ------------------------------------------------------------------ names and constants

    def lookup(self, name: str) -> tuple[ast.AST, int] | None:
        return self.exprs.get(name) or (
            None if name in self.params else self.mod.exprs.get(name))

    def is_var(self, name: str) -> bool:
        return (name in self.taints or name in self.exprs or name in self.params
                or name in self.mod.exprs)

    def const(self, node: ast.AST | None, depth: int = 0) -> str | None:
        """The string an expression always evaluates to; "{?}" marks an unknown part."""
        if node is None or depth > 6:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            found = self.lookup(node.id)
            return self.const(found[0], depth + 1) if found else None
        if isinstance(node, ast.JoinedStr):
            out = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    out.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    out.append(self.const(v.value, depth + 1) or "{?}")
            return "".join(out)
        if isinstance(node, ast.BinOp):
            a, b = self.const(node.left, depth + 1), self.const(node.right, depth + 1)
            if isinstance(node.op, ast.Add) and (a is not None or b is not None):
                return (a or "{?}") + (b or "{?}")
            if isinstance(node.op, ast.Div) and a is not None:
                return f"{a}/{b or '{?}'}"
            return None
        if isinstance(node, ast.Subscript):
            base = _dotted(node.value)
            idx = node.slice
            if base == "sys.argv" and isinstance(idx, ast.Constant):
                return f"argv[{idx.value}]"
            if base in ("os.environ", "environ") and isinstance(idx, ast.Constant):
                return f"${idx.value}"
            return None
        if isinstance(node, ast.Call):
            q = self.qual(node.func)
            name = q[0] if q else None
            if name in PATH_FACTORIES or name in ("os.path.join", "posixpath.join",
                                                  "ntpath.join"):
                parts = [self.const(a, depth + 1) or "{?}" for a in node.args]
                return "/".join(p.rstrip("/") if i < len(parts) - 1 else p
                                for i, p in enumerate(parts)) if parts else "."
            if name in ("pathlib.Path.home", "Path.home", "os.path.expanduser") or (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "home"):
                if node.args:
                    return self.const(node.args[0], depth + 1)
                return "~"
            if name in ("os.path.expandvars", "os.path.abspath", "os.path.realpath", "str",
                        "os.fspath", "os.path.normpath", "urllib.request.Request",
                        "requests.Request") and node.args:
                return self.const(node.args[0], depth + 1)
            if name == "os.getcwd":
                return "."
            if isinstance(node.func, ast.Attribute):
                m = node.func.attr
                recv = self.const(node.func.value, depth + 1)
                if m in ("expanduser", "resolve", "absolute", "as_posix", "strip", "lower"):
                    return recv
                if m in ("joinpath",) and recv is not None:
                    return "/".join([recv] + [self.const(a, depth + 1) or "{?}"
                                              for a in node.args])
                if m == "format" and recv is not None:
                    return recv
                if m in ("with_suffix", "with_name") and recv is not None:
                    return recv
            return None
        if isinstance(node, ast.Attribute) and node.attr in ("parent",):
            return self.const(node.value, depth + 1)
        return None

    def kind_of(self, node: ast.AST) -> tuple | None:
        """What an expression is: a path, a network client, an open file."""
        if isinstance(node, ast.Name):
            return self.kinds.get(node.id)
        if isinstance(node, ast.Call):
            q = self.qual(node.func)
            name = q[0] if q else None
            if name in PATH_FACTORIES or name in ("pathlib.Path.home", "Path.home",
                                                  "os.getcwd"):
                return ("path",)
            if name in ("open", "io.open", "codecs.open"):
                mode = self.const(self.arg(node, 1, "mode")) or "r"
                if any(c in mode for c in "wax+"):
                    f = self._open_fact(node)
                    return ("filew", f.idx) if f is not None else None
            if name in CLIENT_FACTORIES:
                return ("client",)
            if name and name.rsplit(".", 1)[-1] in P.WRITER_FACTORIES:
                return ("writer",)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("home", "joinpath", "resolve", "expanduser", "absolute",
                                      "with_suffix", "with_name"):
                    return ("path",) if self.kind_of(node.func.value) == ("path",) or \
                        node.func.attr == "home" else None
                if node.func.attr in ("client", "resource", "Session"):
                    return ("client",)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return ("path",) if self.kind_of(node.left) == ("path",) else None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return self.kind_of(node.value)
        return None

    def qual(self, node: ast.AST) -> tuple[str, bool] | None:
        """(dotted name, rooted in an import) for a call target, or None for a method call."""
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        root = self.mod.imports.get(cur.id)
        if root is not None:
            return ".".join([root, *reversed(parts)]), True
        if self.is_var(cur.id) and parts:
            return None
        return ".".join([cur.id, *reversed(parts)]), False

    def origin_lines(self, node: ast.AST | None) -> list[tuple[str, int, int]]:
        """Where the names in an expression were defined (module constants, locals)."""
        out: list[tuple[str, int, int]] = []
        if node is None:
            return out
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                found = self.lookup(n.id)
                if found is not None:
                    lo, hi = _line_span(found[0])
                    out.append((self.mod.path, min(lo, found[1]), hi))
        return out

    # ------------------------------------------------------------------ facts and sinks

    def emit(self, cap: C, action: str, node: ast.AST, *, target: str | None = None,
             destination: str | None = None, tags: set | None = None,
             origin_of: list[ast.AST | None] = ()) -> Fact:
        l0, l1 = _line_span(node)
        also: list[tuple[str, int, int]] = []
        for e in origin_of:
            for span in self.origin_lines(e):
                if span not in also and not (span[1] <= l0 <= span[2]):
                    also.append(span)
        t = set(tags or ())
        if target is not None and not target.replace("{?}", "").strip("/. "):
            target = None
        if target is None:
            t.add(DYNAMIC)
        return self.facts.add(Fact(cap, action, self.mod.path, l0, l1, target=target,
                                   destination=destination, origin=self.mod.origin,
                                   scope=self.scope, tags=t, also=also,
                                   conditions=[c for c in self.conds][-3:]))

    def sink(self, kind: str, fact: Fact, tokens: Taint, node: ast.AST) -> None:
        concrete = frozenset(t for t in tokens if isinstance(t, int) or t.startswith("p:"))
        if not concrete:
            return
        use = SinkUse(kind, fact, concrete, None)
        key = (kind, fact.idx, concrete)
        self.func.sinks.setdefault(key, use)
        if any(isinstance(t, int) for t in concrete):
            self.proj.sinks.setdefault(key, SinkUse(kind, fact, concrete,
                                                    (self.mod.path, *_line_span(node))))

    def describe(self, node: ast.AST | None) -> str | None:
        return self.const(node)

    # ------------------------------------------------------------------ expressions

    def expr(self, node: ast.AST | None) -> Taint:
        if node is None or isinstance(node, ast.Constant):
            return EMPTY
        if isinstance(node, ast.Name):
            self.reference(node.id)
            return self.taints.get(node.id) or (
                EMPTY if node.id in self.params else self.mod.taints.get(node.id, EMPTY))
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Attribute) and _dotted(node) in ("os.environ", "environ"):
            if (_dotted(node) == "environ" and self.mod.imports.get("environ") != "os.environ"):
                return EMPTY
            f = self.emit(C.ENV_READ, "read all environment variables", node,
                          target="os.environ (all)", tags={SENSITIVE})
            return frozenset({f.idx})
        if isinstance(node, ast.Subscript) and _dotted(node.value) in ("os.environ", "environ"):
            key = self.const(node.slice)
            f = self.emit(C.ENV_READ, f"read {key or 'an environment variable'}", node,
                          target=f"${key}" if key else None,
                          tags={SENSITIVE} if key is None or P.SECRET_NAME.search(key) else ())
            return frozenset({f.idx}) | self.expr(node.slice)
        if isinstance(node, (ast.Lambda,)):
            return self.expr(node.body)
        out: Taint = EMPTY
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                out = out | self.expr(child)
            elif isinstance(child, ast.comprehension):
                out = out | self.expr(child.iter)
                self.bind(child.target, self.expr(child.iter), None, 0)
                for cond in child.ifs:
                    self.expr(cond)
        return out

    def reference(self, name: str) -> None:
        """A bare name that is a skill function counts as a use (callbacks, map(fn))."""
        if name in self.mod.funcs and name not in ("<module>", "<main>"):
            self.func.calls.add((self.mod.path, name))
        elif name in self.mod.classes:
            for q in self.mod.classes[name]:
                self.func.calls.add((self.mod.path, q))
        elif (target := self.mod.imports.get(name)) is not None:
            hit = self.proj.resolve_import(self.mod, target)
            if hit is not None:
                self.func.calls.add((hit[0].path, hit[1]))
                if hit[1] in hit[0].classes:
                    for q in hit[0].classes[hit[1]]:
                        self.func.calls.add((hit[0].path, q))

    # ------------------------------------------------------------------ calls

    def callee(self, node: ast.Call) -> FuncInfo | None:
        func = node.func
        q = self.qual(func)
        if q is not None:
            name, rooted = q
            if not rooted and name in self.mod.funcs and name not in ("<module>", "<main>"):
                return self.mod.funcs[name]
            if not rooted and name in self.mod.classes:
                return self.mod.funcs.get(f"{name}.__init__")
            if rooted:
                hit = self.proj.resolve_import(self.mod, name)
                if hit is not None:
                    mod, fname = hit
                    return mod.funcs.get(fname) or mod.funcs.get(f"{fname}.__init__")
            return None
        if isinstance(func, ast.Attribute):
            names = [f for qn, f in self.mod.funcs.items()
                     if f.node is not None and qn.rsplit(".", 1)[-1] == func.attr and "." in qn]
            if len(names) == 1:
                return names[0]
        return None

    def apply_summary(self, fn: FuncInfo, node: ast.Call, args: dict[str, Taint]) -> Taint:
        self.func.calls.add((fn.mod, fn.qual))
        if fn.qual.endswith(".__init__"):
            cls = fn.qual.split(".")[0]
            for q in self.proj.modules[fn.mod].classes.get(cls, []):
                self.func.calls.add((fn.mod, q))

        def mapped(tokens: Taint) -> Taint:
            out: set = set()
            for tok in tokens:
                if isinstance(tok, str) and tok.startswith("p:"):
                    out |= args.get(tok[2:], EMPTY)
                else:
                    out.add(tok)
            return frozenset(out)

        for use in list(fn.sinks.values()):
            tokens = mapped(use.tokens)
            if tokens:
                self.sink(use.kind, use.sink, tokens, node)
        return mapped(fn.ret)

    def arg_map(self, fn: FuncInfo | None, node: ast.Call, pos: list[Taint],
                kw: dict[str, Taint], skip_self: bool) -> dict[str, Taint]:
        out: dict[str, Taint] = dict(kw)
        if fn is not None:
            params = [p for p in fn.params if not (skip_self and p in ("self", "cls"))]
            for name, t in zip(params, pos, strict=False):
                out[name] = t
        return out

    def call(self, node: ast.Call) -> Taint:
        func = node.func
        pos = [self.expr(a) for a in node.args]
        kw = {k.arg: self.expr(k.value) for k in node.keywords if k.arg}
        starred = EMPTY
        for k in node.keywords:
            if k.arg is None:
                starred = starred | self.expr(k.value)
        allargs: Taint = starred
        for t in [*pos, *kw.values()]:
            allargs = allargs | t

        q = self.qual(func)
        name = q[0] if q else None
        method = func.attr if isinstance(func, ast.Attribute) else None
        recv_node = func.value if (isinstance(func, ast.Attribute) and q is None) else None
        recv_taint = self.expr(recv_node) if recv_node is not None else EMPTY
        if q is None and isinstance(func, ast.Attribute):
            pass
        elif isinstance(func, ast.Call):
            recv_taint = self.expr(func)

        fn = self.callee(node)
        if fn is not None:
            args = self.arg_map(fn, node, pos, kw, skip_self=recv_node is not None)
            return self.apply_summary(fn, node, args) | recv_taint
        if method and q is None:
            self._method_edge(method)

        # builtins and library calls, by qualified name
        if name is not None:
            res = self.known_call(name, node, pos, kw, allargs)
            if res is not None:
                return res
        if recv_node is not None or isinstance(func, ast.Call):
            res = self.method_call(method or "", recv_node or func, node, pos, kw, allargs,
                                   recv_taint)
            if res is not None:
                return res
        return allargs | recv_taint

    def _method_edge(self, method: str) -> None:
        for qn, f in self.mod.funcs.items():
            if f.node is not None and "." in qn and qn.rsplit(".", 1)[-1] == method:
                self.func.calls.add((self.mod.path, qn))

    def arg(self, node: ast.Call, i: int, kw: str | None = None) -> ast.AST | None:
        if i < len(node.args):
            return node.args[i]
        if kw:
            for k in node.keywords:
                if k.arg == kw:
                    return k.value
        return None

    def known_call(self, name: str, node: ast.Call, pos: list[Taint], kw: dict[str, Taint],
                   allargs: Taint) -> Taint | None:
        a0 = self.arg(node, 0)
        s0 = self.const(a0)

        if name == "open" or name == "io.open" or name == "codecs.open":
            mode_node = self.arg(node, 1, "mode")
            mode = self.const(mode_node) if mode_node is not None else "r"
            if mode is None:
                mode = "r"
            write = any(c in mode for c in "wax+")
            if write:
                self.write_fact(node, s0, [a0], "write")
                return EMPTY
            kind = P.path_class(s0)
            f = self.emit(C.CREDENTIAL_ACCESS if kind == "credential" else C.FS_READ,
                          f"read {s0 or 'a file'}", node, target=s0, origin_of=[a0],
                          tags={SENSITIVE} if kind in ("credential", "sensitive") else ())
            return frozenset({f.idx}) | pos[0]

        if name in P.READ_FULL or name.rsplit(".", 1)[-1] in P.READ_SUFFIX:
            kind = P.path_class(s0)
            f = self.emit(C.CREDENTIAL_ACCESS if kind == "credential" else C.FS_READ,
                          f"read {s0 or 'a file'}", node, target=s0, origin_of=[a0],
                          tags={SENSITIVE} if kind in ("credential", "sensitive") else ())
            return frozenset({f.idx}) | allargs

        if name in P.ENV_CALLS or name == "os.environ.get":
            key = s0
            if name in ("os.environ.copy", "os.environ.items", "os.environ.keys",
                        "os.environ.values", "dotenv.dotenv_values", "dotenv.load_dotenv"):
                f = self.emit(C.ENV_READ, "read all environment variables", node,
                              target="os.environ (all)" if "dotenv" not in name else ".env",
                              tags={SENSITIVE})
            else:
                f = self.emit(C.ENV_READ, f"read {key or 'an environment variable'}", node,
                              target=f"${key}" if key else None,
                              tags={SENSITIVE} if key is None or P.SECRET_NAME.search(key)
                              else ())
            return frozenset({f.idx})

        if name in P.NET_CALLS:
            return self._net(name, node, pos, kw, allargs)

        if name in P.PROC_CALLS:
            return self._proc(node, allargs, name)

        if name in P.EVAL_CALLS:
            if name in ("importlib.import_module", "__import__") and s0 is not None:
                return allargs
            f = self.emit(C.CODE_EVAL, f"{name} a string", node, target=s0,
                          tags={DYNAMIC} if s0 is None else ())
            self.sink("exec", f, allargs, node)
            return allargs | {f.idx}

        if name in P.ENCODE_CALLS:
            f = self.emit(C.ENCODE_OBFUSCATE, f"{name.split('.')[-1]} the data", node,
                          target=name, tags={ENCODE})
            return allargs | {f.idx}

        if name in P.DELETE_CALLS:
            self.emit(C.FS_DELETE, f"delete {s0 or 'a path'}", node, target=s0,
                      origin_of=[a0])
            return EMPTY
        if name in P.ENUM_CALLS:
            self.emit(C.FS_ENUMERATE, f"list files under {s0 or 'a directory'}", node,
                      target=s0, origin_of=[a0])
            return EMPTY
        if name in ("shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree",
                    "shutil.move", "os.rename", "os.replace"):
            src, dst = self.const(a0), self.const(self.arg(node, 1))
            kind = P.path_class(src)
            toks: Taint = EMPTY
            if name != "shutil.move" and not name.startswith("os."):
                r = self.emit(C.CREDENTIAL_ACCESS if kind == "credential" else C.FS_READ,
                              f"read {src or 'a file'}", node, target=src, origin_of=[a0],
                              tags={SENSITIVE} if kind in ("credential", "sensitive") else ())
                toks = frozenset({r.idx})
            f = self.write_fact(node, dst, [self.arg(node, 1)], "copy to")
            self.sink("write", f, toks | allargs, node)
            return EMPTY
        if name in P.WRITE_CALLS:
            if name in ("json.dump", "pickle.dump", "yaml.dump", "yaml.safe_dump"):
                return self._dump(node, pos, allargs)
            t = self.const(a0) if name not in ("tempfile.mkstemp", "csv.writer") else None
            if name in ("os.makedirs", "os.mkdir", "os.chmod", "os.chown", "os.symlink",
                        "os.link", "shutil.make_archive"):
                self.write_fact(node, t, [a0], "create")
            return EMPTY
        if name in P.INTERACT_CALLS:
            self.emit(C.USER_INTERACTION, "ask the user for input", node, target="stdin")
            return EMPTY
        if name in ("print", "len", "str", "int", "float", "bool", "list", "dict", "set",
                    "tuple", "sorted", "enumerate", "zip", "range", "map", "filter", "sum",
                    "min", "max", "repr", "bytes", "bytearray", "isinstance", "getattr",
                    "any", "all", "reversed", "iter", "next", "abs", "round", "format"):
            return allargs
        if name in ("urllib.request.Request", "requests.Request"):
            return (pos[1] if len(pos) > 1 else EMPTY) | (
                frozenset().union(*[kw.get(k, EMPTY) for k in ("data", "json")]))
        return None

    def write_fact(self, node: ast.AST, path: str | None, origin: list, verb: str) -> Fact:
        kind = P.path_class(path)
        if kind == "persistence":
            return self.emit(C.PERSISTENCE, f"{verb} {path}", node, target=path,
                             origin_of=origin)
        if P.is_sysconfig(path) and kind != "sensitive":
            return self.emit(C.SYS_CONFIG_MODIFY, f"{verb} {path}", node, target=path,
                             origin_of=origin)
        return self.emit(C.FS_WRITE, f"{verb} {path or 'a file'}", node, target=path,
                         origin_of=origin)

    def _dump(self, node: ast.Call, pos: list[Taint], allargs: Taint) -> Taint:
        target = self.arg(node, 1, "fp")
        fact = self.file_fact(target)
        if fact is not None and pos:
            self.sink("write", fact, pos[0], node)
        return EMPTY

    def file_fact(self, target: ast.AST | None) -> Fact | None:
        if isinstance(target, ast.Name):
            k = self.kinds.get(target.id)
            if k and k[0] == "filew":
                return self.facts.items[k[1]]
        if isinstance(target, ast.Call):
            q = self.qual(target.func)
            if q and q[0] == "open":
                return self._open_fact(target)
        return None

    def _open_fact(self, call: ast.Call) -> Fact | None:
        l0, _ = _line_span(call)
        for f in self.facts.items:
            if f.file == self.mod.path and f.l0 == l0 and f.cap in WRITE_CAPS:
                return f
        return None

    def _net(self, name: str, node: ast.Call, pos: list[Taint], kw: dict[str, Taint],
             allargs: Taint) -> Taint:
        a0 = self.arg(node, 0, "url")
        url = self.const(a0)
        host = P.host_of(url)
        verb = name.rsplit(".", 1)[-1]
        if name in ("requests.Session", "httpx.Client", "httpx.AsyncClient",
                    "aiohttp.ClientSession", "urllib3.PoolManager"):
            return EMPTY
        if name.endswith(("create_connection", "SMTP", "SMTP_SSL", "FTP", "Telnet",
                          "SSHClient", "HTTPSConnection", "HTTPConnection")):
            host = host or (url if url and "/" not in url else None)
        f = self.emit(C.NET_OUTBOUND, f"{verb} {host or url or 'a remote host'}", node,
                      target=url, destination=host or url, tags={NET_IN},
                      origin_of=[a0])
        sends = verb in ("post", "put", "patch", "request", "urlopen", "stream")
        data: Taint = pos[0] if pos else EMPTY
        for key in ("data", "json", "files", "content", "body", "payload"):
            data = data | kw.get(key, EMPTY)
        if sends and len(pos) > 1:
            data = data | pos[1]
        if name == "urllib.request.urlretrieve":
            dst = self.const(self.arg(node, 1, "filename"))
            w = self.emit(C.FS_WRITE, f"save download to {dst or 'a file'}", node, target=dst,
                          origin_of=[self.arg(node, 1)])
            self.sink("write", w, frozenset({f.idx}), node)
        self.sink("net", f, data, node)
        return frozenset({f.idx}) | (pos[0] if name == "urllib.request.urlopen" else EMPTY)

    def _proc(self, node: ast.Call, allargs: Taint, name: str) -> Taint:
        a0 = self.arg(node, 0, "args")
        cmd = self.command_text(a0)
        f = self.emit(C.PROC_EXEC, f"run {cmd or 'a command'}"[:90], node, target=cmd,
                      origin_of=[a0])
        if cmd:
            analyze_command(cmd, self.mod.path, node.lineno, self.facts, origin=self.mod.origin,
                            scope=self.scope)
        self.sink("exec", f, allargs, node)
        return allargs | {f.idx}

    def command_text(self, node: ast.AST | None) -> str | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            parts = [self.const(e) or "{?}" for e in node.elts]
            return " ".join(parts) if parts else None
        return self.const(node)

    def method_call(self, method: str, recv: ast.AST, node: ast.Call, pos: list[Taint],
                    kw: dict[str, Taint], allargs: Taint, recv_taint: Taint) -> Taint | None:
        kind = self.kind_of(recv)
        path = self.const(recv)
        only_paths = method in ("read_text", "read_bytes", "write_text", "write_bytes",
                                "unlink", "rmdir", "glob", "rglob", "iterdir", "touch")
        single_arg = len(node.args) == 1 and not node.keywords
        looks_path = kind == ("path",) or only_paths or (
            path is not None and method in ("mkdir", "open", "rename", "replace")) or (
            method in ("rename", "replace") and single_arg)
        if kind == ("writer",) and method in ("write", "save") and node.args:
            dst = self.const(node.args[0])
            f = self.write_fact(node, dst, [node.args[0]], "write")
            self.sink("write", f, allargs, node)
            return EMPTY
        if method in P.WRITE_METHODS and node.args:
            dst = self.const(node.args[0])
            f = self.write_fact(node, dst, [node.args[0]], "write")
            self.sink("write", f, recv_taint, node)
            return EMPTY
        if method in ("rename", "replace") and looks_path and single_arg:
            self.write_fact(node, self.const(node.args[0]), [node.args[0]], "rename to")
            return EMPTY
        if method in P.PATH_READ_METHODS or (looks_path and method == "open"):
            writing = method == "open" and any(
                c in (self.const(self.arg(node, 0, "mode")) or "r") for c in "wax+")
            if writing:
                return EMPTY if self.write_fact(node, path, [recv], "write") else EMPTY
            kind_ = P.path_class(path)
            f = self.emit(C.CREDENTIAL_ACCESS if kind_ == "credential" else C.FS_READ,
                          f"read {path or 'a file'}", node, target=path, origin_of=[recv],
                          tags={SENSITIVE} if kind_ in ("credential", "sensitive") else ())
            return frozenset({f.idx}) | recv_taint
        if method in P.PATH_WRITE_METHODS and (looks_path or method.startswith("write_")):
            f = self.write_fact(node, path, [recv], "create" if method in (
                "touch", "mkdir") else "write")
            if method.startswith("write_"):
                self.sink("write", f, allargs, node)
            return EMPTY
        if method in P.PATH_DELETE_METHODS and looks_path:
            self.emit(C.FS_DELETE, f"delete {path or 'a path'}", node, target=path,
                      origin_of=[recv])
            return EMPTY
        if method in P.PATH_ENUM_METHODS and looks_path:
            pat = self.const(self.arg(node, 0))
            tgt = f"{path}/{pat}" if path and pat else path or pat
            self.emit(C.FS_ENUMERATE, f"list files under {tgt or 'a directory'}", node,
                      target=tgt, origin_of=[recv])
            return EMPTY
        # writes to an open file
        if method in ("write", "writelines") and isinstance(recv, ast.Name):
            f = self.file_fact(recv)
            if f is not None:
                self.sink("write", f, allargs, node)
                return EMPTY
        # network clients
        if kind == ("client",) and method not in SKIP_METHODS:
            url = self.const(self.arg(node, 0, "url"))
            if method in ("get", "head", "options") and not url:
                url = None
            f = self.emit(C.NET_OUTBOUND, f"{method} (network client)", node, target=url,
                          destination=P.host_of(url) or url, tags={NET_IN},
                          origin_of=[recv, self.arg(node, 0)])
            self.sink("net", f, allargs if method not in ("get", "head") else (
                pos[0] if pos else EMPTY), node)
            return frozenset({f.idx})
        if method in ("sendall", "sendto", "send"):
            return None
        return None


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def _short(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001 - unparse can fail on odd trees
        return "?"
    return text if len(text) <= 80 else text[:77] + "..."
