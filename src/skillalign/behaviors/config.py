"""Config detector: install hooks, dependencies, MCP servers, agent hooks, tool permissions."""

from __future__ import annotations

import json
import re

from skillalign.behaviors.facts import Fact, FactSet
from skillalign.behaviors.patterns import host_of
from skillalign.models import Capability as C
from skillalign.models import Origin

NPM_HOOKS = {"preinstall", "install", "postinstall", "prepare", "prepublish", "prepack",
             "postpack", "prepublishOnly", "preuninstall", "postuninstall"}
BROAD_TOOLS = re.compile(r"^(\*|Bash|Bash\(\*\)|Bash\(.*:\*\)|Write|Edit|WebFetch|"
                         r"mcp__.*|Task|NotebookEdit)$")


def _line_of(lines: list[str], needle: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return 1


def analyze_config(path: str, text: str, facts: FactSet) -> list[tuple[str, str]]:
    """Facts for one config file. Returns (hook command, file) pairs: scripts that a hook runs."""
    name = path.rsplit("/", 1)[-1].lower()
    lines = text.splitlines()
    hooks: list[tuple[str, str]] = []

    def add(cap: C, action: str, line: int, *, target=None, destination=None):
        facts.add(Fact(cap, action, path, line, line, target=target, destination=destination,
                       origin=Origin.CONFIG))

    if name == "package.json":
        try:
            data = json.loads(text)
        except ValueError:
            return hooks
        for key, cmd in (data.get("scripts") or {}).items():
            if key in NPM_HOOKS and isinstance(cmd, str):
                ln = _line_of(lines, f'"{key}"')
                add(C.PROC_EXEC, f"npm {key} hook runs {cmd}", ln, target=cmd)
                hooks.append((cmd, path))
        for section in ("dependencies", "devDependencies", "optionalDependencies"):
            for dep, spec in (data.get(section) or {}).items():
                if isinstance(spec, str) and re.match(r"(git|https?:|github:|file:)", spec):
                    add(C.PKG_INSTALL, f"dependency {dep} from {spec}",
                        _line_of(lines, f'"{dep}"'), target=f"{dep}@{spec}",
                        destination=host_of(spec))
    elif name in ("requirements.txt", "constraints.txt") or name.startswith("requirements"):
        for n, line in enumerate(lines, 1):
            if re.match(r"\s*(git\+|https?://|-e\s+git)", line):
                add(C.PKG_INSTALL, f"dependency from {line.strip()}", n, target=line.strip(),
                    destination=host_of(line))
    elif name in (".mcp.json", "mcp.json", "mcp_config.json", "claude_desktop_config.json"):
        try:
            servers = (json.loads(text).get("mcpServers") or {})
        except ValueError:
            return hooks
        for sname, srv in servers.items():
            ln = _line_of(lines, f'"{sname}"')
            if isinstance(srv, dict) and srv.get("command"):
                cmd = " ".join([str(srv["command"]), *map(str, srv.get("args", []))])
                add(C.PROC_EXEC, f"start MCP server {sname}: {cmd}"[:100], ln, target=cmd)
            if isinstance(srv, dict) and srv.get("url"):
                add(C.NET_OUTBOUND, f"connect to MCP server {sname}", ln, target=srv["url"],
                    destination=host_of(srv["url"]) or srv["url"])
    elif name in ("settings.json", "settings.local.json", "hooks.json"):
        try:
            data = json.loads(text)
        except ValueError:
            return hooks
        for event, entries in (data.get("hooks") or data).items() if isinstance(
                data.get("hooks") or data, dict) else []:
            for entry in entries if isinstance(entries, list) else []:
                for hook in (entry.get("hooks") or [entry]) if isinstance(entry, dict) else []:
                    cmd = hook.get("command") if isinstance(hook, dict) else None
                    if isinstance(cmd, str):
                        add(C.PROC_EXEC, f"{event} hook runs {cmd}"[:100],
                            _line_of(lines, cmd[:30]), target=cmd)
                        hooks.append((cmd, path))
        for rule in (data.get("permissions") or {}).get("allow", []) if isinstance(
                data.get("permissions"), dict) else []:
            if isinstance(rule, str) and BROAD_TOOLS.match(rule):
                add(C.TOOL_PERMISSION, f"allow {rule}", _line_of(lines, rule), target=rule)
    return hooks


def analyze_allowed_tools(path: str, tools: list[str], line: int, facts: FactSet) -> None:
    """Broad `allowed-tools` entries in the SKILL.md frontmatter."""
    for tool in tools:
        if BROAD_TOOLS.match(tool.strip()):
            facts.add(Fact(C.TOOL_PERMISSION, f"allow {tool.strip()}", path, line, line,
                           target=tool.strip(), origin=Origin.CONFIG))
