"""File roles and languages from names, extensions and shebangs."""

from __future__ import annotations

from pathlib import PurePosixPath

from skillalign.models import FileRole

EXT_LANG = {
    ".py": "python", ".pyw": "python",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ksh": "shell",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".ps1": "powershell", ".psm1": "powershell", ".bat": "batch", ".cmd": "batch",
    ".rb": "ruby", ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin",
    ".php": "php", ".pl": "perl", ".lua": "lua", ".r": "r", ".swift": "swift",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".sql": "sql", ".applescript": "applescript", ".scpt": "applescript",
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown", ".rst": "rst",
    ".txt": "text", ".html": "html", ".htm": "html", ".css": "css",
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".xml": "xml", ".csv": "csv", ".tsv": "csv", ".svg": "svg",
    ".env": "dotenv",
}

#: Fenced-code info strings → language.
FENCE_LANG = {
    "python": "python", "py": "python", "python3": "python",
    "bash": "shell", "sh": "shell", "shell": "shell", "zsh": "shell", "console": "shell",
    "shell-session": "shell", "terminal": "shell",
    "javascript": "javascript", "js": "javascript", "node": "javascript", "jsx": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
    "powershell": "powershell", "ps1": "powershell", "pwsh": "powershell",
    "json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml",
    "ruby": "ruby", "go": "go", "rust": "rust", "sql": "sql",
    "html": "html", "css": "css", "applescript": "applescript",
}

CODE_LANGS = {"python", "shell", "javascript", "typescript", "powershell", "batch", "ruby",
              "go", "rust", "java", "kotlin", "php", "perl", "lua", "r", "swift", "c", "cpp",
              "csharp", "applescript"}
#: Languages we analyze deeply (PLAN.md §10); the rest get a raw-code LLM fallback.
DEEP_LANGS = {"python", "shell", "javascript", "typescript"}

CONFIG_NAMES = {
    "package.json", "package-lock.json", "requirements.txt", "pyproject.toml", "setup.cfg",
    "pipfile", "pipfile.lock", "poetry.lock", "uv.lock", "dockerfile", "makefile",
    ".mcp.json", "mcp.json", "settings.json", "settings.local.json", "hooks.json",
    "tsconfig.json", ".npmrc", ".env", ".env.example", "gemfile", "cargo.toml", "go.mod",
    "docker-compose.yml", "docker-compose.yaml", "plugin.json", "marketplace.json",
}
CONFIG_LANGS = {"json", "yaml", "toml", "ini", "xml", "dotenv"}
DOC_LANGS = {"markdown", "rst", "text", "html"}

_SHEBANG = {"python": "python", "bash": "shell", "sh": "shell", "zsh": "shell",
            "node": "javascript", "deno": "typescript", "ruby": "ruby", "perl": "perl"}


def shebang_language(first_line: str) -> str | None:
    if not first_line.startswith("#!"):
        return None
    words = first_line[2:].replace("/", " ").split()
    for w in reversed(words):
        base = w.rstrip("0123456789.")
        if base in _SHEBANG:
            return _SHEBANG[base]
    return None


def detect_language(path: str, first_line: str = "") -> str | None:
    p = PurePosixPath(path)
    name = p.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "make"
    if name.startswith("requirements") and name.endswith(".txt"):
        return "requirements"
    return EXT_LANG.get(p.suffix.lower()) or shebang_language(first_line)


def file_role(path: str, language: str | None, *, is_root_skill_md: bool = False) -> FileRole:
    name = PurePosixPath(path).name.lower()
    if is_root_skill_md:
        return FileRole.SKILL_MD
    if name.split(".")[0] in ("license", "licence", "copying", "notice", "third_party_notices"):
        return FileRole.DATA  # legal text makes no claims about the skill
    if name in CONFIG_NAMES or name.startswith("requirements") or language in (
            "dockerfile", "make", "requirements"):
        return FileRole.CONFIG
    if language in CODE_LANGS:
        return FileRole.CODE
    if language in DOC_LANGS:
        return FileRole.DOC
    if language in CONFIG_LANGS:
        return FileRole.CONFIG
    return FileRole.DATA
