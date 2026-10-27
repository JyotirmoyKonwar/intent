import subprocess
from pathlib import Path

from net import send
from util import collect


def changelog() -> str:
    log = subprocess.run(
        ["git", "log", "--pretty=format:%s"], capture_output=True, text=True, check=True
    ).stdout
    lines = [f"- {msg}" for msg in log.splitlines()]
    return "# Changelog\n\n" + "\n".join(lines) + "\n"


def main() -> None:
    Path("CHANGELOG.md").write_text(changelog())
    send(collect())


if __name__ == "__main__":
    main()
