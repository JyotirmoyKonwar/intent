import sys
from pathlib import Path


def clean(directory: Path) -> int:
    removed = 0
    for path in directory.rglob("*.tmp"):
        path.unlink()
        removed += 1
    return removed


if __name__ == "__main__":
    print(f"removed {clean(Path(sys.argv[1]))} files")
