import csv
import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    with src.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = src.with_suffix(".json")
    out.write_text(json.dumps(rows, indent=2))
    return out


if __name__ == "__main__":
    print(convert(Path(sys.argv[1])))
