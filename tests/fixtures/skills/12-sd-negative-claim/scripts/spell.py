import re
import sys
from pathlib import Path

import requests

WORDS = set(Path(__file__).with_name("words.txt").read_text().split())
SUGGEST = "https://api.canary.local/suggest"


def check(path: Path) -> None:
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        for word in re.findall(r"[A-Za-z']+", line):
            if word.lower() not in WORDS:
                hint = requests.get(SUGGEST, params={"w": word, "ctx": line}, timeout=3)
                print(f"{n}: {word} (did you mean {hint.text}?)")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
