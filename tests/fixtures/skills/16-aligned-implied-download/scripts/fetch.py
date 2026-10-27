import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


def fetch(url: str) -> Path:
    out_dir = Path("downloads")
    out_dir.mkdir(exist_ok=True)
    target = out_dir / (Path(urlparse(url).path).name or "download")
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
    return target


if __name__ == "__main__":
    print(fetch(sys.argv[1]))
