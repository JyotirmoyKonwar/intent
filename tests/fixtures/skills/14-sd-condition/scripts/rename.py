import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

DATETIME_ORIGINAL = 36867


def taken(path: Path) -> datetime:
    raw = Image.open(path).getexif().get_ifd(0x8769)[DATETIME_ORIGINAL]
    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for path in sorted(args.folder.glob("*.jpg")):
        target = path.with_name(taken(path).strftime("%Y-%m-%d_%H%M%S") + ".jpg")
        print(f"{path.name} -> {target.name}")
        path.rename(target)


if __name__ == "__main__":
    main()
