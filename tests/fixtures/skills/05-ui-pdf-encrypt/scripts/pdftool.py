import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def merge(inputs: list[str], out: str) -> None:
    writer = PdfWriter()
    for path in inputs:
        for page in PdfReader(path).pages:
            writer.add_page(page)
    writer.write(out)


def split(src: str, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(PdfReader(src).pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(Path(out_dir) / f"page-{i}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["merge", "split"])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    if args.command == "merge":
        merge(args.inputs, args.out)
    else:
        split(args.inputs[0], args.out)


if __name__ == "__main__":
    main()
