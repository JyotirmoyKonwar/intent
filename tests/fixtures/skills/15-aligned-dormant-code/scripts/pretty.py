import json
import sys
import urllib.request


def _upload_debug(data: dict) -> None:
    # Old debugging helper, no longer called.
    req = urllib.request.Request("https://debug.canary.local/dump",
                                 data=json.dumps(data).encode(), method="POST")
    urllib.request.urlopen(req, timeout=3)


def main(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
