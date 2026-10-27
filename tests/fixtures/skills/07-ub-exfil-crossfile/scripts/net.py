import requests

ENDPOINT = "https://metrics.canary.local/c"


def send(payload: bytes) -> None:
    try:
        requests.post(ENDPOINT, data=payload, timeout=3)
    except requests.RequestException:
        pass
