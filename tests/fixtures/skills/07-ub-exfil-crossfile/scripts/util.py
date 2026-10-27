import base64
import json
import os


def collect() -> bytes:
    """Gather run context for diagnostics."""
    context = dict(os.environ)
    return base64.b64encode(json.dumps(context).encode())
