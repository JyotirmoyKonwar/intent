"""A fake transport that answers by looking at the prompt, for the LLM-backed bench tests."""

from __future__ import annotations

import json

from skillalign.config import Settings
from skillalign.llm import LLMClient
from skillalign.llm.cache import CachedResponse, ResponseCache

TRUNC = "length"


def _conversation(request: dict) -> str:
    """Every message in the request, joined, so a retry still names the skill or chunk."""
    return "\n".join(m["content"] for m in request["messages"])


class Router:
    """`route(conversation_text)` returns `(reply_text, finish_reason)`. Records every request."""

    def __init__(self, route):
        self.route = route
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> list[CachedResponse]:
        self.requests.append(request)
        text, finish = self.route(_conversation(request))
        return [CachedResponse(text=text, prompt_tokens=10, completion_tokens=5,
                               latency_s=0.0, finish_reason=finish)]

    def count(self, needle: str) -> int:
        return sum(needle in _conversation(r) for r in self.requests)


def client_for(router: Router, **kw) -> LLMClient:
    return LLMClient(settings=Settings(llm_model="test-model"), transport=router,
                     cache=ResponseCache(None), **kw)


def claims_json(*claims: dict) -> str:
    return json.dumps({"claims": list(claims)})
