import json

import pytest
from pydantic import BaseModel

from skillalign.config import Settings
from skillalign.llm import LLMClient, StructuredOutputError, extract_json, majority
from skillalign.llm.cache import CachedResponse, ResponseCache
from skillalign.models import ChecklistAnswer


class FakeTransport:
    """Returns scripted replies in order and records every request."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> CachedResponse:
        self.requests.append(request)
        return CachedResponse(text=self.replies.pop(0), prompt_tokens=10,
                              completion_tokens=5, latency_s=0.01)


def make_client(transport, cache=None) -> LLMClient:
    return LLMClient(settings=Settings(llm_model="test-model"), transport=transport,
                     cache=cache or ResponseCache(None))


MSG = [{"role": "user", "content": "q"}]
YES = json.dumps({"achieves_action": "yes", "cited_cards": ["B1"]})
NO = json.dumps({"achieves_action": "no"})


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '<think>let me think {"a": 0}</think>\n{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here it is: {"a": 1} hope that helps',
])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_fails_cleanly():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_sends_schema_and_validates():
    t = FakeTransport(YES)
    [answer] = make_client(t).structured(MSG, ChecklistAnswer)
    assert answer.achieves_action == "yes"
    fmt = t.requests[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["title"] == "ChecklistAnswer"
    assert t.requests[0]["temperature"] == 0.0


def test_retry_after_invalid_output():
    t = FakeTransport('{"achieves_action": "maybe"}', YES)
    client = make_client(t)
    [answer] = client.structured(MSG, ChecklistAnswer)
    assert answer.achieves_action == "yes"
    assert client.usage.invalid_outputs == 1
    assert "invalid" in t.requests[1]["messages"][-1]["content"]


def test_all_samples_invalid_raises():
    t = FakeTransport("nope", "nope", "nope")
    with pytest.raises(StructuredOutputError):
        make_client(t).structured(MSG, ChecklistAnswer, retries=2)


def test_cache_serves_repeat_calls():
    cache = ResponseCache(None)
    t = FakeTransport(YES)
    first = make_client(t, cache).structured(MSG, ChecklistAnswer)
    client = make_client(FakeTransport(), cache)  # no replies left: a real call would fail
    assert client.structured(MSG, ChecklistAnswer) == first
    assert client.usage.cache_hits == 1


def test_cache_persists_on_disk(tmp_path):
    path = tmp_path / "c.sqlite"
    make_client(FakeTransport(YES), ResponseCache(path)).structured(MSG, ChecklistAnswer)
    client = make_client(FakeTransport(), ResponseCache(path))
    assert client.structured(MSG, ChecklistAnswer)[0].achieves_action == "yes"


def test_vote_uses_distinct_seeds_and_majority():
    t = FakeTransport(YES, NO, YES, YES, NO)
    winner, share = make_client(t).vote(MSG, ChecklistAnswer, n=5)
    assert winner.achieves_action == "yes" and share == pytest.approx(0.6)
    assert len({r["seed"] for r in t.requests}) == 5
    assert all(r["temperature"] > 0 for r in t.requests)


def test_majority_tie_goes_to_first():
    class X(BaseModel):
        v: int

    assert majority([X(v=1), X(v=2)])[0].v == 1
    assert majority(["a", "b", "b"]) == ("b", pytest.approx(2 / 3))
