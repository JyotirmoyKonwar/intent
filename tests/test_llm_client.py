import json
import sqlite3
import sys

import pytest
from pydantic import BaseModel

from skillalign.config import Settings
from skillalign.llm import LLMClient, StructuredOutputError, extract_json, majority
from skillalign.llm.cache import CachedResponse, ResponseCache
from skillalign.models import ChecklistAnswer

TRUNC = object()  # marks a scripted reply that was cut off at max_tokens


class FakeTransport:
    """Returns scripted replies in order (`n` per request) and records every request.

    A reply is a string, or `(text, TRUNC)` for one that hit max_tokens.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> list[CachedResponse]:
        self.requests.append(request)
        out = []
        for _ in range(request.get("n", 1)):
            reply = self.replies.pop(0)
            text, finish = (reply[0], "length") if isinstance(reply, tuple) else (reply, "stop")
            out.append(CachedResponse(text=text, prompt_tokens=10, completion_tokens=5,
                                      latency_s=0.01, finish_reason=finish))
        return out


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


def test_vote_sends_n_samples_in_one_request():
    t = FakeTransport(YES, NO, YES, YES, NO)
    winner, share = make_client(t).vote(MSG, ChecklistAnswer, n=5)
    assert winner.achieves_action == "yes" and share == pytest.approx(0.6)
    assert len(t.requests) == 1
    assert t.requests[0]["n"] == 5 and t.requests[0]["temperature"] > 0


def test_single_sample_request_has_no_n():
    # Keeps cache keys for n=1 the same as before `n` was sent.
    t = FakeTransport(YES)
    make_client(t).structured(MSG, ChecklistAnswer)
    assert "n" not in t.requests[0]


def test_vote_is_cached_as_a_whole():
    cache = ResponseCache(None)
    make_client(FakeTransport(YES, NO, YES), cache).vote(MSG, ChecklistAnswer, n=3)
    client = make_client(FakeTransport(), cache)
    winner, share = client.vote(MSG, ChecklistAnswer, n=3)
    assert winner.achieves_action == "yes" and share == pytest.approx(2 / 3)
    assert client.usage.cache_hits == 3


def test_failed_sample_in_batch_is_retried_alone():
    t = FakeTransport(YES, "garbage", YES, NO)  # 3-sample batch, then one retry
    client = make_client(t)
    answers = client.structured(MSG, ChecklistAnswer, n=3)
    assert [a.achieves_action for a in answers] == ["yes", "no", "yes"]  # sample order
    assert len(t.requests) == 2 and "n" not in t.requests[1]
    assert t.requests[1]["seed"] == t.requests[0]["seed"] + 1  # sample 1's own seed


def test_truncated_reply_is_not_cached():
    cache = ResponseCache(None)
    t = FakeTransport((YES, TRUNC))
    client = make_client(t, cache)
    resp = client.complete(MSG)
    assert resp.truncated and client.usage.truncated == 1
    t.replies.append(NO)
    assert client.complete(MSG).text == NO  # asked again, not served from the cache
    assert len(t.requests) == 2


def test_truncated_reply_is_invalid_even_if_it_parses():
    # Constrained decoding can leave a parseable prefix; a cut-off reply is never trusted.
    t = FakeTransport((YES, TRUNC), NO)
    client = make_client(t)
    [answer] = client.structured(MSG, ChecklistAnswer, max_tokens=100)
    assert answer.achieves_action == "no"
    assert client.usage.invalid_outputs == 1
    retry = t.requests[1]["messages"]
    assert retry[-1]["role"] == "user" and "cut off at the 100-token limit" in retry[-1]["content"]
    assert all(m["role"] != "assistant" for m in retry)  # the cut-off text is not sent back


def test_finish_reason_round_trips_through_cache(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    cache.put("k", "m", CachedResponse("x", 1, 2, 0.1, finish_reason="length"))
    assert ResponseCache(tmp_path / "c.sqlite").get("k").finish_reason == "length"


def test_cache_from_before_finish_reason_still_loads(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, model TEXT, text TEXT,"
               " prompt_tokens INTEGER, completion_tokens INTEGER, latency_s REAL,"
               " created TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.execute("INSERT INTO responses (key, model, text, prompt_tokens, completion_tokens,"
               " latency_s) VALUES ('k', 'm', 'old', 1, 2, 0.1)")
    db.commit()
    db.close()
    hit = ResponseCache(path).get("k")
    assert hit is not None and hit.text == "old" and not hit.truncated


def test_openai_transport_splits_choices(monkeypatch):
    from types import SimpleNamespace as NS

    from skillalign.llm import client as client_mod

    reply = NS(
        usage=NS(prompt_tokens=100, completion_tokens=31),
        choices=[NS(index=1, finish_reason="length", message=NS(content="b")),
                 NS(index=0, finish_reason="stop", message=NS(content="a"))],
    )
    fake_openai = NS(OpenAI=lambda **kw: NS(chat=NS(completions=NS(create=lambda **r: reply))))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    send = client_mod.openai_transport(Settings(llm_model="m"))
    a, b = send({"n": 2})
    assert (a.text, a.finish_reason, b.text, b.finish_reason) == ("a", "stop", "b", "length")
    assert a.prompt_tokens + b.prompt_tokens == 100
    assert a.completion_tokens + b.completion_tokens == 31
    assert b.latency_s == 0.0


def test_majority_tie_goes_to_first():
    class X(BaseModel):
        v: int

    assert majority([X(v=1), X(v=2)])[0].v == 1
    assert majority(["a", "b", "b"]) == ("b", pytest.approx(2 / 3))
