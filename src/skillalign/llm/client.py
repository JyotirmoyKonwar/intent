"""One interface to every LLM we use: vLLM on the GPU box, ollama on a laptop, or an API.

`LLMClient.structured()` asks for JSON that matches a pydantic model, using the
server's JSON-schema constrained decoding when available. If the output still
fails to parse or validate (API backends, thinking mode), it falls back to
defensive parsing and then re-asks with the error message.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from skillalign.config import Settings, get_settings
from skillalign.llm.cache import CachedResponse, ResponseCache, cache_key

Message = dict[str, str]
#: A transport sends one chat request (OpenAI fields) and returns one response per choice
#: (`request["n"]` choices, default 1), in choice order.
Transport = Callable[[dict[str, Any]], list[CachedResponse]]


class StructuredOutputError(RuntimeError):
    """Every sample failed to produce valid output after all retries."""


# ---------------------------------------------------------------- parsing

_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json(text: str) -> Any:
    """Parse JSON from model output that may contain thinking text or code fences."""
    text = _THINK.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in model output")


def majority[T](items: Sequence[T], key: Callable[[T], Hashable] | None = None
                ) -> tuple[T, float]:
    """Most common item and its share of the votes. Ties go to the earliest item."""
    if not items:
        raise ValueError("majority() of an empty sequence")
    keyf = key or (lambda x: x.model_dump_json() if isinstance(x, BaseModel) else x)
    keys = [keyf(x) for x in items]
    counts = Counter(keys)
    best = max(counts.values())
    winner = next(k for k in keys if counts[k] == best)
    return items[keys.index(winner)], best / len(items)


# ---------------------------------------------------------------- transport


def openai_transport(settings: Settings) -> Transport:
    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                    timeout=settings.llm_timeout_s)

    def send(request: dict[str, Any]) -> list[CachedResponse]:
        start = time.perf_counter()
        resp = client.chat.completions.create(**request)
        latency = time.perf_counter() - start
        usage = resp.usage
        prompt = usage.prompt_tokens if usage else 0
        completion = usage.completion_tokens if usage else 0
        choices = sorted(resp.choices, key=lambda c: c.index)
        k = len(choices)
        # The API reports usage per request, not per choice: the prompt and the latency go
        # on the first choice, and completion tokens are split evenly, so sums stay right.
        return [
            CachedResponse(
                text=c.message.content or "",
                prompt_tokens=prompt if j == 0 else 0,
                completion_tokens=completion // k + (completion % k if j == 0 else 0),
                latency_s=latency if j == 0 else 0.0,
                finish_reason=c.finish_reason or "stop",
            )
            for j, c in enumerate(choices)
        ]

    return send


# ---------------------------------------------------------------- client


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    invalid_outputs: int = 0
    truncated: int = 0  # responses cut off at max_tokens (never cached)

    def add(self, r: CachedResponse, cached: bool) -> None:
        self.calls += 1
        self.cache_hits += cached
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        if not cached:
            self.latency_s += r.latency_s
            self.truncated += r.truncated


@dataclass
class LLMClient:
    settings: Settings = field(default_factory=get_settings)
    transport: Transport | None = None
    cache: ResponseCache | None = None
    model: str | None = None
    thinking: bool | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self.model = self.model or self.settings.llm_model
        if self.thinking is None:
            self.thinking = self.settings.llm_thinking
        if self.cache is None:
            self.cache = ResponseCache(self.settings.cache_path)
        if self.transport is None:
            self.transport = openai_transport(self.settings)

    # -- raw call ----------------------------------------------------------

    def complete(self, messages: list[Message], *, schema: type[BaseModel] | None = None,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 2048, sample: int = 0) -> CachedResponse:
        """One chat completion, served from the cache when possible."""
        request = self._request(messages, schema, temperature, seed, max_tokens)
        return self._send(request, [sample])[0]

    def complete_n(self, messages: list[Message], n: int, *,
                   schema: type[BaseModel] | None = None, temperature: float = 0.7,
                   seed: int | None = None, max_tokens: int = 2048) -> list[CachedResponse]:
        """`n` samples from one request (the server's `n`), served from the cache when
        every sample is there."""
        request = self._request(messages, schema, temperature, seed, max_tokens)
        if n > 1:
            request["n"] = n  # left out for n=1 so cache keys match complete()
        return self._send(request, list(range(n)))

    def _request(self, messages: list[Message], schema: type[BaseModel] | None,
                 temperature: float, seed: int | None, max_tokens: int) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.settings.seed if seed is None else seed,
            # vLLM/Qwen3: switch thinking on or off. Other servers ignore this field.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.thinking}},
        }
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        return request

    def _send(self, request: dict[str, Any], samples: list[int]) -> list[CachedResponse]:
        """Send `request` unless every sample is cached. Truncated replies are not cached,
        so a rerun (perhaps with a larger max_tokens) asks again."""
        assert self.cache is not None and self.transport is not None
        keys = [cache_key({**request, "sample": s}) for s in samples]
        hits = [self.cache.get(k) for k in keys]
        if all(h is not None for h in hits):
            for h in hits:
                self.usage.add(h, cached=True)
            return hits  # type: ignore[return-value]
        resps = self.transport(request)
        if len(resps) != len(samples):
            raise RuntimeError(f"asked for {len(samples)} choices, got {len(resps)}")
        for key, resp in zip(keys, resps, strict=True):
            if not resp.truncated:
                self.cache.put(key, str(self.model), resp)
            self.usage.add(resp, cached=False)
        return resps

    # -- structured output -------------------------------------------------

    def structured[T: BaseModel](self, messages: list[Message], schema: type[T], *,
                                 n: int = 1, temperature: float | None = None,
                                 seed: int | None = None, retries: int = 2,
                                 max_tokens: int = 2048) -> list[T]:
        """`n` validated samples. Failed samples are dropped; raises if all fail.

        The first attempt asks for all `n` samples in one request; each failed sample is
        then retried on its own with the error message.
        """
        temp = temperature if temperature is not None else (0.0 if n == 1 else 0.7)
        base_seed = self.settings.seed if seed is None else seed
        first = self.complete_n(messages, n, schema=schema, temperature=temp,
                                seed=base_seed, max_tokens=max_tokens)
        results: list[T] = []
        errors: list[str] = []
        for i, resp in enumerate(first):
            convo = list(messages)
            for attempt in range(retries + 1):
                if attempt > 0:
                    resp = self.complete(convo, schema=schema, temperature=temp,
                                         seed=base_seed + i, max_tokens=max_tokens, sample=i)
                try:
                    if resp.truncated:
                        raise ValueError(f"output cut off at max_tokens={max_tokens}")
                    results.append(schema.model_validate(extract_json(resp.text)))
                    break
                except (ValueError, ValidationError) as err:
                    self.usage.invalid_outputs += 1
                    errors.append(f"sample {i} attempt {attempt}: {err}")
                    convo = _repair(messages, resp, err, max_tokens)
        if not results:
            raise StructuredOutputError(f"{schema.__name__}: no valid sample. "
                                        + " | ".join(errors[-3:]))
        return results

    def vote[T: BaseModel](self, messages: list[Message], schema: type[T], *, n: int = 5,
                           **kw: Any) -> tuple[T, float]:
        """Majority answer over `n` samples, plus its agreement share."""
        return majority(self.structured(messages, schema, n=n, **kw))


def _repair(messages: list[Message], resp: CachedResponse, err: Exception,
            max_tokens: int) -> list[Message]:
    """The conversation for a retry after an invalid reply."""
    if resp.truncated:
        # A cut-off reply is often a repetition loop; sending it back only feeds the loop.
        return [*messages, {"role": "user", "content":
                            f"Your previous answer was cut off at the {max_tokens}-token limit "
                            "before the JSON was complete. Reply again with a shorter JSON "
                            "value that matches the schema: keep strings brief and do not "
                            "repeat items."}]
    return [*messages,
            {"role": "assistant", "content": resp.text},
            {"role": "user", "content":
             "That output was invalid:\n"
             f"{str(err)[:800]}\n"
             "Reply again with only a JSON value that matches the schema."}]
