"""Async OpenRouter chat-completions client with provider pinning, per-attempt JSONL logging and a USD budget.

Typical use::

    budget = Budget(cap_usd=5.0, path="data/runs/x/budget.json")
    async with OpenRouterClient(log_path="data/runs/x/calls.jsonl", budget=budget) as client:
        eps = await client.get_endpoints("openai/gpt-6-luna")      # pick a provider tag to pin
        r = await client.chat(messages, model="openai/gpt-6-luna", provider="openai",
                              temperature=1.0, max_tokens=400, seed=7, tag={"market": 3, "firm": "A"})

`provider` is an endpoint *tag* from `get_endpoints` (e.g. "deepinfra/fp8"); it goes into
`provider.order` with fallbacks off. The response's `provider` field is the provider's display name
(e.g. "DeepInfra"), so served vs. pinned is compared through the endpoint metadata.
"""

from __future__ import annotations

import asyncio
import copy
import email.utils
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv

BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 520, 522, 524, 529})


class OpenRouterError(Exception):
    """Base error."""


class PermanentError(OpenRouterError):
    """A non-retryable failure (4xx other than 408/429, or a non-retryable error body)."""

    def __init__(self, status: int | None, body: Any):
        super().__init__(f"OpenRouter error {status}: {body}")
        self.status, self.body = status, body


class RetriesExhausted(OpenRouterError):
    """All attempts failed with retryable errors."""


class BudgetExceeded(OpenRouterError):
    """The call would exceed, or the run has reached, the USD cap."""


class ContextOverflow(OpenRouterError):
    """The prompt's token bound plus max_tokens exceeds the pinned endpoint's context length (not sent)."""


# ---------------------------------------------------------------- endpoint metadata


@dataclass(frozen=True)
class Endpoint:
    """One provider endpoint of a model, from GET /models/{model}/endpoints.

    `tag` is the slug to pin in `provider.order`; `provider_name` is what responses report as `provider`.
    Prices are USD per token (or per request for `pricing["request"]`).
    """

    tag: str
    provider_name: str
    context_length: int | None
    max_completion_tokens: int | None
    quantization: str | None
    pricing: dict[str, float]
    supported_parameters: tuple[str, ...]
    status: int | None = None

    def supports(self, param: str) -> bool:
        return param in self.supported_parameters

    @property
    def price_known(self) -> bool:
        """Prompt and completion prices are listed and non-negative (OpenRouter uses -1 for variable)."""
        return all(self.pricing.get(k, -1.0) >= 0 for k in ("prompt", "completion"))

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Endpoint:
        pricing = {}
        for k, v in (d.get("pricing") or {}).items():
            try:
                pricing[k] = float(v)
            except (TypeError, ValueError):
                pass
        return cls(
            tag=d.get("tag") or d.get("provider_name", ""),
            provider_name=d.get("provider_name", ""),
            context_length=d.get("context_length"),
            max_completion_tokens=d.get("max_completion_tokens"),
            quantization=d.get("quantization"),
            pricing=pricing,
            supported_parameters=tuple(d.get("supported_parameters") or ()),
            status=d.get("status"),
        )


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_endpoint(endpoints: Sequence[Endpoint], tag: str) -> Endpoint | None:
    return next((e for e in endpoints if e.tag.lower() == tag.lower()), None)


def same_provider(pinned_tag: str, served_name: str | None, endpoints: Sequence[Endpoint]) -> bool:
    """Whether the provider name in a response matches the pinned endpoint tag."""
    if not served_name:
        return False
    ep = find_endpoint(endpoints, pinned_tag)
    names = {_norm(ep.provider_name)} if ep else set()
    names.add(_norm(pinned_tag.split("/")[0]))
    return _norm(served_name) in names


async def fetch_endpoints(http: httpx.AsyncClient, model: str, base_url: str = BASE_URL) -> list[Endpoint]:
    """Fetch a model's provider endpoints (no auth needed). Retries transient failures a few times."""
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = await http.get(f"{base_url}/models/{model}/endpoints")
            if r.status_code == 200:
                return [Endpoint.from_api(e) for e in r.json()["data"]["endpoints"]]
            if r.status_code not in RETRYABLE_STATUS:
                raise PermanentError(r.status_code, r.text[:500])
            last = OpenRouterError(f"endpoints {r.status_code}")
        except httpx.TransportError as e:
            last = e
        await asyncio.sleep(0.5 * 2**attempt)
    raise OpenRouterError(f"could not fetch endpoints for {model}: {last}")


async def get_endpoints(model: str) -> list[Endpoint]:
    """Standalone helper: the provider endpoints of `model`, cheapest prompt price first."""
    async with httpx.AsyncClient(timeout=30) as http:
        eps = await fetch_endpoints(http, model)
    return sorted(eps, key=lambda e: (e.pricing.get("prompt", 0), e.pricing.get("completion", 0)))


# ---------------------------------------------------------------- budget


class Budget:
    """A USD cap shared by all calls of a run, persisted to `path` so a resumed run keeps counting.

    Each attempt reserves an upper bound on its cost before it is sent and settles to the actual cost after.
    `reserve` and `settle` never await, so each runs atomically within the event loop (the cap check
    `spent + reserved + amount > cap` cannot interleave with another call's), and cancellation handlers
    can settle without awaiting. Outstanding reservations are persisted under `in_flight`: if the process
    dies with calls in flight, the next Budget loaded from `path` charges them in full (`recovered_usd`),
    since those requests may have been billed. Once `spent` reaches the cap, every reservation is refused.
    This recovery is safe only because one process at a time uses a run's budget file: `oligopoly.run` holds
    an exclusive lock on the run directory (`lock_run_dir`) before loading it, so any `in_flight` entries
    found on load were left by a process that has exited.
    """

    def __init__(self, cap_usd: float, path: str | Path | None = None):
        self.cap = float(cap_usd)
        self.path = Path(path) if path else None
        self.spent = 0.0
        self.recovered_usd = 0.0
        self.in_flight: dict[str, float] = {}
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.recovered_usd = sum(map(float, (d.get("in_flight") or {}).values()))
            self.spent = float(d["spent_usd"]) + self.recovered_usd
            if self.recovered_usd:
                self._persist()

    @property
    def reserved(self) -> float:
        return sum(self.in_flight.values())

    @property
    def remaining(self) -> float:
        return self.cap - self.spent - self.reserved

    def reserve(self, amount: float) -> str:
        """Reserve `amount` USD; returns the reservation id to settle."""
        if self.spent >= self.cap:
            raise BudgetExceeded(f"cap ${self.cap:.4f} reached (spent ${self.spent:.4f})")
        if self.spent + self.reserved + amount > self.cap:
            raise BudgetExceeded(
                f"reservation ${amount:.6f} exceeds remaining ${self.remaining:.6f} of cap ${self.cap:.4f}"
            )
        rid = uuid.uuid4().hex
        self.in_flight[rid] = amount
        self._persist()
        return rid

    def settle(self, rid: str, actual: float) -> None:
        """Release reservation `rid` and charge `actual` USD."""
        del self.in_flight[rid]
        self.spent += max(0.0, actual)
        self._persist()

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"cap_usd": self.cap, "spent_usd": self.spent, "in_flight": self.in_flight,
                                   "updated": datetime.now(timezone.utc).isoformat()}))
        tmp.replace(self.path)


# ---------------------------------------------------------------- results


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_api(cls, u: dict[str, Any] | None) -> Usage:
        u = u or {}
        pd = u.get("prompt_tokens_details") or {}
        cd = u.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=u.get("prompt_tokens") or 0,
            completion_tokens=u.get("completion_tokens") or 0,
            reasoning_tokens=cd.get("reasoning_tokens") or 0,
            cached_tokens=pd.get("cached_tokens") or 0,
            cache_write_tokens=pd.get("cache_write_tokens") or 0,
            total_tokens=u.get("total_tokens") or 0,
        )


@dataclass
class ChatResult:
    content: str | None
    reasoning: str | None
    reasoning_details: list[Any] | None
    finish_reason: str | None
    native_finish_reason: str | None
    provider: str | None  # display name of the provider that served the call
    model: str | None  # model id returned by OpenRouter
    generation_id: str | None
    usage: Usage
    cost_usd: float | None
    latency_s: float
    attempts: int
    seed_possibly_ignored: bool
    provider_mismatch: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------- helpers


def with_cache_control(messages: Sequence[dict[str, Any]], upto: int | Sequence[int],
                       ttl: str | None = None) -> list[dict[str, Any]]:
    """Copy of `messages` with an Anthropic-style `cache_control` breakpoint on the last content block of
    each message index in `upto` (negative indices allowed). Caching covers everything up to the breakpoint.
    Anthropic allows at most 4 breakpoints per request.
    """
    out = copy.deepcopy(list(messages))
    marker: dict[str, Any] = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    for i in [upto] if isinstance(upto, int) else upto:
        msg = out[i]
        content = msg["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        blocks[-1] = {**blocks[-1], "cache_control": marker}
        msg["content"] = blocks
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


PROMPT_SLACK_TOKENS = 256  # chat-template and provider-added tokens not in the serialised messages

# Context guard. Calibrated on the 2026-10-20 pilots (data/runs/pilot1*/calls.jsonl, ~800 calls): whole prompts
# ran at 3.2-3.8 UTF-8 content bytes per prompt token by model, and the marginal ratio of the numeric
# results-and-prices text that dominates long NoTalk histories at 2.7-3.3 (lowest: qwen3.8-27b 2.69, granite
# 2.82, qwen3-14b 2.85), template tokens included. 2.5 bytes/token is below every model's lowest observed
# marginal ratio; a model whose tokenizer runs closer to it sets a lower `bytes_per_token_min` in models.toml.
DEFAULT_BYTES_PER_TOKEN_MIN = 2.5
TEMPLATE_TOKENS_PER_MESSAGE = 8  # role markers and separators the chat template adds per message


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
    return ""


def prompt_token_bound(messages: Sequence[dict[str, Any]],
                       bytes_per_token_min: float = DEFAULT_BYTES_PER_TOKEN_MIN) -> int:
    """Conservative upper estimate of a prompt's tokens for the context guard: UTF-8 bytes of the message
    contents / bytes_per_token_min, plus TEMPLATE_TOKENS_PER_MESSAGE per message and PROMPT_SLACK_TOKENS.
    (estimate_cost uses the looser one-token-per-byte bound, which is safe for money but would refuse
    prompts that fit.)"""
    nbytes = sum(len(_content_text(m.get("content")).encode()) for m in messages)
    return math.ceil(nbytes / bytes_per_token_min) + TEMPLATE_TOKENS_PER_MESSAGE * len(messages) + PROMPT_SLACK_TOKENS


def estimate_cost(endpoint: Endpoint, messages: Sequence[dict[str, Any]], completion_tokens: int) -> float:
    """Upper bound on a call's cost, reserved before it is sent.

    Prompt tokens are bounded by the UTF-8 byte length of the JSON-serialised messages: byte-level BPE (and
    SentencePiece with byte fallback) never needs more than one token per byte, and the JSON keys and quotes
    per message exceed the few special tokens a chat template adds; PROMPT_SLACK_TOKENS covers the rest.
    Every prompt token is priced at the higher of the input and cache-write prices (no cache discount) and
    every allowed completion token at the output (plus internal-reasoning) price. Raises PermanentError if
    the endpoint's pricing is unknown, so a call is never sent without a bound.
    """
    p = endpoint.pricing
    if not endpoint.price_known:
        raise PermanentError(None, f"pricing of endpoint {endpoint.tag!r} unknown: {p}")
    prompt_tokens = len(json.dumps(list(messages), ensure_ascii=False).encode()) + PROMPT_SLACK_TOKENS
    per_prompt = max(p["prompt"], p.get("input_cache_write", 0.0))
    per_completion = p["completion"] + p.get("internal_reasoning", 0.0)
    return p.get("request", 0.0) + prompt_tokens * per_prompt + completion_tokens * per_completion


RATE_LIMIT_WORDS = re.compile(r"rate.?limit|overloaded|too many requests|capacity", re.I)


def _rate_limited(status: int | None, err: Any) -> bool:
    """HTTP 429 / 529, or an error body (any status) whose code is 429/529 or whose message says rate limited
    or overloaded (e.g. DeepInfra's upstream "engine_overloaded")."""
    if status in (429, 529):
        return True
    if isinstance(err, dict):
        if err.get("code") in (429, 529):
            return True
        err = json.dumps(err, default=str)
    return bool(err) and bool(RATE_LIMIT_WORDS.search(str(err))) and (status is None or status in RETRYABLE_STATUS
                                                                       or status == 200)


def _retry_after_s(value: str | None) -> float | None:
    """Seconds from a Retry-After header (delta-seconds or HTTP date); None if absent or unparseable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _error_retryable(err: Any) -> bool:
    code = err.get("code") if isinstance(err, dict) else None
    return not isinstance(code, int) or code in RETRYABLE_STATUS


# ---------------------------------------------------------------- client

# extra_body keys that are routing or output options, not sampling settings (kept out of the log's `sampling`)
NON_SAMPLING_KEYS = frozenset({"provider", "usage", "models", "route", "transforms", "plugins", "user", "stream"})

NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)  # the request never left


class OpenRouterClient:
    """Async OpenRouter client. Every attempt (failed or not, including budget refusals and cancellations)
    is appended as one JSON line to `log_path`. Use as an async context manager or call `aclose()`.

    Billing of failed attempts: an attempt that may have reached the provider (timeouts, dropped
    connections, cancellation, unexpected errors) is charged its full reservation; one known not to have
    been sent is charged nothing. A response without `usage.cost` is charged nothing if it is an error
    (non-200 status, or a 200 whose body is an error object with no choices: OpenRouter does not bill these)
    and its full reservation otherwise; the log's `charge_reason` says which rule applied. A retry after a timeout may be billed twice (the timed-out generation can
    still complete upstream), so it is logged with `timeout_retry: true`.

    Retry policies: rate limits (HTTP 429/529, or an error saying rate limited / overloaded) are retried
    patiently and do not count against `max_retries`: exponential backoff from `rate_limit_backoff_base_s`
    with jitter, capped at `rate_limit_backoff_max_s` per wait (a longer Retry-After is honoured), until the
    waits for this call total `rate_limit_max_wait_s` or the call has made `max_attempts` attempts in all
    (short waits could otherwise loop for a long time); then RetriesExhausted. Each wait is logged as an
    `event: rate_limit_wait` line. Other transient errors get `max_retries` retries with the normal backoff.
    """

    def __init__(
        self,
        log_path: str | Path,
        budget: Budget,
        *,
        api_key: str | None = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        timeout_s: float = 900.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        rate_limit_backoff_base_s: float = 5.0,
        rate_limit_backoff_max_s: float = 120.0,
        rate_limit_max_wait_s: float = 1800.0,
        max_attempts: int = 60,
        base_url: str = BASE_URL,
        http: httpx.AsyncClient | None = None,
    ):
        if api_key is None:
            load_dotenv(find_dotenv(usecwd=True))
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {api_key}", "X-Title": "SPAR-oligopoly"}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.backoff_base_s, self.backoff_max_s = backoff_base_s, backoff_max_s
        self.rate_limit_backoff_base_s = rate_limit_backoff_base_s
        self.rate_limit_backoff_max_s = rate_limit_backoff_max_s
        self.rate_limit_max_wait_s = rate_limit_max_wait_s
        self.max_attempts = max_attempts
        self.base_url = base_url
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._endpoints: dict[str, list[Endpoint]] = {}
        self._ep_lock = asyncio.Lock()
        if budget.recovered_usd:
            self._log({"ts": _now(), "event": "budget_recovered_in_flight", "charged_usd": budget.recovered_usd,
                       "budget_spent_usd": budget.spent,
                       "note": "reservations left in flight by an earlier process, charged in full"})

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_endpoints(self, model: str) -> list[Endpoint]:
        """Provider endpoints of `model` (cached per client)."""
        async with self._ep_lock:
            if model not in self._endpoints:
                self._endpoints[model] = await fetch_endpoints(self._http, model, self.base_url)
            return self._endpoints[model]

    async def resolve(self, model: str, provider: str) -> Endpoint:
        """The pinned endpoint of `model`, with known pricing; PermanentError otherwise."""
        eps = await self.get_endpoints(model)
        ep = find_endpoint(eps, provider)
        if ep is None:
            raise PermanentError(None, f"provider tag {provider!r} not among endpoints of {model}: "
                                       f"{[e.tag for e in eps]}")
        if not ep.price_known:
            raise PermanentError(None, f"pricing of {model} at {provider!r} unknown: {ep.pricing}")
        return ep

    def _log(self, record: dict[str, Any]) -> None:
        """Append one line. Synchronous (atomic within the event loop), so it also works while cancelled."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        delay = random.uniform(0, min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1)))
        try:
            delay = max(delay, float(retry_after)) if retry_after else delay
        except ValueError:
            pass
        return min(delay, self.backoff_max_s)

    def _rate_limit_backoff(self, k: int, retry_after: str | None) -> float:
        """Wait before the k-th rate-limit retry: exponential with jitter (50-100% of the step), capped at
        rate_limit_backoff_max_s; at least the Retry-After, if given (which may exceed the cap)."""
        step = min(self.rate_limit_backoff_max_s, self.rate_limit_backoff_base_s * 2 ** (k - 1))
        delay = step * random.uniform(0.5, 1.0)
        ra = _retry_after_s(retry_after)
        return max(delay, ra) if ra is not None else delay

    async def seed_flag(self, model: str, pinned: str, served: str | None, seed: int | None) -> tuple[bool, str | None]:
        """(seed_possibly_ignored, reason). False when no seed was requested."""
        if seed is None:
            return False, None
        if not served:
            return True, "served provider unknown"
        try:
            eps = await self.get_endpoints(model)
        except OpenRouterError as e:
            return True, f"endpoint metadata unavailable: {e}"
        if not same_provider(pinned, served, eps):
            return True, "served provider differs from pinned"
        ep = find_endpoint(eps, pinned)
        if ep is None:
            return True, "pinned endpoint not in metadata"
        if not ep.supports("seed"):
            return True, "served endpoint does not list seed in supported_parameters"
        return False, None

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        provider: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        require_parameters: bool = True,
        cache_upto: int | Sequence[int] | None = None,
        tag: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        bytes_per_token_min: float = DEFAULT_BYTES_PER_TOKEN_MIN,
    ) -> ChatResult:
        """One chat completion pinned to endpoint tag `provider` with fallbacks off.

        `max_tokens` is required: the budget reserves against it. `reasoning` is OpenRouter's object, e.g.
        {"effort": "medium"}, {"max_tokens": 2000} or {"enabled": False}. `cache_upto` adds cache_control
        breakpoints (see `with_cache_control`). `tag` is caller metadata (market, firm, period, phase, ...)
        copied into every log line. `timeout_s` overrides the client's HTTP timeout for this call.
        Context guard: if `prompt_token_bound(messages, bytes_per_token_min)` plus the completion allowance
        exceeds the pinned endpoint's context_length, the attempt is logged and ContextOverflow raised without
        sending anything.
        Raises PermanentError, RetriesExhausted, BudgetExceeded or ContextOverflow.
        """
        if max_tokens is None:
            raise ValueError("max_tokens is required: the budget reserves its cost before each call")
        msgs = with_cache_control(messages, cache_upto) if cache_upto is not None else list(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": require_parameters},
            "usage": {"include": True},
            "max_tokens": max_tokens,
        }
        for k, v in (("temperature", temperature), ("top_p", top_p), ("seed", seed),
                     ("reasoning", reasoning), ("response_format", response_format)):
            if v is not None:
                payload[k] = v
        payload.update(extra_body or {})
        timeout = self.timeout_s if timeout_s is None else timeout_s
        base = {
            "tag": tag or {},
            "model_requested": model,
            "provider_requested": provider,
            # everything sent that shapes sampling: the named arguments plus extra_body (top_k, presence_penalty...)
            "sampling": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                         **{k: v for k, v in (extra_body or {}).items() if k not in NON_SAMPLING_KEYS}},
            "seed": seed,
            "reasoning_setting": reasoning,
            "response_format": response_format,
            "require_parameters": require_parameters,
            "cache_upto": cache_upto,
            "timeout_s": timeout,
            "messages_sha256": _sha256(msgs),
            "request": payload,
        }
        try:
            ep = await self.resolve(model, provider)
        except PermanentError as e:
            self._log({"ts": _now(), "attempt": 0, **base, "error": f"PermanentError: {e}", "retryable": False})
            raise
        completion = max_tokens + int((reasoning or {}).get("max_tokens") or 0)  # cost bound: upper side
        bound = prompt_token_bound(msgs, bytes_per_token_min)
        # the context holds prompt + max_tokens; a reasoning budget is part of max_tokens, not extra
        completion_ctx = max_tokens
        base["context_guard"] = {"prompt_tokens_bound": bound, "bytes_per_token_min": bytes_per_token_min,
                                 "max_tokens": completion_ctx, "context_length": ep.context_length}
        if ep.context_length and bound + completion_ctx > ep.context_length:
            err = (f"prompt bound {bound} + max_tokens {completion_ctx} exceeds the context length "
                   f"{ep.context_length} of {model} at {provider!r}")
            self._log({"ts": _now(), "attempt": 0, **base, "error": f"ContextOverflow: {err}", "retryable": False})
            raise ContextOverflow(err)
        estimate = estimate_cost(ep, msgs, completion)

        last_error: str | None = None
        timeout_retry = False
        attempt = failures = rl_retries = 0
        rl_waited = 0.0

        async def retry_wait(rate_limited: bool, retry_after: str | None = None) -> None:
            """Sleep before the next attempt, or raise RetriesExhausted when the relevant policy is spent."""
            nonlocal failures, rl_retries, rl_waited
            if not rate_limited:
                failures += 1
                if failures > self.max_retries:
                    raise RetriesExhausted(f"{failures} non-rate-limit failures in {attempt} attempts; "
                                           f"last error: {last_error}")
                await asyncio.sleep(self._backoff(failures, retry_after))
                return
            if attempt >= self.max_attempts:
                raise RetriesExhausted(f"{attempt} attempts (limit {self.max_attempts}), rate limited for "
                                       f"{rl_waited:.0f} s in total; last error: {last_error}")
            remaining = self.rate_limit_max_wait_s - rl_waited
            if remaining <= 0:
                raise RetriesExhausted(f"rate limited for {rl_waited:.0f} s in total (limit "
                                       f"{self.rate_limit_max_wait_s:.0f} s) over {attempt} attempts; "
                                       f"last error: {last_error}")
            rl_retries += 1
            wait = min(self._rate_limit_backoff(rl_retries, retry_after), remaining)
            rl_waited += wait
            self._log({"ts": _now(), "event": "rate_limit_wait", "tag": base["tag"], "model_requested": model,
                       "provider_requested": provider, "attempt": attempt, "rate_limit_retry": rl_retries,
                       "wait_s": round(wait, 3), "waited_total_s": round(rl_waited, 3),
                       "max_wait_s": self.rate_limit_max_wait_s, "retry_after": retry_after, "error": last_error})
            await asyncio.sleep(wait)

        while True:
            attempt += 1
            rec: dict[str, Any] = {"ts": _now(), "attempt": attempt, **base, "timeout_retry": timeout_retry}
            timeout_retry = False
            resp: httpx.Response | None = None
            async with self._sem:
                try:
                    rid = self.budget.reserve(estimate)
                except BudgetExceeded as e:
                    self._log({**rec, "error": f"BudgetExceeded: {e}", "retryable": False,
                               "budget_spent_usd": self.budget.spent})
                    raise
                rec["reserved_usd"] = estimate
                t0 = time.monotonic()
                try:
                    resp = await self._http.post(f"{self.base_url}/chat/completions", json=payload,
                                                 headers=self._headers, timeout=timeout)
                except BaseException as e:  # incl. CancelledError: settle and log (no awaits) before anything else
                    charged = 0.0 if isinstance(e, NOT_SENT) else estimate
                    charge_reason = "not_sent" if isinstance(e, NOT_SENT) else "sent_no_response"
                    self.budget.settle(rid, charged)
                    transport = isinstance(e, httpx.TransportError)
                    last_error = "cancelled" if isinstance(e, asyncio.CancelledError) else f"{type(e).__name__}: {e}"
                    self._log({**rec, "latency_s": time.monotonic() - t0, "error": last_error,
                               "retryable": transport, "charged_usd": charged, "charge_reason": charge_reason,
                               "budget_spent_usd": self.budget.spent})
                    if not transport:
                        raise
                    timeout_retry = isinstance(e, httpx.TimeoutException) and not isinstance(e, NOT_SENT)
                else:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = None
                    usage_raw = body.get("usage") if isinstance(body, dict) else None
                    cost = usage_raw.get("cost") if isinstance(usage_raw, dict) else None
                    # unknown cost: a 200 carrying a generation may have been billed, so it keeps its
                    # reservation; error responses (non-200, or a 200 whose body is only an error object)
                    # are not billed by OpenRouter
                    error_body = (isinstance(body, dict) and bool(body.get("error"))
                                  and not isinstance(body.get("choices"), list))
                    if cost is not None:
                        charged, charge_reason = float(cost), "usage_cost"
                    elif resp.status_code != 200:
                        charged, charge_reason = 0.0, "error_status"
                    elif error_body:
                        charged, charge_reason = 0.0, "error_body"
                    else:
                        charged, charge_reason = estimate, "cost_unknown_reservation"
                    self.budget.settle(rid, charged)
            if resp is None:
                await retry_wait(False)
                continue

            latency = time.monotonic() - t0
            rec.update(latency_s=latency, status_code=resp.status_code, charged_usd=charged,
                       charge_reason=charge_reason, budget_spent_usd=self.budget.spent)
            if resp.status_code != 200 or not isinstance(body, dict) or not isinstance(body.get("choices"), list):
                if isinstance(body, dict) and body.get("error"):
                    err = body["error"]
                elif resp.status_code == 200:  # malformed body; no code, so retryable
                    err = {"message": f"malformed response body: {resp.text[:2000]}"}
                else:
                    err = {"code": resp.status_code, "message": resp.text[:2000]}
                retryable = (resp.status_code in RETRYABLE_STATUS if resp.status_code != 200
                             else _error_retryable(err))
                last_error = f"HTTP {resp.status_code}: {json.dumps(err, default=str)[:2000]}"
                rate_limited = retryable and _rate_limited(resp.status_code, err)
                self._log({**rec, "error": last_error, "retryable": retryable, "rate_limited": rate_limited})
                if not retryable:
                    raise PermanentError(resp.status_code, err)
                await retry_wait(rate_limited, resp.headers.get("retry-after"))
                continue

            served, model_served = body.get("provider"), body.get("model")
            choices = body["choices"]
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") or {}
            flag, reason = await self.seed_flag(model, provider, served, seed)
            mismatch = not same_provider(provider, served, await self.get_endpoints(model))
            rec.update(
                model_served=model_served, provider_served=served, provider_mismatch=mismatch,
                seed_possibly_ignored=flag, seed_flag_reason=reason, generation_id=body.get("id"),
                content=message.get("content"), reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), usage_raw=usage_raw,
                usage=asdict(Usage.from_api(usage_raw)), cost_usd=cost,
            )
            if not choice or choice.get("error"):
                last_error = f"empty or errored choice: {choice.get('error')}"
                retryable = _error_retryable(choice.get("error")) if choice.get("error") else True
                rate_limited = retryable and bool(choice.get("error")) and _rate_limited(None, choice.get("error"))
                self._log({**rec, "error": last_error, "retryable": retryable, "rate_limited": rate_limited})
                if not retryable:
                    raise PermanentError(resp.status_code, choice.get("error"))
                await retry_wait(rate_limited)
                continue
            self._log({**rec, "error": None})
            return ChatResult(
                content=message.get("content"), reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), provider=served, model=model_served,
                generation_id=body.get("id"), usage=Usage.from_api(usage_raw),
                cost_usd=float(cost) if cost is not None else None, latency_s=latency, attempts=attempt,
                seed_possibly_ignored=flag, provider_mismatch=mismatch, raw=body,
            )
