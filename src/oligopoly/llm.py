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
import hashlib
import json
import os
import random
import re
import time
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


def _find_endpoint(endpoints: Sequence[Endpoint], tag: str) -> Endpoint | None:
    return next((e for e in endpoints if e.tag.lower() == tag.lower()), None)


def same_provider(pinned_tag: str, served_name: str | None, endpoints: Sequence[Endpoint]) -> bool:
    """Whether the provider name in a response matches the pinned endpoint tag."""
    if not served_name:
        return False
    ep = _find_endpoint(endpoints, pinned_tag)
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

    Each attempt reserves a conservative estimate before it is sent and settles to the actual cost after.
    Once `spent` reaches the cap, every further reservation raises BudgetExceeded.
    """

    def __init__(self, cap_usd: float, path: str | Path | None = None):
        self.cap = float(cap_usd)
        self.path = Path(path) if path else None
        self.spent = 0.0
        self.reserved = 0.0
        self._lock = asyncio.Lock()
        if self.path and self.path.exists():
            self.spent = float(json.loads(self.path.read_text())["spent_usd"])

    @property
    def remaining(self) -> float:
        return self.cap - self.spent - self.reserved

    async def reserve(self, amount: float) -> float:
        async with self._lock:
            if self.spent >= self.cap:
                raise BudgetExceeded(f"cap ${self.cap:.4f} reached (spent ${self.spent:.4f})")
            if self.spent + self.reserved + amount > self.cap:
                raise BudgetExceeded(
                    f"reservation ${amount:.6f} exceeds remaining ${self.remaining:.6f} of cap ${self.cap:.4f}"
                )
            self.reserved += amount
            return amount

    async def settle(self, reserved: float, actual: float) -> None:
        async with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += actual
            self._persist()

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"cap_usd": self.cap, "spent_usd": self.spent,
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


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def estimate_cost(endpoint: Endpoint, messages: Sequence[dict[str, Any]], completion_tokens: int) -> float:
    """Conservative upper bound on a call's cost: ~3 chars per prompt token, full (uncached, cache-write)
    prompt price, and every allowed completion token billed, plus a 20% margin."""
    p = endpoint.pricing
    prompt_tokens = len(json.dumps(messages, ensure_ascii=False)) / 3 + 50
    per_prompt = max(p.get("prompt", 0.0), p.get("input_cache_write", 0.0))
    per_completion = p.get("completion", 0.0) + p.get("internal_reasoning", 0.0)
    return 1.2 * (p.get("request", 0.0) + prompt_tokens * per_prompt + completion_tokens * per_completion)


def _error_retryable(err: Any) -> bool:
    code = err.get("code") if isinstance(err, dict) else None
    return not isinstance(code, int) or code in RETRYABLE_STATUS


# ---------------------------------------------------------------- client


class OpenRouterClient:
    """Async OpenRouter client. Every attempt (failed or not, including budget refusals) is appended as one
    JSON line to `log_path`. Use as an async context manager or call `aclose()`."""

    def __init__(
        self,
        log_path: str | Path,
        budget: Budget,
        *,
        api_key: str | None = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        timeout_s: float = 180.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        default_reserve_tokens: int = 8192,
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
        self.backoff_base_s, self.backoff_max_s = backoff_base_s, backoff_max_s
        self.default_reserve_tokens = default_reserve_tokens
        self.base_url = base_url
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._log_lock = asyncio.Lock()
        self._endpoints: dict[str, list[Endpoint]] = {}
        self._ep_lock = asyncio.Lock()

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

    async def _log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        delay = random.uniform(0, min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1)))
        try:
            delay = max(delay, float(retry_after)) if retry_after else delay
        except ValueError:
            pass
        return min(delay, self.backoff_max_s)

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
        ep = _find_endpoint(eps, pinned)
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
    ) -> ChatResult:
        """One chat completion pinned to endpoint tag `provider` with fallbacks off.

        `reasoning` is OpenRouter's object, e.g. {"effort": "medium"}, {"max_tokens": 2000} or
        {"enabled": False}. `cache_upto` adds cache_control breakpoints (see `with_cache_control`).
        `tag` is caller metadata (market, firm, period, phase, ...) copied into every log line.
        Raises PermanentError, RetriesExhausted or BudgetExceeded.
        """
        msgs = with_cache_control(messages, cache_upto) if cache_upto is not None else list(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": require_parameters},
            "usage": {"include": True},
        }
        for k, v in (("temperature", temperature), ("top_p", top_p), ("max_tokens", max_tokens), ("seed", seed),
                     ("reasoning", reasoning), ("response_format", response_format)):
            if v is not None:
                payload[k] = v
        payload.update(extra_body or {})

        eps = await self.get_endpoints(model)
        ep = _find_endpoint(eps, provider)
        if ep is None:
            raise PermanentError(None, f"provider tag {provider!r} not among endpoints of {model}: "
                                       f"{[e.tag for e in eps]}")
        cap_tokens = (max_tokens or self.default_reserve_tokens) + int((reasoning or {}).get("max_tokens") or 0)
        estimate = estimate_cost(ep, msgs, cap_tokens)

        base = {
            "tag": tag or {},
            "model_requested": model,
            "provider_requested": provider,
            "sampling": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
            "seed": seed,
            "reasoning_setting": reasoning,
            "response_format": response_format,
            "require_parameters": require_parameters,
            "cache_upto": cache_upto,
            "messages_sha256": _sha256(msgs),
            "request": payload,
        }
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 2):
            rec: dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(), "attempt": attempt, **base}
            try:
                reserved = await self.budget.reserve(estimate)
            except BudgetExceeded as e:
                await self._log({**rec, "error": f"BudgetExceeded: {e}", "retryable": False,
                                 "budget_spent_usd": self.budget.spent})
                raise
            t0 = time.monotonic()
            retry_after: str | None = None
            try:
                async with self._sem:
                    resp = await self._http.post(f"{self.base_url}/chat/completions", json=payload,
                                                 headers=self._headers)
            except httpx.TransportError as e:
                # A request that may have reached the provider is charged its full reservation.
                not_sent = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
                charged = 0.0 if not_sent else reserved
                await self.budget.settle(reserved, charged)
                last_error = f"{type(e).__name__}: {e}"
                await self._log({**rec, "latency_s": time.monotonic() - t0, "error": last_error,
                                 "retryable": True, "reserved_usd": reserved, "charged_usd": charged,
                                 "budget_spent_usd": self.budget.spent})
                await asyncio.sleep(self._backoff(attempt, None))
                continue
            latency = time.monotonic() - t0
            rec.update(latency_s=latency, status_code=resp.status_code, reserved_usd=reserved)
            try:
                body = resp.json()
            except ValueError:
                body = {"error": {"code": resp.status_code, "message": resp.text[:2000]}}

            if resp.status_code != 200 or "choices" not in body and body.get("error"):
                err = body.get("error", body)
                retryable = (resp.status_code in RETRYABLE_STATUS if resp.status_code != 200
                             else _error_retryable(err))
                await self.budget.settle(reserved, 0.0)
                last_error = f"HTTP {resp.status_code}: {json.dumps(err, default=str)[:2000]}"
                await self._log({**rec, "error": last_error, "retryable": retryable, "charged_usd": 0.0,
                                 "budget_spent_usd": self.budget.spent})
                if not retryable:
                    raise PermanentError(resp.status_code, err)
                await asyncio.sleep(self._backoff(attempt, resp.headers.get("retry-after")))
                continue

            usage_raw = body.get("usage")
            cost = (usage_raw or {}).get("cost")
            charged = float(cost) if cost is not None else reserved  # unknown cost: keep the reservation
            await self.budget.settle(reserved, charged)
            served, model_served = body.get("provider"), body.get("model")
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            flag, reason = await self.seed_flag(model, provider, served, seed)
            mismatch = not same_provider(provider, served, eps)
            rec.update(
                model_served=model_served, provider_served=served, provider_mismatch=mismatch,
                seed_possibly_ignored=flag, seed_flag_reason=reason, generation_id=body.get("id"),
                content=message.get("content"), reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), usage_raw=usage_raw,
                usage=asdict(Usage.from_api(usage_raw)), cost_usd=cost, charged_usd=charged,
                budget_spent_usd=self.budget.spent,
            )
            if not choices or choice.get("error"):
                last_error = f"empty or errored choice: {choice.get('error')}"
                retryable = _error_retryable(choice.get("error")) if choice.get("error") else True
                await self._log({**rec, "error": last_error, "retryable": retryable})
                if not retryable:
                    raise PermanentError(resp.status_code, choice.get("error"))
                await asyncio.sleep(self._backoff(attempt, None))
                continue
            await self._log({**rec, "error": None})
            return ChatResult(
                content=message.get("content"), reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), provider=served, model=model_served,
                generation_id=body.get("id"), usage=Usage.from_api(usage_raw),
                cost_usd=float(cost) if cost is not None else None, latency_s=latency, attempts=attempt,
                seed_possibly_ignored=flag, provider_mismatch=mismatch, raw=body,
            )
        raise RetriesExhausted(f"{self.max_retries + 1} attempts failed; last error: {last_error}")
