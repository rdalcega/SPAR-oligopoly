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
from collections.abc import Callable, Sequence
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


class BudgetUnusable(BudgetExceeded):
    """The budget file could not be written, so no further spending can be accounted for (fail closed)."""


class BudgetWaitTimeout(OpenRouterError):
    """Waited the admission backstop for budget room that never came (a reservation was not settled in time).
    Not a BudgetExceeded: the money may still be there, so only the current market fails (and checkpoints);
    the runner keeps scheduling."""


class ContextOverflow(OpenRouterError):
    """The prompt's token bound plus max_tokens exceeds the pinned endpoint's context length (not sent).
    `guard` holds the guard settings that refused it: bytes_per_token_min, max_tokens, context_length."""

    def __init__(self, message: str, guard: dict[str, Any] | None = None):
        super().__init__(message)
        self.guard = guard


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


NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)  # the request never left


async def fetch_endpoints(http: httpx.AsyncClient, model: str, base_url: str = BASE_URL, connect_wait_s: float = 0,
                          on_wait: Callable[[float, BaseException], None] | None = None) -> list[Endpoint]:
    """Fetch a model's provider endpoints (no auth needed). Retries transient failures a few times; with
    `connect_wait_s` > 0, connection errors (NOT_SENT: e.g. the network is down) are instead retried with
    backoff (jittered, at most 60 s per wait) until the waits total `connect_wait_s`. `on_wait(wait_s, error)`
    is called before each such wait."""
    last: BaseException | None = None
    attempt = net_retries = 0
    net_waited = 0.0
    while True:
        try:
            r = await http.get(f"{base_url}/models/{model}/endpoints")
            if r.status_code == 200:
                return [Endpoint.from_api(e) for e in r.json()["data"]["endpoints"]]
            if r.status_code not in RETRYABLE_STATUS:
                raise PermanentError(r.status_code, r.text[:500])
            last = OpenRouterError(f"endpoints {r.status_code}")
        except httpx.TransportError as e:
            last = e
            if isinstance(e, NOT_SENT) and net_waited < connect_wait_s:
                net_retries += 1
                step = min(60.0, 2.0 * 2 ** (net_retries - 1))
                wait = min(step * random.uniform(0.5, 1.0), connect_wait_s - net_waited)
                net_waited += wait
                if on_wait:
                    on_wait(wait, e)
                await asyncio.sleep(wait)
                continue
        attempt += 1
        if attempt >= 4:
            raise OpenRouterError(f"could not fetch endpoints for {model}: {last}")
        await asyncio.sleep(0.5 * 2 ** (attempt - 1))


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

    Admission (`admit`, used by the client): a reservation that would fit the cap given what is spent but not
    alongside the outstanding reservations waits until settlements make room, instead of failing a call that
    the remaining money covers. It raises BudgetExceeded on true exhaustion (`spent + amount > cap`) or when
    nothing is in flight to wait for, and BudgetWaitTimeout after waiting `max_wait_s` in all (a backstop
    against a reservation that is never settled; it must exceed the longest per-attempt deadline, since an in-flight
    reservation is settled only when its request returns or times out). The cap stays strict: spent + reserved
    never exceeds it.

    Fail closed: if the budget file cannot be written, the budget is marked unusable (`broken`), every waiting
    `admit` is woken, and every later reservation raises BudgetUnusable (a BudgetExceeded, so runners stop
    scheduling). `settle` itself never raises: it is called from cancellation and error handlers.
    This recovery is safe only because one process at a time uses a run's budget file: `oligopoly.run` holds
    an exclusive lock on the run directory (`lock_run_dir`) before loading it, so any `in_flight` entries
    found on load were left by a process that has exited.
    """

    DEFAULT_MAX_WAIT_S = 3600.0  # 2 x the client's default per-attempt deadline (2 x the 900 s per-read timeout)

    def __init__(self, cap_usd: float, path: str | Path | None = None):
        self.cap = float(cap_usd)
        self.path = Path(path) if path else None
        self.spent = 0.0
        self.recovered_usd = 0.0
        self.in_flight: dict[str, float] = {}
        self.broken: str | None = None  # why the budget can no longer be used (persistence failed)
        self._settled: asyncio.Event | None = None  # set (and replaced) on every settle, waking waiting admits
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

    def _check_usable(self) -> None:
        if self.broken:
            raise BudgetUnusable(f"budget unusable: {self.broken}")

    def reserve(self, amount: float) -> str:
        """Reserve `amount` USD; returns the reservation id to settle."""
        self._check_usable()
        if self.spent >= self.cap:
            raise BudgetExceeded(f"cap ${self.cap:.4f} reached (spent ${self.spent:.4f})")
        if self.spent + self.reserved + amount > self.cap:
            raise BudgetExceeded(
                f"reservation ${amount:.6f} exceeds remaining ${self.remaining:.6f} of cap ${self.cap:.4f}"
            )
        rid = uuid.uuid4().hex
        self.in_flight[rid] = amount
        try:
            self._persist()
        except Exception as e:  # the reservation was not recorded on disk: drop it and fail closed
            del self.in_flight[rid]
            self._break(e)
            raise BudgetUnusable(f"budget unusable: {self.broken}") from e
        return rid

    async def admit(self, amount: float, max_wait_s: float | None = None,
                    on_wait: Callable[[], None] | None = None) -> str:
        """Reserve `amount` USD, waiting while it fits what is left after `spent` but not alongside the other
        outstanding reservations. `on_wait` is called once, when the first wait starts. Raises
        BudgetExceeded when `spent + amount > cap` (true exhaustion) or when it does not fit and nothing is in
        flight; BudgetWaitTimeout when the waits total `max_wait_s` (default DEFAULT_MAX_WAIT_S);
        BudgetUnusable once the budget is broken."""
        limit = self.DEFAULT_MAX_WAIT_S if max_wait_s is None else max_wait_s
        deadline: float | None = None
        while True:
            self._check_usable()
            if self.spent >= self.cap or self.spent + amount > self.cap:
                raise BudgetExceeded(f"cap ${self.cap:.4f} exhausted: spent ${self.spent:.4f}, reservation "
                                     f"${amount:.6f}")
            if self.spent + self.reserved + amount <= self.cap:
                return self.reserve(amount)
            if not self.in_flight:  # unreachable while reserved == 0, kept as a guard against waiting forever
                raise BudgetExceeded(f"reservation ${amount:.6f} exceeds remaining ${self.remaining:.6f}")
            now = time.monotonic()
            if deadline is None:
                deadline = now + limit
                if on_wait is not None:
                    on_wait()
            if now >= deadline:
                raise BudgetWaitTimeout(f"waited {limit:.0f} s for budget room (reservation ${amount:.6f}, "
                                     f"spent ${self.spent:.4f}, reserved ${self.reserved:.4f})")
            if self._settled is None:
                self._settled = asyncio.Event()
            try:
                await asyncio.wait_for(self._settled.wait(), deadline - now)
            except TimeoutError:
                pass  # re-checked above: raises unless a settlement made room meanwhile

    def settle(self, rid: str, actual: float) -> None:
        """Release reservation `rid` and charge `actual` USD (an unknown `rid` is ignored). Never awaits and
        never raises; always wakes any waiting `admit`. If the budget file cannot be written the budget is
        marked unusable."""
        try:
            if self.in_flight.pop(rid, None) is not None:
                self.spent += max(0.0, actual)
                self._persist()
        except Exception as e:
            self._break(e)
        finally:
            self._wake()

    def _break(self, e: BaseException) -> None:
        self.broken = self.broken or f"could not persist the budget to {self.path}: {type(e).__name__}: {e}"
        self._wake()

    def _wake(self) -> None:
        if self._settled is not None:
            self._settled.set()
            self._settled = None

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


def _strip_cache_control(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Undo `with_cache_control` on plain-text messages: a content list of one text block becomes its text."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b for b in c]
            if len(c) == 1 and isinstance(c[0], dict) and set(c[0]) == {"type", "text"} and c[0]["type"] == "text":
                c = c[0]["text"]
        out.append({**m, "content": c})
    return out


def _firm_texts(market_json: dict[str, Any], firm: str) -> tuple[list[str], list[dict[str, Any]]]:
    """(instructions, transcript) of `firm` in a market record or failure record (`instructions`,
    `transcripts`) or in a checkpoint (`firms[firm]`)."""
    if "transcripts" in market_json:
        return market_json["instructions"][firm], market_json["transcripts"][firm]
    f = (market_json.get("firms") or {})[firm]
    return f["instructions"], f["transcript"]


def reconstruct_request(market_json: dict[str, Any], firm: str, context: dict[str, Any],
                        cache_upto: int | Sequence[int] | None = None) -> list[dict[str, Any]]:
    """The messages of a request of `firm`, rebuilt from its market record, failure record or checkpoint and the
    attempt's `context` (from the record's decisions). The summaries layout (`layout` "sections") is rebuilt by
    `firm.section_messages`; older contexts hold the number of instruction messages, the memory text and the
    transcript window [lo, hi), and `merged` if same-role messages were merged (absent in older records). Pass
    the log line's `cache_upto` to get the messages as sent (check
    `messages_sha256(result)` against the line's `messages_sha256`), or none to compare with the context's own
    `messages_sha256` (of the messages without cache_control). A call made in a period that a process died in
    and a later invocation replayed differently is not in the final record; `reconstruct_from_log` rebuilds any
    logged request from the log itself."""
    from .firm import LAYOUT, context_messages, section_messages

    instructions, transcript = _firm_texts(market_json, firm)
    if context.get("layout") == LAYOUT:
        try:
            msgs = section_messages(instructions, transcript, context)
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"firm {firm}: context does not fit the record ({type(e).__name__}: {e})") from e
        return with_cache_control(msgs, cache_upto) if cache_upto is not None else msgs
    k, (lo, hi) = context["instructions"], context["window"]
    if not 1 <= k <= len(instructions) or not 0 <= lo < hi <= len(transcript):
        raise ValueError(f"firm {firm}: context {k} instructions, window [{lo}, {hi}) does not fit the record")
    msgs = context_messages(instructions[:k], context["memory"], transcript[lo:hi],
                            merge=bool(context.get("merged")))
    return with_cache_control(msgs, cache_upto) if cache_upto is not None else msgs


def _firm_key(line: dict[str, Any]) -> tuple[Any, ...] | None:
    tag = line.get("tag") or {}
    return (line.get("invocation"), tag.get("run_id"), tag["market"], tag["firm"]) \
        if "market" in tag and "firm" in tag else None


def _compact(line: dict[str, Any]) -> bool:
    return "messages_ref" in line or "messages_new" in line


def reconstruct_from_log(lines: Sequence[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """The messages sent (cache_control included) by the request logged at `lines[index]`, where `lines` are
    a calls.jsonl's records in file order. A line with `request` carries them in full. A compact line refers to
    the latest preceding request line of the same firm, market, run and `invocation` (process), which is how the
    client logged it: `messages_ref` lists, per message, either the index of an identical message in that
    previous request or the message itself; older logs have `messages_from`/`messages_new` (the messages
    appended to it). The chain is followed back to a full line. Raises ValueError if the chain is broken. Check
    the result with `messages_sha256`."""
    line = lines[index]
    if "request" in line:
        return list(line["request"]["messages"])
    if not _compact(line):
        raise ValueError(f"line {index} carries no request")
    key = _firm_key(line)
    if key is None or key[0] is None:
        raise ValueError(f"line {index}: compact line without firm tag or invocation id")
    chain = [line]
    for j in range(index - 1, -1, -1):
        prev = lines[j]
        if "event" in prev or _firm_key(prev) != key:
            continue
        if "request" in prev:
            msgs = _strip_cache_control(prev["request"]["messages"])
            break
        if _compact(prev):
            chain.append(prev)
    else:
        raise ValueError(f"line {index}: no earlier full request of {key}")
    for c in reversed(chain):
        if "messages_ref" in c:
            new = []
            for ref in c["messages_ref"]:
                if isinstance(ref, int) and not isinstance(ref, bool):
                    if not 0 <= ref < len(msgs):
                        raise ValueError(f"line {index}: chain broken (reference {ref} to {len(msgs)} messages)")
                    new.append(dict(msgs[ref]))
                else:
                    new.append(dict(ref))
            msgs = new
            continue
        if c["messages_from"] != len(msgs):
            raise ValueError(f"line {index}: chain broken (messages_from {c['messages_from']} after {len(msgs)} "
                             f"messages)")
        msgs = msgs + [dict(m) for m in c["messages_new"]]
    cache_upto = line.get("cache_upto")
    return with_cache_control(msgs, cache_upto) if cache_upto is not None else msgs


def messages_sha256(messages: Sequence[dict[str, Any]]) -> str:
    """The `messages_sha256` logged for a request with these (cache_control-marked) messages."""
    return _sha256(list(messages))


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


def _cost_value(raw: Any) -> float | None:
    """`usage.cost` as a finite non-negative float; None if absent or not a usable number."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def _error_retryable(err: Any) -> bool:
    code = err.get("code") if isinstance(err, dict) else None
    return not isinstance(code, int) or code in RETRYABLE_STATUS


# ---------------------------------------------------------------- client

# extra_body keys that are routing or output options, not sampling settings (kept out of the log's `sampling`)
NON_SAMPLING_KEYS = frozenset({"provider", "usage", "models", "route", "transforms", "plugins", "user", "stream"})


class OpenRouterClient:
    """Async OpenRouter client. Every attempt (failed or not, including budget refusals and cancellations)
    is appended as one JSON line to `log_path`. Use as an async context manager or call `aclose()`.

    Log size: each line has `invocation` (one id per client, i.e. per runner process), `messages_sha256` (of the
    messages sent, cache_control included), `n_messages` and `prompt_bytes` (UTF-8 bytes of the message
    contents). The full `request` payload is logged for the first request of each (run_id, market, firm) in a
    client's life (so in each process invocation) and for failed attempts other than rate-limited ones. Other
    lines carry `messages_ref` (per message sent, without cache_control: the index of an identical message in
    the firm's previous logged request in this invocation, or the message itself) and `request_params` (the
    payload minus messages); `reconstruct_from_log` chains them from the latest preceding full line of the same
    firm and invocation. Firms' contexts are windowed (instructions, memory, recent turns), so consecutive
    requests share most messages but not a prefix. Calls without market and firm in `tag` always log the full
    request.

    Billing of failed attempts: an attempt that may have reached the provider (timeouts, dropped
    connections, cancellation, unexpected errors) is charged its full reservation; one known not to have
    been sent is charged nothing. A response without `usage.cost` is charged nothing if it is an error
    (non-200 status, or a 200 whose body is an error object with no choices: OpenRouter does not bill these)
    and its full reservation otherwise, as is one whose `usage.cost` is not a usable number
    (`cost_raw` logs it); any other unexpected failure after the reservation charges it in full too, so no
    reservation is ever left unsettled. The log's `charge_reason` says which rule applied. A retry after a timeout may be billed twice (the timed-out generation can
    still complete upstream), so it is logged with `timeout_retry: true`.

    Retry policies: rate limits (HTTP 429/529, or an error saying rate limited / overloaded) are retried
    patiently and do not count against `max_retries`: exponential backoff from `rate_limit_backoff_base_s`
    with jitter, capped at `rate_limit_backoff_max_s` per wait (a longer Retry-After is honoured), until the
    waits for this call total `rate_limit_max_wait_s` or the call has made `max_attempts` attempts in all
    (short waits could otherwise loop for a long time); then RetriesExhausted. Each wait is logged as an
    `event: rate_limit_wait` line. Connection errors where the request provably never left (NOT_SENT:
    connect errors incl. DNS failures, connect and pool timeouts) cost nothing and get the same patient policy,
    with their own wait total (also capped at `rate_limit_max_wait_s`) and `event: network_wait` lines.
    Other transient errors get `max_retries` retries with the normal backoff.
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
        deadline_s: float | None = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        rate_limit_backoff_base_s: float = 5.0,
        rate_limit_backoff_max_s: float = 120.0,
        rate_limit_max_wait_s: float = 1800.0,
        max_attempts: int = 60,
        budget_max_wait_s: float | None = None,
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
        self.deadline_s = deadline_s
        self.backoff_base_s, self.backoff_max_s = backoff_base_s, backoff_max_s
        self.rate_limit_backoff_base_s = rate_limit_backoff_base_s
        self.rate_limit_backoff_max_s = rate_limit_backoff_max_s
        self.rate_limit_max_wait_s = rate_limit_max_wait_s
        self.max_attempts = max_attempts
        # admission backstop: None = the larger of Budget.DEFAULT_MAX_WAIT_S and 2 x the longest per-attempt
        # deadline any call of this client has used (an in-flight reservation is held for up to one deadline)
        self.budget_max_wait_s = budget_max_wait_s
        self._max_deadline_s = self._deadline(timeout_s, deadline_s)
        self.base_url = base_url
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._endpoints: dict[str, list[Endpoint]] = {}
        self._ep_lock = asyncio.Lock()
        # one id per client (= per process invocation of the runner), in every log line
        self.invocation_id = uuid.uuid4().hex
        self._ids = {"invocation": self.invocation_id}
        # (run_id, market, firm) -> sha256 of each message (without cache_control) of its last logged request
        self._logged: dict[tuple[Any, ...], list[str]] = {}
        if budget.recovered_usd:
            self._log({"ts": _now(), "event": "budget_recovered_in_flight", **self._ids, "charged_usd": budget.recovered_usd,
                       "budget_spent_usd": budget.spent,
                       "note": "reservations left in flight by an earlier process, charged in full"})

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def admission_wait_s(self) -> float:
        """How long a call may wait for budget room before BudgetWaitTimeout (see `budget_max_wait_s`)."""
        if self.budget_max_wait_s is not None:
            return self.budget_max_wait_s
        return max(Budget.DEFAULT_MAX_WAIT_S, 2 * self._max_deadline_s)

    async def get_endpoints(self, model: str, connect_wait_s: float = 0,
                            on_wait: Callable[[float, BaseException], None] | None = None) -> list[Endpoint]:
        """Provider endpoints of `model` (cached per client); see `fetch_endpoints` for `connect_wait_s`."""
        async with self._ep_lock:
            if model not in self._endpoints:
                self._endpoints[model] = await fetch_endpoints(self._http, model, self.base_url, connect_wait_s,
                                                               on_wait)
            return self._endpoints[model]

    async def resolve(self, model: str, provider: str, connect_wait_s: float = 0,
                      on_wait: Callable[[float, BaseException], None] | None = None) -> Endpoint:
        """The pinned endpoint of `model`, with known pricing; PermanentError otherwise."""
        eps = await self.get_endpoints(model, connect_wait_s, on_wait)
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

    @staticmethod
    def _deadline(timeout: float, deadline: float | None) -> float:
        """The absolute per-attempt deadline: `deadline` if given, else 2 x the per-read `timeout`."""
        return 2 * timeout if deadline is None else deadline

    async def _post(self, payload: dict[str, Any], timeout: float, deadline: float) -> httpx.Response:
        """POST one attempt. httpx's own timeouts (`timeout` s for connect, pool wait, each write and each read)
        come first; their ConnectTimeout/PoolTimeout/ConnectError are NOT_SENT as before. Since httpx bounds
        each read, a trickling or keep-alive response could still hold the reservation indefinitely, so the
        whole attempt, body included, also runs under a larger absolute `deadline`.

        Hitting the deadline raises httpx.ReadTimeout (charged its reservation, retried with `timeout_retry`)
        if the request may have been sent: httpcore's trace showed request headers being written, or showed
        nothing at all (no trace, e.g. a pool wait or a mock transport: unknown, so conservative). It raises
        httpx.ConnectTimeout (NOT_SENT, charged nothing) only when the trace showed connection activity but no
        request write."""
        phase = "unknown"

        async def trace(name: str, info: dict[str, Any]) -> None:
            nonlocal phase
            if "send_request_headers" in name:
                phase = "sent"
            elif phase == "unknown" and name.startswith("connection."):
                phase = "connecting"

        try:
            async with asyncio.timeout(deadline):
                return await self._http.post(f"{self.base_url}/chat/completions", json=payload,
                                             headers=self._headers, timeout=timeout, extensions={"trace": trace})
        except TimeoutError as e:
            msg = f"absolute deadline of {deadline} s exceeded for the whole request ({phase})"
            if phase == "connecting":
                raise httpx.ConnectTimeout(msg) from e
            raise httpx.ReadTimeout(msg) from e

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
        deadline_s: float | None = None,
        bytes_per_token_min: float = DEFAULT_BYTES_PER_TOKEN_MIN,
    ) -> ChatResult:
        """One chat completion pinned to endpoint tag `provider` with fallbacks off.

        `max_tokens` is required: the budget reserves against it. `reasoning` is OpenRouter's object, e.g.
        {"effort": "medium"}, {"max_tokens": 2000} or {"enabled": False}. `cache_upto` adds cache_control
        breakpoints (see `with_cache_control`). `tag` is caller metadata (market, firm, period, phase, ...)
        copied into every log line. `timeout_s` overrides the client's per-read HTTP timeout (also connect,
        pool and write) for this call; `deadline_s` overrides the absolute per-attempt deadline (default: the
        client's `deadline_s`, else 2 x the per-read timeout).
        Context guard: if `prompt_token_bound(messages, bytes_per_token_min)` plus the completion allowance
        exceeds the pinned endpoint's context_length, the attempt is logged and ContextOverflow raised without
        sending anything.
        Raises PermanentError, RetriesExhausted, BudgetExceeded, BudgetWaitTimeout or ContextOverflow.
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
        deadline = self._deadline(timeout, self.deadline_s if deadline_s is None else deadline_s)
        self._max_deadline_s = max(self._max_deadline_s, deadline)
        base = {
            **self._ids,
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
            "deadline_s": deadline,
            "messages_sha256": _sha256(msgs),
            "n_messages": len(msgs),
            "prompt_bytes": sum(len(_content_text(m.get("content")).encode()) for m in msgs),
        }
        firm_key = ((tag or {}).get("run_id"), tag["market"], tag["firm"]) \
            if tag and "market" in tag and "firm" in tag else None
        request_params = {k: v for k, v in payload.items() if k != "messages"}

        raw = list(messages)
        raw_sha = [_sha256(m) for m in raw]

        def req(full: bool) -> dict[str, Any]:
            """The request part of a log line: the full payload if `full` or for a firm's first logged request in
            this client; otherwise `messages_ref`, each message (without cache_control; `cache_upto` says where
            breakpoints went) as the index of an identical message in the firm's previous logged request, or as
            itself when it is new."""
            prev = self._logged.get(firm_key) if firm_key else None
            if firm_key:
                self._logged[firm_key] = raw_sha
            if full or prev is None:
                return {"request": payload}
            where = {h: i for i, h in reversed(list(enumerate(prev)))}
            return {"messages_ref": [where.get(h, m) for h, m in zip(raw_sha, raw)], "request_params": request_params}
        try:
            ep = await self.resolve(model, provider)
        except PermanentError as e:
            self._log({"ts": _now(), "attempt": 0, **base, **req(True), "error": f"PermanentError: {e}",
                       "retryable": False})
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
            self._log({"ts": _now(), "attempt": 0, **base, **req(True), "error": f"ContextOverflow: {err}",
                       "retryable": False})
            raise ContextOverflow(err, guard={"bytes_per_token_min": bytes_per_token_min, "max_tokens": completion_ctx,
                                              "context_length": ep.context_length})
        estimate = estimate_cost(ep, msgs, completion)

        last_error: str | None = None
        timeout_retry = False
        attempt = failures = rl_retries = net_retries = 0
        rl_waited = net_waited = 0.0

        async def network_wait() -> None:
            """Patient wait after a NOT_SENT connection error (free: nothing was sent), or RetriesExhausted."""
            nonlocal net_retries, net_waited
            if attempt >= self.max_attempts:
                raise RetriesExhausted(f"{attempt} attempts (limit {self.max_attempts}), network unreachable for "
                                       f"{net_waited:.0f} s in total; last error: {last_error}")
            remaining = self.rate_limit_max_wait_s - net_waited
            if remaining <= 0:
                raise RetriesExhausted(f"network unreachable for {net_waited:.0f} s in total (limit "
                                       f"{self.rate_limit_max_wait_s:.0f} s) over {attempt} attempts; "
                                       f"last error: {last_error}")
            net_retries += 1
            wait = min(self._rate_limit_backoff(net_retries, None), remaining)
            net_waited += wait
            self._log({"ts": _now(), "event": "network_wait", **self._ids, "tag": base["tag"], "model_requested": model,
                       "provider_requested": provider, "attempt": attempt, "network_retry": net_retries,
                       "wait_s": round(wait, 3), "waited_total_s": round(net_waited, 3),
                       "max_wait_s": self.rate_limit_max_wait_s, "error": last_error})
            await asyncio.sleep(wait)

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
            self._log({"ts": _now(), "event": "rate_limit_wait", **self._ids, "tag": base["tag"], "model_requested": model,
                       "provider_requested": provider, "attempt": attempt, "rate_limit_retry": rl_retries,
                       "wait_s": round(wait, 3), "waited_total_s": round(rl_waited, 3),
                       "max_wait_s": self.rate_limit_max_wait_s, "retry_after": retry_after, "error": last_error})
            await asyncio.sleep(wait)

        while True:
            attempt += 1
            rec: dict[str, Any] = {"ts": _now(), "attempt": attempt, **base, "timeout_retry": timeout_retry}
            timeout_retry = not_sent = False
            resp: httpx.Response | None = None
            async with self._sem:
                def on_wait() -> None:
                    self._log({"ts": _now(), "event": "budget_wait", **self._ids, "tag": base["tag"],
                               "attempt": attempt, "reservation_usd": estimate, "budget_spent_usd": self.budget.spent,
                               "budget_reserved_usd": self.budget.reserved, "cap_usd": self.budget.cap})
                try:
                    rid = await self.budget.admit(estimate, self.admission_wait_s(), on_wait)
                except (BudgetExceeded, BudgetWaitTimeout) as e:
                    self._log({**rec, **req(True), "error": f"{type(e).__name__}: {e}", "retryable": False,
                               "budget_spent_usd": self.budget.spent})
                    raise
                except asyncio.CancelledError:  # cancelled while waiting for budget room: nothing reserved or sent
                    self._log({**rec, **req(True), "error": "cancelled while waiting for budget room",
                               "retryable": False, "charged_usd": 0.0, "charge_reason": "not_sent",
                               "budget_spent_usd": self.budget.spent})
                    raise
                rec["reserved_usd"] = estimate
                t0 = time.monotonic()
                try:
                    try:
                        resp = await self._post(payload, timeout, deadline)
                    except BaseException as e:  # incl. CancelledError: settle and log (no awaits) first
                        charged = 0.0 if isinstance(e, NOT_SENT) else estimate
                        charge_reason = "not_sent" if isinstance(e, NOT_SENT) else "sent_no_response"
                        self.budget.settle(rid, charged)
                        transport = isinstance(e, httpx.TransportError)
                        last_error = "cancelled" if isinstance(e, asyncio.CancelledError) else f"{type(e).__name__}: {e}"
                        self._log({**rec, **req(True), "latency_s": time.monotonic() - t0, "error": last_error,
                                   "retryable": transport, "charged_usd": charged, "charge_reason": charge_reason,
                                   "budget_spent_usd": self.budget.spent})
                        if not transport:
                            raise
                        not_sent = isinstance(e, NOT_SENT)
                        timeout_retry = isinstance(e, httpx.TimeoutException) and not isinstance(e, NOT_SENT)
                    else:
                        try:
                            body = resp.json()
                        except ValueError:
                            body = None
                        usage_raw = body.get("usage") if isinstance(body, dict) else None
                        cost_raw = usage_raw.get("cost") if isinstance(usage_raw, dict) else None
                        cost = _cost_value(cost_raw)
                        # unknown or invalid cost: a 200 carrying a generation may have been billed, so it keeps
                        # its reservation; error responses (non-200, or a 200 whose body is only an error
                        # object) are not billed by OpenRouter
                        error_body = (isinstance(body, dict) and bool(body.get("error"))
                                      and not isinstance(body.get("choices"), list))
                        if cost is not None:
                            charged, charge_reason = cost, "usage_cost"
                        elif cost_raw is not None:
                            charged, charge_reason = estimate, "cost_invalid_reservation"
                            rec["cost_raw"] = cost_raw
                        elif resp.status_code != 200:
                            charged, charge_reason = 0.0, "error_status"
                        elif error_body:
                            charged, charge_reason = 0.0, "error_body"
                        else:
                            charged, charge_reason = estimate, "cost_unknown_reservation"
                        self.budget.settle(rid, charged)
                finally:  # any unexpected failure above: the request may have been billed, charge it in full
                    if rid in self.budget.in_flight:
                        self.budget.settle(rid, estimate)
                        self._log({**rec, **req(True), "error": "unsettled reservation charged in full after an "
                                   "unexpected error", "charged_usd": estimate,
                                   "charge_reason": "unexpected_error", "budget_spent_usd": self.budget.spent})
            if resp is None:
                if not_sent:
                    await network_wait()
                else:
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
                self._log({**rec, **req(not rate_limited), "error": last_error, "retryable": retryable,
                           "rate_limited": rate_limited})
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
                self._log({**rec, **req(not rate_limited), "error": last_error, "retryable": retryable,
                           "rate_limited": rate_limited})
                if not retryable:
                    raise PermanentError(resp.status_code, choice.get("error"))
                await retry_wait(rate_limited)
                continue
            self._log({**rec, **req(False), "error": None})
            return ChatResult(
                content=message.get("content"), reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), provider=served, model=model_served,
                generation_id=body.get("id"), usage=Usage.from_api(usage_raw),
                cost_usd=cost, latency_s=latency, attempts=attempt,
                seed_possibly_ignored=flag, provider_mismatch=mismatch, raw=body,
            )
