"""Runs one repeated Bertrand market with LLM firms (summary §1–3, DECISIONS.md).

Per period: in Talk parts up to `chat_rounds` simultaneous chat rounds (every firm is called concurrently
on the same state; a round's messages are revealed only after all firms have answered; the chat closes
early when everyone passes), then all firms price concurrently, then payoffs. The results of period t
open period t + 1's user message; the final period's results go with the end-of-part text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from . import market, prompts
from .config import ORDERINGS, ExperimentConfig, ModelConfig
from .firm import Firm
from .llm import BudgetExceeded, ChatResult, ContextOverflow
from .parse import Parsed, parse_message, parse_price, strip_reasoning

Kind = Literal["price", "message"]


class ChatClient(Protocol):
    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult: ...


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from any JSON-serialisable parts."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def environment_seed(base_seed: int, n: int, ordering: str, market_index: int) -> int:
    """Seed of a market's environment draws (die faces, listing orders). It leaves out the model and run id,
    so every model faces the same draws in the same (n, ordering, market) cell."""
    return derive_seed("env", base_seed, n, ordering, market_index)


@dataclass(frozen=True)
class MarketSpec:
    run_id: str
    model: ModelConfig
    n: int
    ordering: str
    market_index: int
    seed: int  # environment seed (see environment_seed)

    def __post_init__(self) -> None:
        if self.ordering not in ORDERINGS:
            raise ValueError(f"unknown ordering {self.ordering!r}; use one of {list(ORDERINGS)}")

    @property
    def key(self) -> str:
        slug = "".join(c if c.isalnum() or c in ".-_" else "-" for c in self.model.name)
        return f"{slug}__n{self.n}__{self.ordering}__m{self.market_index}"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "model": self.model.name, "n": self.n, "ordering": self.ordering,
                "market_index": self.market_index, "seed": self.seed}


def fingerprint(obj: Any) -> str:
    """SHA-256 of the canonical JSON of `obj`."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def texts_digest(exp: ExperimentConfig) -> str:
    """Digest of the firm-facing wording under `exp` (see `prompts.texts_digest`)."""
    return prompts.texts_digest(prompts.ChatSpec(max_rounds=exp.chat_rounds, char_limit=exp.char_limit), exp.notes)


def _fingerprint_from(spec: dict[str, Any], model_identity: dict[str, Any], experiment: dict[str, Any],
                      texts: str) -> str:
    return fingerprint({**spec, "model": model_identity, "experiment": experiment, "texts": texts})


def market_fingerprint(spec: MarketSpec, exp: ExperimentConfig) -> str:
    """Everything that determines a market's play: its spec, the model's settings and the experiment."""
    return _fingerprint_from(spec.to_dict(), spec.model.identity(), exp.to_dict(), texts_digest(exp))


def record_texts_digest_matches(rec: dict[str, Any], digest: str) -> bool:
    """Whether a market record was played under the firm-facing wording with digest `digest`. Records carry
    `settings.texts_digest`; older ones are checked by recomputing their fingerprint with `digest`."""
    st = rec.get("settings") or {}
    if "texts_digest" in st:
        return st["texts_digest"] == digest
    try:
        identity = {k: v for k, v in st["model"].items() if k not in ModelConfig.OPERATIONAL}
        return rec.get("fingerprint") == _fingerprint_from(rec["spec"], identity, st["experiment"], digest)
    except (KeyError, TypeError, AttributeError):
        return False


def is_complete(out_dir: Path, spec: MarketSpec, exp: ExperimentConfig) -> bool | None:
    """True if the market's record exists with the expected fingerprint, False if it is missing, None if it
    exists but was played under other settings (never complete; never overwritten)."""
    path = market_path(out_dir, spec)
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("fingerprint") == market_fingerprint(spec, exp) or None


def market_path(out_dir: Path, spec: MarketSpec) -> Path:
    return Path(out_dir) / "markets" / f"{spec.key}.json"


def failure_path(out_dir: Path, spec: MarketSpec) -> Path:
    return Path(out_dir) / "markets" / f"{spec.key}.failed.json"


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write to a temporary file in the same directory, fsync, then rename over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class Decision:
    kind: Kind
    value: int | str | None = None  # price, message text, or None (pass)
    notes: str | None = None
    fallback: bool = False
    errors: list[str] = field(default_factory=list)  # one per invalid attempt
    warnings: list[str] = field(default_factory=list)  # coercions, truncations, missing notes
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasks(self) -> int:
        return len(self.attempts) - 1

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "notes": self.notes, "valid": not self.fallback, "fallback": self.fallback,
                "reasks": self.reasks, "errors": self.errors, "warnings": self.warnings,
                "attempts": self.attempts}


def pick_error(eg: BaseExceptionGroup) -> BaseException:
    """The first BudgetExceeded among the (possibly nested) members of `eg`, else the first ContextOverflow,
    else its first leaf."""
    def leaves(e: BaseException) -> list[BaseException]:
        return [x for sub in e.exceptions for x in leaves(sub)] if isinstance(e, BaseExceptionGroup) else [e]
    errs = leaves(eg)
    return next((e for kind in (BudgetExceeded, ContextOverflow) for e in errs if isinstance(e, kind)), errs[0])


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens")


class Market:
    def __init__(self, spec: MarketSpec, client: ChatClient, exp: ExperimentConfig):
        self.spec, self.client, self.exp = spec, client, exp
        self.letters = prompts.firm_letters(spec.n)
        self.talk = ORDERINGS[spec.ordering]
        self.chat_spec = prompts.ChatSpec(max_rounds=exp.chat_rounds, char_limit=exp.char_limit)
        self.firms = {L: Firm(L, random.Random(derive_seed("order", spec.seed, L)).sample(self.letters, spec.n))
                      for L in self.letters}
        self.dies = [market.die_schedule(exp.part_lengths[p - 1], exp.part_min_periods(p),
                                         rng=derive_seed("die", spec.seed, p)) for p in (1, 2)]
        self.prev_price = {L: market.P_MAX for L in self.letters}
        self.profit = {L: [0.0, 0.0] for L in self.letters}
        self.periods: list[dict[str, Any]] = []
        self.usage: dict[str, Any] = {k: 0 for k in USAGE_KEYS} | {
            "calls": 0, "cost_usd": 0.0, "cost_unknown_calls": 0, "seed_possibly_ignored_calls": 0,
            "provider_mismatch_calls": 0, "providers_served": []}

    # ------------------------------------------------------------------ calls

    def call_seed(self, firm: str, part: int, period: int, phase: str, attempt: int) -> int:
        s = self.spec
        return derive_seed(s.run_id, s.model.model, s.n, s.ordering, s.market_index, s.seed,
                           firm, part, period, phase, attempt)

    def _account(self, r: ChatResult) -> None:
        u = self.usage
        u["calls"] += 1
        for k in USAGE_KEYS:
            u[k] += getattr(r.usage, k)
        if r.cost_usd is None:
            u["cost_unknown_calls"] += 1
        else:
            u["cost_usd"] += r.cost_usd
        u["seed_possibly_ignored_calls"] += bool(r.seed_possibly_ignored)
        u["provider_mismatch_calls"] += bool(r.provider_mismatch)
        if r.provider and r.provider not in u["providers_served"]:
            u["providers_served"].append(r.provider)

    async def decide(self, L: str, kind: Kind, prompt: str, part: int, period: int, phase: str) -> Decision:
        """Ask firm L, re-asking up to max_reasks times; failed attempts stay in its history."""
        firm, cfg, s = self.firms[L], self.spec.model, self.spec
        if kind == "price":
            fmt = prompts.price_format(self.exp.notes)
            parse = parse_price
        else:
            fmt = prompts.message_format(self.chat_spec, self.exp.notes)
            def parse(c: str | None) -> Parsed:
                return parse_message(c, self.exp.char_limit)
        d = Decision(kind)
        messages = firm.ask(prompt)
        for attempt in range(self.exp.max_reasks + 1):
            call_id = f"{s.key}/{L}/p{part}/t{period}/{phase}/a{attempt}"
            seed = self.call_seed(L, part, period, phase, attempt)
            tag = {"run_id": s.run_id, "market": s.key, "firm": L, "part": part, "period": period,
                   "phase": phase, "attempt": attempt, "call_id": call_id}
            r = await self.client.chat(messages, seed=seed, tag=tag, cache_upto=-1 if cfg.cache_control else None,
                                       **cfg.chat_kwargs())
            self._account(r)
            p = parse(r.content)
            d.attempts.append({
                "call_id": call_id, "seed": seed, "generation_id": r.generation_id, "provider": r.provider,
                "finish_reason": r.finish_reason, "seed_possibly_ignored": r.seed_possibly_ignored,
                "provider_mismatch": r.provider_mismatch, "error": p.error, "warnings": p.warnings,
                "usage": asdict(r.usage), "cost_usd": r.cost_usd,
                **({"content": r.content} if p.error else {})})
            if p.ok:  # the firm's history keeps the value actually used, and its notes as returned
                notes = {"notes": p.obj["notes"]} if "notes" in p.obj else {}
                firm.answer(json.dumps({**notes, kind: p.value}, ensure_ascii=False))
                d.value, d.notes, d.warnings = p.value, p.notes, p.warnings
                return d
            d.errors.append(p.error)
            bad = strip_reasoning(r.content or "")
            firm.answer(bad[:2000] if bad else "(empty response)")
            if attempt < self.exp.max_reasks:
                messages = firm.ask(prompts.reask_text(p.error, fmt))
        d.fallback = True
        d.value = self.prev_price[L] if kind == "price" else None
        firm.tell(prompts.fallback_text(kind, d.value if kind == "price" else None))
        return d

    async def gather(self, coros: dict[str, Any]) -> dict[str, Decision]:
        """Run the firms' decisions concurrently. A failure does not cancel the other firms' calls: they run to
        the end (their requests may already be billed, and their results are kept for a resume), then one error
        is re-raised as is (BudgetExceeded in preference, so the runner stops scheduling markets). If this
        coroutine itself is cancelled, the decisions are cancelled with it."""
        tasks = {L: asyncio.ensure_future(c) for L, c in coros.items()}
        try:
            await asyncio.wait(tasks.values())
        except BaseException:
            for t in tasks.values():
                t.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        errors = [e for t in tasks.values() if not t.cancelled() and (e := t.exception()) is not None]
        if errors:
            raise pick_error(BaseExceptionGroup("firm decisions failed", errors)) from None
        return {L: t.result() for L, t in tasks.items()}

    # ------------------------------------------------------------------ periods

    async def chat(self, part: int, period: int, die_roll: int | None, min_p: int):
        rounds: list[dict[str, Any]] = []
        shown: dict[str, list[prompts.ChatMessage]] = {L: [] for L in self.letters}
        last_round = 0
        for r in range(1, self.exp.chat_rounds + 1):
            ds = await self.gather({L: self.decide(
                L, "message", prompts.chat_prompt(part, period, r, L, shown[L], self.chat_spec,
                                                  die_roll if r == 1 else None, self.exp.notes, min_p),
                part, period, f"chat{r}") for L in self.letters})
            posted = {L: d.value for L, d in ds.items() if d.value is not None}
            # each firm sees the round's messages in its own listing order
            shown = {L: [prompts.ChatMessage(F, posted[F], r) for F in self.firms[L].order if F in posted]
                     for L in self.letters}
            rounds.append({"round": r, "messages": posted, "decisions": {L: d.to_dict() for L, d in ds.items()}})
            last_round = r
            if not posted:
                break
        return rounds, last_round, shown

    async def period(self, part: int, period: int) -> None:
        talk, exp = self.talk[part - 1], self.exp
        min_p = exp.part_min_periods(part)
        die_roll = self.dies[part - 1][period - 2] if period > 1 else None
        rounds, last_round, shown = [], None, {L: [] for L in self.letters}
        if talk:
            rounds, last_round, shown = await self.chat(part, period, die_roll, min_p)
        ds = await self.gather({L: self.decide(
            L, "price", prompts.period_prompt(part, period, None if talk else die_roll, talk, exp.notes,
                                              firm_letter=L, last_round=last_round, last_messages=shown[L],
                                              min_periods=min_p),
            part, period, "price") for L in self.letters})
        prices = {L: d.value for L, d in ds.items()}
        profits = market.payoffs(prices)
        for L in self.letters:
            self.prev_price[L] = prices[L]
            self.profit[L][part - 1] += profits[L]
        die = self.dies[part - 1][period - 1]
        self.periods.append({
            "part": part, "period": period, "talk": talk, "die": die, "chat": rounds, "prices": prices,
            "notes": {L: d.notes for L, d in ds.items()}, "profits": profits,
            "decisions": {L: d.to_dict() for L, d in ds.items()}})
        last = period == exp.part_lengths[part - 1]
        for L, firm in self.firms.items():
            firm.tell(prompts.results_text(
                part, period, L, prices, profits, firm.order, cumulative_part=self.profit[L][part - 1],
                cumulative_total=sum(self.profit[L]) if part == 2 else None))
            if last:
                firm.tell(prompts.part_over_text(part, period, final_part=part == 2,
                                                 total_profit=sum(self.profit[L]) if part == 2 else None,
                                                 min_periods=min_p))
                if part == 1:
                    firm.tell(prompts.part_instructions(self.spec.n, L, 2, self.talk[1], previous_talk=self.talk[0],
                                                        chat_spec=self.chat_spec, notes=self.exp.notes))

    async def run(self) -> None:
        for L, firm in self.firms.items():
            firm.start(prompts.part_instructions(self.spec.n, L, 1, self.talk[0],
                                                 chat_spec=self.chat_spec if self.talk[0] else None,
                                                 notes=self.exp.notes))
        for part in (1, 2):
            for t in range(1, self.exp.part_lengths[part - 1] + 1):
                await self.period(part, t)
        for firm in self.firms.values():
            firm.flush()

    # ------------------------------------------------------------------ output

    def record(self) -> dict[str, Any]:
        totals = {}
        for L in self.letters:
            decs = [p["decisions"][L] for p in self.periods] + [
                r["decisions"][L] for p in self.periods for r in p["chat"]]
            totals[L] = {"profit_part1": self.profit[L][0], "profit_part2": self.profit[L][1],
                         "profit_total": sum(self.profit[L]),
                         "price_fallbacks": sum(p["decisions"][L]["fallback"] for p in self.periods),
                         "chat_fallbacks": sum(r["decisions"][L]["fallback"] for p in self.periods for r in p["chat"]),
                         "reasks": sum(d["reasks"] for d in decs),
                         "invalid_attempts": sum(len(d["errors"]) for d in decs)}
        return {
            "spec": self.spec.to_dict(),
            "fingerprint": market_fingerprint(self.spec, self.exp),
            "settings": {"model": self.spec.model.to_dict(), "experiment": self.exp.to_dict(),
                         "texts_digest": texts_digest(self.exp),
                         "pilot": self.exp.pilot, "talk": dict(zip(("part1", "part2"), self.talk)),
                         "chat_spec": asdict(self.chat_spec), "feedback_all_profits": True, "exchange_rate": None,
                         "die_schedules": self.dies, "listing_orders": {L: f.order for L, f in self.firms.items()}},
            "periods": self.periods,
            "totals": totals,
            "usage": self.usage,
            "histories": {L: f.messages for L, f in self.firms.items()},
        }


def failure_reason(e: BaseException) -> str:
    """Short reason recorded with a failed market: context_overflow, budget or error."""
    if isinstance(e, ContextOverflow):
        return "context_overflow"
    if isinstance(e, BudgetExceeded):
        return "budget"
    return "error"


async def run_market(spec: MarketSpec, client: ChatClient, out_dir: str | Path,
                     exp: ExperimentConfig | None = None) -> Path:
    """Play one market and write its JSON record atomically; returns the path. On failure, writes a
    `.failed.json` record (with the periods completed so far) and re-raises."""
    exp = exp or ExperimentConfig()
    out, fail = market_path(Path(out_dir), spec), failure_path(Path(out_dir), spec)
    m: Market | None = None
    try:
        m = Market(spec, client, exp)
        await m.run()
    except BaseException as e:
        write_json_atomic(fail, {"spec": spec.to_dict(), "reason": failure_reason(e),
                                 **({"guard": e.guard} if isinstance(e, ContextOverflow) else {}),
                                 "error_type": type(e).__name__, "error": str(e),
                                 "traceback": traceback.format_exc(), "periods_completed": len(m.periods) if m else 0,
                                 **({k: v for k, v in m.record().items() if k != "spec"} if m else {})})
        raise
    write_json_atomic(out, m.record())
    fail.unlink(missing_ok=True)
    return out
