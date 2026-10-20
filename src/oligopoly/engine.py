"""Runs one repeated Bertrand market with LLM firms (summary §1–3, DECISIONS.md).

Per period: in Talk parts up to `chat_rounds` simultaneous chat rounds (every firm is called concurrently
on the same state; a round's messages are revealed only after all firms have answered; the chat closes
early when everyone passes), then all firms price concurrently, then payoffs. The results of period t
open period t + 1's user message; the final period's results go with the end-of-part text.

Post-game probe (`post_game_probe`, DECISIONS.md 2026-10-20): after the final period, the final results and
end-of-experiment text open one more user message per firm with comprehension questions (`probe_firm`), scored
into the record's `probe`. It follows all play, so no decision can depend on it; it is checkpointed like a
period, so a market whose probe failed reruns only the probe. The probe sees the same sections as any call
(no chat), so its question about the last part's chat tests the firm's notes.

Context (DECISIONS.md §6 item 3, summaries layout): every request is the firm's instructions and one user
message: the summaries of the last `memory_window` completed periods with its own cumulative profit, its latest
private notes, in Talk parts the current period's chat so far, then the prompt (see `firm.py`). Earlier answers
and earlier periods' chat are not shown again. The record keeps every firm's full transcript (`transcripts`:
summaries, chat rounds, prompts, answers, re-asks) and instructions, and each attempt's `context` (transcript
indices, cumulative-profit text, messages_sha256), so `llm.reconstruct_request` rebuilds every request. The
prompt-cache breakpoint goes on the system message (Part 1's instructions) only.

Checkpoints: `run_market` writes `markets/<key>.ckpt.json` atomically when the market starts and after every
completed period (part boundaries included): the firms' instructions, transcripts, notes and queued text, the
period records, cumulative profits, last prices, usage so far, the market fingerprint and the wording digest
(a resume therefore rebuilds identical windows and memory blocks). When a market
fails, the checkpoint is rewritten with the results of the calls already made in the period in progress
(`partial_calls`, keyed by call id and checked against the request's messages). A rerun with a matching
checkpoint resumes after its last completed period and reuses those results instead of calling again, so
paid calls are not repeated; a checkpoint that does not match (fingerprint, wording, die schedule or listing
orders) is ignored with a printed notice and the market restarts. The checkpoint is deleted once the market
record is written; the record lists in `resumed_from_period` the (1-based, over both parts) period each
resume started from.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from . import market, prompts
from .config import ORDERINGS, ExperimentConfig, ModelConfig
from .firm import Firm
from .llm import BudgetExceeded, BudgetWaitTimeout, ChatResult, ContextOverflow, Usage, messages_sha256
from .parse import PARSER_VERSION, Parsed, extract_object, parse_message, parse_price, parse_probe, strip_reasoning

log = logging.getLogger(__name__)

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


def response_schema(kind: Kind, notes: bool) -> dict[str, Any]:
    """OpenRouter `response_format` for a structured-outputs call (`structured_outputs` in models.toml): a strict
    JSON schema with `notes` (if the run uses notes) and `message` (string or null) or `price` (integer 0-100)."""
    props: dict[str, Any] = {"notes": {"type": "string"}} if notes else {}
    if kind == "price":
        props["price"] = {"type": "integer", "minimum": 0, "maximum": market.P_MAX}
    else:
        props["message"] = {"type": ["string", "null"]}
    return {"type": "json_schema", "json_schema": {
        "name": "price_decision" if kind == "price" else "chat_turn", "strict": True,
        "schema": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}


def texts_digest(exp: ExperimentConfig) -> str:
    """Digest of the firm-facing wording under `exp` (see `prompts.texts_digest`)."""
    return prompts.texts_digest(prompts.ChatSpec(max_rounds=exp.chat_rounds, char_limit=exp.char_limit), exp.notes,
                                exp.memory_window, exp.notes_words)


def _fingerprint_from(spec: dict[str, Any], model_identity: dict[str, Any], experiment: dict[str, Any],
                      texts: str, parser_version: int | None = None) -> str:
    """`parser_version` None (records from before it was stored) leaves it out of the hash."""
    parser = {"parser_version": parser_version} if parser_version is not None else {}
    return fingerprint({**spec, "model": model_identity, "experiment": experiment, "texts": texts, **parser})


def market_fingerprint(spec: MarketSpec, exp: ExperimentConfig, texts: str | None = None) -> str:
    """Everything that determines a market's play: its spec, the model's settings, the experiment, the
    firm-facing wording (`texts`: its digest under `exp`, computed if not given) and the parsing rules
    (`parse.PARSER_VERSION`)."""
    return _fingerprint_from(spec.to_dict(), spec.model.identity(), exp.to_dict(),
                             texts_digest(exp) if texts is None else texts, PARSER_VERSION)


def record_texts_digest_matches(rec: dict[str, Any], digest: str) -> bool:
    """Whether a market record was played under the firm-facing wording with digest `digest`. Records carry
    `settings.texts_digest`; older ones are checked by recomputing their fingerprint with `digest`."""
    st = rec.get("settings") or {}
    if "texts_digest" in st:
        return st["texts_digest"] == digest
    try:
        identity = {k: v for k, v in st["model"].items() if k not in ModelConfig.OPERATIONAL}
        return rec.get("fingerprint") == _fingerprint_from(rec["spec"], identity, st["experiment"], digest,
                                                           st.get("parser_version"))
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


def checkpoint_path(out_dir: Path, spec: MarketSpec) -> Path:
    return Path(out_dir) / "markets" / f"{spec.key}.ckpt.json"


def is_market_record_file(path: Path) -> bool:
    """Whether a file in markets/ is a completed market record (not a failure record or a checkpoint)."""
    return path.name.endswith(".json") and not path.name.endswith((".failed.json", ".ckpt.json"))


def write_text_atomic(path: Path, text: str) -> None:
    """Write to a uniquely named temporary file in the same directory, fsync, then rename over `path`
    (concurrent writers of one path never share a temporary file; the last rename wins)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


_CHECKPOINT_LOCK = threading.Lock()  # makes a checkpoint's sequence check and rename one step in this process
_SEQ_HEAD = re.compile(rb'^\{"seq": (\d+)[,}]')


def checkpoint_seq(path: Path) -> int | None:
    """The `seq` of the checkpoint at `path` (written first in the file, so only its head is read); None if the
    file is missing or has none (written before checkpoints carried one)."""
    try:
        with path.open("rb") as f:
            m = _SEQ_HEAD.match(f.read(64))
    except FileNotFoundError:
        return None
    return int(m.group(1)) if m else None


def with_seq(ck: dict[str, Any], seq: int) -> dict[str, Any]:
    """`ck` with `seq` as its first key (where `checkpoint_seq` reads it)."""
    return {"seq": seq, **{k: v for k, v in ck.items() if k != "seq"}}


def write_checkpoint_atomic(path: Path, text: str, seq: int) -> bool:
    """Like write_text_atomic, but rename over `path` only if the checkpoint there is not newer (its `seq` is
    not above `seq`). Guards against a writer left running in a worker thread after its await was cancelled
    landing after a later checkpoint. Returns whether `path` was replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        with _CHECKPOINT_LOCK:
            on_disk = checkpoint_seq(path)
            if on_disk is not None and on_disk > seq:
                return False
            os.replace(tmp, path)
            return True
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, obj: Any) -> None:
    """`obj` as indented JSON, written atomically (see write_text_atomic)."""
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1))


@dataclass
class Decision:
    kind: Kind
    value: int | str | None = None  # price, message text, or None (pass)
    notes: str | None = None
    fallback: bool = False
    notes_truncated: bool = False  # notes over the cap: `notes` is the capped text the firm carries
    notes_full: str | None = None  # the notes as written, when truncated
    errors: list[str] = field(default_factory=list)  # one per invalid attempt
    warnings: list[str] = field(default_factory=list)  # coercions, truncations, missing notes
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasks(self) -> int:
        return len(self.attempts) - 1

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "notes": self.notes, "valid": not self.fallback, "fallback": self.fallback,
                "reasks": self.reasks, "errors": self.errors, "warnings": self.warnings,
                **({"notes_truncated": True, "notes_full": self.notes_full} if self.notes_truncated else {}),
                "attempts": self.attempts}


def pick_error(eg: BaseExceptionGroup) -> BaseException:
    """The first BudgetExceeded among the (possibly nested) members of `eg`, else the first ContextOverflow,
    else its first leaf."""
    def leaves(e: BaseException) -> list[BaseException]:
        return [x for sub in e.exceptions for x in leaves(sub)] if isinstance(e, BaseExceptionGroup) else [e]
    errs = leaves(eg)
    return next((e for kind in (BudgetExceeded, ContextOverflow) for e in errs if isinstance(e, kind)), errs[0])


PROBE_MAX_REASKS = 1  # post-game probe: one re-ask at most, then recorded as invalid
TRUNCATED = " [truncated]"


def cap_notes(text: str, max_words: int, max_chars: int) -> tuple[str, bool]:
    """`text` cut to at most `max_words` words and `max_chars` characters (the marker included): at the end of
    the last whole word that fits, followed by TRUNCATED. Returns (text, whether it was cut). A first word too
    long to fit is cut inside the word."""
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words and len(text) <= max_chars:
        return text, False
    room = max_chars - len(TRUNCATED)
    end = 0
    for i, w in enumerate(words):
        if i >= max_words or w.end() > room:
            break
        end = w.end()
    return (text[:end] if end else text[:room]) + TRUNCATED, True


def best_common_price(letters: list[str]) -> int:
    """The common price that gives each firm the highest per-period profit when all firms set it."""
    return max(range(market.P_MAX + 1), key=lambda p: (market.payoffs(dict.fromkeys(letters, p))[letters[0]], -p))


def score_probe(answers: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    """Probe answers against the truth: booleans for the two prices, the number and share of firms whose last
    price was recalled exactly. Missing answers (None: also pass None for an invalid or skipped probe, whose
    partial answers must not be scored) count as wrong on every item."""
    a = answers or {}
    recalled = a.get("last_period_prices") or {}
    right = sum(recalled.get(L) == v for L, v in truth["last_period_prices"].items())
    return {"max_consumer_price": a.get("max_consumer_price") == truth["max_consumer_price"],
            "best_common_price": a.get("best_common_price") == truth["best_common_price"],
            "last_period_correct": right, "last_period_share": right / len(truth["last_period_prices"]),
            "last_period_all": right == len(truth["last_period_prices"])}


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens")
CHECKPOINT_VERSION = 5  # 2: `probe` (post-game probe result, None until done); 3: windowed context (firm
# instructions, transcript and notes instead of an append-only history); 4: same-role messages merged;
# 5: summaries layout (transcript entries have a `kind`; firms carry `notes_at`)
# ChatResult fields kept for a call made in a period that did not complete (enough for decide and _account)
RESULT_FIELDS = ("content", "finish_reason", "native_finish_reason", "provider", "model", "generation_id",
                 "cost_usd", "latency_s", "attempts", "seed_possibly_ignored", "provider_mismatch")


def result_to_dict(r: ChatResult) -> dict[str, Any]:
    return {**{k: getattr(r, k) for k in RESULT_FIELDS}, "usage": asdict(r.usage)}


def result_from_dict(d: dict[str, Any]) -> ChatResult:
    return ChatResult(**{k: d[k] for k in RESULT_FIELDS}, usage=Usage(**d["usage"]), reasoning=None,
                      reasoning_details=None)


class Market:
    def __init__(self, spec: MarketSpec, client: ChatClient, exp: ExperimentConfig, checkpoint: Path | None = None):
        self.spec, self.client, self.exp = spec, client, exp
        # computed once (they hash the prompt texts): used by every checkpoint, restore and record
        self.texts_digest = texts_digest(exp)
        self.fingerprint = market_fingerprint(spec, exp, self.texts_digest)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.resumed_from: list[int] = []  # global period index each resume started from
        self.partial_calls: dict[str, dict[str, Any]] = {}  # results of calls in the period in progress
        self.replay: dict[str, dict[str, Any]] = {}  # from a checkpoint: results to reuse instead of calling
        self._boundary: str | None = None  # JSON of the last checkpoint written (a period boundary)
        self._next_boundary: str | None = None  # a boundary serialised but not yet known to be written
        self._seq = 0  # sequence number of the last checkpoint serialised (each write gets a higher one)
        self.letters = prompts.firm_letters(spec.n)
        self.talk = ORDERINGS[spec.ordering]
        self.chat_spec = prompts.ChatSpec(max_rounds=exp.chat_rounds, char_limit=exp.char_limit)
        self.firms = {L: Firm(L, random.Random(derive_seed("order", spec.seed, L)).sample(self.letters, spec.n),
                              window=exp.memory_window)
                      for L in self.letters}
        self.dies = [market.die_schedule(exp.part_lengths[p - 1], exp.part_min_periods(p),
                                         rng=derive_seed("die", spec.seed, p)) for p in (1, 2)]
        self.prev_price = {L: market.P_MAX for L in self.letters}
        self.profit = {L: [0.0, 0.0] for L in self.letters}
        self.periods: list[dict[str, Any]] = []
        self.probe: dict[str, Any] | None = None  # post-game probe result, once asked
        self.usage: dict[str, Any] = {k: 0 for k in USAGE_KEYS} | {
            "calls": 0, "cost_usd": 0.0, "cost_unknown_calls": 0, "seed_possibly_ignored_calls": 0,
            "provider_mismatch_calls": 0, "providers_served": []}

    # ------------------------------------------------------------------ calls

    def global_period(self, part: int, period: int) -> int:
        """1-based period index over both parts (the post-game probe is the total + 1)."""
        return period if part == 1 else self.exp.part_lengths[0] + period

    def totals(self, L: str, part: int) -> str:
        """Firm L's own cumulative profit, shown after the summaries in a call in `part`."""
        return prompts.totals_text(part, self.profit[L][part - 1], sum(self.profit[L]) if part == 2 else None)

    def request(self, L: str, messages: list[dict[str, Any]], ctx: dict[str, Any]) -> tuple[
            list[dict[str, Any]], dict[str, Any], list[int] | None]:
        """Firm L's request: messages, context record and cache breakpoints (on the system message, Part 1's
        instructions: the only stable prefix). `messages_sha256` is of the messages without cache_control."""
        ctx["messages_sha256"] = messages_sha256(messages)
        cache = [0] if self.spec.model.cache_control else None
        return messages, ctx, cache

    def ask(self, L: str, prompt: str, g: int, part: int, chat: bool) -> tuple[
            list[dict[str, Any]], dict[str, Any], list[int] | None]:
        """Open firm L's decision on `prompt` in global period g (`chat`: show this period's chat section)."""
        return self.request(L, *self.firms[L].ask(prompt, g, self.totals(L, part), self.exp.notes, chat))

    def reask(self, L: str, text: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[int] | None]:
        return self.request(L, *self.firms[L].reask(text))

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
            fmt = prompts.price_format(self.exp.notes, self.exp.notes_words)
            parse = parse_price
        else:
            fmt = prompts.message_format(self.chat_spec, self.exp.notes, self.exp.notes_words)
            def parse(c: str | None) -> Parsed:
                return parse_message(c, self.exp.char_limit)
        d = Decision(kind)
        g = self.global_period(part, period)  # the sections are fixed within a decision
        messages, ctx, cache = self.ask(L, prompt, g, part, self.talk[part - 1])
        for attempt in range(self.exp.max_reasks + 1):
            call_id = f"{s.key}/{L}/p{part}/t{period}/{phase}/a{attempt}"
            seed = self.call_seed(L, part, period, phase, attempt)
            tag = {"run_id": s.run_id, "market": s.key, "firm": L, "part": part, "period": period,
                   "phase": phase, "attempt": attempt, "call_id": call_id}
            msha = ctx["messages_sha256"]  # one hash per request: the context's, also the replay key
            cached = self.replay.pop(call_id, None)
            if cached is not None and cached["messages_sha256"] == msha:  # paid for before a failure: reuse
                r = result_from_dict(cached["result"])
            else:
                r = await self.client.chat(messages, seed=seed, tag=tag, cache_upto=cache, **cfg.chat_kwargs(),
                                           **({"response_format": response_schema(kind, self.exp.notes)}
                                              if cfg.structured_outputs else {}))
            self.partial_calls[call_id] = {"messages_sha256": msha, "result": result_to_dict(r)}
            self._account(r)
            p = parse(r.content)
            reask = None
            if kind == "message" and p.obj is None and extract_object(r.content, "price") is not None:
                # a price object on a chat turn: say so plainly (the generic re-ask did not fix it in the pilots)
                p.error, reask = prompts.CHAT_PRICE_ERROR, prompts.chat_price_reask_text(
                    self.chat_spec, self.exp.notes, self.exp.notes_words)
            d.attempts.append({
                "call_id": call_id, "seed": seed, "generation_id": r.generation_id, "provider": r.provider,
                "finish_reason": r.finish_reason, "seed_possibly_ignored": r.seed_possibly_ignored,
                "provider_mismatch": r.provider_mismatch, "error": p.error, "warnings": p.warnings,
                "usage": asdict(r.usage), "cost_usd": r.cost_usd, "context": ctx,
                **({"content": r.content} if p.error else {})})
            if p.ok:  # the firm's transcript keeps the value actually used, and its notes as it carries them
                # (coerced to text, capped): the request's notes section is rebuilt from this answer
                d.value, d.notes, d.warnings = p.value, p.notes, p.warnings
                capped, notes = None, {}
                if p.notes is not None:
                    capped, cut = cap_notes(p.notes, self.exp.notes_max_words, self.exp.notes_max_chars)
                    notes = {"notes": capped}
                    if cut:  # the firm sees only the capped notes
                        d.notes, d.notes_truncated, d.notes_full = capped, True, p.notes
                        d.warnings = [*p.warnings, f"notes truncated to {self.exp.notes_max_words} words / "
                                                   f"{self.exp.notes_max_chars} characters ({len(p.notes)} chars)"]
                # the latest notes replace the memory; an answer without notes keeps it
                firm.answer(json.dumps({**notes, kind: p.value}, ensure_ascii=False), notes=capped)
                firm.close()
                return d
            d.errors.append(p.error)
            bad = strip_reasoning(r.content or "")
            firm.answer(bad[:2000] if bad else "(empty response)")
            if attempt < self.exp.max_reasks:
                messages, ctx, cache = self.reask(L, reask or prompts.reask_text(p.error, fmt))
        firm.close()
        d.fallback = True
        d.value = self.prev_price[L] if kind == "price" else None
        if kind == "price":  # a chat fallback shows in the firm's own line of that round instead
            firm.tell(prompts.fallback_text(d.value))
        return d

    async def probe_firm(self, L: str) -> dict[str, Any]:
        """Ask firm L the post-game probe in its own conversation (after all play, so no decision can depend
        on it), re-asking at most PROBE_MAX_REASKS times. A prompt that would overflow the context is
        recorded as skipped rather than failing a market whose play is complete."""
        firm, cfg, s = self.firms[L], self.spec.model, self.spec
        part, period = 2, self.exp.part_lengths[1]
        fmt = prompts.probe_format(self.letters)
        out: dict[str, Any] = {"answers": None, "valid": False, "errors": [], "warnings": [], "attempts": [],
                               "skipped": None}
        g = self.global_period(2, period) + 1  # after the last period: the same sections, without chat
        messages, ctx, cache = self.ask(L, prompts.probe_prompt(self.letters), g, 2, False)
        for attempt in range(PROBE_MAX_REASKS + 1):
            call_id = f"{s.key}/{L}/probe/a{attempt}"
            seed = self.call_seed(L, part, period, "probe", attempt)
            # tagged apart from play: no part or period of its own (the seed keeps its original coordinates)
            tag = {"run_id": s.run_id, "market": s.key, "firm": L, "part": "probe", "period": None,
                   "phase": "probe", "attempt": attempt, "call_id": call_id}
            msha = ctx["messages_sha256"]  # one hash per request: the context's, also the replay key
            cached = self.replay.pop(call_id, None)
            if cached is not None and cached["messages_sha256"] == msha:
                r = result_from_dict(cached["result"])
            else:
                try:
                    r = await self.client.chat(messages, seed=seed, tag=tag, cache_upto=cache, **cfg.chat_kwargs())
                except ContextOverflow as e:
                    out["skipped"] = "context_overflow"
                    out["errors"].append(f"ContextOverflow: {e}")
                    return out
            self.partial_calls[call_id] = {"messages_sha256": msha, "result": result_to_dict(r)}
            self._account(r)
            p = parse_probe(r.content, self.letters)
            out["attempts"].append({
                "call_id": call_id, "seed": seed, "generation_id": r.generation_id, "provider": r.provider,
                "finish_reason": r.finish_reason, "seed_possibly_ignored": r.seed_possibly_ignored,
                "provider_mismatch": r.provider_mismatch, "error": p.error, "warnings": p.warnings,
                "usage": asdict(r.usage), "cost_usd": r.cost_usd, "context": ctx, "content": r.content})
            out["answers"], out["warnings"] = p.value, p.warnings
            if p.ok:
                firm.answer(json.dumps(p.value, ensure_ascii=False))
                firm.close()
                out["valid"] = True
                return out
            out["errors"].append(p.error)
            bad = strip_reasoning(r.content or "")
            firm.answer(bad[:2000] if bad else "(empty response)")
            if attempt < PROBE_MAX_REASKS:
                messages, ctx, cache = self.reask(L, prompts.reask_text(p.error, fmt))
        firm.close()
        return out

    async def run_probe(self) -> None:
        """The post-game comprehension probe (DECISIONS.md, 2026-10-20), scored against the truth."""
        last = self.periods[-1]
        truth = {"max_consumer_price": market.P_MAX, "best_common_price": best_common_price(self.letters),
                 "last_period_prices": dict(last["prices"])}
        firms = await self.gather({L: self.probe_firm(L) for L in self.letters})
        for f in firms.values():  # an invalid or skipped probe scores wrong on every item; its partial answers
            # stay in `answers` for auditing
            f["score"] = score_probe(f["answers"] if f["valid"] else None, truth)
        self.probe = {"part": last["part"], "period": last["period"], "truth": truth, "firms": firms}

    async def gather(self, coros: dict[str, Any]) -> dict[str, Any]:
        """Run the firms' decisions concurrently. A failure does not cancel the other firms' calls: they run to
        the end (their requests may already be billed, and their results are kept for a resume), then one error
        is re-raised as is (BudgetExceeded in preference, so the runner stops scheduling markets). The exception
        is ContextOverflow: such a market is not retried under the same settings, so the other firms' calls in
        flight are cancelled at once rather than paid for. If this coroutine itself is cancelled, the
        decisions are cancelled with it."""
        tasks = {L: asyncio.ensure_future(c) for L, c in coros.items()}
        try:
            pending = set(tasks.values())
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                if any(not t.cancelled() and isinstance(t.exception(), ContextOverflow) for t in done):
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    break
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

    async def chat(self, part: int, period: int) -> list[dict[str, Any]]:
        rounds: list[dict[str, Any]] = []
        g = self.global_period(part, period)
        for r in range(1, self.exp.chat_rounds + 1):
            ds = await self.gather({L: self.decide(
                L, "message", prompts.chat_prompt(part, period, r, self.chat_spec, self.exp.notes),
                part, period, f"chat{r}") for L in self.letters})
            posted = {L: d.value for L, d in ds.items() if d.value is not None}
            for L, firm in self.firms.items():  # each firm sees the round in its own listing order
                firm.add_chat(g, prompts.round_text(r, posted, firm.order, L, own_failed=ds[L].fallback))
            rounds.append({"round": r, "messages": posted, "decisions": {L: d.to_dict() for L, d in ds.items()}})
            if not posted:
                break
        return rounds

    async def period(self, part: int, period: int) -> None:
        talk, exp = self.talk[part - 1], self.exp
        min_p = exp.part_min_periods(part)
        rounds = await self.chat(part, period) if talk else []
        ds = await self.gather({L: self.decide(
            L, "price", prompts.period_prompt(part, period, talk, exp.notes), part, period, "price")
            for L in self.letters})
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
            if last:
                closing = prompts.part_over_text(part, period, final_part=part == 2,
                                                 total_profit=sum(self.profit[L]) if part == 2 else None,
                                                 min_periods=min_p)
            else:  # the die thrown after this period, from the minimum on (a six would have ended the part)
                closing = prompts.die_line(period, die, min_p) if period >= min_p else None
            firm.summarise(self.global_period(part, period),
                           prompts.results_text(part, period, L, prices, profits, firm.order), closing)
            if last:
                if part == 1:  # shown in every later request, like the humans' instruction sheets
                    firm.add_instructions(prompts.part_instructions(
                        self.spec.n, L, 2, self.talk[1], previous_talk=self.talk[0], chat_spec=self.chat_spec,
                        notes=self.exp.notes, memory_window=self.exp.memory_window,
                        notes_words=self.exp.notes_words))

    async def run(self) -> None:
        """Play the market, or its remaining periods after `restore`."""
        if not any(f.instructions for f in self.firms.values()):
            for L, firm in self.firms.items():
                firm.start(prompts.part_instructions(self.spec.n, L, 1, self.talk[0],
                                                     chat_spec=self.chat_spec if self.talk[0] else None,
                                                     notes=self.exp.notes, memory_window=self.exp.memory_window,
                        notes_words=self.exp.notes_words))
            await self.save_checkpoint()
        schedule = [(part, t) for part in (1, 2) for t in range(1, self.exp.part_lengths[part - 1] + 1)]
        if [(p["part"], p["period"]) for p in self.periods] != schedule[:len(self.periods)]:
            raise RuntimeError("completed periods do not match the schedule")
        for part, t in schedule[len(self.periods):]:
            await self.period(part, t)
            await self.save_checkpoint()
        if self.exp.post_game_probe and self.probe is None:  # after all play; a resume reruns only this
            await self.run_probe()
            await self.save_checkpoint()

    # ------------------------------------------------------------------ checkpoints

    def snapshot(self) -> dict[str, Any]:
        """Everything needed to continue after the last completed period."""
        return {"seq": self._seq, "version": CHECKPOINT_VERSION, "spec": self.spec.to_dict(),
                "fingerprint": self.fingerprint, "texts_digest": self.texts_digest,
                "periods_completed": len(self.periods), "die_schedules": self.dies,
                "listing_orders": {L: f.order for L, f in self.firms.items()},
                "firms": {L: {"instructions": f.instructions, "transcript": f.transcript, "pending": f.pending,
                              "notes": f.notes, "notes_at": f.notes_at} for L, f in self.firms.items()},
                "periods": self.periods, "profit": self.profit, "prev_price": self.prev_price, "usage": self.usage,
                "resumed_from_period": self.resumed_from, "probe": self.probe, "partial_calls": {}}

    async def save_checkpoint(self) -> None:
        """At a period boundary: write the checkpoint and forget the finished period's call results. The
        snapshot is serialised once, here, and both the serialisation and the atomic write run in a worker
        thread, off the event loop. Safe because nothing in this market changes while it is awaited: the
        market's own calls are all finished at a boundary, and other markets never touch its state.
        The call results are kept until the new boundary is written and installed, so a cancellation at either
        await loses nothing: `save_failure_checkpoint` then writes the new boundary if it was serialised, else
        the old one plus those results."""
        if self.checkpoint is None:
            self.partial_calls, self.replay = {}, {}
            return
        self._seq += 1
        seq = self._seq
        text = await asyncio.to_thread(json.dumps, self.snapshot(), ensure_ascii=False)
        self._next_boundary = text
        await asyncio.to_thread(write_checkpoint_atomic, self.checkpoint, text, seq)
        self._boundary, self._next_boundary = text, None
        self.partial_calls, self.replay = {}, {}

    def save_failure_checkpoint(self) -> None:
        """After a failure: the last boundary checkpoint plus the results of the calls already made (or still
        unused from an earlier resume) in the period in progress; or, if the failure interrupted the write of
        a boundary already serialised, that new boundary (falling back to the former if this write fails too).
        Every write carries a `seq` and never replaces a newer checkpoint, so a boundary writer orphaned in its
        worker thread by a cancellation cannot overwrite what is written here."""
        if self.checkpoint is None:
            return
        if self._next_boundary is not None:
            try:
                write_checkpoint_atomic(self.checkpoint, self._next_boundary, self._seq)
                return
            except Exception:  # fall back to the old boundary plus the results still held
                log.warning("market %s: writing the new boundary checkpoint failed; writing the previous one "
                            "plus the period's call results", self.spec.key, exc_info=True)
        if self._boundary is None:
            return
        ck = json.loads(self._boundary)
        ck["partial_calls"] = {**self.replay, **self.partial_calls}
        self._seq += 1
        write_checkpoint_atomic(self.checkpoint, json.dumps(with_seq(ck, self._seq), ensure_ascii=False),
                                self._seq)

    def restore(self, ck: dict[str, Any]) -> str | None:
        """Load checkpoint `ck` into this fresh market. Returns why it does not match (nothing is changed
        then), or None once restored."""
        firms = ck.get("firms") or {}
        checks = [
            (ck.get("version") == CHECKPOINT_VERSION, "version"),
            (ck.get("fingerprint") == self.fingerprint, "market fingerprint"),
            (ck.get("texts_digest") == self.texts_digest, "prompt wording digest"),
            (ck.get("spec") == self.spec.to_dict(), "spec"),
            (ck.get("die_schedules") == self.dies, "die schedule"),
            (ck.get("listing_orders") == {L: f.order for L, f in self.firms.items()}, "listing orders"),
            (sorted(firms) == sorted(self.letters), "firms"),
            (ck.get("periods_completed") == len(ck.get("periods") or []), "period count"),
        ]
        if bad := [what for ok, what in checks if not ok]:
            return "checkpoint does not match: " + ", ".join(bad)
        for L, f in self.firms.items():
            ck_firm = firms[L]
            f.instructions, f.transcript = list(ck_firm["instructions"]), [dict(m) for m in ck_firm["transcript"]]
            f.pending, f.notes, f.notes_at = list(ck_firm["pending"]), ck_firm["notes"], ck_firm["notes_at"]
        self.periods, self.usage = ck["periods"], ck["usage"]
        self.profit = {L: list(v) for L, v in ck["profit"].items()}
        self.prev_price = dict(ck["prev_price"])
        self.probe = ck.get("probe")
        self.resumed_from = [*ck.get("resumed_from_period", []), len(self.periods) + 1]
        self.replay = dict(ck.get("partial_calls") or {})
        self._seq = ck["seq"] if isinstance(ck.get("seq"), int) else 0
        self._boundary = json.dumps({**ck, "resumed_from_period": self.resumed_from, "partial_calls": {}},
                                    ensure_ascii=False)
        return None

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
                         "notes_truncated": sum(bool(d.get("notes_truncated")) for d in decs),
                         "invalid_attempts": sum(len(d["errors"]) for d in decs)}
        return {
            "spec": self.spec.to_dict(),
            "fingerprint": self.fingerprint,
            "settings": {"model": self.spec.model.to_dict(), "experiment": self.exp.to_dict(),
                         "texts_digest": self.texts_digest, "parser_version": PARSER_VERSION,
                         "pilot": self.exp.pilot, "talk": dict(zip(("part1", "part2"), self.talk)),
                         "chat_spec": asdict(self.chat_spec), "feedback_all_profits": True, "exchange_rate": None,
                         "die_schedules": self.dies, "listing_orders": {L: f.order for L, f in self.firms.items()}},
            "periods": self.periods,
            "totals": totals,
            "usage": self.usage,
            # every turn of every firm (not windowed); each attempt's `context` gives the window it was sent
            "instructions": {L: f.instructions for L, f in self.firms.items()},
            "transcripts": {L: f.transcript for L, f in self.firms.items()},
            "resumed_from_period": self.resumed_from,
            "probe": self.probe,  # None when the probe is off (or, in a failure record, not yet asked)
        }


def failure_reason(e: BaseException) -> str:
    """Short reason recorded with a failed market: context_overflow, budget, budget_wait (waited too long for
    budget room; only this market fails, the run goes on) or error."""
    if isinstance(e, ContextOverflow):
        return "context_overflow"
    if isinstance(e, BudgetExceeded):
        return "budget"
    if isinstance(e, BudgetWaitTimeout):
        return "budget_wait"
    return "error"


async def run_market(spec: MarketSpec, client: ChatClient, out_dir: str | Path,
                     exp: ExperimentConfig | None = None) -> Path:
    """Play one market, resuming from its checkpoint if one matches, and write its JSON record atomically;
    returns the path. On failure, writes a `.failed.json` record (with the periods completed so far) and a
    checkpoint with the calls made in the unfinished period, and re-raises."""
    exp = exp or ExperimentConfig()
    out, fail = market_path(Path(out_dir), spec), failure_path(Path(out_dir), spec)
    ckpt = checkpoint_path(Path(out_dir), spec)
    m: Market | None = None
    try:
        m = Market(spec, client, exp, checkpoint=ckpt)
        if ckpt.exists():
            try:
                problem = m.restore(json.loads(ckpt.read_text()))
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                problem = f"checkpoint unreadable: {type(e).__name__}: {e}"
            if problem:
                print(f"CHECKPOINT ignored {spec.key}: {problem}; restarting the market", flush=True)
                # Move it aside so its higher `seq` can't block the restarted market's checkpoints.
                ckpt.replace(ckpt.with_name(f"{spec.key}.rejected-{time.time_ns()}.ckpt.json"))
                m = Market(spec, client, exp, checkpoint=ckpt)
            else:
                print(f"resume  {spec.key} from period {m.resumed_from[-1]} "
                      f"({len(m.replay)} paid call result(s) to reuse)", flush=True)
        await m.run()
    except BaseException as e:
        if m is not None:
            try:
                m.save_failure_checkpoint()
            except Exception as ce:  # never mask the market's own error
                print(f"CHECKPOINT not saved {spec.key}: {type(ce).__name__}: {ce}", flush=True)
        write_json_atomic(fail, {"spec": spec.to_dict(), "reason": failure_reason(e),
                                 **({"guard": e.guard} if isinstance(e, ContextOverflow) else {}),
                                 "error_type": type(e).__name__, "error": str(e),
                                 "traceback": traceback.format_exc(), "periods_completed": len(m.periods) if m else 0,
                                 **({k: v for k, v in m.record().items() if k != "spec"} if m else {})})
        raise
    write_json_atomic(out, m.record())
    fail.unlink(missing_ok=True)
    ckpt.unlink(missing_ok=True)
    return out
