"""Project each model's peak prompt size per cell, for the context-limit exclusions (DECISIONS.md, 2026-10-20).

    uv run python scripts/project_context.py                 # print the table (fetches endpoint metadata)
    uv run python scripts/project_context.py --write         # also write exclude_cells into config/models.toml

Method: every cell (the paper's cells by default, as `oligopoly.run` picks them; --grid for n x ordering; full
part lengths from config/experiment.toml) is played offline by the real
engine with a replay client. The client answers with outputs drawn from that model's completed pilot markets
(data/runs/pilot*/markets/*.json; all pilots pooled if the model has none with chat): real notes and chat
texts, real prices, the observed pass rate and number of chat rounds per period, and the observed rate of
invalid answers per kind (with the real invalid contents, which stay in the history and trigger re-asks). The
peak prompt is the largest prompt sent in the market, converted to tokens with the same upper bound the runtime
guard uses (`llm.prompt_token_bound`, with the model's `bytes_per_token_min`). A cell is OK when the peak over
--markets replays plus the model's max_tokens is at most --threshold (80%) of the pinned endpoint's
context_length; otherwise EXCLUDE. Cells not projected in a call keep their current exclude_cells setting, and a
model whose context length is unknown is reported and left unchanged.

--notes-at-cap: worst case for the notes cap (DECISIONS.md engine details 12): every notes field the replay writes
is exactly at the cap (notes_max_words words, notes_max_chars characters, ASCII) instead of a pilot sample.

--synthetic: no pilot markets needed. A worst-case sample instead of pilot outputs: notes at the cap (as
--notes-at-cap), every chat message at the character limit, every firm posting in every round of every Talk
period (the chat never closes early), and SYNTHETIC_INVALID of answers invalid (2,000 characters, the most the
engine keeps), so re-ask chains occur. An upper envelope for the context limits, not a cost estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oligopoly.config import ROOT, ExperimentConfig, ModelConfig, load_experiment, load_models, select_cells
from oligopoly.engine import ORDERINGS, Market, MarketSpec, derive_seed, environment_seed, is_market_record_file
from oligopoly.llm import ChatResult, Usage, find_endpoint, get_endpoints, prompt_token_bound
from oligopoly.mock import probe_answer
from oligopoly.parse import strip_reasoning

KINDS = ("price", "message")


@dataclass
class Sample:
    """Outputs observed in pilot markets."""

    notes: list[str | None] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    prices: list[int] = field(default_factory=list)
    rounds_used: Counter = field(default_factory=Counter)  # chat rounds per Talk period
    passes: int = 0  # passes in rounds that were not the closing all-pass round
    posts: int = 0
    attempts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    invalid: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in KINDS})
    markets: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.notes and self.prices and self.messages and self.rounds_used)

    @property
    def p_pass(self) -> float:
        return self.passes / max(1, self.passes + self.posts)

    def p_invalid(self, kind: str) -> float:
        return len(self.invalid[kind]) / max(1, self.attempts[kind])

    def add_decision(self, kind: str, d: dict[str, Any]) -> None:
        attempts = d.get("attempts") or [{}]
        self.attempts[kind] += len(attempts)
        for a in attempts:
            if a.get("error"):  # stored as the engine stores it in the firm's history
                bad = strip_reasoning(a.get("content") or "")
                self.invalid[kind].append(bad[:2000] if bad else "(empty response)")
        if not d.get("fallback"):
            self.notes.append(d.get("notes"))


def load_samples(paths: list[Path]) -> dict[str, Sample]:
    """Per model name (and "*" for all models pooled). Only completed markets are used."""
    out: dict[str, Sample] = {}
    for path in paths:
        if not is_market_record_file(path):
            continue
        rec = json.loads(path.read_text())
        name = rec["spec"]["model"]
        for s in (out.setdefault(name, Sample()), out.setdefault("*", Sample())):
            s.markets += 1
            for p in rec.get("periods", []):
                for L, d in p["decisions"].items():
                    s.add_decision("price", d)
                    s.prices.append(p["prices"][L])
                if p["chat"]:
                    s.rounds_used[len(p["chat"])] += 1
                    for r in p["chat"]:
                        closing = not r["messages"]
                        for d in r["decisions"].values():
                            s.add_decision("message", d)
                            if d["value"] is None:
                                s.passes += not closing
                            else:
                                s.posts += 1
                                s.messages.append(d["value"])
    return out


class ReplayClient:
    """Offline client answering with pilot outputs; records the largest prompt bound it was sent."""

    def __init__(self, sample: Sample, bytes_per_token_min: float, seed: int, chat_rounds: int):
        self.s, self.bpt, self.seed, self.chat_rounds = sample, bytes_per_token_min, seed, chat_rounds
        self.peak_tokens = 0

    def _rounds(self, tag: dict[str, Any]) -> int:
        """Rounds the chat of this period lasts (the same draw for every firm), capped at the run's rounds."""
        rng = random.Random(derive_seed("rounds", self.seed, tag["market"], tag["part"], tag["period"]))
        ks, ws = zip(*sorted(self.s.rounds_used.items()))
        return min(rng.choices(ks, ws)[0], self.chat_rounds)

    async def chat(self, messages: list[dict[str, Any]], *, seed: int | None = None,
                   tag: dict[str, Any] | None = None, **kw: Any) -> ChatResult:
        tag = tag or {}
        self.peak_tokens = max(self.peak_tokens, prompt_token_bound(messages, self.bpt))
        rng = random.Random(derive_seed("replay", self.seed, seed))
        kind = "price" if tag.get("phase") == "price" else "message"
        if tag.get("phase") == "probe":  # post-game probe: a plausible answer (no pilot sample of these yet)
            content = json.dumps(probe_answer(messages, rng))
        elif self.s.invalid[kind] and rng.random() < self.s.p_invalid(kind):
            content = rng.choice(self.s.invalid[kind])
        elif tag.get("phase") == "transition":  # notes only (the pilots had no transition call)
            content = json.dumps({"notes": rng.choice([x for x in self.s.notes if x is not None] or [""])},
                                 ensure_ascii=False)
        else:
            notes = rng.choice(self.s.notes)
            obj: dict[str, Any] = {"notes": notes} if notes is not None else {}
            if kind == "price":
                obj["price"] = rng.choice(self.s.prices)
            else:
                r, used = int(tag["phase"][4:]), self._rounds(tag)
                closing = r >= used and used < self.chat_rounds  # the all-pass round that ends the chat
                post = not closing and rng.random() >= self.s.p_pass
                obj["message"] = rng.choice(self.s.messages) if post else None
            content = json.dumps(obj, ensure_ascii=False)
        return ChatResult(content=content, reasoning=None, reasoning_details=None, finish_reason="stop",
                          native_finish_reason="stop", provider="replay", model=kw.get("model"),
                          generation_id=None, usage=Usage(), cost_usd=0.0, latency_s=0.0, attempts=1,
                          seed_possibly_ignored=False, provider_mismatch=False)


SYNTHETIC_INVALID = 0.2  # share of invalid answers in the --synthetic sample


def synthetic_sample(exp: ExperimentConfig) -> Sample:
    """The --synthetic worst-case sample (see the module docstring)."""
    bad = "z" * 2000
    return Sample(notes=[cap_note(exp.notes_max_words, exp.notes_max_chars)],
                  messages=["m" * (exp.char_limit or 300)], prices=list(range(101)),
                  rounds_used=Counter({exp.chat_rounds: 1}), passes=0, posts=1,
                  attempts={k: 10 for k in KINDS}, invalid={k: [bad] * int(10 * SYNTHETIC_INVALID) for k in KINDS},
                  markets=0)


def cap_note(max_words: int, max_chars: int) -> str:
    """ASCII notes at the cap: `max_words` words, `max_chars` characters if that many words can fill them (never
    over either, so the engine keeps them whole)."""
    per = max(1, (max_chars + 1) // max_words - 1)  # word length so that the words and spaces fit
    text = " ".join(["x" * per] * max_words)
    return text + "x" * max(0, max_chars - len(text))  # the last word takes up any slack


async def project(cfg: ModelConfig, sample: Sample, n: int, ordering: str, exp: ExperimentConfig,
                  markets: int) -> int:
    """Largest prompt-token bound over `markets` replays of the cell (played in memory, nothing written)."""
    peak = 0
    for i in range(markets):
        client = ReplayClient(sample, cfg.bytes_per_token_min, derive_seed("project", cfg.name, n, ordering, i),
                              exp.chat_rounds)
        await Market(MarketSpec("project", cfg, n, ordering, i, environment_seed(0, n, ordering, i)),
                     client, exp).run()
        peak = max(peak, client.peak_tokens)
    return peak


async def context_length(cfg: ModelConfig) -> tuple[int | None, str]:
    """(context_length, problem). A None length comes with the reason."""
    if cfg.provider_pending:
        return None, f"provider {cfg.provider!r} not pinned"
    try:
        ep = find_endpoint(await get_endpoints(cfg.model), cfg.provider)
    except Exception as e:
        return None, f"endpoint metadata unavailable ({type(e).__name__}: {e})"
    if ep is None:
        return None, f"pinned endpoint {cfg.provider!r} not found"
    if not ep.context_length:
        return None, "endpoint reports no context_length"
    return ep.context_length, ""


def write_excludes(path: Path, excludes: dict[str, list[tuple[str, int]]]) -> None:
    """Set `exclude_cells` in each model's table of models.toml, keeping everything else as written. An existing
    `exclude_cells` must be a one-line array (as this function writes it)."""
    text = path.read_text()
    for name, cells in excludes.items():
        value = "[" + ", ".join(f'["{o}", {n}]' for o, n in sorted(cells, key=lambda c: (c[1], c[0]))) + "]"
        header = re.compile(rf'^\[(?:"{re.escape(name)}"|{re.escape(name)})\]\s*$', re.M)
        m = header.search(text)
        if not m:
            raise SystemExit(f"table [{name}] not found in {path}")
        nxt = re.compile(r'^\[[^\[\]]+\]\s*$', re.M).search(text, m.end())  # next table header
        end = nxt.start() if nxt else len(text)
        body = text[m.end():end]
        old = re.search(r"^exclude_cells\s*=(.*)$", body, re.M)
        if old and old.group(1).count("[") != old.group(1).count("]"):
            raise SystemExit(f"[{name}]: exclude_cells spans several lines; make it one line before --write")
        line = f"exclude_cells = {value}" if cells else None
        if old and line:
            body = body[:old.start()] + line + body[old.end():]
        elif old:  # no exclusions left: drop the line and its newline
            body = body[:old.start()] + body[old.end() + 1:]
        elif line:
            stripped = body.rstrip("\n")
            body = stripped + "\n" + line + ("\n\n" if nxt else "\n")
        text = text[:m.end()] + body + text[end:]
    path.write_text(text)


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", help="model names (default: every live model in models.toml)")
    ap.add_argument("--n", nargs="+", type=int, help="narrows the paper's cells, as in oligopoly.run")
    ap.add_argument("--orderings", nargs="+", choices=list(ORDERINGS), help="narrows the paper's cells")
    ap.add_argument("--grid", action="store_true", help="every --n x --orderings pair (extension cells)")
    ap.add_argument("--markets", type=int, default=3, help="replays per cell; the peak is their maximum")
    ap.add_argument("--threshold", type=float, default=0.8, help="share of context_length a cell may use")
    ap.add_argument("--pilots", default=str(ROOT / "data" / "runs"), help="directory holding pilot*/markets")
    ap.add_argument("--models-file", default=str(ROOT / "config" / "models.toml"))
    ap.add_argument("--experiment-file", default=str(ROOT / "config" / "experiment.toml"))
    ap.add_argument("--write", action="store_true", help="write exclude_cells into the models file")
    ap.add_argument("--notes-at-cap", action="store_true",
                    help="worst case: every notes field at the cap (notes_max_words / notes_max_chars)")
    ap.add_argument("--synthetic", action="store_true",
                    help="worst-case synthetic sample instead of pilot outputs (no pilot markets needed)")
    a = ap.parse_args(argv)

    registry = load_models(a.models_file)
    names = a.models or [k for k, v in registry.items() if not v.is_mock]
    if unknown := [k for k in names if k not in registry]:
        raise SystemExit(f"unknown models {unknown}")
    exp = load_experiment(a.experiment_file)
    samples = {} if a.synthetic else load_samples(sorted(Path(a.pilots).glob("pilot*/markets/*.json")))
    if a.synthetic:
        samples["*"] = synthetic_sample(exp)
    elif not samples.get("*") or not samples["*"].usable:
        raise SystemExit(f"no completed pilot markets with chat under {a.pilots} (--synthetic needs none)")
    ctx = dict(zip(names, await asyncio.gather(*(context_length(registry[k]) for k in names))))
    cells = select_cells(a.n, a.orderings, a.grid)
    ns, orderings = sorted({n for n, _ in cells}), [o for o in ORDERINGS if any(c[1] == o for c in cells)]
    print(f"Part lengths {exp.part_lengths}; peak over {a.markets} replays per cell; limit = "
          f"{a.threshold:.0%} of context_length; tokens = prompt_token_bound(bytes_per_token_min) + max_tokens")
    excludes: dict[str, list[tuple[str, int]]] = {}
    for name in names:
        cfg = registry[name]
        own = samples.get(name)
        sample = own if own and own.usable else samples["*"]
        if a.notes_at_cap:
            sample = dataclasses.replace(sample, notes=[cap_note(exp.notes_max_words, exp.notes_max_chars)])
        src = "synthetic worst case" if a.synthetic else f"{own.markets} pilot market(s)" if own and own.usable \
            else "no usable pilot of its own: all pooled"
        length, problem = ctx[name]
        limit = int(length * a.threshold) if length else None
        print(f"\n{name}  ({src}; bytes/token {cfg.bytes_per_token_min}; max_tokens {cfg.max_tokens}; "
              f"context {length}; limit {limit})")
        if limit is None:
            print(f"  SKIPPED: {problem}; exclude_cells left unchanged")
            continue
        # cells not projected this time keep their current setting
        excludes[name] = [c for c in cfg.exclude_cells if (c[1], c[0]) not in cells]
        print(f"  {'n':>3} " + " ".join(f"{o:>16}" for o in orderings))
        for n in ns:
            row = []
            for o in orderings:
                if (n, o) not in cells:
                    row.append(f"{'-':>14}")
                    continue
                total = await project(cfg, sample, n, o, exp, a.markets) + (cfg.max_tokens or 0)
                ok = total <= limit
                if not ok:
                    excludes[name].append((o, n))
                row.append(f"{total:>9,} {'OK' if ok else 'EXCL':>4}")
            print(f"  {n:>3} " + " ".join(f"{c:>16}" for c in row), flush=True)
    print("\nexclude_cells:")
    for name, cs in excludes.items():
        print(f"  {name}: {sorted(cs, key=lambda c: (c[1], c[0]))}")
    if a.write:
        write_excludes(Path(a.models_file), excludes)
        print(f"written to {a.models_file}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
