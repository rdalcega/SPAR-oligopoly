"""Project the USD cost, calls and wall-clock time of a full-length run, per cell and in total (§6 item 14).

    uv run python scripts/project_cost.py --model gpt-oss-20b --markets-per-cell 6   # fetches endpoint pricing

Method:
- Every cell (the paper's cells by default, as `oligopoly.run` picks them; --grid for n x ordering; full
  part lengths from config/experiment.toml) is replayed offline by the real engine
  with `project_context`'s replay client (pilot notes, messages, prices, pass rate, chat rounds and invalid-answer
  rate of that model), --replays times. Every call's prompt (UTF-8 bytes of the message contents, as the context
  guard counts them) and phase (chat / price) is recorded.
- Prompt tokens: bytes x the model's empirical tokens-per-byte ratio from the pilot logs
  (data/runs/pilot*/calls.jsonl: sum of prompt_tokens / sum of content bytes, OK calls). The conservative bound of
  the runtime guard (`llm.prompt_token_bound` with the model's bytes_per_token_min) is reported alongside.
- Output tokens (completion_tokens, which include reasoning) are drawn per call from that model's pilot
  distribution for the call's phase.
- Notes: the pilots asked for 100-150 words on every answer; runs now state `notes_words` (500 by default) and
  write notes with prices and in the transition call only. The replay's notes are set to the stated length (in
  every later prompt); each price or transition call adds the stated notes' extra tokens over the pilots' mean
  notes to its drawn output (assumes firms use the full allowance: on the high side, as a budget cap needs), and
  each chat call's draw loses the pilots' mean notes.
- Price: the pinned endpoint's pricing (prompt, completion, per request). Cache-read pricing is applied to the
  pilot's cached share of prompt tokens only if the pilot showed cache hits for the model.
- Time per market: sequential steps (each chat round and each price decision of a period is one step in which all
  firms answer in parallel; a re-ask adds a step) x the pilot's p50 latency for that phase; x p90 as a rougher
  upper figure. With the rolling window (DECISIONS.md §6 item 3, 2026-10-20) prompts stay bounded (peak ~11k
  tokens for gpt-oss at n = 10, ~17k by the guard's bound), close to the pilots' sizes.
- Caveats: the pilots ran with append-only histories and before the chat-turn fix (DECISIONS.md §6 item 1,
  2026-10-20), so their invalid-answer (re-ask) rate on chat turns, which the replay reproduces, is inflated:
  in the n = 10 pilot ~19% of chat turns ended as forced passes after price answers. The projection is on the
  high side for calls and cost.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_context import ReplayClient, Sample, cap_note, load_samples

from oligopoly.config import ROOT, ExperimentConfig, ModelConfig, load_experiment, load_models, select_cells
from oligopoly.engine import ORDERINGS, Market, MarketSpec, derive_seed, environment_seed
from oligopoly.llm import _content_text, find_endpoint, get_endpoints, prompt_token_bound


def content_bytes(messages: list[dict[str, Any]]) -> int:
    return sum(len(_content_text(m.get("content")).encode()) for m in messages)


def phase_kind(phase: str | None) -> str:
    """"price", "probe" (the post-game probe, kept apart from play) or "chat"."""
    return phase if phase in ("price", "probe") else "chat"


def pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


@dataclass
class PilotStats:
    """Per-call statistics of one model's OK calls in the pilot logs."""

    calls: int = 0
    prompt_tokens: int = 0
    prompt_bytes: int = 0
    cached_tokens: int = 0
    output: dict[str, list[int]] = field(default_factory=lambda: {"chat": [], "price": [], "probe": []})
    latency: dict[str, list[float]] = field(default_factory=lambda: {"chat": [], "price": [], "probe": []})

    def draws(self, kind: str) -> str:
        """The kind whose pilot sample stands in for `kind`: the probe uses the price calls' sample until the
        pilots have probe calls."""
        return kind if self.output[kind] else "price"

    @property
    def tokens_per_byte(self) -> float:
        return self.prompt_tokens / self.prompt_bytes

    @property
    def cached_share(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


def pilot_stats(call_logs: list[Path], model_id: str) -> PilotStats:
    s = PilotStats()
    for path in call_logs:
        for line in path.open():
            r = json.loads(line)
            if r.get("event") or r.get("error") or r.get("model_requested") != model_id or not r.get("usage"):
                continue
            u, kind = r["usage"], phase_kind((r.get("tag") or {}).get("phase"))
            s.calls += 1
            s.prompt_tokens += u["prompt_tokens"]
            s.prompt_bytes += content_bytes(r["request"]["messages"]) if "request" in r else r["prompt_bytes"]
            s.cached_tokens += u.get("cached_tokens") or 0
            s.output[kind].append(u["completion_tokens"])
            if r.get("latency_s") is not None:
                s.latency[kind].append(r["latency_s"])
    return s


class CostReplay(ReplayClient):
    """Replay client that also records each call's prompt bytes, token bound and phase."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, *, seed=None, tag=None, **kw):
        tag = tag or {}
        self.calls.append({"bytes": content_bytes(messages), "bound": prompt_token_bound(messages, self.bpt),
                           "kind": phase_kind(tag.get("phase")), "seed": seed,
                           "step": (tag.get("part"), tag.get("period"), tag.get("phase")),
                           "attempt": tag.get("attempt", 0)})
        return await super().chat(messages, seed=seed, tag=tag, **kw)


def price_calls(calls: list[dict[str, Any]], st: PilotStats, pricing: dict[str, float], bound_tpb: bool,
                seed: int, extra_out: float = 0.0, chat_notes_out: float = 0.0) -> tuple[float, int, int]:
    """(USD, prompt tokens, output tokens) of the recorded calls. Prompt tokens are bytes x the empirical
    tokens per byte, or the guard's bound when `bound_tpb`; output tokens are drawn from the pilot per phase."""
    rng = random.Random(seed)
    cache_read = pricing.get("input_cache_read")
    share = st.cached_share if st.cached_tokens and cache_read is not None else 0.0
    usd = ptok = otok = 0
    for c in calls:
        p = c["bound"] if bound_tpb else c["bytes"] * st.tokens_per_byte
        o = rng.choice(st.output[st.draws(c["kind"])])
        phase = c.get("step", (None, None, c["kind"]))[2]
        if phase in ("price", "transition"):  # notes are written with prices and in the transition call
            o += extra_out
        elif c["kind"] == "chat":  # pilot chat answers carried notes; chat answers no longer do
            o = max(1.0, o - chat_notes_out)
        usd += (p * (1 - share) * pricing["prompt"] + p * share * (cache_read or 0.0)
                + o * pricing["completion"] + pricing.get("request", 0.0))
        ptok += p
        otok += o
    return usd, int(ptok), otok


def steps(calls: list[dict[str, Any]]) -> dict[str, int]:
    """Sequential steps per phase kind: one per (part, period, phase), plus one per extra re-ask level."""
    deepest: dict[tuple, int] = defaultdict(int)
    for c in calls:
        deepest[c["step"]] = max(deepest[c["step"]], c["attempt"])
    out = {"chat": 0, "price": 0, "probe": 0}
    for (_, _, phase), a in deepest.items():
        out[phase_kind(phase)] += 1 + a
    return out


def cells_to_project(cfg: ModelConfig, ns: list[int] | None, orderings: list[str] | None, grid: bool = False
                     ) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(cells to project, cells the model excludes), chosen as `oligopoly.run` chooses them (the paper's cells
    without `ns`); excluded cells are not run, so they are left out of the projection, as in a run."""
    pairs = select_cells(ns, orderings, grid)
    return [c for c in pairs if not cfg.excludes(*c)], [c for c in pairs if cfg.excludes(*c)]


async def replay_cell(cfg: ModelConfig, sample: Sample, n: int, ordering: str, exp: ExperimentConfig,
                      i: int) -> list[dict[str, Any]]:
    client = CostReplay(sample, cfg.bytes_per_token_min, derive_seed("cost", cfg.name, n, ordering, i),
                        exp.chat_rounds)
    await Market(MarketSpec("project", cfg, n, ordering, i, environment_seed(0, n, ordering, i)), client, exp).run()
    return client.calls


async def endpoint_pricing(cfg: ModelConfig) -> dict[str, float]:
    ep = find_endpoint(await get_endpoints(cfg.model), cfg.provider)
    if ep is None or not ep.price_known:
        raise SystemExit(f"pinned endpoint {cfg.provider!r} of {cfg.model} not found or pricing unknown")
    return ep.pricing


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="name in config/models.toml")
    ap.add_argument("--markets-per-cell", type=int, default=6)
    ap.add_argument("--n", nargs="+", type=int, help="narrows the paper's cells, as in oligopoly.run")
    ap.add_argument("--orderings", nargs="+", choices=list(ORDERINGS), help="narrows the paper's cells")
    ap.add_argument("--grid", action="store_true", help="every --n x --orderings pair (extension cells)")
    ap.add_argument("--replays", type=int, default=2, help="offline replays per cell; results are their mean")
    ap.add_argument("--pilots", default=str(ROOT / "data" / "runs"), help="directory holding pilot*/")
    ap.add_argument("--models-file", default=str(ROOT / "config" / "models.toml"))
    ap.add_argument("--experiment-file", default=str(ROOT / "config" / "experiment.toml"))
    a = ap.parse_args(argv)

    cfg = load_models(a.models_file)[a.model]
    exp = load_experiment(a.experiment_file)
    pilots = Path(a.pilots)
    samples = load_samples(sorted(pilots.glob("pilot*/markets/*.json")))
    sample = samples.get(a.model)
    if not sample or not sample.usable:
        raise SystemExit(f"no usable pilot markets of {a.model}")
    pilot_notes = [len(x) for x in sample.notes if x]
    if exp.notes:  # notes at the stated length (about 6 characters per word)
        stated = cap_note(exp.notes_words, 6 * exp.notes_words)
        sample = dataclasses.replace(sample, notes=[stated])
    st = pilot_stats(sorted(pilots.glob("pilot*/calls.jsonl")), cfg.model)
    if not st.calls or not st.output["chat"] or not st.output["price"]:
        raise SystemExit(f"no OK pilot calls of {cfg.model} for both phases")
    pricing = await endpoint_pricing(cfg)
    # output tokens the stated notes add over the pilots' (bytes x the pilot's tokens per byte)
    extra_out = max(0.0, 6 * exp.notes_words - statistics.mean(pilot_notes or [0])) * st.tokens_per_byte \
        if exp.notes else 0.0
    chat_notes_out = statistics.mean(pilot_notes or [0]) * st.tokens_per_byte  # pilot notes in chat outputs
    print(f"notes at the stated {exp.notes_words} words: +{extra_out:.0f} output tokens per price or transition "
          f"call; chat calls -{chat_notes_out:.0f} (chat answers carry no notes)")
    lat = {k: (statistics.median(v), pctl(v, 0.9)) for k, v in st.latency.items() if v}
    lat.setdefault("probe", lat["price"])
    share = st.cached_share if pricing.get("input_cache_read") is not None else 0.0
    print(f"{a.model} at {cfg.provider}: ${pricing['prompt'] * 1e6:.3f}/M prompt, ${pricing['completion'] * 1e6:.3f}/M "
          f"completion; cache {'read at $%.3f/M on %.0f%% of prompt tokens' % (pricing['input_cache_read'] * 1e6, 100 * share) if share else 'not applied (no cache hits in the pilot)'}")
    print(f"pilot: {st.calls} OK calls; {st.tokens_per_byte:.3f} prompt tokens/byte (bound: "
          f"{1 / cfg.bytes_per_token_min:.3f} + per-message overhead); output tokens mean chat "
          f"{statistics.mean(st.output['chat']):.0f} / price {statistics.mean(st.output['price']):.0f}; latency p50/p90 "
          f"chat {lat['chat'][0]:.1f}/{lat['chat'][1]:.1f} s, price {lat['price'][0]:.1f}/{lat['price'][1]:.1f} s")
    print(f"part lengths {exp.part_lengths}; {a.markets_per_cell} markets per cell; mean of {a.replays} replays\n")
    hdr = (f"{'n':>3} {'ordering':<6} {'calls/mkt':>9} {'steps':>6} {'peak tok':>9} {'$/mkt':>8} {'$/cell':>8} "
           f"{'$/cell bound':>12} {'h/mkt p50':>9} {'h/mkt p90':>9}")
    print(hdr)
    tot = defaultdict(float)
    todo, excluded = cells_to_project(cfg, a.n, a.orderings, a.grid)
    for n, o in todo:
        reps = [await replay_cell(cfg, sample, n, o, exp, i) for i in range(a.replays)]
        r = defaultdict(float)
        for i, calls in enumerate(reps):
            usd, _, _ = price_calls(calls, st, pricing, False, derive_seed("out", n, o, i), extra_out, chat_notes_out)
            usd_b, _, _ = price_calls(calls, st, pricing, True, derive_seed("out", n, o, i), extra_out, chat_notes_out)
            sp = steps(calls)
            r["calls"] += len(calls)
            r["steps"] += sp["chat"] + sp["price"] + sp["probe"]
            r["peak"] = max(r["peak"], max(c["bytes"] for c in calls) * st.tokens_per_byte)
            r["usd"] += usd
            r["usd_b"] += usd_b
            r["h50"] += sum(sp[k] * lat[k][0] for k in sp) / 3600
            r["h90"] += sum(sp[k] * lat[k][1] for k in sp) / 3600
        m = {k: (v if k == "peak" else v / a.replays) for k, v in r.items()}
        k = a.markets_per_cell
        print(f"{n:>3} {o:<6} {m['calls']:>9,.0f} {m['steps']:>6,.0f} {m['peak']:>9,.0f} {m['usd']:>8.3f} "
              f"{m['usd'] * k:>8.2f} {m['usd_b'] * k:>12.2f} {m['h50']:>9.2f} {m['h90']:>9.2f}", flush=True)
        for key in ("calls", "usd", "usd_b", "h50", "h90"):
            tot[key] += m[key] * k
        tot["markets"] += k
    if excluded:
        print(f"excluded (exclude_cells, not run, not projected): {', '.join(f'{o} n={n}' for n, o in excluded)}")
    print(f"\nTOTAL {tot['markets']:.0f} markets: {tot['calls']:,.0f} calls, ${tot['usd']:.2f} "
          f"(${tot['usd_b']:.2f} with the conservative token bound); market-hours {tot['h50']:.0f} (p50) / "
          f"{tot['h90']:.0f} (p90), i.e. wall-clock / concurrent markets")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
