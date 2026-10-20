"""Runner CLI: plays every (model, n, ordering, market) cell of a run, resumably.

    uv run python -m oligopoly.run --run-id pilot1 --models mock --n 2 4 --orderings NT-T \\
        --markets-per-cell 2 --part-lengths 6,4
    uv run python -m oligopoly.run --run-id sweep1 --models gpt-oss-20b --n 2 4 6 8 10 \\
        --orderings NT-T T-NT NT-NT --markets-per-cell 6 --budget-usd 40 --dry-run

Output goes to data/runs/<run-id>/: markets/<key>.json per completed market (skipped on rerun),
markets/<key>.failed.json for failed ones (retried on rerun), calls.jsonl (every LLM call), budget.json
and run.json (one entry per invocation). Part lengths below 20 are for pilots only: the instructions still
say "at least 20 periods" while the die schedule shrinks to the part length.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT, ConfigError, ExperimentConfig, ModelConfig, load_experiment, load_models
from .engine import ORDERINGS, ChatClient, MarketSpec, environment_seed, market_path, run_market
from .llm import Budget, BudgetExceeded, OpenRouterClient, get_endpoints
from .mock import MockClient


def build_specs(run_id: str, models: list[ModelConfig], ns: list[int], orderings: list[str],
                markets_per_cell: int, base_seed: int = 0) -> list[MarketSpec]:
    return [MarketSpec(run_id, m, n, o, i, environment_seed(base_seed, n, o, i))
            for m, n, o, i in itertools.product(models, ns, orderings, range(markets_per_cell))]


async def run_all(specs: list[MarketSpec], clients: dict[str, ChatClient], out_dir: Path, exp: ExperimentConfig,
                  max_concurrent_markets: int = 4) -> dict[str, list[str]]:
    """Run every spec whose output file does not exist yet. `clients` maps model name -> client. A failed
    market is recorded and the others continue; BudgetExceeded stops new markets from starting."""
    summary: dict[str, list[str]] = {"done": [], "skipped": [], "failed": [], "not_started": []}
    sem = asyncio.Semaphore(max_concurrent_markets)
    stop = asyncio.Event()

    async def one(spec: MarketSpec) -> None:
        if market_path(out_dir, spec).exists():
            summary["skipped"].append(spec.key)
            return
        async with sem:
            if stop.is_set():
                summary["not_started"].append(spec.key)
                return
            try:
                await run_market(spec, clients[spec.model.name], out_dir, exp)
                summary["done"].append(spec.key)
                print(f"done    {spec.key}", flush=True)
            except BudgetExceeded as e:
                stop.set()
                summary["failed"].append(spec.key)
                print(f"BUDGET  {spec.key}: {e}; no new markets will start", flush=True)
            except Exception as e:  # recorded in <key>.failed.json by run_market
                summary["failed"].append(spec.key)
                print(f"FAILED  {spec.key}: {type(e).__name__}: {e}", flush=True)

    await asyncio.gather(*(one(s) for s in specs))
    return summary


# ---------------------------------------------------------------------------------------------- dry run


def _tokens(messages: list[dict[str, Any]], chars_per_token: float) -> int:
    return int(sum(len(m["content"]) for m in messages) / chars_per_token) + 4 * len(messages)


async def simulate_tokens(n: int, ordering: str, exp: ExperimentConfig, chars_per_token: float) -> dict[str, float]:
    """Calls and input tokens (total, and cacheable as the previous call's prompt) of one mock market."""
    mock = ModelConfig("mock", "mock", "mock")
    client = MockClient(record=True)
    with tempfile.TemporaryDirectory() as tmp:
        await run_market(MarketSpec("dry-run", mock, n, ordering, 0, environment_seed(0, n, ordering, 0)),
                         client, Path(tmp), exp)
    last: dict[str, int] = {}
    total = cached = 0
    for c in client.calls:
        firm = c["tag"]["firm"]
        t = _tokens(c["messages"], chars_per_token)
        total += t
        cached += last.get(firm, 0)
        last[firm] = t
    return {"calls": len(client.calls), "input": total, "cached": cached}


async def _pricing(cfg: ModelConfig) -> tuple[dict[str, float] | None, str]:
    """USD per token for the model's pinned endpoint (or its configured `pricing`, per million tokens)."""
    if cfg.pricing:
        return {k: v / 1e6 for k, v in cfg.pricing.items()}, "config pricing"
    try:
        eps = await get_endpoints(cfg.model)
    except Exception as e:
        return None, f"endpoint pricing unavailable ({type(e).__name__})"
    if not eps:
        return None, "no endpoints"
    if cfg.provider_pending:
        return eps[0].pricing, f"provider TODO: cheapest endpoint {eps[0].tag}"
    ep = next((e for e in eps if e.tag.lower() == cfg.provider.lower()), None)
    return (ep.pricing, f"endpoint {ep.tag}") if ep else (None, f"pinned tag {cfg.provider!r} not found")


async def dry_run(specs: list[MarketSpec], out_dir: Path, exp: ExperimentConfig, out_tokens: int,
                  reasoning_tokens: int, chars_per_token: float) -> None:
    cells: dict[tuple[str, int, str], list[MarketSpec]] = defaultdict(list)
    for s in specs:
        cells[(s.model.name, s.n, s.ordering)].append(s)
    sims = {(n, o): await simulate_tokens(n, o, exp, chars_per_token) for n, o in {(s.n, s.ordering) for s in specs}}
    models = {s.model.name: s.model for s in specs}
    prices = {name: await _pricing(cfg) for name, cfg in models.items()}
    print(f"Run {specs[0].run_id if specs else '?'}: {len(cells)} cells, {len(specs)} markets; part lengths "
          f"{exp.part_lengths}{' (PILOT: die schedule shortened)' if exp.pilot else ''}")
    for name, (_, src) in prices.items():
        print(f"  pricing {name}: {src}")
    print(f"  token model: chars/token {chars_per_token}, +4/message; output {out_tokens} + reasoning "
          f"{reasoning_tokens} tokens per call; cached input = previous prompt of the same firm; "
          f"mock re-asks included")
    hdr = f"{'model':18} {'n':>3} {'order':6} {'mkts':>4} {'todo':>4} {'calls/mkt':>9} {'in tok/mkt':>11} " \
          f"{'cached':>10} {'out tok/mkt':>11} {'$/mkt':>8} {'$ todo':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    tot = defaultdict(float)
    for (name, n, o), ss in sorted(cells.items()):
        sim, (p, _) = sims[(n, o)], prices[name]
        todo = sum(not market_path(out_dir, s).exists() for s in ss)
        out = sim["calls"] * (out_tokens + reasoning_tokens)
        if p:
            fresh = sim["input"] - sim["cached"]
            cache_price = p.get("input_cache_read", p.get("prompt", 0.0))
            cost = (fresh * p.get("prompt", 0) + sim["cached"] * cache_price + out * p.get("completion", 0)
                    + sim["calls"] * reasoning_tokens * p.get("internal_reasoning", 0) + sim["calls"] * p.get("request", 0))
            nocache = cost + sim["cached"] * (p.get("prompt", 0) - cache_price)
        else:
            cost = nocache = float("nan")
        print(f"{name:18} {n:>3} {o:6} {len(ss):>4} {todo:>4} {sim['calls']:>9} {sim['input']:>11,.0f} "
              f"{sim['cached']:>10,.0f} {out:>11,.0f} {cost:>8.4f} {cost * todo:>9.2f}")
        for k, v in (("markets", todo), ("calls", sim["calls"] * todo), ("input", sim["input"] * todo),
                     ("cached", sim["cached"] * todo), ("output", out * todo), ("cost", cost * todo),
                     ("nocache", nocache * todo)):
            tot[k] += v
    print("-" * len(hdr))
    print(f"To run: {tot['markets']:.0f} markets, {tot['calls']:,.0f} calls, {tot['input']:,.0f} input tokens "
          f"({tot['cached']:,.0f} cacheable), {tot['output']:,.0f} output tokens")
    print(f"Estimated cost: ${tot['cost']:.2f} with cache reads, ${tot['nocache']:.2f} without")


# ---------------------------------------------------------------------------------------------- CLI


def _git() -> dict[str, Any]:
    def g(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="oligopoly.run", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="names from config/models.toml")
    ap.add_argument("--n", nargs="+", type=int, required=True)
    ap.add_argument("--orderings", nargs="+", choices=list(ORDERINGS), default=list(ORDERINGS))
    ap.add_argument("--markets-per-cell", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, help="USD cap for the run (required for live models)")
    ap.add_argument("--max-concurrent-markets", type=int, default=4)
    ap.add_argument("--max-concurrency", type=int, default=16, help="max simultaneous LLM calls")
    ap.add_argument("--part-lengths", default=None, help="e.g. 29,24 (default from config); <20 = pilot only")
    ap.add_argument("--seed", type=int, default=0, help="base seed of the environment draws")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and a cost estimate; no calls")
    ap.add_argument("--assume-output-tokens", type=int, default=80, help="dry run: content tokens per call")
    ap.add_argument("--assume-reasoning-tokens", type=int, default=1000, help="dry run: reasoning tokens per call")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--models-file", default=str(ROOT / "config" / "models.toml"))
    ap.add_argument("--experiment-file", default=str(ROOT / "config" / "experiment.toml"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "runs"))
    return ap.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    registry = load_models(a.models_file)
    if missing := [m for m in a.models if m not in registry]:
        raise ConfigError(f"unknown models {missing}; known: {sorted(registry)}")
    models = [registry[m] for m in a.models]
    exp = load_experiment(a.experiment_file)
    if a.part_lengths:
        exp = exp.with_part_lengths(tuple(int(x) for x in a.part_lengths.split(",")))
    out_dir = Path(a.out_root) / a.run_id
    specs = build_specs(a.run_id, models, a.n, a.orderings, a.markets_per_cell, a.seed)

    if a.dry_run:
        await dry_run(specs, out_dir, exp, a.assume_output_tokens, a.assume_reasoning_tokens, a.chars_per_token)
        return 0

    live = [m for m in models if not m.is_mock]
    for m in live:
        m.check_live()
    if live and a.budget_usd is None:
        raise ConfigError("--budget-usd is required for live models")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "run.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entries.append({"started": datetime.now(timezone.utc).isoformat(), "argv": sys.argv[1:] if argv is None else argv,
                    "git": _git(), "experiment": exp.to_dict(), "models": {m.name: m.to_dict() for m in models},
                    "budget_usd": a.budget_usd})
    manifest.write_text(json.dumps(entries, indent=1))

    log = out_dir / "calls.jsonl"
    mock = MockClient(log)
    live_client = OpenRouterClient(log, Budget(a.budget_usd, out_dir / "budget.json"),
                                   max_concurrency=a.max_concurrency) if live else None
    try:
        clients = {m.name: (mock if m.is_mock else live_client) for m in models}
        summary = await run_all(specs, clients, out_dir, exp, a.max_concurrent_markets)
    finally:
        if live_client:
            await live_client.aclose()
    print(" ".join(f"{k}={len(v)}" for k, v in summary.items()))
    if live_client:
        print(f"spent ${live_client.budget.spent:.4f} of ${live_client.budget.cap:.2f}")
    return 1 if summary["failed"] or summary["not_started"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
