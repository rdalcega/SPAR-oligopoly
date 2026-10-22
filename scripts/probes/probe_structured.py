"""Live check of OpenRouter structured outputs (`structured_outputs = true` in models.toml) on the pinned endpoints.

    uv run python scripts/probes/probe_structured.py                # gpt-oss-20b and qwen3-14b, 2 calls each
    uv run python scripts/probes/probe_structured.py --models qwen3-14b --calls 1

The prompt is a realistic chat-turn request: the engine plays a full-length NoTalk -> Talk market of n = 4 offline
(mock answers) and takes firm A's request for chat round 2 of Part 2 period 5, with the instructions and the
summaries-layout sections (period summaries, totals, notes, chat so far) exactly as a real run builds them. Each
call sends the model's own settings from models.toml plus the strict `chat_turn` json_schema
(`engine.response_schema`) and require_parameters, so a provider that cannot honour the schema must refuse the
call. Reports per call whether the provider accepted it and whether the content is valid
JSON matching the schema (exactly a `message` string-or-null; chat turns carry no notes). Budget cap $0.05
(--budget-usd); calls are logged to data/probes/structured-outputs/calls.jsonl and the summary to
results.json in the same folder (a rerun appends to the log and overwrites the summary).

Outputs go to untracked scratch (data/probes/structured-outputs/); the 2026-10-20 outputs are archived in
results/2026-10-20-probes/structured-outputs/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oligopoly.config import ROOT, load_experiment, load_models
from oligopoly.engine import MarketSpec, derive_seed, environment_seed, response_schema, run_market
from oligopoly.llm import Budget, OpenRouterClient, OpenRouterError
from oligopoly.mock import MockClient
from oligopoly.parse import parse_message

OUT = ROOT / "data" / "probes" / "structured-outputs"


async def chat_turn_request() -> list[dict[str, Any]]:
    """Firm A's chat round 2 request in Part 2 period 5 of an offline NT-T market with n = 4."""
    exp = replace(load_experiment(), post_game_probe=False)
    mock = load_models()["mock"]
    client = MockClient(record=True, malformed_rate=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        await run_market(MarketSpec("structured-probe", mock, 4, "NT-T", 0, environment_seed(0, 4, "NT-T", 0)),
                         client, Path(tmp), exp)
    want = [c for c in client.calls if c["tag"]["firm"] == "A" and c["tag"]["part"] == 2
            and c["tag"]["phase"].startswith("chat")]
    pick = next((c for c in want if c["tag"]["period"] == 5 and c["tag"]["phase"] == "chat2"), want[-1])
    return pick["messages"]


def schema_valid(content: str | None) -> tuple[bool, str]:
    """Whether `content` is, as it stands, a JSON object with exactly a string-or-null `message` (the `chat_turn`
    schema of `engine.response_schema`)."""
    try:
        obj = json.loads(content or "")
    except json.JSONDecodeError as e:
        return False, f"not JSON: {e}"
    if not isinstance(obj, dict) or set(obj) != {"message"}:
        return False, f"keys {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}"
    if not (obj["message"] is None or isinstance(obj["message"], str)):
        return False, "wrong types"
    return True, "ok"


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["gpt-oss-20b", "qwen3-14b"])
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--budget-usd", type=float, default=0.05)
    a = ap.parse_args(argv)
    registry = load_models()
    exp = load_experiment()
    messages = await chat_turn_request()
    print(f"prompt: {len(messages)} messages, {sum(len(m['content']) for m in messages):,} characters; last:\n"
          f"  {messages[-1]['content'][-300:]!r}\n")
    fmt = response_schema("message", exp.notes)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    async with OpenRouterClient(OUT / "calls.jsonl", Budget(a.budget_usd, OUT / "budget.json")) as client:
        for name in a.models:
            cfg = registry[name]
            kw = {**cfg.chat_kwargs(), "require_parameters": True}
            for i in range(a.calls):
                seed = derive_seed("structured-probe", name, i)
                row: dict[str, Any] = {"model": name, "provider": cfg.provider, "call": i, "seed": seed}
                try:
                    r = await client.chat(messages, seed=seed, response_format=fmt,
                                          tag={"purpose": "structured_probe", "model": name, "call": i}, **kw)
                except OpenRouterError as e:
                    row.update(accepted=False, error=f"{type(e).__name__}: {e}"[:600])
                else:
                    ok, why = schema_valid(r.content)
                    p = parse_message(r.content, exp.char_limit)
                    row.update(accepted=True, served=r.provider, finish_reason=r.finish_reason,
                               schema_valid=ok, schema_check=why, engine_parse_ok=p.ok, content=r.content,
                               completion_tokens=r.usage.completion_tokens, cost_usd=r.cost_usd,
                               seed_possibly_ignored=r.seed_possibly_ignored)
                results.append(row)
                print(f"{name:12} call {i}: " + (
                    f"accepted (served by {row['served']}, finish {row['finish_reason']}); valid JSON per schema: "
                    f"{row['schema_valid']} ({row['schema_check']}); content {row['content']!r:.200}"
                    if row["accepted"] else f"REFUSED: {row['error']}"), flush=True)
        spent = client.budget.spent
    (OUT / "results.json").write_text(json.dumps(
        {"run": datetime.now(timezone.utc).isoformat(), "response_format": fmt, "results": results,
         "spent_usd": spent}, indent=1, ensure_ascii=False) + "\n")
    print(f"\nspent ${spent:.4f}; results in {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
