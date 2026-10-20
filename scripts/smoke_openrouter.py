"""Live smoke test of the OpenRouter client (costs a fraction of a cent; budget cap $0.05).

Snapshots the provider endpoints of candidate models to data/endpoints_snapshot.json, then makes one call
to a cheap model pinned to a seed-supporting provider. Logs go to data/runs/smoke/.

    uv run python scripts/smoke_openrouter.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from oligopoly.llm import Budget, OpenRouterClient, get_endpoints

CANDIDATES = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash",
              "google/gemini-3.8-flash", "qwen/qwen3.8-flash", "z-ai/glm-5.3"]
SMOKE_MODEL = "deepseek/deepseek-v4.1-flash"
ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    snapshot = {"fetched": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model in CANDIDATES:
        eps = await get_endpoints(model)
        snapshot["models"][model] = [asdict(e) for e in eps]
        print(f"\n{model}")
        for e in eps:
            print(f"  {e.tag:32} prompt ${e.pricing.get('prompt', 0) * 1e6:7.3f}/M  "
                  f"completion ${e.pricing.get('completion', 0) * 1e6:7.3f}/M  "
                  f"seed={'y' if e.supports('seed') else 'n'}  quant={e.quantization}  ctx={e.context_length}")
    out = ROOT / "data" / "endpoints_snapshot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"\nwrote {out}")

    run = ROOT / "data" / "runs" / "smoke"
    async with OpenRouterClient(run / "calls.jsonl", Budget(0.05, run / "budget.json")) as client:
        eps = await client.get_endpoints(SMOKE_MODEL)
        pinned = min((e for e in eps if e.supports("seed") and e.status in (0, None)),
                     key=lambda e: e.pricing.get("prompt", 0) + e.pricing.get("completion", 0))
        r = await client.chat(
            [{"role": "user", "content": "Reply with a single integer between 0 and 100."}],
            model=SMOKE_MODEL, provider=pinned.tag, temperature=1.0, max_tokens=64, seed=1,
            reasoning={"enabled": False}, tag={"purpose": "smoke"},
        )
    print(f"\npinned={pinned.tag}  served={r.provider}  model={r.model}  finish={r.finish_reason}")
    print(f"content={r.content!r}  reasoning={'yes' if r.reasoning else 'no'}")
    print(f"usage={asdict(r.usage)}  cost=${r.cost_usd}  latency={r.latency_s:.2f}s")
    print(f"seed_possibly_ignored={r.seed_possibly_ignored}  provider_mismatch={r.provider_mismatch}")
    print(f"raw response keys={sorted(r.raw)}  usage keys={sorted(r.raw.get('usage') or {})}")


if __name__ == "__main__":
    asyncio.run(main())
