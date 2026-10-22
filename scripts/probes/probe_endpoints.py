"""Snapshot OpenRouter endpoints of the 2026-10-19 probed models to results/2026-10-20-probes/provider/endpoints.json
(no cost). Archived with the provider probe; a rerun overwrites the snapshot.

    uv run python scripts/probes/probe_endpoints.py
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODELS = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "qwen/qwen3.5-27b", "qwen/qwen3.5-9b",
          "google/gemma-4-26b-a4b-it", "openai/gpt-oss-20b", "ibm-granite/granite-4.2-8b",
          "liquid/lfm-2.5-24b-a2b"]
ROOT = Path(__file__).resolve().parents[2]
KEYS = ["seed", "reasoning", "include_reasoning", "response_format", "structured_outputs", "top_k", "min_p",
        "temperature", "top_p"]


async def main():
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "models": {}}
    async with httpx.AsyncClient(timeout=30) as h:
        for m in MODELS:
            r = await h.get(f"https://openrouter.ai/api/v1/models/{m}/endpoints")
            d = r.json()
            out["models"][m] = d.get("data", d)
            print(f"\n{m}  status={r.status_code}")
            for e in (d.get("data") or {}).get("endpoints", []):
                p = e.get("pricing", {})
                sp = set(e.get("supported_parameters") or [])

                def f(k):
                    return float(p.get(k) or 0) * 1e6
                print(f"  {str(e.get('tag')):26} q={str(e.get('quantization')):7} in={f('prompt'):.3f} "
                      f"out={f('completion'):.3f} cr={f('input_cache_read'):.3f} ctx={e.get('context_length')} "
                      f"maxc={e.get('max_completion_tokens')} st={e.get('status')} "
                      + " ".join(k for k in KEYS if k in sp)
                      + f" | up30m={e.get('uptime_last_30m')}")
    path = ROOT / "results/2026-10-20-probes/provider/endpoints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
