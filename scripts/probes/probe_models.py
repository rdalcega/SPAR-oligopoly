"""Characterise candidate OpenRouter models for provider pinning and sampling settings.

Archived: this is the 2026-10-19/20 provider probe behind the pins and sampling in config/models.toml
(DECISIONS.md "Providers and sampling" and §6 item 9), for the lineup of that date (it still lists the dropped
LFM-2.5-24B and the old DeepInfra pin for qwen3.5-9b). Its requests use the old append-only multi-turn layout
(period_prompt / results_text chains), not the summaries layout real runs now send, so a rerun no longer reflects
real requests.

Every attempt is logged by the client to results/2026-10-20-probes/provider/probe_calls.jsonl; a compact per-call
table goes to probe_results_<which>.json in the same folder, and scripts/probes/probe_summarise.py turns those into
summary_table.md there. Budget cap $3 shared across runs (budget.json there).

    uv run python scripts/probes/probe_models.py [primary|alt|long|all] [model-substring]
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from oligopoly.llm import Budget, OpenRouterClient, OpenRouterError
from oligopoly.market import payoffs
from oligopoly.prompts import (ChatSpec, chat_prompt, die_line, firm_letters, part_instructions, period_prompt,
                               results_text, round_text)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "2026-10-20-probes" / "provider"

PRIMARY = {
    "qwen/qwen3.8-27b": "deepinfra/bf16",
    "qwen/qwen3.6-27b": "deepinfra/fp8",
    "qwen/qwen3.5-27b": "deepinfra/fp8",
    "qwen/qwen3.5-9b": "deepinfra/bf16",
    "google/gemma-4-26b-a4b-it": "deepinfra/fp8",
    "openai/gpt-oss-20b": "deepinfra/bf16",
    "ibm-granite/granite-4.2-8b": "deepinfra/bf16",
    "liquid/lfm-2.5-24b-a2b": "liquid/fp8",
}
ALTERNATES = {
    "qwen/qwen3.8-27b": "alibaba",
    "qwen/qwen3.6-27b": "alibaba",
    "qwen/qwen3.5-27b": "alibaba",
    "qwen/qwen3.5-9b": "parasail/bf16",
    "google/gemma-4-26b-a4b-it": "google-vertex/global",
    "openai/gpt-oss-20b": "groq",
    "ibm-granite/granite-4.2-8b": "coreweave/bf16",
}
# Model-card sampling for thinking mode (see summary_table.md and DECISIONS.md §6 item 9); params an endpoint lacks are dropped.
SAMPLING = {
    "qwen/qwen3.8-27b": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "qwen/qwen3.6-27b": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "qwen/qwen3.5-27b": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "qwen/qwen3.5-9b": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "google/gemma-4-26b-a4b-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "openai/gpt-oss-20b": {"temperature": 1.0, "top_p": 1.0},
    "ibm-granite/granite-4.2-8b": {"temperature": 1.0, "top_p": 0.95},
    "liquid/lfm-2.5-24b-a2b": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05},
}
N, FIRM = 4, "B"
CHAT = ChatSpec()
MAXTOK = 16000


def price_messages() -> list[dict]:
    order = ["A", "B", "C", "D"]
    hist = [{"A": 80, "B": 75, "C": 90, "D": 70}, {"A": 72, "B": 70, "C": 69, "D": 70}]
    msgs = [{"role": "system", "content": part_instructions(N, FIRM, 1, talk=False)},
            {"role": "user", "content": period_prompt(1, 1)}]
    cum = 0.0
    for t, prices in enumerate(hist, start=1):
        msgs.append({"role": "assistant", "content": json.dumps({"price": prices[FIRM]})})
        prof = payoffs(prices)
        cum += prof[FIRM]
        msgs.append({"role": "user", "content": results_text(1, t, FIRM, prices, prof, order, cumulative_part=cum)
                     + "\n\n" + period_prompt(1, t + 1)})
    return msgs


def chat_messages() -> list[dict]:
    """Part 2 Talk (after Part 1 NoTalk), period 1, round 1, after a short Part 1 history."""
    order = ["A", "B", "C", "D"]
    p1 = [{"A": 80, "B": 75, "C": 90, "D": 70}, {"A": 72, "B": 70, "C": 69, "D": 70},
          {"A": 60, "B": 65, "C": 62, "D": 59}]
    msgs = [{"role": "system", "content": part_instructions(N, FIRM, 1, talk=False)},
            {"role": "user", "content": period_prompt(1, 1)}]
    cum = 0.0
    for t, prices in enumerate(p1, start=1):
        msgs.append({"role": "assistant", "content": json.dumps({"price": prices[FIRM]})})
        prof = payoffs(prices)
        cum += prof[FIRM]
        nxt = period_prompt(1, t + 1) if t < len(p1) else ""
        msgs.append({"role": "user", "content": results_text(1, t, FIRM, prices, prof, order, cumulative_part=cum)
                     + ("\n\n" + nxt if nxt else "")})
        if not nxt:
            msgs[-1]["content"] += "\n\n[...]\n\n" + part_instructions(N, FIRM, 2, talk=True, previous_talk=False,
                                                                        chat_spec=CHAT)
            msgs[-1]["content"] += "\n\n" + chat_prompt(2, 1, 1, CHAT)
    return msgs


def long_messages(target_chars: int = 240_000) -> list[dict]:
    """Synthetic n=10 history: Part 1 NoTalk (25 periods) then Part 2 Talk with 3 chat rounds per period,
    grown until ~target_chars (~60k tokens)."""
    n, me = 10, "C"
    L = firm_letters(n)
    msgs = [{"role": "system", "content": part_instructions(n, me, 1, talk=False)},
            {"role": "user", "content": period_prompt(1, 1)}]
    for t in range(1, 26):
        prices = {f: 40 + (i * 7 + t * 3) % 50 for i, f in enumerate(L)}
        msgs.append({"role": "assistant", "content": json.dumps({"price": prices[me]})})
        msgs.append({"role": "user", "content": results_text(1, t, me, prices, payoffs(prices), L)
                     + ("\n" + die_line(t, 3) if t >= 20 else "") + "\n\n" + period_prompt(1, t + 1)})
    msgs[-1]["content"] = msgs[-1]["content"].split("\n\nPart 1, period")[0] + "\n\n" + part_instructions(
        n, me, 2, talk=True, previous_talk=False, chat_spec=CHAT)
    filler = ("I propose we all hold at {p} again this period; last time {f} undercut and everyone lost. "
              "If anyone deviates we go back to low prices for several periods. Agreed?")
    t = 0
    while sum(len(m["content"]) for m in msgs) < target_chars:
        t += 1
        for r in range(1, 4):
            posted = {f: filler.format(p=90 + t % 10, f=L[(t + i) % n]) for i, f in enumerate(L) if f < me}
            msgs[-1]["content"] += "\n\n" + round_text(r, posted, L, me) + "\n\n" + chat_prompt(2, t, r, CHAT)
            msgs.append({"role": "assistant", "content": json.dumps({"message": filler.format(p=95, f="A")})})
            msgs.append({"role": "user", "content": f"Round {r} messages after you: " + " | ".join(
                f"Firm {f}: " + filler.format(p=92, f="B") for f in L if f > me)})
        msgs[-1]["content"] += "\n\n" + period_prompt(2, t, talk=True)
        prices = {f: 90 + (i + t) % 8 for i, f in enumerate(L)}
        msgs.append({"role": "assistant", "content": json.dumps({"price": prices[me]})})
        msgs.append({"role": "user", "content": results_text(2, t, me, prices, payoffs(prices), L)
                     + ("\n" + die_line(t, 3) if t >= 20 else "")})
    msgs[-1]["content"] += "\n\n" + period_prompt(2, t + 1, talk=True)
    return msgs


FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse(content: str | None, key: str) -> dict:
    """mode: strict | fenced | embedded (JSON inside prose) | empty | none."""
    if content is None or not content.strip():
        return {"mode": "empty", "ok": False, "value": None}
    s = content.strip()
    for mode, cand in (("strict", s), ("fenced", (FENCE.match(s) or [None, None])[1])):
        if cand is None:
            continue
        try:
            v = json.loads(cand)
            if isinstance(v, dict) and key in v:
                return {"mode": mode, "ok": True, "value": v[key]}
        except ValueError:
            pass
    for m in re.finditer(r"\{[^{}]*\}", s):
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict) and key in v:
                return {"mode": "embedded", "ok": True, "value": v[key]}
        except ValueError:
            pass
    return {"mode": "none", "ok": False, "value": None}


async def sampling_for(client, model, provider) -> tuple[dict, dict]:
    eps = await client.get_endpoints(model)
    ep = next(e for e in eps if e.tag == provider)
    s = dict(SAMPLING[model])
    kw = {k: s[k] for k in ("temperature", "top_p") if k in s and ep.supports(k)}
    extra = {k: s[k] for k in ("top_k", "min_p", "presence_penalty", "repetition_penalty") if k in s and ep.supports(k)}
    return kw, extra


async def call(client, rows, model, provider, label, msgs, key="price", **kw):
    extra = kw.pop("extra_body", None) or None
    try:
        r = await client.chat(msgs, model=model, provider=provider, extra_body=extra,
                              tag={"probe": "models", "label": label}, **kw)
    except OpenRouterError as e:
        rows.append({"model": model, "provider": provider, "label": label, "error": str(e)[:500]})
        print(f"[{model} {provider} {label}] ERROR {str(e)[:300]}", flush=True)
        return None
    p = parse(r.content, key)
    content = r.content or ""
    row = {
        "model": model, "provider": provider, "label": label, "served": r.provider, "model_served": r.model,
        "seed": kw.get("seed"), "reasoning_setting": kw.get("reasoning"), "max_tokens": kw.get("max_tokens"),
        "sampling": {k: kw.get(k) for k in ("temperature", "top_p")} | (extra or {}),
        "finish": r.finish_reason, "native_finish": r.native_finish_reason,
        "content": content[:2000], "content_len": len(content),
        "reasoning_present": bool(r.reasoning), "reasoning_chars": len(r.reasoning or ""),
        "reasoning_details_types": sorted({d.get("type", "?") for d in (r.reasoning_details or [])}),
        "think_tag_in_content": "<think>" in content or "</think>" in content,
        "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
        "reasoning_tokens": r.usage.reasoning_tokens, "cached_tokens": r.usage.cached_tokens,
        "cache_write_tokens": r.usage.cache_write_tokens, "cost": r.cost_usd, "latency_s": round(r.latency_s, 2),
        "attempts": r.attempts, "json_mode": p["mode"], "json_ok": p["ok"], "value": p["value"],
        "reasoning_full": r.reasoning, "seed_possibly_ignored": r.seed_possibly_ignored,
        "provider_mismatch": r.provider_mismatch,
    }
    rows.append(row)
    print(f"[{model} {provider} {label}] fin={r.finish_reason} val={p['value']!r} mode={p['mode']} "
          f"rtok={r.usage.reasoning_tokens} rch={row['reasoning_chars']} ctok={r.usage.completion_tokens} "
          f"ptok={r.usage.prompt_tokens} cache={r.usage.cached_tokens} ${r.cost_usd} {r.latency_s:.1f}s",
          flush=True)
    return r


async def probe_primary(client, rows, model, provider):
    s, extra = await sampling_for(client, model, provider)
    eps = await client.get_endpoints(model)
    ep = next(e for e in eps if e.tag == provider)
    maxtok = min(MAXTOK, ep.max_completion_tokens or MAXTOK)
    pm = price_messages()
    med = {"effort": "medium"}
    base = dict(**s, max_tokens=maxtok, extra_body=extra)
    for i, seed in enumerate([11, 11, 12, 13, 14]):
        await call(client, rows, model, provider, f"price_medium_{i}", pm, seed=seed, reasoning=med, **base)
    await call(client, rows, model, provider, "price_default", pm, seed=15, **base)
    for i in range(2):
        await call(client, rows, model, provider, f"price_disabled_{i}", pm, seed=21,
                   reasoning={"enabled": False}, **base)
    await call(client, rows, model, provider, "price_low", pm, seed=31, reasoning={"effort": "low"}, **base)
    await call(client, rows, model, provider, "price_high", pm, seed=32, reasoning={"effort": "high"}, **base)
    await call(client, rows, model, provider, "price_maxtok400", pm, seed=41, reasoning=med,
               **{**base, "max_tokens": 400})
    await call(client, rows, model, provider, "chat_medium", chat_messages(), key="message", seed=51,
               reasoning=med, **base)
    await call(client, rows, model, provider, "chat_medium_b", chat_messages(), key="message", seed=52,
               reasoning=med, **base)


async def probe_alt(client, rows, model, provider):
    s, extra = await sampling_for(client, model, provider)
    pm = price_messages()
    base = dict(**s, max_tokens=MAXTOK, extra_body=extra)
    for i in range(2):
        await call(client, rows, model, provider, f"alt_price_medium_{i}", pm, seed=11,
                   reasoning={"effort": "medium"}, **base)
    await call(client, rows, model, provider, "alt_price_disabled", pm, seed=21, reasoning={"enabled": False},
               **base)


async def probe_long(client, rows, model, provider):
    s, extra = await sampling_for(client, model, provider)
    lm = long_messages()
    await call(client, rows, model, provider, "long_60k", lm, seed=61, reasoning={"effort": "low"},
               **s, max_tokens=4000, extra_body=extra)


async def main(which: str, only: str | None) -> None:
    rows: list[dict] = []
    budget = Budget(3.0, OUT / "budget.json")
    sel = lambda d: {m: p for m, p in d.items() if not only or only in m}  # noqa: E731
    async with OpenRouterClient(OUT / "probe_calls.jsonl", budget, max_concurrency=12, timeout_s=400) as client:
        tasks = []
        if which in ("all", "primary"):
            tasks += [probe_primary(client, rows, m, p) for m, p in sel(PRIMARY).items()]
        if which in ("all", "alt"):
            tasks += [probe_alt(client, rows, m, p) for m, p in sel(ALTERNATES).items()]
        if which in ("all", "long"):
            tasks += [probe_long(client, rows, m, p) for m, p in sel(PRIMARY).items()]
        await asyncio.gather(*tasks)
    path = OUT / f"probe_results_{which}{'_' + only.replace('/', '_') if only else ''}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"spent ${budget.spent:.4f} (cap $3); wrote {path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "all", sys.argv[2] if len(sys.argv) > 2 else None))
