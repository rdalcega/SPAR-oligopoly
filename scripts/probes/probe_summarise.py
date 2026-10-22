"""Summarise the provider probe's probe_results_*.json and probe_calls.jsonl (results/2026-10-20-probes/provider/)
into a per-model table (stdout, markdown; saved there as summary_table.md).

    uv run python scripts/probes/probe_summarise.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "2026-10-20-probes" / "provider"
rows = []
for f in ("probe_results_primary.json", "probe_results_alt.json", "probe_results_long.json"):
    if (OUT / f).exists():
        rows += json.loads((OUT / f).read_text())

calls = [json.loads(line) for line in (OUT / "probe_calls.jsonl").open()]
spent = sum(c.get("charged_usd") or 0 for c in calls)
n429 = defaultdict(int)
for c in calls:
    if "429" in str(c.get("error") or ""):
        n429[(c["model_requested"], c["provider_requested"])] += 1

by = defaultdict(list)
for r in rows:
    by[(r["model"], r["provider"])].append(r)


def g(rs, label):
    return next((r for r in rs if r["label"] == label), None)


def fmt(x, nd=0):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


print("| model | provider | med rtok (mean, range) | low / high / default rtok | disabled | "
      "max ctok seen | JSON fail (stop calls) | modes | med latency s | $/price call (med) | "
      "same-seed repro (content / reasoning) | cached on repeat | 429 attempts | errors |")
print("|" + "---|" * 14)
for (m, p), rs in sorted(by.items()):
    ok = [r for r in rs if "error" not in r]
    med = [r for r in ok if r["label"].startswith(("price_medium_", "alt_price_medium_"))]
    rt = [r["reasoning_tokens"] for r in med]
    stops = [r for r in ok if r["finish"] == "stop"]
    fails = [r for r in stops if not r["json_ok"]]
    modes = sorted({r["json_mode"] for r in ok})
    lo, hi, de = (g(rs, x) for x in ("price_low", "price_high", "price_default"))
    dis = g(rs, "price_disabled_0") or g(rs, "alt_price_disabled")
    dis_s = dis["error"][:40] if dis and "error" in dis else (f"rtok={dis['reasoning_tokens']}" if dis else "-")
    a, b = (g(rs, "price_medium_0"), g(rs, "price_medium_1")) if g(rs, "price_medium_0") else \
        (g(rs, "alt_price_medium_0"), g(rs, "alt_price_medium_1"))
    if a and b and "error" not in a and "error" not in b:
        rep = f"{a['content'] == b['content']} / {a['reasoning_full'] == b['reasoning_full']}"
        cached = b["cached_tokens"]
    else:
        rep, cached = "-", "-"
    errs = sorted({r["label"] + ": " + r["error"][:60] for r in rs if "error" in r})
    print(f"| {m} | {p} | {fmt(st.mean(rt)) if rt else '-'} ({min(rt) if rt else '-'}–{max(rt) if rt else '-'}) | "
          f"{fmt(lo and lo.get('reasoning_tokens'))} / {fmt(hi and hi.get('reasoning_tokens'))} / "
          f"{fmt(de and de.get('reasoning_tokens'))} | {dis_s} | "
          f"{max((r['completion_tokens'] for r in ok if r['finish'] == 'stop'), default='-')} | "
          f"{len(fails)}/{len(stops)} | {','.join(modes)} | "
          f"{fmt(st.mean(r['latency_s'] for r in med), 1) if med else '-'} | "
          f"{fmt(st.mean(r['cost'] for r in med), 5) if med else '-'} | {rep} | {cached} | "
          f"{n429.get((m, p), 0)} | {'; '.join(errs) or '-'} |")
print(f"\nTotal charged across all logged attempts: ${spent:.4f}")
print("\nChat messages (chat_medium*):")
for (m, p), rs in sorted(by.items()):
    for r in rs:
        if r["label"].startswith("chat") and "error" not in r:
            print(f"- {m} [{p}] {r['label']} ({r['json_mode']}, {len(str(r['value'] or ''))} chars, "
                  f"rtok={r['reasoning_tokens']}): {r['value']!r}")
print("\nLong context (long_60k):")
for (m, p), rs in sorted(by.items()):
    r = g(rs, "long_60k")
    if r:
        print(f"- {m} [{p}]: " + (r["error"][:100] if "error" in r else
                                    f"prompt_tokens={r['prompt_tokens']} finish={r['finish']} json={r['json_mode']} "
                                    f"value={r['value']} latency={r['latency_s']}s ${r['cost']:.4f}"))
