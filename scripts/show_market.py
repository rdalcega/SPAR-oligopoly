"""Print a market record compactly for reading: per period the prices, chat rounds and each firm's notes.

    uv run python scripts/show_market.py data/runs/<run>/markets/<key>.json [--periods 1.1-1.5,2.1-2.24] [--no-notes]

Probe answers are printed at the end. Notes are the firms' private memory (DECISIONS.md §6 item 3).
"""
from __future__ import annotations

import argparse
import json


def in_ranges(part: int, period: int, ranges: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
    return not ranges or any(a <= (part, period) <= b for a, b in ranges)


def parse_ranges(s: str | None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out = []
    for chunk in (s or "").split(","):
        if chunk:
            lo, _, hi = chunk.partition("-")
            a = tuple(int(x) for x in lo.split("."))
            out.append((a, tuple(int(x) for x in hi.split(".")) if hi else a))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--periods")
    ap.add_argument("--no-notes", action="store_true")
    a = ap.parse_args()
    rec = json.load(open(a.path))
    ranges = parse_ranges(a.periods)
    s = rec["spec"]
    print(f"# {s.get('model')} n={s.get('n')} {s.get('ordering')} market {s.get('market_index')}")
    for p in rec["periods"]:
        if not in_ranges(p["part"], p["period"], ranges):
            continue
        prices = " ".join(f"{f}={v}" for f, v in p["prices"].items())
        die = f" die={p['die']}" if p.get("die") else ""
        print(f"\n## {p['part']}.{p['period']} {'TALK' if p['talk'] else 'notalk'}{die} | {prices}")
        for r in p.get("chat") or []:
            msgs = [(f, x.get("value")) for f, x in (r.get("decisions") or {}).items()]
            said = [f"{f}: {m}" for f, m in msgs if m]
            print(f"  chat r{r.get('round')}: " + (" || ".join(said) if said else "(all passed)"))
        if not a.no_notes:
            for f, n in (p.get("notes") or {}).items():
                print(f"  note {f}: {n}")
    for f, x in ((rec.get("probe") or {}).get("firms") or {}).items():
        print(f"\nprobe {f}: valid={x.get('valid')} {json.dumps(x.get('answers'), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
