"""Load market records written by `oligopoly.engine` and flatten them into tidy tables."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import benchmarks as b

PHASE = {True: "Talk", False: "NoTalk"}
KEY_COLS = ["run_id", "market_id", "model", "n", "ordering", "market_index"]


@dataclass
class Tables:
    firm_period: pd.DataFrame  # one row per firm x period
    market_period: pd.DataFrame  # one row per market x period
    chat: pd.DataFrame  # one row per firm x chat round (message or pass)
    markets: pd.DataFrame  # one row per market: design info and part lengths


def load_records(run_ids: list[str], runs_root: str | Path) -> list[dict[str, Any]]:
    """All completed market records (`<key>.json`, not `.failed.json`) of the given runs."""
    recs = []
    for rid in run_ids:
        d = Path(runs_root) / rid / "markets"
        files = sorted(p for p in d.glob("*.json") if not p.name.endswith(".failed.json"))
        if not files:
            raise FileNotFoundError(f"no completed markets in {d}")
        for p in files:
            rec = json.loads(p.read_text())
            rec["_market_id"] = f"{rid}/{p.stem}"
            recs.append(rec)
    return recs


def num(x) -> float | None:
    """Float for JSON/CSV output; None for None, NaN or inf."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def check_uniform_design(records: list[dict[str, Any]]) -> None:
    """Refuse to pool markets of one model played under different settings (experiment config or model
    identity), e.g. a pilot with a full run or a record left over from an earlier configuration."""
    seen: dict[str, dict[str, str]] = {}
    for rec in records:
        st = rec["settings"]
        model = {k: v for k, v in st.get("model", {}).items() if k not in ("pricing", "timeout_s")}
        key = json.dumps({"experiment": st["experiment"], "model": model}, sort_keys=True)
        seen.setdefault(rec["spec"]["model"], {}).setdefault(key, rec["_market_id"])
    mixed = {m: list(v.values()) for m, v in seen.items() if len(v) > 1}
    if mixed:
        raise ValueError(f"markets of the same model use different settings (one example market per variant): {mixed}")


def _coerced(warnings: list[str]) -> bool:
    return any("coerced" in w for w in warnings)


def tidy(records: list[dict[str, Any]]) -> Tables:
    check_uniform_design(records)
    fp, mp, ch, mk = [], [], [], []
    for rec in records:
        s = rec["spec"]
        lengths = rec["settings"]["experiment"]["part_lengths"]
        key = {"run_id": s["run_id"], "market_id": rec["_market_id"], "model": s["model"], "n": s["n"],
               "ordering": s["ordering"], "market_index": s["market_index"]}
        talk = rec["settings"]["talk"]
        mk.append(key | {"part1_len": lengths[0], "part2_len": lengths[1], "phase1": PHASE[talk["part1"]],
                         "phase2": PHASE[talk["part2"]], "pilot": rec["settings"].get("pilot", False),
                         "paper_lengths": tuple(lengths) == (b.PART1_PERIODS, b.PART2_PERIODS)})
        for p in rec["periods"]:
            part, t = p["part"], p["period"]
            g = t if part == 1 else lengths[0] + t
            pk = key | {"part": part, "period": t, "gperiod": g, "phase": PHASE[p["talk"]],
                        "part_len": lengths[part - 1]}
            prices = p["prices"]
            sp = min(prices.values())
            mp.append(pk | {"sp": sp, "n_at_min": sum(v == sp for v in prices.values()),
                            "mean_posted": sum(prices.values()) / len(prices), "die": p["die"],
                            "state3": b.markov_state(sp), "bin4": b.price_bin4(sp),
                            "all100": all(v == 100 for v in prices.values())})
            for L, d in p["decisions"].items():
                fp.append(pk | {"firm": L, "price": prices[L], "profit": p["profits"][L], "valid": d["valid"],
                                "coerced": _coerced(d["warnings"]), "reasks": d["reasks"],
                                "invalid_attempts": len(d["errors"]), "fallback": d["fallback"],
                                "notes": d["notes"] or "", "warnings": "; ".join(d["warnings"]),
                                "errors": " | ".join(e[:120] for e in d["errors"])})
            for r in p["chat"]:
                for L, d in r["decisions"].items():
                    msg = d["value"]
                    ch.append(pk | {"round": r["round"], "firm": L, "message": msg, "passed": msg is None,
                                    "chars": len(msg) if msg else 0, "notes": d["notes"] or "",
                                    "fallback": d["fallback"], "reasks": d["reasks"],
                                    "invalid_attempts": len(d["errors"]),
                                    "truncated": any("truncated" in w for w in d["warnings"]),
                                    "coerced": _coerced(d["warnings"])})
    sort = ["market_id", "part", "period"]
    chat = pd.DataFrame(ch)
    if chat.empty:
        chat = pd.DataFrame(columns=KEY_COLS + ["part", "period", "gperiod", "phase", "round", "firm", "message",
                                                "passed", "chars", "notes", "fallback", "reasks",
                                                "invalid_attempts", "truncated", "coerced"])
    return Tables(firm_period=pd.DataFrame(fp).sort_values(sort + ["firm"], ignore_index=True),
                  market_period=pd.DataFrame(mp).sort_values(sort, ignore_index=True),
                  chat=chat.sort_values(sort + ["round", "firm"], ignore_index=True) if len(chat) else chat,
                  markets=pd.DataFrame(mk).sort_values("market_id", ignore_index=True))
