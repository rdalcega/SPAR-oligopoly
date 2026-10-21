"""Paper measures (summary §5, `benchmarks.py` docstring), computed with markets as units."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import benchmarks as b
from .. import market
from ..config import PAPER_CELLS
from .load import KEY_COLS, num

CELL = ["model", "n", "ordering"]

WINDOWS: dict[str, str] = {
    "all": "all periods of both parts",
    "p1_all": "Part 1, all periods",
    "p1_from6": "Part 1, periods 6 on (the paper's window)",
    "p2_all": "Part 2, all periods (reading 1: paper numbers periods 1-53, only Part 1's first 5 dropped)",
    "p2_from6": "Part 2, its periods 6 on (reading 2: per-part numbering)",
    "p2_last5": "Part 2, last 5 periods (fn. 14 window)",
}


def window_mask(df: pd.DataFrame, w: str) -> pd.Series:
    first5 = df["period"] > b.EXCLUDED_EARLY_PERIODS
    return {
        "all": pd.Series(True, index=df.index),
        "p1_all": df["part"] == 1,
        "p1_from6": (df["part"] == 1) & first5,
        "p2_all": df["part"] == 2,
        "p2_from6": (df["part"] == 2) & first5,
        "p2_last5": (df["part"] == 2) & (df["period"] > df["part_len"] - 5),
    }[w]


def in_paper(n: int, ordering: str, variant: str = "baseline") -> bool:
    """Cells with a human counterpart: NoTalk->Talk at n in {2,4,6,8}; both controls at n = 4; baseline prompt
    only (a prompt variant is an extension, DECISIONS.md 2026-10-21)."""
    return (n, ordering) in PAPER_CELLS and variant == "baseline"


def _variants(mt: pd.DataFrame) -> dict[str, str]:
    """Each model's prompt variant (one per model: `load.check_uniform_design` refuses mixed settings)."""
    if "prompt_variant" not in mt:
        return {}
    return mt.groupby("model")["prompt_variant"].first().to_dict()


# ------------------------------------------------------------------------------------------- price levels


def market_table(mp: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """One row per market: mean selling price (`sp_<w>`) and mean posted price (`posted_<w>`) per window, and
    the gain from talking (Talk minus NoTalk; NoTalk = the NoTalk part's window, Talk = the Talk part's)."""
    out = markets.set_index("market_id").copy()
    for w in WINDOWS:
        g = mp[window_mask(mp, w)].groupby("market_id")
        out[f"sp_{w}"] = g["sp"].mean()
        out[f"posted_{w}"] = g["mean_posted"].mean()
    sign = out["ordering"].map({"NT-T": 1.0, "T-NT": -1.0, "NT-NT": np.nan})
    for w2 in ("p2_all", "p2_from6", "p2_last5"):
        out[f"gain_{w2}"] = sign * (out[f"sp_{w2}"] - out["sp_p1_from6"])
    v = out["prompt_variant"] if "prompt_variant" in out else ["baseline"] * len(out)
    out["in_paper"] = [in_paper(n, o, x) for n, o, x in zip(out["n"], out["ordering"], v)]
    return out.reset_index()


def cell_stats(mt: pd.DataFrame) -> pd.DataFrame:
    """Long table: per cell and measure, mean / SD (ddof 1, across market means) / median / n markets."""
    cols = [c for c in mt.columns if c.startswith(("sp_", "posted_", "gain_"))]
    long = mt.melt(id_vars=CELL + ["market_id"], value_vars=cols, var_name="measure").dropna(subset=["value"])
    g = long.groupby(CELL + ["measure"])["value"]
    out = g.agg(mean="mean", sd="std", median="median", min="min", max="max", n_markets="count").reset_index()
    variants = _variants(mt)
    out["in_paper"] = [in_paper(n, o, variants.get(m, "baseline"))
                       for m, n, o in zip(out["model"], out["n"], out["ordering"])]
    return out


# ------------------------------------------------------------------------------------------- Table 2


def table2(mp: pd.DataFrame) -> pd.DataFrame:
    """Percent of market-periods per selling-price bin (100 / 99 / 2-98 / <=1), per cell, part and window."""
    rows = []
    for w in ("p1_from6", "p2_all", "p2_from6"):
        sub = mp[window_mask(mp, w)]
        for key, g in sub.groupby(CELL):
            shares = g["bin4"].value_counts(normalize=True) * 100
            rows.append(dict(zip(CELL, key)) | {"window": w, "part": int(w[1]), "phase": g["phase"].iloc[0],
                                                "n_obs": len(g)} | {k: float(shares.get(k, 0.0)) for k in b.BINS4})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- Markov


def transition_counts(mp: pd.DataFrame) -> pd.DataFrame:
    """3x3 counts of selling-price state transitions t -> t+1 within each market's part."""
    counts = pd.DataFrame(0, index=list(b.STATES3), columns=list(b.STATES3))
    for _, g in mp.sort_values("period").groupby(["market_id", "part"]):
        s, t = g["state3"].tolist(), g["period"].tolist()
        for i in range(len(s) - 1):
            if t[i + 1] == t[i] + 1:
                counts.loc[s[i], s[i + 1]] += 1
    return counts


def ergodic(P: np.ndarray) -> np.ndarray | None:
    """Stationary distribution π = πP of a row-stochastic matrix; None if any row is undefined or π is not
    unique (several closed classes, e.g. markets stuck in different absorbing states)."""
    k = len(P)
    if np.isnan(P).any() or np.linalg.matrix_rank(P.T - np.eye(k)) < k - 1:
        return None
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    pi, *_ = np.linalg.lstsq(A, np.r_[np.zeros(k), 1.0], rcond=None)
    return pi


def markov_summary(counts: pd.DataFrame, states: pd.Series) -> dict:
    """Transition probabilities, Shorrocks φ, ergodic and observed shares from counts and a state series."""
    rows = counts.sum(axis=1)
    P = counts.div(rows.replace(0, np.nan), axis=0)
    pi = ergodic(P.to_numpy(dtype=float))
    out = {"n_transitions": int(rows.sum()), "n_obs": int(len(states)),
           "shorrocks": None if P.isna().any().any() else b.shorrocks({s: P.loc[s].to_dict() for s in b.STATES3}),
           "matrix": {s: {t: num(P.loc[s, t]) for t in b.STATES3} for s in b.STATES3},
           "counts": {s: {t: int(counts.loc[s, t]) for t in b.STATES3} for s in b.STATES3}}
    obs = states.value_counts(normalize=True)
    out["observed"] = {s: float(obs.get(s, 0.0)) for s in b.STATES3}
    out["ergodic"] = None if pi is None else dict(zip(b.STATES3, map(float, pi)))
    return out


def markov(mp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Per cell and part (all periods; the paper uses Talk parts): long transition table and summaries."""
    long, summ = [], []
    for key, g in mp.groupby(CELL + ["part"]):
        cell = dict(zip(CELL + ["part"], key)) | {"phase": g["phase"].iloc[0]}
        c = transition_counts(g)
        s = markov_summary(c, g["state3"])
        summ.append(cell | s)
        for fr in b.STATES3:
            for to in b.STATES3:
                long.append(cell | {"from": fr, "to": to, "count": s["counts"][fr][to], "prob": s["matrix"][fr][to]})
    return pd.DataFrame(long), summ



# ------------------------------------------------------------------------------------------- §5.3 strategy


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    """(start index, length) of maximal runs of True."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - start))
            start = None
    return out


def turn_firm(prices: dict[str, int]) -> str | None:
    """The firm at 99 when exactly one firm is at 99 and all others at 100 (taking-turns pattern)."""
    at99 = [f for f, p in prices.items() if p == 99]
    return at99[0] if len(at99) == 1 and all(p in (99, 100) for p in prices.values()) else None


def strategy_part(mp: pd.DataFrame, fp: pd.DataFrame) -> dict:
    """Price-based §5.3 measures for one market's part (sorted by period)."""
    mp = mp.sort_values("period")
    prices = [dict(zip(g["firm"], g["price"])) for _, g in fp.sort_values("period").groupby("period")]
    all100, sp = mp["all100"].tolist(), mp["sp"].tolist()
    spells = [ln for _, ln in runs(all100)]
    turns = [turn_firm(p) for p in prices]
    turn_periods = sum(ln for st, ln in runs([t is not None for t in turns])
                       if len(set(turns[st:st + ln])) >= 2)
    defect = kept = 0
    for t in range(2, len(sp)):  # fn. 18: all-100 at t-2, some firm at 99 at t-1; kept if selling price >= 99 at t
        if all100[t - 2] and 99 in prices[t - 1].values():
            defect += 1
            kept += sp[t] >= 99
    return {"first_all100": bool(all100[0]), "n_all100_spells": len(spells),
            "mean_all100_spell": float(np.mean(spells)) if spells else np.nan, "all100_periods": sum(all100),
            "turn_taking_periods": turn_periods, "defections": defect, "defections_kept": kept,
            "defections_punished": defect - kept}


def strategy(mp: pd.DataFrame, fp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-market-part strategy measures for every part, and cell aggregates (pooled defection counts)."""
    rows = []
    fpg = dict(tuple(fp.groupby(["market_id", "part"])))
    for (mid, part), g in mp.groupby(["market_id", "part"]):
        rows.append({c: g[c].iloc[0] for c in KEY_COLS} | {"part": part, "phase": g["phase"].iloc[0]}
                    | strategy_part(g, fpg[(mid, part)]))
    per = pd.DataFrame(rows)
    agg = per.groupby(CELL + ["part", "phase"]).agg(
        n_markets=("market_id", "count"), share_first_all100=("first_all100", "mean"),
        markets_first_all100=("first_all100", "sum"), mean_all100_spell=("mean_all100_spell", "mean"),
        markets_with_all100_spell=("n_all100_spells", lambda s: int((s > 0).sum())),
        markets_turn_taking=("turn_taking_periods", lambda s: int((s > 0).sum())),
        turn_taking_periods=("turn_taking_periods", "sum"), defections=("defections", "sum"),
        defections_kept=("defections_kept", "sum"), defections_punished=("defections_punished", "sum"),
    ).reset_index()
    return per, agg


# ------------------------------------------------------------------------------------------- series, theory


def timeseries(mp: pd.DataFrame) -> pd.DataFrame:
    """Mean selling price per cell and global period with the spread across markets."""
    g = mp.groupby(CELL + ["gperiod", "part", "period", "phase"])
    out = g["sp"].agg(mean="mean", sd="std", min="min", max="max", n_markets="count")
    out["q25"], out["q75"] = g["sp"].quantile(0.25), g["sp"].quantile(0.75)
    out["mean_posted"] = g["mean_posted"].mean()
    out["share_all100"] = g["all100"].mean()
    return out.reset_index()


def theory(ns: list[int], delta: float = b.CONTINUATION_PROB) -> list[dict]:
    """Grim-trigger threshold for full collusion per n at δ (exact integer-price condition)."""
    out = []
    for n in sorted(ns):
        th = market.collusion_threshold(n)
        out.append({"n": int(n), "delta": delta, "threshold_exact": th.exact, "threshold_continuous": th.continuous,
                    "collusion_equilibrium": bool(th.holds(delta)), "margin": delta - th.exact,
                    "side": "inside" if th.holds(delta) else "outside"})
    return out
