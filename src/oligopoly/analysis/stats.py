"""Nonparametric tests as in the paper (summary §5), markets as units; hysteresis comparison (Result 5).

- Spearman correlation of market mean price with n (trend in n).
- Two-sided Mann-Whitney U between cells (pairs of n; orderings; gain from talking across n).
- Two-sided Wilcoxon signed-rank Part 1 vs Part 2 within markets.
- Time trend: Spearman(selling price, period) per market and part, then a two-sided sign test on the signs.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import benchmarks as b
from .load import num
from .measures import CELL

# (label, ordering, window) of the NoTalk and Talk observations in each ordering (T-T: an extension, Talk in both)
TALK_WINDOWS = {"NT-T": ("p2_all", "p2_from6"), "T-NT": ("p1_from6",), "T-T": ("p1_from6", "p2_all", "p2_from6")}
NOTALK_WINDOWS = {"NT-T": ("p1_from6",), "T-NT": ("p2_all", "p2_from6"), "NT-NT": ("p1_from6", "p2_all", "p2_from6")}


def _row(model, test, group, stat, p, n_obs, **kw) -> dict:
    return {"model": model, "test": test, "group": group, "statistic": num(stat), "p": num(p), "n_obs": n_obs,
            "scope": None} | kw


def _quiet(fn, *a, **kw):
    """Run a scipy test; degenerate inputs (ties only, too few points) give (nan, nan)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = fn(*a, **kw)
            return float(r[0]), float(r[1])
        except ValueError:
            return np.nan, np.nan


def spearman(x, y):
    return (np.nan, np.nan) if len(x) < 3 else _quiet(st.spearmanr, x, y)


def mannwhitney(x, y):
    return (np.nan, np.nan) if min(len(x), len(y)) == 0 else _quiet(st.mannwhitneyu, x, y, alternative="two-sided")


def wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    return (np.nan, np.nan) if len(d) == 0 or not np.any(d) else _quiet(st.wilcoxon, x, y)


def sign_test(values) -> tuple[int, int, float]:
    v = [x for x in values if np.isfinite(x) and x != 0]
    neg = sum(x < 0 for x in v)
    p = st.binomtest(neg, len(v)).pvalue if v else np.nan
    return neg, len(v) - neg, p


def run_tests(mt: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, m in mt.groupby("model"):
        # trend in n (paper: the 4 paper n; extension: all n present)
        for o in dict.fromkeys([*NOTALK_WINDOWS, *TALK_WINDOWS]):  # every ordering with a window
            for w in sorted(set(NOTALK_WINDOWS.get(o, ())) | set(TALK_WINDOWS.get(o, ()))):
                for scope, sub in (("paper_n", m[m["n"].isin(b.NS)]), ("all_n", m)):
                    sub = sub[(sub["ordering"] == o)].dropna(subset=[f"sp_{w}"])
                    if sub["n"].nunique() > 1:
                        r, p = spearman(sub["n"], sub[f"sp_{w}"])
                        rows.append(_row(model, "spearman_price_vs_n", f"{o} {w}", r, p, len(sub), scope=scope))
                # every pair of n
                sub = m[m["ordering"] == o]
                for n1, n2 in itertools.combinations(sorted(sub["n"].unique()), 2):
                    x, y = (sub.loc[sub["n"] == k, f"sp_{w}"].dropna() for k in (n1, n2))
                    u, p = mannwhitney(x, y)
                    rows.append(_row(model, "mw_pair_n", f"{o} {w} n={n1} vs n={n2}", u, p, len(x) + len(y),
                                     scope="paper_n" if {n1, n2} <= set(b.NS) else "all_n"))
        # Part 1 vs Part 2 within markets
        for (n, o), sub in m.groupby(["n", "ordering"]):
            for w2 in ("p2_all", "p2_from6"):
                s = sub.dropna(subset=["sp_p1_from6", f"sp_{w2}"])
                stat, p = wilcoxon(s["sp_p1_from6"], s[f"sp_{w2}"])
                higher = int((s[f"sp_{w2}"] > s["sp_p1_from6"]).sum())
                rows.append(_row(model, "wilcoxon_part1_vs_part2", f"n={n} {o} p1_from6 vs {w2}", stat, p, len(s),
                                 part2_higher=higher))
        # gain from talking across n (NT-T)
        g = m[m["ordering"] == "NT-T"]
        for w2 in ("p2_all", "p2_from6"):
            for n1, n2 in itertools.combinations(sorted(g["n"].unique()), 2):
                x, y = (g.loc[g["n"] == k, f"gain_{w2}"].dropna() for k in (n1, n2))
                u, p = mannwhitney(x, y)
                rows.append(_row(model, "mw_gain_pair_n", f"NT-T gain_{w2} n={n1} vs n={n2}", u, p, len(x) + len(y)))
        # between orderings, per n (order effect on Talk; hysteresis; learning)
        for n, sub in m.groupby("n"):
            for label, (o1, w1), (o2, wb) in ORDERING_COMPARISONS:
                x = sub.loc[sub["ordering"] == o1, f"sp_{w1}"].dropna()
                y = sub.loc[sub["ordering"] == o2, f"sp_{wb}"].dropna()
                if len(x) and len(y):
                    u, p = mannwhitney(x, y)
                    rows.append(_row(model, "mw_orderings", f"n={n} {label}", u, p, len(x) + len(y),
                                     mean_a=float(x.mean()), mean_b=float(y.mean()),
                                     caveat=transition_caveat(sub, (o1, w1), (o2, wb))))
    rows += trend_tests(mp)
    return pd.DataFrame(rows)


ORDERING_COMPARISONS = (
    ("Talk order effect: NT-T p2_all vs T-NT p1_from6", ("NT-T", "p2_all"), ("T-NT", "p1_from6")),
    ("Talk order effect: NT-T p2_from6 vs T-NT p1_from6", ("NT-T", "p2_from6"), ("T-NT", "p1_from6")),
    ("hysteresis: T-NT p2_all vs NT-T p1_from6", ("T-NT", "p2_all"), ("NT-T", "p1_from6")),
    ("hysteresis: T-NT p2_from6 vs NT-T p1_from6", ("T-NT", "p2_from6"), ("NT-T", "p1_from6")),
    ("hysteresis: T-NT p2_all vs NT-NT p2_all", ("T-NT", "p2_all"), ("NT-NT", "p2_all")),
    ("hysteresis: T-NT p2_from6 vs NT-NT p2_from6", ("T-NT", "p2_from6"), ("NT-NT", "p2_from6")),
    ("learning: NT-NT p1_from6 vs NT-T p1_from6", ("NT-NT", "p1_from6"), ("NT-T", "p1_from6")),
    ("learning: NT-NT p2_all vs NT-T p1_from6", ("NT-NT", "p2_all"), ("NT-T", "p1_from6")),
    # T-T (extension): Talk after Talk vs Talk after NoTalk; the same Part 1 as T-NT; Talk kept vs Talk removed
    ("prior talk: T-T p2_all vs NT-T p2_all", ("T-T", "p2_all"), ("NT-T", "p2_all")),
    ("prior talk: T-T p2_from6 vs NT-T p2_from6", ("T-T", "p2_from6"), ("NT-T", "p2_from6")),
    ("replication: T-T p1_from6 vs T-NT p1_from6", ("T-T", "p1_from6"), ("T-NT", "p1_from6")),
    ("talk kept vs removed: T-T p2_all vs T-NT p2_all", ("T-T", "p2_all"), ("T-NT", "p2_all")),
    ("talk kept vs removed: T-T p2_from6 vs T-NT p2_from6", ("T-T", "p2_from6"), ("T-NT", "p2_from6")),
)


TRANSITION_CAVEAT = ("Only one side's markets had the transition call (notes written between the parts), so a "
                     "Part 2 difference includes its effect.")


def transition_caveat(sub: pd.DataFrame, a: tuple[str, str], b_: tuple[str, str]) -> str | None:
    """The caveat for an ordering comparison involving a Part 2 window where exactly one side had the transition call
    (it runs in NT-T and T-NT only; DECISIONS.md §6 item 3)."""
    if "transition" not in sub or not any(w.startswith("p2") for _, w in (a, b_)):
        return None
    had = [bool(sub.loc[sub["ordering"] == o, "transition"].any()) for o, _ in (a, b_)]
    return TRANSITION_CAVEAT if had[0] != had[1] else None


def trend_tests(mp: pd.DataFrame) -> list[dict]:
    """Spearman(selling price, period) per market and part (all periods), then a sign test per cell and part."""
    rows = []
    for key, cell in mp.groupby(CELL + ["part"]):
        rhos = [spearman(g["period"], g["sp"])[0] for _, g in cell.groupby("market_id")]
        neg, pos, p = sign_test(rhos)
        model, n, o, part = key
        rows.append(_row(model, "sign_test_time_trend", f"n={n} {o} part {part} ({cell['phase'].iloc[0]})",
                         None, p, len(rhos), negative=neg, positive=pos,
                         undefined=len(rhos) - neg - pos))
    return rows


def hysteresis(mt: pd.DataFrame) -> pd.DataFrame:
    """Per model and n: T-NT Part 2 (NoTalk after Talk) vs NT-NT Part 2 and NT-T Part 1 (NoTalk, no prior Talk)."""
    rows = []
    for (model, n), sub in mt.groupby(["model", "n"]):
        def mean(o, w):
            v = sub.loc[sub["ordering"] == o, f"sp_{w}"].dropna()
            return (float(v.mean()), float(v.std()) if len(v) > 1 else None, len(v)) if len(v) else (None, None, 0)
        r = {"model": model, "n": n}
        for label, o, w in (("tnt_p1_talk", "T-NT", "p1_from6"), ("tnt_p2_all", "T-NT", "p2_all"),
                            ("tnt_p2_from6", "T-NT", "p2_from6"), ("ntnt_p2_all", "NT-NT", "p2_all"),
                            ("ntnt_p2_from6", "NT-NT", "p2_from6"), ("ntt_p1", "NT-T", "p1_from6"),
                            ("ntt_p2_talk", "NT-T", "p2_all"), ("tt_p1_talk", "T-T", "p1_from6"),
                            ("tt_p2_talk", "T-T", "p2_all")):
            r[label], r[f"{label}_sd"], r[f"{label}_n"] = mean(o, w)
        if "transition" in sub and sub["transition"].any():
            r["caveat"] = ("T-NT and NT-T markets had the transition call, NT-NT and T-T did not: Part 2 "
                           "differences between them include its effect.")
        if n == 4:
            r |= {"human_tnt_p2": b.CONTROLS_SELLING[("Talk->NoTalk", 2)][1].mean,
                  "human_ntnt_p2": b.CONTROLS_SELLING[("NoTalk->NoTalk", 2)][1].mean,
                  "human_ntt_p1": b.MAIN_NOTALK_SELLING[4].mean}
        rows.append(r)
    return pd.DataFrame(rows)
