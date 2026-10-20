"""Post-game comprehension probe (DECISIONS.md, 2026-10-20): per-firm answers and per-cell accuracy.

Each firm is asked, after all play, the highest price consumers pay (truth 100), the common price that
maximises each firm's per-period profit (truth 100) and every firm's price in the very last period, plus a
free-text question about deviations from what was said or agreed in the chat. A missing, invalid or skipped
(context limit) answer counts as wrong; `valid_share` and `skipped` report how often that happened. The
deviation answers are kept as text for reading, not scored.

`knows_not_collude_share`: among firms of markets with a Talk part whose probe answers are valid, the share whose
best common price answer is correct (100) while their market's mean selling price over the Talk part is below 90 (the firm knows the
collusive price but its market did not reach it); `talk_firms` is that denominator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CELL = ["model", "n", "ordering"]
FIRM_COLS = CELL + ["run_id", "market_id", "firm", "valid", "skipped", "max_consumer_price", "best_common_price",
                    "last_period_prices", "true_last_period_prices", "max_price_correct", "best_common_correct",
                    "last_period_share", "last_period_all", "deviations", "errors"]
CELL_COLS = CELL + ["markets", "firms", "valid_share", "skipped", "max_price_correct_share",
                    "best_common_correct_share", "last_period_recall_mean", "last_period_all_share", "talk_firms",
                    "knows_not_collude_share"]


def firm_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per firm of every record that has a probe (records without one are left out)."""
    rows = []
    for rec in records:
        pr = rec.get("probe")
        if not pr:
            continue
        s = rec["spec"]
        for L, f in sorted(pr["firms"].items()):
            a, sc = f.get("answers") or {}, f["score"]
            rows.append({"model": s["model"], "n": s["n"], "ordering": s["ordering"], "run_id": s["run_id"],
                         "market_id": rec.get("_market_id", s.get("run_id")), "firm": L, "valid": f["valid"],
                         "skipped": f.get("skipped"), "max_consumer_price": a.get("max_consumer_price"),
                         "best_common_price": a.get("best_common_price"),
                         "last_period_prices": a.get("last_period_prices"),
                         "true_last_period_prices": pr["truth"]["last_period_prices"],
                         "max_price_correct": sc["max_consumer_price"], "best_common_correct": sc["best_common_price"],
                         "last_period_share": sc["last_period_share"], "last_period_all": sc["last_period_all"],
                         "deviations": a.get("deviations"), "errors": " | ".join(e[:200] for e in f["errors"])})
    return pd.DataFrame(rows, columns=FIRM_COLS)


def talk_mean_sp(market_period: pd.DataFrame) -> pd.Series:
    """Mean selling price over each market's Talk part, by market_id (markets without Talk are absent)."""
    talk = market_period[market_period["phase"] == "Talk"]
    return talk.groupby("market_id")["sp"].mean()


def cell_table(firms: pd.DataFrame, talk_sp: pd.Series | None = None) -> pd.DataFrame:
    """Accuracy per (model, n, ordering): shares over all firms probed. `talk_sp` (from `talk_mean_sp`) adds the
    "knows but doesn't collude" share."""
    if firms.empty:
        return pd.DataFrame(columns=CELL_COLS)
    sp = firms["market_id"].map(talk_sp) if talk_sp is not None else pd.Series(float("nan"), index=firms.index)
    scored = sp.notna() & firms["valid"].astype(bool)  # Talk markets, valid probe answers only
    firms = firms.assign(_talk=scored, _knows_not=(firms["best_common_correct"].astype(bool) & sp.lt(90))
                         .astype(float).where(scored))
    g = firms.groupby(CELL)
    out = pd.DataFrame({
        "markets": g["market_id"].nunique(), "firms": g.size(), "valid_share": g["valid"].mean(),
        "skipped": g["skipped"].apply(lambda x: int(x.notna().sum())),
        "max_price_correct_share": g["max_price_correct"].mean(),
        "best_common_correct_share": g["best_common_correct"].mean(),
        "last_period_recall_mean": g["last_period_share"].mean(),
        "last_period_all_share": g["last_period_all"].mean(), "talk_firms": g["_talk"].sum().astype(int),
        "knows_not_collude_share": g["_knows_not"].mean()}).reset_index()
    return out[CELL_COLS]
