"""Side-by-side LLM vs human values for the cells the paper ran (NT-T at n = 2, 4, 6, 8; controls at n = 4).

Every human number comes from `oligopoly.benchmarks`. Each row carries the caveat that applies to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import benchmarks as b
from .load import num

CAV = {
    "p2_window": "Part 2 window ambiguous in the paper: compare both p2_all and p2_from6 (WINDOW_AMBIGUITY).",
    "posted": "fn. 10 NoTalk posted price; the paper does not state its window (assumed periods 6 on).",
    "fn14": b.FN14_WINDOW,
    "table2_sum": "Human column sums to 101.0 as transcribed (Table 2 rounding/printing).",
    "table3": "Human matrix rounded to 2 decimals; Shorrocks reported from unrounded matrices.",
    "table4_erg8": "Human ergodic shares for n=8 sum to 1.031 as transcribed; do not cross-check with Table 3.",
    "table4_label": "WP labels the third state 'p < 100' for n=6, 8 (probably 'p < 99').",
    "turns": "Paper says taking turns was seen in 5 markets but describes only four spells.",
    "defections": "Paper removed 2 deliberate turn switches by reading chats; LLM counts are price-based candidates.",
    "spell": "Paper's unit is a market's average all-100 agreement length; LLM mean over markets with >= 1 spell.",
    "pooled": "Pooled over n = 2, 4, 6, 8 (24 human markets).",
    "transition": ("Played with the transition call (notes written between the parts), which human subjects did "
                   "not have; Part 2 comparisons are not like for like."),
}


def comparison(cells: pd.DataFrame, t2: pd.DataFrame, markov_summ: list[dict], strat: pd.DataFrame,
               tests: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    ci = cells.set_index(["model", "n", "ordering", "measure"])
    t2i = t2.set_index(["model", "n", "ordering", "window"]) if len(t2) else None
    mk = {(m["model"], m["n"], m["ordering"], m["part"]): m for m in markov_summ}
    si = strat.set_index(["model", "n", "ordering", "part"]) if len(strat) else None

    for model in sorted(cells["model"].unique()):
        mm = markets[markets["model"] == model]
        if "prompt_variant" in mm and (mm["prompt_variant"] != "baseline").any():
            continue  # a prompt variant is an extension (DECISIONS.md 2026-10-21): no human comparison
        lengths_ok = bool(mm["paper_lengths"].all())
        # the transition call between the parts departs from the paper (DECISIONS.md §6 item 3): it can affect only
        # Part 2, in the orderings whose markets had it
        with_transition = set(mm.loc[mm["transition"], "ordering"]) if "transition" in mm else set()

        def add(block, measure, n, o, window, llm, human, sd=None, hsd=None, n_llm=None, cav=()):
            cav = list(cav) + ([] if lengths_ok else ["Part lengths differ from the paper's 29/24."]) + (
                [CAV["transition"]] if o in with_transition and "p2" in str(window) else [])
            rows.append({"model": model, "block": block, "measure": measure, "n": n, "ordering": o,
                         "window": window, "llm": num(llm), "llm_sd": num(sd), "llm_n_markets": n_llm,
                         "human": human, "human_sd": hsd,
                         "diff": None if num(llm) is None or human is None else num(llm) - human,
                         "caveat": " ".join(cav)})

        def cell(n, o, measure, block, label, window, human, hsd=None, cav=()):
            key = (model, n, o, measure)
            if key in ci.index:
                r = ci.loc[key]
                add(block, label, n, o, window, r["mean"], human, r["sd"], hsd, int(r["n_markets"]), cav)

        for n in b.NS:
            h = b.MAIN_NOTALK_SELLING[n]
            cell(n, "NT-T", "sp_p1_from6", "price", "NoTalk selling price", "p1_from6", h.mean, h.sd)
            cell(n, "NT-T", "posted_p1_from6", "price", "NoTalk posted price", "p1_from6",
                 b.MAIN_NOTALK_POSTED[n], cav=[CAV["posted"]])
            for w in ("p2_all", "p2_from6"):
                h = b.MAIN_TALK_SELLING[n]
                cell(n, "NT-T", f"sp_{w}", "price", "Talk selling price", w, h.mean, h.sd, [CAV["p2_window"]])
                cell(n, "NT-T", f"gain_{w}", "price", "Gain from talking", w, b.MAIN_GAIN_FROM_TALK[n],
                     cav=[CAV["p2_window"]])
            cell(n, "NT-T", "sp_p2_last5", "price", "fn14 Talk selling price (last 5)", "p2_last5",
                 b.FN14_TALK_PROFIT[n], cav=[CAV["fn14"]])
            cell(n, "NT-T", "gain_p2_last5", "price", "fn14 gain (last 5)", "p2_last5", b.FN14_GAIN[n],
                 cav=[CAV["fn14"]])
        for (treat, part), (phase, h) in b.CONTROLS_SELLING.items():
            o = "T-NT" if treat == "Talk->NoTalk" else "NT-NT"
            for w in (("p1_from6",) if part == 1 else ("p2_all", "p2_from6")):
                cell(4, o, f"sp_{w}", "controls", f"{phase} selling price (part {part})", w, h.mean, h.sd,
                     [CAV["p2_window"]] if part == 2 else [])

        # Table 2
        for (phase, n), hbins in b.TABLE2.items():
            for w in (("p1_from6",) if phase == "NoTalk" else ("p2_all", "p2_from6")):
                if t2i is not None and (model, n, "NT-T", w) in t2i.index:
                    r = t2i.loc[(model, n, "NT-T", w)]
                    cav = ([CAV["table2_sum"]] if (phase, n) in b.TABLE2_SUM_ANOMALIES else []) + \
                          ([CAV["p2_window"]] if phase == "Talk" else [])
                    for k in b.BINS4:
                        add("table2", f"{phase} % selling price {k}", n, "NT-T", w, r[k], hbins[k],
                            n_llm=int(r["n_obs"]), cav=cav)

        # Tables 3-4 (Talk = NT-T part 2, all periods)
        for n in b.NS:
            m = mk.get((model, n, "NT-T", 2))
            if not m:
                continue
            for fr in b.STATES3:
                for to in b.STATES3:
                    add("table3", f"P({fr} -> {to})", n, "NT-T", "p2_all", m["matrix"][fr][to], b.TABLE3[n][fr][to],
                        n_llm=m["counts"][fr]["100"] + m["counts"][fr]["99"] + m["counts"][fr]["<99"],
                        cav=[CAV["table3"]])
            add("table3", "Shorrocks index", n, "NT-T", "p2_all", m["shorrocks"], b.TABLE3_SHORROCKS[n],
                cav=[CAV["table3"]])
            for s in b.STATES3:
                add("table4", f"observed share {s}", n, "NT-T", "p2_all", m["observed"][s], b.TABLE4_OBSERVED[n][s],
                    cav=[CAV["table4_label"]] if n in (6, 8) else [])
                add("table4", f"ergodic share {s}", n, "NT-T", "p2_all", (m["ergodic"] or {}).get(s),
                    b.TABLE4_ERGODIC[n][s], cav=([CAV["table4_erg8"]] if n == 8 else [])
                    + ([CAV["table4_label"]] if n in (6, 8) else []))
            if n in b.DEFECTIONS.p_leave_all100 and m["matrix"]["100"]["100"] is not None:
                add("strategy", "P(leave all-100 state)", n, "NT-T", "p2_all", 1 - m["matrix"]["100"]["100"],
                    b.DEFECTIONS.p_leave_all100[n], cav=["Paper gives this only for n=4 and n=8."])

        # §5.3 strategy (price-based), NT-T Talk part
        if si is not None:
            for n in b.NS:
                if (model, n, "NT-T", 2) in si.index:
                    r = si.loc[(model, n, "NT-T", 2)]
                    add("strategy", "mean all-100 agreement length", n, "NT-T", "p2_all", r["mean_all100_spell"],
                        b.AGREEMENT_LENGTH[n], n_llm=int(r["markets_with_all100_spell"]), cav=[CAV["spell"]])
            sub = strat[(strat["model"] == model) & (strat["ordering"] == "NT-T") & (strat["part"] == 2)
                        & strat["n"].isin(b.NS)]
            if len(sub):
                nm = int(sub["n_markets"].sum())
                add("strategy", "share of markets all-100 in first Talk period", None, "NT-T", "p2 period 1",
                    sub["markets_first_all100"].sum() / nm, b.FIRST_TALK_PERIOD_ALL100[0] / b.FIRST_TALK_PERIOD_ALL100[1],
                    n_llm=nm, cav=[CAV["pooled"]])
                add("strategy", "markets with taking turns", None, "NT-T", "p2_all",
                    int(sub["markets_turn_taking"].sum()), b.TAKING_TURNS_MARKETS_REPORTED, n_llm=nm,
                    cav=[CAV["pooled"], CAV["turns"]])
                d = int(sub["defections"].sum())
                add("strategy", "defections (fn18 price pattern)", None, "NT-T", "p2_all", d,
                    b.DEFECTIONS.candidate_cases, n_llm=nm, cav=[CAV["pooled"], CAV["defections"]])
                add("strategy", "share of defections kept (p >= 99 next)", None, "NT-T", "p2_all",
                    sub["defections_kept"].sum() / d if d else None,
                    b.DEFECTIONS.candidate_kept / b.DEFECTIONS.candidate_cases, n_llm=d,
                    cav=[CAV["pooled"], CAV["defections"]])

        # headline tests
        tm = tests[tests["model"] == model] if len(tests) else tests
        for grp, label, human in (("NT-T p1_from6", "Spearman rho, NoTalk price vs n", -0.93),
                                  ("NT-T p2_all", "Spearman rho, Talk price vs n", -0.51),
                                  ("NT-T p2_from6", "Spearman rho, Talk price vs n", -0.51)):
            r = tm[(tm["test"] == "spearman_price_vs_n") & (tm["group"] == grp) & (tm["scope"] == "paper_n")]
            if len(r):
                add("tests", f"{label} (p={_fmt(r['p'].iloc[0])})", None, "NT-T", grp.split()[1],
                    r["statistic"].iloc[0], human, n_llm=int(r["n_obs"].iloc[0]),
                    cav=[CAV["p2_window"]] if "p2" in grp else [])
        for w2 in ("p2_all", "p2_from6"):
            r = tm[(tm["test"] == "wilcoxon_part1_vs_part2")
                   & tm["group"].str.match(rf"n=[2468] NT-T p1_from6 vs {w2}$")]
            if len(r):
                add("tests", "markets with Talk > NoTalk", None, "NT-T", w2, int(r["part2_higher"].sum()),
                    b.TALK_MARKETS_HIGHER_THAN_NOTALK[0], n_llm=int(r["n_obs"].sum()), cav=[CAV["pooled"]])
    out = pd.DataFrame(rows)
    return out.astype({"n": "Int64", "llm_n_markets": "Int64"}) if len(out) else out


def _fmt(p) -> str:
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.3g}"
