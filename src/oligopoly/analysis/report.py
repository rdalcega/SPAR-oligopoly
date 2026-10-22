"""Build every analysis table from one or more runs and write CSV/JSON files plus `summary.json`."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import benchmarks as b
from ..config import ROOT, load_models
from . import chat as chat_mod
from . import human, measures, mistakes, stats
from . import probe as probe_mod
from .load import load_records, tidy

LIMITATIONS = [
    "Chat coding is regex-based: prices, endorsements and agreement markers can be missed or misattributed "
    "(e.g. sarcasm, numbers in unusual phrasings, agreements spread over periods). Read the saved examples.",
    "An 'agreement' needs a strict majority backing one price within the period; implicit continuation of an "
    "earlier agreement only counts when a firm replies with a bare agreement marker.",
    "Punishment is a price cut at t+1 by a firm that did not undercut at t; it does not separate punishment "
    "from a firm simply following the market down.",
    "Deception signals rely on the notes written with the price stating a lower own price or a deviation plan; "
    "notes are the firm's self-written memory, not full reasoning, so absence of a signal is weak evidence.",
    "Agreements outlive their period only through a firm's notes (summaries layout), so a firm that breaks an "
    "agreement its latest notes don't mention may have forgotten it; the coding does not separate this from "
    "deliberate cheating.",
    "When the `transition` setting is on, the notes-only transition call between the parts runs only in NT-T and "
    "T-NT, not NT-NT or T-T, and the humans had no such step: Part 2 comparisons across orderings and with humans "
    "include its effect (rows say when).",
    "Mistake heuristics (notes contradicting price, low price without context, misread rules) are screening "
    "flags for manual reading, not final classifications.",
    "fn. 18 defections are counted from prices only; the paper removed deliberate turn switches by reading chats.",
    "Comparisons with humans are valid only for runs with the paper's part lengths (29/24); rows say otherwise.",
]


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", double_precision=15)) if len(df) else []


def chat_examples(msgs: pd.DataFrame, periods: pd.DataFrame, chat: pd.DataFrame, fp: pd.DataFrame,
                  per_model: int = 8) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    mcols = ["model", "market_id", "n", "part", "period", "round", "firm", "message", "prices", "endorsed"]
    for cat in ("proposal", "agreement", "threat", "turns", "accusation", "apology"):
        sel = msgs[msgs[cat]].groupby("model").head(per_model) if len(msgs) else msgs
        out[f"message_{cat}"] = records(sel[mcols]) if len(sel) else []
    chat_g = dict(tuple(chat.groupby(["market_id", "part", "period"]))) if len(chat) else {}
    fp_g = dict(tuple(fp.groupby(["market_id", "part", "period"])))

    def transcript(r) -> dict:
        key = (r["market_id"], r["part"], r["period"])
        c, f = chat_g.get(key), fp_g[key]
        lines = [] if c is None else [f"r{x['round']} {x['firm']}: {x['message']}" for _, x in c.iterrows()
                                      if not x["passed"]]
        return {"model": r["model"], "market_id": r["market_id"], "n": int(r["n"]), "part": int(r["part"]),
                "period": int(r["period"]), "agreed_price": None if pd.isna(r["agreed_price"]) else int(r["agreed_price"]),
                "undercutters": r["undercutters"], "deception": r["deception_notes"], "chat": lines,
                "prices": dict(zip(f["firm"], map(int, f["price"]))), "notes": dict(zip(f["firm"], f["notes"]))}

    if len(periods):
        cats = {"agreement_kept": periods["kept"].eq(True), "agreement_broken": periods["kept"].eq(False),
                "breach_punished": periods["punished_next"].eq(True),
                "breach_unpunished": periods["punished_next"].eq(False),
                "deception_notes": periods["deception_notes"].str.len() > 0,
                "deception_priced": periods["deception_priced"].str.len() > 0}
        for cat, mask in cats.items():
            sel = periods[mask].groupby("model").head(per_model)
            out[f"period_{cat}"] = [transcript(r) for _, r in sel.iterrows()]
    return out


def excluded_cells_from_config(models_file: str | Path | None = None) -> dict[str, set[tuple[str, int]]]:
    """{model name: {(ordering, n), ...}} from `exclude_cells` in models.toml (DECISIONS.md: context limits)."""
    models = load_models(models_file) if models_file else load_models()
    return {name: set(m.exclude_cells) for name, m in models.items() if m.exclude_cells}


def drop_excluded(records: list[dict[str, Any]], excluded: dict[str, set[tuple[str, int]]]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records outside excluded cells, and one row per excluded cell of a model present in the data (with the
    number of completed markets found there and left out). Excluded cells are reported as missing, not imputed."""
    keep, dropped = [], {}
    for rec in records:
        sp = rec["spec"]
        cell = (sp["model"], sp["ordering"], int(sp["n"]))
        if (cell[1], cell[2]) in excluded.get(cell[0], ()):
            dropped[cell] = dropped.get(cell, 0) + 1
        else:
            keep.append(rec)
    models = {r["spec"]["model"] for r in records}
    rows = [{"model": m, "ordering": o, "n": n, "markets_dropped": dropped.get((m, o, n), 0)}
            for m in sorted(models) for o, n in sorted(excluded.get(m, ()), key=lambda c: (c[1], c[0]))]
    return keep, rows


def build(run_ids: list[str], runs_root: str | Path,
          excluded: dict[str, set[tuple[str, int]]] | None = None,
          skip_mismatched_prompts: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """`excluded`: cells to leave out per model; default: `exclude_cells` in config/models.toml.
    `skip_mismatched_prompts`: leave out (with a warning) records played under other prompt wording than
    their run records, instead of refusing (see `load_records`)."""
    recs, excluded_rows = drop_excluded(load_records(run_ids, runs_root, skip_mismatched_prompts),
                                           excluded_cells_from_config() if excluded is None else excluded)
    if not recs:
        raise ValueError("every completed market is in an excluded cell")
    t = tidy(recs)
    mt = measures.market_table(t.market_period, t.markets)
    cells = measures.cell_stats(mt)
    t2 = measures.table2(t.market_period)
    mk_long, mk_summ = measures.markov(t.market_period)
    strat_m, strat = measures.strategy(t.market_period, t.firm_period)
    tests = stats.run_tests(mt, t.market_period)
    hyst = stats.hysteresis(mt)
    msgs = chat_mod.code_messages(t.chat) if len(t.chat) else pd.DataFrame(
        columns=list(t.chat.columns) + ["threat", "endorsed", "agreement"])
    periods = chat_mod.code_periods(msgs, t.chat, t.firm_period)
    chat_cells = chat_mod.chat_summary(periods, t.chat)
    ff = mistakes.flag_firm_periods(t.firm_period, msgs)
    fc = mistakes.flag_chat(t.chat) if len(t.chat) else t.chat
    ft = mistakes.flag_transition(t.transition) if len(t.transition) else None
    mis = mistakes.rates(ff, fc, ft)
    comp = human.comparison(cells, t2, mk_summ, strat, tests, t.markets)
    ts = measures.timeseries(t.market_period)
    ns = sorted(t.markets["n"].unique())
    probe_firms = probe_mod.firm_table(recs)
    probe_cells = probe_mod.cell_table(probe_firms, probe_mod.talk_mean_sp(t.market_period))
    mis_cells = mistakes.cell_rates(ff, fc, ft)

    tables = {
        "firm_period": ff, "market_period": t.market_period, "markets": mt, "cells": cells,
        "extensions": cells[~cells["in_paper"]], "table2": t2, "markov_transitions": mk_long,
        "strategy_market_part": strat_m, "strategy_cells": strat, "tests": tests, "hysteresis": hyst,
        "human_comparison": comp, "chat_messages": msgs, "chat_turns": fc, "chat_periods": periods,
        "chat_cells": chat_cells, "mistakes_by_model": mis, "mistakes_cells": mis_cells, "timeseries": ts,
        "excluded_cells": pd.DataFrame(excluded_rows, columns=["model", "ordering", "n", "markets_dropped"]),
        "probe_answers": probe_firms, "probe_cells": probe_cells,
    }
    counts = t.markets.groupby(["model", "n", "ordering"]).size().rename("markets").reset_index()
    summary = {
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "run_ids": run_ids,
                 "git_commit": _git(), "models": sorted(t.markets["model"].unique()), "n": [int(n) for n in ns],
                 "orderings": sorted(t.markets["ordering"].unique()), "markets_per_cell": records(counts),
                 "part_lengths": sorted({(int(a), int(c)) for a, c in zip(t.markets["part1_len"], t.markets["part2_len"])}),
                 "paper_lengths_all": bool(t.markets["paper_lengths"].all()), "windows": measures.WINDOWS,
                 "excluded_cells": excluded_rows},
        "definitions": {"measures": b.__doc__, "chat_coding": chat_mod.__doc__, "mistakes": mistakes.__doc__,
                        "tests": stats.__doc__, "probe": probe_mod.__doc__},
        "theory": measures.theory(ns),
        "cells": records(cells), "human_comparison": records(comp), "extensions": records(tables["extensions"]),
        "table2": records(t2), "markov": mk_summ, "strategy": records(strat), "tests": records(tests),
        "hysteresis": records(hyst), "chat": records(chat_cells), "mistakes": records(mis),
        "mistakes_cells": records(mis_cells),  # format slips (re-asks) apart from strategic mistakes
        "timeseries": records(ts),
        "probe": records(probe_cells),  # deviation answers are in probe_answers.csv, for reading
        "caveats": {"human_data": list(b.CAVEATS), "analysis": LIMITATIONS},
    }
    tables["_examples_chat"] = chat_examples(msgs, periods, t.chat, t.firm_period)
    tables["_examples_mistakes"] = mistakes.examples(ff, fc)
    return tables, summary


def write(out_dir: str | Path, tables: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, obj in tables.items():
        if name.startswith("_"):
            path = out / f"{name[1:]}.json"
            path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
        else:
            path = out / f"{name}.csv"
            obj.to_csv(path, index=False)
        files.append(path.name)
    summary["files"] = sorted(files + ["summary.json"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return summary["files"]


def _git() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or None
