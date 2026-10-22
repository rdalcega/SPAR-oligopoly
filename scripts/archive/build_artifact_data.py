"""Build the compact datasets behind the 2026-10-20 sweep dashboard (results/2026-10-20-sweep/dashboard/).

Archived and frozen: the dashboard is a record of the 2026-10-20 sweep (gpt-oss-20b and the incomplete Qwen3-14B
sweep, n up to 10, and the ab-w3/ab-w60 memory A/B), kept for provenance. Experiments are paused, so it is not
meant to be re-run; a rerun would also fold every later run's budget into the "pilots and probes" cost row.

Reads only: data/analysis/<sweep>/ outputs, data/runs/*/budget.json, the A/B pilot market JSONs in
data/runs/ab-w3 and data/runs/ab-w60 (all untracked, present only in the main checkout's data/; tracked copies of
the analysis are in results/2026-10-20-sweep/), the probe budgets in results/2026-10-20-probes/, and the
hand-maintained inputs in results/2026-10-20-sweep/dashboard/manual/ (findings_edits.json, screen_checks.json).
Never writes outside results/2026-10-20-sweep/dashboard/.

    uv run python scripts/archive/build_artifact_data.py      # from the main checkout

Per-cell datasets keep only cells with all six markets complete; run_status lists the partial cells (not imputed).

Outputs (all small): cells.csv, timeseries.csv, probe.csv, chat.csv, mistakes.csv, mistakes_model.csv,
findings.json, ab_memory.csv, run_status.csv, screen_checks.csv, build_meta.json.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")  # untracked: data/analysis and data/runs exist only in the main checkout
OUT = os.path.join(ROOT, "results", "2026-10-20-sweep", "dashboard")
MANUAL = os.path.join(OUT, "manual")
# The probe budgets that were in data/probes/ when the dashboard was built (structured/ was a subfolder, not counted)
PROBE_BUDGETS = [os.path.join(ROOT, "results", "2026-10-20-probes", d, "*budget*.json")
                 for d in ("provider", "contamination")]

SWEEPS = {"gpt-oss-20b": "sweep-gptoss", "qwen3-14b": "sweep-qwen14b"}
N_VALUES = [2, 4, 6, 8, 10]
ORDERINGS = ["NT-T", "T-NT", "NT-NT"]
MARKETS_PER_CELL = 6
# Cells compared directly with the human results (DECISIONS.md, Scope).
HUMAN_CELLS = {(n, "NT-T") for n in (2, 4, 6, 8)} | {(4, "T-NT"), (4, "NT-NT")}
CELL_MEASURES = ["sp_p1_from6", "sp_p1_all", "sp_p2_all", "sp_p2_from6", "gain_p2_all", "gain_p2_from6"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def rnd(x, d=2):
    v = num(x)
    return None if v is None else round(v, d)


def pct(x, d=1):
    v = num(x)
    return None if v is None else round(100 * v, d)


def write_csv(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def cid(*parts):
    return "|".join(str(p) for p in parts)


def analysis(model, name):
    return read_csv(os.path.join(DATA, "analysis", SWEEPS[model], name))


_COUNTS = {}


def market_counts(model):
    """Completed markets per (n, ordering), from the analysis's markets.csv."""
    if model not in _COUNTS:
        per = defaultdict(int)
        for r in analysis(model, "markets.csv"):
            per[(int(r["n"]), r["ordering"])] += 1
        _COUNTS[model] = per
    return _COUNTS[model]


def complete(model, r):
    """Per-cell datasets keep only cells with all markets done; partial cells are reported in run_status."""
    return market_counts(model).get((int(r["n"]), r["ordering"]), 0) >= MARKETS_PER_CELL


# ---------------------------------------------------------------- cells (+ human benchmarks)

def build_cells():
    rows = []
    human = {}
    for model in SWEEPS:
        for r in analysis(model, "cells.csv"):
            if not complete(model, r):
                continue
            if r["measure"] not in CELL_MEASURES:
                continue
            n, o = int(r["n"]), r["ordering"]
            rows.append(dict(cell_id=cid(model, n, o, r["measure"]), model=model, n=n, ordering=o,
                             measure=r["measure"], mean=rnd(r["mean"]), sd=rnd(r["sd"]),
                             n_markets=int(float(r["n_markets"])), human_comparable=(n, o) in HUMAN_CELLS))
        # Human benchmarks, mapped onto the same measure names.
        for r in analysis(model, "human_comparison.csv"):
            if r["block"] not in ("price", "controls") or num(r["human"]) is None:
                continue
            m, w, o = r["measure"], r["window"], r["ordering"]
            if m.startswith("NoTalk selling price (part 1)") or m == "NoTalk selling price" or m.startswith("Talk selling price (part 1)"):
                meas = "sp_p1_from6" if w == "p1_from6" else None
            elif m == "Talk selling price" or m.startswith("NoTalk selling price (part 2)"):
                meas = {"p2_all": "sp_p2_all", "p2_from6": "sp_p2_from6"}.get(w)
            elif m == "Gain from talking":
                meas = {"p2_all": "gain_p2_all", "p2_from6": "gain_p2_from6"}.get(w)
            else:
                meas = None
            if meas:
                key = (int(r["n"]), o, meas)
                human[key] = dict(cell_id=cid("humans", key[0], o, meas), model="humans", n=key[0], ordering=o,
                                  measure=meas, mean=rnd(r["human"]), sd=rnd(r["human_sd"]),
                                  n_markets=MARKETS_PER_CELL, human_comparable=True)
    rows += [human[k] for k in sorted(human)]
    write_csv("cells.csv", rows, ["cell_id", "model", "n", "ordering", "measure", "mean", "sd", "n_markets",
                                  "human_comparable"])
    return rows


# ---------------------------------------------------------------- time series

def build_timeseries():
    rows = []
    for model in SWEEPS:
        for r in analysis(model, "timeseries.csv"):
            if not complete(model, r):
                continue
            rows.append(dict(series=cid(model, r["n"], r["ordering"]), model=model, n=int(r["n"]),
                             ordering=r["ordering"], gperiod=int(r["gperiod"]), part=int(r["part"]),
                             period=int(r["period"]), mean=rnd(r["mean"], 1), q25=rnd(r["q25"], 1),
                             q75=rnd(r["q75"], 1), n_markets=int(float(r["n_markets"]))))
    write_csv("timeseries.csv", rows, ["series", "model", "n", "ordering", "gperiod", "part", "period", "mean",
                                       "q25", "q75", "n_markets"])


# ---------------------------------------------------------------- probe vs pricing

def build_probe():
    sp = {}
    for model in SWEEPS:
        for r in analysis(model, "cells.csv"):
            if not complete(model, r):
                continue
            sp[(model, int(r["n"]), r["ordering"], r["measure"])] = rnd(r["mean"], 1)
    rows = []
    for model in SWEEPS:
        for r in analysis(model, "probe_cells.csv"):
            if not complete(model, r):
                continue
            n, o = int(r["n"]), r["ordering"]
            talk = {"NT-T": sp.get((model, n, o, "sp_p2_all")), "T-NT": sp.get((model, n, o, "sp_p1_all"))}.get(o)
            rows.append(dict(cell_id=cid(model, n, o), model=model, n=n, ordering=o, firms=int(float(r["firms"])),
                             best_common_pct=pct(r["best_common_correct_share"]),
                             max_price_pct=pct(r["max_price_correct_share"]),
                             recall_pct=pct(r["last_period_recall_mean"]),
                             knows_not_collude_pct=pct(r["knows_not_collude_share"]),
                             talk_sp=talk, valid_pct=pct(r["valid_share"])))
    write_csv("probe.csv", rows, ["cell_id", "model", "n", "ordering", "firms", "best_common_pct", "max_price_pct",
                                  "recall_pct", "knows_not_collude_pct", "talk_sp", "valid_pct"])


# ---------------------------------------------------------------- chat screen

def build_chat():
    rows = []
    for model in SWEEPS:
        for r in analysis(model, "chat_cells.csv"):
            if not complete(model, r):
                continue
            n, o = int(r["n"]), r["ordering"]
            ag, kept = int(float(r["agreements"])), int(float(r["kept"]))
            rows.append(dict(cell_id=cid(model, n, o), model=model, n=n, ordering=o, talk_part=int(r["part"]),
                             talk_periods=int(float(r["talk_periods"])), agreements=ag,
                             agreed_100=int(float(r["agreed_100"])), kept=kept, broken=ag - kept,
                             broken_undercut=int(float(r["breaches_undercut"])),
                             broken_above_only=int(float(r["breaches_above_only"])),
                             threat_pct=pct(r["share_threat"]), punished_next_pct=pct(r["share_punished_next"]),
                             deception_flagged=int(float(r["periods_deception_notes"])),
                             deception_priced=int(float(r["periods_deception_priced"]))))
    write_csv("chat.csv", rows, ["cell_id", "model", "n", "ordering", "talk_part", "talk_periods", "agreements",
                                 "agreed_100", "kept", "broken", "broken_undercut", "broken_above_only",
                                 "threat_pct", "punished_next_pct", "deception_flagged", "deception_priced"])


# ---------------------------------------------------------------- mistakes

def build_mistakes():
    rows, model_rows = [], []
    for model in SWEEPS:
        zero = defaultdict(lambda: [0, 0])
        for r in analysis(model, "firm_period.csv"):
            k = (int(r["n"]), r["ordering"])
            zero[k][1] += 1
            zero[k][0] += r["price"] == "0"
        for r in analysis(model, "mistakes_cells.csv"):
            if not complete(model, r):
                continue
            n, o = int(r["n"]), r["ordering"]
            z, tot = zero.get((n, o), [0, 0])
            rows.append(dict(cell_id=cid(model, n, o), model=model, n=n, ordering=o,
                             price_decisions=int(float(r["price_decisions"])),
                             price_reask_pct=pct(r["price_reask_rate"], 2), chat_reask_pct=pct(r["chat_reask_rate"], 2),
                             price_fallback_pct=pct(r["price_fallback"], 2), price_zero_pct=pct(z / tot if tot else None, 1),
                             misread_notes_pct=pct(r["misread_notes"], 2), misread_chat_pct=pct(r["misread_chat"], 2),
                             notes_contradict_pct=pct(r["notes_contradict"], 2),
                             low_price_no_context_pct=pct(r["low_price_no_context"], 2)))
        for r in analysis(model, "mistakes_by_model.csv"):
            mis = sum(num(r[k]) or 0 for k in ("notes_misread_highest_wins", "notes_misread_tie_full", "notes_misread_horizon"))
            model_rows.append(dict(model=model, price_decisions=int(float(r["price_decisions"])),
                                   chat_decisions=int(float(r["chat_decisions"])),
                                   price_zero_pct=pct(r["price_zero"], 1),
                                   price_invalid_pct=pct(r["price_any_invalid"], 2),
                                   chat_invalid_pct=pct(r["chat_any_invalid"], 1),
                                   price_fallback_pct=pct(r["price_fallback"], 3),
                                   chat_fallback_pct=pct(r["chat_fallback"], 3),
                                   misread_notes_pct=pct(mis, 2),
                                   notes_contradict_pct=pct(r["notes_contradict"], 2)))
    write_csv("mistakes.csv", rows, ["cell_id", "model", "n", "ordering", "price_decisions", "price_reask_pct",
                                     "chat_reask_pct", "price_fallback_pct", "price_zero_pct", "misread_notes_pct",
                                     "misread_chat_pct", "notes_contradict_pct", "low_price_no_context_pct"])
    write_csv("mistakes_model.csv", model_rows, ["model", "price_decisions", "chat_decisions", "price_zero_pct",
                                                 "price_invalid_pct", "chat_invalid_pct", "price_fallback_pct",
                                                 "chat_fallback_pct", "misread_notes_pct", "notes_contradict_pct"])


# ---------------------------------------------------------------- verified findings

def short_market(m):
    # gpt-oss-20b__n4__NT-T__m4 -> n4 NT-T m4
    parts = m.split("__")
    return " ".join(parts[1:]) if len(parts) >= 4 else m


def build_findings():
    with open(os.path.join(MANUAL, "findings_edits.json")) as f:
        edits = json.load(f)
    rows, unedited = [], []
    for model, sweep in SWEEPS.items():
        path = os.path.join(DATA, "analysis", sweep, "reading_workflow.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            wf = json.load(f)
        for g in wf.get("result", []):
            verdicts = {v["id"]: v for v in g.get("verdicts", [])}
            for c in g.get("claims", []):
                key = f"{model}/{g['group']}/{c['id']}"
                v = verdicts.get(c["id"], {})
                verdict = v.get("verdict", "unverified")
                e = edits.get(key, {})
                if verdict == "refuted" or e.get("drop"):
                    continue
                if verdict == "partly" and "claim" not in e:
                    unedited.append(key)
                ev = c.get("evidence") or [{}]
                q = ev[min(e.get("ev", 0), len(ev) - 1)]
                rows.append(dict(fid=key, model=model, group=g["group"], theme=e.get("theme", "other"),
                                 category=c.get("category", ""), verdict=verdict,
                                 claim=e.get("claim", c["claim"]), edited="claim" in e,
                                 verifier_note=v.get("note", ""), quote=q.get("quote", ""),
                                 market=q.get("market", ""), market_short=short_market(q.get("market", "")),
                                 period=str(q.get("period", ""))))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)
    return unedited


# ---------------------------------------------------------------- memory-window A/B pilot

def build_ab():
    rows = []
    for arm, run in (("window 3", "ab-w3"), ("window 60", "ab-w60")):
        d = os.path.join(DATA, "runs", run, "markets")
        keys = sorted({os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(d, "*.json"))})
        for k in keys:
            model, _, _, m = k.split("__")
            full = os.path.join(d, k + ".json")
            if os.path.exists(full):
                with open(full) as f:
                    rec = json.load(f)
                p2 = [min(p["prices"].values()) for p in rec["periods"] if p["part"] == 2]
                rows.append(dict(ab_id=cid(run, k), arm=arm, model=model, market=m, status="complete",
                                 talk_sp=round(sum(p2) / len(p2), 1) if p2 else None,
                                 final_sp=p2[-1] if p2 else None, periods_done=len(rec["periods"])))
            else:
                done, partial = None, None
                ck = os.path.join(d, k + ".ckpt.json")
                if os.path.exists(ck):
                    with open(ck) as f:
                        ckpt = json.load(f)
                    done = ckpt.get("periods_completed")
                    p2 = [min(p["prices"].values()) for p in ckpt.get("periods", []) if p.get("part") == 2]
                    partial = round(sum(p2) / len(p2), 1) if p2 else None
                reason = ""
                fp = os.path.join(d, k + ".failed.json")
                if os.path.exists(fp):
                    with open(fp) as f:
                        reason = json.load(f).get("reason", "")
                rows.append(dict(ab_id=cid(run, k), arm=arm, model=model, market=m,
                                 status="stopped: context guard" if reason == "context_overflow" else f"stopped: {reason}",
                                 talk_sp=None, talk_sp_partial=partial, periods_done=done))
    write_csv("ab_memory.csv", rows, ["ab_id", "arm", "model", "market", "status", "talk_sp", "final_sp",
                                      "talk_sp_partial", "periods_done"])


# ---------------------------------------------------------------- run status and cost

def spent(path):
    try:
        with open(path) as f:
            return float(json.load(f).get("spent_usd", 0))
    except (OSError, ValueError):
        return 0.0


def build_run_status():
    rows = []
    for model, sweep in SWEEPS.items():
        markets = analysis(model, "markets.csv")
        per = defaultdict(int)
        for r in markets:
            per[(int(r["n"]), r["ordering"])] += 1
        complete = {k for k, v in per.items() if v >= MARKETS_PER_CELL}
        partial = "; ".join(f"n{n} {o} {per.get((n, o), 0)}/{MARKETS_PER_CELL}" for n in N_VALUES for o in ORDERINGS
                            if (n, o) not in complete)
        b = os.path.join(DATA, "runs", sweep, "budget.json")
        cap = None
        try:
            with open(b) as f:
                cap = json.load(f).get("cap_usd")
        except OSError:
            pass
        rows.append(dict(model=model, cells_planned=len(N_VALUES) * len(ORDERINGS), cells_complete=len(complete),
                         human_cells_planned=len(HUMAN_CELLS), human_cells_complete=len(complete & HUMAN_CELLS),
                         markets_planned=len(N_VALUES) * len(ORDERINGS) * MARKETS_PER_CELL,
                         markets_complete=len(markets), cost_usd=round(spent(b), 2), cap_usd=cap,
                         cells_incomplete=partial or "none"))
    pilot = sum(spent(p) for p in glob.glob(os.path.join(DATA, "runs", "*", "budget.json"))
                if not os.path.basename(os.path.dirname(p)).startswith("sweep-"))
    pilot += sum(spent(p) for pattern in PROBE_BUDGETS for p in glob.glob(pattern))
    rows.append(dict(model="pilots and probes", cost_usd=round(pilot, 2)))
    write_csv("run_status.csv", rows, ["model", "cells_planned", "cells_complete", "human_cells_planned",
                                       "human_cells_complete", "markets_planned", "markets_complete", "cost_usd",
                                       "cap_usd", "cells_incomplete"])


def build_screen_checks():
    with open(os.path.join(MANUAL, "screen_checks.json")) as f:
        rows = json.load(f)["rows"]
    for r in rows:
        r["correct_pct"] = round(100 * r["correct"] / r["sample"])
        r["deliberate_pct"] = None if r["deliberate"] is None else round(100 * r["deliberate"] / r["sample"])
    write_csv("screen_checks.csv", rows, ["check", "label", "sample", "correct", "correct_pct", "deliberate",
                                          "deliberate_pct"])


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = build_cells()
    build_timeseries()
    build_probe()
    build_chat()
    build_mistakes()
    unedited = build_findings()
    build_ab()
    build_run_status()
    build_screen_checks()
    gen = {}
    for model, sweep in SWEEPS.items():
        try:
            with open(os.path.join(DATA, "analysis", sweep, "summary.json")) as f:
                gen[model] = json.load(f)["meta"]["generated_at"]
        except (OSError, KeyError):
            pass
    with open(os.path.join(OUT, "build_meta.json"), "w") as f:
        json.dump({"analysis_generated_at": gen, "partly_claims_without_rewrite": unedited}, f, indent=1)
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name)
        if os.path.isfile(p):
            print(f"{name:22s} {os.path.getsize(p):>8,d} bytes")
    if unedited:
        print("WARNING: 'partly' claims kept unedited (add rewrites to manual/findings_edits.json):", *unedited)


if __name__ == "__main__":
    main()
