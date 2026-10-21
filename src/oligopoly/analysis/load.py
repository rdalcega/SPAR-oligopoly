"""Load market records written by `oligopoly.engine` and flatten them into tidy tables."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .. import benchmarks as b
from ..config import ConfigError, ModelConfig
from ..engine import is_market_record_file, record_texts_digest_matches

PHASE = {True: "Talk", False: "NoTalk"}
KEY_COLS = ["run_id", "market_id", "model", "n", "ordering", "market_index"]


TRANSITION_COLS = ["run_id", "market_id", "model", "n", "ordering", "market_index", "part", "period", "firm",
                   "notes", "valid", "fallback", "reasks", "invalid_attempts", "coerced", "warnings", "errors"]


@dataclass
class Tables:
    firm_period: pd.DataFrame  # one row per firm x period
    market_period: pd.DataFrame  # one row per market x period
    chat: pd.DataFrame  # one row per firm x chat round (message or pass)
    markets: pd.DataFrame  # one row per market: design info and part lengths
    # one row per firm x market: the transition call between the parts (DECISIONS.md §6 item 3), if the run had it
    transition: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRANSITION_COLS))


def run_texts_digest(run_dir: Path) -> str | None:
    """The firm-facing wording digest recorded in the run's run_config.json (None for older runs)."""
    path = run_dir / "run_config.json"
    return json.loads(path.read_text()).get("texts_digest") if path.exists() else None


def load_records(run_ids: list[str], runs_root: str | Path,
                 skip_mismatched_prompts: bool = False) -> list[dict[str, Any]]:
    """All completed market records (`<key>.json`, not `.failed.json`) of the given runs.

    A record not played under the run's recorded prompt wording (`texts_digest` in run_config.json) raises
    ValueError, or with `skip_mismatched_prompts` is left out with a printed warning. Older records without
    their own `settings.texts_digest` are checked by recomputing their fingerprint, and get the run's digest
    when it matches."""
    recs = []
    for rid in run_ids:
        d = Path(runs_root) / rid / "markets"
        files = sorted(p for p in d.glob("*.json") if is_market_record_file(p))
        if not files:
            raise FileNotFoundError(f"no completed markets in {d}")
        digest = run_texts_digest(d.parent)
        off = []
        for p in files:
            rec = json.loads(p.read_text())
            rec["_market_id"] = f"{rid}/{p.stem}"
            if digest is not None:
                if not record_texts_digest_matches(rec, digest):
                    off.append(rec["_market_id"])
                    continue
                rec.setdefault("settings", {})["texts_digest"] = digest
            recs.append(rec)
        if off and not skip_mismatched_prompts:
            raise ValueError(f"{len(off)} market(s) of run {rid!r} were played under other prompt wording than "
                             f"its run_config.json records, e.g. {off[0]}; pass skip_mismatched_prompts "
                             f"(--skip-mismatched-prompts) to leave them out")
        for m in off:
            print(f"WARNING: skipped {m}: played under other prompt wording than run {rid!r} records")
    return recs


def num(x) -> float | None:
    """Float for JSON/CSV output; None for None, NaN or inf."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def model_identity(settings_model: dict[str, Any]) -> dict[str, Any]:
    """`ModelConfig.identity()` of a record's stored model settings (operational fields dropped, defaults
    filled in); fields this code does not know are kept, so they still separate variants."""
    known = set(ModelConfig.__dataclass_fields__)
    play = {k: v for k, v in settings_model.items() if k not in ModelConfig.OPERATIONAL}
    unknown = {k: v for k, v in play.items() if k not in known}
    try:
        identity = ModelConfig(**{k: v for k, v in play.items() if k in known}).identity()
    except (TypeError, ConfigError):  # missing required fields: compare as stored
        identity = {k: v for k, v in play.items() if k in known}
    return {**identity, **unknown}


def check_uniform_design(records: list[dict[str, Any]]) -> None:
    """Refuse to pool markets of one model played under different settings (experiment config, model
    identity, firm-facing prompt wording or answer-parsing rules), e.g. a pilot with a full run or a record left over from an
    earlier configuration."""
    seen: dict[str, dict[str, str]] = {}
    for rec in records:
        st = rec["settings"]
        key = json.dumps({"experiment": st["experiment"], "model": model_identity(st.get("model", {})),
                          "texts_digest": st.get("texts_digest"), "parser_version": st.get("parser_version")},
                         sort_keys=True, default=str)
        seen.setdefault(rec["spec"]["model"], {}).setdefault(key, rec["_market_id"])
    mixed = {m: list(v.values()) for m, v in seen.items() if len(v) > 1}
    if mixed:
        raise ValueError(f"markets of the same model use different settings (one example market per variant): {mixed}")


def _coerced(warnings: list[str]) -> bool:
    return any("coerced" in w for w in warnings)


def tidy(records: list[dict[str, Any]]) -> Tables:
    check_uniform_design(records)
    fp, mp, ch, mk, tr = [], [], [], [], []
    for rec in records:
        s = rec["spec"]
        lengths = rec["settings"]["experiment"]["part_lengths"]
        key = {"run_id": s["run_id"], "market_id": rec["_market_id"], "model": s["model"], "n": s["n"],
               "ordering": s["ordering"], "market_index": s["market_index"]}
        talk = rec["settings"]["talk"]
        mk.append(key | {"part1_len": lengths[0], "part2_len": lengths[1], "phase1": PHASE[talk["part1"]],
                         "phase2": PHASE[talk["part2"]], "pilot": rec["settings"].get("pilot", False),
                         "paper_lengths": tuple(lengths) == (b.PART1_PERIODS, b.PART2_PERIODS),
                         "transition": rec.get("transition") is not None,  # the call ran in this market
                         # an added Part 1 sentence (DECISIONS.md, prompt variants): never compared with humans
                         "prompt_variant": rec["settings"]["experiment"].get("prompt_variant", "baseline")})
        for L, d in (rec.get("transition") or {}).items():
            tr.append(key | {"part": 2, "period": 0, "firm": L, "notes": d["notes"] or "", "valid": d["valid"],
                             "fallback": d["fallback"], "reasks": d["reasks"], "invalid_attempts": len(d["errors"]),
                             "coerced": _coerced(d["warnings"]), "warnings": "; ".join(d["warnings"]),
                             "errors": " | ".join(e[:120] for e in d["errors"])})
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
    transition = (pd.DataFrame(tr).sort_values(["market_id", "firm"], ignore_index=True) if tr
                  else pd.DataFrame(columns=TRANSITION_COLS))
    return Tables(transition=transition,
                  firm_period=pd.DataFrame(fp).sort_values(sort + ["firm"], ignore_index=True),
                  market_period=pd.DataFrame(mp).sort_values(sort, ignore_index=True),
                  chat=chat.sort_values(sort + ["round", "firm"], ignore_index=True) if len(chat) else chat,
                  markets=pd.DataFrame(mk).sort_values("market_id", ignore_index=True))
