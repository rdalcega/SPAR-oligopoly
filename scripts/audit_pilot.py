"""Read-only audit of the pilot1 / pilot1-q14 markets and call logs.

    uv run python scripts/audit_pilot.py            # writes data/audits/pilot1_audit.json, prints a summary

Checks engine/prompt invariants against the recorded markets and call logs (A), extracts per-model behaviour
(B), and token/cost statistics (C). Never modifies run data.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oligopoly import market as mk  # noqa: E402
from oligopoly.analysis.chat import code_message, code_period  # noqa: E402
from oligopoly.prompts import format_ecu  # noqa: E402

RUNS = [ROOT / "data/runs/pilot1", ROOT / "data/runs/pilot1-q14"]
OUT = ROOT / "data/audits/pilot1_audit.json"


def content_text(m: dict) -> str:
    c = m["content"]
    if isinstance(c, list):  # cache_control blocks
        return "".join(b.get("text", "") for b in c)
    return c


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


# ---------------------------------------------------------------------------------------------- A: engine


def check_market(rec: dict) -> dict:
    issues: list[str] = []
    info: dict = {}
    letters = sorted(rec["histories"])
    orders = rec["settings"]["listing_orders"]
    lengths = rec["settings"]["experiment"]["part_lengths"]
    periods = rec["periods"]
    talk = rec["settings"]["talk"]

    # numbering
    exp_seq = [(p, t) for p in (1, 2) for t in range(1, lengths[p - 1] + 1)]
    got_seq = [(x["part"], x["period"]) for x in periods]
    if got_seq != exp_seq:
        issues.append(f"period sequence {got_seq} != {exp_seq}")

    # payoffs and cumulative totals
    cum = {L: [0.0, 0.0] for L in letters}
    for x in periods:
        pay = mk.payoffs(x["prices"])
        if any(abs(pay[L] - x["profits"][L]) > 1e-9 for L in letters):
            issues.append(f"payoff mismatch p{x['part']} t{x['period']}: {x['profits']} vs {pay}")
        for L in letters:
            cum[L][x["part"] - 1] += x["profits"][L]
        if x["talk"] != talk[f"part{x['part']}"]:
            issues.append(f"talk flag wrong p{x['part']} t{x['period']}")
        if not x["talk"] and x["chat"]:
            issues.append(f"chat in NoTalk period p{x['part']} t{x['period']}")
        # chat structure
        rounds = x["chat"]
        for i, r in enumerate(rounds):
            if not r["messages"] and i != len(rounds) - 1:
                issues.append(f"chat continued after an all-pass round p{x['part']} t{x['period']} r{r['round']}")
            for L, msg in r["messages"].items():
                if len(msg) > 300:
                    issues.append(f"message over 300 chars p{x['part']} t{x['period']} {L}")
        if x["talk"] and len(rounds) < 3 and rounds and rounds[-1]["messages"]:
            issues.append(f"chat ended early without an all-pass round p{x['part']} t{x['period']}")
        # fallback = previous own price
    for L in letters:
        t = rec["totals"][L]
        if abs(t["profit_part1"] - cum[L][0]) > 1e-6 or abs(t["profit_part2"] - cum[L][1]) > 1e-6:
            issues.append(f"totals mismatch {L}: {t} vs {cum[L]}")
    prev = {L: 100 for L in letters}
    for x in periods:
        for L in letters:
            d = x["decisions"][L]
            if d["fallback"] and x["prices"][L] != prev[L]:
                issues.append(f"fallback price {x['prices'][L]} != previous {prev[L]} ({L} p{x['part']} t{x['period']})")
            prev[L] = x["prices"][L]

    # histories as the firm sees them
    all_notes = defaultdict(set)  # firm -> its notes strings
    for x in periods:
        for L, d in x["decisions"].items():
            if d["notes"]:
                all_notes[L].add(d["notes"])
        for r in x["chat"]:
            for L, d in r["decisions"].items():
                if d["notes"]:
                    all_notes[L].add(d["notes"])
    for L in letters:
        h = rec["histories"][L]
        roles = [m["role"] for m in h]
        if roles[0] != "system" or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
            issues.append(f"{L}: role alternation broken")
        sysmsg = h[0]["content"]
        if "Part 2" in sysmsg:
            issues.append(f"{L}: system prompt mentions Part 2")
        users = [m["content"] for m in h if m["role"] == "user"]
        full_user = "\n".join(users)
        p2i = next(i for i, u in enumerate(users) if "Instructions – Part 2" in u)
        part1_text = "\n".join([sysmsg, *users[:p2i]]) + users[p2i].split("Part 1 is over.")[0]
        if re.search(r"Part 2|second part", part1_text):
            issues.append(f"{L}: Part 1 text mentions Part 2")
        p2_instr = users[p2i]
        if talk["part1"] and not talk["part2"]:
            if "will not be able to communicate" not in p2_instr:
                issues.append(f"{L}: Part 2 NoTalk-after-Talk sentence missing")
            if "chat box, which you can use" in p2_instr:
                issues.append(f"{L}: Part 2 describes a chat box")
        after = "\n".join(users[p2i:])
        if talk["part2"] is False and re.search(r": chat, round|chat box has closed|Messages posted", after.split("Instructions – Part 2")[1]):
            issues.append(f"{L}: chat prompt in Part 2 NoTalk")
        # die lines: only at the end of each part (pilot: min_periods = part length)
        dies = re.findall(r"Period (\d+) is over\. The computer threw the virtual dice and it showed (\d)", full_user)
        exp_dies = [(str(lengths[0]), "6"), (str(lengths[1]), "6")]
        if dies != exp_dies:
            issues.append(f"{L}: die lines {dies} != {exp_dies}")
        # prompt period headers
        heads = re.findall(r"^Part (\d), period (\d+)(?::|\.)", full_user, re.M)
        info.setdefault("headers", {})[L] = len(heads)
        # results texts: prices/profits/cumulative and listing order
        res = re.findall(r"Results of period (\d+) \(Part (\d)\):\n((?:Firm .*\n?)+)(Your total profit in this part so far: ([\d,\.]+) ECU)?(?:\nYour total profit in the experiment so far: ([\d,\.]+) ECU)?", full_user)
        if len(res) != len(periods):
            issues.append(f"{L}: {len(res)} results texts for {len(periods)} periods")
        run = [0.0, 0.0]
        for (t, p, block, _, cp, ct), x in zip(res, periods):
            if (int(p), int(t)) != (x["part"], x["period"]):
                issues.append(f"{L}: results header p{p} t{t} out of order")
            rows = re.findall(r"Firm (\w+)( \(you\))?: price (\d+), profit ([\d,\.]+) ECU", block)
            if [r[0] for r in rows] != orders[L]:
                issues.append(f"{L}: listing order {[r[0] for r in rows]} != {orders[L]}")
            for F, you, pr, pf in rows:
                if bool(you) != (F == L):
                    issues.append(f"{L}: (you) marker wrong")
                if int(pr) != x["prices"][F] or pf != format_ecu(x["profits"][F]):
                    issues.append(f"{L}: results text wrong for {F} p{p} t{t}")
            run[x["part"] - 1] += x["profits"][L]
            if cp != format_ecu(run[x["part"] - 1]):
                issues.append(f"{L}: cumulative part profit shown {cp} != {format_ecu(run[x['part'] - 1])} p{p} t{t}")
            if x["part"] == 2 and ct != format_ecu(sum(run)):
                issues.append(f"{L}: cumulative total shown {ct} != {format_ecu(sum(run))}")
            if x["part"] == 1 and ct:
                issues.append(f"{L}: experiment total shown in Part 1")
        final = re.search(r"The sum of your firm's profits during the experiment is ([\d,\.]+) ECU", users[-1])
        if not final or final.group(1) != format_ecu(sum(run)):
            issues.append(f"{L}: final total line wrong/missing")
        # other firms' notes never in this firm's history
        for F in letters:
            if F == L:
                continue
            for nt in all_notes[F]:
                if len(nt) > 30 and nt in "\n".join(m["content"] for m in h):
                    issues.append(f"{L}: sees notes of {F}")
                    break
        # round-message blocks show exactly the previous round's posted messages, in L's listing order
        for x in periods:
            for r in x["chat"]:
                exp_lines = [f"Firm {F}{' (you)' if F == L else ''}: {r['messages'][F]}" for F in orders[L] if F in r["messages"]]
                block = (f"Messages posted in round {r['round']}:\n" + "\n".join(exp_lines)) if exp_lines else f"No firm posted a message in round {r['round']}."
                if block not in full_user:
                    issues.append(f"{L}: round block p{x['part']} t{x['period']} r{r['round']} not shown verbatim")
    return {"issues": issues, "info": info}


def check_calls(rec: dict, calls: list[dict]) -> dict:
    """Request snapshots must be prefixes of the final history, and never contain same-round/same-period
    information of other firms."""
    issues: list[str] = []
    key = rec["spec"]["run_id"], None
    hist = {L: [(m["role"], m["content"]) for m in h] for L, h in rec["histories"].items()}
    per = {(x["part"], x["period"]): x for x in rec["periods"]}
    posted = {(x["part"], x["period"], r["round"]): r["messages"] for x in rec["periods"] for r in x["chat"]}
    n_ok = 0
    for c in calls:
        if c.get("error") is not None:
            continue
        n_ok += 1
        tg = c["tag"]
        L = tg["firm"]
        req = [(m["role"], content_text(m)) for m in c["request"]["messages"]]
        if req != hist[L][:len(req)]:
            issues.append(f"request not a prefix of history: {tg['call_id']}")
            continue
        if req[-1][0] != "user":
            issues.append(f"request does not end with a user turn: {tg['call_id']}")
        text = "\n".join(t for _, t in req)
        x = per[(tg["part"], tg["period"])]
        others = [F for F in hist if F != L]
        # same-period results never visible before deciding
        if f"Results of period {tg['period']} (Part {tg['part']})" in text:
            issues.append(f"same-period results visible: {tg['call_id']}")
        if tg["phase"].startswith("chat"):
            here = (tg["part"], tg["period"], int(tg["phase"][4:]))
        else:
            here = (tg["part"], tg["period"], 99)  # price: every round of this period is visible
        earlier = {m for k, ms in posted.items() if k < here for m in ms.values()}
        for k, ms in posted.items():
            if k >= here:
                for F, m in ms.items():
                    if F in others and len(m) > 25 and m not in earlier and m in text:
                        issues.append(f"future/same-round message of {F} visible: {tg['call_id']}")
        for xx in per.values():
            for F, d in xx["decisions"].items():
                if F in others and d["notes"] and len(d["notes"]) > 30 and d["notes"] in text:
                    issues.append(f"price notes of {F} visible: {tg['call_id']}")
    return {"issues": issues, "ok_calls": n_ok}


# ---------------------------------------------------------------------------------------------- B: behaviour


def behaviour(rec: dict) -> dict:
    letters = sorted(rec["histories"])
    out: dict = {"prices": [], "chat": [], "agreements": []}
    n_msg = n_slot = 0
    lens = []
    reasks = Counter()
    fallbacks = Counter()
    warns = Counter()
    errors = Counter()
    prev_agreed = None
    for x in rec["periods"]:
        out["prices"].append({"part": x["part"], "period": x["period"],
                              "prices": "/".join(str(x["prices"][L]) for L in letters)})
        decs = [("price", d) for d in x["decisions"].values()]
        msgs = []
        for r in x["chat"]:
            for L, d in r["decisions"].items():
                decs.append(("chat", d))
                n_slot += 1
                if d["value"] is not None:
                    n_msg += 1
                    lens.append(len(d["value"]))
                    cm = code_message(d["value"])
                    msgs.append({"round": r["round"], "firm": L, **cm})
        for kind, d in decs:
            reasks[kind] += d["reasks"]
            fallbacks[kind] += d["fallback"]
            for w in d["warnings"]:
                warns[re.sub(r"\d+", "#", w)] += 1
            for e in d["errors"]:
                errors[re.sub(r"\d+", "#", e)] += 1
        if x["talk"]:
            cp = code_period(sorted(msgs, key=lambda m: m["round"]), x["prices"], prev_agreed)
            prev_agreed = cp["agreed_price"] if cp["agreement"] else prev_agreed
            out["agreements"].append({"part": x["part"], "period": x["period"], "agreed": cp["agreed_price"],
                                      "support": cp["support"], "kept": cp["kept"],
                                      "undercutters": cp["undercutters"], "above": cp["above"],
                                      "rounds": len(x["chat"])})
    out["pass_rate"] = 1 - n_msg / n_slot if n_slot else None
    out["msg_len_mean"] = mean(lens)
    out["msg_len_max"] = max(lens) if lens else None
    out["msg_over_280"] = sum(v >= 280 for v in lens)
    out["reasks"] = dict(reasks)
    out["fallbacks"] = dict(fallbacks)
    out["warnings"] = dict(warns)
    out["errors"] = dict(errors)
    out["profits"] = {L: rec["totals"][L]["profit_total"] for L in letters}
    return out


# ---------------------------------------------------------------------------------------------- C: tokens


def tokens(calls: list[dict]) -> dict:
    ok = [c for c in calls if c.get("error") is None]
    errs = [c for c in calls if c.get("error") is not None]
    by_period = defaultdict(list)
    phase = defaultdict(lambda: defaultdict(list))
    ratio = []
    lat = []
    cost = []
    cached = 0
    seed_flags = Counter()
    providers = Counter()
    models_served = Counter()
    finish = Counter()
    reasoning_present = 0
    for c in ok:
        u = c["usage"]
        tg = c["tag"]
        ph = "chat" if tg["phase"].startswith("chat") else "price"
        if tg["attempt"] == 0:
            by_period[(tg["part"], tg["period"], tg["phase"])].append(u["prompt_tokens"])
        phase[ph]["completion"].append(u["completion_tokens"])
        phase[ph]["reasoning"].append(u["reasoning_tokens"])
        phase[ph]["cost"].append(c["cost_usd"] or 0)
        b = sum(len(content_text(m).encode()) for m in c["request"]["messages"])
        ratio.append(u["prompt_tokens"] / b)
        lat.append(c["latency_s"])
        cost.append(c["cost_usd"])
        cached += u["cached_tokens"]
        seed_flags[(c["seed_possibly_ignored"], c["seed_flag_reason"])] += 1
        providers[(c["provider_requested"], c["provider_served"], c["provider_mismatch"])] += 1
        models_served[c["model_served"]] += 1
        finish[(c["finish_reason"], c.get("native_finish_reason"))] += 1
        reasoning_present += bool(c.get("reasoning") or c.get("reasoning_details"))
    # growth: mean prompt tokens of the round-1/price prompt per period
    mp = {k: mean(v) for k, v in by_period.items()}

    def first_phase(part, t):
        return mp.get((part, t, "chat1")) or mp.get((part, t, "price"))

    p1 = [first_phase(1, t) for t in range(1, 7)]
    p2 = [first_phase(2, t) for t in range(1, 5)]
    g1 = [b - a for a, b in zip(p1, p1[1:]) if a and b]
    g2 = [b - a for a, b in zip(p2, p2[1:]) if a and b]
    return {
        "calls_ok": len(ok), "calls_err": len(errs),
        "err_kinds": dict(Counter(re.sub(r"\d{3,}", "#", str(c["error"]))[:80] for c in errs)),
        "timeout_retries": sum(bool(c.get("timeout_retry")) for c in calls),
        "prompt_by_period_first_call": {"part1": p1, "part2": p2},
        "price_prompt_by_period": {f"p{p}t{t}": mp.get((p, t, "price")) for p in (1, 2) for t in range(1, 7) if mp.get((p, t, "price"))},
        "growth_talk_per_period": mean(g1), "growth_notalk_per_period": mean(g2),
        "completion_mean": {k: mean(v["completion"]) for k, v in phase.items()},
        "reasoning_mean": {k: mean(v["reasoning"]) for k, v in phase.items()},
        "cost_per_call": {k: mean(v["cost"]) for k, v in phase.items()},
        "cost_total": sum(x or 0 for x in cost), "cost_missing": sum(x is None for x in cost),
        "cached_tokens": cached,
        "latency_p50": pct(lat, .5), "latency_p90": pct(lat, .9), "latency_max": max(lat) if lat else None,
        "tok_per_byte": mean(ratio), "tok_per_byte_min": min(ratio) if ratio else None,
        "tok_per_byte_max": max(ratio) if ratio else None,
        "prompt_tokens_max": max((c["usage"]["prompt_tokens"] for c in ok), default=None),
        "seed_flags": {str(k): v for k, v in seed_flags.items()},
        "providers": {str(k): v for k, v in providers.items()},
        "models_served": dict(models_served),
        "finish": {str(k): v for k, v in finish.items()},
        "reasoning_returned_calls": reasoning_present,
    }


def main() -> int:
    calls_by_market = defaultdict(list)
    for run in RUNS:
        for line in (run / "calls.jsonl").open():
            c = json.loads(line)
            calls_by_market[c["tag"]["market"]].append(c)
    report = {}
    for run in RUNS:
        for f in sorted((run / "markets").glob("*.json")):
            if f.name.endswith(".failed.json"):
                rec = json.loads(f.read_text())
                key = rec["spec"]["model"]
                report[key] = {"failed": True, "error": rec["error"][:300],
                               "periods_completed": rec["periods_completed"],
                               "tokens": tokens(calls_by_market[f.stem.removesuffix(".failed")])}
                continue
            rec = json.loads(f.read_text())
            mkey = f.stem
            calls = calls_by_market[mkey]
            report[rec["spec"]["model"]] = {
                "market": mkey,
                "engine": check_market(rec),
                "calls": check_calls(rec, calls),
                "usage_record": rec["usage"],
                "behaviour": behaviour(rec),
                "tokens": tokens(calls),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    for m, r in report.items():
        print("=" * 30, m)
        if r.get("failed"):
            print("FAILED", r["error"][:200], r["tokens"]["calls_ok"], r["tokens"]["calls_err"])
            continue
        print("engine issues:", r["engine"]["issues"][:20])
        print("call issues:", r["calls"]["issues"][:10], "ok calls", r["calls"]["ok_calls"], "usage calls", r["usage_record"]["calls"])
        b = r["behaviour"]
        print("prices:", " | ".join(f"{p['part']}.{p['period']}:{p['prices']}" for p in b["prices"]))
        print("agreements:", [(a["period"], a["agreed"], a["support"], a["kept"], a["undercutters"], a["rounds"]) for a in b["agreements"]])
        print({k: b[k] for k in ("pass_rate", "msg_len_mean", "msg_len_max", "msg_over_280", "reasks", "fallbacks", "warnings", "errors", "profits")})
        t = r["tokens"]
        print({k: v for k, v in t.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
