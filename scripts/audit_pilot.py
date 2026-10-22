"""Read-only audit of a run's markets and call logs.

    uv run python scripts/audit_pilot.py pilot3-n2 pilot3-n10 # any run ids under data/runs/
    uv run python scripts/audit_pilot.py --runs-dir DIR --out FILE RUN_ID ...

With no run ids it audits pilot1 and pilot1-q14, the first append-only pilots (a leftover default). data/runs/
is untracked and exists only in the main checkout; from a worktree pass --runs-dir. A run id with no directory
there is skipped silently, so check the report is not empty.

Writes data/audits/<run ids>_audit.json (untracked scratch; or --out, e.g. a results/<folder>/audits/ file) and
prints a summary. Checks engine/prompt invariants against the recorded markets and call logs (A), extracts
per-market behaviour (B), and token/cost statistics (C). Never modifies run data.

Records with windowed or summaries-layout context (`instructions` and `transcripts`, DECISIONS.md §6 item 3;
the summaries layout's transcript entries carry a `kind`): every
attempt's request is rebuilt from the record with `llm.reconstruct_request` and its messages_sha256 checked
against the attempt's context and against the call log's line for that call (where the log has one); the leak
checks (no other firm's notes; no same-round messages or same-period results of others) run on the rebuilt
requests. Older records (`histories`, append-only) keep the prefix check: every logged request must be a
prefix of the firm's final history.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oligopoly import market as mk  # noqa: E402
from oligopoly.engine import is_market_record_file  # noqa: E402
from oligopoly.llm import messages_sha256, reconstruct_from_log, reconstruct_request  # noqa: E402
from oligopoly.analysis.chat import code_message, code_period  # noqa: E402
from oligopoly.prompts import OWN_MESSAGE, format_ecu, round_text, shown_message, totals_text  # noqa: E402

DEFAULT_RUNS = ["pilot1", "pilot1-q14"]  # the first append-only pilots; pass run ids for anything else
RUNS_DIR = ROOT / "data/runs"
AUDITS_DIR = ROOT / "data/audits"  # untracked scratch; tracked audits are copied into results/<folder>/audits/


def content_text(m: dict) -> str:
    c = m["content"]
    if isinstance(c, list):  # cache_control blocks
        return "".join(b.get("text", "") for b in c)
    return c


def load_calls(path: Path) -> list[dict]:
    """The call log's records in file order; compact lines (`messages_ref`, or `messages_new` in older logs) get
    `request.messages` rebuilt by `reconstruct_from_log` (or `reconstruct_error` if the chain is broken)."""
    recs = [json.loads(line) for line in path.open() if line.strip()]
    for i, c in enumerate(recs):
        if "request" not in c and ("messages_new" in c or "messages_ref" in c):
            try:
                c["request"] = {"messages": reconstruct_from_log(recs, i)}
            except ValueError as e:
                c["reconstruct_error"] = str(e)
    return recs


def market_files(run: Path) -> list[Path]:
    """Completed and failed market records of a run (checkpoints are neither)."""
    return sorted(f for f in (run / "markets").glob("*.json")
                  if is_market_record_file(f) or f.name.endswith(".failed.json"))


def request_bytes(c: dict) -> int | None:
    if "prompt_bytes" in c:
        return c["prompt_bytes"]
    if "request" not in c:  # e.g. offline mock logs
        return None
    return sum(len(content_text(m).encode()) for m in c["request"]["messages"])


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


def letters_of(rec: dict) -> list[str]:
    return sorted(rec["totals"])


def firm_view(rec: dict, L: str) -> dict:
    """What firm L was shown, in one shape for both record formats: the system message, the user texts in order,
    the index of the first user text of Part 2, the Part 2 instructions, the Part 1 text (to check it never
    mentions Part 2) and every text the firm saw."""
    L1 = part_lengths(rec)[0]
    if "transcripts" in rec:  # windowed or summaries layout: instructions apart from the transcript
        instr, tr = rec["instructions"][L], rec["transcripts"][L]
        users = [m["content"] for m in tr if m["role"] == "user"]
        periods = [m["period"] for m in tr if m["role"] == "user"]
        p2i = next((i for i, g in enumerate(periods) if g > L1), len(users))
        p2_instr = instr[1] if len(instr) > 1 else ""
        if tr and "kind" in tr[0]:  # summaries layout: each prompt or re-ask is answered, and only those are;
            # the last entry may be the probe's unanswered prompt or re-ask (a probe skipped on context overflow)
            asks = [i for i, m in enumerate(tr) if m["kind"] in ("prompt", "reask")]
            answered = [i for i in asks if i + 1 < len(tr)]
            roles_ok = all(tr[i + 1]["kind"] == "answer" for i in answered) and \
                [i - 1 for i, m in enumerate(tr) if m["role"] == "assistant"] == answered and \
                (answered == asks or tr[-1]["kind"] in ("prompt", "reask") and tr[-1]["period"] > len(rec["periods"]))
        else:
            roles_ok = all(r == ("user", "assistant")[i % 2] for i, r in enumerate(m["role"] for m in tr))
        return {"system": instr[0], "users": users, "p2i": p2i, "p2_instr": p2_instr,
                "part1_text": "\n".join([instr[0], *users[:p2i]]), "roles": [m["role"] for m in tr],
                "roles_ok": roles_ok, "sections": bool(tr) and "kind" in tr[0],
                "chat_rounds": [m["content"] for m in tr if m.get("kind") == "chat"],
                "all_text": "\n".join([*instr, *(m["content"] for m in tr)])}
    h = rec["histories"][L]  # append-only history
    roles = [m["role"] for m in h]
    users = [m["content"] for m in h if m["role"] == "user"]
    p2i = next((i for i, u in enumerate(users) if "Instructions – Part 2" in u), len(users))
    part1_text = "\n".join([h[0]["content"], *users[:p2i]]) + (users[p2i].split("Part 1 is over.")[0]
                                                               if p2i < len(users) else "")
    return {"system": h[0]["content"], "users": users, "p2i": p2i, "p2_instr": users[p2i] if p2i < len(users) else "",
            "part1_text": part1_text, "roles": roles, "sections": False, "chat_rounds": [],
            "roles_ok": roles[0] == "system" and all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)),
            "all_text": "\n".join(m["content"] for m in h)}


def firm_notes(rec: dict) -> dict:
    """Every private notes text each firm wrote (chat and price decisions; the full text of truncated notes too),
    except notes the firm also posted in the chat (public, so not a leak when others see them)."""
    out = defaultdict(set)
    posted = defaultdict(list)
    for x in rec["periods"]:
        for r in x["chat"]:
            for L, m in r["messages"].items():
                posted[L].append(m)
        for d in [x["decisions"]] + [r["decisions"] for r in x["chat"]]:
            for L, dd in d.items():
                out[L] |= {t for t in (dd.get("notes"), dd.get("notes_full")) if t}
    for L, dd in (rec.get("transition") or {}).items():
        out[L] |= {t for t in (dd.get("notes"), dd.get("notes_full")) if t}
    return {L: {t for t in ts if not any(t in m for m in posted[L])} for L, ts in out.items()}


def expected_dies(rec: dict) -> list[tuple[str, str]]:
    """(period, face) of every die line the results show: periods with a throw in the pre-drawn schedules."""
    scheds = rec["settings"].get("die_schedules")
    if scheds is None:  # older records: pilots showed the die only after each part's final period
        return [(str(n), "6") for n in part_lengths(rec)]
    return [(str(t), str(d)) for sched in scheds for t, d in enumerate(sched, 1) if d is not None]


def chat_spec(rec: dict) -> tuple[int, int]:
    """(chat_rounds, char_limit) from the record's settings (pilot defaults 3 and 300 for records without them)."""
    exp = (rec.get("settings") or {}).get("experiment") or {}
    return exp.get("chat_rounds", 3), exp.get("char_limit", 300)


def part_lengths(rec: dict) -> tuple[int, int]:
    """Part lengths from the record's settings (the pilots' 6 and 4 for records without them)."""
    lengths = ((rec.get("settings") or {}).get("experiment") or {}).get("part_lengths") or (6, 4)
    return int(lengths[0]), int(lengths[1])


def check_market(rec: dict) -> dict:
    issues: list[str] = []
    info: dict = {}
    letters = letters_of(rec)
    orders = rec["settings"]["listing_orders"]
    lengths = part_lengths(rec)
    max_rounds, char_limit = chat_spec(rec)
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
                if len(msg) > char_limit:
                    issues.append(f"message over {char_limit} chars p{x['part']} t{x['period']} {L}")
        if len(rounds) > max_rounds:
            issues.append(f"{len(rounds)} chat rounds > {max_rounds} p{x['part']} t{x['period']}")
        if x["talk"] and len(rounds) < max_rounds and rounds and rounds[-1]["messages"]:
            issues.append(f"chat ended early without an all-pass round p{x['part']} t{x['period']}")
    for L in letters:
        t = rec["totals"][L]
        if abs(t["profit_part1"] - cum[L][0]) > 1e-6 or abs(t["profit_part2"] - cum[L][1]) > 1e-6:
            issues.append(f"totals mismatch {L}: {t} vs {cum[L]}")
    # fallback = previous own price
    prev = {L: 100 for L in letters}
    for x in periods:
        for L in letters:
            d = x["decisions"][L]
            if d["fallback"] and x["prices"][L] != prev[L]:
                issues.append(f"fallback price {x['prices'][L]} != previous {prev[L]} ({L} p{x['part']} t{x['period']})")
            prev[L] = x["prices"][L]

    # what each firm was shown
    all_notes = firm_notes(rec)
    for L in letters:
        v = firm_view(rec, L)
        if not v["roles_ok"]:
            issues.append(f"{L}: role alternation broken")
        if "Part 2" in v["system"]:
            issues.append(f"{L}: system prompt mentions Part 2")
        users = v["users"]
        full_user = "\n".join(users)
        if re.search(r"Part 2|second part", v["part1_text"]):
            issues.append(f"{L}: Part 1 text mentions Part 2")
        p2_instr = v["p2_instr"]
        if talk["part1"] and not talk["part2"]:
            if "will not be able to communicate" not in p2_instr:
                issues.append(f"{L}: Part 2 NoTalk-after-Talk sentence missing")
            if "chat box, which you can use" in p2_instr:
                issues.append(f"{L}: Part 2 describes a chat box")
        after = "\n".join(users[v["p2i"]:])
        if "Instructions – Part 2" in after:  # older records: the instructions sit in the history
            after = after.split("Instructions – Part 2", 1)[1]
        if talk["part2"] is False and re.search(r": chat,? round|chat box has closed|Messages posted", after):
            issues.append(f"{L}: chat prompt in Part 2 NoTalk")
        dies = re.findall(r"Period (\d+) is over\. The computer threw the virtual dice and it showed (\d)", full_user)
        if dies != expected_dies(rec):
            issues.append(f"{L}: die lines {dies} != {expected_dies(rec)}")
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
            if v["sections"]:  # cumulative profit follows the summaries in each request, not each results text
                if cp or ct:
                    issues.append(f"{L}: cumulative profit inside a period summary p{p} t{t}")
                continue
            if cp != format_ecu(run[x["part"] - 1]):
                issues.append(f"{L}: cumulative part profit shown {cp} != {format_ecu(run[x['part'] - 1])} p{p} t{t}")
            if x["part"] == 2 and ct != format_ecu(sum(run)):
                issues.append(f"{L}: cumulative total shown {ct} != {format_ecu(sum(run))}")
            if x["part"] == 1 and ct:
                issues.append(f"{L}: experiment total shown in Part 1")
        final = re.search(r"The sum of your firm's profits during the experiment is ([\d,\.]+) ECU", full_user)
        if not final or final.group(1) != format_ecu(sum(run)):
            issues.append(f"{L}: final total line wrong/missing")
        # other firms' notes never in anything this firm saw
        for F in letters:
            if F != L and any(len(nt) > 30 and nt in v["all_text"] for nt in all_notes[F]):
                issues.append(f"{L}: sees notes of {F}")
        # round-message blocks show exactly the previous round's posted messages, in L's listing order
        if v["sections"]:  # each chat round as the firm saw it: every firm in its listing order
            want = [round_text(r["round"], r["messages"], orders[L], L, own_failed=r["decisions"][L]["fallback"])
                    for x in periods for r in x["chat"]]
            if v["chat_rounds"] != want:
                issues.append(f"{L}: chat rounds shown differ from the record")
        for x in periods if not v["sections"] else []:
            for r in x["chat"]:
                blocks = []  # the firm's own message as a placeholder (current) or repeated (older records)
                for own in (OWN_MESSAGE, None):
                    lines = [f"Firm {F} (you): {own or r['messages'][F]}" if F == L else f"Firm {F}: {r['messages'][F]}"
                             for F in orders[L] if F in r["messages"]]
                    blocks.append(f"Messages posted in round {r['round']}:\n" + "\n".join(lines) if lines
                                  else f"No firm posted a message in round {r['round']}.")
                if not any(b in full_user for b in blocks):
                    issues.append(f"{L}: round block p{x['part']} t{x['period']} r{r['round']} not shown verbatim")
        if "totals" in rec and rec["totals"][L].get("notes_truncated"):
            info.setdefault("notes_truncated", {})[L] = rec["totals"][L]["notes_truncated"]
    return {"issues": issues, "info": info}


def record_attempts(rec: dict) -> list[tuple[str, tuple, dict]]:
    """(firm, (part, period, phase), attempt) of every attempt in the record; the probe as ("probe", None, "probe")."""
    out = []
    for x in rec["periods"]:
        for r in x["chat"]:
            for L, d in r["decisions"].items():
                out += [(L, (x["part"], x["period"], f"chat{r['round']}"), a) for a in d["attempts"]]
        for L, d in x["decisions"].items():
            out += [(L, (x["part"], x["period"], "price"), a) for a in d["attempts"]]
    for L, d in (rec.get("transition") or {}).items():
        out += [(L, (2, 0, "transition"), a) for a in d["attempts"]]
    for L, f in ((rec.get("probe") or {}).get("firms") or {}).items():
        out += [(L, ("probe", None, "probe"), a) for a in f["attempts"]]
    return out


LEAK_MIN_CHARS = 25  # shorter messages ("ok", "agreed, 100") recur by chance


def leaks(rec: dict, L: str, where: tuple, text: str, all_notes: dict, posted: dict,
          own_notes: set[str] | None = None) -> list[str]:
    """Information firm L must not see in a request at `where`: other firms' notes; for play calls also the
    same period's results and other firms' messages of the current or later rounds.

    A message counts as leaked if its text (over LEAK_MIN_CHARS chars) appears anywhere in the request, in any
    form (a round-block line, results text, ...), once everything L legitimately saw is removed: every earlier
    round's message lines ("Firm X: <text>" and "Firm X (you): <text>"), and L's own messages and notes (`own_notes`;
    default: L's notes in the record). So a firm, or another firm earlier, quoting a sentence that someone posts
    later is not a leak."""
    out = []
    for F, notes in all_notes.items():
        if F != L and any(len(nt) > 30 and nt in text for nt in notes):
            out.append(f"notes of {F} visible")
    part, period, phase = where
    if part == "probe":  # after all play
        return out
    if f"Results of period {period} (Part {part})" in text:
        out.append("same-period results visible")
    here = (part, period, int(phase[4:]) if phase.startswith("chat") else 99)  # price: every round is visible
    earlier = {(F, m) for k, ms in posted.items() if k < here for F, m in ms.items()}
    if own_notes is None:
        own_notes = all_notes.get(L, set())
    def one_line(m: str) -> str:  # as the summaries layout shows a message (prompts.shown_message, unquoted)
        return " ".join(m.splitlines())

    visible = {f"Firm {F}: {m}" for F, m in earlier} | {f"Firm {F} (you): {m}" for F, m in earlier}
    visible |= {f"Firm {F}: {shown_message(m)}" for F, m in earlier}
    visible |= {f"Firm {F} (you): {shown_message(m)}" for F, m in earlier}
    visible |= {x for ms in posted.values() for F, m in ms.items() if F == L for x in (m, one_line(m))}
    visible |= set(own_notes)
    rest = text
    for v in sorted((v for v in visible if v), key=len, reverse=True):  # longest first: lines before their texts
        rest = rest.replace(v, "\n")
    for k, ms in posted.items():
        if k >= here:
            for F, m in ms.items():
                if F != L and len(m) > LEAK_MIN_CHARS and (F, m) not in earlier and \
                        (m in rest or one_line(m) in rest):
                    out.append(f"future/same-round message of {F} visible")
    return out


def _slot(firm, part, period, phase) -> tuple:
    """A decision slot: (firm, part, period, phase), or (firm, "probe") for the post-game probe."""
    return (firm, "probe") if phase == "probe" else (firm, part, period, phase)


def unlogged_market(rec: dict, calls: list[dict]) -> list[str]:
    """A record with attempts but no call-log line for any of them (missing or mismatched log). An attempt is
    logged if its call_id is among the log's; an attempt without a call_id counts as unlogged, except in a legacy
    (append-only) record, whose attempts may lack call ids: there a log line for the same firm and decision slot
    (part, period, phase) counts."""
    attempts = record_attempts(rec)
    tags = [c.get("tag") or {} for c in calls]
    logged_ids = {t["call_id"] for t in tags if t.get("call_id") is not None}
    legacy = "histories" in rec
    logged_slots = {_slot(t.get("firm"), t.get("part"), t.get("period"), t.get("phase")) for t in tags} if legacy \
        else set()

    def is_logged(L, where, a) -> bool:
        if a.get("call_id") is not None:
            return a["call_id"] in logged_ids
        return legacy and _slot(L, *where) in logged_slots

    if attempts and not any(is_logged(*x) for x in attempts):
        return [f"record has {len(attempts)} attempts but no matching call-log lines"]
    return []


def check_calls(rec: dict, calls: list[dict]) -> dict:
    """Every request is rebuilt from the record and its hash checked against the record and the call log; no
    request shows other firms' notes or same-round/same-period information of other firms. Older (append-only)
    records: requests must be prefixes of the final history."""
    if "histories" in rec:
        res = check_calls_legacy(rec, calls)
        res["issues"] = unlogged_market(rec, calls) + res["issues"]
        return res
    issues: list[str] = unlogged_market(rec, calls)
    all_notes = firm_notes(rec)
    posted = {(x["part"], x["period"], r["round"]): r["messages"] for x in rec["periods"] for r in x["chat"]}
    logged = defaultdict(list)
    for c in calls:
        if c.get("error") is None and "messages_sha256" in c:
            logged[c["tag"].get("call_id")].append(c)
    n_ok = n_log = 0
    for L, where, a in record_attempts(rec):
        cid, ctx = a["call_id"], a.get("context")
        if not ctx:
            issues.append(f"no context recorded: {cid}")
            continue
        try:
            msgs = reconstruct_request(rec, L, ctx)
        except ValueError as e:
            issues.append(f"request not reconstructable: {cid}: {e}")
            continue
        n_ok += 1
        if messages_sha256(msgs) != ctx.get("messages_sha256"):
            issues.append(f"rebuilt request does not match its context hash: {cid}")
        if (ctx.get("merged") or ctx.get("layout")) and any(x["role"] == y["role"] for x, y in zip(msgs, msgs[1:])):
            issues.append(f"roles do not alternate: {cid}")
        if msgs[-1]["role"] != "user":
            issues.append(f"request does not end with a user turn: {cid}")
        if ctx.get("layout") and ctx.get("totals") != expected_totals(rec, L, where):
            issues.append(f"cumulative profit shown {ctx.get('totals')!r} != record: {cid}")
        if lines := logged.get(cid):  # the log hashes the messages as sent (cache_control included)
            n_log += 1
            if not any(messages_sha256(reconstruct_request(rec, L, ctx, c.get("cache_upto"))) == c["messages_sha256"]
                       for c in lines):
                issues.append(f"rebuilt request does not match the call log: {cid}")
        text = "\n".join(content_text(m) for m in msgs)
        issues += [f"{x}: {cid}" for x in leaks(rec, L, where, text, all_notes, posted)]
    if any(t and "kind" in t[0] for t in rec["transcripts"].values()):
        issues += check_sections(rec)
    return {"issues": issues, "ok_calls": n_ok, "checked_against_log": n_log}


def expected_totals(rec: dict, L: str, where: tuple) -> str:
    """Firm L's cumulative-profit text for a call at `where` ((part, period, phase), or the probe): the profits of
    the periods completed before it."""
    if where[0] == "probe":
        done, part = rec["periods"], 2
    else:
        part, period = where[0], where[1]
        done = [x for x in rec["periods"] if (x["part"], x["period"]) < (part, period)]
    cum = [sum(x["profits"][L] for x in done if x["part"] == p) for p in (1, 2)]
    return totals_text(part, cum[part - 1], sum(cum) if part == 2 else None)


def decisions_in_order(rec: dict, L: str) -> list[tuple[tuple, dict]]:
    """((part, period, phase), decision) of firm L in the order they were made (the transition call before Part 2
    period 1; the probe last)."""
    out = []
    for x in rec["periods"]:
        if (x["part"], x["period"]) == (2, 1) and rec.get("transition"):
            out.append(((2, 0, "transition"), rec["transition"][L]))
        out += [((x["part"], x["period"], f"chat{r['round']}"), r["decisions"][L]) for r in x["chat"]]
        out.append(((x["part"], x["period"], "price"), x["decisions"][L]))
    if (f := ((rec.get("probe") or {}).get("firms") or {}).get(L)) is not None:
        out.append((("probe", None, "probe"), {**f, "notes": None, "fallback": not f["valid"]}))
    return out


def check_sections(rec: dict) -> list[str]:
    """Summaries layout: each attempt's context selects what the design says. "In the transcript" no longer means
    "shown", and a rebuilt request always matches its own hash, so the selection itself is checked: the last W
    summaries before the call, every chat round of the call's period so far (Talk play calls only; none for the
    transition call, the probe and NoTalk parts) and the notes of the latest valid answer that carried any."""
    issues: list[str] = []
    W = ((rec.get("settings") or {}).get("experiment") or {}).get("memory_window", 3)
    talk = rec["settings"]["talk"]
    for L in letters_of(rec):
        tr, notes = rec["transcripts"][L], None
        for (part, period, phase), d in decisions_in_order(rec, L):
            for a in d["attempts"]:
                ctx = a.get("context") or {}
                if ctx.get("layout") is None:
                    continue
                lo = ctx["turns"][0]
                summ = [i for i in range(lo) if tr[i]["kind"] == "summary"]
                if ctx["summaries"] != summ[-W:]:
                    issues.append(f"{L}: summaries shown {ctx['summaries']} != the last {W} {summ[-W:]}: {a['call_id']}")
                in_talk = part in (1, 2) and phase != "transition" and talk[f"part{part}"]
                g = tr[lo]["period"]
                rounds = [i for i in range(lo) if tr[i]["kind"] == "chat" and tr[i]["period"] == g]
                want = rounds if in_talk else None
                if ctx["chat"] != want:
                    issues.append(f"{L}: chat section {ctx['chat']} != {want}: {a['call_id']}")
                if in_talk and len(rounds) != (int(phase[4:]) - 1 if phase.startswith("chat") else
                                               len(next(x for x in rec["periods"]
                                                        if (x["part"], x["period"]) == (part, period))["chat"])):
                    issues.append(f"{L}: {len(rounds)} chat rounds shown: {a['call_id']}")
                at = ctx.get("notes_at")
                shown = json.loads(tr[at]["content"]).get("notes") if at is not None else None
                if ctx.get("notes") and shown != notes:
                    issues.append(f"{L}: notes shown are not the latest valid notes: {a['call_id']}")
            if not d.get("fallback") and d.get("notes") is not None:
                notes = d["notes"]
    return issues


def check_calls_legacy(rec: dict, calls: list[dict]) -> dict:
    """Append-only records: request snapshots must be prefixes of the final history, and never contain
    same-round/same-period information of other firms."""
    issues: list[str] = []
    hist = {L: [(m["role"], m["content"]) for m in h] for L, h in rec["histories"].items()}
    all_notes = firm_notes(rec)
    posted = {(x["part"], x["period"], r["round"]): r["messages"] for x in rec["periods"] for r in x["chat"]}
    n_ok = 0
    for c in calls:
        if c.get("error") is not None or c["tag"].get("phase") == "probe":  # the probe follows all play
            continue
        n_ok += 1
        if "request" not in c:
            issues.append(f"request not reconstructable: {c['tag'].get('call_id')}: {c.get('reconstruct_error')}")
            continue
        tg = c["tag"]
        L = tg["firm"]
        req = [(m["role"], content_text(m)) for m in c["request"]["messages"]]
        if req != hist[L][:len(req)]:
            issues.append(f"request not a prefix of history: {tg['call_id']}")
            continue
        if req[-1][0] != "user":
            issues.append(f"request does not end with a user turn: {tg['call_id']}")
        text = "\n".join(t for _, t in req)
        issues += [f"{x}: {tg['call_id']}" for x in
                   leaks(rec, L, (tg["part"], tg["period"], tg["phase"]), text, all_notes, posted)]
    return {"issues": issues, "ok_calls": n_ok}


# ---------------------------------------------------------------------------------------------- B: behaviour


def behaviour(rec: dict) -> dict:
    letters = letters_of(rec)
    char_limit = chat_spec(rec)[1]
    out: dict = {"prices": [], "chat": [], "agreements": []}
    n_msg = n_slot = 0
    lens = []
    reasks = Counter()
    fallbacks = Counter()
    warns = Counter()
    errors = Counter()
    prev_agreed, prev_part = None, None
    for x in rec["periods"]:
        if x["part"] != prev_part:  # as analysis.chat.code_periods: agreements are coded within a part
            prev_agreed, prev_part = None, x["part"]
        out["prices"].append({"part": x["part"], "period": x["period"],
                              "prices": "/".join(str(x["prices"][L]) for L in letters)})
        decs = [("price", d) for d in x["decisions"].values()]
        if (x["part"], x["period"]) == (2, 1):
            decs += [("transition", d) for d in (rec.get("transition") or {}).values()]
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
    out["msg_near_limit"] = sum(v >= char_limit - 20 for v in lens)  # within 20 chars of char_limit
    out["msg_over_280"] = out["msg_near_limit"]  # former name (char_limit 300), kept for older readers
    out["reasks"] = dict(reasks)
    out["fallbacks"] = dict(fallbacks)
    out["warnings"] = dict(warns)
    out["errors"] = dict(errors)
    out["profits"] = {L: rec["totals"][L]["profit_total"] for L in letters}
    return out


# ---------------------------------------------------------------------------------------------- C: tokens


def tokens(calls: list[dict], lengths: tuple[int, int]) -> dict:
    """Token, cost and latency statistics of a market's calls; `lengths` are its part lengths (`part_lengths`)."""
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
        ph = "chat" if tg["phase"].startswith("chat") else tg["phase"]  # price or probe
        if tg["attempt"] == 0:
            by_period[(tg["part"], tg["period"], tg["phase"])].append(u["prompt_tokens"])
        phase[ph]["completion"].append(u["completion_tokens"])
        phase[ph]["reasoning"].append(u["reasoning_tokens"])
        phase[ph]["cost"].append(c.get("cost_usd") or 0)
        b = request_bytes(c)
        if b:
            ratio.append(u["prompt_tokens"] / b)
        if c.get("latency_s") is not None:
            lat.append(c["latency_s"])
        cost.append(c.get("cost_usd"))
        cached += u.get("cached_tokens") or 0
        seed_flags[(c.get("seed_possibly_ignored"), c.get("seed_flag_reason"))] += 1
        providers[(c.get("provider_requested"), c.get("provider_served"), c.get("provider_mismatch"))] += 1
        models_served[c.get("model_served")] += 1
        finish[(c.get("finish_reason"), c.get("native_finish_reason"))] += 1
        reasoning_present += bool(c.get("reasoning") or c.get("reasoning_details"))
    # growth: mean prompt tokens of the round-1/price prompt per period
    mp = {k: mean(v) for k, v in by_period.items()}

    def first_phase(part, t):
        return mp.get((part, t, "chat1")) or mp.get((part, t, "price"))

    p1 = [first_phase(1, t) for t in range(1, lengths[0] + 1)]
    p2 = [first_phase(2, t) for t in range(1, lengths[1] + 1)]
    g1 = [b - a for a, b in zip(p1, p1[1:]) if a and b]
    g2 = [b - a for a, b in zip(p2, p2[1:]) if a and b]
    return {
        "calls_ok": len(ok), "calls_err": len(errs),
        "err_kinds": dict(Counter(re.sub(r"\d{3,}", "#", str(c["error"]))[:80] for c in errs)),
        "timeout_retries": sum(bool(c.get("timeout_retry")) for c in calls),
        "prompt_by_period_first_call": {"part1": p1, "part2": p2},
        "price_prompt_by_period": {f"p{p}t{t}": mp.get((p, t, "price")) for p in (1, 2)
                                   for t in range(1, lengths[p - 1] + 1) if mp.get((p, t, "price"))},
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS, help="run ids (directories under --runs-dir)")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", help="report path (default data/audits/<run ids>_audit.json)")
    a = ap.parse_args(argv)
    runs = [Path(a.runs_dir) / r for r in a.runs]
    out = Path(a.out) if a.out else AUDITS_DIR / f"{'+'.join(a.runs)}_audit.json"
    calls_by_market = defaultdict(list)  # (run id, market key) -> its calls
    for run in runs:
        log = run / "calls.jsonl"
        for c in load_calls(log) if log.exists() else []:
            tag = c.get("tag") or {}
            if "event" not in c and tag.get("market"):
                calls_by_market[(tag.get("run_id", run.name), tag["market"])].append(c)
    report = {}
    for run in runs:
        for f in market_files(run):
            rec = json.loads(f.read_text())
            lengths = part_lengths(rec)
            mkey = f.name.removesuffix(".json").removesuffix(".failed")
            calls = calls_by_market[(rec["spec"]["run_id"], mkey)]
            key = f"{run.name}/{mkey}"
            if f.name.endswith(".failed.json"):
                report[key] = {"failed": True, "reason": rec.get("reason"), "error": rec["error"][:300],
                               "periods_completed": rec["periods_completed"],
                               "tokens": tokens(calls, lengths)}
                continue
            report[key] = {
                "market": mkey,
                "engine": check_market(rec),
                "calls": check_calls(rec, calls),
                "usage_record": rec["usage"],
                "behaviour": behaviour(rec),
                "tokens": tokens(calls, lengths),
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    for m, r in report.items():
        print("=" * 30, m)
        if r.get("failed"):
            print("FAILED", r["reason"], r["error"][:200], r["tokens"]["calls_ok"], r["tokens"]["calls_err"])
            continue
        print("engine issues:", r["engine"]["issues"][:20])
        print("call issues:", r["calls"]["issues"][:10], "checked", r["calls"]["ok_calls"], "of which against the log",
              r["calls"].get("checked_against_log"), "usage calls", r["usage_record"]["calls"])
        b = r["behaviour"]
        print("prices:", " | ".join(f"{p['part']}.{p['period']}:{p['prices']}" for p in b["prices"]))
        print("agreements:", [(a["period"], a["agreed"], a["support"], a["kept"], a["undercutters"], a["rounds"]) for a in b["agreements"]])
        print({k: b[k] for k in ("pass_rate", "msg_len_mean", "msg_len_max", "msg_near_limit", "reasks", "fallbacks", "warnings", "errors", "profits")})
        t = r["tokens"]
        print({k: v for k, v in t.items()})
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
