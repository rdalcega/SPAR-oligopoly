import json

import numpy as np
import pandas as pd
import pytest

from oligopoly import market
from oligopoly.analysis import chat as C
from oligopoly.analysis import measures as M
from oligopoly.analysis import mistakes as X
from oligopoly.analysis.load import tidy
from oligopoly.analysis.report import build, write

TALK = {"NT-T": (False, True), "T-NT": (True, False), "NT-NT": (False, False)}


def dec(value, notes="", fallback=False, errors=(), warnings=()):
    return {"value": value, "notes": notes, "valid": not fallback, "fallback": fallback, "reasks": len(errors),
            "errors": list(errors), "warnings": list(warnings), "attempts": []}


def record(prices: dict[int, list[dict[str, int]]], ordering="NT-T", n=None, model="m", idx=0, run="r",
           chats=None, notes=None):
    """Minimal engine-style record. prices[part] = list of {firm: price}; chats[(part, period)] = list of
    rounds, each {firm: message or None}; notes[(part, period)] = {firm: text}."""
    chats, notes = chats or {}, notes or {}
    firms = sorted(prices[1][0])
    n = n or len(firms)
    lengths = [len(prices[1]), len(prices[2])]
    periods = []
    for part in (1, 2):
        for t, pr in enumerate(prices[part], start=1):
            nt = notes.get((part, t), {})
            rounds = [{"round": r, "messages": {f: m for f, m in rd.items() if m is not None},
                       "decisions": {f: dec(m) for f, m in rd.items()}}
                      for r, rd in enumerate(chats.get((part, t), []), start=1)]
            periods.append({"part": part, "period": t, "talk": TALK[ordering][part - 1], "die": None,
                            "chat": rounds, "prices": pr, "notes": {f: nt.get(f, "") for f in firms},
                            "profits": market.payoffs(pr),
                            "decisions": {f: dec(pr[f], nt.get(f, "")) for f in firms}})
    return {"spec": {"run_id": run, "model": model, "n": n, "ordering": ordering, "market_index": idx, "seed": 0},
            "settings": {"experiment": {"part_lengths": lengths}, "pilot": False,
                         "talk": {"part1": TALK[ordering][0], "part2": TALK[ordering][1]}},
            "periods": periods, "_market_id": f"{run}/{model}-{n}-{ordering}-{idx}"}


def flat(*ps):
    return [{"A": p, "B": q} for p, q in ps]


# ------------------------------------------------------------------------------------------- measures


def test_markov_counts_within_parts_only():
    rec = record({1: flat((100, 100), (100, 100), (99, 100), (50, 60), (100, 100)),
                  2: flat((100, 100), (99, 100))})
    t = tidy([rec])
    c = M.transition_counts(t.market_period)
    assert c.loc["100", "100"] == 1 and c.loc["100", "99"] == 2 and c.loc["99", "<99"] == 1
    assert c.loc["<99", "100"] == 1
    assert c.to_numpy().sum() == 5  # 4 in part 1 + 1 in part 2; no transition across parts


def test_ergodic_and_shorrocks():
    P = np.array([[0.9, 0.1, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    pi = M.ergodic(P)
    assert pi @ P == pytest.approx(pi) and pi.sum() == pytest.approx(1)
    counts = pd.DataFrame([[9, 1, 0], [0, 5, 5], [5, 0, 5]], index=["100", "99", "<99"], columns=["100", "99", "<99"])
    s = M.markov_summary(counts, pd.Series(["100"] * 3 + ["<99"]))
    assert s["shorrocks"] == pytest.approx((3 - 1.9) / 2)
    assert s["observed"] == {"100": 0.75, "99": 0.0, "<99": 0.25}
    assert M.ergodic(np.array([[1, 0, 0], [np.nan] * 3, [0, 0, 1.0]])) is None
    assert M.ergodic(np.eye(3)) is None  # every distribution is stationary


def test_windows_gain_and_cells():
    p1 = flat(*[(0, 0)] * 5 + [(10, 20)] * 3)  # periods 1-5 at 0 (dropped), 6-8 selling 10
    p2 = flat(*[(100, 100)] * 5 + [(50, 50)] * 2)
    recs = [record({1: p1, 2: p2}, idx=i) for i in range(2)]
    t = tidy(recs)
    mt = M.market_table(t.market_period, t.markets)
    r = mt.iloc[0]
    assert r["sp_p1_from6"] == 10 and r["posted_p1_from6"] == 15
    assert r["sp_p2_all"] == pytest.approx((500 + 100) / 7) and r["sp_p2_from6"] == 50
    assert r["gain_p2_from6"] == 40 and r["sp_p2_last5"] == pytest.approx((300 + 100) / 5)
    cells = M.cell_stats(mt).set_index("measure")
    assert cells.loc["sp_p1_from6", "n_markets"] == 2 and cells.loc["sp_p1_from6", "sd"] == 0
    assert t.market_period.query("market_index == 0")["gperiod"].tolist() == list(range(1, 16))


def test_table2_bins():
    p1 = flat(*[(100, 100)] * 5 + [(100, 100), (99, 100), (50, 98), (1, 5)])
    t = tidy([record({1: p1, 2: flat((0, 0))})])
    row = M.table2(t.market_period).query("window == 'p1_from6'").iloc[0]
    assert (row["100"], row["99"], row["2-98"], row["<=1"]) == (25.0, 25.0, 25.0, 25.0)


def test_strategy_spells_defections_turns():
    p2 = flat((100, 100), (100, 100), (99, 100), (100, 100), (100, 100), (99, 100), (40, 100),
              (99, 100), (100, 99), (99, 100))
    t = tidy([record({1: flat((5, 5)), 2: p2})])
    per, _ = M.strategy(t.market_period, t.firm_period)
    s = per.query("part == 2").iloc[0]
    assert s["first_all100"] and s["n_all100_spells"] == 2 and s["mean_all100_spell"] == 2
    # defections: t-2 all100, t-1 at 99 -> periods 3 (kept: 100 at 4) and 6 (punished: 40 at 7)
    assert (s["defections"], s["defections_kept"], s["defections_punished"]) == (2, 1, 1)
    assert s["turn_taking_periods"] == 3  # periods 8-10 alternate the 99 firm; period 3 and 6 are single


def test_theory_threshold_sides():
    side = {r["n"]: r["side"] for r in M.theory([2, 4, 6, 8, 10])}
    assert side == {2: "inside", 4: "inside", 6: "inside", 8: "outside", 10: "outside"}


# ------------------------------------------------------------------------------------------- chat coding


@pytest.mark.parametrize("msg, prices, endorsed", [
    ("Shall we all set 100?", [100], 100),
    ("I will keep my price at 100.", [100], 100),
    ("Let's all go 100 again. If anyone undercuts, I will go to 0.", [100, 0], 100),
    ("Last period was 45, terrible. Let us return to 100", [45, 100], 100),
    ("You undercut us at 99! Why did you do that?", [99], None),
    ("We have 20 periods at least and 4 firms; 95 for everyone", [95], 95),
    ("Period 3 was bad. ok", [], None),
    ("Profit was 15000 last time", [], None),
])
def test_message_prices_and_endorsement(msg, prices, endorsed):
    c = C.code_message(msg)
    assert c["prices"] == prices and c["endorsed"] == endorsed


@pytest.mark.parametrize("msg, cats", [
    ("Agreed, 100 it is.", {"agreement"}),
    ("If anyone undercuts, I will go to 0 forever.", {"threat"}),
    ("This is your last chance before I start hitting 0", {"threat"}),
    ("Let's take turns: A at 99 this period, B next", {"turns"}),
    ("You undercut us! You promised 100.", {"accusation"}),
    ("Sorry, my mistake, back to 100", {"apology", "proposal"}),
    ("Let's not undercut each other", set()),
])
def test_message_categories(msg, cats):
    c = C.code_message(msg)
    found = {k for k in ("agreement", "threat", "turns", "accusation", "apology", "proposal") if c[k]}
    assert found == cats


def msgs(*rows):
    return [{"round": r, "firm": f, **C.code_message(m)} for r, f, m in rows]


def test_agreement_kept_and_bare_ok_support():
    m = msgs((1, "A", "Let's all set 100"), (2, "B", "ok"), (2, "C", "sure"))
    r = C.code_period(m, {"A": 100, "B": 100, "C": 100}, None)
    assert r["agreed_price"] == 100 and r["support"] == 3 and r["kept"]


def test_agreement_broken_undercut_and_above():
    m = msgs((1, "A", "100?"), (1, "B", "I'll set 100"), (1, "C", "90 is safer"), (2, "C", "fine, agreed"))
    r = C.code_period(m, {"A": 100, "B": 99, "C": 100}, None)
    assert r["agreed_price"] == 100 and r["kept"] is False and r["undercutters"] == ["B"] and r["max_undercut"] == 1
    m = msgs((1, "A", "Let's do 90"), (1, "B", "90 works"), (1, "C", "no"))
    r = C.code_period(m, {"A": 90, "B": 100, "C": 85}, None)
    assert r["agreed_price"] == 90 and r["undercutters"] == ["C"] and r["above"] == ["B"]


def test_no_majority_no_agreement():
    m = msgs((1, "A", "100"), (1, "B", "90"), (1, "C", "80"), (1, "D", "100"))
    assert C.code_period(m, {f: 50 for f in "ABCD"}, None)["agreement"] is False  # 2 of 4 is not a majority


def test_bare_ok_refers_to_previous_agreement():
    r = C.code_period(msgs((1, "A", "ok"), (1, "B", "yes")), {"A": 100, "B": 100}, 100)
    assert r["agreed_price"] == 100 and r["kept"]


def test_code_periods_punishment_threat_and_deception():
    chats = {(2, 1): [{"A": "Let's all set 100. If anyone undercuts, I will go to 0.", "B": "Agreed, 100"}],
             (2, 2): [{"A": "100 again", "B": "yes 100"}],
             (2, 3): [{"A": "You undercut! Back to 100", "B": "sorry, 100"}]}
    notes = {(2, 2): {"B": "They trust me. I will set 99 to take the whole market."}}
    p2 = flat((100, 100), (100, 99), (50, 100), (100, 100))
    rec = record({1: flat((10, 10)), 2: p2}, chats=chats, notes=notes)
    t = tidy([rec])
    cm = C.code_messages(t.chat)
    per = C.code_periods(cm, t.chat, t.firm_period).set_index("period")
    assert per.loc[1, "kept"] and per.loc[1, "threat"]
    assert per.loc[2, "kept"] is False
    assert per.loc[2, "undercutters"] == "B" and per.loc[2, "threatened_so_far"]
    assert per.loc[2, "punished_next"] and per.loc[2, "sp_next_lt99"]  # A cut from 100 to 50
    assert per.loc[2, "deception_notes"] == "B" and per.loc[2, "deception_priced"] == "B"
    assert per.loc[3, "accusation"] and per.loc[3, "apology"] and per.loc[3, "agreed_price"] == 100
    assert per.loc[4, "rounds_used"] == 0 and not per.loc[4, "agreement"]
    summ = C.chat_summary(per.reset_index(), t.chat).iloc[0]
    assert summ["agreements"] == 3 and summ["breaches_undercut"] == 2 and summ["share_threatened_before"] == 1


# ------------------------------------------------------------------------------------------- mistakes


def test_stated_price_and_negated_deviation():
    assert C.stated_own_price("Others at 100. I will set 99 to grab the market.") == 99
    assert C.stated_own_price("My price last period was 50. Keep cooperating.") is None
    assert C.stated_own_price("I'll keep my price at 100") == 100
    assert not C.has_unnegated(C.DEVIATE, "I will not undercut them.")
    assert C.has_unnegated(C.DEVIATE, "Time to undercut slightly.")


def test_mistake_flags():
    notes = {(1, 2): {"A": "I will set 100 to signal cooperation."}, (1, 3): {"B": "go low"}}
    p1 = flat((100, 100), (99, 100), (100, 0))
    t = tidy([record({1: p1, 2: flat((100, 100))}, notes=notes)])
    ff = X.flag_firm_periods(t.firm_period, C.code_messages(t.chat)).set_index(["part", "period", "firm"])
    assert ff.loc[(1, 2, "A"), "notes_contradict"] and not ff.loc[(1, 2, "B"), "notes_contradict"]
    assert ff.loc[(1, 3, "B"), "low_price_no_context"] and ff.loc[(1, 3, "B"), "price_zero"]
    assert ff.loc[(1, 3, "A"), "others_min"] == 0 and ff.loc[(1, 3, "B"), "prev_others_min"] == 99


def test_low_price_with_punishment_notes_not_flagged():
    notes = {(1, 2): {"B": "Punish A for cheating."}}
    t = tidy([record({1: flat((100, 100), (100, 0)), 2: flat((1, 1))}, notes=notes)])
    ff = X.flag_firm_periods(t.firm_period, C.code_messages(t.chat))
    assert not ff["low_price_no_context"].any()


@pytest.mark.parametrize("text, period, cats", [
    ("The highest price wins, so 100.", 3, ["highest_wins"]),
    ("If we tie at 100 I get the entire market", 3, ["tie_full"]),
    ("If we tie we split the market equally, full market otherwise", 3, []),
    ("This is the last period, so undercut.", 7, ["horizon"]),
    ("This might be the final period.", 22, []),
    ("Last period everyone was at 100.", 5, []),
])
def test_misreads(text, period, cats):
    assert X.misreads(text, period) == cats


# ------------------------------------------------------------------------------------------- pipeline


def test_build_and_write(tmp_path):
    runs = tmp_path / "runs" / "r" / "markets"
    runs.mkdir(parents=True)
    chats = {(2, t): [{"A": "Let's all set 100", "B": None}, {"A": None, "B": "ok"}] for t in range(1, 4)}
    for o in ("NT-T", "T-NT", "NT-NT"):
        for i in range(3):
            ch = chats if o == "NT-T" else {(1, t): v for (_, t), v in chats.items()} if o == "T-NT" else {}
            rec = record({1: flat(*[(i * 10, 50)] * 7), 2: flat(*[(100, 100 - i)] * 6)}, ordering=o, idx=i, chats=ch,
                         n=2)
            rec.pop("_market_id")
            (runs / f"m__n2__{o}__m{i}.json").write_text(json.dumps(rec))
    tables, summary = build(["r"], tmp_path / "runs")
    files = write(tmp_path / "out", tables, summary)
    assert "summary.json" in files and "human_comparison.csv" in files
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert {"theory", "human_comparison", "markov", "tests", "chat", "mistakes", "timeseries"} <= set(s)
    comp = pd.DataFrame(s["human_comparison"])
    assert set(comp["ordering"]) == {"NT-T"}  # n=2 controls are extensions, not compared
    assert not s["meta"]["paper_lengths_all"] and comp["caveat"].str.contains("Part lengths").all()
    ext = pd.DataFrame(s["extensions"])
    assert set(ext["ordering"]) == {"T-NT", "NT-NT"}
    talk = pd.DataFrame(s["chat"]).query("ordering == 'NT-T'").iloc[0]
    assert talk["agreements"] == 9


def test_excluded_cells_skipped_and_reported(tmp_path):
    runs = tmp_path / "runs" / "r" / "markets"
    runs.mkdir(parents=True)
    for o in ("NT-NT", "T-NT"):
        for i in range(2):
            rec = record({1: flat(*[(100, 100)] * 3), 2: flat(*[(90, 90)] * 3)}, ordering=o, idx=i, n=2,
                         chats={(1, t): [{"A": "Let's all set 100", "B": "ok"}] for t in range(1, 4)} if o == "T-NT" else {})
            rec.pop("_market_id")
            (runs / f"m__n2__{o}__m{i}.json").write_text(json.dumps(rec))
    model = rec["spec"]["model"]
    tables, summary = build(["r"], tmp_path / "runs", {model: {("T-NT", 2), ("NT-T", 10)}})
    assert set(tables["markets"]["ordering"]) == {"NT-NT"}
    assert summary["meta"]["excluded_cells"] == [
        {"model": model, "ordering": "T-NT", "n": 2, "markets_dropped": 2},
        {"model": model, "ordering": "NT-T", "n": 10, "markets_dropped": 0}]
    assert len(tables["excluded_cells"]) == 2
    tables, summary = build(["r"], tmp_path / "runs", {})
    assert set(tables["markets"]["ordering"]) == {"NT-NT", "T-NT"} and summary["meta"]["excluded_cells"] == []


def test_uniform_design_ignores_operational_fields_only():
    from oligopoly.analysis.load import check_uniform_design

    base = {"name": "m", "model": "acme/m", "provider": "p", "temperature": 1.0, "max_tokens": 100}

    def rec(i, digest="d1", **model):
        r = record({1: flat((100, 100)), 2: flat((100, 100))}, idx=i)
        r["settings"].update(model={**base, **model}, texts_digest=digest)
        return r

    check_uniform_design([rec(0), rec(1, pricing={"prompt": 1}, timeout_s=5, exclude_cells=[["NT-T", 8]],
                                      bytes_per_token_min=2.0)])
    with pytest.raises(ValueError, match="different settings"):
        check_uniform_design([rec(0), rec(1, temperature=0.5)])
    with pytest.raises(ValueError, match="different settings"):
        check_uniform_design([rec(0), rec(1, digest="d2")])


def test_records_off_the_run_prompt_digest_refused_or_skipped(tmp_path, capsys):
    from oligopoly.analysis.load import load_records

    runs = tmp_path / "runs" / "r" / "markets"
    runs.mkdir(parents=True)
    (runs.parent / "run_config.json").write_text(json.dumps({"texts_digest": "d1"}))
    for i, d in enumerate(("d1", "d1", "d2")):
        rec = record({1: flat((100, 100)), 2: flat((100, 100))}, idx=i, n=2)
        rec["settings"]["texts_digest"] = d
        rec.pop("_market_id")
        (runs / f"m__n2__NT-T__m{i}.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="other prompt wording"):
        load_records(["r"], tmp_path / "runs")
    recs = load_records(["r"], tmp_path / "runs", skip_mismatched_prompts=True)
    assert [r["_market_id"] for r in recs] == ["r/m__n2__NT-T__m0", "r/m__n2__NT-T__m1"]
    assert "WARNING: skipped r/m__n2__NT-T__m2" in capsys.readouterr().out


async def test_legacy_records_checked_by_fingerprint(tmp_path):
    from oligopoly.analysis.load import load_records
    from oligopoly.config import ExperimentConfig, ModelConfig
    from oligopoly.engine import MarketSpec, environment_seed, run_market, texts_digest
    from oligopoly.mock import MockClient

    exp = ExperimentConfig(part_lengths=(2, 1))
    spec = MarketSpec("r", ModelConfig("mock", "mock", "mock", max_tokens=10, timeout_s=3), 2, "NT-NT", 0,
                      environment_seed(0, 2, "NT-NT", 0))
    run = tmp_path / "runs" / "r"
    path = await run_market(spec, MockClient(), run, exp)
    rec = json.loads(path.read_text())
    del rec["settings"]["texts_digest"]  # as written before records stored it
    path.write_text(json.dumps(rec))
    (run / "run_config.json").write_text(json.dumps({"texts_digest": texts_digest(exp)}))
    [loaded] = load_records(["r"], tmp_path / "runs")
    assert loaded["settings"]["texts_digest"] == texts_digest(exp)
    (run / "run_config.json").write_text(json.dumps({"texts_digest": "other"}))
    with pytest.raises(ValueError, match="other prompt wording"):
        load_records(["r"], tmp_path / "runs")


# ------------------------------------------------------------------------------------------- negative controls


def test_refusal_is_not_agreement():
    m = msgs((1, "A", "Let's all set 100"), (2, "B", "I do not agree."))
    r = C.code_period(m, {"A": 100, "B": 90}, None)
    assert not r["agreement"] and r["backed"]["B"] is None
    assert C.code_message("I do not agree to 100")["endorsed"] is None
    m = msgs((1, "A", "100"), (1, "B", "100"), (2, "B", "No, I changed my mind"))
    assert C.code_period(m, {"A": 100, "B": 90}, None)["backed"]["B"] is None


def test_avoiding_price_war_is_not_threat():
    assert not C.code_message("Let us avoid a price war. I will set 100.")["threat"]
    assert C.code_message("Keep 100 or there will be a price war.")["threat"]


def test_rival_talk_is_not_own_plan():
    assert C.stated_own_price("I will set 100. B is setting 99.") == 100
    assert C.stated_own_price("If B undercuts, I will set 0.") is None
    assert C.stated_own_price("I will not set 99.") is None
    chats = {(2, 1): [{"A": "Let's all set 100", "B": "Agreed, 100"}]}
    notes = {(2, 1): {"A": "B may undercut. I will set 100."}}
    t = tidy([record({1: flat((10, 10)), 2: flat((100, 100))}, chats=chats, notes=notes)])
    per = C.code_periods(C.code_messages(t.chat), t.chat, t.firm_period)
    assert per.iloc[0]["kept"] and per.iloc[0]["deception_notes"] == ""


def test_deception_plan_in_chat_notes():
    rec = record({1: flat((10, 10)), 2: flat((100, 99))}, chats={(2, 1): [{"A": "Let's all set 100", "B": None},
                                                                     {"A": None, "B": "ok"}]})
    rec["periods"][1]["chat"][1]["decisions"]["B"]["notes"] = "Say yes, then I will set 99."
    t = tidy([rec])
    per = C.code_periods(C.code_messages(t.chat), t.chat, t.firm_period)
    assert per.iloc[0]["deception_priced"] == "B"


@pytest.mark.parametrize("text", ["This is not the final period.", "If we tie, I do not get the entire market.",
                                  "A higher price earns more if everyone matches."])
def test_correct_rule_statements_not_misreads(text):
    assert X.misreads(text, 3) == []


def test_fn18_defection_with_deeper_cut():
    t = tidy([record({1: flat((5, 5)), 2: flat((100, 100), (100, 100), (99, 98), (100, 100))})])
    per, _ = M.strategy(t.market_period, t.firm_period)
    s = per.query("part == 2").iloc[0]
    assert (s["defections"], s["defections_kept"]) == (1, 1)


@pytest.mark.parametrize("msg, prices, endorsed", [
    ("Let's secure another 7,500 ECU together!", [], None),
    ("Undercutting gives 29,700 vs 7,500, but let's keep 100.", [100], 100),
    ("At 100 we each earn 11,250 ECU. Let's all set 100.", [100, 100], 100),
    ("prices 7,500ECU", [], None),
    ("We each get 75 profit", [], None),
    ("Let us all set 100, ok?", [100], 100),
])
def test_thousands_separators_are_not_prices(msg, prices, endorsed):
    assert C.prices_in(msg) == prices
    assert C.endorsed_price(msg) == endorsed


def test_stated_price_ignores_profit_amounts():
    assert C.stated_own_price("I will set 11,250 ECU of profit") is None
    assert C.stated_own_price("I will set 13,200") is None
    assert C.stated_own_price("I will set 100, then see") == 100


@pytest.mark.parametrize("msg, prices, endorsed", [
    ("Let's set 100 ECU", [100], 100),
    ("Agreed, 100 ECU", [100], 100),
    ("We each get 75 ECU", [], None),
    ("We each earn 75 ECU per period", [], None),
    ("75 ECU each is good", [], None),
])
def test_ecu_prices_vs_profit_amounts(msg, prices, endorsed):
    assert C.prices_in(msg) == prices
    assert C.endorsed_price(msg) == endorsed


@pytest.mark.parametrize("msg, prices", [
    ("7,500 ECU profit", []),
    ("29,700", []),
    ("We earn 1,100 in total", []),
    ("10,000", []),
    ("99,100 alternating", [99, 100]),
    ("100,100,100", [100, 100, 100]),
])
def test_comma_joined_price_lists(msg, prices):
    assert C.prices_in(msg) == prices


@pytest.mark.parametrize("msg", ["We should not set 100", "let's not go to 100", "I won't price at 100",
                                 "don't set 99", "We should not set 100. 100?"])
def test_negated_proposals_refuse(msg):
    c = C.code_message(msg)
    assert c["endorsed"] is None and c["refusal"]


def test_negated_proposal_clears_support_and_ruling_out_lower_prices_does_not():
    m = msgs((1, "A", "Let's all set 100"), (1, "B", "100"), (2, "B", "We should not set 100"))
    assert C.code_period(m, {"A": 100, "B": 90}, None)["backed"]["B"] is None
    assert C.code_message("We should not set 100. Let's set 99.")["endorsed"] == 99
    for msg in ("Don't go below 100.", "I won't go below 100"):
        c = C.code_message(msg)
        assert not c["refusal"] and c["endorsed"] == 100
