"""Reading this period's own price and deviation plans from memory-style notes (analysis/notes.py)."""

import json
from pathlib import Path

import pytest

from oligopoly.analysis import chat as C
from oligopoly.analysis import mistakes as X
from oligopoly.analysis.notes import deviation_plan, stated_own_price

LABELS = json.loads((Path(__file__).parent / "data" / "notes_labels.json").read_text())["notes"]


@pytest.mark.parametrize("text, period, price", [
    ("I will set 52 for this period.", 4, 52),
    ("Set my price to 52 now.", 4, 52),
    ("Price: 52", 4, 52),
    ("I'll keep my price at 100", None, 100),
    ("Maintain price 1 this period to keep the split.", 9, 1),
    ("Setting 96 is a strategic move to reclaim share.", 5, 96),
    ("I'll set 99 ECU in period 2.", 2, 99),
    ("To avoid getting zero if A lowers its price, I'll set 99.", 2, 99),
    # differences, other firms, history, contingent plans, ranges, profits
    ("To regain a payoff I will undercut by one, setting 86 to win.", 5, 86),
    ("I'll keep this rule: always set price 1 lower than Firm A's last price.", 5, None),
    ("Firm A last set 87 and I set 88, getting no sales.", 5, None),
    ("Firm A confirmed the 52-price split and a brief 51-price retaliation if either side deviates. "
     "I will set 52 for this period. After this period, I'll remain ready to enforce the 51-price punishment "
     "if needed and then return to 52.", 4, 52),
    ("If B deviates, I'll set 39 as agreed in prior rounds.", 3, None),
    ("I will set 30 if anyone undercuts.", 3, None),
    ("I plan to set price 2 in the next period.", 5, None),
    ("I'll set my price in period 7 to 40.", 6, None),
    ("I will set between 50 and 60.", 3, None),
    ("I will set 50 or 51.", 3, None),
    ("I will set 11,250 ECU of profit", None, None),
    ("I will not set 99.", None, None),
    ("Setting 32 would result in a split.", 3, None),
    ("Last period I set 88 to win the market.", 5, None),
    ("I will undercut to 30 to punish B.", 5, None),
    ("I will set 99 next period instead of 100 this period.", 5, None),
    ("I will set 99.5 this period.", 5, None),
    ("Last period I decided to set 40.", 5, None),
    ("I will set 99 once again.", 5, 99),
    ("I will set 99 ECU of profit.", 5, None),
    # disagreeing statements: the one with a this-period cue wins, otherwise None
    ("I said I'd go to 40. I will set 45 this period.", 3, 45),
    ("I will set 40. I will set 45.", 3, None),
])
def test_stated_own_price(text, period, price):
    assert stated_own_price(text, period) == price


@pytest.mark.parametrize("text, plan", [
    ("They trust me. I will undercut them this period.", True),
    ("I will deviate now - set 29 to capture the full market.", True),
    ("If B deviates, I'll undercut to 74 to reclaim the market.", False),
    ("I will not undercut them.", False),
    ("I could undercut, but that would break trust.", False),
    ("Firm A undercut me last period.", False),
    ("Next period I will undercut to 50.", False),
    ("Last period I set 88 to win the market.", False),
    ("I will undercut to 30 to punish B.", False),
])
def test_deviation_plan(text, plan):
    assert deviation_plan(text, 5) is plan


@pytest.mark.parametrize("text", [
    "Last period I set 100 expecting a match; instead A set 65 and took the entire market.",
    "Up to period 18 A had been matching our price; now it chose to step down to capture the full market.",
    "If B keeps 1, a tie gives 150; if B raises above 1, we capture the entire market (300).",
    "All alternatives give 0 profit (higher price gets undercut, lower price yields 0).",
    "This period (period 4) is the second and final period of the penalty.",
    "We are in the final period of the two-period punishment.",
])
def test_pilot_misread_false_positives(text):
    assert X.misreads(text, 4) == []


@pytest.mark.parametrize("text, cat", [
    ("If we tie we each get the full market.", "tie_full"),
    ("Matching gives me the entire market.", "tie_full"),
    ("The highest price wins the buyers.", "highest_wins"),
    ("This is the final period, so I will undercut.", "horizon"),
])
def test_misreads_still_found(text, cat):
    assert cat in X.misreads(text, 4)


@pytest.mark.parametrize("msg, endorsed", [
    ("We'll stay at price 1 this period. Next period we can move to 2 if we both agree.", 1),
    ("We stay at 1 today. Next period we shift to 2 only if we both keep it; otherwise we fall back to 1.", 1),
    ("Agreed, let's both set price 2 next period.", 2),  # no this-period cue: often the price about to be posted
    ("Let's keep 100 for this and all future periods.", 100),
    ("Once again, let's all set 100.", 100),
    ("The best option is 100 for all of us.", 100),
    ("If you undercut now, I'll go to 0. Let's set 2 next period.", 2),
    ("We can discuss moving to 2 later.", None),
])
def test_endorsement_skips_later_period_plans(msg, endorsed):
    assert C.endorsed_price(msg) == endorsed


@pytest.mark.parametrize("notes, flagged", [
    ("B threatened to punish. I will set 99 this period.", True),
    ("We agreed a 0-price punishment. I will set 99 this period to comply with the punishment.", False),
])
def test_deception_punishment_exemption_is_own_punishment_only(notes, flagged):
    from oligopoly.analysis.load import tidy
    from test_analysis import flat, record
    chats = {(2, 1): [{"A": "Let's all set 100", "B": "Agreed, 100"}]}
    t = tidy([record({1: flat((10, 10)), 2: flat((100, 99))}, chats=chats, notes={(2, 1): {"B": notes}})])
    per = C.code_periods(C.code_messages(t.chat), t.chat, t.firm_period)
    assert (per.iloc[0]["deception_notes"] == "B") is flagged


def test_chat_module_reexports_stated_price():
    assert C.stated_own_price("I will set 100. B is setting 99.") == 100


def _scores(rows):
    tp = fp = fn = 0
    for r in rows:
        got, lab = stated_own_price(r["notes"], r["period"]), r["label_stated_price"]
        if got is not None:
            tp, fp = tp + (got == lab), fp + (got != lab)
        fn += lab is not None and got != lab
    return tp / (tp + fp), tp / (tp + fn)


def test_hand_labelled_notes_precision_and_recall():
    """>= 60 hand-labelled notes from the pilots (both models): precision >= 0.9 and recall >= 0.7 on the
    stated this-period price, overall and on the held-out split; the deviation plan is precise."""
    assert len(LABELS) >= 60 and {r["model"] for r in LABELS} == {"gpt-oss-20b", "qwen3-14b"}
    for rows in (LABELS, [r for r in LABELS if r["split"] == "heldout"]):
        precision, recall = _scores(rows)
        assert precision >= 0.9 and recall >= 0.7, (precision, recall)
    got = [deviation_plan(r["notes"], r["period"]) for r in LABELS]
    lab = [r["label_deviation_plan"] for r in LABELS]
    tp = sum(g and y for g, y in zip(got, lab))
    assert tp / sum(got) >= 0.9 and tp / sum(lab) >= 0.6
