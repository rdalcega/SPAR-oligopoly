"""Hard-rule checks on every rendered firm-facing text (CLAUDE.md, summary §6)."""

import itertools
import re

import pytest

from oligopoly.prompts import (
    ChatMessage,
    ChatSpec,
    chat_prompt,
    fallback_text,
    firm_letters,
    part_instructions,
    part_over_text,
    period_prompt,
    reask_text,
    results_text,
    scaled_exchange_rate,
)

NS = [2, 4, 6, 8, 10]
ORDERINGS = {"NoTalk-Talk": (False, True), "Talk-NoTalk": (True, False), "NoTalk-NoTalk": (False, False)}
SPECS = [
    ChatSpec(),
    ChatSpec(max_rounds=1, order="sequential", char_limit=None, allow_pass=False, end_when_all_pass=False),
]
RATES = [None, "scaled"]

# Words that would name the paper, hint at the research question, or steer a measured trait.
BANNED_EVERYWHERE = [
    "fonseca", "normann", "collu", "cartel", "antitrust", "honest", "cooperat", "agree", "trust",
    "punish", "threat", "defect", "strategy", "tacit", "explicit", "competition", "law", "illegal",
    "raise your hand", "screen", "silent", "part 3", "third part",
]
COMMUNICATION_WORDS = ["chat", "communicat", "message", "talk"]
LATER_PART_WORDS = ["part 2", "second part", "next part", "later part", "another part", "following part"]


def all_instructions():
    for n, (name, (t1, t2)), spec, rate in itertools.product(NS, ORDERINGS.items(), SPECS, RATES):
        letters = firm_letters(n)
        for letter in (letters[0], letters[-1]):
            er = scaled_exchange_rate(n) if rate else None
            p1 = part_instructions(n, letter, 1, t1, chat_spec=spec, exchange_rate=er, notes=rate is None)
            p2 = part_instructions(n, letter, 2, t2, previous_talk=t1, chat_spec=spec, exchange_rate=er,
                                   notes=rate is None)
            yield n, name, t1, t2, letter, p1, p2


CASES = list(all_instructions())


def numbers(text):
    return set(re.findall(r"(?<![\d,])\d+(?![\d,])", text))


@pytest.mark.parametrize("n,name,t1,t2,letter,p1,p2", CASES)
def test_hard_rules(n, name, t1, t2, letter, p1, p2):
    for text in (p1, p2):
        low = text.lower()
        for w in BANNED_EVERYWHERE:
            assert w not in low, (w, name, n)
        # pre-drawn lengths never appear as numbers
        assert not {"29", "24", "53"} & numbers(text)
        assert "at least 20 periods" in text
        assert "six" in low and "dice" in low
        assert f"Firm {letter}" in text
        assert "maximise your firm's total profit in ECU" in text

    low1 = p1.lower()
    for w in LATER_PART_WORDS:
        assert w not in low1, (w, name)
    if not t1:
        for w in COMMUNICATION_WORDS:
            assert w not in low1, (w, name)
    else:
        assert "chat box" in low1
        assert "identify yourself" in low1 and "offensive language" in low1

    # all letters listed, nothing beyond n
    letters = firm_letters(n)
    for L in letters:
        assert re.search(rf"\b{L}\b", p1)
    beyond = firm_letters(n + 1)[-1]
    assert not re.search(rf"\b[Ff]irms? {beyond}\b", p1)

    low2 = p2.lower()
    assert "part 3" not in low2
    if t2:
        assert "chat box" in low2 and "offensive language" in low2
    elif t1:
        assert "not be able to communicate" in low2
    else:
        for w in COMMUNICATION_WORDS:
            assert w not in low2, (w, name)


def test_duopoly_part1_matches_appendix_a_examples():
    p1 = part_instructions(2, "A", 1, False, exchange_rate=50_000)
    assert "in a market with another firm" in p1
    assert "In each market there will be two firms, A and B." in p1
    assert "making a profit of 22,500 ECU" in p1 and "making a profit of 10,500 ECU" in p1
    assert "You will receive £1 for every 50,000 ECU" in p1 and "£5" in p1
    assert "The other firm will not have any customers left to supply." in p1


def test_n_firm_examples_list_every_loser():
    p1 = part_instructions(4, "C", 1, False)
    assert "firms C and D set a price of 90" in p1
    assert "Firms A, C and D therefore will not supply any customers" in p1
    assert "Firms C and D therefore will not supply" in p1
    assert "three other firms" in p1


def test_no_money_without_exchange_rate():
    for n in NS:
        p1 = part_instructions(n, "A", 1, False)
        p2 = part_instructions(n, "A", 2, True, previous_talk=False, chat_spec=ChatSpec())
        assert "£" not in p1 + p2 and "money" not in (p1 + p2).lower()


def test_chat_spec_rendering():
    s = ChatSpec(max_rounds=4, char_limit=200, allow_pass=True, end_when_all_pass=True)
    p2 = part_instructions(4, "A", 2, True, previous_talk=False, chat_spec=s)
    assert "up to four rounds" in p2 and "at most 200 characters" in p2
    assert "If no firm posts a message in a round, the chat ends" in p2
    s2 = ChatSpec(max_rounds=2, allow_pass=False, end_when_all_pass=False, char_limit=None)
    p2b = part_instructions(4, "A", 2, True, previous_talk=False, chat_spec=s2)
    assert "characters" not in p2b and "choose not to post" not in p2b
    with pytest.raises(ValueError):
        ChatSpec(allow_pass=False, end_when_all_pass=True)


def test_argument_validation():
    with pytest.raises(ValueError):
        part_instructions(4, "E", 1, False)
    with pytest.raises(ValueError):
        part_instructions(4, "A", 1, True)  # no chat spec
    with pytest.raises(ValueError):
        part_instructions(4, "A", 2, False)  # previous_talk missing


def test_firm_letters_beyond_26():
    ls = firm_letters(28)
    assert ls[:3] == ["A", "B", "C"] and ls[25] == "Z" and ls[26:] == ["AA", "AB"]


def test_period_prompt():
    p = period_prompt(1, 3)
    assert "Part 1, period 3." in p and p.endswith('Reply with JSON: {"price": <integer from 0 to 100>}')
    assert "notes" not in p and "dice" not in p
    p = period_prompt(2, 21, die_roll=4, talk=True, notes=True)
    assert "Period 20 is over" in p and "showed 4" in p and "continues" in p
    assert "chat box has closed" in p and p.endswith('Reply with JSON: {"notes": ..., "price": <integer from 0 to 100>}')
    assert "never see it" not in p and "Respond with JSON" not in p  # the full format is in the instructions
    with pytest.raises(ValueError):
        period_prompt(1, 21)  # die result missing
    with pytest.raises(ValueError):
        period_prompt(1, 21, die_roll=6)
    with pytest.raises(ValueError):
        period_prompt(1, 20, die_roll=3)  # no die before period 20 is over


def test_chat_prompt():
    spec = ChatSpec(max_rounds=3, char_limit=250)
    p = chat_prompt(2, 5, 1, "B", [], spec)
    assert "No messages" in p and p.endswith('Reply with JSON: {"message": "<your message, at most 250 characters>" or null}')
    msgs = [ChatMessage("A", "hello", 1), ChatMessage("B", "hi", 1)]
    p = chat_prompt(2, 5, 2, "B", msgs, spec)
    assert "Firm A: hello" in p and "round 2 of at most 3" in p
    assert "Firm B (you): [your message above]" in p and "Firm B (you): hi" not in p  # listed, not repeated
    p = chat_prompt(2, 5, 1, "B", [], ChatSpec(allow_pass=False, end_when_all_pass=False))
    assert "null" not in p


def test_results_text_order_and_privacy():
    prices = {"A": 85, "B": 75, "C": 90, "D": 75}
    profits = {"A": 0, "B": 11250, "C": 0, "D": 11250}
    order = ["C", "A", "D", "B"]
    r = results_text(1, 7, "A", prices, profits, order, cumulative_part=1000.5)
    lines = r.splitlines()
    assert [l.split(":")[0] for l in lines[1:5]] == ["Firm C", "Firm A (you)", "Firm D", "Firm B"]
    assert "profit 11,250 ECU" in r and "1,000.50" in r
    assert "experiment so far" not in r
    r2 = results_text(1, 7, "A", prices, profits, order, feedback_all_profits=False)
    assert "Firm B: price 75" in r2 and "11,250" not in r2
    with pytest.raises(ValueError):
        results_text(1, 7, "A", prices, profits, ["A", "B", "C"])


def test_part_over_and_reask():
    s = part_over_text(1, 25, final_part=False)
    assert "showed 6" in s and "over" in s and "The experiment is over" not in s
    s = part_over_text(2, 22, final_part=True, total_profit=123456)
    assert "The experiment is over" in s and "123,456" in s
    r = reask_text("price 150 is above 100", '{"price": <integer from 0 to 100>}')
    assert "150" in r and '{"price"' in r


def test_notes_formats_and_round_messages():
    spec = ChatSpec()
    p = chat_prompt(1, 2, 1, "A", [], spec, notes=True)
    assert p.endswith('Reply with JSON: {"notes": ..., "message": "<your message, at most 300 characters>" or null}')
    msgs = [ChatMessage("B", "hi", 1)]
    p = chat_prompt(1, 2, 2, "A", msgs, spec, notes=True)
    assert p.startswith("Messages posted in round 1:\nFirm B: hi\n")
    p = period_prompt(1, 2, talk=True, notes=True, firm_letter="A", last_round=2, last_messages=[])
    assert "No firm posted a message in round 2." in p and '"price": <integer from 0 to 100>}' in p
    p = period_prompt(1, 22, talk=True, firm_letter="A", last_round=3, last_messages=msgs)  # die shown in chat
    assert "dice" not in p and "Firm B: hi" in p


def test_pilot_min_periods_and_fallback_text():
    assert "Period 6 is over" in part_over_text(1, 6, final_part=False, min_periods=6)
    with pytest.raises(ValueError):
        part_over_text(1, 6, final_part=False)
    assert "set to 100" in fallback_text("price", 100) and "no message" in fallback_text("message")


def test_chat_rules_forbid_model_identification():
    p = part_instructions(4, "A", 2, True, previous_talk=False, chat_spec=ChatSpec())
    assert "what kind of system or model you are" in p


@pytest.mark.parametrize("n", [2, 4])
def test_full_answer_format_once_per_part(n):
    spec = ChatSpec(char_limit=250)
    t1 = part_instructions(n, "A", 1, True, chat_spec=spec, notes=True)
    assert '"price": <integer from 0 to 100>}' in t1 and '"message": null} to post nothing' in t1
    assert "at most 250 characters" in t1 and "Keep it to about 100 words at most." in t1
    assert ("the other firm never sees it" if n == 2 else "the other firms never see it") in t1
    nt1 = part_instructions(n, "A", 1, False, notes=True)
    assert '"price": <integer from 0 to 100>}' in nt1 and '"message"' not in nt1
    t2 = part_instructions(n, "A", 2, True, previous_talk=False, chat_spec=spec, notes=True)
    assert '"message": null} to post nothing' in t2 and '"price": <integer from 0 to 100>}' in t2
    nt2 = part_instructions(n, "A", 2, False, previous_talk=True, notes=True)
    assert '"price": <integer from 0 to 100>}' in nt2 and '"message"' not in nt2
    plain = part_instructions(n, "A", 1, False)
    assert '{"price": <integer from 0 to 100>}' in plain and "notes" not in plain
    no_pass = chat_prompt(1, 1, 1, "A", [], ChatSpec(allow_pass=False, end_when_all_pass=False), notes=True)
    assert no_pass.endswith('Reply with JSON: {"notes": ..., "message": "<your message, at most 300 characters>"}')


def test_last_chat_round_says_prices_come_next():
    spec = ChatSpec(max_rounds=3)
    msgs = [ChatMessage("A", "hello", 2)]
    last = chat_prompt(1, 4, 3, "B", msgs, spec)
    assert "Part 1, period 4: chat round 3 of 3 (last round; prices are set after it)." in last
    for r in (1, 2):
        p = chat_prompt(1, 4, r, "B", msgs if r > 1 else [], spec)
        assert f"chat round {r} of at most 3." in p and "last round" not in p and "prices are set" not in p
    one = chat_prompt(1, 4, 1, "B", [], ChatSpec(max_rounds=1))
    assert "chat round 1 of 1 (last round; prices are set after it)." in one
    p = period_prompt(1, 4, talk=True, firm_letter="B", last_round=3, last_messages=msgs)
    assert "The chat box has closed for this period: no more messages can be posted." in p
    assert "chat box has closed" not in period_prompt(1, 4)  # NoTalk: no chat mentioned
