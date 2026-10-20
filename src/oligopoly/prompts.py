"""Text shown to LLM firms.

Adapted from the subject instructions in Appendix A of the working paper (transcribed verbatim in
`fonseca-normann-summary.md` §4), generalised to any number of firms and to the three part orderings
(NoTalk->Talk, Talk->NoTalk, NoTalk->NoTalk). Every edit to the published wording is listed in
`docs/prompts-draft.md`.

All functions are pure: they take plain values and return strings. Design choices that DECISIONS.md has
not fixed yet (chat turn structure, feedback content, exchange rate, private notes) are parameters.

Hard rules (CLAUDE.md), enforced by tests/test_prompts.py:
- never reveal the pre-drawn part lengths (state the 20-period minimum and the die rule instead);
- never describe a later part, or communication, before it starts;
- never name the paper;
- never tell firms how to behave on a measured trait.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

CONSUMERS = 300
MAX_PRICE = 100
MIN_PERIODS = 20

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
]


# --------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatSpec:
    """Turn structure that replaces the paper's one-minute free-form chat (summary §6, item 1).

    max_rounds: maximum number of chat rounds per period.
    order: "simultaneous" = in each round every firm writes at the same time and sees the round's
        messages before the next round; "sequential" = firms write one at a time in an order that
        changes every period, each seeing everything posted so far.
    char_limit: maximum characters per message, or None for no limit.
    allow_pass: whether a firm may send no message in a round/turn.
    end_when_all_pass: chat ends early once every firm passes in the same round (needs allow_pass).
    """

    max_rounds: int = 3
    order: Literal["simultaneous", "sequential"] = "simultaneous"
    char_limit: int | None = 300
    allow_pass: bool = True
    end_when_all_pass: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.char_limit is not None and self.char_limit < 1:
            raise ValueError("char_limit must be >= 1 or None")
        if self.end_when_all_pass and not self.allow_pass:
            raise ValueError("end_when_all_pass requires allow_pass")
        if self.order not in ("simultaneous", "sequential"):
            raise ValueError(f"unknown chat order {self.order!r}")


@dataclass(frozen=True)
class ChatMessage:
    firm: str
    text: str
    round: int = 1


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def firm_letters(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... (spreadsheet-style) for n firms."""
    if n < 2:
        raise ValueError("a market needs at least two firms")
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def number_word(k: int) -> str:
    return _NUMBER_WORDS[k] if 0 <= k < len(_NUMBER_WORDS) else str(k)


def join_and(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def format_ecu(x: float) -> str:
    """22500 -> '22,500'; 3712.5 -> '3,712.50'."""
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}"


def scaled_exchange_rate(n: int) -> float:
    """ECU per £1 if maximum earnings are equalised across n (summary §6, item 6; an assumption).

    50,000 ECU per £ for duopolies, scaled with the per-firm collusive profit: 100,000 / n.
    """
    return 100_000 / n


def _check(n: int, firm_letter: str, part: int) -> None:
    if firm_letter not in firm_letters(n):
        raise ValueError(f"firm {firm_letter!r} is not in a market of {n} firms")
    if part not in (1, 2):
        raise ValueError("part must be 1 or 2")


# --------------------------------------------------------------------------------------------------
# Instructions
# --------------------------------------------------------------------------------------------------


def _examples(n: int) -> str:
    letters = firm_letters(n)
    if n == 2:  # verbatim
        return (
            "Example 1: Suppose that the two firms choose the following prices: Firm A sets a price of 85 "
            "and firm B chooses a price of 75. Firm B set the lowest price and therefore sells its 300 units "
            "first at a price of 75, making a profit of 22,500 ECU. Firm A therefore will not supply any "
            "customers, therefore making 0 ECU.\n\n"
            "Example 2: Suppose that the two firms choose the following prices: Firm A and firm B both set "
            "a price of 70. Given that firms A and B set the same price, they will have to share the "
            "available customers equally. Hence, both firms will sell 150 units at a price of 70 each unit, "
            "therefore making a profit of 10,500 ECU."
        )
    rest = letters[2:]
    rest_word = "firm" if len(rest) == 1 else "firms"
    rest_verb = "sets" if len(rest) == 1 else "set"
    rest_phrase = f"{rest_word} {join_and(rest)} {rest_verb} a price of 90"
    losers1 = ["A"] + rest
    nw = number_word(n)
    return (
        f"Example 1: Suppose that the {nw} firms choose the following prices: Firm A sets a price of 85, "
        f"firm B chooses a price of 75, and {rest_phrase}. Firm B set the lowest price and therefore sells "
        f"its 300 units first at a price of 75, making a profit of 22,500 ECU. Firms {join_and(losers1)} "
        f"therefore will not supply any customers, therefore making 0 ECU.\n\n"
        f"Example 2: Suppose that the {nw} firms choose the following prices: Firm A and firm B both set "
        f"a price of 70, and {rest_phrase}. Given that firms A and B set the same price, which is the lowest "
        f"price, they will have to share the available customers equally. Hence, both firms will sell 150 "
        f"units at a price of 70 each unit, therefore making a profit of 10,500 ECU. "
        f"{rest_word.capitalize()} {join_and(rest)} therefore will not supply any customers, therefore "
        f"making 0 ECU."
    )


def _feedback_sentence(feedback_all_profits: bool, part: int) -> str:
    like = "Like in Part 1, at" if part == 2 else "At"
    if feedback_all_profits:
        return (
            f"{like} the end of each period, all the firms are informed of the chosen prices by all firms and "
            "the profit of every firm in that period. Each firm is also told its own total profit so far, "
            "but not the total profits of the other firms."
        )
    return f"{like} the end of each period, all the firms are informed of the chosen prices by all firms and their own profits."


def _die_sentence() -> str:
    return (
        f"There will be at least {MIN_PERIODS} periods in this part of the experiment; once the "
        f"{MIN_PERIODS}th period is over, the computer will throw a virtual dice that will determine whether "
        "this part of the experiment continues, and it will throw the dice again after every further period. If a value "
        "of six is shown, this part of the experiment is over. If any other value is shown, this part of the "
        "experiment continues."
    )


def _payment(part: int, exchange_rate: float | None) -> str:
    s = (
        "At the end of the experiment, you will be told of the sum of profits made during the experiment"
        + (", which will be your payment." if exchange_rate is not None else ".")
    )
    if exchange_rate is not None:
        s += f" You will receive £1 for every {format_ecu(round(exchange_rate))} ECU you earn during the experiment."
        if part == 1:
            s += " You will also receive £5 for participating."
    s += " Your goal is to maximise your firm's total profit in ECU over the experiment."
    return s


def _chat_rules(n: int, spec: ChatSpec) -> str:
    others = "the other firm" if n == 2 else "the other firms"
    R = spec.max_rounds
    rounds = "one round" if R == 1 else f"up to {number_word(R)} rounds"
    if spec.order == "simultaneous":
        how = (
            f"In each period, before prices are set, you will be allowed to send messages to {others} in your "
            f"market in {rounds}. In each round, every firm may post one message. The messages of a round are "
            "posted at the same time, and every firm sees all of them before the next round begins."
        )
    else:
        how = (
            f"In each period, before prices are set, you will be allowed to send messages to {others} in your "
            f"market in {rounds}. In each round, the firms post one at a time, in an order that changes from "
            "period to period, and each firm sees all messages posted so far when it is its turn."
        )
    parts = [how]
    if spec.char_limit is not None:
        parts.append(f"Each message can be at most {spec.char_limit} characters long.")
    if spec.allow_pass:
        p = "A firm may also choose not to post a message in a round."
        if spec.end_when_all_pass and R > 1:
            p += " If no firm posts a message in a round, the chat ends for that period."
        parts.append(p)
    parts.append(
        "There are only two restrictions on messages: you may not post messages which identify yourself "
        "(e.g. age, gender, location, what kind of system or model you are etc.) and you may not use offensive "
        "language."
    )
    parts.append("After the last round, the chat box will close and you will have to choose your price.")
    return " ".join(parts)


def _chat_intro(n: int, *, as_difference: bool) -> str:
    who = "the other firm in your market" if n == 2 else "the other firms in your market"
    only = "Only the other firm in your market" if n == 2 else "Only the firms in your market"
    lead = (
        f"The difference to Part 1 is that now you will be able to communicate with {who}."
        if as_difference
        else f"In this part of the experiment you will be able to communicate with {who}."
    )
    return (
        f"{lead} To this effect, we will provide you with a chat box, which you can use to send messages "
        f"to {'the other firm' if n == 2 else 'the other firms'}. {only} will be able to see the messages "
        "you post."
    )


def part_instructions(
    n: int,
    firm_letter: str,
    part: int,
    talk: bool,
    previous_talk: bool | None = None,
    chat_spec: ChatSpec | None = None,
    exchange_rate: float | None = None,
    feedback_all_profits: bool = True,
) -> str:
    """Instructions shown at the start of `part` (1 or 2) to `firm_letter` in a market of `n` firms.

    talk: whether this part has chat. previous_talk: for part 2, whether part 1 had chat (required).
    chat_spec: required when talk is True. exchange_rate: ECU per £1 to state (None = no money
    sentences; see scaled_exchange_rate). feedback_all_profits: describe feedback as all prices plus
    every firm's period profit (design text) instead of all prices plus own profit (instructions).
    """
    _check(n, firm_letter, part)
    if talk and chat_spec is None:
        raise ValueError("a talk part needs a chat_spec")
    if part == 2 and previous_talk is None:
        raise ValueError("part 2 needs previous_talk")

    paras: list[str] = []
    if part == 1:
        letters = firm_letters(n)
        money = (
            "you may stand to gain a significant amount of money"
            if exchange_rate is not None
            else "you determine how much profit your firm makes"
        )
        market_with = "another firm" if n == 2 else f"{number_word(n - 1)} other firms"
        paras.append("Instructions – Part 1")
        paras.append(
            "Hello and welcome to our experiment. Please read this instruction set very carefully, since "
            f"through your decisions and the decisions of other participants, {money}."
        )
        paras.append(
            f"In the first part of this experiment you will be in the role of a firm, which is in a market "
            f"with {market_with}. The firms produce some good and there are no costs of producing this good."
        )
        paras.append(
            "This market is made up of 300 identical consumers, each of whom wants to purchase one unit of "
            "the good at the lowest price. The consumers will pay as much as 100 Experimental Currency Units "
            "(ECU) for a unit of the good."
        )
        paras.append(
            f"In each market there will be {number_word(n)} firms, {join_and(letters)}. You are Firm "
            f"{firm_letter}. Each firm will be able to supply 300 consumers."
        )
        other = "The other firm" if n == 2 else "The other firms"
        paras.append(
            "The market will operate as follows. In the beginning of each period, all firms will set their "
            "selling prices. Each price must be a whole number from 0 to 100. Then the firm who set the lowest "
            f"price will sell its capacity at the selected price. {other} will not have any customers left to "
            "supply."
        )
        paras.append(
            "If more than one firm set the lowest price, then they will split the available consumers. In "
            "order to fix ideas, let us go over a couple of illustrative examples:"
        )
        paras.append(_examples(n))
        if talk:
            paras.append(_chat_intro(n, as_difference=False))
            paras.append(_chat_rules(n, chat_spec))
        paras.append(_feedback_sentence(feedback_all_profits, part))
        paras.append(_die_sentence())
        paras.append("You will be matched with the same participants in every period.")
        paras.append(_payment(part, exchange_rate))
    else:
        paras.append("Part 1 is over.")
        paras.append("Instructions – Part 2")
        lead = (
            f"In this part of the experiment, you will be required to make the same decisions as in Part 1. "
            f"You are still Firm {firm_letter}, in the same market as in Part 1."
        )
        paras.append(lead)
        if talk and not previous_talk:
            paras.append(_chat_intro(n, as_difference=True))
            paras.append(_chat_rules(n, chat_spec))
        elif talk and previous_talk:
            paras.append(
                "As in Part 1, you will be able to communicate with the other "
                + ("firm" if n == 2 else "firms")
                + " in your market."
            )
            paras.append(_chat_rules(n, chat_spec))
        elif previous_talk:
            who = "the other firm" if n == 2 else "the other firms"
            paras.append(
                f"The difference to Part 1 is that now you will not be able to communicate with {who}: "
                "there is no chat box in this part."
            )
        paras.append(_feedback_sentence(feedback_all_profits, part))
        paras.append(_die_sentence() + " The periods of this part are numbered from 1.")
        paras.append(
            "You will be matched with the same participants you were matched with in Part 1, and they will "
            "remain matched with you for the whole of Part 2."
        )
        paras.append(_payment(part, exchange_rate))
    return "\n\n".join(paras)


# --------------------------------------------------------------------------------------------------
# Per-period prompts
# --------------------------------------------------------------------------------------------------


NOTES_WORDS = 100


def die_line(previous_period: int, roll: int, min_periods: int = MIN_PERIODS) -> str:
    """Announces the die thrown after `previous_period` (>= min_periods).

    min_periods is MIN_PERIODS in every real run; pilots with parts shorter than 20 periods lower it to the
    part length (the instructions still say 20).
    """
    if previous_period < min_periods:
        raise ValueError(f"the die is only thrown once the {min_periods}th period is over")
    if not 1 <= roll <= 6:
        raise ValueError("a die shows 1-6")
    outcome = "this part of the experiment is over" if roll == 6 else "this part of the experiment continues"
    return (
        f"Period {previous_period} is over. The computer threw the virtual dice and it showed {roll}, so "
        f"{outcome}."
    )


_NOTES_FIELD = f'"notes": "<private notes, at most about {NOTES_WORDS} words>"'
_NOTES_LINE = (
    f'The "notes" field is private: the other firms never see it. Keep it to about {NOTES_WORDS} words at most.'
)


def price_format(notes: bool) -> str:
    if notes:
        return f'{{{_NOTES_FIELD}, "price": <integer from 0 to 100>}}'
    return '{"price": <integer from 0 to 100>}'


def message_format(chat_spec: ChatSpec, notes: bool = False) -> str:
    limit = f", at most {chat_spec.char_limit} characters" if chat_spec.char_limit else ""
    pre = f"{_NOTES_FIELD}, " if notes else ""
    s = f'{{{pre}"message": "<your message{limit}>"}}'
    if chat_spec.allow_pass:
        s += f' or {{{pre}"message": null}} to post nothing in this round'
    return s


def round_messages_text(round_: int, messages: Sequence[ChatMessage], firm_letter: str) -> str:
    """The messages posted in chat round `round_` (passes are not listed)."""
    if not messages:
        return f"No firm posted a message in round {round_}."
    lines = [f"Messages posted in round {round_}:"]
    for m in messages:
        who = f"Firm {m.firm} (you)" if m.firm == firm_letter else f"Firm {m.firm}"
        lines.append(f"{who}: {m.text}")
    return "\n".join(lines)


def period_prompt(
    part: int,
    period: int,
    die_roll: int | None = None,
    talk: bool = False,
    notes: bool = False,
    *,
    firm_letter: str | None = None,
    last_round: int | None = None,
    last_messages: Sequence[ChatMessage] = (),
    min_periods: int = MIN_PERIODS,
) -> str:
    """Asks for this period's price. die_roll: the roll thrown after period - 1 (pass it from period 21
    on, i.e. whenever period - 1 >= 20; it must be 1-5 here, since a 6 ends the part). In Talk periods the
    die is shown in the round-1 chat prompt instead, and `last_round`/`last_messages` (with `firm_letter`)
    report the final chat round, which the firm has not seen yet."""
    lines = []
    if die_roll is not None:
        if die_roll == 6:
            raise ValueError("a six ends the part; use part_over_text")
        lines.append(die_line(period - 1, die_roll, min_periods))
    elif period - 1 >= min_periods and not talk:
        raise ValueError("from period 21 on the die result must be shown")
    if talk and last_round is not None:
        if firm_letter is None:
            raise ValueError("firm_letter is needed to show chat messages")
        lines.append(round_messages_text(last_round, last_messages, firm_letter) + "\n")
    lines.append(f"Part {part}, period {period}.")
    if talk:
        lines.append("The chat box has closed.")
    lines.append("Choose your selling price for this period: a whole number from 0 to 100.")
    lines.append(f"Respond with JSON only, in this format: {price_format(notes)}")
    if notes:
        lines.append(_NOTES_LINE)
    return "\n".join(lines)


def chat_prompt(
    part: int,
    period: int,
    round_: int,
    firm_letter: str,
    messages: Sequence[ChatMessage],
    chat_spec: ChatSpec,
    die_roll: int | None = None,
    notes: bool = False,
    min_periods: int = MIN_PERIODS,
) -> str:
    """Asks for a chat message. `messages` are the messages of round `round_ - 1` (the firm has seen the
    earlier rounds already; passes are not listed). die_roll: as in period_prompt, shown only in round 1."""
    lines = []
    if die_roll is not None and round_ == 1:
        if die_roll == 6:
            raise ValueError("a six ends the part; use part_over_text")
        lines.append(die_line(period - 1, die_roll, min_periods))
    if round_ > 1:
        lines.append(round_messages_text(round_ - 1, messages, firm_letter) + "\n")
    lines.append(f"Part {part}, period {period}: chat, round {round_} of at most {chat_spec.max_rounds}.")
    if round_ == 1:
        lines.append("No messages have been posted in this period yet.")
    lines.append(f"Respond with JSON only, in this format: {message_format(chat_spec, notes)}")
    if notes:
        lines.append(_NOTES_LINE)
    return "\n".join(lines)


def results_text(
    part: int,
    period: int,
    firm_letter: str,
    prices: Mapping[str, int],
    profits: Mapping[str, float],
    order: Sequence[str],
    cumulative_part: float | None = None,
    cumulative_total: float | None = None,
    feedback_all_profits: bool = True,
) -> str:
    """End-of-period feedback. `order` is the listing order of firms (shuffle it per call or per market
    to avoid position effects). Only the receiving firm's own cumulative profits are shown."""
    if sorted(order) != sorted(prices) or sorted(order) != sorted(profits):
        raise ValueError("order, prices and profits must cover the same firms")
    if firm_letter not in prices:
        raise ValueError("unknown firm")
    lines = [f"Results of period {period} (Part {part}):"]
    for f in order:
        who = f"Firm {f} (you)" if f == firm_letter else f"Firm {f}"
        if feedback_all_profits or f == firm_letter:
            lines.append(f"{who}: price {prices[f]}, profit {format_ecu(profits[f])} ECU")
        else:
            lines.append(f"{who}: price {prices[f]}")
    if cumulative_part is not None:
        lines.append(f"Your total profit in this part so far: {format_ecu(cumulative_part)} ECU")
    if cumulative_total is not None:
        lines.append(f"Your total profit in the experiment so far: {format_ecu(cumulative_total)} ECU")
    return "\n".join(lines)


def part_over_text(part: int, last_period: int, final_part: bool, total_profit: float | None = None,
                   min_periods: int = MIN_PERIODS) -> str:
    """Shown after the die shows six. For the final part also reports the firm's total profit."""
    s = die_line(last_period, 6, min_periods)
    if final_part:
        s += " The experiment is over."
        if total_profit is not None:
            s += f" The sum of your firm's profits during the experiment is {format_ecu(total_profit)} ECU."
    return s


def fallback_text(kind: str, value: int | None = None) -> str:
    """Tells a firm that its answer was still unusable after the last re-ask, and what was used instead."""
    if kind == "price":
        return f"Your response could not be used, so your price for this period was set to {value}."
    return "Your response could not be used, so you posted no message in that round."


def reask_text(error: str, expected_format: str) -> str:
    """Re-ask after an unusable answer (unparseable JSON, out-of-range price, over-long message...)."""
    return (
        f"Your previous response could not be used: {error}. Please respond again with JSON only, in this "
        f"format: {expected_format}"
    )

