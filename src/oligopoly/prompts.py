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

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

CONSUMERS = 300
MAX_PRICE = 100
MIN_PERIODS = 20
NOTES_WORDS = 500  # default stated length of the notes (`notes_words` in config/experiment.toml)

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
    notes: bool = False,
    memory_window: int | None = None,
    notes_words: int = NOTES_WORDS,
) -> str:
    """Instructions shown at the start of `part` (1 or 2) to `firm_letter` in a market of `n` firms.

    talk: whether this part has chat. previous_talk: for part 2, whether part 1 had chat (required).
    chat_spec: required when talk is True. exchange_rate: ECU per £1 to state (None = no money
    sentences; see scaled_exchange_rate). feedback_all_profits: describe feedback as all prices plus
    every firm's period profit (design text) instead of all prices plus own profit (instructions).
    notes: whether answers carry the private "notes" field. Each part's instructions end with the full answer
    format (see response_format_text); the per-turn prompts only repeat a one-line reminder.
    memory_window: the number of completed periods each request shows in full (DECISIONS.md §6 item 3,
    2026-10-20), stated with the answer format; None = the whole history (nothing is said).
    notes_words: the stated maximum length of the notes.
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
        paras.append(response_format_text(n, chat_spec if talk else None, notes, memory_window, notes_words))
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
        paras.append(response_format_text(n, chat_spec if talk else None, notes, memory_window, notes_words))
    return "\n\n".join(paras)


# --------------------------------------------------------------------------------------------------
# Per-period prompts
# --------------------------------------------------------------------------------------------------




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


def _notes_field(notes_words: int) -> str:
    return f'"notes": "<private notes, at most about {notes_words} words>"'


def price_format(notes: bool, notes_words: int = NOTES_WORDS) -> str:
    if notes:
        return f'{{{_notes_field(notes_words)}, "price": <integer from 0 to 100>}}'
    return '{"price": <integer from 0 to 100>}'


def message_format(chat_spec: ChatSpec, notes: bool = False, notes_words: int = NOTES_WORDS) -> str:
    limit = f", at most {chat_spec.char_limit} characters" if chat_spec.char_limit else ""
    pre = f"{_notes_field(notes_words)}, " if notes else ""
    s = f'{{{pre}"message": "<your message{limit}>"}}'
    if chat_spec.allow_pass:
        s += f' or {{{pre}"message": null}} to post nothing in this round'
    return s


def response_format_text(n: int, chat_spec: ChatSpec | None, notes: bool, memory_window: int | None = None,
                         notes_words: int = NOTES_WORDS) -> str:
    """The full answer format, stated once per part at the end of its instructions. chat_spec: the part's chat
    rules, or None in a NoTalk part (then nothing about messages is said). memory_window: see
    part_instructions; with it, the notes are described as the firm's private memory (mechanics only)."""
    s = f"How to answer: when you are asked for your selling price, respond with JSON only, in this format: " \
        f"{price_format(notes, notes_words)}"
    if chat_spec is not None:
        s += (f"\nWhen you are asked for a chat message, respond with JSON only, in this format: "
              f"{message_format(chat_spec, notes, notes_words)}.")
    others = "the other firm never sees it" if n == 2 else "the other firms never see it"
    if memory_window is not None:
        s += "\n" + window_text(memory_window, chat_spec is not None, notes)
        if notes:
            s += (f'\nThe "notes" field is private: {others}. Your notes are shown to you at your next decision, '
                  "where they replace your previous notes, so write in them everything you want to remember. Keep "
                  f"them to about {notes_words} words at most.")
    elif notes:
        s += f'\nThe "notes" field is private: {others}. Keep it to about {notes_words} words at most.'
    s += "\nEach request ends with a short reminder of the format."
    return s


def window_text(memory_window: int, talk: bool, notes: bool = True) -> str:
    """What each request shows (DECISIONS.md §6 item 3, summaries layout). Mechanics only."""
    if memory_window < 1:
        raise ValueError("memory_window must be >= 1")
    k = "the last period" if memory_window == 1 else f"the last {number_word(memory_window)} periods"
    shown = [f"the results of {k}", "your total profit"]
    if notes:
        shown.append("your notes")
    if talk:
        shown.append("the chat messages posted so far in the current period")
    earlier = "Earlier periods, earlier chat messages and your earlier answers" if talk else \
        "Earlier periods and your earlier answers"
    return f"Each request shows, in this order, {join_and(shown)}. {earlier} are not shown again."


# --------------------------------------------------------------------------------------------------
# Request sections (DECISIONS.md §6 item 3, summaries layout): after the instructions, every request is one
# user message made of these sections in order, then the prompt
# --------------------------------------------------------------------------------------------------

NO_PERIODS_YET = "No periods have been played yet."
NO_NOTES_YET = "You have not written any notes yet."
NO_MESSAGES_YET = "No messages have been posted in this period yet."


def summaries_text(summaries: Sequence[str], totals: str) -> str:
    """The summaries of the last completed periods (oldest first), then the firm's own cumulative profit."""
    if not summaries:
        return f"{NO_PERIODS_YET}\n{totals}"
    return "\n\n".join([*summaries, totals])


def totals_text(part: int, cumulative_part: float, cumulative_total: float | None = None) -> str:
    """The firm's own cumulative profit at the time of the request (Part 2 also shows the experiment total)."""
    lines = [f"Your total profit in this part so far: {format_ecu(cumulative_part)} ECU"]
    if part == 2:
        if cumulative_total is None:
            raise ValueError("Part 2 shows the total over the experiment")
        lines.append(f"Your total profit in the experiment so far: {format_ecu(cumulative_total)} ECU")
    return "\n".join(lines)


def notes_text(notes: str | None) -> str:
    """The firm's latest private notes."""
    if notes is None:
        return NO_NOTES_YET
    return f"Your most recent notes:\n{notes if notes.strip() else '(empty)'}"


def chat_text(rounds: Sequence[str]) -> str:
    """The chat rounds of the current period so far (each from round_text)."""
    if not rounds:
        return NO_MESSAGES_YET
    return "Chat messages posted so far in this period:\n\n" + "\n\n".join(rounds)


NO_MESSAGE = "(no message)"
OWN_FAILED = "(no message: your response could not be used)"
OWN_MESSAGE = "[your message above]"  # older records only: the firm's own message was its answer just above


def round_text(round_: int, posted: Mapping[str, str], order: Sequence[str], firm_letter: str,
               own_failed: bool = False) -> str:
    """Chat round `round_` as `firm_letter` sees it: every firm in its listing order `order`, with its message or
    NO_MESSAGE (OWN_FAILED for the firm's own turn when its answer could not be used)."""
    lines = [f"Round {round_}:"]
    for f in order:
        if f == firm_letter:
            lines.append(f"Firm {f} (you): " + (OWN_FAILED if own_failed else posted.get(f, NO_MESSAGE)))
        else:
            lines.append(f"Firm {f}: {posted.get(f, NO_MESSAGE)}")
    return "\n".join(lines)


def price_reminder(notes: bool) -> str:
    """One-line format reminder closing every price prompt (the full format is in the instructions)."""
    pre = '"notes": ..., ' if notes else ""
    return f'Reply with JSON: {{{pre}"price": <integer from 0 to 100>}}'


def message_reminder(chat_spec: ChatSpec, notes: bool) -> str:
    """One-line format reminder closing every chat prompt (keeps the character limit in view)."""
    limit = f", at most {chat_spec.char_limit} characters" if chat_spec.char_limit else ""
    value = f'"<your message{limit}>"' + (" or null" if chat_spec.allow_pass else "")
    pre = '"notes": ..., ' if notes else ""
    return f'Reply with JSON: {{{pre}"message": {value}}}'


# Stated in every chat-turn prompt (2026-10-20, after the n = 10 pilot, where ~19% of chat turns ended as forced
# passes because the firm answered a chat turn with a price): restates timing already in the instructions.
CHAT_TURN_LINE = "This is a chat turn: prices cannot be set now; you will set your price after the chat closes."


def period_prompt(part: int, period: int, talk: bool = False, notes: bool = False) -> str:
    """Asks for this period's price (the die and, in Talk, the chat are in the sections above it)."""
    lines = [f"Part {part}, period {period}."]
    if talk:
        lines.append("The chat box has closed for this period: no more messages can be posted.")
    lines.append("Choose your selling price for this period.")
    lines.append(price_reminder(notes))
    return "\n".join(lines)


def chat_prompt(part: int, period: int, round_: int, chat_spec: ChatSpec, notes: bool = False) -> str:
    """Asks for a chat message in round `round_` (the earlier rounds are in the chat section above it)."""
    if round_ == chat_spec.max_rounds:
        lines = [f"Part {part}, period {period}: chat round {round_} of {chat_spec.max_rounds} "
                 "(last round; prices are set after it)."]
    else:
        lines = [f"Part {part}, period {period}: chat round {round_} of at most {chat_spec.max_rounds}."]
    lines.append(CHAT_TURN_LINE)
    lines.append(message_reminder(chat_spec, notes))
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
    to avoid position effects). Only the receiving firm's own cumulative profits are shown (the engine leaves
    them out: the summaries are followed by totals_text instead)."""
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


def fallback_text(value: int) -> str:
    """Tells a firm, in that period's summary, that its price answer was still unusable after the last re-ask."""
    return f"Your price response in this period could not be used, so your price was set to {value}."


CHAT_PRICE_ERROR = 'a price was given on a chat turn (no "message" field)'


def chat_price_reask_text(chat_spec: ChatSpec, notes: bool, notes_words: int = NOTES_WORDS) -> str:
    """Re-ask after a chat turn was answered with a price object (and no message object)."""
    s = "You answered with a price, but this is a chat turn. Prices are set after the chat closes. Post a message"
    s += ", or null to post nothing." if chat_spec.allow_pass else "."
    return s + f" Respond with JSON only, in this format: {message_format(chat_spec, notes, notes_words)}"


def reask_text(error: str, expected_format: str) -> str:
    """Re-ask after an unusable answer (unparseable JSON, out-of-range price, over-long message...)."""
    return (
        f"Your previous response could not be used: {error}. Please respond again with JSON only, in this "
        f"format: {expected_format}"
    )


# Post-game comprehension probe (DECISIONS.md, 2026-10-20): asked once after all play, in each firm's own
# conversation, so it cannot affect any decision. Neutral wording; field name -> question.
PROBE_QUESTIONS = {
    "max_consumer_price": "What is the highest price at which consumers would buy the good?",
    "best_common_price": "If all firms set the same price, which common price would give each firm the highest "
                         "profit in a period?",
    "last_period_prices": "What price did each firm set in the very last period?",
    "deviations": "In the last part, did any firm price differently from what it had said in the chat or had "
                  "agreed? If so, which firm, in which period, and how? If there was no chat, write 'no chat'.",
}


def probe_format(letters: Sequence[str]) -> str:
    prices = ", ".join(f'"{L}": <int>' for L in letters)
    return (f'{{"max_consumer_price": <integer>, "best_common_price": <integer>, '
            f'"last_period_prices": {{{prices}}}, "deviations": "<text>"}}')


def probe_prompt(letters: Sequence[str]) -> str:
    """The post-game probe: the questions, then the JSON format with one price per firm letter."""
    questions = "\n".join(f"- {k}: {q}" for k, q in PROBE_QUESTIONS.items())
    return (f"The experiment is over. Please answer these questions about it.\n{questions}\n"
            f"Reply with JSON: {probe_format(letters)}")


def texts_digest(chat_spec: ChatSpec, notes: bool, memory_window: int | None = None,
                 notes_words: int = NOTES_WORDS) -> str:
    """SHA-256 of a fixed sample of every kind of firm-facing text, rendered with the run's settings. It goes
    into every market's fingerprint, so a resumed run never mixes markets played under different wording."""
    posted = {"A": "m1", "B": "m2"}
    rounds = [round_text(1, posted, ["B", "A", "C"], "A"), round_text(2, {"B": "m3"}, ["B", "A", "C"], "A", True)]
    texts = [
        *(part_instructions(n, "A", p, t, previous_talk=pt, chat_spec=chat_spec, notes=notes,
                            memory_window=memory_window, notes_words=notes_words)
          for n in (2, 4) for t1, t2 in ((False, True), (True, False), (False, False))
          for p, t, pt in ((1, t1, None), (2, t2, t1))),
        period_prompt(1, 21, notes=notes),
        period_prompt(2, 5, talk=True, notes=notes),
        chat_prompt(1, 21, 1, chat_spec, notes=notes),
        chat_prompt(1, 5, chat_spec.max_rounds, chat_spec, notes=notes),
        results_text(2, 3, "A", {"A": 90, "B": 80}, {"A": 0, "B": 24000}, ["B", "A"]),
        die_line(21, 3), part_over_text(2, 22, final_part=True, total_profit=10),
        fallback_text(7),
        reask_text("e", price_format(notes, notes_words)), reask_text("e", message_format(chat_spec, notes, notes_words)),
        chat_price_reask_text(chat_spec, notes, notes_words),
        summaries_text([], totals_text(1, 0)), summaries_text(["s1", "s2"], totals_text(2, 1.5, 2.5)),
        notes_text(None), notes_text("x"), chat_text([]), chat_text(rounds),
        probe_prompt(["A", "B"]), reask_text("e", probe_format(["A", "B"])),
    ]
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()
