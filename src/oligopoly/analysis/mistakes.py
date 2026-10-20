"""Mistakes vs deliberate choices: invalid-output rates and a small set of heuristic flags.

Output problems (from the engine's logs): a decision with any invalid attempt, re-asks, fallbacks (price set to
the firm's previous price; chat counted as a pass), coerced values (numeric strings or integral floats), and
truncated chat messages.

Heuristic flags on price decisions (meant for manual reading, examples saved):
    notes_contradict     the private notes state a first-person own price ("I will set 100", "my price: 99")
                         and the valid posted price differs (last such statement counts; fallbacks excluded)
    low_price_no_context price <= 1 at period >= 2 of a part, while every other firm was at >= 90 in the
                         previous period, and no punishment context: the firm has not sent a threat in this part
                         so far and its notes contain no punishment wording
    price_zero           price 0 (weakly dominated by 1: it always earns 0). Counted, not called a mistake
Rule-misreading flags on notes and chat messages (regex; skipped when negated within 30 characters before or
inside the match; examples saved):
    highest_wins         "higher price wins / attracts the buyers", in a sentence without if/when/unless
    tie_full             a tie or matching price earning the full/entire market, with no split/share wording
                         within 80 characters
    horizon              "this is the last/final period", "only 20 periods", "game ends at 20", "the game ends after
                         this period" before period 20 (claims about periods or the game only: "last/final round"
                         refers to the chat rounds, whose end the prompts announce, and is not flagged)
"""

from __future__ import annotations

import re

import pandas as pd

from .chat import has_unnegated, stated_own_price
from .load import num

I = re.I
PUNISH = re.compile(r"punish|retaliat|teach|lesson|price war|revenge|tit[- ]for[- ]tat|in response to|respond to|"
                    r"signal|warn|because (?:they|someone|firm|[a-z]) (?:undercut|cheated|defected)", I)
MISREAD = {
    "highest_wins": re.compile(
        r"\b(?:highest|higher) price[sd]? (?:wins?|gets?|captures?|takes?|attracts?|sells?|earns? (?:more|the most))|"
        r"\b(?:buyers|customers|consumers|demand) (?:buy|go|goes|prefer|choose|flows?) (?:from |to )?(?:the )?"
        r"(?:highest|higher)", I),
    "tie_full": re.compile(
        r"(?:\btie\b|\btied\b|same price|equal price|\bmatch(?:ing)?\b)[^.!?]{0,60}?\b(?:full|entire|whole|all(?: the)?|"
        r"100% of the)\s+(?:profit|market|demand|sales|buyers|customers)", I),
    "horizon": re.compile(
        r"\b(?:this is|this will be|this might be|this could be|it'?s) (?:the |likely the |probably the )?"
        r"(?:last|final) period|\bfinal period\b|game ends (?:at|after) (?:period )?20\b|only 20 periods|"
        r"20 periods (?:in )?total|"
        r"\b(?:game|market|experiment) (?:ends|will end|is over|is ending) (?:now|after this (?:period|one)|this period)",
        I),
}
SPLIT = re.compile(r"split|share|divid|half|equally", I)
NEG = re.compile(r"\b(?:not|never|no)\b|n't", I)
IF = re.compile(r"\b(?:if|when|unless|as long as)\b", I)
FIRM_KEYS = ["market_id", "part", "period"]


def misreads(text: str, period: int) -> list[str]:
    """Misreading categories found in `text` written at `period` of its part."""
    out = []
    text = text or ""
    for k, rx in MISREAD.items():
        for m in rx.finditer(text):
            sent = _sentence(text, m.start(), m.end())
            if NEG.search(text[max(0, m.start() - 30):m.end()]):  # "this is not the final period"
                continue
            if k == "highest_wins" and IF.search(sent):  # "a higher price earns more if everyone matches"
                continue
            if k == "tie_full" and SPLIT.search(text[max(0, m.start() - 80):m.end() + 80]):
                continue
            if k == "horizon" and period >= 20:
                continue
            out.append(k)
            break
    return out


def _sentence(text: str, start: int, end: int) -> str:
    a = max(text.rfind(c, 0, start) for c in ".!?\n") + 1
    ends = [i for i in (text.find(c, end) for c in ".!?\n") if i >= 0]
    return text[a:min(ends) if ends else len(text)]


def flag_firm_periods(fp: pd.DataFrame, chat_msgs: pd.DataFrame) -> pd.DataFrame:
    """firm-period table plus heuristic flag columns."""
    f = fp.sort_values(FIRM_KEYS + ["firm"]).copy()
    keys = [f[k] for k in FIRM_KEYS]
    mn = f.groupby(keys)["price"].transform("min")
    second = f["price"].where(f["price"] > mn).groupby(keys).transform("min")  # lowest price above the min
    sole_min = f["price"].eq(mn) & f["price"].eq(mn).groupby(keys).transform("sum").eq(1)
    f["others_min"] = second.where(sole_min, mn)
    f["prev_others_min"] = f.groupby(["market_id", "part", "firm"])["others_min"].shift(1)
    # threats sent by the firm so far in this part
    if len(chat_msgs):
        th = chat_msgs[chat_msgs["threat"]].groupby(["market_id", "part", "firm"])["period"].min()
        first_threat = f.set_index(["market_id", "part", "firm"]).index.map(th.to_dict().get)
        f["threat_so_far"] = [t is not None and t <= p for t, p in zip(first_threat, f["period"])]
    else:
        f["threat_so_far"] = False
    f["stated_price"] = f["notes"].map(stated_own_price)
    f["notes_contradict"] = f["stated_price"].notna() & ~f["fallback"] & f["stated_price"].ne(f["price"])
    f["low_price_no_context"] = (f["price"].le(1) & f["period"].ge(2) & f["prev_others_min"].ge(90)
                                 & ~f["threat_so_far"] & ~f["notes"].map(lambda t: has_unnegated(PUNISH, t)))
    f["price_zero"] = f["price"].eq(0)
    mr = [misreads(t, p) for t, p in zip(f["notes"], f["period"])]
    for k in MISREAD:
        f[f"misread_{k}"] = [k in m for m in mr]
    return f


def flag_chat(chat: pd.DataFrame) -> pd.DataFrame:
    """Chat turns (messages and passes) plus misreading flags on the message and its notes."""
    c = chat.copy()
    mr = [misreads(f"{m or ''} {n}", p) for m, n, p in zip(c["message"], c["notes"], c["period"])]
    for k in MISREAD:
        c[f"misread_{k}"] = [k in m for m in mr]
    return c


def rates(ff: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Per model: output-problem rates (price and chat decisions) and heuristic flag rates."""
    price = ff.assign(invalid=ff["invalid_attempts"] > 0).groupby("model").agg(
        price_decisions=("price", "size"), price_any_invalid=("invalid", "mean"), price_reasks=("reasks", "sum"),
        price_fallback=("fallback", "mean"), price_coerced=("coerced", "mean"),
        notes_contradict=("notes_contradict", "mean"), low_price_no_context=("low_price_no_context", "mean"),
        price_zero=("price_zero", "mean"),
        **{f"notes_misread_{k}": (f"misread_{k}", "mean") for k in MISREAD})
    out = price
    if len(fc):
        chat = fc.assign(invalid=fc["invalid_attempts"] > 0).groupby("model").agg(
            chat_decisions=("round", "size"), chat_any_invalid=("invalid", "mean"), chat_reasks=("reasks", "sum"),
            chat_fallback=("fallback", "mean"), chat_truncated=("truncated", "mean"),
            **{f"chat_misread_{k}": (f"misread_{k}", "mean") for k in MISREAD})
        out = price.join(chat, how="left")
    return out.reset_index()


def examples(ff: pd.DataFrame, fc: pd.DataFrame, per_model: int = 10) -> dict[str, list[dict]]:
    """Up to `per_model` examples per model for each flag (for manual reading)."""
    cols = ["model", "market_id", "part", "period", "firm"]
    out: dict[str, list[dict]] = {}

    def take(df: pd.DataFrame, flag: str, text_cols: list[str]) -> list[dict]:
        sel = df[df[flag]].groupby("model").head(per_model)
        return [{c: _py(r[c]) for c in cols + text_cols if c in r} for _, r in sel.iterrows()]

    for flag in ("notes_contradict", "low_price_no_context", "price_zero"):
        out[flag] = take(ff, flag, ["price", "stated_price", "prev_others_min", "notes"])
    out["price_fallback"] = take(ff, "fallback", ["price", "errors"])
    for k in MISREAD:
        out[f"misread_{k}_notes"] = take(ff, f"misread_{k}", ["price", "notes"])
        if len(fc):
            out[f"misread_{k}_chat"] = take(fc, f"misread_{k}", ["round", "message", "notes"])
    return out


def _py(v):
    """numpy scalars -> Python; NaN -> None (strings and bools pass through)."""
    v = v.item() if hasattr(v, "item") else v
    return num(v) if isinstance(v, float) else v
