"""Mistakes vs deliberate choices: invalid-output rates and a small set of heuristic flags.

Output problems (from the engine's logs): a decision with any invalid attempt, re-asks, fallbacks (price set to
the firm's previous price; chat counted as a pass), coerced values (numeric strings, integral floats, or JSON
with a string closed by a typographic quote), and truncated chat messages.

Heuristic flags on price decisions (meant for manual reading, examples saved):
    notes_contradict     the price-decision notes state this period's own price explicitly ("I will set 52 for
                         this period", "my price: 99"; `notes.stated_own_price`, which ignores history, other
                         firms, differences such as "undercut by one", contingent plans and ranges, and returns
                         None when statements disagree) and the valid posted price differs (fallbacks excluded)
    low_price_no_context price <= 1 at period >= 2 of a part, while every other firm was at >= 90 in the
                         previous period, and no punishment context: the firm has not sent a threat in this part
                         so far and its notes contain no punishment wording
    price_zero           price 0 (weakly dominated by 1: it always earns 0). Counted, not called a mistake
Rule-misreading flags on notes and chat messages (regex; skipped when negated within 30 characters before or
inside the match; examples saved):
    highest_wins         "higher price wins / attracts the buyers", in a sentence without if/when/unless ("a
                         higher price gets undercut" doesn't count)
    tie_full             a tie or matching price earning the full/entire market, within one clause free of
                         contrast, condition, undercut or loss-avoidance wording ("A set 65 instead and took the
                         entire market", "I may match to avoid losing all customers" don't count), with no
                         split/share wording within 80 characters
    horizon              "this is the last/final period", "only 20 periods", "game ends at 20", "the game ends after
                         this period" before period 20, outside hypotheticals ("if I can confirm it's the last
                         period", "in the shortest scenario (game ends at 20)") (claims about periods or the game
                         only: "last/final round" refers to the chat rounds, whose end the prompts announce, and is
                         not flagged; nor is "the final period of the punishment")
"""

from __future__ import annotations

import re

import pandas as pd

from .chat import has_unnegated
from .notes import stated_own_price
from .load import num

I = re.I
PUNISH = re.compile(r"punish|retaliat|teach|lesson|price war|revenge|tit[- ]for[- ]tat|in response to|respond to|"
                    r"signal|warn|because (?:they|someone|firm|[a-z]) (?:undercut|cheated|defected)", I)
MISREAD = {
    "highest_wins": re.compile(
        r"\b(?:highest|higher) price[sd]? (?:wins?|gets?|captures?|takes?|attracts?|sells?|earns? (?:more|the most))\b"
        r"(?! (?:undercut|beaten|nothing|no\b|zero|0\b|less|lower))|"
        r"\b(?:buyers|customers|consumers|demand) (?:buy|go|goes|prefer|choose|flows?) (?:from |to )?(?:the )?"
        r"(?:highest|higher)", I),
    "tie_full": re.compile(
        r"(?:\btie\b|\btied\b|same price|equal price|\bmatch(?:ing)?\b)"
        r"(?:(?!\b(?:instead|but|if|when|whenever|now|while|whereas|otherwise|undercuts?|undercutting|lower|below|"
        r"raises?|above|higher|fails?|step|steps|loses?|lost|losing|(?:avoid|risk)(?:s|ing)? los\w+)\b)"
        r"[^.!?;,()]){0,60}?\b(?:full|entire|whole|all(?: the)?|"
        r"100% of the)\s+(?:profit|market|demand|sales|buyers|customers)", I),
    "horizon": re.compile(
        r"\b(?:this is|this will be|this might be|this could be|it'?s) (?:the |likely the |probably the )?"
        r"(?:last|final) period(?!\s+(?:of|in|for)\b)|\bfinal period\b(?!\s+(?:of|in|for)\b)|game ends (?:at|after) (?:period )?20\b|only 20 periods|"
        r"20 periods (?:in )?total|"
        r"\b(?:game|market|experiment) (?:ends|will end|is over|is ending) (?:now|after this (?:period|one)|this period)",
        I),
}
SPLIT = re.compile(r"split|share|divid|half|equally", I)
NEG = re.compile(r"\b(?:not|never|no)\b|n't", I)
IF = re.compile(r"\b(?:if|when|unless|as long as)\b", I)
# a hypothetical opening the clause of a horizon claim: "if I can confirm it's the last period", "in the shortest
# scenario (game ends at 20)" ("This is the final period when retaliation ends" is still a claim)
HYPO_BEFORE = re.compile(r"\b(?:if|unless|in case|suppose|whether|scenario)\b", I)
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
            if k == "horizon" and (period >= 20 or _hypothetical_before(text, m.start())):
                continue
            out.append(k)
            break
    return out


def _hypothetical_before(text: str, start: int) -> bool:
    """Whether a hypothetical word comes before `start` in its clause (from the last sentence or clause break)."""
    lo = max(text.rfind(c, 0, start) for c in ".!?\n,;") + 1
    return bool(HYPO_BEFORE.search(text[lo:start]))


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
    f["stated_price"] = pd.array([stated_own_price(t, p) for t, p in zip(f["notes"], f["period"])], dtype="Int64")
    f["notes_contradict"] = f["stated_price"].notna() & ~f["fallback"] & f["stated_price"].ne(f["price"])
    f["low_price_no_context"] = (f["price"].le(1) & f["period"].ge(2) & f["prev_others_min"].ge(90)
                                 & ~f["threat_so_far"] & ~f["notes"].map(lambda t: has_unnegated(PUNISH, t)))
    f["price_zero"] = f["price"].eq(0)
    mr = [misreads(t, p) for t, p in zip(f["notes"], f["period"])]
    for k in MISREAD:
        f[f"misread_{k}"] = [k in m for m in mr]
    return f


def flag_chat(chat: pd.DataFrame) -> pd.DataFrame:
    """Chat turns (messages and passes) plus misreading flags on the message and its notes (chat turns carry notes
    only in records with `PARSER_VERSION` < 4, e.g. the 2026-10-20 sweeps; newer ones are flagged on the message alone)."""
    c = chat.copy()
    mr = [misreads(f"{m or ''} {n}", p) for m, n, p in zip(c["message"], c["notes"], c["period"])]
    for k in MISREAD:
        c[f"misread_{k}"] = [k in m for m in mr]
    return c


def flag_transition(tr: pd.DataFrame) -> pd.DataFrame:
    """Transition-call notes plus misreading flags (they are read as written before Part 2 period 1)."""
    t = tr.copy()
    mr = [misreads(n, 1) for n in t["notes"]]
    for k in MISREAD:
        t[f"misread_{k}"] = [k in m for m in mr]
    return t


def _transition_agg(ft: pd.DataFrame | None, by: list[str]) -> pd.DataFrame | None:
    if ft is None or not len(ft):
        return None
    mr = [f"misread_{k}" for k in MISREAD]
    t = ft.assign(reask=ft["invalid_attempts"] > 0, misread=ft[mr].any(axis=1))
    return t.groupby(by).agg(transition_decisions=("firm", "size"), transition_reask_rate=("reask", "mean"),
                             transition_fallback=("fallback", "mean"), misread_transition=("misread", "mean"))


def rates(ff: pd.DataFrame, fc: pd.DataFrame, ft: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per model: output-problem rates (price, chat and transition decisions) and heuristic flag rates."""
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
    if (t := _transition_agg(ft, ["model"])) is not None:
        out = out.join(t, how="left")
    return out.reset_index()


CELL = ["model", "n", "ordering"]


def cell_rates(ff: pd.DataFrame, fc: pd.DataFrame, ft: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (model, n, ordering): format slips kept apart from strategic mistakes.

    Format slips: the share of price decisions and of chat turns that needed a re-ask (any invalid attempt), and
    their fallback shares. Strategic mistakes: notes contradicting the posted price, a near-zero price without
    context, and any rule misread in price notes or in chat turns (message, plus notes in records with
    `PARSER_VERSION` < 4). With `ft`, also the transition call's decision count and re-ask, fallback and misread
    rates."""
    mr = [f"misread_{k}" for k in MISREAD]
    p = ff.assign(reask=ff["invalid_attempts"] > 0, misread=ff[mr].any(axis=1))
    out = p.groupby(CELL).agg(
        price_decisions=("price", "size"), price_reask_rate=("reask", "mean"), price_fallback=("fallback", "mean"),
        notes_contradict=("notes_contradict", "mean"), low_price_no_context=("low_price_no_context", "mean"),
        misread_notes=("misread", "mean"))
    if len(fc):
        c = fc.assign(reask=fc["invalid_attempts"] > 0, misread=fc[mr].any(axis=1))
        out = out.join(c.groupby(CELL).agg(
            chat_turns=("round", "size"), chat_reask_rate=("reask", "mean"), chat_fallback=("fallback", "mean"),
            misread_chat=("misread", "mean")), how="left")
    else:
        out = out.assign(chat_turns=0, chat_reask_rate=None, chat_fallback=None, misread_chat=None)
    if (t := _transition_agg(ft, CELL)) is not None:
        out = out.join(t, how="left")
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
    if v is pd.NA:
        return None
    v = v.item() if hasattr(v, "item") else v
    return num(v) if isinstance(v, float) else v
