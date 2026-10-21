"""Reading a firm's private notes (its running memory, rewritten with every price and in the transition call;
chat turns carry no notes since 2026-10-20, older records have them) for this period's own price
and for an explicit plan to undercut or deviate this period.

Notes carry history, rules and contingency plans ("Firm A last set 87", "undercut by one", "if B deviates I'll set
30", "a brief 51-price retaliation"), so only an explicit statement about this period's own price counts.

`stated_own_price(text, period=None)` - candidates are a pricing verb with a first-person subject ("I will set
52", "I'll undercut to 92", "we set the price to 0 this period"), a clause-initial imperative, gerund or
infinitive ("Maintain price 1", "Setting 96 is a strategic move", "Proceed with 50", "..., setting my price to
83", "my choice is to set 0"), or "my price (for this period) is/: 52" and "Price: 52". Between verb and number
only filler words are allowed ("my price at", "the price to", "by one point to"). Rejected candidates:
    differences      "undercut by 1", "1 lower than", "2 points below"
    other firms      no first-person or clause-initial subject ("Firm A set 87")
    past             a past cue in the clause ("last period", "previous", "was", "in period K" for K != period),
                     and a bare "I set N" / "chose N" unless a this-period cue follows it
    contingent       an if/unless/whether/when/once/as long as/otherwise before it in the sentence or right after
                     the number, a modal of possibility (could, might, would, may), punishment wording before it
                     in the clause or right after the number ("51-price retaliation"), or a later-period cue
                     ("next period", "then", "after", "future")
    ranges           "50-60", "50 or 51", "between 50 and 60", examples ("e.g.")
    evaluations      a clause-initial move followed by a bad outcome ("Setting 32 would result in a split"), or a
                     price cut weighed against its cost ("undercutting to 99 gains 22,200, but triggers ...")
    profits          numbers > 100 and "N ECU each / of profit"
    negated          "I will not set 99"
    hypotheticals    a move under a deviation heading ("Deviation analysis: undercut to 99 gives +22,200"), a rule
                     mapping a condition to a response ("severe (p≤50) → I go to 1"), hyphenated counts
                     ("the 2-firm split")
An explicit decision label ("MY DECISION: Setting 94", "Decision: GO TO 93", "P26 DECISION: Price 89") wins over
every other statement. Otherwise, when several candidates remain, those whose clause has a this-period cue ("this
period", "this round", "now", "today", "for period t") win, the closest to its cue first; candidates that disagree
return None.

`deviation_plan(text, period=None)` - an un-negated, non-conditional first-person plan to undercut, deviate,
defect, cheat or grab the market this period ("I will undercut", "I will deviate now", "I'll try the
undercut"); "could/might/would" plans, rules about later periods and punishment plans don't count.
"""

from __future__ import annotations

import re

I = re.I
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
# clause boundaries inside a sentence: commas, colons, dashes, parentheses, " and ", " but "
_CLAUSE = re.compile(r"[,:;()–—]|\s-\s|\band\b|\bbut\b|\bso\b|\bwhile\b|\bwhich\b", I)
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?![.,]?\d)"

_MODAL = (r"(?:'ll|’ll|\s+will|\s+shall|\s+am going to|'m going to|’m going to|\s+plan to|\s+intend to|"
          r"\s+choose to|\s+decided? to|\s+aim to|\s+want to|\s+need to|\s+must|\s+should|\s+have to|\s+opt to|"
          r"\s+will (?:therefore|also|now|still|simply|again|then|just|continue to)|\s+(?:therefore|also|now|simply|"
          r"just)(?:\s+will)?|\s+will try to|\s+am|'m|’m)?")
_SET = (r"set(?:ting)?|choos(?:e|ing)|chose|pick(?:ing)?|post(?:ing)?|charg(?:e|ing)|submit(?:ting)?|keep(?:ing)?|"
        r"maintain(?:ing)?|stay(?:ing)?|stick(?:ing)?|remain(?:ing)?|hold(?:ing)?|continu(?:e|ing)(?: with| to set| "
        r"setting| at)?|proceed(?:ing)?(?: with| to set)?|match(?:ing)?|go(?:ing)?|pric(?:e|ing)(?=\s+(?:at|to|it|my|our))|bid(?:ding)?|"
        r"offer(?:ing)?|use|using|select(?:ing)?|play(?:ing)?")
_MOVE = (r"undercut(?:ting)?|lower(?:ing)?|drop(?:ping)?|cut(?:ting)?|rais(?:e|ing)|increas(?:e|ing)|"
         r"decreas(?:e|ing)|reduc(?:e|ing)|mov(?:e|ing)|shift(?:ing)?|return(?:ing)?|revert(?:ing)?|go(?:ing)? (?:down|up|back)|"
         r"adjust(?:ing)?|come|coming|step(?:ping)? (?:down|up)")
# filler between a set-type verb and its number: "set my price to", "keep the price at", "stick to the", ...
_SET_GAP = (r"(?:\s+(?:my|our|the|a|its|this|own))?(?:\s+(?:own|final|current|same|agreed|cooperative|established))?"
            r"(?:\s+price)?(?:\s+(?:for|in) (?:this|the current) (?:period|round))?"
            r"(?:\s+(?:again|still|once more|exactly|firmly|steady))?"
            r"(?:\s+(?:at|to|of|with|on|as|for))?(?:\s+(?:the|a|price|exactly|just|our|my))?(?:\s+price)?\s+\$?")
# filler between a move-type verb and its number: "undercut A by one unit to", "lower my price further to"
_MOVE_GAP = (r"(?:\s+(?:my|our|the) price)?(?:\s+(?:them|him|her|it|firm\s+[a-z]|[a-z]|their price|the others|"
             r"everyone|the rival|the opponent))?(?:\s+by\s+(?:one|two|three|a|an|\d{1,2})(?:\s+(?:units?|points?|ecu|"
             r"notch|step))?)?(?:\s+(?:further|again|slightly|all the way|back|down|up))?\s+(?:to|at)\s+\$?")
_FIRST = (r"\b(?:I|we)" + _MODAL + r"\s+(?:(?:also|then|now|simply|just|therefore|again|still|will)\s+)?"
          r"(?:(?:comply|agree|cooperate|follow suit|follow through|play along|go ahead|test|try) (?:and|by)\s+)?")
_VERB = rf"(?:(?:{_SET}){_SET_GAP}|(?:{_MOVE}){_MOVE_GAP}|try (?:the|an) undercut (?:to|at)\s+)"
FIRST_PERSON = re.compile(rf"{_FIRST}(?P<verb>{_VERB}){_NUM}", I)
CLAUSE_START = re.compile(rf"(?:^|(?<=[,:;(–—])|(?<=\s-)|\b(?:therefore|thus|so|hence|then|now|instead),?)"
                          rf"\s*(?:(?:i\s+)?(?:will|'ll|to|is to|plan is to|decision is to|choice is to|(?:the )?best "
                          rf"(?:action|move|choice|option|response) is to|(?:it is |it's )?(?:best|safest|better|wisest|"
                          rf"optimal|wise|rational) to)\s+)?(?P<verb>{_VERB}){_NUM}", I)
INFINITIVE = re.compile(rf"\b(?:my|our|the) (?:\w+\s+){{0,3}}?(?:choice|decision|plan|move|action|strategy|play|"
                        rf"option|response)\b[^.!?\d]{{0,60}}?\b(?:is|remains) to (?P<verb>{_VERB}){_NUM}", I)
MY_PRICE = re.compile(rf"(?:\bmy (?:own )?(?:price|choice|decision|bid|final price)(?: (?:for|in) (?:this|the current) "
                      rf"(?:period|round)| this (?:period|round)| now| today)?\s*(?:is|will be|:|=|stays at|remains at|"
                      rf"will stay at|will remain at|is set to|set to)|^\W*(?:final )?(?:price|decision|choice)\s*[:=])"
                      rf"(?P<verb>)\s*(?:at\s+)?\$?{_NUM}", I)
# an explicit decision label, the strongest statement: "MY DECISION: Setting 94", "P26 DECISION: Price 89",
# "Decision: GO TO 93", "DECISION: 100"
DECISION = re.compile(rf"\b(?:my |final |price )?decision\b(?: (?:for|in) (?:this|the current) (?:period|round))?\s*[:=]\s*"
                      rf"(?:(?:price|set(?:ting)?|go(?:ing)? to|stay(?:ing)? at|keep(?:ing)?|hold(?:ing)?|match(?:ing)?|"
                      rf"maintain(?:ing)?|continu(?:e|ing)(?: at)?|(?:undercut(?:ting)?|lower(?:ing)?|drop(?:ping)?|"
                      rf"cut(?:ting)?|rais(?:e|ing)|increas(?:e|ing)|mov(?:e|ing)) to)\s+(?:(?:at|to|price)\s+)?)?"
                      rf"(?P<verb>)\$?{_NUM}", I)
# ... only the firm's own label, at the start of its sentence (after a period tag such as "P26" or "PERIOD 3 -") or
# as "my decision": not "Firm B's decision: 90", "Group decision: 90"
DECISION_LEAD = re.compile(r"^\W*(?:(?:p|period|part)\s*(?P<tag>\d+)\W*)*$", I)
PATTERNS = (DECISION, FIRST_PERSON, INFINITIVE, MY_PRICE, CLAUSE_START)
# a hypothetical move under a deviation heading: "Deviation analysis: undercut to 99 gives +22,200", "Deviation from
# 100: set 99 → 29,700"
DEVIATION_LABEL = re.compile(r"\b(?:deviation|defection|undercut(?:ting)?)\s+(?:analysis|math|check|calc\w*|scenario|payoff|"
                             r"gain|incentive)s?\b(?:\s+\w+)?\s*[:=]|\bdeviation from \d+\s*[:=]", I)
# where a heading's or rule's scope ends: "Deviation check: not worth it, keep 100", "..., so hold at 100"
SCOPE_END = re.compile(r",(?!\d)|;|\b(?:so|therefore|thus|hence|instead)\b", I)  # not "14,700"
# a rule mapping a condition to a response: "severe (p≤50) → I go to 1", "if B undercuts -> I set 0"
RULE_ARROW = re.compile(r"(?:[<>≤≥]\s*=?\s*\d|\bif\b|deviat\w*|undercut\w*|defect\w*|\bsevere\b|\bmild\b)"
                        r"(?:(?!→|->|=>|⇒).){0,40}(?:→|->|=>|⇒)(?P<rest>(?:(?!→|->|=>|⇒).)*)$", I)
# a hyphenated count, not a price: "the 2-firm split", "a 1-step ratchet"
HYPHEN_COUNT = re.compile(r"^[-‐‑–](?:step|firm|way|period|round|player|unit|notch|point|time|fold|split|day)s?\b", I)

THIS_PERIOD = re.compile(r"\bthis (?:period|round|time|turn)\b|\bright now\b|\bnow\b|\btoday\b|\bcurrent (?:period|"
                         r"round)\b|\bimmediately\b|\bfor this (?:one|decision)\b", I)
PERIOD_REF = re.compile(r"\bperiods?\s+(\d{1,3})\b", I)
PAST = re.compile(r"\b(?:last|previous(?:ly)?|prior|earlier|before|ago|yesterday|was|were|had|did|got|earned|"
                  r"captured|already|so far|history|historical)\b", I)
LATER = re.compile(r"\b(?:next|future|later|after(?! (?:the )?(?:most )?(?:recent|last|previous))|afterwards|then|"
                   r"eventually|subsequent|following|from now on|thereafter|resume|once(?! again| more))\b", I)
_IF = r"(?:if|unless|when|whenever|once(?! again| more)|in case|as long as|until|otherwise|or else|should|suppose)"
IF_START = re.compile(rf"^\W*(?:\d+[.)]\s*)?(?:(?:however|but|and|then|also|still|now)\W+)?{_IF}\b", I)  # "(7) If"
IF_MID = re.compile(rf"(?:[,;:(–—]|\band\b|\bbut\b|\bor\b)\s*{_IF}\b|\b(?:whether|otherwise|or else|alternatively)\b", I)
IF_AFTER = re.compile(r"^[^.!?]{0,25}?(?<!test )(?<!testing )(?<!see )(?<!check )(?<!gauge )\b(?:if|unless|in case|once(?! again| more)|"
                      r"when|whenever|as long as|until|should (?:they|he|she|it|firm|anyone|any|[a-z]\b)|only if|"
                      r"provided)\b", I)
RESET = re.compile(r"\b(?:so|therefore|thus|hence)\b|;", I)
HYPOTHETICAL = re.compile(r"\b(?:could|might|would|may|'d)\b", I)
BAD_OUTCOME = re.compile(r"^\s*(?:ecu\s+)?(?:\w+\s+)?(?:yields? (?:no|zero|nothing|0)|gives? (?:me |us )?(?:no|zero|nothing|0)|"
                         r"means? (?:no|zero|losing))\b|"
                         r"^[^.!?]*?\b(?:risks|loses?|would|could|might|offers? no|no (?:gain|advantage|benefit)|worse|"
                         r"only reduces?|reduces? (?:my|our))\b", I)
# a price cut weighed against its cost, not a plan: "undercutting to 99 gains +22,200 ECU this period, but expected
# punishment ..." ("Setting 1 gives us a share" is the decision with its reason)
CUT_VERB = re.compile(r"(?:undercut|lower|drop|cut|decreas|reduc|deviat)", I)
# (the "but" clause names a cost and no first-person decision follows: "..., but I accept the retaliation risk" is one)
GAIN_BUT = re.compile(r"^\s*(?:ecu\s+)?(?:\S+\s+){0,3}?(?:yields?|gains?|earns?|nets?|gives?|brings?|generates?|pays?)\b"
                      r"(?:[^.!?]|\.\d)*?\b(?:but|however|yet|though|although)\b"
                      r"(?=(?:[^.!?]|\.\d)*?\b(?:punish|retaliat|trigger|price war|risk|los[es]|losing|destroy|not worth|"
                      r"cost|future|remaining|long[- ]run|shadow)\w*)"
                      r"(?!(?:[^.!?]|\.\d)*?\b(?:I|we)(?:'ll|’ll| will| accept| am going to|'m going to| still)\b)", I)
PUNISH = re.compile(r"punish|retaliat|penalt|enforce|in response|tit[- ]for[- ]tat|revenge|threat", I)
PUNISH_AFTER = re.compile(r"^\s*(?:ecu)?\s*[-‑–]?\s*(?:price\s+)?(?:punish|retaliat|penalt|threat|response|deterr)", I)
DIFF_AFTER = re.compile(r"^\s*(?:ecu\s+|units?\s+|points?\s+|notch(?:es)?\s+|steps?\s+)?(?:lower|higher|below|above|"
                        r"under|over|less|more|cheaper|down|up|off)\b", I)
RANGE_AFTER = re.compile(r"^\s*(?:[-‑–]|to|or|and|/)\s*\$?\d|^\s*\(\s*(?:or|to)\s+\$?\d", I)  # "50-60", "0 (or 1)"
RANGE_BEFORE = re.compile(r"(?:\bbetween|\bfrom|\be\.g\.,?|\bsuch as|\bfor example|\bfor instance|\bby|\bor|[-‑–]|/)\s*\$?$",
                          I)
PROFIT_AFTER = re.compile(r"^\s*(?:ecu\s+)?(?:each|apiece|(?:of |in )?profit|per (?:period|round|unit)|in profit|total|units?|"
                          r"customers|consumers|buyers|%|percent|periods?|rounds?|times|firms|of (?:us|them|the))", I)
NEGATED = re.compile(r"(?:\bnot|n't|n’t|\bnever|\bno|\bavoid|\bwithout|\brefrain from|\binstead of|\brather than)"
                     r"\W+(?:\w+\W+){0,3}$", I)
_PLAN_SUBJ = (r"\b(?:I|we)(?:'ll|’ll| will| shall| am going to|'m going to|’m going to| plan to| intend to| decided? to|"
              r" choose to| must| should| need to| have to| want to| aim to)?(?:\s+(?:now|then|simply|also|therefore|"
              r"actually|just|try to|still))?\s+")
DEVIATE_VERB = re.compile(
    _PLAN_SUBJ + r"(?:undercut|deviate|defect|cheat|betray|grab|steal|break (?:the|our|this) "
    r"(?:agreement|deal|promise|pact)|try (?:the|an) undercut|go (?:below|under|lower)|price (?:below|under)|"
    r"set (?:a )?(?:slightly )?lower price|drop below|capture (?:the )?(?:whole|entire|full) market|"
    r"take (?:the )?(?:whole|entire|full) market)|"
    r"\brather than (?:stay|keep|stick|remain)(?:ing)? (?:at |to |with )?(?:the |our )?agreed", I)
GRAB_PURPOSE = re.compile(
    _PLAN_SUBJ + r"[^.!?]{0,40}?\bto (?:win|capture|take|grab|regain|reclaim|win back|steal)\s+(?:(?:the|all|whole|"
    r"entire|full|of)\s+)*(?:market|customers|buyers|demand)", I)


def _sentences(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for part in _SENT.split(text or ""):
        start = text.find(part, pos)
        if part.strip():
            out.append((start, part))
        pos = start + len(part)
    return out


def _clause_bounds(sent: str, a: int, b: int) -> tuple[int, int]:
    """The clause of `sent` around the span [a, b): from the last boundary before a to the first after b."""
    lo = max((m.end() for m in _CLAUSE.finditer(sent, 0, a)), default=0)
    hi = next((m.start() for m in _CLAUSE.finditer(sent, b)), len(sent))
    return lo, hi


def _conditional(sent: str, a: int) -> bool:
    """Whether position `a` of `sent` lies in a conditional or deliberative scope: the sentence starts with
    "if/unless/when/...", or an "if" clause or "whether/either/otherwise" comes before it (a "so/therefore"
    conclusion resets the scope); an "if" inside a clause ("to avoid zero if A lowers, I'll set 99") does not."""
    prefix = sent[:a]
    cut = max((m.end() for m in RESET.finditer(prefix)), default=0)
    prefix = prefix[cut:]
    return bool(IF_START.search(prefix) or IF_MID.search(prefix))


def _under_deviation_heading(prefix: str) -> bool:
    """Whether a statement after `prefix` (its sentence so far) sits in the clause right after a deviation heading."""
    label = None
    for label in DEVIATION_LABEL.finditer(prefix):
        pass
    return label is not None and not SCOPE_END.search(prefix[label.end():])


def _rule_response(prefix: str) -> bool:
    """Whether a statement after `prefix` is the response of a condition → response rule ("severe (p≤50) → I go to
    1"), in the arrow's own clause ("All prices >= 95 -> cooperation intact, I will hold at 100" is not)."""
    m = RULE_ARROW.search(prefix)
    return bool(m) and not SCOPE_END.search(m.group("rest"))


def _own_decision(prefix: str, label: str, period: int | None) -> bool:
    """Whether a decision label is the firm's own for this period: "my decision", or a bare label at the start of its
    sentence after at most a period tag that names this period ("P26 DECISION: 89" at period 26, not "P25 ...")."""
    if not label.lower().startswith("my") and not DECISION_LEAD.match(prefix):
        return False
    tags = [int(t) for t in re.findall(r"\b(?:p|period)\s*(\d+)", prefix, I)]  # "P25 MY DECISION" at period 26: not this one
    return period is None or not tags or tags[-1] == period


def _period_refs(span: str, period: int | None) -> tuple[bool, bool]:
    """(refers to this period, refers to another period) by "period K" mentions in `span`."""
    ks = [int(k) for k in PERIOD_REF.findall(span)]
    if period is None or not ks:
        return False, False
    return period in ks, any(k != period for k in ks)


def _excluded_context(sent: str, a: int, b: int, period: int | None) -> tuple[bool, bool]:
    """(excluded, has a this-period cue) for a statement spanning [a, b) of `sent`: negated, conditional,
    hypothetical, about punishment, about another period, past or later."""
    lo, hi = _clause_bounds(sent, a, b)
    before, after = sent[lo:a], sent[b:hi]
    this_ref, other_ref = _period_refs(sent[lo:hi], period)
    cue = bool(THIS_PERIOD.search(before) or THIS_PERIOD.search(after) or this_ref)
    near = sent[b:min(hi, b + 30)]
    later_near, this_near = LATER.search(near), THIS_PERIOD.search(near)
    excluded = (NEGATED.search(before) or _conditional(sent, a) or IF_AFTER.search(sent[b:])
                or re.search(rf"\b(?:{_IF}|whether|either)\b", before, I)
                or HYPOTHETICAL.search(before) or HYPOTHETICAL.search(after) or PUNISH.search(before)
                or (PUNISH.search(after) and not cue) or (other_ref and not this_ref)
                # "set 99 next period instead of 100 this period": a later cue right after the number wins
                or (later_near and (not this_near or later_near.start() < this_near.start()))
                or _rule_response(sent[:a])
                or ((PAST.search(before) or PAST.search(near) or LATER.search(before)) and not cue))
    return bool(excluded), cue


def _candidates(text: str, period: int | None) -> list[tuple[int, int, int, int | None, str]]:
    """(value, tier, position, distance to a this-period cue or None, the sentence and the one before it) of every
    accepted statement; tier -1 for a decision label ("DECISION: Set 94"), 0 for a first-person or "my price"
    statement, 1 for a clause-initial imperative or gerund."""
    out = []
    prev = ""
    for s0, sent in _sentences(text):
        context, prev = f"{prev} {sent}", sent
        seen = set()
        for rx in PATTERNS:
            tier = -1 if rx is DECISION else int(rx is CLAUSE_START)
            for m in rx.finditer(sent):
                a, b = m.start(), m.end()
                raw = m.group("num")
                if b in seen or "," in raw:  # found by another pattern; a thousands number (above 100)
                    continue
                v = int(raw)
                after, before = sent[b:], sent[:m.start("num")]
                if v > 100 or DIFF_AFTER.search(after) or RANGE_AFTER.search(after) or PROFIT_AFTER.search(after) \
                        or HYPHEN_COUNT.search(after):
                    continue
                if RANGE_BEFORE.search(before) or PUNISH_AFTER.search(after):
                    continue
                if rx is CLAUSE_START and _under_deviation_heading(sent[:a]):
                    continue  # "Deviation analysis: undercut to 99 gives +22,200": a hypothetical
                if rx is DECISION and not _own_decision(sent[:a], m.group(0), period):
                    continue
                if rx is CLAUSE_START and BAD_OUTCOME.search(after):
                    continue  # "Setting 32 would result in a split", "Raising to 42 risks ..."
                if rx is CLAUSE_START and CUT_VERB.match(m.group("verb") or "") and GAIN_BUT.search(after):
                    continue
                excluded, cue = _excluded_context(sent, a, b, period)
                if excluded:
                    continue
                verb = (m.group("verb") or "").strip().lower()
                bare_past = rx is FIRST_PERSON and bool(re.match(r"(?:set|chose|undercut|cut|bid|put)\b", verb)) and not re.search(
                    r"(?:'ll|’ll|will|shall|going to|plan to|intend to|decided? to|choose to|must|should|need to|"
                    r"have to|want to|aim to|opt to|and|by)\s+(?:\w+\s+)?(?:set|chose|undercut|cut|bid|put)", m.group(0), I)
                if (bare_past or verb.startswith("chose")) and not (cue or re.match(r"\s*(?:ecu\s+)?to\s+\w", after, I)):
                    continue  # "I set 88" may be past; only with "this period" or a purpose ("I set 99 to win")
                dist = None
                if cue:
                    cues = [abs(c.start() - b) for c in THIS_PERIOD.finditer(sent)]
                    if period is not None:
                        cues += [abs(c.start() - b) for c in PERIOD_REF.finditer(sent) if int(c.group(1)) == period]
                    dist = min(cues) if cues else 0
                seen.add(b)
                out.append((v, tier, s0 + b, dist, context))
    return out


def stated_own_price(text: str | None, period: int | None = None) -> int | None:
    """This period's own price as stated in the notes, or None when absent or ambiguous (see module docstring)."""
    return stated_own_price_context(text, period)[0]


def stated_own_price_context(text: str | None, period: int | None = None) -> tuple[int | None, str]:
    """`stated_own_price` and the sentence that states it with the one before it ("" when None)."""
    cands = _candidates(text or "", period)
    if not cands:
        return None, ""
    decided = [c for c in cands if c[1] < 0]  # an explicit "DECISION: N" beats everything else
    cued = [c for c in cands if c[3] is not None]
    pool = decided or cued or cands
    if not decided and cued:  # closest to its this-period cue first
        best = min(c[3] for c in pool)
        pool = [c for c in pool if c[3] == best]
    tier = min(c[1] for c in pool)
    pool = [c for c in pool if c[1] == tier]
    vals = {c[0] for c in pool}
    return (pool[0][0], " ".join(dict.fromkeys(c[4] for c in pool))) if len(vals) == 1 else (None, "")


def deviation_plan(text: str | None, period: int | None = None) -> bool:
    """Whether the notes contain an explicit first-person plan to undercut or deviate this period."""
    for _, sent in _sentences(text or ""):
        for rx in (DEVIATE_VERB, GRAB_PURPOSE):
            for m in rx.finditer(sent):
                a, b = m.start(), m.end()
                excluded, cue = _excluded_context(sent, a, b, period)
                lo, hi = _clause_bounds(sent, a, b)
                if excluded or re.search(r"\b(?:can|could|might|would|may)\b", m.group(0), I) \
                        or PUNISH.search(sent[lo:hi]):  # a punishment plan is not a deviation
                    continue
                if re.match(r"(?:I|we)\s+(?:undercut|deviate|defect|cheat|betray|grab|steal)\b", m.group(0), I) \
                        and not cue and not re.match(r"\s*\w*\s*(?:to|at|by|now|here)\b", sent[b:], I):
                    continue  # bare "I undercut" may be past
                return True
    return False
