"""Rule-based coding of chat messages and Talk periods (DECISIONS.md: no LLM judges).

Per message (`code_message`):
    prices      integers 0-100 in a price-like context (not followed by "periods", "%", "of us", "profit", ...;
                not preceded by "period", "round", "firm", "#", ...). Counts and durations are not prices
                ("5-period", "3 more rounds", "2 consecutive", "for the next 3", "per 2", "10 units"), nor is
                the price of a punishment ("the 0-price penalty"). "N ECU" is a price ("Let's set 100 ECU")
                unless profit wording precedes it in the sentence ("we each earn 75 ECU") or follows the ECU
                ("75 ECU each", "75 ECU of profit"). A comma group is a thousands number ("7,500", "29,700")
                unless every group is a plausible price and the sentence has no profit wording: "99,100" and
                "100,100,100" are lists of prices
    endorsed    the price the sender backs for this period, read clause by clause (`endorsement`): the first
                price of a clause with a proposal, commitment or agreement cue. Sentences with deliberation
                ("we can discuss/revisit/consider 2", "open to 2"), refusals or taking-turns proposals are
                skipped; a conditional, threat or accusation ends its sentence ("We keep 100, but if you
                undercut, I set 0": 100); clauses about the past, describing rivals' prices ("G and B set 30")
                or evaluating a move ("Moving to 12 now would harm everyone") are skipped. Clauses about this
                period ("this period", "now", "today", "keep/stay at 10") beat clauses with no time cue, which
                beat the next period ("next period", "tomorrow"), which beats later times ("then", "after a few
                rounds", "once the penalty ends"); a later clause makes what follows later too. A message only
                about the next period still endorses ("set 2 next period" often means the price about to be
                posted), unless the period's chat also speaks of this period (`support`). Numbers in a punishment
                rule are not endorsed ("keep the 5-period 0-price penalty rule"), and a clause carrying out a
                punishment ("keep 1 this period to finish the punishment": `punish_now`) or bidding to win the
                market alone ("set 24 this period to capture the whole market": `solo_now`) endorses nothing
                and, unless the message endorses a price for this period elsewhere, neither do its later plans.
                Failing a cued clause, the only distinct price in the usable clauses
    proposal    a proposal/commitment cue together with an endorsed price
    agreement   an un-negated agreement marker ("agree", "ok", "deal", "yes", "sounds good", "I'm in", ...) in a
                message without a refusal
    refusal     "disagree", "I don't agree", "won't set", "no, ...", negated proposals ("we should not set 100",
                "let's not go to 100", "I won't price at 100"; not "don't go below 100") (clears the sender's
                backing; its sentences endorse nothing, and a price they reject is not endorsed by the
                single-price fallback)
    threat      un-negated punishment language ("punish", "retaliate", "price war", "if anyone undercuts ...",
                "we will all go to 0", "last chance", "I'll match the lowest", ...); "let's avoid a price
                war" is not a threat
    turns       taking-turns proposals ("take turns", "rotate", "alternate", "one firm at 99", ...)
    accusation  complaints about a defection ("you undercut", "cheated", "betrayed", "you promised", ...),
                excluding conditional sentences
    apology     "sorry", "apologise", "my mistake", ...

Per Talk period (`code_period`):
    support     each firm backs its last endorsed price of the period; a bare agreement marker (no price)
                backs the most recent price endorsed by another firm in an earlier round, else the agreed
                price of period t-1 (if any); a refusal, punishment price or solo bid clears the backing. When
                any message of the period speaks of this period, endorsements of a later period are ignored
    agreed      the price backed by a strict majority of firms (> n/2); ties go to the higher price
    kept        every firm posted exactly the agreed price; otherwise broken, with deviators split into
                undercutters (< agreed) and firms above it
    punished    after a breach with undercutting at t, some firm that did not undercut at t posts a lower
                price at t+1 than its own price at t (same part); also the paper's "selling price < 99 at t+1"
    threatened  a threat-coded message by any firm in the same part at or before period t
    deception   a firm backed the agreed price P in chat, while the notes of its price decision that period
                (written after the chat; not its chat-turn notes) state this period's own price below P, or an
                explicit first-person plan to undercut or deviate this period (`notes.stated_own_price` and
                `notes.deviation_plan`: history, other firms' prices, differences, contingent and punishment plans
                don't count; nor does a lower stated price with punishment wording in its sentence or the one
                before, e.g. an agreed 0-price punishment; nor a lower price or vague plan in notes that also say
                in the first person the firm keeps P this period, `keeps_price`, nor a vague plan when the stated price is P);
                `deception_priced` additionally requires that it then posted below P
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .load import KEY_COLS
from . import notes as _notes
from .notes import deviation_plan, stated_own_price, stated_own_price_context  # noqa: F401 (re-exported)

I = re.I
# A whole number, thousands separators included ("7,500" is 7500, not 7); values > 100 are discarded later.
NUMBER = r"\d{1,3}(?:,\d{3})+|\d+"
NUM = re.compile(rf"(?<![\w#.\-‐‑–])\$?({NUMBER})(?![\d%]|,\d{{3}}(?!\d)|\.\d|[a-z])", I)
# counts and durations, not prices: "period 5", "Part 2", "per 3", "for the next 5", "another 3", "every 2"
# ("period 5" but not "the 10-firm 100-price pact")
NOT_BEFORE = re.compile(r"(?:(?<![\-‐‑–])(?:period|round|part|turn|firm|player)|number|no\.|step|day|#|top|first|"
                        r"last|\bper|\bnext|\banother|\bevery|\bfor (?:the )?(?:next|following|first|last|remaining|"
                        r"another))\s*$", I)
# "5 periods", "5-period", "5‑round", "3 more rounds", "2 consecutive", "4 firms", "10 units", "the other 3"
NOT_AFTER = re.compile(r"^\s*[-‐‑–]?\s*(?:(?:more|consecutive|straight|additional|extra|further|full|subsequent|"
                       r"following|remaining)\b[\s\-‐‑–]*)?(?:periods?|rounds?|times?|turns?|firms?|players?|"
                       r"of (?:us|you|them)|companies|competitors|rivals|others|percent|points|profits?|units?|"
                       r"customers|buyers|consumers|(?:th|st|nd|rd|x)\b|consecutive|straight|in a row|"
                       # the price of a punishment, not an agreement: "the 0-price penalty", "a 0 punishment"
                       r"(?:price|ecu)?[\s\-‐‑–]*(?:penalt|punish|retaliat|deterr|reset|sanction|shock|threat))", I)
# profit wording: an "N ECU" amount with it before (same sentence) is a profit, not a price, and so is a comma group
PROFIT_CTX = re.compile(r"profit|earn|payoff|income|revenue|\btotal|\bsplit|\bsum\b|"
                        r"\b(?:each|we|you|i|both|all|they|everyone)\s+(?:would\s+|will\s+|could\s+|can\s+|'d\s+|"
                        r"'ll\s+)?(?:get|gets|make|makes|receive|pocket|take home|walk away with)\b", I)
ECU_PROFIT_AFTER = re.compile(r"^\s*ecu\b\s*(?:of\s+)?(?:profits?|each|apiece|in total|total|per (?:period|round)|"
                              r"earn|payoff|income|revenue)", I)
ECU_AFTER = re.compile(r"^\s*ecu\b", I)
SENT = re.compile(r"(?<=[.!?;\n])\s+|\n")
CONDITIONAL = re.compile(r"^\W*(?:if|otherwise|or else|unless|whoever|anyone who|should anyone|else|when|whenever|"
                         r"once(?! again| more)|"
                         r"as soon as)\b|\bif (?:any|some|you)|\b(?:if|once|when|whenever|as soon as) (?:we|we're|we’re|"
                         r"you|you're|both|everyone|they|all)\b", I)
# deliberation about a possible later move, not an endorsement ("we can discuss moving to 2", "open to 2")
# the firm's own punishment: "set 0 to comply with the punishment", "we retaliate with 0" (not "B threatened")
OWN_PUNISH = re.compile(r"\b(?:punish(?:ment|ing)?|retaliat\w*|penalty)\b(?<!threatened to punish)", I)
DISCUSS = re.compile(r"\b(?:discuss|revisit|consider|talk about|think about|explore)(?:ing)?\b(?! further)[^.!?]{0,30}?"
                     r"\b(?:mov|rais|switch|go(?:ing)?\b|shift|bump|increas|higher|later|next)|\bopen to\b|"
                     r"\boption (?:of|to)\b", I)
# "before" is about the past unless a later action follows ("hold 1 before jointly raising to 12")
PAST = re.compile(r"\b(?:was|were|had|last (?:period|round)|previous(?:ly)?|earlier|got|earned|made|ago)\b|"
                  r"(?<!as )\bbefore\b(?!\s+(?:\w+ly\s+)?\w+ing\b|\s+(?:we|you|the next|any|anyone|everyone|all)\b)", I)
# about a later period: the next one ("next period", "from the next round") or later still ("then", "after a few
# rounds", "in the future")
NEXT = re.compile(r"\b(?:next (?:period|round|time|turn)|from (?:the )?next|tomorrow|"
                  r"(?:for )?the next (?:\d+|few|two|three|four|five|several) (?:periods|rounds))\b", I)
AFTER = re.compile(r"^\W*(?:and\s+)?then\b|\b(?:(?:in )?(?:the )?(?:near )?future|upcoming|later|eventually|"
                   r"after (?:this|a few|several|some|that|the next|\d|two|three|four|five|more)|before (?:jointly|"
                   r"collectively|together|we|raising|moving|lifting|shifting)|thereafter|afterwards?|"
                   r"in (?:a few|several|two|three|\d+) (?:more )?(?:periods|rounds)|"
                   r"(?:following|subsequent|remaining|future|later) (?:periods|rounds)|after (?:period|round|the|penalt|"
                   r"punish)|(?:once|when|until) (?:the|this|that|it|its|our|all|every)|resum\w*)\b", I)
THIS_NOW = re.compile(r"(?<!after )(?<!until )(?<!before )(?<!beyond )\bthis (?:period|round)\b|\bthis and\b|"
                      r"\bnow\b|\btoday\b|\bright away\b|\bimmediately\b", I)
# a strong cue names the period; "now" may only mean the talking ("say yes now ... 10 next period")
THIS_PERIOD = re.compile(r"(?<!after )(?<!until )(?<!before )(?<!beyond )\bthis (?:period|round)\b|\bthis and\b|"
                         r"\btoday\b|\bfor now\b|\bright now\b", I)
# staying at a price is about this period: "let's keep at 10", "stay at 10", "hold the price at 10"
STAY = re.compile(r"\b(?:keep|stay|stick|remain|hold|maintain|continu)\w*\b[^.!?\d]{0,25}\d", I)
# clause boundaries within a sentence: commas, dashes, "but", "then", "and" before a new verb phrase
_CLAUSE_START = (r"(?:and|but|then|so|or|we|i|let|let's|let’s|if|once|when|while|unless|otherwise|after|before|next|"
                 r"from|keep|stay|set|move|raise|hold|return|which|this|that|please|everyone|all|it|no|yes|ok|okay|"
                 r"agreed|for)\b")
CLAUSE = re.compile(r"\s*(?:,\s+(?=" + _CLAUSE_START + r")|\s[—–-]\s|—|\bbut\b|(?=\b(?:and )?then\b)|"
                    r"\band\b(?=\s+(?:we|i|let|move|go|raise|switch|shift|bump|increase|keep|stay|set|from|after|next|"
                    r"in|once|if|when|then|plan|hold|stick|maintain|remain|commit|lift|jump|join|match|continue|return|"
                    r"enforce|uphold|will|i'll|we'll|i’ll|we’ll|also)\b))\s*", I)
# a punishment rule: its numbers are the punishment price or its length, not the agreed price
# ("keep the 5-period 0-price penalty rule", "retaliate with 0 for 3 rounds", "reset to 0 for 2 periods")
PUNISH_RULE = re.compile(r"penalt|punish|retaliat|deterr|sanction|grim|trigger|\breset (?:rule|clause|mechanism|plan)|"
                         r"\b(?:reset|revert|drop)\w*\b[^.!?,]{0,25}?\b(?:to|at)\s+(?:price\s+)?(?:0|zero)\b", I)
# carrying out a punishment, not mentioning a rule: "set 0 this period to comply with the penalty", "price 10 as a
# punishment", "keep 0 to stay in line with the punishment rules", "we remain in the two-round penalty"
PUNISH_ACT = re.compile(r"\b(?:as|in|into|due to|under|per|during|of|by|because of|(?:honou?r|comply with|finish|"
                        r"complete|enforce|serve|apply|carry out|follow|uphold|accept|start|begin|continue|implement|"
                        r"compliant with|compliance with|in line with)\w*)\s+"
                        r"(?:\S+\s+){0,4}?(?:penalt|punish|retaliat|sanction)\w*"
                        # ... but not a rule kept for later: "uphold the penalty rule for any deviation"
                        r"(?!\s+(?:rule|threat|mechanism|trigger|clause|plan|for any|if|next|in place|ready|active|remains))|"
                        r"\bto (?:punish|retaliat|penali[sz])|"
                        r"\b(?:penalt\w*|punish\w*) (?:period|phase|round|cycle)", I)
# an evaluation of a move, not a proposal: "Moving to 12 now would harm everyone"
HYPOTHETICAL = re.compile(r"^\W*(?:moving|raising|switching|jumping|shifting|lifting|bumping|dropping|lowering|cutting|"
                          r"undercutting|deviating|going)\b[^.!?]{0,40}?\b(?:would|could|might|may)\b", I)
# staying through later periods still covers this one ("stay at 100 for the remaining periods", "until the end");
# a specific later start does not ("keep 1 next period", "then stay at 2")
STAY_LATER = re.compile(r"\bnext (?:period|round|turn|time)\b|\btomorrow\b|^\W*(?:and\s+)?then\b|\bafter\b|"
                        r"\bonce\b|\bwhen\b|\bfrom (?:the )?next\b", I)
# a solo bid is the sender's own: "I'll set 24 to win the market", "Set 44 to capture ..." (not "no one sets 99 to
# capture the whole market")
OWN_BID = re.compile(r"^\W*(?:set|setting|price|pricing|go|going)\b|\b(?:i|we|my|our|let'?s|let us)\b", I)
NOBODY = re.compile(r"\b(?:no one|nobody|noone|anyone|no firm|none of us)\b", I)
# a plan to win the market alone, not a common price: "set 47 to win the market", "become the sole lowest bidder"
SOLO = re.compile(r"\b(?:win|capture|grab|take|secure|get|aim for)\w*\s+(?:the\s+)?"
                  r"(?:whole\s+|entire\s+|full\s+|all\s+)?(?:market|sales|customers|demand)\b|\bfull market share|"
                  r"\bsole (?:lowest|winner|seller)|\blowest bidder", I)
# a description of rivals' prices, not a proposal: "G and B set 30", "Firm C posted 1"
RIVALS_SET = re.compile(r"(?-i:\b(?:[Ff]irms? )?(?!I\b)[A-Z](?:(?:,\s*|\s+and\s+|\s*&\s*)(?!I\b)[A-Z])*\b)\s+"
                        r"(?:set|sets|chose|posted|priced|went|stayed|kept|moved|raised|dropped|lowered|undercut)\b")
PROPOSE = re.compile(r"\b(?:let'?s|let us|shall we|how about|what about|propose|suggest|we'll|"
                     r"we (?:all )?(?:should|could|can|must|will|keep|stay|set|go|stick)|everyone|every firm|all of us|we all|both of us|agree|keep|stay|stick|"
                     r"maintain|hold|set|charge|price|go (?:with|for|back)|return|back to|i(?:'ll| will| am| shall)|"
                     r"i'm|my price|(?:i|we) (?:am |are )?commit\w*)\b", I)
AGREE = re.compile(r"\b(?:agree[ds]?|agreed|ok(?:ay)?|deal|yes|yep|sure|sounds good|confirm(?:s|ed)?|count me in|i'?m in|"
                   r"i am in|same here|let'?s do (?:it|this|that)|on board|fine (?:by|with) me|works for me|"
                   r"happy to|absolutely|definitely|of course|will do|me too)\b", I)
_THREAT_TALK = (r"punish|retaliat|price war|tit[- ]for[- ]tat|grim trigger|consequence|last chance|"
                r"teach [^.!?]{0,20}lesson|"
                r"\bif (?:any(?:one|body)?|some(?:one|body)|you|a firm|any (?:firm|of you)|firm [a-z]{1,2})\b[^.!?]{0,60}?"
                r"\b(?:undercut|deviat|cheat|defect|go(?:es)? (?:below|lower|under)|lower|drop|break|betray|price below)|"
                r"undercut (?:you|them|him|her|whoever|anyone)|race to the bottom|"
                r"\b(?:i|we)(?:'ll| will) (?:match|follow) (?:any|the lowest|your)")
# "we will all go to 0", "hitting 0": a threat, but also how a firm states a price of 0 or 1 for this period
_LOW_PRICE_TALK = (r"\b(?:i|we)(?:'ll| will| would| shall)? (?:all )?(?:go|drop|cut|set|price|respond|match)[^.!?\d]{0,20}?"
                   r"\b(?:to|at) (?:0|zero|1)\b|\bhit(?:ting)? (?:0|zero)\b")
THREAT = re.compile(f"{_THREAT_TALK}|{_LOW_PRICE_TALK}", I)
THREAT_TALK = re.compile(_THREAT_TALK, I)  # ends an endorsement clause; "I'll set my price at 0" does not
TURNS = re.compile(r"take turns|taking turns|\brotat|\balternat|\bin turns?\b|turn[- ]taking|"
                   r"each (?:of us |firm )?(?:gets|takes) (?:a|its|their|one) turn|"
                   r"one (?:of us|firm) (?:at|sets?|prices? at|goes(?: to)?) 99|(?:your|my|his|her|their) turn (?:to|at)", I)
ACCUSE = re.compile(
    r"(?i:\b(?:you|someone|somebody|who|why did|firm\s+[a-z]{1,2})\b[^.!?]{0,40}?\b(?:undercut|cheated|betrayed|"
    r"defected|broke|deviated|lied|went (?:below|lower|under)|dropped|backstabbed))|"
    r"(?i:\b(?:betray(?:ed|al)?|cheat(?:ed|er|ing)|liar|not fair|unfair|you said|you promised|broke (?:the|our|your) "
    r"(?:deal|agreement|promise|word)))|\b(?!I\b)[A-Z]{1,2}\b (?:undercut|cheated|defected|deviated|dropped|broke)")
APOLOGY = re.compile(r"\bsorry\b|apolog|my (?:bad|mistake)|mistake on my|won'?t happen again", I)
NEGATION = re.compile(r"(?:\bnot|n't|\bnever|\bno|\bavoid|\bwithout|\bagainst)\W+(?:\w+\W+){0,2}$", I)
# a negated pricing verb followed by a price is a refusal ("we should not set 100", "I won't price at 100"),
# unless it only rules out lower prices ("don't go below 100")
_NOT_LOWER = r"(?![^.!?;,]{0,20}?\b(?:below|under|lower|beneath|less|cheaper|down)\b)"
REFUSE = re.compile(r"\b(?:disagree|refuse|reject|no deal|(?:do not|don'?t|won'?t|will not|can'?t|cannot) (?:agree|accept|"
                    r"do (?:that|this|it)|commit))\b|"
                    rf"\b(?:(?:do not|don'?t|won'?t|will not|can'?t|cannot)|not (?:going to|gonna)) (?:set|go|price|charge)\b"
                    rf"{_NOT_LOWER}|"
                    r"(?:\bnot|\bnever|\bcannot|\b(?:do|does|did|wo|should|would|could|ca|must|sha)n['’]?t)\s+"
                    r"(?:(?:going to|gonna|to|even|really|just|all|both|ever)\s+)?"
                    rf"(?:set|go|price|charge|keep|stay|stick|post)\b{_NOT_LOWER}(?=[^.!?;\d]{{0,25}}\d)|"
                    r"^\W*no\b(?! problem| worries| doubt)", I)
DEVIATE = re.compile(r"undercut|defect|deviat|cheat|slightly (?:lower|below|under)|below (?:the )?(?:agreed|others|them)|"
                     r"grab|steal|capture (?:the )?(?:whole |entire |full )?market|exploit|take (?:the )?(?:whole|entire|"
                     r"full) (?:market|profit)|betray", I)
_KEEP_AUX = (r"(?:'ll|’ll|\s+will|\s+shall|\s+am|\s+are|\s+going|\s+to|\s+must|\s+also|\s+just|\s+simply|"
             r"\s+continue|\s+have|\s+has|\s+agreed|\s+plan|\s+intend|\s+decide|\s+choose|\s+therefore|\s+thus|"
             r"\s+still|\s+again|\s+each|\s+all)*")


def _other_time(sent: str, a: int, b: int, period: int | None) -> bool:
    """Whether the statement at [a, b) of a notes sentence is about the past or a later period, not this one:
    its clause (with a short lead-in such as "Last period,") has a past or later cue, or names another period,
    and no this-period cue ("Last period I set 100", "Next period I will keep 100")."""
    lo, hi = _notes._clause_bounds(sent, a, b)
    prev = max((m.end() for m in _notes._CLAUSE.finditer(sent, 0, max(lo - 1, 0))), default=0)
    if lo and len(sent[prev:lo].split()) <= 4:  # "Last period, I set 100"
        lo = prev
    span = sent[lo:hi]
    this_ref, other_ref = _notes._period_refs(span, period)
    if _notes.THIS_PERIOD.search(span) or this_ref:
        return False
    return bool(_notes.PAST.search(span) or _notes.LATER.search(span) or other_ref)


def keeps_price(notes: str | None, price: int, period: int | None = None) -> bool:
    """Whether the notes state in the first person that the firm keeps or sets `price` this period ("I stay at the
    agreed collusive price of 11", "I will keep 1 ECU", "myself have agreed to set price 24", "By staying at 24
    we"), in a sentence that names no lower price and no deviation ("I will set 10 instead of 11" does not keep
    11). Statements about the past or a later period don't count ("Last period I set 100. This period I will set
    99." does not keep 100)."""
    rx = re.compile(rf"(?:\b(?:I|we|myself){_KEEP_AUX}|\bby)\s+(?:stay|keep|remain|stick|set|choose|maintain|hold|post|"
                    rf"go with)\w*\b(?:\s+(?:at|to|with|on|the|my|our|price|agreed|collusive|cooperative|of))*\s*"
                    rf"(?<![\d.,])\b{price}\b(?![\d,.]\d|\s*(?:ECU\s+)?(?:profit|periods?|rounds?|units?|%))", I)
    for sent in SENT.split((notes or "").replace("’", "'")):
        if any(not _other_time(sent, m.start(), m.end(), period) for m in rx.finditer(sent)) \
                and not CONDITIONAL.search(sent) and not DEVIATE.search(sent) \
                and not re.search(r"\binstead\b", sent, I) and all(v >= price for v in prices_in(sent)):
            return True
    return False


def _sentence_before(text: str, start: int, width: int = 60) -> str:
    """Up to `width` characters before `start`, within the same sentence."""
    before = text[max(0, start - width):start]
    cut = max(before.rfind(c) for c in ".!?;\n")
    return before[cut + 1:]


def _price_list(groups: list[str]) -> bool:
    """Whether comma groups read as a list of prices: every group a price 0-100 without a leading zero."""
    return all(g.isdigit() and int(g) <= 100 and (g == "0" or not g.startswith("0")) for g in groups)


def prices_in(text: str, end: int | None = None) -> list[int]:
    """Integers 0-100 in a price-like context, in order of appearance (only those starting before `end`)."""
    out = []
    text = text or ""
    for m in NUM.finditer(text):
        if end is not None and m.start() >= end:
            break
        raw, after = m.group(1), text[m.end():m.end() + 25]
        if NOT_BEFORE.search(text[max(0, m.start() - 30):m.start()]) or NOT_AFTER.search(after):
            continue
        if "," in raw:  # "99,100" (a list of prices) vs "29,700" (a thousands number)
            groups = raw.split(",")
            sentence = _sentence_before(text, m.start(), 200) + text[m.end():m.end() + 60].split(".")[0]
            if _price_list(groups) and not PROFIT_CTX.search(sentence) and not ECU_AFTER.search(after):
                out.extend(int(g) for g in groups)
            continue  # otherwise a thousands number: above 100, not a price
        v = int(raw)
        if v > 100:
            continue
        if ECU_AFTER.search(after) and (ECU_PROFIT_AFTER.search(after)
                                        or PROFIT_CTX.search(_sentence_before(text, m.start()))):
            continue  # "we each earn 75 ECU", "75 ECU each": a profit amount
        out.append(v)
    return out


def _unnegated(pattern: re.Pattern, text: str) -> re.Match | None:
    """The first match of `pattern` not preceded (within three words) by a negation."""
    return next((m for m in pattern.finditer(text or "") if not NEGATION.search(text[max(0, m.start() - 30):m.start()])),
                None)


def has_unnegated(pattern: re.Pattern, text: str) -> bool:
    """Whether `pattern` matches somewhere not preceded (within three words) by a negation."""
    return any(not NEGATION.search(text[max(0, m.start() - 30):m.start()]) for m in pattern.finditer(text or ""))


def _clauses(sentence: str) -> list[str]:
    return [c for c in CLAUSE.split(sentence) if c and c.strip()]


def endorsement(text: str) -> tuple[int | None, int | None, int | None, int | None]:
    """(endorsed price, its time tier, this-period punishment price, this-period solo bid) of a message.

    Tier 0: the clause names this period ("set 1 this period") or staying at a price ("keep at 10", unless it
    continues a later period named earlier: "Next period we set 30. We keep 30 for three rounds" is tier 2); 1: no time
    cue; 2: the next period; 3: later ("then", "after a few rounds"). The lowest tier wins, the last clause within it.
    A clause that states this period's price as a punishment ("keep 1 this period to finish the punishment") or as
    a bid to win the market alone ("I'll set 24 this period to capture the whole market") is not an endorsement:
    unless the message endorses a price for this period elsewhere, it endorses nothing (its later plans don't count
    either) and that price is returned instead."""
    text = (text or "").replace("’", "'")
    sents = [s for s in SENT.split(text) if s.strip()]
    usable = [s for s in sents if not any(rx.search(s) for rx in (REFUSE, TURNS, DISCUSS))]
    # clause by clause: a conditional, threat or accusation clause ends the sentence's usable part ("We keep 100,
    # but if you undercut, I set 0"); a clause carrying out a punishment, about the past, describing rivals'
    # prices, evaluating a move or planning to win the market alone is skipped ("We were at 10 last period, let's
    # stay at 10")
    cued, clauses, punish_now, solo_now = [], [], None, None  # clauses: the prices of each usable clause
    floor = 1  # a later-period clause makes what follows later too, until this period is named again
    for s in usable:
        done = False
        for c in _clauses(s):
            if done:
                break
            punish = has_unnegated(PUNISH_RULE, c)
            stop = [m.start() for m in (CONDITIONAL.search(c), None if punish else _unnegated(THREAT_TALK, c),
                                        ACCUSE.search(c)) if m is not None]
            if stop:  # keep what comes before the condition: "We keep 5 this period, with 0 if any firm deviates"
                done = True
                c = c[:min(stop)]
                if not c.strip():
                    break
                punish = has_unnegated(PUNISH_RULE, c)
            later = 3 if AFTER.search(c) else 2 if NEXT.search(c) else 1
            # "now" alone names this period only outside talk of a later one ("Ready to set 10 now" after "10
            # for the next 3 periods" is about the coalition), except in a staying statement, where it returns to
            # the present ("Next period we will set 30. We keep 1 now.")
            now = bool(THIS_PERIOD.search(c)) or (bool(THIS_NOW.search(c)) and later == 1
                                                  and (floor == 1 or bool(STAY.search(c))))
            tier = 0 if now else max(floor, later)
            inherited = floor > 1  # a later time named earlier ("Next period we set 30. We keep 30 for 3 rounds")
            floor = 1 if now else max(floor, tier)  # "Next period, we go to 30. Let's all raise to 30"
            # "let's stay at 1 ... let's move to 2": about now, unless it continues a plan for a later period
            if STAY.search(c) and not STAY_LATER.search(c) and not inherited:
                tier = 0
            ps = prices_in(c)
            if punish:
                if PUNISH_ACT.search(c):  # this period's price is a punishment, not an agreement
                    if tier <= 1 and ps and punish_now is None:
                        punish_now = ps[0]
                    continue
                # a punishment rule mentioned alongside a price ("we stay at 10 this period, respecting the 0-price
                # rule"): only prices before the rule count ("keep the 5-period 0-price penalty rule": none)
                ps = prices_in(c, _unnegated(PUNISH_RULE, c).start())
            if has_unnegated(SOLO, c) and OWN_BID.search(c) and not NOBODY.search(c):
                if tier <= 1 and ps and solo_now is None:
                    solo_now = ps[0]
                continue
            if PAST.search(c) or RIVALS_SET.search(c) or HYPOTHETICAL.search(c):
                continue
            clauses.append((tier, ps))
            if ps and (PROPOSE.search(c) or has_unnegated(AGREE, c)):
                cued.append((tier, ps[0]))
    # a firm's own bid to win the market alone beats urging others ("I will set 10 this period to capture the
    # market ... Everyone else should stay at 11")
    if solo_now is not None or (punish_now is not None and not any(t == 0 for t, _ in cued)):
        return None, None, punish_now, solo_now
    if cued:
        best = min(t for t, _ in cued)
        return [p for t, p in cued if t == best][-1], best, None, None
    rejected = {p for s in sents if REFUSE.search(s) for p in prices_in(s)}  # "We should not set 100. 100?"
    distinct = {p for _, ps in clauses for p in ps} - rejected
    if len(distinct) != 1:
        return None, None, None, None
    p = distinct.pop()
    return p, min(t for t, ps in clauses if p in ps), None, None


def endorsed_price(text: str) -> int | None:
    return endorsement(text)[0]


def code_message(text: str | None) -> dict:
    t = (text or "").replace("’", "'")
    nonconditional = " ".join(s for s in SENT.split(t) if not CONDITIONAL.search(s))
    e, tier, punish_now, solo_now = endorsement(t)
    refusal = bool(REFUSE.search(t))
    return {"prices": prices_in(t), "endorsed": e, "endorsed_tier": tier, "punish_now": punish_now,
            "solo_now": solo_now, "this_now": bool(THIS_PERIOD.search(t)), "proposal": e is not None and bool(PROPOSE.search(t)),
            "agreement": not refusal and has_unnegated(AGREE, t), "refusal": refusal,
            "threat": has_unnegated(THREAT, t), "turns": bool(TURNS.search(t)),
            "accusation": bool(ACCUSE.search(nonconditional)), "apology": bool(APOLOGY.search(t))}


def code_messages(chat: pd.DataFrame) -> pd.DataFrame:
    """Per-message coding (passes dropped)."""
    msgs = chat[~chat["passed"].astype(bool)].copy() if len(chat) else chat.copy()
    coded = pd.DataFrame([code_message(m) for m in msgs.get("message", [])], index=msgs.index,
                         columns=list(code_message("")))
    msgs = pd.concat([msgs, coded], axis=1).astype({"endorsed": "Int64", "endorsed_tier": "Int64",
                                                    "punish_now": "Int64", "solo_now": "Int64"})
    msgs["prices"] = msgs["prices"].apply(lambda v: " ".join(map(str, v)) if isinstance(v, list) else "")
    return msgs


def _effective(msgs: list[dict]) -> list[dict]:
    """Messages with endorsements of a later period dropped when the period's chat also speaks of this period:
    once firms tell "this period" from "next period", a plan for the next period is not this period's price
    ("I'll stay at 1 this period" ... "I commit to 10 next period"). So is an endorsement with no time cue of a
    price the period's chat endorses only for later ("I confirm the 10-ECU coalition" among "10 next period")."""
    if not any(m.get("this_now") for m in msgs):
        return msgs
    tiers: dict[int, set[int]] = {}
    for m in msgs:
        if m["endorsed"] is not None and m.get("endorsed_tier") is not None:
            tiers.setdefault(m["endorsed"], set()).add(m["endorsed_tier"])
    later_only = {p for p, ts in tiers.items() if min(ts) >= 1 and max(ts) >= 2}
    out = []
    for m in msgs:
        tier = m.get("endorsed_tier")
        if m["endorsed"] is not None and tier is not None and (tier >= 2 or m["endorsed"] in later_only):
            continue  # neither an endorsement nor a bare "ok" for this period
        out.append(m)
    return out


def support(msgs: list[dict], firms: list[str], prev_agreed: int | None) -> dict[str, int | None]:
    """Price each firm backs at the end of a period's chat (`msgs`: dicts with round, firm, endorsed,
    agreement, sorted by round). A this-period punishment price or solo bid (`punish_now`, `solo_now`) clears
    the firm's backing."""
    backed: dict[str, int | None] = {f: None for f in firms}
    msgs = _effective(msgs)
    for m in msgs:
        if m["endorsed"] is not None:
            backed[m["firm"]] = m["endorsed"]
        elif m.get("refusal") or m.get("punish_now") is not None or m.get("solo_now") is not None:
            backed[m["firm"]] = None
        elif m["agreement"]:
            earlier = [x["endorsed"] for x in msgs if x["round"] < m["round"] and x["firm"] != m["firm"]
                       and x["endorsed"] is not None]
            ref = earlier[-1] if earlier else prev_agreed
            if ref is not None:
                backed[m["firm"]] = ref
    return backed


def agreed_price(backed: dict[str, int | None]) -> tuple[int | None, int]:
    c = Counter(v for v in backed.values() if v is not None)
    if not c:
        return None, 0
    price, k = max(c.items(), key=lambda kv: (kv[1], kv[0]))
    return (price, k) if k > len(backed) / 2 else (None, k)


def code_period(msgs: list[dict], prices: dict[str, int], prev_agreed: int | None) -> dict:
    """Agreement and compliance for one Talk period."""
    firms = sorted(prices)
    backed = support(msgs, firms, prev_agreed)
    agreed, k = agreed_price(backed)
    out = {"agreed_price": agreed, "support": k, "agreement": agreed is not None, "backed": backed,
           "kept": None, "undercutters": [], "above": [], "max_undercut": None}
    if agreed is not None:
        under = [f for f in firms if prices[f] < agreed]
        out |= {"kept": all(prices[f] == agreed for f in firms), "undercutters": under,
                "above": [f for f in firms if prices[f] > agreed],
                "max_undercut": max((agreed - prices[f] for f in under), default=None)}
    return out


def code_periods(chat_msgs: pd.DataFrame, chat: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """Per Talk period: chat activity, agreement, compliance, punishment, threats and deception signals."""
    rows = []
    talk_fp = fp[fp["phase"] == "Talk"]
    msg_groups = dict(tuple(chat_msgs.groupby(["market_id", "part", "period"]))) if len(chat_msgs) else {}
    chat_groups = dict(tuple(chat.groupby(["market_id", "part", "period"]))) if len(chat) else {}
    for (mid, part), fpp in talk_fp.groupby(["market_id", "part"]):
        prev_agreed, threatened = None, False
        periods = sorted(fpp["period"].unique())
        by_t = {t: g for t, g in fpp.groupby("period")}
        coded = []
        for t in periods:
            g = by_t[t]
            prices = dict(zip(g["firm"], g["price"]))
            m = msg_groups.get((mid, part, t))
            mlist = m.sort_values(["round", "firm"]).to_dict("records") if m is not None else []
            for x in mlist:  # NaN -> None, float -> int after the DataFrame round trip
                for k in ("endorsed", "endorsed_tier", "punish_now", "solo_now"):
                    x[k] = None if pd.isna(x.get(k)) else int(x[k])
            c = chat_groups.get((mid, part, t))
            threatened = threatened or any(x["threat"] for x in mlist)
            r = code_period(mlist, prices, prev_agreed)
            notes = dict(zip(g["firm"], g["notes"]))
            dec, dec_priced = [], []
            if r["agreement"]:
                for f, v in r["backed"].items():
                    if v != r["agreed_price"]:
                        continue
                    nt = notes.get(f, "")  # price-decision notes, written after the chat
                    stated, context = stated_own_price_context(nt, t)
                    keeps = keeps_price(nt, v, t)  # "I stay at the agreed price of 11": a lower price is a rival's
                    lower = stated is not None and stated < v and not has_unnegated(OWN_PUNISH, context) and not keeps
                    # a vague plan ("I'll keep 1 to capture the market") is no deviation when the stated price is P
                    if lower or (deviation_plan(nt, t) and (stated is None or stated < v) and not keeps):
                        dec.append(f)
                        if prices[f] < v:
                            dec_priced.append(f)
            posted = c[~c["passed"]] if c is not None else None
            rows_used = int(posted["round"].max()) if posted is not None and len(posted) else 0
            coded.append({c0: g[c0].iloc[0] for c0 in KEY_COLS} | {
                "part": part, "period": t, "gperiod": g["gperiod"].iloc[0], "sp": min(prices.values()),
                "rounds_used": rows_used, "messages": 0 if posted is None else len(posted),
                "passes": 0 if c is None else int(c["passed"].sum()),
                "chars": 0 if posted is None else int(posted["chars"].sum()),
                "proposals": sum(x["proposal"] for x in mlist), "threat": any(x["threat"] for x in mlist),
                "threatened_so_far": threatened, "turns_talk": any(x["turns"] for x in mlist),
                "accusation": any(x["accusation"] for x in mlist), "apology": any(x["apology"] for x in mlist),
                "agreement": r["agreement"], "agreed_price": r["agreed_price"], "support": r["support"],
                "kept": r["kept"], "broken": None if r["kept"] is None else not r["kept"],
                "undercutters": " ".join(r["undercutters"]), "n_undercut": len(r["undercutters"]),
                "above": " ".join(r["above"]), "n_above": len(r["above"]), "max_undercut": r["max_undercut"],
                "undercutters_backed": sum(r["backed"][f] == r["agreed_price"] for f in r["undercutters"]),
                "deception_notes": " ".join(dec), "deception_priced": " ".join(dec_priced),
                "_prices": prices, "_under": r["undercutters"]})
            prev_agreed = r["agreed_price"]  # a bare "ok" can only refer to the previous period's agreement
        for i, row in enumerate(coded):  # punishment after a breach with undercutting
            nxt = coded[i + 1] if i + 1 < len(coded) and coded[i + 1]["period"] == row["period"] + 1 else None
            breach = bool(row["n_undercut"])
            row["punished_next"] = None if not breach or nxt is None else any(
                nxt["_prices"][f] < row["_prices"][f] for f in row["_prices"] if f not in row["_under"])
            row["sp_next_lt99"] = None if not breach or nxt is None else nxt["sp"] < 99
            rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    return pd.DataFrame(rows)


def chat_summary(periods: pd.DataFrame, chat: pd.DataFrame) -> pd.DataFrame:
    """Per cell and part: agreement, compliance, punishment, deception and activity rates."""
    if periods.empty:
        return pd.DataFrame()
    keys = ["model", "n", "ordering", "part"]
    p = periods.assign(
        agreed100=periods["agreement"] & periods["agreed_price"].eq(100), kept_=periods["kept"].eq(True),
        breach_under=periods["n_undercut"] > 0,
        breach_above_only=periods["agreement"] & periods["n_undercut"].eq(0) & periods["n_above"].gt(0),
        dn=periods["deception_notes"].str.len() > 0, dp=periods["deception_priced"].str.len() > 0)
    out = p.groupby(keys).agg(
        talk_periods=("period", "count"), agreements=("agreement", "sum"), agreed_100=("agreed100", "sum"),
        kept=("kept_", "sum"), breaches_undercut=("breach_under", "sum"),
        breaches_above_only=("breach_above_only", "sum"), periods_deception_notes=("dn", "sum"),
        periods_deception_priced=("dp", "sum"), mean_rounds_used=("rounds_used", "mean"),
        messages_per_period=("messages", "mean"), chars_per_period=("chars", "mean"),
        share_threat=("threat", "mean"), share_turns_talk=("turns_talk", "mean"),
        share_accusation=("accusation", "mean"), share_apology=("apology", "mean")).reset_index()
    out["share_agreement"] = out["agreements"] / out["talk_periods"]
    out["share_kept"] = out["kept"] / out["agreements"].replace(0, float("nan"))
    rate = lambda s: s.dropna().astype(float).mean()
    br = p[p["breach_under"]].groupby(keys).agg(
        share_punished_next=("punished_next", rate), share_sp_next_lt99=("sp_next_lt99", rate),
        share_threatened_before=("threatened_so_far", "mean"), mean_max_undercut=("max_undercut", "mean"),
        breaches_by_backers=("undercutters_backed", lambda s: int((s > 0).sum())))
    out = out.merge(br.reset_index(), on=keys, how="left")
    talk_chat = chat[chat["phase"] == "Talk"]
    if len(talk_chat):
        a = talk_chat.assign(posted=~talk_chat["passed"]).groupby(keys).agg(
            chat_turns=("passed", "size"), pass_rate=("passed", "mean"), n_messages=("posted", "sum"),
            truncated=("truncated", "sum")).reset_index()
        out = out.merge(a, on=keys, how="left")
        out["messages_per_firm_period"] = out["n_messages"] / (out["talk_periods"] * out["n"])
    return out
