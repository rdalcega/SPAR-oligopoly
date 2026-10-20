"""Human benchmarks from Fonseca & Normann (2012), as transcribed in ``fonseca-normann-summary.md`` §7.

Every number here comes from the summary, which was checked against the working paper (DICE DP 65, Aug 2012;
page/table references are to that version). The journal version (EER 56(8), 2012) has not been checked, so
small numeric differences from it are possible (see ``CAVEATS``). Nothing here is estimated or derived; where
the summary only describes something qualitatively it is stored as a string.

Measures (summary §5), so analysis code can replicate them exactly:

- ``selling_price(prices)`` = min over the market's posted prices (the market price). Industry profit is
  300 × selling price, so "profit" and "selling price" tables are the same measure.
- Posted price = each firm's own price, reported separately (only for NoTalk means, fn. 10).
- Unit of observation = one market. A market's mean selling price over the window is one observation, so a
  treatment cell has 6 observations; reported SDs are across those 6 market means.
- Window: periods 1-5 are dropped (``EXCLUDED_EARLY_PERIODS``). Whether Part 2's first five periods are also
  dropped is ambiguous (``WINDOW_AMBIGUITY``); report both.
- Price-state distribution (Table 2): shares of market-periods in p = 100, p = 99, 2 ≤ p ≤ 98, p ≤ 1, from
  period 6 on (same Part 2 ambiguity).
- Markov analysis (Tables 3-4): Talk phase only, ALL periods, three states 100 / 99 / <99 of the selling price;
  Shorrocks mobility φ = (k − tr(M)) / (k − 1), k = 3.
- Gain from talking = Talk minus NoTalk mean selling price, per market.
- Defection = all firms at 100 in t−2 and some firm at 99 in t−1; "kept" means selling price ≥ 99 in t;
  "punishment" means any selling price < 99 in t (not a lasting price war).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Design constants and measure definitions (summary §1, §3, §5)
# ---------------------------------------------------------------------------

NS: tuple[int, ...] = (2, 4, 6, 8)
MARKETS_PER_CELL = 6  # p. 6 text and the 168-subject count; WP Table 1 prints "4" (likely typo)
PART1_PERIODS = 29  # periods 1-29 in the paper's continuous numbering
PART2_PERIODS = 24  # periods 30-53
CONTINUATION_PROB = 5 / 6
EXCLUDED_EARLY_PERIODS = 5  # periods 1-5 dropped from selling-price averages and Table 2

WINDOW_AMBIGUITY = (
    "Summary §5/§6 item 12: the paper drops 'periods 1 to 5'. Figure 3 and §4.4 number periods 1-53 across "
    "both parts (Talk = periods 30-53), which would mean only Part 1's first five are dropped; but §5.3 chat "
    "quotations restart at 'period one' for Part 2, which suggests per-part numbering. Table 2 ('from period "
    "six onwards') has the same ambiguity. Report Part 2 both with and without its first five periods."
)

Part = Literal[1, 2]
Phase = Literal["NoTalk", "Talk"]
State3 = Literal["100", "99", "<99"]
STATES3: tuple[State3, ...] = ("100", "99", "<99")


def selling_price(prices: list[int] | tuple[int, ...] | dict[str, int]) -> int:
    """Market (selling) price as the paper defines it: the lowest posted price."""
    values = list(prices.values()) if isinstance(prices, dict) else list(prices)
    return min(values)


def price_bin4(p: int) -> Literal["100", "99", "2-98", "<=1"]:
    """Table 2 bins of the selling price."""
    if p == 100:
        return "100"
    if p == 99:
        return "99"
    if p <= 1:
        return "<=1"
    return "2-98"


def markov_state(p: int) -> State3:
    """Table 3/4 states of the selling price."""
    if p == 100:
        return "100"
    if p == 99:
        return "99"
    return "<99"


def shorrocks(matrix: dict[State3, dict[State3, float]]) -> float:
    """Shorrocks mobility index φ = (k − tr(M)) / (k − 1)."""
    k = len(matrix)
    return (k - sum(matrix[s][s] for s in matrix)) / (k - 1)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellStat:
    """Mean (and SD across the 6 market means, if reported) for one treatment cell."""

    mean: float
    sd: float | None
    source: str
    window: str = "Part 1 from period 6; Part 2 window ambiguous (see WINDOW_AMBIGUITY)"


@dataclass(frozen=True)
class Test:
    """A reported statistical test."""

    description: str
    p: float | None  # None when only 'not significant' / a bound is given
    p_bound: str | None = None  # e.g. "<0.001", "<0.05"
    statistic: float | None = None  # e.g. Spearman rho
    source: str = ""


# ---------------------------------------------------------------------------
# Main design (NoTalk -> Talk), summary §7 table "Main design"
# ---------------------------------------------------------------------------

_MAIN_SRC = "Summary §7 'Main design' table (WP Results 1-4); SD across 6 market means"

MAIN_NOTALK_SELLING: dict[int, CellStat] = {
    2: CellStat(50.4, 21.2, _MAIN_SRC, "Part 1, periods 6-29"),
    4: CellStat(6.0, 0.6, _MAIN_SRC, "Part 1, periods 6-29"),
    6: CellStat(2.5, 2.0, _MAIN_SRC, "Part 1, periods 6-29"),
    8: CellStat(1.1, 0.3, _MAIN_SRC, "Part 1, periods 6-29"),
}
MAIN_TALK_SELLING: dict[int, CellStat] = {
    2: CellStat(93.5, 9.3, _MAIN_SRC),
    4: CellStat(81.3, 30.1, _MAIN_SRC),
    6: CellStat(65.0, 42.9, _MAIN_SRC),
    8: CellStat(55.2, 35.3, _MAIN_SRC),
}
#: Talk minus NoTalk selling price (per-market gains averaged). Reported values; may differ from the
#: difference of the rounded means by ~0.1.
MAIN_GAIN_FROM_TALK: dict[int, float] = {2: 43.2, 4: 75.3, 6: 62.5, 8: 54.1}
#: NoTalk mean posted (own) price, all firms; fn. 10. Window assumed same as selling price (not stated).
MAIN_NOTALK_POSTED: dict[int, float] = {2: 54.23, 4: 18.18, 6: 23.41, 8: 29.36}

#: Footnote 14: Talk profits (in selling-price units) and gains using, most likely, ONLY the last 5 Talk
#: periods (wording "if we include the last 5 periods ... only" is ambiguous; summary §7 argues for 'only').
FN14_WINDOW = "Most likely only the last 5 periods of the Talk phase (periods 49-53); wording ambiguous"
FN14_TALK_PROFIT: dict[int, float] = {2: 90.57, 4: 62.97, 6: 58.63, 8: 38.73}
FN14_GAIN: dict[int, float] = {2: 40.21, 4: 56.95, 6: 56.09, 8: 37.64}

MAIN_TESTS: tuple[Test, ...] = (
    Test("NoTalk: Spearman price vs n over 24 market means", None, "<0.001", -0.93, "Result 1"),
    Test("NoTalk: every pair of n differs (Mann-Whitney)", None, "<0.05", None, "Result 1"),
    Test("Talk: Spearman price vs n over 24 market means", 0.011, None, -0.51, "Result 2"),
    Test("Talk: n=2 vs n=8 (MW); only significant pair", 0.036, None, None, "Result 2"),
    Test("Talk > NoTalk in all 24 markets (Wilcoxon signed-rank)", None, "<0.001", None, "Result 3"),
    Test("Gain from talking n=4 vs n=2 (MW)", 0.055, None, None, "Result 4"),
    Test("Gain from talking n=6 vs n=2 (MW), n.s.", 0.337, None, None, "Result 4"),
    Test("Gain from talking n=8 vs n=2 (MW), n.s.", 0.631, None, None, "Result 4"),
    Test("Talk time trend n=8: sign test on 6 market Spearman(price, period)", 0.031, None, None, "§4.4"),
)
TALK_MARKETS_HIGHER_THAN_NOTALK = (24, 24)  # all 24 markets price higher in Talk (Result 3)

MAIN_QUALITATIVE: tuple[str, ...] = (
    "NoTalk: only duopolies price clearly above cost; for n >= 4 prices sit near marginal cost (Result 1).",
    "NoTalk: n > 2 markets converge to near Nash within about 5 periods.",
    "Talk: relationship with n roughly linear, unlike the NoTalk cliff between n=2 and n=4.",
    "Eight firms with Talk end up about where duopolies are without Talk (fn. 11).",
    "Gain from talking is an inverted U in n (n=4 highest) (Result 4).",
    "Talk (periods 30-53): prices trend down in all n; significant only for n=8.",
    "Figure 2 (explicit cartel / tacit / competition split at hypothetical fine 50): share of explicit "
    "cartels inverted-U in n, n=4 highest; no numbers given.",
)

# ---------------------------------------------------------------------------
# Controls (n = 4), summary §7 "Controls"
# ---------------------------------------------------------------------------

_CTRL_SRC = "Summary §7 'Controls (n = 4)' table (WP Result 5); SD across 6 market means"

#: Keyed by (treatment, part). Same window convention as the main design (assumed; summary §7 preamble).
CONTROLS_SELLING: dict[tuple[str, Part], tuple[Phase, CellStat]] = {
    ("Talk->NoTalk", 1): ("Talk", CellStat(89.0, 8.7, _CTRL_SRC, "Part 1, periods 6-29")),
    ("Talk->NoTalk", 2): ("NoTalk", CellStat(67.0, 33.8, _CTRL_SRC)),
    ("NoTalk->NoTalk", 1): ("NoTalk", CellStat(2.84, 3.63, _CTRL_SRC, "Part 1, periods 6-29")),
    ("NoTalk->NoTalk", 2): ("NoTalk", CellStat(5.69, 8.42, _CTRL_SRC)),
}

CONTROL_TESTS: tuple[Test, ...] = (
    Test("Hysteresis: Talk->NoTalk Part 2 (67.0) vs main NoTalk n=4 (6.0), MW", 0.055, None, None, "Result 5"),
    Test("No order effect on Talk: 81.3 (main) vs 89.0 (Talk->NoTalk), MW", 0.873, None, None, "§7"),
)
CONTROL_QUALITATIVE: tuple[str, ...] = (
    "One Talk->NoTalk market priced higher in its NoTalk phase than in its Talk phase.",
    "Talk->NoTalk: prices trend down in the NoTalk phase; no trend in the Talk phase.",
    "NoTalk->NoTalk prices do not differ from main-design NoTalk n=4 prices (not just learning); "
    "no p-value given in the summary.",
)

# ---------------------------------------------------------------------------
# Table 2: distribution of selling prices, % of market-periods, period 6 onward
# ---------------------------------------------------------------------------

TABLE2_WINDOW = "From period 6 on (Part 2 numbering ambiguous; see WINDOW_AMBIGUITY)"
TABLE2_SOURCE = "Summary §7 'Table 2' (WP Table 2)"
Bin4 = Literal["100", "99", "2-98", "<=1"]
BINS4: tuple[Bin4, ...] = ("100", "99", "2-98", "<=1")

#: TABLE2[(phase, n)][bin] = percent of market-periods.
TABLE2: dict[tuple[Phase, int], dict[Bin4, float]] = {
    ("NoTalk", 2): {"100": 0.0, "99": 2.1, "2-98": 97.9, "<=1": 0.0},
    ("NoTalk", 4): {"100": 0.0, "99": 0.0, "2-98": 64.9, "<=1": 36.1},
    ("NoTalk", 6): {"100": 0.0, "99": 0.0, "2-98": 35.0, "<=1": 66.0},
    ("NoTalk", 8): {"100": 0.0, "99": 0.0, "2-98": 5.6, "<=1": 94.4},
    ("Talk", 2): {"100": 60.5, "99": 14.9, "2-98": 24.6, "<=1": 0.0},
    ("Talk", 4): {"100": 74.6, "99": 2.6, "2-98": 11.2, "<=1": 11.4},
    ("Talk", 6): {"100": 23.7, "99": 28.9, "2-98": 28.1, "<=1": 19.3},
    ("Talk", 8): {"100": 12.3, "99": 23.7, "2-98": 45.6, "<=1": 18.4},
}
#: Columns whose printed values do not sum to 100 (as transcribed): NoTalk n=4 and n=6 sum to 101.0.
TABLE2_SUM_ANOMALIES: frozenset[tuple[Phase, int]] = frozenset({("NoTalk", 4), ("NoTalk", 6)})

TABLE2_TESTS: tuple[Test, ...] = (
    Test("NoTalk: share of p<=1 rises with n (Spearman)", None, "<0.001", 0.867, "§7 under Table 2"),
    Test("Talk: share of p>=99 falls with n (Spearman)", 0.054, None, -0.400, "§7 under Table 2"),
)
TABLE2_NOTE = (
    "In Talk, p = 99 always coincides with at least one firm at 100: state 99 is a defection from all-100, "
    "taking turns, or a coordination failure, never a common price of 99."
)

# ---------------------------------------------------------------------------
# Table 3: Talk-phase Markov transitions (all Talk periods), Table 4: state shares
# ---------------------------------------------------------------------------

MARKOV_WINDOW = "Talk phase, all periods"
TABLE3_SOURCE = "Summary §7 'Table 3' (WP Table 3); rows = from-state, entries are probabilities"

TABLE3: dict[int, dict[State3, dict[State3, float]]] = {
    2: {
        "100": {"100": 0.92, "99": 0.02, "<99": 0.06},
        "99": {"100": 0.04, "99": 0.87, "<99": 0.09},
        "<99": {"100": 0.19, "99": 0.00, "<99": 0.81},
    },
    4: {
        "100": {"100": 0.95, "99": 0.03, "<99": 0.02},
        "99": {"100": 0.10, "99": 0.70, "<99": 0.20},
        "<99": {"100": 0.09, "99": 0.03, "<99": 0.88},
    },
    6: {
        "100": {"100": 0.85, "99": 0.03, "<99": 0.12},
        "99": {"100": 0.05, "99": 0.82, "<99": 0.13},
        "<99": {"100": 0.04, "99": 0.09, "<99": 0.87},
    },
    8: {
        "100": {"100": 0.78, "99": 0.19, "<99": 0.03},
        "99": {"100": 0.07, "99": 0.72, "<99": 0.21},
        "<99": {"100": 0.01, "99": 0.04, "<99": 0.95},
    },
}
#: Reported Shorrocks φ (from unrounded matrices; recomputing from TABLE3 differs by up to ~0.01).
TABLE3_SHORROCKS: dict[int, float] = {2: 0.20, 4: 0.24, 6: 0.24, 8: 0.28}

TABLE4_SOURCE = (
    "Summary §7 (WP Table 4). WP labels the third state 'p < 100' for n=6 and 8, probably a typo for 'p < 99'."
)
#: Observed state shares in Talk, all periods.
TABLE4_OBSERVED: dict[int, dict[State3, float]] = {
    2: {"100": 0.609, "99": 0.167, "<99": 0.225},
    4: {"100": 0.685, "99": 0.109, "<99": 0.207},
    6: {"100": 0.239, "99": 0.275, "<99": 0.486},
    8: {"100": 0.225, "99": 0.203, "<99": 0.572},
}
#: Ergodic shares as printed. n=8 sums to 1.031 as transcribed.
TABLE4_ERGODIC: dict[int, dict[State3, float]] = {
    2: {"100": 0.610, "99": 0.175, "<99": 0.215},
    4: {"100": 0.689, "99": 0.121, "<99": 0.192},
    6: {"100": 0.239, "99": 0.275, "<99": 0.486},
    8: {"100": 0.265, "99": 0.183, "<99": 0.583},
}
TABLE4_ERGODIC_SUM_ANOMALIES: frozenset[int] = frozenset({8})
MARKOV_NOTE = "Adding a fourth state (p in [50, 98]) changes nothing qualitatively (authors)."

# ---------------------------------------------------------------------------
# Section 5.3: collusive strategies seen in chat (Observation 6)
# ---------------------------------------------------------------------------

_S53 = "Summary §7 'Collusive strategies' (WP §5.3)"

#: Observations (market-periods) with a common price other than 100.
COMMON_PRICE_NOT_100_OBS = 6
#: Markets reaching all-100 in their very first Talk period, out of 24.
FIRST_TALK_PERIOD_ALL100 = (14, 24)
FIRST_TALK_PERIOD_NOTE = "A price of 100 is usually proposed immediately."

#: Mean length (periods) of all-100 agreements; unit = market's average agreement length.
AGREEMENT_LENGTH: dict[int, float] = {2: 10.8, 4: 16.0, 6: 4.3, 8: 3.6}
AGREEMENT_TESTS: tuple[Test, ...] = (
    Test("Agreement length vs n (Spearman)", 0.020, None, -0.47, _S53),
    Test("Agreement length n=2 vs n=6 (MW)", 0.092, None, None, _S53),
    Test("Agreement length n=2 vs n=8 (MW)", 0.076, None, None, _S53),
    Test("Agreement length n=4 vs n=6 (MW)", 0.037, None, None, _S53),
)
FN17_NOTE = (
    "n=4 Talk markets are more often in all-100, stay longer and have longer agreements than duopolies yet "
    "earn less: no n=4 market takes turns (duopoly collusion partly sits in state 99), and n=4 prices are very "
    "low whenever p < 100 (polarised, 'unforgiving after collusion collapses')."
)


@dataclass(frozen=True)
class TakingTurnsSpell:
    n: int
    periods: int


TAKING_TURNS_DEFINITION = "One firm at 99 and the rest at 100, rotating."
#: Summary says 'seen in 5 markets' but lists only these four spells; the fifth is unspecified.
TAKING_TURNS_MARKETS_REPORTED = 5
TAKING_TURNS_SPELLS: tuple[TakingTurnsSpell, ...] = (
    TakingTurnsSpell(2, 20),
    TakingTurnsSpell(6, 12),
    TakingTurnsSpell(6, 18),
    TakingTurnsSpell(8, 14),
)
TAKING_TURNS_N4_MARKETS = 0
#: p = 99 observations accounted for by taking-turns markets, out of all p = 99 observations.
TAKING_TURNS_P99_SHARE = (64, 120)
TAKING_TURNS_DEFECTIONS = 0


@dataclass(frozen=True)
class DefectionStats:
    """fn. 18. Defection = all-100 at t−2, some firm at 99 at t−1; 'kept' = p >= 99 at t."""

    candidate_cases: int = 21
    candidate_kept: int = 10
    deliberate_turn_switches: int = 2
    real_defections: int = 19
    real_kept: int = 8
    #: Only these per-n splits are given: (kept, real defections).
    by_n: dict[int, tuple[int, int]] = field(default_factory=lambda: {4: (1, 5), 8: (5, 7)})
    #: Probability of moving from all-100 to a lower state: range given only for n=4 and n=8.
    p_leave_all100: dict[int, float] = field(default_factory=lambda: {4: 0.047, 8: 0.226})
    source: str = _S53 + ", fn. 18"


DEFECTIONS = DefectionStats()

PUNISHMENT_DEFINITION = "'Punishment' = any selling price < 99 at t after a defection; not a lasting price war."

EIGHT_FIRM_MARKET_4 = (
    "Only coordinated zero-price punishment reported. n=8 market 4 held all-100 in Talk periods 1-12; "
    "deviations in 13 (one firm at 99), 14 (99 and 98), 15 (99); each followed by chat agreement to return to "
    "100 (apology + threat in 13; threats in 14: 'this is your last chance before i start hitting 0'; in 15 "
    "renewed 100 proposals first dismissed, then accepted by the threatening firm). After a fourth deviation "
    "in 16 (99 and 88) two firms priced at 0; collusion never recovered. Authors say 'the remaining seven "
    "periods' averaged 38.4, although Part 2's 24 periods imply eight after period 16 (unresolved; check "
    "journal version). Read as threats losing credibility after not being carried out."
)
EIGHT_FIRM_MARKET_4_REMAINING_AVG = 38.4

STRATEGY_QUALITATIVE: tuple[str, ...] = (
    "Collusion with a common price works through all firms at 100.",
    "Chat mediates conflict after defections: apologies, offers to compensate the cheated firm, pleas to "
    "return to 100.",
    "Threats are rare and mostly come after a deviation.",
    "Price wars (p <= 1) mostly reflect coordination failure or a cartel in decline, not planned punishment.",
    "Observation 6: successful Talk collusion is mostly all-100, less often taking turns; defections occur "
    "but often do not lower prices because chat mediates the conflict.",
    "Threats, punishments and price wars were coded qualitatively from transcripts; no formal coding scheme.",
)

# ---------------------------------------------------------------------------
# Caveats (summary header, §2, §3, §5, §7)
# ---------------------------------------------------------------------------

CAVEATS: tuple[str, ...] = (
    "All numbers are from the working paper (DICE DP 65); the journal version has not been checked. To "
    "confirm there: Table 1 cell counts, results-screen feedback, Part 2 analysis window, fn. 14 wording, "
    "Table 4 'p < 100' labels, and the 'remaining seven periods' count for eight-firm market 4.",
    WINDOW_AMBIGUITY,
    "fn. 14 most likely uses only the last 5 Talk periods (see FN14_WINDOW).",
    "Table 2 NoTalk n=4 and n=6 columns sum to 101.0 as transcribed (rounding or a printing error).",
    "Table 4 ergodic shares for n=8 sum to 1.031 as transcribed; and ergodic shares recomputed from the "
    "two-decimal Table 3 matrices do not match Table 4 closely (badly for n=8), so do not use Table 3 to "
    "validate Table 4 or vice versa.",
    "Taking turns: 'seen in 5 markets' but only four spells (n=2: 20, n=6: 12 and 18, n=8: 14 periods) are "
    "described.",
    "Authors: interpret results qualitatively, not quantitatively; there is no 'magic' number of firms.",
    "Only 6 markets per cell; the inverted U rests on modest p-values (e.g. 0.055 for n=4 vs n=2).",
)
