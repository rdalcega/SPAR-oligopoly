"""Stage game and horizon of the Fonseca & Normann (2012) Bertrand market (summary §1 and §3)."""

from __future__ import annotations

import random
from dataclasses import dataclass

M = 300
P_MAX = 100


def payoffs(prices: dict[str, int], m: int = M, p_max: int = P_MAX) -> dict[str, float]:
    """Bertrand payoffs: the lowest price earns m·p, split equally on ties; everyone else earns 0.

    Raises ValueError unless every price is an integer in 0..p_max (bools are rejected).
    """
    if not prices:
        raise ValueError("no prices")
    for firm, p in prices.items():
        if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= p_max:
            raise ValueError(f"price for {firm!r} must be an integer in 0..{p_max}, got {p!r}")
    low = min(prices.values())
    winners = [f for f, p in prices.items() if p == low]
    share = m * low / len(winners)
    return {f: (share if p == low else 0.0) for f, p in prices.items()}


@dataclass(frozen=True)
class Threshold:
    """Minimum δ for all firms at p_max to be a grim-trigger equilibrium."""

    n: int
    exact: float  # integer prices: the best deviation is p_max - step
    continuous: float  # the paper's (n - 1)/n, deviation profit bounded by p_max·m

    def holds(self, delta: float) -> bool:
        """Whether full collusion is an equilibrium at δ under the exact (integer-price) condition."""
        return delta >= self.exact


def collusion_threshold(n: int, p_max: int = P_MAX, step: int = 1) -> Threshold:
    """Grim-trigger thresholds: exact 1 − p_max/((p_max − step)·n) and continuous (n − 1)/n."""
    if n < 2:
        raise ValueError("n must be at least 2")
    return Threshold(n=n, exact=1 - p_max / ((p_max - step) * n), continuous=(n - 1) / n)


def die_schedule(length: int, min_periods: int = 20, rng: random.Random | int | None = None) -> list[int | None]:
    """Die outcome shown after each period of a part with pre-drawn `length`.

    Index i is period i + 1. Periods before `min_periods` have no throw (None). Periods `min_periods` to
    `length − 1` show a face from 1–5 (play continues); the final period shows a 6. `rng` is a
    `random.Random` or a seed.
    """
    if length < min_periods:
        raise ValueError(f"length {length} is below the minimum of {min_periods} periods")
    r = rng if isinstance(rng, random.Random) else random.Random(rng)
    return [None if t < min_periods else 6 if t == length else r.randint(1, 5) for t in range(1, length + 1)]
