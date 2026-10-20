import random

import pytest

from oligopoly.market import collusion_threshold, die_schedule, payoffs


def test_instruction_examples():
    assert payoffs({"A": 85, "B": 75}) == {"A": 0.0, "B": 22_500.0}
    assert payoffs({"A": 70, "B": 70}) == {"A": 10_500.0, "B": 10_500.0}


def test_ties_among_many():
    assert payoffs({"A": 90, "B": 60, "C": 60, "D": 60}) == {"A": 0.0, "B": 6000.0, "C": 6000.0, "D": 6000.0}
    assert payoffs({"A": 0, "B": 0}) == {"A": 0.0, "B": 0.0}


@pytest.mark.parametrize("bad", [-1, 101, 50.0, "50", True, None])
def test_invalid_prices(bad):
    with pytest.raises(ValueError):
        payoffs({"A": 50, "B": bad})


def test_custom_p_max_and_m():
    assert payoffs({"A": 150, "B": 200}, m=10, p_max=200) == {"A": 1500.0, "B": 0.0}
    with pytest.raises(ValueError):
        payoffs({"A": 101})


def test_thresholds_at_five_sixths():
    holds = {n: collusion_threshold(n).holds(5 / 6) for n in (2, 4, 6, 8, 10)}
    assert holds == {2: True, 4: True, 6: True, 8: False, 10: False}
    t6 = collusion_threshold(6)
    assert t6.exact == pytest.approx(1 - 100 / 594)
    assert t6.continuous == pytest.approx(5 / 6)
    assert collusion_threshold(2).exact == pytest.approx(0.4949, abs=1e-4)
    assert collusion_threshold(8).exact == pytest.approx(0.8737, abs=1e-4)


def test_die_schedule():
    s = die_schedule(29, rng=1)
    assert len(s) == 29
    assert s[:19] == [None] * 19
    assert all(1 <= f <= 5 for f in s[19:28])
    assert s[28] == 6
    assert die_schedule(24, rng=random.Random(3)) == die_schedule(24, rng=3)
    assert die_schedule(20) == [None] * 19 + [6]
    with pytest.raises(ValueError):
        die_schedule(19)
