import pytest

from oligopoly import benchmarks as b


def test_cells_cover_all_n():
    for table in (b.MAIN_NOTALK_SELLING, b.MAIN_TALK_SELLING, b.MAIN_GAIN_FROM_TALK, b.MAIN_NOTALK_POSTED,
                  b.FN14_TALK_PROFIT, b.FN14_GAIN, b.TABLE3, b.TABLE3_SHORROCKS, b.TABLE4_OBSERVED,
                  b.TABLE4_ERGODIC, b.AGREEMENT_LENGTH):
        assert set(table) == set(b.NS)


def test_gain_matches_talk_minus_notalk():
    for n in b.NS:
        diff = b.MAIN_TALK_SELLING[n].mean - b.MAIN_NOTALK_SELLING[n].mean
        assert b.MAIN_GAIN_FROM_TALK[n] == pytest.approx(diff, abs=0.15)
        assert b.FN14_GAIN[n] == pytest.approx(b.FN14_TALK_PROFIT[n] - b.MAIN_NOTALK_SELLING[n].mean, abs=0.1)


def test_talk_above_notalk_and_gain_inverted_u():
    for n in b.NS:
        assert b.MAIN_TALK_SELLING[n].mean > b.MAIN_NOTALK_SELLING[n].mean
    g = b.MAIN_GAIN_FROM_TALK
    assert max(g, key=g.get) == 4 and g[4] > g[2] and g[4] > g[8]
    f = b.FN14_GAIN
    assert max(f, key=f.get) == 4


def test_posted_at_least_selling():
    for n in b.NS:
        assert b.MAIN_NOTALK_POSTED[n] >= b.MAIN_NOTALK_SELLING[n].mean


def test_prices_in_range():
    stats = [*b.MAIN_NOTALK_SELLING.values(), *b.MAIN_TALK_SELLING.values(),
             *(s for _, s in b.CONTROLS_SELLING.values())]
    for s in stats:
        assert 0 <= s.mean <= 100 and (s.sd is None or s.sd >= 0) and s.source


@pytest.mark.parametrize("key", sorted(b.TABLE2))
def test_table2_columns_sum_to_100(key):
    total = sum(b.TABLE2[key][k] for k in b.BINS4)
    tol = 1.05 if key in b.TABLE2_SUM_ANOMALIES else 0.25
    assert total == pytest.approx(100, abs=tol)


def test_table2_anomalies_are_real():
    for key in b.TABLE2_SUM_ANOMALIES:
        assert abs(sum(b.TABLE2[key].values()) - 100) > 0.25


@pytest.mark.parametrize("n", b.NS)
def test_table3_rows_are_distributions(n):
    m = b.TABLE3[n]
    assert set(m) == set(b.STATES3)
    for row in m.values():
        assert set(row) == set(b.STATES3)
        assert all(0 <= v <= 1 for v in row.values())
        assert sum(row.values()) == pytest.approx(1, abs=0.005)


@pytest.mark.parametrize("n", b.NS)
def test_shorrocks_recomputed_from_table3(n):
    assert b.shorrocks(b.TABLE3[n]) == pytest.approx(b.TABLE3_SHORROCKS[n], abs=0.015)


def test_leave_all100_matches_table3():
    for n, p in b.DEFECTIONS.p_leave_all100.items():
        assert 1 - b.TABLE3[n]["100"]["100"] == pytest.approx(p, abs=0.01)


@pytest.mark.parametrize("n", b.NS)
def test_table4_shares_sum_to_one(n):
    assert sum(b.TABLE4_OBSERVED[n].values()) == pytest.approx(1, abs=0.002)
    tol = 0.035 if n in b.TABLE4_ERGODIC_SUM_ANOMALIES else 0.003
    assert sum(b.TABLE4_ERGODIC[n].values()) == pytest.approx(1, abs=tol)


def test_taking_turns_spells_account_for_64_p99_obs():
    assert sum(s.periods for s in b.TAKING_TURNS_SPELLS) == b.TAKING_TURNS_P99_SHARE[0]
    assert all(s.n != 4 for s in b.TAKING_TURNS_SPELLS)
    assert len(b.TAKING_TURNS_SPELLS) < b.TAKING_TURNS_MARKETS_REPORTED  # documented gap


def test_defection_counts_consistent():
    d = b.DEFECTIONS
    assert d.candidate_cases - d.deliberate_turn_switches == d.real_defections
    assert d.candidate_kept - d.deliberate_turn_switches == d.real_kept
    for kept, total in d.by_n.values():
        assert 0 <= kept <= total
    assert sum(t for _, t in d.by_n.values()) <= d.real_defections


def test_measure_helpers():
    assert b.selling_price({"A": 85, "B": 75}) == 75
    assert [b.price_bin4(p) for p in (100, 99, 98, 2, 1, 0)] == ["100", "99", "2-98", "2-98", "<=1", "<=1"]
    assert [b.markov_state(p) for p in (100, 99, 98)] == ["100", "99", "<99"]


def test_design_constants():
    assert b.PART1_PERIODS + b.PART2_PERIODS == 53
    assert b.FIRST_TALK_PERIOD_ALL100[0] <= b.FIRST_TALK_PERIOD_ALL100[1] == b.MARKETS_PER_CELL * len(b.NS)
