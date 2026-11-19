from datetime import date, timedelta
from functools import lru_cache
from itertools import product
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from bts import picks
from bts.simulate import exact, mdp, monte_carlo, tail_policy
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.strategies import Strategy
from scratch import inject


HIT, MISS, DOUBLE, MIXED, VOID, SKIP = (
    ("hit",), ("miss",), ("hit", "hit"), ("hit", "miss"), ("void",), (),
)
PLAY = (HIT, MISS, DOUBLE, MIXED, SKIP)
ALL = PLAY + (("miss", "hit"), VOID, ("void", "void"),
              ("hit", "void"), ("void", "hit"), ("miss", "void"), ("void", "miss"))


def step(state, slots):
    streak, best, saver = state
    if "miss" in slots:
        if saver and streak in range(10, 16):
            saver = False
        else:
            streak = 0
    else:
        streak += slots.count("hit")
    return streak, max(best, streak), saver


def fold(days, state=(0, 0, True)):
    for slots in days:
        state = step(state, slots)
    return state


def seasons(outcomes):
    # Bias toward the saver zone and both edges. Repeated misses exercise consumption.
    return st.builds(
        lambda count, misses, suffix: (DOUBLE,) * (count // 2) + (HIT,) * (count % 2)
        + misses + tuple(suffix),
        st.integers(8, 17),
        st.sampled_from(((), (MISS,), (MIXED, MISS), (MISS, MIXED))),
        st.lists(st.sampled_from(outcomes), max_size=12),
    )


def apply_live(directory, day, slots):
    results = dict(zip(("pick", "double_down"), slots))
    return picks.update_streak(picks.active_streak_results(results), directory, date=day)


def save_day(directory, day, slots):
    pick = picks.Pick("Synthetic", 1, "TST", 1, "Synthetic", 2, 0.8, [], False, 1, day)
    results = dict(zip(("pick", "double_down"), slots))
    daily = picks.DailyPick(day, day, pick, pick if len(slots) == 2 else None, None,
                            result=picks.effective_daily_result(results), slot_results=results)
    picks.save_pick(daily, directory)


@settings(max_examples=80, deadline=None, database=None)
@given(days=seasons(ALL))
def test_update(days):
    with TemporaryDirectory() as temp, pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, picks, "update_streak", "current + len(results)", "current + 1")
        directory = Path(temp)
        best = 0
        for i, slots in enumerate(days):
            day = (date(2027, 4, 1) + timedelta(days=i)).isoformat()
            streak = apply_live(directory, day, slots)
            best = max(best, streak)
            assert (streak, best, picks.load_saver_available(directory)) == fold(days[:i + 1])
            before = (directory / "streak.json").read_bytes()
            assert apply_live(directory, day, MISS) == streak
            assert (directory / "streak.json").read_bytes() == before
        assert apply_live(directory, "2028-04-02", HIT) == 1
        assert picks.load_saver_available(directory)
        assert picks.load_applied_dates(directory) == ["2028-04-02"]
        assert apply_live(directory, "2028-04-01", MISS) == 0
        assert picks.load_applied_dates(directory) == ["2028-04-02", "2028-04-01"]


@settings(max_examples=80, deadline=None, database=None)
@given(days=seasons(ALL))
def test_replay(days):
    with TemporaryDirectory() as temp, pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, picks, "_replay_season_streak",
                   'if r == "void":\n            continue',
                   'if r == "void":\n            streak = 0\n            continue')
        directory = Path(temp)
        dates = [(date(2027, 4, 1) + timedelta(days=i)).isoformat() for i in range(len(days))]
        for day, slots in reversed(list(zip(dates, days))):
            if slots:  # A skipped date has no pick file.
                save_day(directory, day, slots)
        best = 0
        for i, day in enumerate(dates):
            streak, saver = picks._replay_season_streak(directory, 2027, day)
            best = max(best, streak)
            assert (streak, best, saver) == fold(days[:i + 1])


def action(slots):
    return {0: "skip", 1: "single", 2: "double"}[len(slots)]


@settings(max_examples=80, deadline=None, database=None)
@given(days=seasons(PLAY), saver=st.booleans())
def test_mdp(days, saver):
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, mdp, "transition_outcomes", "10 <= streak <= 15", "11 <= streak <= 15")
        state = (0, 0, saver)
        for i, slots in enumerate(days):
            streak, best, available = state
            hit = float("miss" not in slots)
            branch, = [b for b in mdp.transition_outcomes(
                action(slots), streak, available, p_hit=hit, p_both=hit
            ) if b.probability == 1.0]
            state = (branch.next_streak, max(best, branch.next_streak), branch.saver_available)
            assert state == fold(days[:i + 1], (0, 0, saver))


def profile(slots):
    confidence = {0: (0.1, 1.0), 1: (0.7, 0.1), 2: (0.9, 0.9)}[len(slots)]
    return monte_carlo.DailyProfile(confidence[0], bool(slots and slots[0] == "hit"),
                                   confidence[1], len(slots) == 2 and slots[1] == "hit")


@settings(max_examples=80, deadline=None, database=None)
@given(days=seasons(PLAY), saver=st.booleans())
def test_monte_carlo(days, saver):
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, monte_carlo, "simulate_season", "streak += 2", "streak += 1")
        strategy = Strategy("generated", skip_threshold=0.5, double_threshold=0.5, streak_saver=saver)
        profiles = [profile(slots) for slots in days]
        result = monte_carlo.simulate_season(profiles, strategy)
        _, best, available = fold(days, (0, 0, saver))
        assert (result.max_streak, result.play_days, result.streak_saver_used) == (
            best, sum(bool(slots) for slots in days), saver and not available,
        )


def deterministic_matrix(slots, saver):
    hit = float("miss" not in slots)
    bins = QualityBins([QualityBin(0, (0.75, 0.75), hit, hit, 1.0)], [])
    strategy = Strategy("generated", skip_threshold=1.0 if not slots else None,
                        double_threshold=0.0 if len(slots) == 2 else None, streak_saver=saver)
    return exact.build_transition_matrix(strategy, bins)


@settings(max_examples=80, deadline=None, database=None)
@given(days=seasons(PLAY), state=st.integers(8, 17), saver=st.booleans())
def test_exact_matrix(days, state, saver):
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, exact, "build_transition_matrix", "next_s = min(s + 2, 57)",
                   "next_s = min(s + 1, 57)")
        # The matrix lacks a saver-used coordinate. Compare one transition with
        # the saver enabled, and full sequences with the saver disabled.
        assert np.argmax(deterministic_matrix(days[-1], saver)[state]) == (
            fold(days[-1:], (state, state, saver))[0]
        )
        distribution = np.eye(58)[state]
        for slots in days:
            distribution = distribution @ deterministic_matrix(slots, False)
        assert np.argmax(distribution) == fold(days, (state, state, False))[0]


@settings(max_examples=50, deadline=None, database=None)
@given(count=st.integers(8, 17), doubles=st.integers(1, 4), skips=st.integers(0, 2))
def test_exact_policy(count, doubles, skips):
    # A single always hits, while a double independently hits both with p=1/2.
    # Force the stochastic stretch between a certain opening and closing run.
    bins = QualityBins([QualityBin(0, (0.75, 0.75), 1.0, 0.5, 1.0)], [])
    actions = [1] * count + [2] * doubles + [0] * skips + [1] * (57 - count)
    policy = np.zeros((58, len(actions) + 1, 2, 1), dtype=np.int8)
    policy[:, 1:, :, 0] = actions[::-1]
    expected = sum(
        fold((HIT,) * count + middle + (SKIP,) * skips + (HIT,) * (57 - count))[1] >= 57
        for middle in product((DOUBLE, MIXED), repeat=doubles)
    ) / 2**doubles
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, exact, "exact_p57_policy_table", "(1 - pb) * ev(s, 0)", "(1 - pb) * ev(s, saver)")
        assert exact.exact_p57_policy_table(policy, bins) == expected


@settings(max_examples=80, deadline=None, database=None)
@given(streak=st.integers(8, 17), lead=st.integers(0, 3), days=st.integers(2, 6),
       saver=st.booleans(),
       rates=st.sampled_from(((0.5, 0.25), (0.75, 0.5), (1.0, 0.5), (0.5, 0.5))))
def test_tail(streak, lead, days, saver, rates):
    ph, pb = rates
    best = streak + lead
    target = 32  # All generated paths fit below the production cap.

    @lru_cache(None)
    def expected(state, left):
        if left == 0:
            return float(state[1])
        return max(expected(step(state, SKIP), left - 1),
                   ph * expected(step(state, HIT), left - 1)
                   + (1 - ph) * expected(step(state, MISS), left - 1),
                   pb * expected(step(state, DOUBLE), left - 1)
                   + (1 - pb) * expected(step(state, MIXED), left - 1))

    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, tail_policy, "solve_emax_season_best", "EV[Sg, Mg, 0]", "EV[Sg, Mg, sv]")
        solved = tail_policy.solve_emax_season_best([1.0], [ph], [pb], target=target, max_days=days)
        assert solved.value[streak, best, days, int(saver)] == expected((streak, best, saver), days)


@settings(max_examples=50, deadline=None, database=None)
@given(length=st.integers(31, 45))
def test_date_marker_survives_long_retry(length):
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        first = date(2027, 4, 1)
        for offset in range(length):
            apply_live(directory, (first + timedelta(days=offset)).isoformat(), HIT)
        assert apply_live(directory, first.isoformat(), HIT) == length
