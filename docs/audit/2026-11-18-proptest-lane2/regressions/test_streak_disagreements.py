from datetime import date, timedelta

import numpy as np

from bts.picks import update_streak
from bts.simulate.exact import build_transition_matrix
from bts.simulate.mdp import transition_outcomes
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.strategies import Strategy


def test_evicted_date_is_still_idempotent(tmp_path):
    first = date(2027, 4, 1)
    for offset in range(31):
        update_streak([True], tmp_path, date=(first + timedelta(days=offset)).isoformat())
    assert update_streak([True], tmp_path, date=first.isoformat()) == 31


def test_dated_update_is_chronological(tmp_path):
    update_streak([True], tmp_path, date="2027-04-02")
    assert update_streak([False], tmp_path, date="2027-04-01") == 1


def test_fixed_exact_and_mdp_agree_after_two_misses_at_ten():
    bins = QualityBins([QualityBin(0, (0.8, 0.8), 0.0, 0.0, 1.0)], [])
    matrix = build_transition_matrix(Strategy("single"), bins)
    exact_streak = int(np.argmax((matrix @ matrix)[10]))
    streak, saver = 10, True
    for _ in range(2):
        miss = transition_outcomes("single", streak, saver, p_hit=0.0, p_both=0.0)[1]
        streak, saver = miss.next_streak, miss.saver_available
    assert exact_streak == streak
