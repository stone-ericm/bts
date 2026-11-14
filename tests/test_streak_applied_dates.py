"""update_streak applies a contest date at most once (incident L04, 2026-11-13).

The scorers write streak.json and then the pick's terminal result. A death between the two used to make the
restarted scorer apply the same result again. The date now rides in the same atomic write as the streak.
"""
from __future__ import annotations

import json

from bts.picks import (APPLIED_DATES_KEPT, load_applied_dates, load_saver_available, load_streak, save_streak,
                       update_streak)


def test_the_same_date_is_applied_once(tmp_path):
    save_streak(5, tmp_path, saver_available=True)
    assert update_streak([True], tmp_path, date="2027-04-02") == 6
    assert update_streak([True], tmp_path, date="2027-04-02") == 6
    assert load_streak(tmp_path) == 6 and load_applied_dates(tmp_path) == ["2027-04-02"]


def test_a_repeated_saver_miss_does_not_reset_the_streak(tmp_path):
    save_streak(12, tmp_path, saver_available=True)
    assert update_streak([False], tmp_path, date="2027-04-02") == 12       # the saver absorbs it
    assert update_streak([False], tmp_path, date="2027-04-02") == 12       # applied again it would reset to 0
    assert (load_streak(tmp_path), load_saver_available(tmp_path)) == (12, False)


def test_different_dates_each_apply(tmp_path):
    save_streak(5, tmp_path, saver_available=True)
    update_streak([True], tmp_path, date="2027-04-02")
    assert update_streak([True, True], tmp_path, date="2027-04-03") == 8
    assert load_applied_dates(tmp_path) == ["2027-04-02", "2027-04-03"]


def test_without_a_date_the_old_behaviour_is_kept(tmp_path):
    save_streak(5, tmp_path, saver_available=True)
    update_streak([True], tmp_path)
    assert update_streak([True], tmp_path) == 7 and load_applied_dates(tmp_path) == []


def test_other_writers_of_the_streak_keep_the_marker(tmp_path):
    save_streak(5, tmp_path, saver_available=True)
    update_streak([True], tmp_path, date="2027-04-02")
    save_streak(6, tmp_path, saver_available=False)             # e.g. reconcile's replay, the saver CLI
    assert load_applied_dates(tmp_path) == ["2027-04-02"]
    assert update_streak([True], tmp_path, date="2027-04-02") == 6


def test_a_legacy_streak_file_has_no_marker_and_still_works(tmp_path):
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 3, "saver_available": True}))
    assert load_applied_dates(tmp_path) == []
    assert update_streak([True], tmp_path, date="2027-04-02") == 4


def test_the_marker_list_is_bounded(tmp_path):
    save_streak(0, tmp_path, saver_available=True)
    for day in range(1, APPLIED_DATES_KEPT + 6):
        update_streak([True], tmp_path, date=f"2027-{4 + day // 31:02d}-{day % 31 + 1:02d}")
    assert len(load_applied_dates(tmp_path)) == APPLIED_DATES_KEPT
