"""Resume-date suspended games must be excluded from the candidate slate.

Per BTS rules the resumed portion of a suspended game is never evaluated, so a pick on
the resume day can never score; `is_resume_date_game` filters them out. It lives in
`bts.picks` (lightgbm-free) so this regression runs even without the model extra.
Regression for the 2026-06-17 live_forward_resolution stall (game 824912 resumed 06-16).
See docs/audit/2026-06-29-skip-threshold-and-discrimination.md.
"""
import pandas as pd

from bts.picks import is_resume_date_game


def test_excludes_resume_day_suspended_game():
    # game 824912: suspended 06-16, resumed 06-17 -> officialDate stays 06-16
    assert is_resume_date_game({"officialDate": "2026-06-16"}, "2026-06-17") is True


def test_keeps_normal_same_day_game():
    assert is_resume_date_game({"officialDate": "2026-06-17"}, "2026-06-17") is False


def test_keeps_game_with_missing_official_date():
    # defensive: absent officialDate must not drop a real game
    assert is_resume_date_game({}, "2026-06-17") is False


def test_normalizes_timestamp_and_datetime_string_forms():
    # date passed as Timestamp or "...00:00:00" must still compare against YYYY-MM-DD
    assert is_resume_date_game({"officialDate": "2026-06-17"}, pd.Timestamp("2026-06-17")) is False
    assert is_resume_date_game({"officialDate": "2026-06-16"}, "2026-06-17 00:00:00") is True


def test_keeps_game_with_later_official_date():
    # only games resumed from an EARLIER date are excluded; a later/odd official date
    # must not be silently dropped (uses `<`, not `!=`)
    assert is_resume_date_game({"officialDate": "2026-06-18"}, "2026-06-17") is False
