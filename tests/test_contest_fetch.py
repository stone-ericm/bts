from __future__ import annotations

import datetime as dt

import pytest

from bts.contest_fetch import (
    ContestFetchError,
    build_observation,
    derive_source_date,
    fetch_profile,
    validate_fetch,
)


def test_fetch_profile_returns_success_payload():
    success = {
        "activeStreak": 0,
        "seasonBestStreak": 9,
        "predictions": [],
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": success}

    class Client:
        calls = []

        @classmethod
        def get(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return Response()

    assert fetch_profile(50311, {"oktaid": "uid"}, "xsid_1", client=Client) == success

    args, kwargs = Client.calls[0]
    assert "/50311/profile?xSid=xsid_1" in args[0]
    assert kwargs["cookies"] == {"oktaid": "uid"}
    assert kwargs["headers"]["Accept"] == "application/json"


def test_derive_source_date_latest_settled():
    rounds = {
        10: dt.date(2026, 6, 4),
        11: dt.date(2026, 6, 5),
        12: dt.date(2026, 6, 6),
    }
    predictions = [
        {"roundId": 10, "result": "hit"},
        {"roundId": 11, "result": "miss"},
        {"roundId": 12, "result": None},
    ]

    assert derive_source_date(predictions, rounds) == dt.date(2026, 6, 5)


def test_derive_source_date_none_when_no_settled():
    assert (
        derive_source_date(
            [{"roundId": 1, "result": None}],
            {1: dt.date(2026, 6, 6)},
        )
        is None
    )


@pytest.mark.parametrize(
    "success",
    [
        {"activeStreak": -1, "seasonBestStreak": 9},
        {"activeStreak": 1.5, "seasonBestStreak": 9},
        {"activeStreak": "1", "seasonBestStreak": 9},
        {"activeStreak": 1, "seasonBestStreak": None},
    ],
)
def test_validate_fetch_rejects_non_int_or_negative_streaks(success):
    with pytest.raises(ContestFetchError):
        validate_fetch(success)


def test_validate_fetch_rejects_best_less_than_active():
    with pytest.raises(ContestFetchError):
        validate_fetch({"activeStreak": 5, "seasonBestStreak": 3})


def test_validate_fetch_accepts_zero_active_streak():
    validate_fetch({"activeStreak": 0, "seasonBestStreak": 9})


def test_build_observation_returns_auto_schema():
    recorded_at = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    observation = build_observation(
        {"activeStreak": 0, "seasonBestStreak": 9},
        source_date=dt.date(2026, 6, 6),
        user_id=50311,
        username="stonehengee",
        recorded_at=recorded_at,
    )

    assert observation == {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 0,
        "best_streak": 9,
        "source": "mlb_bts_profile",
        "source_date": "2026-06-06",
        "recorded_at": "2026-06-06T18:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
        "saver_available": None,
    }


def test_build_observation_allows_none_source_date():
    """Snapshot (activeStreak) must persist even when ledger coverage is unknown —
    the predictions array lags the counter, so source_date can be None."""
    obs = build_observation(
        {"activeStreak": 8, "seasonBestStreak": 9},
        source_date=None,
        user_id=50311,
        username="stonehengee",
        recorded_at=dt.datetime(2026, 6, 17, 12, 0, tzinfo=dt.UTC),
    )
    assert obs["active_streak"] == 8
    assert obs["source_date"] is None
    assert obs["recorded_at"].endswith("Z")


class TestHasPredictionFor:
    def test_pending_prediction_counts_as_entered(self):
        from datetime import date
        from bts.contest_fetch import has_prediction_for
        success = {"predictions": [{"roundId": 7, "result": "pending"}]}
        assert has_prediction_for(success, {7: date(2026, 6, 12)}, date(2026, 6, 12)) is True

    def test_settled_prediction_counts(self):
        from datetime import date
        from bts.contest_fetch import has_prediction_for
        success = {"predictions": [{"roundId": 7, "result": "hit"}]}
        assert has_prediction_for(success, {7: date(2026, 6, 12)}, date(2026, 6, 12)) is True

    def test_absent_date_is_not_entered(self):
        from datetime import date
        from bts.contest_fetch import has_prediction_for
        success = {"predictions": [{"roundId": 6, "result": "hit"}]}
        rounds = {6: date(2026, 6, 11), 7: date(2026, 6, 12)}
        assert has_prediction_for(success, rounds, date(2026, 6, 12)) is False

    def test_unknown_round_id_ignored(self):
        from datetime import date
        from bts.contest_fetch import has_prediction_for
        success = {"predictions": [{"roundId": 999, "result": "hit"}, {"result": "hit"}]}
        assert has_prediction_for(success, {7: date(2026, 6, 12)}, date(2026, 6, 12)) is False


def test_derive_source_date_counts_not_hit_rounds():
    """MLB profiles use 'not_hit' for a settled miss; derive_source_date must treat it
    as settled, else freshness is biased against reset days (the RESOLVED vocab bug)."""
    rounds = {1: dt.date(2026, 6, 8), 2: dt.date(2026, 6, 9)}
    preds = [{"roundId": 1, "result": "hit"}, {"roundId": 2, "result": "not_hit"}]
    assert derive_source_date(preds, rounds) == dt.date(2026, 6, 9)
