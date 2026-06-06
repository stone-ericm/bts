from __future__ import annotations

import datetime as dt

from bts.contest_fetch import derive_source_date, fetch_profile


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
