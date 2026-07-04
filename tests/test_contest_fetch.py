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
    assert "application/json" in kwargs["headers"]["Accept"]
    assert "Chrome/" in kwargs["headers"]["User-Agent"]


def test_fetch_pending_predictions_returns_current_round_rows():
    # GET api/predictions — the endpoint that DOES expose the pending
    # same-day entry (the profile endpoint is settled-only; 2026-06-12 lesson).
    pending = [{"roundId": 923, "unitId": 1323, "playerId": 377, "number": 1,
                "streak": 17, "result": None}]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": {"predictions": pending, "resentHistory": []}}

    class Client:
        calls = []

        @classmethod
        def get(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return Response()

    from bts.contest_fetch import fetch_pending_predictions
    assert fetch_pending_predictions({"oktaid": "uid"}, "xsid_1", client=Client) == pending
    args, kwargs = Client.calls[0]
    assert "/api/predictions?xSid=xsid_1" in args[0]
    assert kwargs["cookies"] == {"oktaid": "uid"}


def test_fetch_pending_predictions_empty_list_when_no_pending():
    # An authenticated response with an explicit empty predictions list is the
    # legitimate "no pick entered yet" state — returns [], does not raise.
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": {"predictions": []}}

    class Client:
        @classmethod
        def get(cls, *args, **kwargs):
            return Response()

    from bts.contest_fetch import fetch_pending_predictions
    assert fetch_pending_predictions({}, "x", client=Client) == []


def test_fetch_pending_predictions_raises_on_schema_drift():
    # A 200 whose shape drifted (no success.predictions) must RAISE, not collapse
    # to [] — else check-pick-entered reads it as "not entered" and false-alarms
    # (the v1 class). Raising routes it to the caller's quiet-skip path.
    from bts.contest_fetch import ContestFetchError, fetch_pending_predictions

    for bad in ({"success": {}}, {"errors": ["nope"]}, {"success": {"predictions": "x"}}):
        class Response:
            _b = bad

            def raise_for_status(self):
                pass

            def json(self):
                return self._b

        class Client:
            @classmethod
            def get(cls, *args, **kwargs):
                return Response()

        with pytest.raises(ContestFetchError):
            fetch_pending_predictions({}, "x", client=Client)


class TestPickEntryStatus:
    ROUNDS = {7: dt.date(2026, 6, 12)}
    TARGET = dt.date(2026, 6, 12)
    XWALK = {100: 1, 200: 2}  # BTS id -> MLB feed id

    def _status(self, profile=None, pending=None, required=None, xwalk=None):
        from bts.contest_fetch import pick_entry_status
        return pick_entry_status(
            {"predictions": profile or []}, pending or [], self.ROUNDS, self.TARGET,
            required if required is not None else {1},
            self.XWALK if xwalk is None else xwalk)

    def test_no_pick_when_nothing_entered(self):
        assert self._status() == (False, "no_pick")

    def test_match_on_pending_row(self):
        assert self._status(pending=[{"roundId": 7, "playerId": 100}]) == (True, "match")

    def test_match_on_nested_profile_row(self):
        prof = [{"roundId": 7, "roundPredictions": [{"playerId": 100}]}]
        assert self._status(profile=prof) == (True, "match")

    def test_mismatch_wrong_player(self):
        assert self._status(pending=[{"roundId": 7, "playerId": 200}]) == (False, "mismatch")

    def test_double_down_missing_slot_is_mismatch(self):
        # required both MLB 1 and 2; only 100->1 entered
        assert self._status(pending=[{"roundId": 7, "playerId": 100}],
                            required={1, 2}) == (False, "mismatch")

    def test_double_down_both_slots_match(self):
        rows = [{"roundId": 7, "playerId": 100}, {"roundId": 7, "playerId": 200}]
        assert self._status(pending=rows, required={1, 2}) == (True, "match")

    def test_unresolved_crosswalk_is_present_unverified(self):
        # entered BTS 999 not in the crosswalk -> can't prove a mismatch
        assert self._status(pending=[{"roundId": 7, "playerId": 999}]) == (True, "present_unverified")

    def test_empty_crosswalk_is_present_unverified(self):
        assert self._status(pending=[{"roundId": 7, "playerId": 100}], xwalk={}) == (True, "present_unverified")

    def test_other_date_rows_ignored(self):
        # a pick for a different round/date is not "entered for target"
        rounds = {7: dt.date(2026, 6, 12), 8: dt.date(2026, 6, 13)}
        from bts.contest_fetch import pick_entry_status
        ok, reason = pick_entry_status(
            {"predictions": []}, [{"roundId": 8, "playerId": 100}], rounds,
            dt.date(2026, 6, 12), {1}, self.XWALK)
        assert (ok, reason) == (False, "no_pick")

    def test_no_required_ids_is_present_unverified(self):
        assert self._status(pending=[{"roundId": 7, "playerId": 100}], required=set()) == (True, "present_unverified")


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


def _mock_fetch(monkeypatch, profile, rounds):
    """Patch the auth + fetch + rounds calls of the fetch-contest-streak CLI.
    The command does function-local imports, so patch the source modules."""
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    import bts.cli as climod

    class _Sess:
        xsid = "x"; user_id = 50311; username = "stonehengee"
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "u"})
    monkeypatch.setattr(auth, "extract_uid", lambda c: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda uid, cookies: _Sess())
    monkeypatch.setattr(cf, "fetch_profile", lambda uid, cookies, xsid: profile)
    monkeypatch.setattr(climod, "_fetch_rounds", lambda: rounds)


def test_fetch_cli_persists_current_activestreak_despite_lag(tmp_path, monkeypatch):
    """Incident: activeStreak=8 is current, but the predictions array lags (latest settled
    row = 6/15) while a local pick is resolved through 6/16. The CLI must still WRITE 8."""
    import json
    from click.testing import CliRunner
    from bts.cli import cli
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))   # local resolved 6/16
    _mock_fetch(monkeypatch,
                {"activeStreak": 8, "seasonBestStreak": 9,
                 "predictions": [{"roundId": 1, "result": "hit"}]},          # only 6/15 settled
                {1: dt.date(2026, 6, 15)})
    res = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                   "--expected-username", "stonehengee"])
    assert res.exit_code == 0, res.output
    written = json.loads((picks / "account_state" / "contest_streak.json").read_text())
    assert written["active_streak"] == 8


def test_fetch_cli_persists_snapshot_when_no_settled_predictions(tmp_path, monkeypatch):
    """No settled predictions -> derive_source_date None. The snapshot must still persist
    (activeStreak set, source_date null) — the snapshot/coverage split."""
    import json
    from click.testing import CliRunner
    from bts.cli import cli
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    _mock_fetch(monkeypatch,
                {"activeStreak": 8, "seasonBestStreak": 9,
                 "predictions": [{"roundId": 1, "result": None}]},          # nothing settled
                {1: dt.date(2026, 6, 16)})
    res = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                   "--expected-username", "stonehengee"])
    assert res.exit_code == 0, res.output
    written = json.loads((picks / "account_state" / "contest_streak.json").read_text())
    assert written["active_streak"] == 8 and written["source_date"] is None


def test_fetch_cli_persists_per_round_ledger(tmp_path, monkeypatch):
    """The full per-round MLB ledger is persisted (append-only) for analysis."""
    import json
    from click.testing import CliRunner
    from bts.cli import cli
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))
    preds = [{"roundId": 1, "result": "hit", "streak": 8, "streakIncrease": 1,
              "roundPredictions": [{"playerId": 1, "result": "hit", "hits": 2, "atBats": 4}]}]
    _mock_fetch(monkeypatch,
                {"activeStreak": 8, "seasonBestStreak": 9, "predictions": preds},
                {1: dt.date(2026, 6, 16)})
    res = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                   "--expected-username", "stonehengee"])
    assert res.exit_code == 0, res.output
    ledger = picks / "account_state" / "contest_ledger.jsonl"
    assert ledger.exists()
    row = json.loads(ledger.read_text().strip().splitlines()[-1])
    assert row["active_streak"] == 8 and len(row["predictions"]) == 1
    assert row["recorded_at"].endswith("Z")


def _run_fetch(picks):
    from click.testing import CliRunner
    from bts.cli import cli
    return CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                    "--expected-username", "stonehengee"])


def test_fetch_auto_earn_cold_file_at_10_stays_uninitialized(tmp_path, monkeypatch):
    """Fail-closed: a cold saver_state file + best_streak>=10 must NOT auto-become active
    (the account could have earned-and-used the saver before we ever observed it)."""
    import json
    from bts.saver_state import load_saver_state
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))
    _mock_fetch(monkeypatch, {"activeStreak": 10, "seasonBestStreak": 10,
                              "predictions": [{"roundId": 1, "result": "hit", "streak": 10}]},
                {1: dt.date(2026, 6, 16)})
    assert _run_fetch(picks).exit_code == 0
    assert load_saver_state(picks, season=2026).state == "uninitialized"


def test_fetch_auto_earn_below_10_inits_not_earned_then_promotes_at_10(tmp_path, monkeypatch):
    import json
    from bts.saver_state import load_saver_state
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))
    _mock_fetch(monkeypatch, {"activeStreak": 8, "seasonBestStreak": 8,
                              "predictions": [{"roundId": 1, "result": "hit", "streak": 8}]},
                {1: dt.date(2026, 6, 16)})
    assert _run_fetch(picks).exit_code == 0
    assert load_saver_state(picks, season=2026).state == "not_earned"   # safe cold-init below 10
    _mock_fetch(monkeypatch, {"activeStreak": 10, "seasonBestStreak": 10,
                              "predictions": [{"roundId": 2, "result": "hit", "streak": 10}]},
                {2: dt.date(2026, 6, 17)})
    assert _run_fetch(picks).exit_code == 0
    assert load_saver_state(picks, season=2026).state == "active"       # sound auto-earn from not_earned
