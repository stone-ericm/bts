"""Complete v3 season table and policy-reserve saver contract; no live data."""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health.contest_state import check

NOW = datetime(2027, 3, 24, 16, tzinfo=timezone.utc)
CURRENT = "2027-03-24T12:00:00Z"
OLD = "2026-09-26T12:00:00Z"


def _setup(picks, recorded, source, count=0, *, manual=False, streak=9):
    directory = picks / "account_state"
    directory.mkdir(exist_ok=True)
    body = {"active_streak": streak, "recorded_at": recorded, "source_date": source}
    if manual:
        body["override_expires_at"] = "2027-03-24T15:00:00Z"
    name = "contest_streak.manual.json" if manual else "contest_streak.json"
    (directory / name).write_text(json.dumps(body))
    (picks / "streak.json").write_text('{"streak":7}')
    for i in range(count):
        (picks / f"2027-03-{21+i}.json").write_text('{"result":"hit"}')


# row, recording, round, settled count, decision streak, status
STREAK_ROWS = [
    (1, OLD, "2027-03-24", 0, 9, "fresh"),
    (1, CURRENT, "2027-03-24", 1, 9, "fresh"),
    (1, None, "2027-03-24", 0, 9, "fresh"),
    (1, "invalid", "2027-03-24", 1, 9, "fresh"),
    (2, CURRENT, None, 0, 0, "season_start"),
    (2, CURRENT, "2026-09-26", 0, 0, "season_start"),
    (3, CURRENT, None, 1, 9, "stale"),
    (3, CURRENT, None, 2, 9, "stale"),
    (3, CURRENT, None, 3, 9, "stale"),
    (4, CURRENT, "2026-09-26", 1, 7, "model_only"),
    (5, OLD, None, 0, 0, "model_only"),
    (5, OLD, "2026-09-26", 1, 7, "model_only"),
    (6, None, "2026-09-26", 0, 0, "model_only"),
    (6, "invalid", "2026-09-26", 1, 7, "model_only"),
    (7, None, None, 0, 9, "stale"),
    (7, "invalid", None, 1, 9, "stale"),
]


@pytest.mark.parametrize("row,recorded,source,count,streak,status", STREAK_ROWS)
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_s1_v3_table(tmp_path, row, recorded, source, count, streak, status, manual, required):
    _setup(tmp_path, recorded, source, count, manual=manual)
    if row in (4, 5, 6) and required:
        with pytest.raises(ContestStateError) as caught:
            load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
        assert "season" in str(caught.value)
        if row in (5, 6):
            assert "2026" in str(caught.value)
    else:
        state = load_decision_streak_state(tmp_path, now=NOW, require_contest_state=required)
        assert (state.streak, state.status) == (streak, status)
        assert state.source == ("model" if row in (4, 5, 6) else "contest")


@pytest.mark.parametrize("recorded,source,count", [(CURRENT, None, 0), (CURRENT, "2026-09-26", 1), (OLD, None, 0)])
@pytest.mark.parametrize("required", [False, True])
def test_s1_manual_override_beside_rows_2_4_5(tmp_path, recorded, source, count, required):
    _setup(tmp_path, recorded, source, count)
    (tmp_path / "account_state" / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 13, "recorded_at": OLD,
        "override_expires_at": "2027-03-25T00:00:00Z"}))
    state = load_decision_streak_state(tmp_path, now=NOW, require_contest_state=required)
    assert (state.streak, state.source, state.status) == (13, "contest", "stale")


@pytest.mark.parametrize("row,recorded,source,count,streak,status", STREAK_ROWS)
@pytest.mark.parametrize("manual", [False, True])
def test_s2_v2_table(tmp_path, row, recorded, source, count, streak, status, manual):
    _setup(tmp_path, recorded, source, count, manual=manual)
    critical = [a.message for a in check(tmp_path, expected=True, now=NOW) if a.level == "CRITICAL"]
    expected = row in (4, 5, 6, 7) or (row == 3 and count >= 3)
    assert bool(critical) == expected
    if row in (5, 6):
        assert "2026" in critical[0] and "fetch" in critical[0]
    if row == 4:
        assert "local" in critical[0] and "settled" in critical[0] and "season" in critical[0]
    if row == 3 and count >= 3:
        assert "investigate the fetch" in critical[0]


@pytest.mark.parametrize("flag,streak,count,expected", [
    ("active", 8, 1, True), ("active", 12, 1, True),
    ("used", 8, 1, False), ("used", 12, 1, False),
    ("not_earned", 9, 1, True), ("not_earned", 10, 1, False),
    ("uninitialized", 12, 0, True), ("uninitialized", 12, 1, False),
])
@pytest.mark.parametrize("contest", [False, True])
def test_s7_policy_reserve_table(tmp_path, flag, streak, count, expected, contest):
    _setup(tmp_path, CURRENT, "2027-03-24", count, streak=streak)
    if not contest:
        (tmp_path / "account_state" / "contest_streak.json").unlink()
        (tmp_path / "streak.json").write_text(json.dumps({"streak": streak}))
    if flag != "uninitialized":
        (tmp_path / "account_state" / "saver_state.json").write_text(json.dumps({"season": 2027, "state": flag}))
    assert load_decision_streak_state(tmp_path, now=NOW).saver_available is expected


def test_s7_not_earned_selects_single_in_tracked_policy(tmp_path):
    import numpy as np
    from bts.strategy import DecisionContext, resolve_policy_decision

    _setup(tmp_path, CURRENT, "2027-03-24", 1, streak=8)
    (tmp_path / "account_state" / "saver_state.json").write_text('{"season":2027,"state":"not_earned"}')
    state = load_decision_streak_state(tmp_path, now=NOW)
    # Read the tracked artifact directly: no global policy cache or generated table.
    with np.load(Path(__file__).resolve().parents[1] / "data/models/mdp_policy.npz") as artifact:
        mdp = {"policy_table": artifact["policy_table"], "boundaries": artifact["boundaries"].tolist(),
               "season_length": int(artifact["season_length"])}
    ctx = DecisionContext(0.70, None, False, "2027-03-24", True, mdp)
    assert resolve_policy_decision(ctx, state.streak, False).action == "skip"
    decision = resolve_policy_decision(ctx, state.streak, state.saver_available)
    assert decision.days_effective == 180
    assert decision.action == "single"


@pytest.mark.parametrize("flag_season,flag,expected", [(2026, "used", True), (2027, "active", True)])
def test_s7_decision_uses_current_saver_season(tmp_path, flag_season, flag, expected):
    _setup(tmp_path, CURRENT, "2026-09-26", streak=12)
    (tmp_path / "account_state" / "saver_state.json").write_text(json.dumps({"season": flag_season, "state": flag}))
    assert load_decision_streak_state(tmp_path, now=NOW).saver_available is expected


@pytest.mark.parametrize("flag_season,expected", [(2026, "uninitialized"), (2027, "active")])
def test_s7_dashboard_uses_current_saver_season(tmp_path, flag_season, expected):
    from bts.web import saver_dashboard_context
    _setup(tmp_path, CURRENT, "2026-09-26", streak=12)
    (tmp_path / "account_state" / "saver_state.json").write_text(json.dumps({"season": flag_season, "state": "active"}))
    assert saver_dashboard_context(tmp_path, now=NOW).state == expected


@pytest.mark.parametrize("flag_season,expected_warn", [(2026, True), (2027, False)])
def test_s7_health_uses_current_saver_season(tmp_path, flag_season, expected_warn):
    _setup(tmp_path, CURRENT, "2026-09-26", streak=12)
    (tmp_path / "account_state" / "saver_state.json").write_text(json.dumps({"season": flag_season, "state": "active"}))
    warns = [a for a in check(tmp_path, expected=False, now=NOW) if "Streak Saver" in a.message]
    assert bool(warns) is expected_warn


@pytest.mark.parametrize("source,best,initial,expected", [
    ("2026-09-26", 14, "not_earned", "not_earned"),
    ("2027-03-23", 14, "not_earned", "active"),
    (None, 14, "not_earned", "not_earned"),
    ("2026-09-26", 9, None, "not_earned"),
])
def test_s7_fetch_earns_only_from_current_round(tmp_path, monkeypatch, source, best, initial, expected):
    import datetime as dt
    from click.testing import CliRunner
    import bts.cli as cli_module
    import bts.contest_fetch as fetch
    import bts.leaderboard.auth as auth
    from bts.saver_state import load_saver_state

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(dt, "datetime", FixedDatetime)
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {})
    monkeypatch.setattr(auth, "extract_uid", lambda _: "fixture")
    monkeypatch.setattr(auth, "fetch_login_session", lambda **kwargs: auth.AuthSession(xsid="fixture", user_id=1, username="fixture"))
    monkeypatch.setattr(fetch, "fetch_profile", lambda *args: {"activeStreak": 0, "seasonBestStreak": best,
        "predictions": [{"roundId": 1, "result": "hit"}] if source else []})
    monkeypatch.setattr(cli_module, "_fetch_rounds", lambda: {1: date.fromisoformat(source)} if source else {})
    directory = tmp_path / "account_state"
    directory.mkdir()
    if initial:
        (directory / "saver_state.json").write_text(json.dumps({"season": 2027, "state": initial}))
    result = CliRunner().invoke(cli_module.cli, ["fetch-contest-streak", "--picks-dir", str(tmp_path)])
    assert result.exit_code == 0, repr(result.exception)
    assert load_saver_state(tmp_path, season=2027).state == expected
