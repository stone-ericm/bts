"""Real entry-check receipts: batter-only match, then already-confirmed runs (Gemini B3)."""
import gzip
import itertools
import json
from datetime import date as date_cls
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from bts.cli import cli
from bts.daily_decision import write_decision
from bts.entry_receipt import discover
from bts.picks import DailyPick, Pick, save_pick


@pytest.fixture(autouse=True)
def _fixed_test_provenance(monkeypatch):
    """Receipt provenance must not launch Git even indirectly from these tests."""
    monkeypatch.setattr("bts.picks._git_head_sha", lambda *a, **k: "0" * 40)


def _setup_committed_day(root: Path, date: str, batter_id: int = 592450,
                         game_time_utc: str = "2027-05-15T23:10:00Z") -> Path:
    picks = root / "picks"
    picks.mkdir(parents=True, exist_ok=True)
    daily = DailyPick(
        date=date,
        run_time="17:00 ET",
        pick=Pick(
            batter_name="Aaron Judge", batter_id=batter_id, team="NYY",
            lineup_position=2, pitcher_name="Brayan Bello", pitcher_id=678394,
            p_game_hit=0.81, flags=[], projected_lineup=False,
            game_pk=9001, game_time=game_time_utc,
        ),
        double_down=None,
        runner_up=None,
        delivered_at=f"{date}T17:05:00-04:00",
        notification_sent=True, notification_channel="bluesky_dm",
    )
    save_pick(daily, picks)
    write_decision(
        date, picks, action="single", source="mdp",
        primary={"batter_id": batter_id, "batter_name": "Aaron Judge", "p_game_hit": 0.81},
        delivery_status="delivered", scoreable=True,
    )
    return picks


def _write_units_snapshot(root: Path, date: str, feed_id: int | None, round_id: int = 42,
                          unit_id: int = 700, snap_name: str = "2027-05-15T180000Z") -> None:
    units_dir = root / "leaderboard" / "static_snapshots" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": snap_name,
        "units": [{
            "id": unit_id,
            "roundId": round_id,
            "unitType": "ROUND",
            "feedId": feed_id,
            "startDate": f"{date}T04:00:00Z",
            "endDate": f"{date}T23:59:59Z",
            "title": date,
        }],
    }
    gz_path = units_dir / f"{snap_name}.json.gz"
    gz_path.write_bytes(gzip.compress(json.dumps(payload).encode()))


def _mock_contest_env(date: str, bts_player_id: int = 99, mlb_id: int = 592450,
                      round_id: int = 42, unit_id: int = 700):
    d_obj = date_cls.fromisoformat(date)
    session = SimpleNamespace(user_id="uid-1", username="stonehengee", xsid="xsid-1")
    profile = {"current_streak": 5, "predictions": []}
    pending = [{
        "roundId": round_id,
        "unitId": unit_id,
        "playerId": bts_player_id,
        "number": 1,
        "result": "pending",
    }]
    rounds = {round_id: d_obj}
    crosswalk = {bts_player_id: mlb_id}
    return session, profile, pending, rounds, crosswalk


def _batter_only_sequence(tmp_path, monkeypatch, later_feed_id):
    date = "2027-05-15"
    picks = _setup_committed_day(tmp_path, date)
    _write_units_snapshot(tmp_path, date, feed_id=None)
    session, profile, pending, rounds, crosswalk = _mock_contest_env(date)
    runner = CliRunner()
    with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
         patch("bts.leaderboard.auth.extract_uid", return_value="uid-1"), \
         patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
         patch("bts.contest_fetch.fetch_profile", return_value=profile), \
         patch("bts.contest_fetch.fetch_pending_predictions", return_value=pending) as fetch_pending, \
         patch("bts.cli._fetch_rounds", return_value=rounds), \
         patch("bts.cli._fetch_bts_to_mlb", return_value=crosswalk):
        for time in ("18:05", "18:20", "18:35"):
            if time == "18:20":
                _write_units_snapshot(tmp_path, date, feed_id=later_feed_id, snap_name="2027-05-15T221000Z")
            res = runner.invoke(cli, ["check-pick-entered", "--picks-dir", str(picks),
                "--expected-username", "stonehengee", "--now-et", f"{date}T{time}:00"])
            assert res.exit_code == 0, res.output
    assert fetch_pending.call_count == 1  # already_confirmed makes no new observation
    receipts = discover(picks, date)
    assert sorted(r["outcome"] for r in receipts) == ["already_confirmed", "already_confirmed", "observed"]
    observed = next(r for r in receipts if r["outcome"] == "observed")
    assert observed["verifier"]["reason"] == "match"
    assert [s["state"] for s in observed["qualification"]["slots"]] == ["unit_unverified"]
    cfg = tmp_path / "orch.toml"
    cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda date: [{"gamePk": 9001}])
    dms = []
    monkeypatch.setattr("bts.dm.send_dm", lambda *args: dms.append(args))
    result = runner.invoke(cli, ["day-outcome", "--date", date, "--config", str(cfg),
        "--picks-dir", str(picks), "--expected-username", "stonehengee", "--dm-recipient", "me.bsky.social"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "entered_game_unverified" and dms == []
    return picks, observed, crosswalk


def test_batter_only_observation_then_already_confirmed_is_clean(tmp_path, monkeypatch):
    _batter_only_sequence(tmp_path, monkeypatch, later_feed_id=9001)


def test_limit_later_wrong_game_binding_is_not_reobserved_after_batter_only_confirmation(tmp_path, monkeypatch):
    """Accepted limit: a clean batter-only result does not prove the right doubleheader game."""
    from bts.entry_receipt import load_units, qualify
    picks, observed, crosswalk = _batter_only_sequence(tmp_path, monkeypatch, later_feed_id=9002)
    units, _ = load_units(picks, 2027)
    assert units[700] == (9002, 42)
    assert observed["selection"]["slots"][0]["game_pk"] == 9001
    # A fresh observation would identify the wrong game; sealed receipts retain the earlier unverified result.
    now_qualified = qualify(observed["selection"]["slots"], observed["observation"]["rows"], crosswalk, units)
    assert [s["state"] for s in now_qualified["slots"]] == ["wrong_game"]


PRECEDENCE_CASES = [
    pytest.param(order, state, id="-".join(order))
    for kinds, state in [
        (("absent",), "not_entered"),
        (("late", "absent"), "entered_late"),
        (("batter", "late", "absent"), "entered_game_unverified"),
        (("full", "batter", "late", "absent"), "entered"),
    ]
    for order in itertools.permutations(kinds)
]


@pytest.mark.parametrize("order,expected", PRECEDENCE_CASES)
def test_receipt_precedence_is_independent_of_observation_order(tmp_path, order, expected):
    """Reviewer probe: all 33 orders preserve full > batter-only > late > absence."""
    from bts.daily_decision import load_decision_bytes
    from bts.day_outcome import classify_day
    from bts.entry_receipt import _sha
    date = "2027-03-24"
    picks = tmp_path / "picks"
    write_decision(date, picks, action="single", source="mdp", scoreable=True, delivery_status="delivered",
                   primary={"batter_id": 1, "batter_name": "A", "game_pk": 9001, "p_game_hit": .8})
    decision, raw = load_decision_bytes(date, picks)

    def receipt(kind):
        return {"schema": "bts_pick_entry_receipt_v1", "outcome": "observed", "et_date": date, "season": 2027,
                "degraded": [], "account": {"username": "owner"},
                "selection": {"commit": {"decision_sha256": _sha(raw)}},
                "observation": {"before_cutoff": kind != "late", "response_completed_at": date + "T23:06:00Z"},
                "verifier": {"reason": "no_pick" if kind == "absent" else "match"},
                "qualification": {"all_confirmed": kind in ("full", "late"),
                                  "slots": [{"state": "unit_unverified" if kind == "batter" else
                                                       "confirmed" if kind in ("full", "late") else "missing"}]}}

    outcome = classify_day(date=date, games=1, decision=decision, decision_sha256=_sha(raw),
                           receipts=[receipt(kind) for kind in order], intent="enter", expected_username="owner")
    assert outcome.state == expected
