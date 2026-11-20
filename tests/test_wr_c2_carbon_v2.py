"""Adversarial regression tests for whole-range C2 (unknown-outcome alert after restart).

Kills all four mutations surviving tests/test_wr_c2_recovery.py:
  1. Calling `_alert_delivery_outcome_unknown` BEFORE `_write_classification_decision` in `run_day`
     (survived test_wr_c2_recovery.py because assertions inside `alert.send_dm.side_effect` raise
     `Exception` subclasses caught by `_dispatch_with_acknowledgement`).
  2. Dropping `O_NOFOLLOW` on the dedicated receipt directory open (reading/writing through `dir_link`).
  3. Dropping `if not require_message_id:` around legacy resurface-banner computation when a prior
     receipt or symlink target carries `status="failed"` with `critical_count > 0`.
  4. Dropping the post-send `_check_receipt_entry` guard in `_replace_receipt` when an alias or
     special file appears at `<slate>.json` while `send_dm` is in flight.
Also covers first-encounter post-midnight slate keying (ordinary midnight + New Year), read-only
`0o500` receipt directory dedup vs write failure, acknowledged-receipt preservation without a
recipient, and `crash_then_corrected` `season-contract status` + B2 watchdog agreement.
"""

from __future__ import annotations

from hashlib import sha256

from datetime import datetime
import json
import os
from pathlib import Path
import signal
import stat
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import cli, dm, picks, scheduler, strategy
from bts.daily_decision import decision_path, load_decision, write_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


PREFIX = "Unknown-outcome alert failed: "


@pytest.fixture(autouse=True)
def _restore_mdp_cache(monkeypatch):
    monkeypatch.setattr(strategy, "_mdp_cache", strategy._mdp_cache)


def _own(directory: Path, day: str = DAY) -> Path:
    return directory.parent / "health_state" / "delivery_outcome_unknown" / f"{day}.json"


def _shared(directory: Path) -> Path:
    return directory.parent / "health_state" / "health_dm_delivery_status.json"


def _copy(day: str = DAY) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def _invoke_route(route: str, config: dict, directory: Path, day: str = DAY) -> None:
    if route == "direct":
        daily = picks.load_pick(day, directory)
        st = scheduler_state()
        st.date = day
        assert scheduler._deliver_and_lock_pick(daily, config, directory, st, day, "c2-carbon") is False
    else:
        scheduler.run_day(day, config)


@pytest.mark.parametrize("prior", [None, "locked_unconfirmed", "delivered", "terminal", "write_none"])
def test_classification_alert_runs_strictly_after_decision_commit(world, monkeypatch, capsys, prior):
    """Witnesses disk state OUTSIDE send_dm so _dispatch_with_acknowledgement cannot swallow failures."""
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    if prior == "terminal":
        daily.result = "hit"
    picks.save_pick(daily, directory)
    path = decision_path(DAY, directory)
    if prior in {"locked_unconfirmed", "delivered", "terminal"}:
        write_decision(
            DAY, directory, action="single", source="mdp",
            delivery_status=prior if prior != "terminal" else "locked_unconfirmed",
            primary={"batter_id": daily.pick.batter_id, "batter_name": daily.pick.batter_name},
            scoreable=True,
        )
    if prior == "write_none":
        monkeypatch.setattr("bts.daily_decision.write_decision", Mock(return_value=None))
    original = path.read_bytes() if path.exists() else None

    witnessed: list[dict] = []

    def capture_send(recipient, body):
        st = scheduler.load_state(DAY, directory)
        dec = load_decision(DAY, directory)
        raw = path.read_bytes() if path.exists() else None
        witnessed.append({
            "pick_locked": st.pick_locked if st else None,
            "committed_pick_written": st.committed_pick_written if st else None,
            "decision_status": dec.get("delivery_status") if dec else None,
            "raw": raw,
        })
        return None if len(witnessed) == 1 else "ack-after-commit"

    alert.send_dm.side_effect = capture_send
    scheduler.run_day(DAY, config)
    scheduler.run_day(DAY, config)

    unknown_calls = [c.args[1] for c in alert.send_dm.call_args_list if "DELIVERY OUTCOME UNKNOWN" in c.args[1]]
    assert unknown_calls == [_copy(DAY), _copy(DAY)]
    assert json.loads(_own(directory).read_bytes()) == {
        "schema_version": "bts_health_dm_delivery_status_v1",
        "status": "sent",
        "message_id": "ack-after-commit",
        "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"],
        "body_sha256": sha256(_copy(DAY).encode("utf-8")).hexdigest(),
    }
    first_snap = witnessed[0]
    assert first_snap["pick_locked"] is True
    if original is not None:
        assert first_snap["raw"] == original
        assert first_snap["committed_pick_written"] is True
    elif prior == "write_none":
        assert first_snap["raw"] is None
        assert first_snap["committed_pick_written"] is False
    else:
        assert first_snap["decision_status"] == "locked_unconfirmed"
        assert first_snap["committed_pick_written"] is True
    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(PREFIX)]
    assert len(err_lines) == 1 and "send_dm returned no usable message id" in err_lines[0]


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("target_has_sent_receipt", [False, True])
def test_symlinked_receipt_directory_neither_reads_nor_writes_target(
    world, monkeypatch, capsys, route, target_has_sent_receipt,
):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    target_dir = directory.parent / "elsewhere_dir"
    target_dir.mkdir(parents=True)
    target_receipt = target_dir / f"{DAY}.json"
    seed_bytes = json.dumps({
        "schema_version": "bts_health_dm_delivery_status_v1",
        "status": "sent", "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"], "message_id": "forged",
    }).encode()
    if target_has_sent_receipt:
        target_receipt.write_bytes(seed_bytes)

    own_dir = _own(directory).parent
    own_dir.parent.mkdir(parents=True, exist_ok=True)
    own_dir.symlink_to(target_dir, target_is_directory=True)

    _invoke_route(route, config, directory)

    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert own_dir.is_symlink()
    if target_has_sent_receipt:
        assert target_receipt.read_bytes() == seed_bytes
    else:
        assert not target_receipt.exists()
    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(PREFIX)]
    assert len(err_lines) == 1 and "open receipt directory" in err_lines[0]


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("seeded_via", ["own_file", "symlink_to_shared"])
def test_prior_failed_with_critical_count_never_adds_resurface_banner(
    world, monkeypatch, capsys, route, seeded_via,
):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    legacy_failed = json.dumps({
        "schema_version": "bts_health_dm_delivery_status_v1",
        "status": "failed", "critical_count": 1, "error": "prior outage",
    }).encode()
    own_path = _own(directory)
    own_path.parent.mkdir(parents=True)
    shared_path = _shared(directory)
    if seeded_via == "own_file":
        own_path.write_bytes(legacy_failed)
    else:
        shared_path.write_bytes(legacy_failed)
        own_path.symlink_to(shared_path)

    _invoke_route(route, config, directory)

    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert "Previous health alert DM FAILED" not in alert.send_dm.call_args.args[1]
    if seeded_via == "symlink_to_shared":
        assert shared_path.read_bytes() == legacy_failed
        assert own_path.is_symlink()
        err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(PREFIX)]
        assert len(err_lines) == 1 and "read receipt" in err_lines[0]
    else:
        assert json.loads(own_path.read_bytes())["status"] == "sent"


@pytest.mark.parametrize("race_kind", ["symlink", "hardlink", "fifo"])
def test_mid_send_alias_or_special_at_receipt_is_left_in_place(world, capsys, race_kind):
    config, directory, _, _ = world
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    shared_path = _shared(directory)
    shared_path.parent.mkdir(parents=True)
    shared_seed = b'{"status": "sent", "message_id": "keep"}\n'
    shared_path.write_bytes(shared_seed)
    own_path = _own(directory)

    def race_during_send(recipient, body):
        if race_kind == "symlink":
            own_path.symlink_to(shared_path)
        elif race_kind == "hardlink":
            os.link(shared_path, own_path)
        else:
            os.mkfifo(own_path)
        return "ack-after-race"

    alert.send_dm.side_effect = race_during_send

    def on_alarm(*_):
        raise BaseException("receipt write blocked on special file")

    prev = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(5)
    try:
        _invoke_route("direct", config, directory)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert shared_path.read_bytes() == shared_seed
    assert list(own_path.parent.glob(f".{DAY}.json.*")) == []
    if race_kind == "symlink":
        assert own_path.is_symlink()
    elif race_kind == "hardlink":
        assert own_path.lstat().st_nlink == 2
    else:
        assert stat.S_ISFIFO(own_path.lstat().st_mode)
    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(PREFIX)]
    assert len(err_lines) == 1 and "write receipt" in err_lines[0]


@pytest.mark.parametrize("has_valid_sent", [False, True])
def test_readonly_receipt_directory_dedups_sent_and_reports_write_failure(world, capsys, has_valid_sent):
    config, directory, _, _ = world
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    own_path = _own(directory)
    own_path.parent.mkdir(parents=True)
    if has_valid_sent:
        own_path.write_text(json.dumps({
            "schema_version": "bts_health_dm_delivery_status_v1",
            "status": "sent", "sent_day": DAY,
            "sent_sources": ["CRITICAL:late_delivery"], "message_id": "prior-ack",
            "body_sha256": sha256(_copy(DAY).encode("utf-8")).hexdigest(),
        }))
    own_path.parent.chmod(0o500)
    try:
        _invoke_route("direct", config, directory)
    finally:
        own_path.parent.chmod(0o700)

    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(PREFIX)]
    if has_valid_sent:
        alert.send_dm.assert_not_called()
        assert err_lines == []
    else:
        alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
        assert len(err_lines) == 1 and "write receipt" in err_lines[0]


@pytest.mark.parametrize(("day", "following"), [("2027-03-25", "2027-03-26"), ("2027-12-31", "2028-01-01")])
def test_first_encounter_after_midnight_keys_receipt_by_slate_and_preserves_ack_without_recipient(
    world, monkeypatch, day, following,
):
    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    et = ZoneInfo("America/New_York")
    games = {}
    for slate in (day, following):
        dp = daily_pick()
        dp.date = slate
        dp.pick.game_time = f"{slate}T23:10:00Z"
        if dp.double_down:
            dp.double_down.game_time = f"{slate}T23:40:00Z"
        dp.delivery_attempted = True
        picks.save_pick(dp, directory)
        games[slate] = [{
            "gamePk": 101, "gameDate": dp.pick.game_time,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "H"}}},
        }]
    scheduler.fetch_schedule.side_effect = games.get

    now[0] = datetime.fromisoformat(f"{following}T00:03:00").replace(tzinfo=et)
    scheduler.run_day(day, config)
    alert.send_dm.assert_called_once_with("offline.test", _copy(day))
    assert _own(directory, day).is_file() and not _own(directory, following).exists()
    saved_bytes = _own(directory, day).read_bytes()
    assert json.loads(saved_bytes)["sent_day"] == day

    no_recip_cfg = json.loads(json.dumps(config))
    no_recip_cfg["bluesky"].pop("dm_recipient")
    scheduler.run_day(day, no_recip_cfg)
    assert _own(directory, day).read_bytes() == saved_bytes
    alert.send_dm.assert_called_once()


def test_corrected_state_operator_flow_agrees_with_status_and_watchdog(world, monkeypatch):
    import datetime as calendar
    from types import SimpleNamespace
    from bts import contest_fetch
    from bts.leaderboard import auth

    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    monkeypatch.setattr(calendar, "datetime", alert.datetime)
    monkeypatch.setattr(cli, "datetime", alert.datetime)

    set_dated(directory, streak=1, best=10, saver="not_earned")
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    set_dated(directory, streak=46, best=46, saver="used")

    now[0] = now[0].replace(hour=18, minute=40)
    scheduler.run_day(DAY, config)
    assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"
    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert daily.pick.batter_name not in alert.send_dm.call_args.args[1]

    status_out = CliRunner().invoke(
        cli.cli, ["season-contract", "status", "--picks-dir", str(directory)], catch_exceptions=False,
    )
    assert status_out.exit_code == 0
    assert "live decision now: accepted; source=contest; streak=46; status=fresh; saver=used" in status_out.output

    monkeypatch.setattr(auth, "load_session_cookies", lambda: {})
    monkeypatch.setattr(auth, "extract_uid", lambda _: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
    monkeypatch.setattr(contest_fetch, "fetch_profile", lambda *_: {"activeStreak": 46, "seasonBestStreak": 46, "predictions": []})
    monkeypatch.setattr(contest_fetch, "fetch_pending_predictions", lambda *_: [])
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: now[0].date()})
    monkeypatch.setattr(cli, "_fetch_bts_to_mlb", lambda: {daily.pick.batter_id: daily.pick.batter_id})

    wd = CliRunner().invoke(cli.cli, [
        "check-pick-entered", "--picks-dir", str(directory),
        "--now-et", now[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "--expected-username", "offline", "--dm-recipient", "offline.test",
    ])
    assert wd.exit_code == 1
    dm.send_dm.assert_called_once()
    wd_body = dm.send_dm.call_args.args[1]
    assert "Delivery of this pick was NOT confirmed" in wd_body
    assert "Check your DMs and MLB entry before entering anything" in wd_body
    assert "Fix it now!" not in wd_body
