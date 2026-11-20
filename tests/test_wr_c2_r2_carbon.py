"""Round-2 adversarial regression tests for whole-range C2 (receipt provenance, source order, and N1 boundary)."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner
import pytest

from bts import cli, contest_fetch, dm, picks, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from bts.leaderboard import auth
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


@pytest.fixture(autouse=True)
def _restore_mdp_cache(monkeypatch):
    monkeypatch.setattr(strategy, "_mdp_cache", strategy._mdp_cache)


def _own(directory: Path, day: str = DAY) -> Path:
    return directory.parent / "health_state" / "delivery_outcome_unknown" / f"{day}.json"


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


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize(
    "seed_kind",
    [
        "same_day_cutoff_shared_status",
        "copied_other_slate_c2_receipt",
        "failed_status_with_sent_fields",
        "superset_sent_sources",
        "duplicate_sent_sources",
        "nbsp_message_id",
        "boolean_schema_version",
    ],
)
def test_foreign_or_noncanonical_receipt_never_suppresses_c2(world, monkeypatch, route, seed_kind):
    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    own_path = _own(directory)
    own_path.parent.mkdir(parents=True, exist_ok=True)
    valid = {
        "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": "sent",
        "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"],
        "message_id": "prior-ack",
    }

    if seed_kind == "same_day_cutoff_shared_status":
        # Produce a real legacy status file for same-day CRITICAL:late_delivery (lacks message_id)
        legacy_path = directory.parent / "tmp_legacy_cutoff.json"
        alert.dispatch_dm_for_health_alerts(
            [alert.Alert("CRITICAL", "late_delivery", "cutoff refusal")],
            "offline.test",
            status_path=legacy_path,
            now_et_date=now[0].date(),
        )
        own_path.write_bytes(legacy_path.read_bytes())
        alert.send_dm.reset_mock()
    elif seed_kind == "copied_other_slate_c2_receipt":
        own_path.write_text(json.dumps(valid | {"sent_day": "2027-07-14"}))
    elif seed_kind == "failed_status_with_sent_fields":
        own_path.write_text(json.dumps(valid | {"status": "failed", "error": "prior failure"}))
    elif seed_kind == "superset_sent_sources":
        own_path.write_text(json.dumps(valid | {"sent_sources": ["CRITICAL:late_delivery", "CRITICAL:pick_delivery"]}))
    elif seed_kind == "duplicate_sent_sources":
        own_path.write_text(json.dumps(valid | {"sent_sources": ["CRITICAL:late_delivery", "CRITICAL:late_delivery"]}))
    elif seed_kind == "nbsp_message_id":
        own_path.write_text(json.dumps(valid | {"message_id": "\u00a0\t\n"}))
    elif seed_kind == "boolean_schema_version":
        own_path.write_text(json.dumps(valid | {"schema_version": True}))

    if route == "direct":
        assert scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "r2") is False
    else:
        assert scheduler.run_day(DAY, config) is None

    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert json.loads(own_path.read_bytes()) == valid | {
        "message_id": "health-dm", "body_sha256": sha256(_copy(DAY).encode("utf-8")).hexdigest(),
    }
    assert scheduler.run_day(DAY, config) is None
    alert.send_dm.assert_called_once()


def test_opt_in_multi_source_receipt_rejects_unsorted_sources_and_dedups_swapped_inputs(world, tmp_path):
    _, _, now, _ = world
    receipt_path = tmp_path / "multi.json"
    alerts_ba = [alert.Alert("CRITICAL", "b_check", "b"), alert.Alert("CRITICAL", "a_check", "a")]
    alerts_ab = [alert.Alert("CRITICAL", "a_check", "a"), alert.Alert("CRITICAL", "b_check", "b")]

    receipt_path.write_text(json.dumps({
        "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": "sent",
        "sent_day": DAY,
        "sent_sources": ["CRITICAL:b_check", "CRITICAL:a_check"],
        "message_id": "forged-unsorted",
        "body_sha256": sha256(alert.format_health_dm_body(alerts_ba).encode("utf-8")).hexdigest(),
    }))

    assert alert.dispatch_dm_for_health_alerts(
        alerts_ba, "offline.test", status_path=receipt_path, now_et_date=now[0].date(), require_message_id=True,
    ) is True
    alert.send_dm.assert_called_once()
    saved = json.loads(receipt_path.read_text())
    assert saved["sent_sources"] == ["CRITICAL:a_check", "CRITICAL:b_check"]
    assert alert.dispatch_dm_for_health_alerts(
        alerts_ba, "offline.test", status_path=receipt_path, now_et_date=now[0].date(), require_message_id=True,
    ) is False
    alert.send_dm.assert_called_once()

    assert alert.dispatch_dm_for_health_alerts(
        alerts_ab, "offline.test", status_path=receipt_path, now_et_date=now[0].date(), require_message_id=True,
    ) is True
    assert alert.send_dm.call_count == 2
    body = alert.format_health_dm_body(alerts_ab)
    assert alert.send_dm.call_args.args[1] == body
    assert json.loads(receipt_path.read_text()) == saved | {
        "body_sha256": sha256(body.encode("utf-8")).hexdigest(),
    }
    assert alert.dispatch_dm_for_health_alerts(
        alerts_ab, "offline.test", status_path=receipt_path, now_et_date=now[0].date(), require_message_id=True,
    ) is False
    assert alert.send_dm.call_count == 2


@pytest.mark.parametrize("route", ["direct", "classification"])
def test_double_down_unconfirmed_locks_double_without_naming_either_batter_and_handles_null_bluesky(
    world, monkeypatch, capsys, route,
):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.double_down = picks.Pick(
        "Juan Soto", 665742, "NYM", 2, "Blake Snell", 605483, 0.81, [], False, 102, f"{DAY}T23:40:00Z",
    )
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)

    if route == "direct":
        assert scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "dd") is False
    else:
        assert scheduler.run_day(DAY, config) is None

    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    sent_body = alert.send_dm.call_args.args[1]
    assert daily.pick.batter_name not in sent_body
    assert daily.double_down.batter_name not in sent_body
    dec = load_decision(DAY, directory)
    assert dec["action"] == "double" and dec["delivery_status"] == "locked_unconfirmed"

    # Malformed config["bluesky"] = None is contained with one bounded stderr diagnostic
    bad_dir = directory.with_name(f"bad_{route}")
    picks.save_pick(daily, bad_dir)
    bad_cfg = json.loads(json.dumps(config))
    bad_cfg["orchestrator"]["picks_dir"] = str(bad_dir)
    bad_cfg["bluesky"] = None
    if route == "direct":
        assert scheduler._deliver_and_lock_pick(daily, bad_cfg, bad_dir, scheduler_state(), DAY, "bad") is False
    else:
        assert scheduler.run_day(DAY, bad_cfg) is None
    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("Unknown-outcome alert failed: ")]
    assert len(err_lines) == 1


def test_recommendation_send_returning_none_matches_ledgered_watchdog_divergence(world, monkeypatch):
    import datetime as calendar
    from tests.test_pick_chosen_on_cs5a import record

    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    monkeypatch.setattr(calendar, "datetime", alert.datetime)
    monkeypatch.setattr(cli, "datetime", alert.datetime)
    set_dated(directory)

    daily = daily_pick()
    daily.chosen_on = record()
    picks.save_pick(daily, directory)
    dm.send_dm.return_value = None

    assert scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "lineup") is True
    dec = load_decision(DAY, directory)
    stored = picks.load_pick(DAY, directory)
    assert dec["scoreable"] is True and dec["delivery_status"] == "delivered"
    assert stored.delivery_attempted is True and not picks.pick_was_delivered(stored)

    dm.send_dm.reset_mock()
    scheduler.run_day(DAY, config)
    dm.send_dm.assert_not_called()
    alert.send_dm.assert_called_once_with("offline.test", _copy(DAY))
    assert "Do not enter or change a pick based on this alert." in alert.send_dm.call_args.args[1]

    dm.send_dm.return_value = "wd-ack"
    with patch.object(auth, "load_session_cookies", lambda: {}), \
            patch.object(auth, "extract_uid", lambda _: "u"), \
            patch.object(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x")), \
            patch.object(contest_fetch, "fetch_profile", lambda *_: {"predictions": []}), \
            patch.object(contest_fetch, "fetch_pending_predictions", lambda *_: []), \
            patch.object(cli, "_fetch_rounds", lambda: {1: date.fromisoformat(DAY)}), \
            patch.object(cli, "_fetch_bts_to_mlb", lambda: {daily.pick.batter_id: daily.pick.batter_id}):
        wd = CliRunner().invoke(cli.cli, [
            "check-pick-entered", "--picks-dir", str(directory),
            "--now-et", now[0].strftime("%Y-%m-%dT%H:%M:%S"),
            "--expected-username", "offline", "--dm-recipient", "offline.test",
        ])
    assert wd.exit_code == 1
    dm.send_dm.assert_called_once()
    assert "Fix it now!" in dm.send_dm.call_args.args[1]
