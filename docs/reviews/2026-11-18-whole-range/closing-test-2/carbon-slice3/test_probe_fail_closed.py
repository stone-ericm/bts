"""Slice 3 probe family 3: fail-closed and fail-loud paths across all new refusals and raises in 619abda7..HEAD.

Adoptable as a pytest regression suite and runnable standalone via:
    nice -n 19 .venv/bin/python .carbon-review/whole-range/test_probe_fail_closed.py
No module-level import of bts.model.predict, no leaked process-wide state, no network.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import (
    cli,
    contest_fetch,
    contest_state,
    daily_decision,
    day_outcome,
    dm,
    entry_intent,
    picks,
    scheduler,
    strategy,
    util,
)
from bts.data import build as data_build, pull as data_pull
from bts.features import probable_pitcher_cache as pp_cache
from bts.health import (
    alert as health_alert,
    pick_entry as health_pick_entry,
    probable_pitcher_cache as health_pp_cache,
)
from bts.leaderboard import auth as lb_auth

ET = ZoneInfo("America/New_York")


@contextmanager
def block_network():
    def _deny(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args[:2]}")

    with ExitStack() as stack:
        for method in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, method, _deny))
        stack.enter_context(patch.object(socket, "getaddrinfo", _deny))
        yield


def test_live_gate_refusal_has_no_dm_but_leaves_heartbeat_stale(tmp_path: Path):
    """6a live_gate_problem exits 2 before heartbeat/DM; check_heartbeat.is_stale catches missing/aged heartbeat."""
    import importlib.util

    hb_path = tmp_path / "data" / ".heartbeat"
    cfg_path = tmp_path / "orch.toml"
    cfg_path.write_text(
        f'[orchestrator]\npicks_dir = "{tmp_path / "picks"}"\nheartbeat_path = "{hb_path}"\n'
        '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
        '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n'
    )
    dms: list[tuple[str, str]] = []
    runner = CliRunner()
    with block_network(), patch.object(health_alert, "send_dm", lambda r, b: dms.append((r, b))):
        res = runner.invoke(cli.cli, ["schedule", "--config", str(cfg_path), "--date", "2027-07-15"])
    assert res.exit_code == 2
    assert "refusing to start" in res.stderr
    assert dms == []  # No DM sent by bts schedule itself
    assert not hb_path.exists()

    # Operator-visible signal: check_heartbeat.py sees missing/stale heartbeat (and NRestarts churn)
    spec = importlib.util.spec_from_file_location(
        "check_heartbeat_probe", Path(__file__).resolve().parents[2] / "scripts" / "check_heartbeat.py"
    )
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)
    stale, reason = ch.is_stale(hb_path)
    assert stale is True and "not found" in reason


def test_fetch_contest_streak_prior_non_dict_or_bad_utf8_crashes_without_alert(tmp_path: Path):
    """Item 3 finding: corrupt prior contest_streak.json ([] or bad UTF-8) crashes fetch-contest-streak before _fail()."""
    picks_dir = tmp_path / "data" / "picks"
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True)
    status_path = tmp_path / "data" / "health_state" / "contest_streak_fetch_status.json"
    session = SimpleNamespace(user_id=1, username="stonehengee", xsid="x")
    runner = CliRunner()

    for bad_bytes, expected_exc in ((b"[]", AttributeError), (b"\xff", UnicodeDecodeError)):
        (acct / "contest_streak.json").write_bytes(bad_bytes)
        dms: list[str] = []
        with block_network(), \
             patch.object(lb_auth, "load_session_cookies", return_value={}), \
             patch.object(lb_auth, "extract_uid", return_value="u1"), \
             patch.object(lb_auth, "fetch_login_session", return_value=session), \
             patch.object(dm, "send_dm", lambda _r, msg: dms.append(msg) or "id"):
            res = runner.invoke(cli.cli, [
                "fetch-contest-streak", "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "operator.bsky.social",
            ])
        assert res.exit_code == 1
        assert isinstance(res.exception, expected_exc)
        # Crash bypassed _fail(): NO fetch alert DM sent and NO status file written!
        assert dms == []
        assert not status_path.exists()


def test_watchdog_persistent_auth_or_identity_failure_is_silent_until_morning(tmp_path: Path):
    """Item 3: check-pick-entered on persistent AuthError or identity_mismatch sends 0 DMs and 0 EOD alerts;
    only 08:30 day-outcome (and fetch-contest-streak in enter mode) surfaces ENTRY UNVERIFIABLE."""
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    day = "2027-07-15"
    dp = picks.DailyPick(
        date=day, run_time="2027-07-15T18:00:00-04:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-07-15T19:10:00-04:00"),
        double_down=None, runner_up=None,
        delivered_at="2027-07-15T18:00:00-04:00",
        notification_sent=True, notification_channel="bluesky_dm", notification_id="dm1",
    )
    picks.save_pick(dp, picks_dir)
    daily_decision.write_decision(
        day, picks_dir, action="single", source="mdp",
        primary={"batter_id": 680757, "batter_name": "Steven Kwan", "game_pk": 101},
        delivery_status="delivered", scoreable=True,
    )
    runner = CliRunner()
    dms: list[str] = []
    wrong_session = SimpleNamespace(user_id="u2", username="wrong_user", xsid="x2")

    with block_network(), \
         patch.object(lb_auth, "load_session_cookies", return_value={}), \
         patch.object(lb_auth, "extract_uid", return_value="u2"), \
         patch.object(lb_auth, "fetch_login_session", return_value=wrong_session), \
         patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40), \
         patch.object(dm, "send_dm", lambda _r, msg: dms.append(msg) or "id"):
        for slot in ("18:00", "18:15", "18:30", "18:45", "19:00"):
            res = runner.invoke(cli.cli, [
                "check-pick-entered", "--cron", "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "operator.bsky.social",
                "--now-et", f"{day}T{slot}:00",
            ])
            assert res.exit_code == 0

    assert dms == []
    assert not (tmp_path / "data" / "health_state" / "pick_entry_check.json").exists()
    # EOD pick_entry health check is ALSO silent because no marker was written!
    assert health_pick_entry.check(
        picks_dir, today=date(2027, 7, 15), now=datetime(2027, 7, 15, 23, 0, tzinfo=ET)
    ) == []
    # Next morning 08:30 day-outcome catches it as entry_unverifiable (exit 1 -> DM)
    out = day_outcome.gather(
        day, picks_dir, {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}},
        expected_username="stonehengee",
        fetch_schedule=lambda _d: [{"gamePk": 101, "status": {"detailedState": "Final"}}],
    )
    assert out.state == "entry_unverifiable" and not out.ok


def test_data_build_untyped_feed_and_pull_non_dict_raises(tmp_path: Path):
    """Item 3: build_season raises ValueError on untyped raw feed or vanished R feeds when parquet exists;
    download_game_feed raises ValueError on non-dict JSON body."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "2027").mkdir(parents=True)
    # 1) Untyped feed in pre-season raw/2027 raises every 03:00 cron run until a regular-season game exists
    (raw_dir / "2027" / "100.json").write_text(json.dumps({"gameData": {"game": {}}}))
    with pytest.raises(ValueError, match="missing or unreadable gameData.game.type"):
        data_build.build_season(raw_dir, tmp_path / "pa_2027.parquet", 2027)

    # 2) Existing parquet with 0 eligible R feeds raises instead of leaving stale parquet silently
    (raw_dir / "2027" / "100.json").write_text(json.dumps({"gameData": {"game": {"type": "S"}}}))
    out_pq = tmp_path / "pa_2027.parquet"
    out_pq.write_bytes(b"PAR1")
    with pytest.raises(ValueError, match="season output already exists"):
        data_build.build_season(raw_dir, out_pq, 2027)

    # 3) download_game_feed refuses non-object JSON without overwriting target
    import io

    with block_network(), patch.object(data_pull, "urlopen", side_effect=lambda *_a, **_k: io.BytesIO(b"[1, 2, 3]")):
        with pytest.raises(ValueError, match="must be a JSON object"):
            data_pull.download_game_feed(777, raw_dir / "2027")
    assert not (raw_dir / "2027" / "777.json").exists()


def test_malformed_schedule_and_bounded_readers_and_pp_cache_signals(tmp_path: Path):
    """Item 3: W5 schedule ValueError, 6b-2 non-blocking status reader, F4 streak.json FIFO, W15 corrupt status."""
    # 1) W5 fetch_schedule raises ValueError on malformed bodies (NaN, non-dict, missing games list)
    import io

    for payload in (b"[]", b'{"dates": null}', b'{"dates": [{"games": "bad"}]}', b'{"dates": [], "x": NaN}'):
        with block_network(), patch.object(scheduler, "retry_urlopen", side_effect=lambda *_a, p=payload, **_k: io.BytesIO(p)):
            with pytest.raises(ValueError):
                scheduler.fetch_schedule("2027-07-15")

    # 2) 6b-2 DM status FIFO does not block and still sends CRITICAL DM
    fifo_status = tmp_path / "fifo_status.json"
    os.mkfifo(fifo_status)
    sent: list[str] = []
    with block_network(), patch.object(health_alert, "send_dm", lambda _r, b: sent.append(b) or "ok"):
        ok = health_alert.dispatch_dm_for_health_alerts(
            [health_alert.Alert("CRITICAL", "probe", "test alert")],
            "operator.bsky.social",
            status_path=fifo_status,
            now_et_date=date(2027, 7, 15),
        )
    assert ok is True and len(sent) == 1

    # 3) F4 streak.json FIFO is caught non-blocking by day_outcome._streak_problem
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    os.mkfifo(picks_dir / "streak.json")
    assert day_outcome._streak_problem(picks_dir / "streak.json") == "not a regular file"

    # 4) W15 probable_pitcher_cache corrupt status record still emits WARN alert at EOD
    pp_status_dir = tmp_path / "pp_status"
    pp_status_dir.mkdir()
    (pp_status_dir / "2027-07-15.invalid_entries.000000000000.json").write_text("{corrupt")
    alerts = health_pp_cache.check(pp_status_dir, today=date(2027, 7, 15))
    assert len(alerts) == 1 and alerts[0].level == "WARN" and "status unreadable" in alerts[0].message


if __name__ == "__main__":
    import tempfile

    checks = [
        test_live_gate_refusal_has_no_dm_but_leaves_heartbeat_stale,
        test_fetch_contest_streak_prior_non_dict_or_bad_utf8_crashes_without_alert,
        test_watchdog_persistent_auth_or_identity_failure_is_silent_until_morning,
        test_data_build_untyped_feed_and_pull_non_dict_raises,
        test_malformed_schedule_and_bounded_readers_and_pp_cache_signals,
    ]
    results = []
    for fn in checks:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            results.append({"test": fn.__name__, "status": "PASS"})
    out_dir = Path(__file__).resolve().parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_fail_closed.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"probe_fail_closed: {len(results)}/{len(checks)} PASS")
