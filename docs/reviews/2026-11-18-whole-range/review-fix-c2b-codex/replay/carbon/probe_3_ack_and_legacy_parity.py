"""Item 3 probe: message-id acknowledgement rules and byte-identical legacy callers vs base."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_SRC = HERE.parents[1] / "base" / "src"
HEAD_SRC = ROOT / "src"
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
PINNED_NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)


def u7_copy(day: str = DAY) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def _legacy_worker(mode: str) -> dict:
    """Run every existing caller against base or head (omitted / explicit False)."""
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
    from bts.health import alert, runner

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

    results = {}

    def capture_logs():
        records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append((record.levelname, record.getMessage()))

        handler = ListHandler(level=logging.DEBUG)
        alert.log.addHandler(handler)
        prev_level = alert.log.level
        alert.log.setLevel(logging.DEBUG)
        return records, handler, prev_level

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch.object(contest_state, "datetime", PinnedClock), \
                patch.object(daily_decision, "datetime", PinnedClock), \
                patch.object(alert, "datetime", PinnedClock), \
                patch.object(scheduler, "_now_et", lambda: PINNED_NOW), \
                patch.object(dm, "send_dm", Mock(return_value="pick-dm")), \
                patch.object(posting, "post_to_bluesky", Mock(return_value="at://1")), \
                patch.object(strategy, "_mdp_cache", None):

            # If mode == 'explicit_false', wrap dispatch_dm_for_health_alerts to pass require_message_id=False
            orig_dispatch = alert.dispatch_dm_for_health_alerts
            if mode == "explicit_false":
                def wrapped_dispatch(*a, **kw):
                    kw.setdefault("require_message_id", False)
                    return orig_dispatch(*a, **kw)
                dispatch_patch = patch.Multiple
            else:
                wrapped_dispatch = orig_dispatch

            with (patch.object(alert, "dispatch_dm_for_health_alerts", wrapped_dispatch)
                  if mode == "explicit_false" else nullcontext()), \
                 (patch.object(runner, "dispatch_dm_for_health_alerts", wrapped_dispatch)
                  if mode == "explicit_false" else nullcontext()):

                # 1. EOD runner.run_all_checks
                sub = root / "eod"
                pdir = sub / "picks"
                pdir.mkdir(parents=True)
                sent = Mock(return_value="eod-id")
                records, h, lvl = capture_logs()
                err = StringIO()
                try:
                    eod_alerts = [
                        alert.Alert("CRITICAL", "pick_delivery", "EOD pick missing"),
                        alert.Alert("WARN", "saver_state", "Saver warn"),
                    ]
                    with patch.object(alert, "send_dm", sent), \
                            patch.object(runner, "_safe_run", lambda n, _: eod_alerts if n == "post_failure" else []), \
                            patch.object(runner, "apply_warn_attention_policy",
                                         lambda al, **_: ([], [al[1]])), \
                            patch.object(runner.realized_calibration, "_current_deploy_iso", lambda _: None), \
                            redirect_stderr(err):
                        runner.run_all_checks(pdir, sub / "models", "offline.test", today=PINNED_NOW.date())
                finally:
                    alert.log.removeHandler(h)
                    alert.log.setLevel(lvl)
                results["eod"] = {
                    "dm_calls": [c.args for c in sent.call_args_list],
                    "status_bytes": (sub / "health_state/health_dm_delivery_status.json").read_text(),
                    "logs": records,
                    "stderr": err.getvalue(),
                }

                # 2. 4c stale-reading warning
                sub = root / "4c"
                pdir = sub / "picks"
                pdir.mkdir(parents=True)
                sent = Mock(return_value="4c-id")
                records, h, lvl = capture_logs()
                err = StringIO()
                try:
                    rec = {
                        "schema": "bts_pick_automatic_reading_v1",
                        "recorded_at": "2027-07-14T22:35:00-04:00",
                        "age_hours": 20.0, "stale": True, "reason": "old",
                    }
                    cfg = {"bluesky": {"dm_recipient": "offline.test"}, "orchestrator": {"picks_dir": str(pdir)}}
                    with patch.object(alert, "send_dm", sent), redirect_stderr(err):
                        scheduler._warn_stale_contest_reading(rec, cfg, pdir)
                finally:
                    alert.log.removeHandler(h)
                    alert.log.setLevel(lvl)
                results["4c"] = {
                    "dm_calls": [c.args for c in sent.call_args_list],
                    "status_bytes": (sub / "health_state/contest_stale_reading_dm_delivery_status.json").read_text(),
                    "logs": records,
                    "stderr": err.getvalue(),
                }

                # 3. 6b-1 morning check (refusal CRITICAL + stale WARN)
                for mkind in ("refusal", "stale"):
                    sub = root / f"6b1_{mkind}"
                    pdir = sub / "picks"
                    pdir.mkdir(parents=True)
                    if mkind == "stale":
                        acct = pdir / "account_state"
                        acct.mkdir()
                        (acct / "season_contract.json").write_text(json.dumps({
                            "schema_version": "bts_season_contract_v1", "season": 2027,
                            "status": "automatic", "activation_id": "a" * 32,
                            "switched_at": PINNED_NOW.isoformat(), "revoked_at": None, "note": None,
                        }))
                        (acct / "contest_streak.json").write_text(json.dumps({
                            "schema_version": "bts_contest_streak_auto_v2", "season": 2027,
                            "activation_id": "a" * 32, "active_streak": 7, "best_streak": 12,
                            "source": "mlb_bts_profile", "source_date": DAY,
                            "recorded_at": "2027-07-14T10:00:00-04:00",
                        }))
                    sent = Mock(return_value="6b1-id")
                    records, h, lvl = capture_logs()
                    err = StringIO()
                    try:
                        cfg = {
                            "bluesky": {"dm_recipient": "offline.test"},
                            "orchestrator": {"picks_dir": str(pdir)},
                            "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                        }
                        with patch.object(alert, "send_dm", sent), redirect_stderr(err):
                            scheduler._morning_state_check(
                                cfg, pdir, DAY, PINNED_NOW.replace(hour=10, minute=0),
                                PINNED_NOW.replace(hour=16, minute=10),
                            )
                    finally:
                        alert.log.removeHandler(h)
                        alert.log.setLevel(lvl)
                    fname = (
                        "contest_morning_check_dm_delivery_status.json" if mkind == "refusal"
                        else "contest_stale_reading_dm_delivery_status.json"
                    )
                    results[f"6b1_{mkind}"] = {
                        "dm_calls": [c.args for c in sent.call_args_list],
                        "status_bytes": (sub / "health_state" / fname).read_text(),
                        "logs": records,
                        "stderr": err.getvalue().replace(str(pdir), "<pdir>"),
                    }

                # 4. Cutoff refusal, 5. Void refusal, 6. Missed pick (live & optional), 7. Contest state failure
                for caller in ("cutoff", "void", "missed_live", "missed_opt", "contest_fail"):
                    sub = root / caller
                    pdir = sub / "picks"
                    pdir.mkdir(parents=True)
                    p1 = picks.Pick("Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{DAY}T22:30:00Z")
                    dp = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None)
                    picks.save_pick(dp, pdir)
                    st = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                    cfg = {
                        "bluesky": {"dm_recipient": "offline.test"},
                        "orchestrator": {"picks_dir": str(pdir)},
                        "scheduler": {"contest_state_required": caller != "missed_opt", "pick_delivery": "dm"},
                    }
                    sent = Mock(return_value="caller-id")
                    records, h, lvl = capture_logs()
                    err = StringIO()
                    try:
                        with patch.object(alert, "send_dm", sent), redirect_stderr(err):
                            if caller == "cutoff":
                                scheduler._refuse_late_delivery(
                                    dp, cfg, pdir, st, DAY, "lineup",
                                    PINNED_NOW - timedelta(minutes=10), PINNED_NOW,
                                )
                            elif caller == "void":
                                scheduler._refuse_void_game_delivery(
                                    dp, cfg, pdir, st, DAY, "lineup", {101: "Postponed"},
                                )
                            elif caller.startswith("missed_"):
                                scheduler._alert_missed_pick(cfg, dp, 8.0)
                            else:
                                scheduler._alert_contest_state_failure(cfg, RuntimeError("refused state"))
                    finally:
                        alert.log.removeHandler(h)
                        alert.log.setLevel(lvl)
                    results[caller] = {
                        "dm_calls": [c.args for c in sent.call_args_list],
                        "status_bytes": (sub / "health_state/health_dm_delivery_status.json").read_text(),
                        "logs": records,
                        "stderr": err.getvalue(),
                    }

                # 8. Direct dispatcher cases: no_recipient, duplicate, exception + resurface retry, missing_id (None, "")
                for dcase in ("no_recipient", "duplicate", "exception_and_resurface", "missing_id_none", "missing_id_empty"):
                    sub = root / dcase
                    spath = sub / "status.json"
                    a = [alert.Alert("CRITICAL", "contest_state", f"case {dcase}")]
                    sent = Mock()
                    if dcase == "exception_and_resurface":
                        sent.side_effect = [OSError("net down"), "recovered-id"]
                    elif dcase == "missing_id_none":
                        sent.return_value = None
                    elif dcase == "missing_id_empty":
                        sent.return_value = ""
                    else:
                        sent.return_value = "ok-id"
                    records, h, lvl = capture_logs()
                    err = StringIO()
                    rets = []
                    snaps = []
                    try:
                        with patch.object(alert, "send_dm", sent), redirect_stderr(err):
                            recip = None if dcase == "no_recipient" else "offline.test"
                            rets.append(alert.dispatch_dm_for_health_alerts(
                                a, recip, status_path=spath, now_et_date=PINNED_NOW.date()
                            ))
                            snaps.append(spath.read_text())
                            if dcase in {"duplicate", "exception_and_resurface", "missing_id_none", "missing_id_empty"}:
                                rets.append(alert.dispatch_dm_for_health_alerts(
                                    a, recip, status_path=spath, now_et_date=PINNED_NOW.date()
                                ))
                                snaps.append(spath.read_text())
                    finally:
                        alert.log.removeHandler(h)
                        alert.log.setLevel(lvl)
                    results[dcase] = {
                        "returns": rets,
                        "dm_calls": [c.args for c in sent.call_args_list],
                        "status_snaps": snaps,
                        "logs": records,
                        "stderr": err.getvalue(),
                    }
    return results


def run_legacy_subprocess(src_dir: Path, mode: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(src_dir), PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), __file__, "--legacy-worker", mode],
        env=env, capture_output=True, text=True, timeout=25, check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    signal.alarm(60)
    try:
        # Part B first via clean subprocesses: base vs head-omitted vs head-explicit-False
        base_legacy = run_legacy_subprocess(BASE_SRC, "omitted")
        head_omitted = run_legacy_subprocess(HEAD_SRC, "omitted")
        head_false = run_legacy_subprocess(HEAD_SRC, "explicit_false")
        assert base_legacy == head_omitted == head_false, "Legacy caller mismatch between base and head!"

        # Part A: require_message_id=True acknowledgement matrix on both routes
        sys.path.insert(0, str(HEAD_SRC))
        from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.health import alert

        class PinnedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

        replies = [
            ("none", None, False),
            ("empty", "", False),
            ("space", " ", False),
            ("tabs_newlines", " \t\r\n ", False),
            ("zero_int", 0, False),
            ("pos_int", 17, False),
            ("bool_true", True, False),
            ("bool_false", False, False),
            ("list_id", ["ack"], False),
            ("dict_id", {"id": "ack"}, False),
            ("exception", RuntimeError("transport down"), False),
            ("clean_id", "msg-123", True),
            ("padded_id", "  msg-padded  ", True),
        ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            games = [{
                "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "H"}}},
            }]
            for route in ("direct", "classification"):
                for name, reply, is_ack in replies:
                    sub = root / f"{route}_{name}"
                    pdir = sub / "picks"
                    pdir.mkdir(parents=True)
                    own_path = sub / "health_state/delivery_outcome_unknown" / f"{DAY}.json"
                    # Seed a prior failed receipt with critical_count=1 to prove NO resurface banner on opt-in route!
                    own_path.parent.mkdir(parents=True)
                    own_path.write_text(json.dumps({"status": "failed", "critical_count": 1, "error": "seed"}))

                    p1 = picks.Pick("Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{DAY}T23:10:00Z")
                    dp = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None, delivery_attempted=True)
                    picks.save_pick(dp, pdir)
                    cfg = {
                        "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(sub / "hb")},
                        "bluesky": {"dm_recipient": "offline.test"},
                        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                                      "live_forward_capture_on_lock": False},
                        "health_checks": {"enabled": False},
                        "tiers": [],
                    }
                    sent = Mock(side_effect=[reply, "retry-ack"])
                    err = StringIO()
                    with patch.object(contest_state, "datetime", PinnedClock), \
                            patch.object(daily_decision, "datetime", PinnedClock), \
                            patch.object(alert, "datetime", PinnedClock), \
                            patch.object(scheduler, "_now_et", lambda: PINNED_NOW), \
                            patch.object(scheduler, "fetch_schedule", side_effect=lambda d, g=games: g if d == DAY else []), \
                            patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}), \
                            patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None), \
                            patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None), \
                            patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()), \
                            patch.object(scheduler, "notify_ready", Mock()), \
                            patch.object(scheduler, "notify_watchdog", Mock()), \
                            patch.object(scheduler, "run_result_polling", Mock(return_value="final")), \
                            patch.object(dm, "send_dm", Mock()), \
                            patch.object(alert, "send_dm", sent), \
                            patch.object(posting, "post_to_bluesky", Mock()), \
                            patch.object(strategy, "_mdp_cache", None), \
                            redirect_stderr(err):
                        if route == "direct":
                            st = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                            assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, "r1") is False
                        else:
                            scheduler.run_day(DAY, cfg)

                        rec1 = json.loads(own_path.read_text())
                        err1_lines = [ln for ln in err.getvalue().splitlines()
                                      if ln.startswith("Unknown-outcome alert failed: ")]
                        if is_ack:
                            assert rec1 == {
                                "schema_version": "bts_health_dm_delivery_status_v1",
                                "status": "sent",
                                "message_id": reply,
                                "sent_day": DAY,
                                "sent_sources": ["CRITICAL:late_delivery"],
                            }, rec1
                            assert err1_lines == []
                        else:
                            expected_err = "transport down" if isinstance(reply, Exception) else "send_dm returned no usable message id"
                            assert rec1 == {
                                "schema_version": "bts_health_dm_delivery_status_v1",
                                "status": "failed",
                                "error": expected_err,
                            }, rec1
                            assert len(err1_lines) == 1 and expected_err in err1_lines[0]

                        # Second encounter (and check what happens if dm_recipient removed AFTER ack)
                        if route == "direct":
                            st2 = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                            assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st2, DAY, "r2") is False
                        else:
                            scheduler.run_day(DAY, cfg)

                        bodies = [c.args[1] for c in sent.call_args_list]
                        assert bodies == [u7_copy()] * (1 if is_ack else 2), bodies

                        # Once acknowledged, removing dm_recipient still dedups without overwriting receipt
                        cfg_no_recip = json.loads(json.dumps(cfg))
                        cfg_no_recip["bluesky"].pop("dm_recipient")
                        before_no_recip = own_path.read_bytes()
                        st3 = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(dp, cfg_no_recip, pdir, st3, DAY, "r3") is False
                        assert own_path.read_bytes() == before_no_recip

                # Direct return boolean check on dispatch_dm_for_health_alerts(..., require_message_id=True)
                bpath = root / f"bool_{route}.json"
                al = [alert.Alert("CRITICAL", "late_delivery", "body")]
                with patch.object(alert, "send_dm", Mock(side_effect=[None, "ack-bool"])), redirect_stderr(StringIO()):
                    assert alert.dispatch_dm_for_health_alerts([], "offline.test", status_path=bpath,
                                                               now_et_date=PINNED_NOW.date(), require_message_id=True) is False
                    assert not bpath.exists()
                    assert alert.dispatch_dm_for_health_alerts(al, None, status_path=bpath,
                                                               now_et_date=PINNED_NOW.date(), require_message_id=True) is False
                    assert json.loads(bpath.read_text())["status"] == "skipped_no_recipient"
                    assert alert.dispatch_dm_for_health_alerts(al, "offline.test", status_path=bpath,
                                                               now_et_date=PINNED_NOW.date(), require_message_id=True) is True
                    assert json.loads(bpath.read_text())["status"] == "failed"
                    assert alert.dispatch_dm_for_health_alerts(al, "offline.test", status_path=bpath,
                                                               now_et_date=PINNED_NOW.date(), require_message_id=True) is True
                    assert json.loads(bpath.read_text())["status"] == "sent"
                    assert alert.dispatch_dm_for_health_alerts(al, "offline.test", status_path=bpath,
                                                               now_et_date=PINNED_NOW.date(), require_message_id=True) is False

        print("PROBE 3 OK: 13 reply shapes x 2 routes + no-recipient/dedup + 14 legacy caller parity checks vs base verified.")
    finally:
        if str(HEAD_SRC) in sys.path:
            sys.path.remove(str(HEAD_SRC))
        signal.alarm(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy-worker":
        print(json.dumps(_legacy_worker(sys.argv[2])))
    else:
        main()
