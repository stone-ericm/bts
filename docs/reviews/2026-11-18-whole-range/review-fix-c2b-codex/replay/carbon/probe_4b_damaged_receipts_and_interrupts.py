"""Item 4 probe (ExitStack): special/damaged receipts, TOCTOU races, error bounding, and interrupts."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HEAD_SRC = ROOT / "src"
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
PINNED_NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
PREFIX = "Unknown-outcome alert failed: "


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


def main() -> None:
    def on_alarm(*_):
        raise BaseException("probe_4b timed out on blocking I/O")

    prev_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(30)
    chmod_restore: list[tuple[Path, int]] = []
    sys.path.insert(0, str(HEAD_SRC))
    try:
        from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.daily_decision import load_decision
        from bts.health import alert

        class PinnedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

        games = [{
            "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "H"}}},
        }]

        def enter_common(stack: ExitStack, sent_mock, pick_dm_mock, err_buf):
            for p in (
                patch.object(contest_state, "datetime", PinnedClock),
                patch.object(daily_decision, "datetime", PinnedClock),
                patch.object(alert, "datetime", PinnedClock),
                patch.object(scheduler, "_now_et", lambda: PINNED_NOW),
                patch.object(scheduler, "fetch_schedule", side_effect=lambda d, g=games: g if d == DAY else []),
                patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
                patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
                patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
                patch.object(scheduler, "notify_ready", Mock()),
                patch.object(scheduler, "notify_watchdog", Mock()),
                patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
                patch.object(dm, "send_dm", pick_dm_mock),
                patch.object(alert, "send_dm", sent_mock),
                patch.object(posting, "post_to_bluesky", Mock()),
                patch.object(strategy, "_mdp_cache", None),
                redirect_stderr(err_buf),
            ):
                stack.enter_context(p)

        injuries = [
            "truncated", "empty", "array", "null", "int_json", "invalid_utf8", "deep_nesting",
            "bad_sent_null_id", "bad_sent_blank_id", "bad_sent_int_id", "bad_sent_wrong_day", "bad_sent_wrong_sources",
            "directory", "nonempty_directory", "fifo", "unreadable_file",
            "symlink_to_shared_sent", "symlink_to_shared_failed", "dangling_symlink", "hardlink_to_shared",
            "unwritable_parent_missing", "unwritable_parent_malformed", "unwritable_parent_valid_sent",
            "unsearchable_parent", "blocked_parent_file", "blocked_grandparent_file",
            "parent_symlink_to_shared_dir", "parent_dangling_symlink",
            "race_symlink_during_send", "race_hardlink_during_send", "race_fifo_during_send",
            "both_storage_and_transport_fail",
        ]
        readable_malformed = {
            "truncated", "empty", "array", "null", "int_json", "invalid_utf8", "deep_nesting",
            "bad_sent_null_id", "bad_sent_blank_id", "bad_sent_int_id", "bad_sent_wrong_day", "bad_sent_wrong_sources",
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                for route in ("direct", "classification"):
                    for injury in injuries:
                        sub = root / f"{route}_{injury}"
                        pdir = sub / "picks"
                        pdir.mkdir(parents=True)
                        health_dir = sub / "health_state"
                        shared = health_dir / "health_dm_delivery_status.json"
                        own_dir = health_dir / "delivery_outcome_unknown"
                        own = own_dir / f"{DAY}.json"

                        valid_sent = json.dumps({
                            "schema_version": "bts_health_dm_delivery_status_v1",
                            "status": "sent", "sent_day": DAY,
                            "sent_sources": ["CRITICAL:late_delivery"], "message_id": "shared-old",
                        }).encode()
                        failed_banner_seed = json.dumps({
                            "schema_version": "bts_health_dm_delivery_status_v1",
                            "status": "failed", "critical_count": 1, "error": "shared-fail",
                        }).encode()

                        if injury == "blocked_grandparent_file":
                            health_dir.write_bytes(b"not-a-dir")
                            shared_expected = None
                        else:
                            health_dir.mkdir(parents=True)
                            shared_seed = failed_banner_seed if injury == "symlink_to_shared_failed" else valid_sent
                            shared.write_bytes(shared_seed)
                            shared_expected = shared_seed

                        if injury == "blocked_parent_file":
                            own_dir.write_bytes(b"blocked")
                        elif injury == "parent_symlink_to_shared_dir":
                            own_dir.symlink_to(health_dir, target_is_directory=True)
                        elif injury == "parent_dangling_symlink":
                            own_dir.symlink_to(sub / "does_not_exist_dir")
                        elif injury != "blocked_grandparent_file":
                            own_dir.mkdir()
                            if injury == "directory":
                                own.mkdir()
                            elif injury == "nonempty_directory":
                                own.mkdir()
                                (own / "inside.txt").write_text("keep")
                            elif injury == "fifo":
                                os.mkfifo(own)
                            elif injury == "unreadable_file":
                                own.write_bytes(valid_sent)
                                own.chmod(0)
                                chmod_restore.append((own, 0o600))
                            elif injury in {"symlink_to_shared_sent", "symlink_to_shared_failed"}:
                                own.symlink_to(shared)
                            elif injury == "dangling_symlink":
                                own.symlink_to(sub / "missing_target.json")
                            elif injury == "hardlink_to_shared":
                                os.link(shared, own)
                            elif injury == "unwritable_parent_missing":
                                own_dir.chmod(0o500)
                                chmod_restore.append((own_dir, 0o700))
                            elif injury == "unwritable_parent_malformed":
                                own.write_bytes(b"[]")
                                own_dir.chmod(0o500)
                                chmod_restore.append((own_dir, 0o700))
                            elif injury == "unwritable_parent_valid_sent":
                                own.write_bytes(valid_sent)
                                own_dir.chmod(0o500)
                                chmod_restore.append((own_dir, 0o700))
                            elif injury == "unsearchable_parent":
                                own_dir.chmod(0o000)
                                chmod_restore.append((own_dir, 0o700))
                            elif injury == "both_storage_and_transport_fail":
                                own.symlink_to(shared)
                            elif injury in readable_malformed:
                                payloads = {
                                    "truncated": b"{",
                                    "empty": b"",
                                    "array": b"[]",
                                    "null": b"null",
                                    "int_json": b"123",
                                    "invalid_utf8": b"\xff\xfe\x80",
                                    "deep_nesting": b"[" * 4000 + b"]" * 4000,
                                    "bad_sent_null_id": valid_sent.replace(b'"shared-old"', b"null"),
                                    "bad_sent_blank_id": valid_sent.replace(b'"shared-old"', b'"  "'),
                                    "bad_sent_int_id": valid_sent.replace(b'"shared-old"', b"99"),
                                    "bad_sent_wrong_day": valid_sent.replace(DAY.encode(), b"2027-07-14"),
                                    "bad_sent_wrong_sources": valid_sent.replace(b"late_delivery", b"other"),
                                }
                                own.write_bytes(payloads[injury])

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

                        def send_fn(recipient, body, inj=injury, own_p=own, sh_p=shared):
                            if inj == "race_symlink_during_send":
                                own_p.symlink_to(sh_p)
                            elif inj == "race_hardlink_during_send":
                                os.link(sh_p, own_p)
                            elif inj == "race_fifo_during_send":
                                os.mkfifo(own_p)
                            elif inj == "both_storage_and_transport_fail":
                                raise RuntimeError("lineA\nlineB\r" + "q" * 300)
                            return "fresh-ack"

                        sent = Mock(side_effect=send_fn)
                        err = StringIO()
                        t0 = time.monotonic()
                        with ExitStack() as stack:
                            enter_common(stack, sent, Mock(), err)
                            if route == "direct":
                                st = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                                assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, injury) is False
                            else:
                                scheduler.run_day(DAY, cfg)
                        elapsed = time.monotonic() - t0
                        assert elapsed < 2.0, f"{injury} took {elapsed:.2f}s"

                        assert scheduler.load_state(DAY, pdir).pick_locked is True
                        assert load_decision(DAY, pdir)["delivery_status"] == "locked_unconfirmed"

                        if shared_expected is not None:
                            assert shared.read_bytes() == shared_expected, f"shared touched on {injury}"
                        if injury == "parent_symlink_to_shared_dir":
                            assert not (health_dir / f"{DAY}.json").exists(), "wrote through symlinked parent!"
                            assert own_dir.is_symlink()

                        err_lines = [ln for ln in err.getvalue().splitlines() if ln.startswith(PREFIX)]
                        if injury == "unwritable_parent_valid_sent":
                            assert sent.call_count == 0
                            assert err_lines == []
                        elif injury in readable_malformed:
                            assert sent.call_count == 1 and sent.call_args.args == ("offline.test", u7_copy())
                            assert json.loads(own.read_bytes())["message_id"] == "fresh-ack"
                            assert err_lines == []
                        else:
                            assert sent.call_count == 1 and sent.call_args.args == ("offline.test", u7_copy())
                            assert len(err_lines) == 1, (injury, err_lines)
                            detail = err_lines[0].removeprefix(PREFIX)
                            assert len(detail) <= 200 and "\n" not in detail and "\r" not in detail
                            if injury in {"directory", "nonempty_directory"}:
                                assert own.is_dir()
                            elif injury in {"fifo", "race_fifo_during_send"}:
                                assert stat.S_ISFIFO(own.lstat().st_mode)
                            elif injury in {"symlink_to_shared_sent", "symlink_to_shared_failed",
                                            "dangling_symlink", "race_symlink_during_send"}:
                                assert own.is_symlink()
                            elif injury in {"hardlink_to_shared", "race_hardlink_during_send"}:
                                assert own.lstat().st_nlink == 2
                            if own_dir.is_dir() and not own_dir.is_symlink() and (own_dir.stat().st_mode & 0o500) == 0o500:
                                leftovers = [p.name for p in own_dir.iterdir() if p.name.startswith(f".{DAY}.json.")]
                                assert leftovers == [], (injury, leftovers)

                # Interrupts across 3 windows on both routes
                for route in ("direct", "classification"):
                    for intr in (KeyboardInterrupt, SystemExit):
                        for window in ("after_accept_before_write", "during_fsync_before_replace", "after_durable_receipt"):
                            sub = root / f"intr_{route}_{intr.__name__}_{window}"
                            pdir = sub / "picks"
                            pdir.mkdir(parents=True)
                            own_dir = sub / "health_state/delivery_outcome_unknown"
                            own = own_dir / f"{DAY}.json"
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
                            remote_wire_sends = []

                            def wire_send(recipient, body, rw=remote_wire_sends, w=window, exc_cls=intr, pd=pdir):
                                assert scheduler.load_state(DAY, pd).pick_locked is True
                                assert load_decision(DAY, pd)["delivery_status"] == "locked_unconfirmed"
                                rw.append(body)
                                if w == "after_accept_before_write" and len(rw) == 1:
                                    raise exc_cls("cut after remote accept")
                                return f"ack-{len(rw)}"

                            orig_fsync = os.fsync
                            orig_dispatch = alert.dispatch_dm_for_health_alerts

                            def fsync_hook(fd, w=window, exc_cls=intr, rw=remote_wire_sends, od=own_dir):
                                orig_fsync(fd)
                                if w == "during_fsync_before_replace" and len(rw) == 1 and od.exists():
                                    if list(od.glob(f".{DAY}.json.*")):
                                        raise exc_cls("cut during fsync")

                            def after_durable_hook(*a, w=window, exc_cls=intr, **kw):
                                res = orig_dispatch(*a, **kw)
                                if w == "after_durable_receipt":
                                    raise exc_cls("cut after durable receipt")
                                return res

                            pick_dm = Mock()
                            with ExitStack() as stack:
                                enter_common(stack, Mock(side_effect=wire_send), pick_dm, StringIO())
                                caught = None
                                with patch.object(os, "fsync", fsync_hook), \
                                        (patch.object(alert, "dispatch_dm_for_health_alerts", after_durable_hook)
                                         if window == "after_durable_receipt" else nullcontext()):
                                    try:
                                        if route == "direct":
                                            st = scheduler.SchedulerState(
                                                DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None
                                            )
                                            scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, DAY, "intr")
                                        else:
                                            scheduler.run_day(DAY, cfg)
                                    except BaseException as exc:
                                        caught = exc
                                assert isinstance(caught, intr), (route, intr, window, caught)
                                assert scheduler.load_state(DAY, pdir).pick_locked is True
                                assert load_decision(DAY, pdir)["delivery_status"] == "locked_unconfirmed"
                                if own_dir.exists():
                                    assert list(own_dir.glob(f".{DAY}.json.*")) == []

                                if window == "after_durable_receipt":
                                    assert own.is_file() and json.loads(own.read_bytes())["message_id"] == "ack-1"
                                    scheduler.run_day(DAY, cfg)
                                    scheduler.run_day(DAY, cfg)
                                    assert remote_wire_sends == [u7_copy()]
                                else:
                                    assert not own.exists()
                                    scheduler.run_day(DAY, cfg)
                                    assert own.is_file() and json.loads(own.read_bytes())["message_id"] == "ack-2"
                                    scheduler.run_day(DAY, cfg)
                                    assert remote_wire_sends == [u7_copy(), u7_copy()]
                                assert pick_dm.call_count == 0
            finally:
                for p, mode in reversed(chmod_restore):
                    try:
                        p.chmod(mode)
                    except OSError:
                        pass

        print("PROBE 4B OK: 32 damaged/special/race receipt cases x 2 routes + 12 interrupt windows verified.")
    finally:
        if str(HEAD_SRC) in sys.path:
            sys.path.remove(str(HEAD_SRC))
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


if __name__ == "__main__":
    main()
