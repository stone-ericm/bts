"""Item 1 probe (ExitStack): real run_day restart paths, scoreable/terminal/write_none states, and base parity."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime
from io import StringIO
import json
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
SLICE1_CODEX = ROOT / "docs/reviews/2026-11-18-whole-range/slice-1-codex"
ET = ZoneInfo("America/New_York")
DAY = "2027-03-25"
PINNED_NOW = datetime(2027, 3, 25, 18, 35, 0, tzinfo=ET)


def expected_u7_copy(day: str) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def _worker_collect() -> dict:
    import pandas as pd
    import opening_edges as edges
    import opening_rehearsal as opening
    from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.daily_decision import decision_path, write_decision
    from bts.health import alert, post_failure

    opening.REPO = ROOT
    out: dict = {"restarts": {}, "classified": {}, "controls": {}}

    with TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        # 1. Real run_day restart paths via opening_edges + a second restart on the same slate
        for phase in ("before_remote_accept", "after_remote_accept", "corrected_after_marker"):
            phase_dir = tmp_root / f"rehearsal_{phase}"
            phase_dir.mkdir()
            season = opening.Season(phase_dir)
            opening.write_json(season.account_dir / "contest_streak.json", {
                "schema_version": "bts_contest_streak_auto_v1",
                "season": 2026,
                "active_streak": 17,
                "best_streak": 26,
                "saver_available": False,
                "source_date": opening.OLD_DAY,
                "recorded_at": "2026-10-21T14:00:00Z",
                "source": "manual_cli",
            })
            orig_finish = edges.finish
            second_restart_snap = {}

            def restart_twice_then_finish(s, d, outcome_exit, snap=second_restart_snap, fin=orig_finish):
                dec_file = decision_path(d, s.picks)
                before_second = dec_file.read_bytes()
                msgs_after_first = len([m for m in s.messages if "DELIVERY OUTCOME UNKNOWN" in m["body"]])
                s.command("schedule", "--config", s.config, "--date", d)
                after_second = dec_file.read_bytes()
                snap["decision_unchanged_on_second_restart"] = (before_second == after_second)
                snap["unknown_dms_after_first"] = msgs_after_first
                snap["unknown_dms_after_second"] = len(
                    [m for m in s.messages if "DELIVERY OUTCOME UNKNOWN" in m["body"]]
                )
                return fin(s, d, outcome_exit)

            with ExitStack() as st_rehearsal:
                st_rehearsal.enter_context(patch.object(edges, "finish", restart_twice_then_finish))
                st_rehearsal.enter_context(patch.object(strategy, "_mdp_cache", None))
                st_rehearsal.enter_context(season.run_offline())
                season.restore()
                ev = edges.restart_at_send(season, phase)
            day = ev["snapshot"]["day"]
            unknown_bodies = [m["body"] for m in season.messages if "DELIVERY OUTCOME UNKNOWN" in m["body"]]
            watchdog_bodies = [m["body"] for m in season.messages if "Delivery of this pick was NOT confirmed" in m["body"]]
            receipt_file = season.root / "data/health_state/delivery_outcome_unknown" / f"{day}.json"
            shared_file = season.root / "data/health_state/health_dm_delivery_status.json"
            out["restarts"][phase] = {
                "day": day,
                "picks_sent": len(edges.picks_sent(season)),
                "unknown_bodies": unknown_bodies,
                "watchdog_count": len(watchdog_bodies),
                "pick_locked": ev["snapshot"]["scheduler"]["pick_locked"],
                "committed_pick_written": ev["snapshot"]["scheduler"]["committed_pick_written"],
                "decision": ev["snapshot"]["decision"],
                "next_morning_state": ev["next_morning"]["state"],
                "second_restart": second_restart_snap,
                "own_receipt_exists": receipt_file.is_file(),
                "shared_status": json.loads(shared_file.read_text()) if shared_file.is_file() else None,
            }

        def make_env(case_name: str):
            base = tmp_root / case_name
            picks_dir = base / "picks"
            picks_dir.mkdir(parents=True)
            now_box = [PINNED_NOW]

            class PinnedClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

            games = [{
                "gamePk": 101,
                "gameDate": f"{DAY}T23:10:00Z",
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
            }]
            cfg = {
                "orchestrator": {
                    "picks_dir": str(picks_dir),
                    "heartbeat_path": str(base / "hb"),
                    "models_dir": str(base / "models"),
                },
                "bluesky": {"dm_recipient": "offline.test"},
                "scheduler": {
                    "contest_state_required": True,
                    "pick_delivery": "dm",
                    "live_forward_capture_on_lock": False,
                },
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            p1 = picks.Pick(
                "Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                0.84, [], False, 101, f"{DAY}T23:10:00Z",
            )
            daily = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None)
            return base, picks_dir, now_box, PinnedClock, games, cfg, daily

        def apply_patches(stack: ExitStack, PinnedClock, now_box, games, statuses, pick_dm, health_dm, poll_mock, sched_fn=None):
            for p in (
                patch.object(contest_state, "datetime", PinnedClock),
                patch.object(daily_decision, "datetime", PinnedClock),
                patch.object(alert, "datetime", PinnedClock),
                patch.object(scheduler, "_now_et", lambda: now_box[0]),
                patch.object(scheduler, "fetch_schedule", side_effect=sched_fn or (lambda d, g=games: g if d == DAY else [])),
                patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
                patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
                patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
                patch.object(scheduler, "notify_ready", Mock()),
                patch.object(scheduler, "notify_watchdog", Mock()),
                patch.object(scheduler, "run_result_polling", poll_mock),
                patch.object(picks, "get_game_statuses_detailed", lambda *_, s=statuses: s),
                patch.object(strategy, "get_game_statuses_detailed", lambda *_, s=statuses: s),
                patch.object(picks, "get_game_statuses", lambda *_, s=statuses: {101: s[101]["abstract"]}),
                patch.object(dm, "send_dm", pick_dm),
                patch.object(alert, "send_dm", health_dm),
                patch.object(posting, "post_to_bluesky", Mock(return_value="at://1")),
                patch.object(strategy, "_mdp_cache", None),
                redirect_stderr(StringIO()),
            ):
                stack.enter_context(p)

        # 2. Classified restart variations
        for variant in (
            "fresh_unconfirmed",
            "prior_locked_unconfirmed",
            "prior_delivered_null_id",
            "prior_private_locked_carried",
            "prior_unknown_label",
            "terminal_hit",
            "terminal_miss",
            "terminal_void",
            "write_decision_none",
        ):
            base, picks_dir, now_box, PinnedClock, games, cfg, daily = make_env(f"cls_{variant}")
            statuses = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
            pick_dm = Mock(return_value=None if variant == "prior_delivered_null_id" else "pick-id")
            health_dm = Mock(return_value="alert-ack")
            poll_mock = Mock(return_value="final")

            with ExitStack() as stack:
                apply_patches(stack, PinnedClock, now_box, games, statuses, pick_dm, health_dm, poll_mock)
                if variant == "prior_delivered_null_id":
                    with patch.object(scheduler, "_contest_state_required", lambda _: False), \
                            patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=5)):
                        st0 = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                        picks.save_pick(daily, picks_dir)
                        assert scheduler._deliver_and_lock_pick(daily, cfg, picks_dir, st0, DAY, "initial") is True
                        assert pick_dm.call_count == 1
                        pick_dm.reset_mock()
                        health_dm.reset_mock()
                else:
                    daily.delivery_attempted = True
                    if variant.startswith("terminal_"):
                        daily.result = variant.removeprefix("terminal_")
                    picks.save_pick(daily, picks_dir)
                    if variant in {"prior_locked_unconfirmed", "prior_private_locked_carried", "prior_unknown_label"}:
                        lbl = {
                            "prior_locked_unconfirmed": "locked_unconfirmed",
                            "prior_private_locked_carried": "private_locked",
                            "prior_unknown_label": "custom_label",
                        }[variant]
                        write_decision(
                            DAY, picks_dir, action="single", source="mdp",
                            primary={"batter_id": 680757, "batter_name": "Steven Kwan",
                                     "team": "CLE", "game_pk": 101, "p_game_hit": 0.84},
                            delivery_status=lbl, scoreable=True, now=PINNED_NOW,
                        )

                dec_file = decision_path(DAY, picks_dir)
                prior_dec_bytes = dec_file.read_bytes().decode() if dec_file.exists() else None
                pick_before = (picks_dir / f"{DAY}.json").read_bytes().decode()

                if variant == "write_decision_none":
                    stack.enter_context(patch.object(daily_decision, "write_decision", Mock(return_value=None)))
                ret1 = scheduler.run_day(DAY, cfg)
                ret2 = scheduler.run_day(DAY, cfg)

                st = scheduler.load_state(DAY, picks_dir)
                dec_after_bytes = dec_file.read_bytes().decode() if dec_file.exists() else None
                pick_after = (picks_dir / f"{DAY}.json").read_bytes().decode()
                eod_alerts = post_failure.check(picks_dir, today=PINNED_NOW.date(), now=PINNED_NOW.replace(hour=22))
                bodies = [c.args[1] for c in health_dm.call_args_list]
                out["classified"][variant] = {
                    "returns": [ret1, ret2],
                    "pick_dm_calls": pick_dm.call_count,
                    "unknown_dms": [b for b in bodies if "DELIVERY OUTCOME UNKNOWN" in b],
                    "missed_pick_dms": len([b for b in bodies if "[missed_pick]" in b]),
                    "pick_locked": st.pick_locked,
                    "committed_pick_written": st.committed_pick_written,
                    "result_status": st.result_status,
                    "poll_calls": poll_mock.call_count,
                    "prior_dec_preserved": (dec_after_bytes == prior_dec_bytes) if prior_dec_bytes is not None else None,
                    "dec_after": json.loads(dec_after_bytes) if dec_after_bytes else None,
                    "pick_bytes_unchanged": (pick_before == pick_after),
                    "eod_pick_delivery_fires": any(a.source == "pick_delivery" for a in eod_alerts),
                }

        # 3. Controls
        for ctrl in (
            "confirmed_dm", "confirmed_public", "fresh_private", "stale_preview_game_live",
            "pre_attempt", "dry_run", "off_day", "schedule_error",
        ):
            base, picks_dir, now_box, PinnedClock, games, cfg, daily = make_env(f"ctrl_{ctrl}")
            statuses = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
            if ctrl == "stale_preview_game_live":
                statuses[101] = {"abstract": "L", "detailed": "In Progress", "code": "I"}
            health_dm = Mock(return_value="alert-ack")
            pick_dm = Mock(return_value="pick-ack")
            sched_fn = (
                OSError("sched down") if ctrl == "schedule_error"
                else (lambda d: []) if ctrl == "off_day"
                else (lambda d, g=games: g if d == DAY else [])
            )
            with ExitStack() as stack:
                apply_patches(stack, PinnedClock, now_box, games, statuses, pick_dm, health_dm, Mock(return_value="final"), sched_fn)
                stack.enter_context(patch.object(orchestrator, "run_cascade", Mock(return_value=(None, None))))
                stack.enter_context(patch.object(scheduler, "_contest_state_required", lambda _: False))
                stack.enter_context(patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=5)))
                if ctrl in {"confirmed_dm", "confirmed_public", "fresh_private"}:
                    cfg["scheduler"]["pick_delivery"] = {
                        "confirmed_dm": "dm", "confirmed_public": "public", "fresh_private": "private",
                    }[ctrl]
                    picks.save_pick(daily, picks_dir)
                    st0 = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(daily, cfg, picks_dir, st0, DAY, ctrl) is True
                    scheduler.run_day(DAY, cfg)
                else:
                    if ctrl in {"dry_run", "off_day", "schedule_error"}:
                        daily.delivery_attempted = True
                    picks.save_pick(daily, picks_dir)
                    if ctrl == "schedule_error":
                        try:
                            scheduler.run_day(DAY, cfg)
                        except OSError:
                            pass
                    else:
                        scheduler.run_day(DAY, cfg, dry_run=(ctrl == "dry_run"))
                bodies = [c.args[1] for c in health_dm.call_args_list]
                receipt = base / "health_state/delivery_outcome_unknown" / f"{DAY}.json"
                out["controls"][ctrl] = {
                    "unknown_dms": [b for b in bodies if "DELIVERY OUTCOME UNKNOWN" in b],
                    "receipt_exists": receipt.exists(),
                }
    return out


def run_child(src_dir: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src_dir}:{SLICE1_CODEX}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "America/New_York"
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), __file__, "--worker"],
        env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    signal.alarm(60)
    try:
        base_res = run_child(BASE_SRC)
        head_res = run_child(HEAD_SRC)

        for phase, expected_picks in (
            ("before_remote_accept", 0),
            ("after_remote_accept", 1),
            ("corrected_after_marker", 0),
        ):
            b = base_res["restarts"][phase]
            h = head_res["restarts"][phase]
            day = h["day"]
            assert b["unknown_bodies"] == [], f"base unexpectedly sent DM on {phase}"
            assert h["unknown_bodies"] == [expected_u7_copy(day)], h["unknown_bodies"]
            assert h["second_restart"] == {
                "decision_unchanged_on_second_restart": True,
                "unknown_dms_after_first": 1,
                "unknown_dms_after_second": 1,
            }, h["second_restart"]
            assert h["picks_sent"] == b["picks_sent"] == expected_picks
            assert h["pick_locked"] is b["pick_locked"] is True
            assert h["committed_pick_written"] is b["committed_pick_written"] is True
            assert h["decision"] == b["decision"]
            assert h["next_morning_state"] == b["next_morning_state"] == "not_entered"
            assert h["watchdog_count"] == b["watchdog_count"] == 1
            assert h["own_receipt_exists"] is True and b["own_receipt_exists"] is False
            assert h["shared_status"] == b["shared_status"]

        for variant, h in head_res["classified"].items():
            b = base_res["classified"][variant]
            assert b["unknown_dms"] == []
            assert h["unknown_dms"] == [expected_u7_copy(DAY)]
            assert h["pick_dm_calls"] == b["pick_dm_calls"] == 0
            assert h["returns"] == b["returns"] == [None, None]
            assert h["pick_locked"] == b["pick_locked"] is True
            assert h["committed_pick_written"] == b["committed_pick_written"] == (variant != "write_decision_none")
            assert h["missed_pick_dms"] == b["missed_pick_dms"] == (2 if variant == "write_decision_none" else 0)
            expected_polls = 0 if (variant.startswith("terminal_") or variant == "write_decision_none") else 2
            assert h["poll_calls"] == b["poll_calls"] == expected_polls
            assert h["result_status"] == b["result_status"]
            assert h["prior_dec_preserved"] == b["prior_dec_preserved"]
            assert h["dec_after"] == b["dec_after"]
            assert h["pick_bytes_unchanged"] is b["pick_bytes_unchanged"] is True
            assert h["eod_pick_delivery_fires"] is b["eod_pick_delivery_fires"] is True

        for ctrl, h in head_res["controls"].items():
            b = base_res["controls"][ctrl]
            assert h == b == {"unknown_dms": [], "receipt_exists": False}, (ctrl, h, b)

        print("PROBE 1B OK: 3 real restart phases (x2 restarts), 9 classified variants, 8 controls verified vs base.")
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(_worker_collect()))
    else:
        main()
