import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import ExitStack
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler


DATE = "2027-06-12"
NOW = datetime(2027, 6, 12, 10, tzinfo=scheduler.ET)
GAME = {
    "gamePk": 101, "gameDate": DATE + "T23:00:00Z",
    "teams": {"away": {"team": {"name": "North"}}, "home": {"team": {"name": "South"}}},
}


def state():
    return scheduler.SchedulerState(DATE, NOW.isoformat(), [], [], [], True, NOW.isoformat(),
                                    "final", None, committed_pick_written=True)


def config(root, mode):
    return {
        "orchestrator": {"picks_dir": root / "picks", "heartbeat_path": root / "heartbeat"},
        "scheduler": {"pick_delivery": mode, "contest_state_required": True,
                      "live_forward_capture_on_lock": False},
        "bluesky": {"dm_recipient": "synthetic.invalid"},
        "health_checks": {"enabled": False},
    }


def seed_pick(picks_dir, mode, committed):
    pick = picks.Pick("Example Batter", 1, "NTH", 1, "Example Pitcher", 2, .8, [],
                      False, GAME["gamePk"], GAME["gameDate"])
    daily = picks.DailyPick(DATE, NOW.isoformat(), pick, None, None, result="hit",
                            bluesky_posted=mode == "public", notification_sent=mode == "dm",
                            notification_id="synthetic-dm" if mode == "dm" else None,
                            bluesky_uri="at://synthetic/post" if mode == "public" else None,
                            delivery_attempted=True)
    pick_path = picks.save_pick(daily, picks_dir)
    decision_path = daily_decision.decision_path(DATE, picks_dir)
    if committed:
        assert daily_decision.write_decision(
            DATE, picks_dir, action="single", source="mdp",
            primary={"batter_id": 1, "batter_name": "Example Batter", "game_pk": 101},
            streak=12, saver_available=True, scoreable=True, delivery_status="delivered", now=NOW,
        )
    return daily, pick_path, decision_path


def patch_external(stack):
    mocks = []
    for module, name in [
        (dm, "send_dm"), (posting, "post_to_bluesky"), (orchestrator, "run_and_pick"),
        (picks, "get_game_statuses_detailed"), (contest_state, "load_decision_streak_state"),
    ]:
        mocks.append(stack.enter_context(patch.object(module, name, side_effect=AssertionError(name))))
    stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("network")))
    return mocks


def run_restart(root, compatible_json, mode, *, future_field, committed, deployed):
    cfg = config(root, mode)
    picks_dir = cfg["orchestrator"]["picks_dir"]
    _, pick_path, decision_path = seed_pick(picks_dir, mode, committed)
    original_pick = pick_path.read_bytes()
    original_decision = decision_path.read_bytes() if committed else None
    state_path = picks_dir / DATE / "scheduler_state.json"
    data = json.loads(compatible_json)
    if future_field:
        data["synthetic_future_field"] = True
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data))
    assert {field.name for field in fields(scheduler.SchedulerState)} == set(json.loads(compatible_json))

    clock = NOW
    snapshots = []
    real_check = scheduler.run_single_check

    def advance(seconds):
        nonlocal clock
        clock += timedelta(seconds=seconds)

    def observe_check(*args, **kwargs):
        snapshots.append(scheduler.load_state(DATE, picks_dir).pick_locked)
        return real_check(*args, **kwargs)

    with ExitStack() as stack:
        external = patch_external(stack)
        for name, replacement in {
            "_now_et": lambda: clock, "fetch_schedule": lambda *_: [GAME],
            "_watchdog_ping_sleep": advance, "notify_ready": lambda: None,
            "notify_watchdog": lambda: None, "count_new_confirmations": lambda *_: 1,
            "_next_day_wakeup": lambda *_: NOW + timedelta(days=1),
            "_idle_until_next_wakeup": lambda *_: None, "run_single_check": observe_check,
        }.items():
            stack.enter_context(patch.object(scheduler, name, replacement))
        morning = None if deployed else stack.enter_context(
            patch.object(scheduler, "_morning_state_check", wraps=scheduler._morning_state_check))
        scheduler.run_day(DATE, cfg)
        if morning is not None:
            assert morning.call_count == int(future_field)
        for mock in external:
            mock.assert_not_called()
    assert snapshots == [False], snapshots
    assert bool(list(state_path.parent.glob("scheduler_state.json.corrupt-*"))) == future_field
    assert scheduler.load_state(DATE, picks_dir).pick_locked
    assert pick_path.read_bytes() == original_pick
    decision = daily_decision.load_decision(DATE, picks_dir)
    assert decision["scoreable"] and decision["primary"]["batter_id"] == 1
    if committed:
        assert decision_path.read_bytes() == original_decision
    else:
        assert decision["delivery_status"] == "delivered"
    print(f"restart {mode}: future={future_field}, quarantine={future_field}, startup lock false, "
          f"pick bytes unchanged, decision {'preserved' if committed else 'recovered'}")


def delivery_chokepoint(root, mode):
    cfg = config(root, mode)
    picks_dir = cfg["orchestrator"]["picks_dir"]
    daily, _, decision_path = seed_pick(picks_dir, mode, committed=True)
    decision_before = decision_path.read_bytes()
    fresh = state()
    fresh.pick_locked = False
    with ExitStack() as stack:
        external = patch_external(stack)
        if hasattr(picks, "saved_pick_refusal"):
            stack.enter_context(patch.object(picks, "saved_pick_refusal",
                                             side_effect=AssertionError("5a ran")))
        stack.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock"))
        assert scheduler._deliver_and_lock_pick(daily, cfg, picks_dir, fresh, DATE, "probe")
        for mock in external:
            mock.assert_not_called()
    assert fresh.pick_locked
    assert decision_path.read_bytes() != decision_before
    assert daily_decision.load_decision(DATE, picks_dir)["streak"] is None
    print(f"direct {mode}: locks with no transport or state read, rewrites commit with less provenance")


def skip_notice(root, compatible_json):
    cfg = config(root, "dm")
    picks_dir = cfg["orchestrator"]["picks_dir"]
    previous = state()
    previous.skip_notified_at = NOW.isoformat()
    path = scheduler.save_state(previous, picks_dir)
    summary = {"best_batter": "Example Batter", "best_team": "NTH", "best_p": .7, "streak": 12}
    for unknown in (False, True):
        data = json.loads(compatible_json) | {"skip_notified_at": NOW.isoformat()}
        if unknown:
            data["synthetic_future_field"] = True
        path.write_text(json.dumps(data))
        current = state()
        current.skip_notified_at = None
        scheduler.carry_forward_skip_state(current, scheduler.load_state(DATE, picks_dir))
        sends = []
        notified = scheduler.maybe_notify_skip(current, summary, cfg, now_iso=NOW.isoformat(),
                                                send=lambda *args: sends.append(args))
        assert notified == unknown and len(sends) == int(unknown)
    print("skip notice: same-day compatible reload suppresses duplicate, quarantine allows it")


def exercise(root, compatible_json, deployed):
    print("scheduler:", scheduler.__file__)
    for mode in ("dm", "public"):
        for future_field, committed in ((False, True), (True, True), (True, False)):
            run_restart(root / f"{mode}-{future_field}-{committed}", compatible_json, mode,
                        future_field=future_field, committed=committed, deployed=deployed)
        delivery_chokepoint(root / f"direct-{mode}", mode)
    skip_notice(root / "skip-notice", compatible_json)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        if len(sys.argv) > 1:
            exercise(root, Path(sys.argv[1]).read_text(), deployed=True)
        else:
            fixture = scheduler.save_state(state(), root / "head-fixture")
            exercise(root / "head", fixture.read_text(), deployed=False)
            archive = subprocess.check_output(["git", "archive", "619abda7", "src"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root / "deployed", filter="data")
            sys.stdout.flush()
            subprocess.run([sys.executable, __file__, str(fixture)], check=True,
                           env=os.environ | {"PYTHONPATH": str(root / "deployed" / "src")})
