"""Run each package through the same summary, lineup, alert and state inputs."""
import contextlib
import dataclasses
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pytest import MonkeyPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

STRIP = {"best_batter", "best_team", "best_p", "basis", "candidate_verified"}


def protect_socket(*args, **kwargs):
    raise AssertionError("network is forbidden")


def state_fixture(scheduler, day, summary=None):
    return scheduler.SchedulerState(
        date=day, schedule_fetched_at=None, games=[], confirmed_game_pks=[], runs_completed=[],
        pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None, skip_summary=summary,
    )


def summaries(scheduler, current):
    import pandas as pd

    class BadFrame:
        @property
        def columns(self):
            raise ValueError("columns")

    class BadCandidate(dict):
        def get(self, key, default=None):
            if key == "team":
                raise ValueError(key)
            return super().get(key, default)

    frame = pd.DataFrame([dict(batter_name="High", team="A", p_game_hit=.99),
                          dict(batter_name="Low", team="B", p_game_hit=.75)])
    frames = [
        frame, None, BadFrame(), [], {}, pd.DataFrame(), frame.iloc[:0],
        frame.drop(columns="team"), frame.drop(columns="batter_name"), frame.drop(columns="p_game_hit"),
        frame.set_axis(["p_game_hit", "team", "p_game_hit"], axis=1),
        pd.concat([frame, frame]), frame.assign(p_game_hit=pd.Series([pd.NA, pd.NA], dtype="Float64")),
    ]
    frames.extend(frame.assign(p_game_hit=[bad, bad]) for bad in (math.nan, math.inf, -math.inf, None, "0.75"))
    frames.append(frame.assign(p_game_hit=[[], {}]))
    candidates = (None, {}, {"p_game_hit": "0.75"}, {"p_game_hit": True}, {"p_game_hit": math.nan},
                  {"p_game_hit": 10**400}, BadCandidate(p_game_hit=.75), {"p_game_hit": .75},
                  {"p_game_hit": .75, "batter_name": "Low", "team": "B"},
                  {"p_game_hit": .75, "batter_name": pd.NA, "team": []})
    results = []
    for predictions in frames:
        for dec in (None, SimpleNamespace(reason="Tail stop", objective="tail")):
            reference = scheduler.build_skip_summary(predictions, 8, .8, dec)
            for index, candidate in enumerate(candidates):
                if current:
                    result = scheduler.build_skip_summary(predictions, 8, .8, dec, candidate)
                    assert (result is None) == (reference is None)
                    if result is not None:
                        verified = dec is not None and index >= 7
                        assert result["candidate_verified"] is verified
                        if verified:
                            assert (result["best_batter"], result["best_team"], result["best_p"]) == (
                                ("Low", "B", .75) if index == 8 else ("?", "?", .75)
                            )
                else:
                    result = reference
                results.append({k: v for k, v in result.items() if k not in STRIP} if result else result)
    return results


def checks(scheduler, root, mp):
    from bts import contest_state, orchestrator, picks, strategy
    from scripts.audit.season_walk import walk

    results = {}
    for case, day, streak, best, p, detail in [
        ("removed_top", "2027-07-15", 10, 10, .75, "removed"),
        ("none_eligible", "2027-07-15", 10, 10, .75, "final"),
        ("status_failure", "2027-07-15", 10, 10, .75, "raises"),
        ("heuristic", "2027-07-15", 10, 10, .75, "heuristic"),
        ("terminal_tail", "2027-09-19", 0, 18, .72, ""),
        ("clamped_double", "2027-07-15", 0, 0, .90, "clamp"),
        ("above_bar", "2027-08-30", 56, 56, .99, ""),
        ("no_bar", "2027-07-15", 56, 56, .99, ""),
        ("empty", "2027-07-15", 10, 10, .75, "empty"),
        ("no_selection", "2027-07-15", 10, 10, .75, "none"),
    ]:
        directory = root / case
        now = datetime.fromisoformat(day + "T10:00:00-04:00")
        calendar_day = now.date()
        frame = walk._predictions(calendar_day, p)
        if detail == "removed":
            frame.loc[0, "p_game_hit"] = .999
        if detail == "empty":
            frame = frame.iloc[:0].copy()
        config = walk._config(directory)
        config["orchestrator"]["heartbeat_path"] = str(directory / "heartbeat")
        states = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        if detail == "removed":
            states[70000] = {"abstract": "F", "detailed": "Final"}
        if detail == "final":
            states = dict.fromkeys(states, {"abstract": "F", "detailed": "Final"})

        def statuses(*args):
            if detail == "raises":
                raise OSError("offline")
            return states

        with walk.offline(now), mp.context() as patch:
            assert walk.install_state(directory, calendar_day, walk.State("dated", streak, best, "used", p)) == []
            patch.setattr(orchestrator, "run_cascade", lambda *args, **kwargs: (frame.copy(), "offline"))
            patch.setattr(picks, "get_game_statuses_detailed", statuses)
            patch.setattr(scheduler, "count_new_confirmations", lambda *args, **kwargs: 0)
            patch.setattr(scheduler, "heartbeat_watchdog", lambda *args, **kwargs: contextlib.nullcontext())
            if detail == "heuristic":
                patch.setattr(strategy, "_load_mdp", lambda *a, **k: None)
            if detail == "none":
                patch.setattr(orchestrator, "run_and_pick", lambda *a, **kw: (frame.copy(), None, "offline"))
            if detail == "clamp":
                load = contest_state.load_decision_streak_state
                patch.setattr(contest_state, "load_decision_streak_state",
                              lambda *a, **k: dataclasses.replace(load(*a, **k), allow_double=False))
            result = scheduler.run_single_check(date=day, all_game_pks=list(states),
                                                confirmed_sides=set(), config=config, early_lock_gap=.03)
        if detail == "none":
            assert result["selection"] is None
        if detail == "clamp":
            decision = result["selection"].decision
            assert (decision.policy_action, decision.action) == ("double", "single")
        summary = result.get("skip_summary")
        if summary:
            print(case, summary, file=sys.stderr)
            result["skip_summary"] = {k: v for k, v in summary.items() if k not in STRIP}
        results[case] = result
    return results


def bars(strategy):
    artifact = strategy._load_mdp()
    assert artifact["tail"] is not None
    end = datetime(2027, 9, 27).date()
    values = []
    regimes = set()
    for days in range(182):
        day = (end - timedelta(days=days)).isoformat()
        for streak in range(57):
            regime, _ = strategy._regime(artifact, streak, day)
            regimes.add(regime)
            bests = [(streak, "trusted")]
            if regime == strategy.OBJECTIVE_TAIL:
                bests += [(56, "trusted"), (56, "untrusted")]
            for saver in (False, True):
                for best, trust in bests:
                    values.append(strategy.effective_pick_bar(streak, day, saver, artifact, best, trust))
    assert len(regimes) == 2
    return values


def skip_lifecycle(scheduler, root, mp):
    from bts import picks
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick

    config = {"scheduler": {"pick_delivery": "dm"}, "bluesky": {"dm_recipient": "offline.test"}}
    summ = {"best_batter": "A", "best_team": "T", "best_p": .75, "streak": 10}
    now = NOW.isoformat()
    results = {}
    for effect in ("id", "no_id", "error", "interrupt", "exit"):
        state = state_fixture(scheduler, DAY, summ)
        scheduler.save_state(state, root)
        events = []

        def send(*args):
            events.append(args[0])
            if len(events) == 1:
                if effect in ("error", "interrupt", "exit"):
                    raise {"error": OSError, "interrupt": KeyboardInterrupt, "exit": SystemExit}[effect]()
                return "message ID" if effect == "id" else None
            return "message ID"

        def notify(s):
            try:
                return scheduler.maybe_notify_skip(s, s.skip_summary, config, now_iso=now, send=send)
            except (KeyboardInterrupt, SystemExit) as exc:
                return type(exc).__name__

        first = notify(state)
        before_save = scheduler.load_state(DAY, root)
        retry = notify(before_save)
        scheduler.save_state(before_save, root)
        restarted = scheduler.carry_forward_skip_state(state_fixture(scheduler, DAY), scheduler.load_state(DAY, root))
        assert restarted.skip_summary == summ
        results[effect] = first, state.skip_notified_at, retry, restarted.skip_notified_at, notify(restarted), events

    mp.setattr(scheduler, "_now_et", lambda: NOW)
    mp.setattr(scheduler, "_earliest_pick_game_et", lambda *_: NOW)
    mp.setattr(scheduler, "_alert_missed_pick", Mock())
    picks.save_pick(daily_pick(), root)
    gate = []
    for summary in (None, {}, summ, summ | {"basis": "no_decision"}):
        scheduler._maybe_alert_missed_pick(config, DAY, root, 10, None, state_fixture(scheduler, DAY, summary))
        gate.append(scheduler._alert_missed_pick.call_count)
    assert gate == [1, 2, 2, 2], gate
    return results, gate


def delivery_effects(scheduler, root, mp):
    from bts import picks
    from bts.health import alert
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic
    from tests.test_saved_pick_paths_cs5a import world

    config, directory, clock, _ = world.__wrapped__(root, mp)
    deliveries = []

    def capture(recipient, body):
        deliveries.append((recipient, body.splitlines()[0]))
        if len(deliveries) == 1:
            raise OSError("synthetic first-send failure")
        return "confirmed"

    set_automatic(directory, now=NOW)
    reading = directory / "account_state/contest_streak.json"
    reading.write_text(json.dumps(dict(json.loads(reading.read_text()), recorded_at="2027-07-14T01:00:00-04:00")))
    mp.setattr(alert, "send_dm", capture)
    record = {"stale": True, "reason": "old", "recorded_at": "2027-07-14T01:00:00-04:00", "age_hours": 36.0}
    status_file = directory.parent / "health_state/contest_stale_reading_dm_delivery_status.json"
    status_snapshots = []

    def status():
        return {k: v for k, v in json.loads(status_file.read_text()).items() if k != "body_sha256"}

    clock[0] = NOW
    scheduler._morning_state_check(config, directory, DAY, NOW, NOW + timedelta(minutes=30))
    status_snapshots.append(status())
    scheduler._warn_stale_contest_reading(record, config, directory)
    status_snapshots.append(status())
    scheduler._morning_state_check(config, directory, DAY, NOW, NOW + timedelta(minutes=30))
    assert len(deliveries) == 2
    clock[0] = datetime(2027, 12, 31, 23, 59, tzinfo=NOW.tzinfo)
    scheduler._warn_stale_contest_reading(record, config, directory)
    status_snapshots.append(status())
    clock[0] += timedelta(minutes=2)
    scheduler._warn_stale_contest_reading(record, config, directory)
    status_snapshots.append(status())
    assert [s.get("sent_day") for s in status_snapshots] == [None, DAY, "2027-12-31", "2028-01-01"]
    assert all(s.get("sent_sources") == ["WARN_ATTENTION:contest_state:stale_reading"] for s in status_snapshots[1:])

    clock[0] = NOW
    interrupt_results = {}
    for kind in (OSError, KeyboardInterrupt, SystemExit):
        for hook in ("Alert", "dispatch_dm_for_health_alerts"):
            with mp.context() as patch:
                patch.setattr(alert, hook, Mock(side_effect=kind("planted")))
                scheduler._warn_stale_contest_reading(record, config, directory)
                try:
                    scheduler._morning_state_check(config, directory, DAY, NOW, NOW + timedelta(minutes=30))
                except BaseException as error:
                    interrupt_results[kind.__name__ + "/" + hook] = type(error).__name__
                else:
                    raise AssertionError(f"morning inner route did not raise {kind.__name__}")

    refusals = {}
    for route in ("late", "void"):
        daily = daily_pick()
        target = directory / route
        picks.save_pick(daily, target)
        state = state_fixture(scheduler, DAY)
        with mp.context() as patch:
            dispatch = Mock()
            patch.setattr(alert, "dispatch_dm_for_health_alerts", dispatch)
            if route == "late":
                scheduler._refuse_late_delivery(daily, config, target, state, DAY, route, NOW - timedelta(minutes=3), NOW)
            else:
                scheduler._refuse_void_game_delivery(daily, config, target, state, DAY, route, {101: "Postponed"})
        assert picks.load_pick(DAY, target) is None
        delivered, = dispatch.call_args.args[0]
        loaded = scheduler.load_state(DAY, target)
        assert loaded.delivery_refusals == state.delivery_refusals and not loaded.pick_locked
        refusal, = loaded.delivery_refusals
        archive = target / DAY / refusal["archive"]
        archive_data = json.loads(archive.read_text())
        refusals[route] = {
            "metadata": refusal, "archive": archive_data,
            "level_source": (delivered.level, delivered.source),
            "status_path": str(dispatch.call_args.kwargs["status_path"].relative_to(root)),
        }
    return {"warning_status": status_snapshots, "warning_deliveries": deliveries,
            "interrupts": interrupt_results, "refusals": refusals}


def serialization(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(type(obj))


def worker(reference, family, root):
    import bts
    from bts import scheduler, strategy

    expected = (HERE / "base/src" if reference else ROOT / "src").resolve()
    actual = Path(bts.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
    print("bts.__file__", actual, file=sys.stderr)
    with MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", protect_socket)
        mp.setattr(socket, "create_connection", protect_socket)
        mp.setattr(socket, "getaddrinfo", protect_socket)
        jobs = {
            "summary": lambda: summaries(scheduler, not reference),
            "checks": lambda: checks(scheduler, root, mp),
            "bars": lambda: bars(strategy),
            "lifecycle": lambda: skip_lifecycle(scheduler, root, mp),
            "effects": lambda: delivery_effects(scheduler, root, mp),
        }
        return jobs[family]()


def main():
    family = sys.argv[1]
    if len(sys.argv) > 2:
        root = Path(sys.argv[3])
        outcome = worker(sys.argv[2] == "base", family, root)
        print(json.dumps(outcome, default=serialization, sort_keys=True))
        return
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        records = []
        for reference, source in (("base", HERE / "base/src"), ("head", ROOT / "src")):
            target = Path(temp) / reference
            target.mkdir()
            process = subprocess.run([sys.executable, str(__file__), family, reference, str(target)],
                                     env=dict(os.environ, PYTHONPATH=str(source)), stdout=subprocess.PIPE,
                                     cwd=ROOT, check=True, timeout=12, text=True)
            records.append(json.loads(process.stdout))
        before, after = records
        assert before == after, (before, after)
        print(f"{family}: identical, {len(before)} entries")
        if family == "checks":
            print({name: result["selection"] and result["selection"]["decision"] for name, result in before.items()})
        if family == "lifecycle":
            print(before)


if __name__ == "__main__":
    main()
