"""Compare existing delivery call sites, alerts, result polling, and disk output."""

from contextlib import redirect_stderr
import hashlib
import io
import json
import sys
from urllib.parse import urlsplit

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, health, offline


REAL_POLL = scheduler.run_result_polling


def run(case):
    with offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
        ])
        if case in {"final", "missed", "unknown"}:
            world.predictions = lambda: world.frame((0.99,), (True,))
            seed = scheduler.run_single_check(DAY, [101], set(), world.config, 0.03)["pick_result"].daily
            if case == "missed":
                world.dated(0, best=1)
            if case == "unknown":
                seed.delivery_attempted = True
                picks.save_pick(seed, world.picks)

        labels = []
        calls = []
        stderr = io.StringIO()
        deliver = scheduler._deliver_and_lock_pick

        def forecasts():
            calls.append(world.now.isoformat())
            if len(calls) > 1 or case in {"final", "missed", "unknown"}:
                return None
            return world.frame((0.99,), (case != "lineup",))

        def transport(daily, config, directory, state, date, label, **kwargs):
            labels.append([label, sorted(kwargs)])
            assert "reused_check_cascade" not in kwargs
            return deliver(daily, config, directory, state, date, label, **kwargs)

        def http(url, **kwargs):
            path = urlsplit(getattr(url, "full_url", url)).path
            if not path.endswith("/feed/live"):
                return world.http(url, **kwargs)
            assert world.now >= at(19, 20)
            batting = {"ID601": {"person": {"id": 601, "fullName": "Synthetic Batter 101"},
                                 "stats": {"batting": {"atBats": 4, "hits": 1}}}}
            final = {"gameData": {"status": {"abstractGameCode": "F", "detailedState": "Final"}},
                     "liveData": {"boxscore": {"teams": {"away": {"players": batting},
                                                          "home": {"players": {}}}}}}
            return io.BytesIO(json.dumps(final).encode())

        world.predictions = forecasts
        mp.setattr(scheduler, "_deliver_and_lock_pick", transport)
        mp.setattr(scheduler, "retry_urlopen", http)
        mp.setattr(picks, "retry_urlopen", http)
        mp.setattr(scheduler, "run_result_polling", REAL_POLL)
        with redirect_stderr(stderr):
            scheduler.run_day(DAY, world.config)

        state = scheduler.load_state(DAY, world.picks)
        daily = picks.load_pick(DAY, world.picks)
        record = daily_decision.load_decision(DAY, world.picks)
        files = {}
        for file in sorted(world.picks.rglob("*")):
            if file.is_file() and "scheduler_state" not in file.name:
                data = file.read_bytes()
                files[str(file.relative_to(world.picks))] = hashlib.sha256(data).hexdigest()
        result = {"case": case, "labels": labels, "prediction": calls, "selected_file_sha256": files,
                  "locked": state.pick_locked, "committed": state.committed_pick_written,
                  "result": state.result_status, "scored": daily.result if daily else None,
                  "streak": picks.load_streak(world.picks, season=2027), "health": health(world),
                  "journals": stderr.getvalue().replace(str(world.root), "<fixture>"),
                  "pick_dm": world.sends("dm"), "decision": record}
        print("LEGACY", json.dumps(result, sort_keys=True), flush=True)
        if case == "missed":
            assert not record and not world.sends("dm")
            assert any("[missed_pick]" in alert for alert in result["health"])
        else:
            assert state.result_status == "final" and daily.result == "hit" and result["streak"] == 1
        expected = {"lineup": [["lineup", ["selection"]]], "slept": [["fallback", ["selection"]]],
                    "final": [["final fallback", ["selection"]]], "missed": [["final fallback", ["selection"]]],
                    "unknown": []}
        assert labels == expected[case]


def abandoned():
    with offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
        ])
        calls = []
        labels = []
        load = picks.load_pick_bytes
        deliver = scheduler._deliver_and_lock_pick
        stderr = io.StringIO()

        def forecasts():
            calls.append(world.now.isoformat())
            if len(calls) == 1:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def corrected(date, directory, **kwargs):
            if kwargs.get("regular_only"):
                path = directory / f"{date}.json"
                path.write_bytes(path.read_bytes() + b"\n")
                world.dated(0, best=1)
            return load(date, directory, **kwargs)

        def guarded(daily, config, picks_dir, state, date, label, **kwargs):
            labels.append([label, list(kwargs)])
            assert daily is not None and label == "final fallback" and set(kwargs) == {"selection"}
            return deliver(daily, config, picks_dir, state, date, label, **kwargs)

        world.predictions = forecasts
        mp.setattr(picks, "load_pick_bytes", corrected)
        mp.setattr(scheduler, "_deliver_and_lock_pick", guarded)
        with redirect_stderr(stderr):
            scheduler.run_day(DAY, world.config)
        assert labels == [["final fallback", ["selection"]]] and len(calls) == 2
        assert not world.sends("dm") and daily_decision.load_decision(DAY, world.picks) is None
        assert sorted(heading.split("]")[0] for heading in health(world)) == ["- [contest_state", "- [missed_pick"]
        assert not picks.load_pick(DAY, world.picks).delivery_attempted
        print("ABANDONED", json.dumps({"cascades": calls, "labels": labels, "health": health(world),
                                        "journals": stderr.getvalue().replace(str(world.root), "<fixture>")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["abandoned"]:
        abandoned()
    else:
        for name in ["lineup", "slept", "final", "missed", "unknown"]:
            run(name)
