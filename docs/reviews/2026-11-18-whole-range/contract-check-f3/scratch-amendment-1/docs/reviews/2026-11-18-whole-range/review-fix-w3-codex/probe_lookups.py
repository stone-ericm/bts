"""Exercise the added poll using real response parsers and bounded fake HTTP bodies."""

import io
import json
import sys
from threading import Event, current_thread, main_thread
from time import monotonic
from urllib.parse import urlsplit

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, game, health, offline, pending_window


REAL_CONFIRMATIONS = scheduler.check_confirmed_lineups


def boxscore(sides):
    teams = {side: {"players": {"ID500": {"battingOrder": "100" if side in sides else ""}}}
             for side in ("home", "away")}
    return io.BytesIO(json.dumps({"liveData": {"boxscore": {"teams": teams}}}).encode())


class Body:
    def __init__(self, payload, read_hook):
        self.payload = payload
        self.read_hook = read_hook

    def read(self):
        self.read_hook()
        return self.payload.read()


def run(case):
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        pending_window(world, mp, scheduler)
        if case == "partial-late":
            world.schedule[DAY].append(game(103, at(19, 50)))
            mp.setattr(scheduler, "plan_run_times", lambda *_args, **_kwargs: [
                {"time_et": at(18, 25), "game_pks": [101]},
                {"time_et": at(18, 50), "game_pks": [102]},
                {"time_et": at(19, 5), "game_pks": [103]},
            ])
        starts = []
        samples = []
        calls = []
        worker_started = Event()
        worker_finished = Event()
        workers = []
        release = Event()
        snapshots = []
        real_plan = scheduler.plan_fallback_action
        real_replan = scheduler._replan_after_reuse_poll

        def slow_body():
            assert current_thread().daemon
            workers.append(current_thread())
            worker_started.set()
            try:
                assert release.wait(6)
            finally:
                worker_finished.set()

        def delayed_confirmation():
            assert not Event().wait(1.20)

        def always_fails():
            raise OSError("fixture body read failed")

        def feed(url, **kwargs):
            path = urlsplit(getattr(url, "full_url", url)).path
            if not path.endswith("/feed/live"):
                return world.http(url, **kwargs)
            pk = int(path.split("/")[-3])
            if current_thread() is main_thread():
                return boxscore({"home", "away"} if pk == 101 else {"home"})
            calls.append(("lineup", pk, monotonic()))
            if case == "shared-budget":
                return Body(boxscore({"home"}), delayed_confirmation)
            if case == "partial-late" and pk == 103:
                return Body(boxscore({"away"}), slow_body)
            if case == "partial-late" and pk == 102:
                return boxscore({"away"})
            if case == "fail":
                return Body(boxscore({"away"}), always_fails)
            if case == "cutoff-before-status":
                world.advance(17 * 60 + 58)
            return boxscore({"home"})

        def schedule_body(url, **kwargs):
            body = world.http(url, **kwargs)
            if current_thread() is main_thread():
                return body
            calls.append(("status", None, monotonic()))
            if case == "shared-budget":
                return Body(body, slow_body)
            if case == "fail":
                return Body(body, always_fails)
            if case == "void":
                games = world.schedule[DAY]
                payload = json.loads(body.getvalue())
                assert payload["dates"][0]["games"][0]["gamePk"] == games[0]["gamePk"]
                payload["dates"][0]["games"][0]["status"].update(
                    statusCode="PO", detailedState="Postponed")
                return io.BytesIO(json.dumps(payload).encode())
            return body

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance({"at-reserve": 39 * 60 + 57}.get(case, 22 * 60))
                return world.frame((0.99, 0.58, 0.59), (True, True, True))
            if worker_started.is_set():
                assert not worker_finished.is_set()
                release.set()
                assert worker_finished.wait(1)
            frame = world.frame((0.40, 0.99, 0.40), (True, False, True))
            return frame.iloc[[1, 0] + ([2] if case == "partial-late" else [])]

        def observed_plan(**kwargs):
            snapshots.append({"now": kwargs["now"].isoformat(), "cutoff": kwargs["cutoff"].isoformat(),
                              "should_post": kwargs["should_post"], "ungated": kwargs["should_post_ungated"],
                              "confirmed": sorted(kwargs["confirmed_sides"]), "block": kwargs["block_reason"],
                              "runs": [run["game_pks"] for run in kwargs["remaining_runs"]],
                              "budget": kwargs["budget_min"], "reserve": kwargs["operator_reserve_min"],
                              "contenders": list(kwargs["contender_game_pks"])})
            return real_plan(**kwargs)

        def observed_replan(daily, date, cutoff, planner_inputs):
            begin = monotonic()
            result = real_replan(daily, date, cutoff, planner_inputs)
            samples.append((begin, monotonic() - begin, planner_inputs["confirmed_sides"]))
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "check_confirmed_lineups", REAL_CONFIRMATIONS)
        mp.setattr(scheduler, "retry_urlopen", feed)
        mp.setattr(picks, "retry_urlopen", schedule_body)
        mp.setattr(scheduler, "plan_fallback_action", observed_plan)
        mp.setattr(scheduler, "_replan_after_reuse_poll", observed_replan)
        try:
            scheduler.run_day(DAY, world.config)
        finally:
            release.set()
            if worker_started.is_set():
                assert worker_finished.wait(1)
            for worker in workers:
                worker.join(1)
                assert not worker.is_alive()
        began, elapsed, confirmations = samples[0]
        state = scheduler.load_state(DAY, world.picks)
        record = daily_decision.load_decision(DAY, world.picks)
        facts = {"case": case, "wall_elapsed_sec": round(elapsed, 3),
                 "lookup_start_sec": [(kind, pk, round(stamp - began, 3)) for kind, pk, stamp in calls],
                 "cascades": [t.strftime("%H:%M:%S") for t in starts], "confirmed": sorted(confirmations),
                 "planner": snapshots, "refreshes": state.fallback_refreshes,
                 "health": health(world), "commit": record and record["primary"]["game_pk"]}
        print("LOOKUPS", json.dumps(facts), flush=True)
        assert facts["commit"] == 102
        assert elapsed < 2.6
        if case == "shared-budget":
            assert starts[1] == at(18, 50) and worker_started.is_set()
            assert [kind for kind, _, _ in calls] == ["lineup", "status"]
            assert 1.1 < calls[1][2] - began < 1.8 and 1.9 <= elapsed < 2.6
        if case == "partial-late":
            assert {(102, "home"), (102, "away"), (103, "home")} <= confirmations
            assert (103, "away") not in confirmations
            assert worker_started.is_set() and worker_finished.is_set()
            assert len(calls) == 2 and all(kind == "lineup" for kind, _, _ in calls)
        if case == "fail":
            assert (102, "home") in confirmations and (102, "away") not in confirmations
            assert facts["refreshes"][0]["action"] == "defer"
        if case in {"cutoff-before-status", "at-reserve"}:
            assert all(kind != "status" for kind, _, _ in calls)
            assert len(calls) == (0 if case == "at-reserve" else 1)
        if case == "void":
            assert sum("[postponed_pick]" in s for s in health(world)) == 1
            assert not list((world.picks / DAY).glob("deferred_fallback*.json"))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["shared-budget", "partial-late", "fail", "at-reserve",
                                "cutoff-before-status", "void"]:
        run(name)
