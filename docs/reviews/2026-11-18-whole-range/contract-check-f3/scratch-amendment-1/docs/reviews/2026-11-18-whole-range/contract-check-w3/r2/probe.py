"""Offline comparisons on 59575cae. Inject only the proposed consumer inputs."""

from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

from bts import contest_state, daily_decision, picks, scheduler, strategy
from bts.util import read_regular_file_text


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex"))
import offline

offline.HERE = HERE
offline.ROOT = ROOT
DAY = "2027-03-24"


@contextmanager
def deadline():
    def expired(*_):
        raise TimeoutError("decision read exceeded probe limit")

    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(1)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def decision_read(path, day):
    try:
        text = read_regular_file_text(path)
    except OSError:
        return "invalid" if os.path.lexists(path) else "absent"
    try:
        record = daily_decision.parse_decision(text)
    except (ValueError, TypeError):
        return "invalid"
    if not record or record["date"] != day:
        return "invalid"
    return "scoreable" if record["scoreable"] else "unscoreable"


def run(case, implementation, day=DAY):
    calendar = case == "midnight"
    start = offline.instant(day, 23, 20) if calendar else offline.instant(day, 18, 25)
    with offline.offline(start) as (world, mp):
        games = world.schedule[day]
        if calendar:
            games[:] = [offline.game(101, start + timedelta(minutes=70))]
            mp.setattr(strategy, "_load_mdp", lambda: None)
            mp.setattr(scheduler, "plan_run_times",
                       lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
            world.config["scheduler"].update({"fallback_deadline_min_morning": 40})
        elif case != "attempt_poll":
            games[:] = games[:1]
        world.dated(0)
        pick_path = world.picks / f"{day}.json"
        decision_path = daily_decision.decision_path(day, world.picks)
        real_check = scheduler.run_single_check
        real_lock = scheduler._lock_decision_from_predictions
        real_write = picks.atomic_write_text
        real_diagnostics = contest_state._load_model_diagnostics
        real_plan = scheduler.plan_fallback_action
        issued = []
        frames = []
        returned = False
        decision_status = None
        changed = False

        def add_attempt():
            nonlocal changed
            assert not changed
            changed = True
            candidate = picks.load_pick(day, world.picks)
            candidate.delivery_attempted = True
            picks.save_pick(candidate, world.picks)

        def forecast():
            frames.append(world.now.isoformat())
            if len(frames) == 1:
                world.advance(2399 if case in {"cutoff", "midnight"} else 2100)
                return world.frame((0.99, 0.80), (True, True))
            if case in {"cutoff", "midnight"}:
                world.advance(2)
                if calendar:
                    world.dated(0)
            if case == "attempt_poll" and not changed:
                add_attempt()
                return world.frame((0.99, 0.80), (True, True))
            return None

        def serialize(path, text):
            real_write(path, text)
            if path == pick_path:
                issued.append(text.encode("ascii"))
                data = json.loads(text)
                if case == "cutoff" and data["delivery_attempted"] and not data["notification_sent"]:
                    world.advance(1.25)

        def diagnostics(*args, **kwargs):
            result = real_diagnostics(*args, **kwargs)
            # The actual reader has already captured its clock, and still has
            # to perform its file reads. Keep the old dated bytes during I/O.
            if calendar and returned and world.now.date().isoformat() == day:
                world.advance(2)
            return result

        def confirmations(game_pks):
            return {pk: {"away", "home"} for pk in game_pks if pk == 101 or changed}

        def plan(**kwargs):
            choice = real_plan(**kwargs)
            if (case == "attempt_poll" and implementation != "legacy"
                    and choice.action == "defer" and not changed):
                add_attempt()
                kwargs["confirmed_sides"].update({(102, "away"), (102, "home")})
                return real_plan(**kwargs)
            return choice

        def check(*args, **kwargs):
            nonlocal returned, decision_status
            captured = []

            def capture(*a, **kw):
                result = real_lock(*a, **kw)
                captured.append(result)
                return result

            with patch.object(scheduler, "_lock_decision_from_predictions", capture):
                result = real_check(*args, **kwargs)
            if not captured or returned:
                return result
            returned = True
            assert not result["should_post"]
            original_bytes = issued[-1]
            if case in {"fifo", "malformed", "wrong_date"}:
                decision_path.parent.mkdir(exist_ok=True)
                if case == "fifo":
                    os.mkfifo(decision_path)
                elif case == "malformed":
                    decision_path.write_text("{")
                else:
                    wrong = daily_decision.write_decision(
                        day, world.picks, action="single", source="mdp",
                        delivery_status="private_locked", scoreable=True,
                    )
                    decision_path.write_text(json.dumps(dict(wrong, date="2027-03-23")))
            if implementation == "legacy":
                return result
            reloaded, raw = picks.load_pick_bytes(day, world.picks)
            assert raw == original_bytes and reloaded.date == day
            if implementation == "plain":
                record, record_bytes = daily_decision.load_decision_bytes(day, world.picks)
                decision_status = "scoreable" if record and record["scoreable"] else (
                    "invalid" if record_bytes else "absent")
            else:
                decision_status = decision_read(decision_path, day)
            if decision_status == "absent":
                result["lock_decision"], = captured
                result["saved_pick_bytes"] = raw
            return result

        world.predictions = forecast
        mp.setattr(picks, "atomic_write_text", serialize)
        mp.setattr(contest_state, "_load_model_diagnostics", diagnostics)
        mp.setattr(scheduler, "check_confirmed_lineups", confirmations)
        mp.setattr(scheduler, "plan_fallback_action", plan)
        mp.setattr(scheduler, "run_single_check", check)
        timed_out = False
        try:
            with deadline():
                scheduler.run_day(day, world.config)
        except TimeoutError:
            timed_out = True
        dms = [when.isoformat(timespec="milliseconds") for kind, when, _ in world.events if kind == "dm"]
        alerts = world.sends("health-dm")
        decision_kind = decision_read(decision_path, day)
        print("RESULT", json.dumps({
            "case": case, "date": day, "implementation": implementation,
            "status_at_reuse": decision_status, "status_after_run": decision_kind,
            "timeout": timed_out, "predictions": frames, "pick_dm": dms,
            "alerts": [body.splitlines()[-1] for body in alerts],
        }), flush=True)
        return timed_out, dms, alerts, decision_kind


if __name__ == "__main__":
    for case in sys.argv[1:]:
        if case == "midnight":
            for day in ("2027-06-18", "2026-12-31"):
                old = run(case, "legacy", day)
                reused = run(case, "regular", day)
                assert not old[1] and any("[contest_state]" in alert for alert in old[2])
                assert len(reused[1]) == 1
        elif case == "fifo":
            assert len(run(case, "legacy")[1]) == 1
            assert run(case, "plain")[0]
            assert len(run(case, "regular")[1]) == 1
        elif case in {"malformed", "wrong_date"}:
            assert len(run(case, "regular")[1]) == 1
        else:
            old = run(case, "legacy")
            reused = run(case, "regular")
            reason = "[late_delivery]" if case == "cutoff" else "OUTCOME UNKNOWN"
            assert not old[1] and any(reason in alert for alert in old[2])
            assert len(reused[1]) == 1
