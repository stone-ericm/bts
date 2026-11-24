"""Generate the legacy snapshot and rehearse the first three days."""

import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

from bts import picks, scheduler
from bts.saver_state import transition_saver_state

from offline import Interrupted, REVIEW, World, at

OPEN = "2027-03-24"
DAY2 = "2027-03-25"
DAY3 = "2027-03-26"
OLD = "2026-09-14"
SNAPSHOT = REVIEW / "synthetic-snapshot"


def seed():
    SNAPSHOT.mkdir(exist_ok=True)
    with World(OLD, SNAPSHOT) as w:
        w.config["scheduler"].update(pick_delivery="private", entry_intent="research")
        w.config["scheduler"].pop("contest_state_required")
        w.config["health_checks"].pop("contest_state_expected")
        w.profile = {"activeStreak": 17, "seasonBestStreak": 18,
                     "predictions": [{"roundId": 1, "result": "hit"}]}
        picks.save_streak(17, w.directory, False)
        transition_saver_state(w.directory, expected_prior="uninitialized", new_state="used",
                               season=2026, source="fixture")
        w.fetch()
        w.run(via_cli=True)
        w.toml()
        (w.root / "restored-crontab.txt").write_text(
            "10 1,2,10,13 * * * uv run bts fetch-contest-streak # synthetic October snapshot\n")
        w.snapshot()
        w.dump("seed")


def set_slate(w, day):
    w.day = day
    w.now = at(day)
    w.finish = at(day, "22:00")
    if day not in w.games:
        w.game("19:10")
        w.game("20:10", 102, pair="B")
        w.game("21:10", 103, pair="C")
    w.rounds = {1: at(OPEN).date(), 2: at(DAY2).date(), 3: at(DAY3).date()}
    w.prediction = None
    w.pending = []


def enter_pick(w):
    def on_send(recipient, body):
        if body.startswith(("Today's pick:", "Today's picks (double down):")):
            round_id = next(key for key, day in w.rounds.items() if day == w.now.date())
            w.pending = [{"roundId": round_id, "playerId": i} for i in (1, 2) if f"Batter {i}" in body]
    return on_send


def schedule_restart(w, mode):
    if mode == "midday":
        def interrupt():
            raise Interrupted()
        w.actions.append((at(w.day, "13:00"), interrupt))
    elif mode == "mid_send":
        def interrupt_send(recipient, body):
            if body.startswith(("Today's pick:", "Today's picks (double down):")):
                w.on_send = None
                w.record("fixture_transport", "accepted request, process stopped before acknowledgement")
                raise Interrupted()
        w.on_send = interrupt_send
    w.actions.sort(key=lambda entry: entry[0])


def run_with_restart(w, label):
    try:
        w.run()
    except Interrupted:
        w.record("process_restart", label)
        w.snapshot()
        w.now += timedelta(seconds=30)
        w.run()


def account_streak(w, prior, round_id):
    daily = picks.load_pick(w.day, w.directory)
    entered = len(w.pending)
    if daily and daily.result == "hit" and entered:
        streak = prior + entered
        w.profile["predictions"].append({"roundId": round_id, "result": "hit"})
    else:
        streak = prior
    w.profile.update(activeStreak=streak, seasonBestStreak=streak)
    w.record("fixture_account", "settled profile", json.loads(json.dumps(w.profile)))
    return streak


def opening(case, restart=None):
    with World(OPEN) as w:
        shutil.copytree(SNAPSHOT / "data/picks", w.directory, dirs_exist_ok=True)
        shutil.copytree(SNAPSHOT / "data/health_state", w.directory.parent / "health_state", dirs_exist_ok=True)
        w.now = at("2027-03-23", "12:00")
        config = w.toml()
        w.invoke("entry-intent", "--config", str(config), expect=0)
        w.switch("revoke", "--reason", "restored from the offseason snapshot")
        w.switch("status")
        w.now = at(OPEN)
        w.rounds = {1: at(OPEN).date(), 2: at(DAY2).date(), 3: at(DAY3).date()}
        if case == "midnight":
            w.games[OPEN] = []
            w.game("22:10")
            w.finish = at(DAY2, "00:20")
        if case not in {"missing", "between", "too_late", "preview_only"}:
            w.dated(30 if case in {"corrected", "corrected_after_selection", "corrected_after_delivery"} else 0)
        if case == "between":
            w.actions.append((at(OPEN, "12:00"), w.dated))
        if case == "too_late":
            w.actions.append((at(OPEN, "18:26"), w.dated))
        if case == "corrected":
            w.actions.append((at(OPEN, "12:00"), w.dated))
        if case == "corrected_after_selection":
            def projected():
                confirmed = w.now >= at(OPEN, "18:25")
                return w.frame(projected=not confirmed), "offline"
            w.prediction = projected
            w.actions.append((at(OPEN, "17:35"), w.dated))
        if case == "corrected_after_delivery":
            w.actions.append((at(OPEN, "17:35"), w.dated))
        if case in {"fallback_refusal", "fallback_matching", "fallback_in_loop"}:
            def failed():
                if w.now >= at(OPEN, "16:11"):
                    if case == "fallback_in_loop" and w.now == at(OPEN, "18:25"):
                        return w.frame(projected=True), "offline"
                    return (None, None)
                return w.frame(projected=True), "offline"
            w.prediction = failed
            if case != "fallback_matching":
                w.actions.append((at(OPEN, "16:11"), lambda: w.dated(1)))
            if case == "fallback_in_loop":
                statuses = w.statuses
                def warmup(day):
                    table = statuses(day)
                    if at(OPEN, "18:20") <= w.now < at(OPEN, "19:10") and 101 in table:
                        table[101].update(abstract="L", code="PW", detailed="Warmup")
                    return table
                w.patch(picks, "get_game_statuses_detailed", warmup)
        if case == "preview_only":
            w.now = at(OPEN, "03:00")
            # The actual optional-state preview can still see the old automatic
            # file if there is no dated observation yet.
            w.invoke("preview", "--date", OPEN, expect=0)
            w.snapshot()
            w.now = at(OPEN)
            w.dated()
            w.prediction = lambda: (None, None)
        w.on_send = enter_pick(w)
        check = at(OPEN, "21:45" if case == "midnight" else "18:45")
        w.actions.append((check, w.watchdog))
        schedule_restart(w, restart)
        run_with_restart(w, restart)
        if w.actions:
            w.wait(max(0, (check - w.now).total_seconds()))
        w.snapshot()
        streak = account_streak(w, 0, 1)
        set_slate(w, DAY2)
        if case in {"preview_only", "fallback_refusal"}:
            w.now = at(DAY2, "02:00")
            w.invoke("reconcile", expect=0)
            w.snapshot(OPEN)
        w.now = at(DAY2, "08:30")
        w.outcome(OPEN)
        w.now = at(DAY2)
        w.on_send = enter_pick(w)
        w.actions = [(at(DAY2, "18:45"), w.watchdog)]
        w.dated(streak)
        w.now = at(DAY2, "10:30")
        w.fetch()
        w.run()
        w.snapshot()
        streak = account_streak(w, streak, 2)
        set_slate(w, DAY3)
        w.now = at(DAY3, "08:30")
        w.outcome(DAY2)
        w.now = at(DAY3)
        w.actions = [(at(DAY3, "18:45"), w.watchdog)]
        if case in {"automatic", "old_auto", "old_profile"}:
            w.switch("status")
            w.switch("enable")
            if case != "old_auto":
                if case == "old_profile":
                    w.profile.update(activeStreak=17, seasonBestStreak=18,
                                     predictions=[{"roundId": 0, "result": "hit"}])
                    w.rounds[0] = at(OLD).date()
                w.fetch()
            w.switch("status")
            w.invoke("saver-state")
            schedule_restart(w, restart)
        else:
            w.dated(streak)
        run_with_restart(w, restart)
        w.snapshot()
        w.now = at("2027-03-27", "08:30")
        w.outcome(DAY3)
        w.dump(f"opening-{case}-{restart or 'steady'}")


def restart_after_lock():
    with World(OPEN) as w:
        w.dated()
        def stop_after_delivery():
            raise Interrupted()
        w.actions = [(at(OPEN, "17:00"), stop_after_delivery)]
        try:
            w.run()
        except Interrupted:
            w.record("process_restart", "confirmed delivery before grading")
            w.snapshot()
        (w.directory / "account_state/contest_streak.manual.json").unlink()
        w.actions = [(at(OPEN, "18:45"), w.watchdog)]
        w.run()
        w.snapshot()
        w.dump("opening-delivered-restart")


def unknown_and_watchdog():
    with World(OPEN) as w:
        w.dated()
        def accept_then_die(recipient, body):
            w.on_send = None
            raise Interrupted()
        w.on_send = accept_then_die
        try:
            w.run()
        except Interrupted:
            w.record("process_restart", "after remote acceptance before acknowledgement")
        w.now = at(OPEN, "17:00")
        w.actions = [(at(OPEN, "18:45"), w.watchdog)]
        w.run()
        w.now = at(DAY2, "08:30")
        w.outcome(OPEN)
        w.snapshot()
        w.dump("opening-unknown-watchdog")


if __name__ == "__main__":
    if sys.argv[1:] == ["seed"]:
        seed()
    else:
        for case in ("correct", "missing", "between", "too_late", "corrected",
                     "corrected_after_selection", "corrected_after_delivery",
                     "automatic", "old_auto", "old_profile",
                     "midnight", "fallback_refusal", "fallback_matching", "fallback_in_loop", "preview_only"):
            for restart in (None, "midday", "mid_send"):
                opening(case, restart)
        restart_after_lock()
        unknown_and_watchdog()
