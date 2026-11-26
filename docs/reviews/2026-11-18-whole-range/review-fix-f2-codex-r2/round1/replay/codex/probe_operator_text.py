import datetime as dt
import json
import re
import shlex
from contextlib import chdir
from pytest import MonkeyPatch

from bts import contest_state, orchestrator, scheduler, strategy, web
from probe_support import (
    DAY, DECEMBER, INCIDENT, SECOND, automatic, bts, clock, command, eod, flag,
    health, observed, receipt, scratch,
)


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def render(picks, day=DAY, *, wall=None):
    raw, sends = eod(picks, day, wall=wall)
    alert, = (item for item in raw if item.incident_key == INCIDENT)
    _, body = sends[0]
    assert str(picks) not in body
    if wall is not None:
        assert "`" not in body and body.endswith(health.HISTORICAL_POINTER)
    else:
        assert re.findall(r"`([^`]+)`", body) == ["bts season-contract status"]
    return alert, body


def status_from_dm(picks, day, body):
    executable = re.findall(r"`([^`]+)`", body)[0]
    verb, *args = shlex.split(executable)
    assert verb == "bts"
    return command(day, *args, picks=picks)


def read_flag(picks, day):
    return command(day, "saver-state", picks=picks)


def mark_used(picks, day):
    return command(day, "saver-state", "--use", picks=picks)


def wrong_observation(root):
    picks = root / "scheduler directory" / "picks"
    manual = observed(picks)
    filename = flag(picks, "active")
    before = filename.read_bytes()
    _, body = render(picks)
    print("LIVE DM\n" + body)
    with chdir(root):
        # The shell command name in the DM defaults to a different, valid directory.
        flag(root / "data/picks", "used")
        default_status = status_from_dm(None, DAY, body)
        scheduler_status = status_from_dm(picks, DAY, body)
    assert "saver: used" in default_status and "saver: active" in scheduler_status
    assert "saver=used" in scheduler_status
    assert filename.read_bytes() == before
    # MLB confirms active, so the spent-saver correction does not apply.
    observed(picks, DAY + dt.timedelta(minutes=10), state="active")
    assert json.loads(manual.read_text())["saver_state"] == "active"
    assert filename.read_bytes() == before
    now = DAY + dt.timedelta(minutes=15)
    assert "saver=active" in status_from_dm(picks, now, body)
    assert not eod(picks, now)[1]
    print("Wrong dated side: real status shows the disagreement; corrected current dated row leaves flag active. "
          "Rerunning the printed command later sees the correction when --picks-dir is supplied.")


def dashboard(root):
    picks = root / "dashboard" / "picks"
    observed(picks)
    filename = flag(picks, "active")
    _, body = render(picks)
    now = DAY + dt.timedelta(minutes=10)

    def change(before, after, instant):
        result = web.saver_transition_response(picks, expected_prior=before, new_state=after,
                                               same_origin=True, now=instant)
        assert result[0] == 200, result

    change("active", "used", now)
    assert "saver: used" in status_from_dm(picks, now, body)
    assert "saver_state: used" in read_flag(picks, now)
    # If the dashboard correction happens before the current inspection, the warning is harmless.
    change("used", "active", now + dt.timedelta(minutes=1))
    assert "saver: active" in status_from_dm(picks, now, body)
    assert json.loads(filename.read_text())["state"] == "active"
    print("Dashboard changes before rechecking: real status sees used and then active; no stale command in DM.")

    # Model an official scoring correction between operator A's fresh MLB / flag checks and action.
    # Operator B sees the correction and presses Undo. A is still following B8a's previously read result.
    assert "saver_state: active" in read_flag(picks, now)
    change("active", "used", now + dt.timedelta(minutes=2))
    change("used", "active", now + dt.timedelta(minutes=3))
    observed(picks, now + dt.timedelta(minutes=3), state="active")
    outcome = mark_used(picks, now + dt.timedelta(minutes=4))
    assert "Marked saver used" in outcome and json.loads(filename.read_text())["state"] == "used"
    print("Intervening scoring/observation correction after B8a verification: CLI --use accepts the dashboard's "
          "restored active state and writes used. The ordinary CLI guard compares the current state, "
          "not the snapshot checked by the operator.")


def next_morning(root):
    from tests.test_saved_pick_paths_cs5a import world

    with MonkeyPatch.context() as mp:
        config, picks, now, _ = world.__wrapped__(root / "morning", mp)
        mp.setattr(strategy, "_mdp_cache", None)
        manual = observed(picks)
        original = manual.read_bytes()
        flag(picks, "active")
        _, body = render(picks)
        when = DAY.replace(hour=11) + dt.timedelta(days=1)
        now[0] = when
        automatic(picks, when, streak=0, coverage=when.date() - dt.timedelta(days=1))
        frame, tier = orchestrator.run_cascade.return_value
        date = str(when.date())
        orchestrator.run_cascade.return_value = (frame.assign(
            p_game_hit=[0.99, 0.96], game_time=[f"{date}T23:10:00Z", f"{date}T23:40:00Z"]), tier)
        with clock(when):
            selected = scheduler._refresh_pick_at_fallback_decision(config, date, None, 0.03)
        assert selected.daily is not None, selected
        assert selected.daily.chosen_on["kind"] == "automatic"
        before = (picks / f"{date}.json").read_bytes()
        assert "saver: active" in status_from_dm(picks, when, body)
        assert "saver_state: active" in read_flag(picks, when)
        assert "Marked saver used" in mark_used(picks, when)
        assert not contest_state.load_decision_streak_state(picks, now=when, require_contest_state=True).saver_available
        after = (picks / f"{date}.json").read_bytes()
        assert before == after and manual.read_bytes() == original
        print("Next morning after a real automatic pick was saved: verified used repair changes the following "
              "decision read, leaves the saved pick and expired observation bytes unchanged.")


def year(root):
    picks = root / "year" / "picks"
    observed(picks, DECEMBER)
    flag(picks, "not_earned", DECEMBER.year)
    _, live = render(picks, DECEMBER)
    jan = (DECEMBER + dt.timedelta(days=1)).replace(hour=0, minute=6)
    receipt(picks, SECOND).unlink()
    _, history = render(picks, DECEMBER, wall=jan)
    print("DECEMBER LIVE DM\n" + live)
    print("HISTORICAL DM\n" + history)
    assert "2027-12-31 ET" in live and "season 2027" in live
    assert "January" not in history and "season 2027" in history
    assert "inspected season: 2027" in status_from_dm(picks, DECEMBER, live)
    assert "season 2027" in read_flag(picks, DECEMBER)
    # The original year's instruction permits a same-year repair. The C1 pointer permits inspection only.
    assert "Marked saver used" in mark_used(picks, DECEMBER)
    flag(picks, "active", jan.year)
    before = snapshot(picks)
    assert "inspected season: 2028" in status_from_dm(picks, jan, live)
    assert "saver_state: active (season 2028)" in read_flag(picks, jan)
    assert snapshot(picks) == before
    print("Late December reads ET season despite UTC rollover. Next ET year, B8a's D-year guard prohibits "
          "acting on the old DM, and read-only status leaves the current season untouched.")


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    with scratch() as directory:
        wrong_observation(directory)
        dashboard(directory)
        next_morning(directory)
        year(directory)
