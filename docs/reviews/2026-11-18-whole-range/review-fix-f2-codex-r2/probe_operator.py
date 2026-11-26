"""Walk B8a from rendered warnings through real CLI, dashboard and policy calls."""

import datetime as dt
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from bts import orchestrator, scheduler, strategy, web
from bts.simulate.mdp import load_policy
from review_support import (
    DAY, DECEMBER, automatic, cli_call, clock, contest_state, eod, files,
    flag, health, observe, select,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = ROOT / "docs/ops/2027-season-start.md"


def render(picks, day=DAY, wall=None):
    rows, sends = eod(picks, day, wall=wall)
    raw, = select(rows)
    (_, body), = sends
    assert str(picks) not in raw.message and str(picks) not in body
    if wall:
        assert "`" not in body and body.endswith(health.HISTORICAL_POINTER)
    else:
        assert re.findall(r"`([^`]+)`", body) == ["bts season-contract status"]
        assert "with the scheduler's picks directory as shown in checklist B8a." in body
    return body


def status(picks, now):
    return cli_call(picks, now, "season-contract", "status")


def dashboard_write(picks, now, old, new):
    code, detail = web.saver_transition_response(
        picks, expected_prior=old, new_state=new, same_origin=True, now=now,
    )
    assert code == 200, detail


def ordinary_walkthrough(root):
    picks = root / "operator test's scheduled picks"
    observe(picks)
    path = flag(picks, "active")
    before = path.read_bytes()
    print("LIVE DM\n" + render(picks))
    assert "saver=used" in status(picks, DAY) and "saver: active" in status(picks, DAY)
    # MLB in this case never spent the saver. Following the new B8a bullet preserves the right flag.
    assert path.read_bytes() == before
    later = DAY + dt.timedelta(minutes=5)
    cli_call(picks, later, "set-contest-streak", "--date", str(DAY.date()), "--streak", "14",
             "--best-streak", "16", "--saver", "active")
    assert "saver=active" in status(picks, later) and not select(eod(picks, later)[0])
    assert path.read_bytes() == before

    observe(picks, later)
    dashboard_write(picks, later, "active", "used")
    assert "saver_state: used" in cli_call(picks, later, "saver-state")
    dashboard_write(picks, later, "used", "active")
    observe(picks, later, saver="active")
    # The initial MLB check showed used. The added last refresh sees corrected MLB as active
    # and dashboard Undo. Follow "stop" and make no stale --use call.
    assert "saver_state: active" in cli_call(picks, later, "saver-state")
    assert "saver=active" in status(picks, later)
    assert not select(eod(picks, later)[0])
    print("Wrong dated side with a correct flag, same-day dated correction, and dashboard/MLB recheck held")

    picks = root / "year" / "picks"
    observe(picks, DECEMBER)
    flag(picks, "not_earned", DECEMBER.year)
    live = render(picks, DECEMBER)
    assert "2027-12-31 ET" in live and "season 2027" in live
    assert "inspected season: 2027" in status(picks, DECEMBER)
    jan = (DECEMBER + dt.timedelta(days=1)).replace(hour=0, minute=9)
    history = render(picks, DECEMBER, wall=jan)
    print("HISTORICAL DM\n" + history)
    assert "season 2027" in history
    new = flag(picks, "active", jan.year)
    before = new.read_bytes()
    assert "inspected season: 2028" in status(picks, jan)
    assert new.read_bytes() == before
    print("Dec 31 used the ET year; January status read the new season and did not edit it")


def after_automatic_pick(root):
    from tests.test_saved_pick_paths_cs5a import world

    for initial in ("active", "not_earned"):
        with MonkeyPatch.context() as patches:
            config, picks, now, _ = world.__wrapped__(root / initial, patches)
            patches.setattr(strategy, "_mdp_cache", None)
            manual = observe(picks)
            flag(picks, initial)
            render(picks)
            tomorrow = DAY.replace(hour=11) + dt.timedelta(days=1)
            now[0] = tomorrow
            automatic(picks, tomorrow, streak=9, coverage=DAY.date())
            frame, tier = orchestrator.run_cascade.return_value
            date = str(tomorrow.date())
            orchestrator.run_cascade.return_value = (frame.assign(
                p_game_hit=[0.99, 0.96], game_time=[f"{date}T23:10:00Z", f"{date}T23:40:00Z"],
            ), tier)
            with clock(tomorrow):
                selection = scheduler._refresh_pick_at_fallback_decision(config, date, None, 0.03)
            assert selection.daily is not None and selection.daily.chosen_on["kind"] == "automatic"
            saved = {path: path.read_bytes() for path in (manual, picks / f"{date}.json")}
            assert f"saver: {initial}" in status(picks, tomorrow)
            assert contest_state.load_decision_streak_state(picks, now=tomorrow,
                                                          require_contest_state=True).saver_available
            # Prior entries are settled, MLB still shows used, and the immediate flag refresh agrees.
            assert f"saver_state: {initial}" in cli_call(picks, tomorrow, "saver-state")
            assert "Marked saver used" in cli_call(picks, tomorrow, "saver-state", "--use")
            assert not contest_state.load_decision_streak_state(picks, now=tomorrow,
                                                              require_contest_state=True).saver_available
            assert all(path.read_bytes() == original for path, original in saved.items())
    print("Next morning: verified repair for either flag changed the future automatic read and preserved the saved pick")


def wrong_pick_after_b8a(root):
    picks = root / "b8a-case" / "picks"
    tomorrow = DAY.replace(month=8, day=20, hour=11)
    day = tomorrow.replace(day=19, hour=22)
    observe(picks, day, streak=9)
    path = flag(picks, "active")
    assert "read active" in render(picks, day)
    # Before A reads the DM, B trusts the mistaken dated row and clicks Mark used.
    # In this ordinary case MLB actually shows an earned, unused saver throughout.
    dashboard_write(picks, day, "active", "used")
    before = files(picks)
    screen = status(picks, day)
    assert "saver=used" in screen and "saver: used" in screen
    assert "the flag is right: leave the flag as it is" in CHECKLIST.read_text()
    # A obeys that sentence after comparing both rows with MLB. No state write follows.
    assert files(picks) == before
    automatic(picks, tomorrow, streak=9, coverage=day.date())
    table, bins, length = load_policy(ROOT / "data/models/mdp_policy.npz")
    ctx = strategy.DecisionContext(.79, .80, True, str(tomorrow.date()), True, {
        "policy_table": table, "boundaries": bins, "season_length": length,
    })
    stale = contest_state.load_decision_streak_state(picks, now=tomorrow, require_contest_state=True)
    wrong = strategy.resolve_policy_decision(ctx, stale.streak, stale.saver_available)
    assert not stale.saver_available and wrong.source == "mdp" and wrong.action == "skip"
    dashboard_write(picks, tomorrow, "used", "active")
    correct = contest_state.load_decision_streak_state(picks, now=tomorrow, require_contest_state=True)
    right = strategy.resolve_policy_decision(ctx, correct.streak, correct.saver_available)
    assert correct.saver_available and right.source == "mdp" and right.action == "single"
    assert "active" in path.read_text()
    print(f"B8a counterexample: {day.date()} alert read active, dashboard then wrote used, MLB still active. "
          f"Next automatic {tomorrow.date()} at streak 9: checklist's retained used => {wrong.action}; "
          f"reconciled active => {right.action} (real MDP policy, .79/.80)")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="f2-text-", dir="/tmp") as directory:
        root = Path(directory)
        ordinary_walkthrough(root)
        after_automatic_pick(root)
        wrong_pick_after_b8a(root)
