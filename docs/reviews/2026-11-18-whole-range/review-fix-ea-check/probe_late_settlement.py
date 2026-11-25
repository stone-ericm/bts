"""Compare the old and revised instructions when a suspended entry settles as Pass."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from bts import picks, scheduler
from scripts.audit.season_walk import walk

from probe_contest import WORK, at, command, lineup, observation, send_stale, stale, state


DOC = "docs/ops/2027-season-start.md"
BEFORE = "wait until MLB has settled that slate's entry or confirms there was none."
AFTER = "wait until MLB has settled all prior entries or confirmed their absence after their entry windows closed."
HANDOFF = "have Eric verify any pending or delivered pick against MLB before acting on it."


old = subprocess.run(["git", "show", f"1ed57c47:{DOC}"], cwd=WORK.parent.parent, timeout=5,
                     capture_output=True, check=True, text=True).stdout
new = (WORK.parent.parent / DOC).read_text()
assert BEFORE in old and AFTER in new and HANDOFF in new
assert "that day's ET date, run on that day) before its first lineup check." in old

sent_at = at("2027-07-15T18:35")
today = "2027-07-17"
with TemporaryDirectory(dir=WORK) as temp, walk.offline(sent_at):
    root = Path(temp)
    revised = root / "revised" / "picks"
    stale(revised, sent_at, streak=0, best=15, saver="used")
    send_stale(revised, sent_at)
    original = root / "original" / "picks"
    fixed = root / "fixed" / "picks"
    for dest in (original, fixed):
        shutil.copytree(revised, dest)

    walk._set_clock(at(today + "T09:30"))
    assert "automatic path alone: accepted (streak 0" in command(revised, "season-contract", "status")
    print("MLB at 09:30: July 15 entry settled as hit; July 16 entry pending in suspended game; displays 15/15/used.")
    print("Old B8a allows the July 17 filing at 09:30; revised B8a requires waiting for July 16.")
    assert "in effect today" in observation(original, today, 15, 15, "used")
    assert (state(original).streak, state(revised).streak) == (15, 0)

    games = [{"gamePk": 70000, "gameDate": today + "T23:05:00Z"},
             {"gamePk": 70001, "gameDate": today + "T23:10:00Z"}]
    first, second, *_ = scheduler.plan_run_times(games, now=at(today + "T10:00"))
    assert [check["time_et"].strftime("%H:%M") for check in (first, second)] == ["16:05", "17:05"]
    rows = walk._predictions(first["time_et"].date(), .80).iloc[:2].copy()
    rows["p_game_hit"] = [.80, .80]
    for dest in (original, revised, fixed):
        result, action, sent = lineup(dest, first["time_et"], rows.iloc[:0], "first check")
        assert result["pick_result"] is None and action is None and sent == []

    walk._set_clock(at(today + "T16:10"))
    print("MLB at 16:10: July 16 settled as Pass at 16:08; MLB still 15/15/used. No later fetch succeeded.")
    assert picks.load_pick(today, revised) is None
    assert "automatic path alone: accepted (streak 0" in command(revised, "season-contract", "status")
    print("Revised text's first check has started; Eric has no pending or delivered pick to verify. No filing follows.")
    assert "in effect today" in observation(fixed, today, 15, 15, "used")

    for dest, expected_streak, expected_action, dm_count in ((original, 15, "skip", 0),
                                                           (revised, 0, "double", 1),
                                                           (fixed, 15, "skip", 0)):
        result, action, sent = lineup(dest, second["time_et"], rows, dest.parent.name)
        assert (result["selection"].streak, action, len(sent)) == (expected_streak, expected_action, dm_count)
    print("Differential: only the revised instructions send a wrong pick; an after-settlement correction before the next check prevents it.")
