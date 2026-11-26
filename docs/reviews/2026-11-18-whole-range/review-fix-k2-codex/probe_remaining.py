import ast
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from offline import ROOT
from probe_journal_pipe import PipeJournal
from probe_startup import DAY, HAS_WARNING, Startup, game, scheduler
from bts import strategy
from bts.simulate.mdp import ACTIONS
from scripts.audit.season_walk import walk
from tests.test_stale_delivery_cs4c import seed_automatic


def stale_morning(root):
    results = []
    for age in (3, 20):
        for at in ("15:54:55", "15:55:01"):
            start = Startup(root / f"{age}-{at}", mode="dm", required=True)
            at_et = datetime.fromisoformat(f"{DAY}T{at}").replace(tzinfo=scheduler.ET)
            seed_automatic(start.root / "picks", (at_et - timedelta(hours=age)).isoformat(),
                           now=at_et, streak=20)
            checks, journal = start.run([game()], at=at)
            morning = [body for body in start.bodies if "[contest_state] Morning check" in body]
            expected = age == 20 and at == "15:54:55"
            assert bool(morning) == expected, (morning, journal)
            if expected:
                assert start.bodies[0] == morning[0] and "about 20.00 hours ago" in morning[0]
            results.append([age, at, checks, morning])
    print("stale+fresh", json.dumps(results))


def pipe_at_cutoff(root):
    start = Startup(root / "late")
    late = "18:59:59"

    def on_drain():
        start.now += timedelta(seconds=2)

    journal = PipeJournal(270, on_drain)
    try:
        checks, output = start.run([game()], at=late, journal=journal)
        assert bool(journal.blocked_write) is HAS_WARNING, output
        if HAS_WARNING:
            assert "SEASON BOUNDARY BEHIND SCHEDULE" in journal.blocked_write, journal.blocked_write
        assert not start.bodies
        expected = (f"{DAY}T19:00:01-04:00", [301]) if HAS_WARNING else (f"{DAY}T{late}-04:00", [])
        assert checks == [expected], output
        print("late restart", checks, "physical journal blocked", bool(journal.blocked_write))
    finally:
        journal.finish()


def wrong_action_column():
    if not HAS_WARNING:
        return
    state = walk.State("dated", 55, 55, "used", .8)
    shipped = strategy._load_mdp()
    changed = shipped["policy_table"].copy()
    changed[55, 1, :, :] = ACTIONS.index("single")
    before = walk.walk_one(walk.MAKEUP_GAME_DAY, state)
    with patch.object(strategy, "_load_mdp", return_value=shipped | {"policy_table": changed}):
        after = walk.walk_one(walk.MAKEUP_GAME_DAY, state)
    assert (before["action"], after["action"]) == ("double", "single")
    assert walk.check([after]) == []
    print("walk accepts wrong one-day reach57 action", before["action"], after["action"])


def matched_script_syntax():
    manifest = ROOT / ".codex-review/wr-k2-r1/matched-files.txt"
    scripts = [ROOT / line for line in manifest.read_text().splitlines()
               if line.startswith("scripts/") and line.endswith(".py")]
    for script in scripts:
        ast.parse(script.read_bytes(), filename=str(script))
    print("search-matched source scripts parsed", len(scripts))


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        stale_morning(root)
        pipe_at_cutoff(root)
    wrong_action_column()
    matched_script_syntax()
