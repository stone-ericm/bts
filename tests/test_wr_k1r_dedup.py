"""K1r (R75): the focused read's two orders deliver every alert again (FINAL2's wall-clock dedup).

The focused Codex read of FINAL3 (`WRK12READ-43f82e3d`) found that K1's slate-date dedup
suppressed a distinct alert FINAL2 delivered when the first send for slate D came BEFORE
midnight. These replay its two orders through the real `run_day`, using the read's own
harness (`closing-test-2/addendum-final3/codex-read/rollover.py`), and require every alert
to reach the wire. Both fail on K1 (main `5cd58a0c`) and pass after K1r.
"""

from datetime import timedelta
from pathlib import Path
import importlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
READ = ROOT / "docs/reviews/2026-11-18-whole-range/closing-test-2/addendum-final3/codex-read"


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(READ))
    return importlib.import_module("rollover")


def test_two_void_refusals_across_midnight_both_deliver(harness, tmp_path):
    """23:58 ET: game 1 refused as postponed; 00:24 ET: game 2 of the same slate refused as cancelled."""
    from bts import picks, scheduler
    from scripts.audit.season_walk import walk

    day = "2027-06-14"
    start = harness.clock(day, 23, 58)
    later = start + timedelta(minutes=26)
    with harness.scenario(tmp_path, start, [start + timedelta(minutes=44),
                                            later + timedelta(minutes=45)]) as (sim, _):
        state = walk.State("automatic", 57, 57, "used", .94)
        assert not walk.install_state(sim.picks, start.date(), state)
        walk._set_clock(start)
        sim.cascade.side_effect = [(sim.predictions.iloc[[i]], "synthetic") for i in (0, 1)]

        def last_status(_):
            picked = picks.load_pick(day, sim.picks).pick.game_pk
            detail = "Postponed" if picked == sim.games[0]["gamePk"] else "Cancelled"
            return {picked: {"detailed": detail}}

        sim.delivery_status.side_effect = last_status
        sim.run()
        first, second = (str(game["gamePk"]) for game in sim.games)
        refused = scheduler.load_state(day, sim.picks).delivery_refusals
        assert [r["void_games"] for r in refused] == [{first: "Postponed"}, {second: "Cancelled"}]
        assert not sim.pick_dms
        assert [(r["sources"], r["wall"], r["wire"]) for r in sim.dispatches] == [
            (["postponed_pick"], start.isoformat(), True),
            (["postponed_pick"], later.isoformat(), True),
        ]
        assert len(sim.health_dms) == 2
        assert f"DELIVERY REFUSED for {day}: game {first} is Postponed." in sim.health_dms[0]
        assert f"DELIVERY REFUSED for {day}: game {second} is Cancelled." in sim.health_dms[1]
        assert sim.read_receipt()["sent_day"] == "2027-06-15"


def test_digest_then_restart_then_live_refusal_all_deliver(harness, tmp_path, monkeypatch):
    """23:46 ET end-of-day digest (malformed automatic file); restart at 23:59 on the same slate;
    00:01 ET live refusal (invalid streak -1), then that run's own end-of-day digest."""
    from bts.health import runner
    from scripts.audit.season_walk import walk

    day = "2027-06-14"
    start = harness.clock(day, 23, 46)
    with harness.scenario(tmp_path, start, [start + timedelta(minutes=44)]) as (sim, mp):
        original = runner._safe_run
        mp.setattr(runner, "_safe_run", lambda key, fn: original(key, fn) if key == "contest_state" else [])
        mp.setattr(runner, "read_systemd_nrestarts", lambda: None)
        sim.config["health_checks"]["enabled"] = True
        sim.write_bad_observation()
        sim.cascade.return_value = (None, None)
        sim.run()
        assert len(sim.dispatches) == 1 and sim.dispatches[0]["caller"].startswith("run_all_checks<-")
        assert sim.dispatches[0]["sources"] == ["contest_state"] and sim.dispatches[0]["wire"]

        walk._set_clock(harness.clock(day, 23, 59))
        live_at = start + timedelta(minutes=15)

        def cascade(*_):
            walk._set_clock(live_at)
            (sim.picks / "account_state" / "contest_streak.json").write_text('{"active_streak": -1}')
            return sim.predictions, "synthetic"

        sim.cascade.side_effect = cascade
        sim.run()
        assert not sim.pick_dms
        live = sim.dispatches[1]
        assert live["caller"].startswith("_alert_contest_state_failure<-run_single_check:")
        assert (live["sources"], live["wall"], live["wire"]) == (["contest_state"], live_at.isoformat(), True)
        assert sim.dispatches[2]["caller"].startswith("run_all_checks<-")
        assert sim.dispatches[2]["wire"]
        assert len(sim.health_dms) == 3
        assert "contest streak state malformed" in sim.health_dms[0]
        assert f"No pick was sent for {day} at this check. Reason: " in sim.health_dms[1]
        assert "Prior slate 2027-06-14 ET." in sim.health_dms[2]
