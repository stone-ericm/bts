"""Scoring serialization across the daemon and the 1am cron scorer (review F13).

Both run load daily -> resolve -> update_streak -> save_pick around 01:00 ET
(the daemon polls until 05:00; cron scores yesterday at 01:00 — same date).
Unlocked interleaving can double-apply or lose a streak update. Contract: a
shared flock serializes the read-modify-write, and each scorer RE-CHECKS the
pick's result inside the lock before touching the streak.
"""
import json
import threading
import time
from pathlib import Path

from click.testing import CliRunner

from bts.picks import scoring_lock


def test_scoring_lock_is_mutually_exclusive(tmp_path):
    a_holds = threading.Event()
    release_a = threading.Event()
    b_acquired = threading.Event()

    def holder():
        with scoring_lock(tmp_path):
            a_holds.set()
            release_a.wait(timeout=5)

    def contender():
        a_holds.wait(timeout=5)
        with scoring_lock(tmp_path):
            b_acquired.set()

    ta = threading.Thread(target=holder)
    tb = threading.Thread(target=contender)
    ta.start(); tb.start()
    assert a_holds.wait(timeout=5)
    time.sleep(0.15)
    assert not b_acquired.is_set(), "second scorer must block while the first holds the lock"
    release_a.set()
    assert b_acquired.wait(timeout=5), "second scorer must proceed once the lock is released"
    ta.join(timeout=5); tb.join(timeout=5)


def _plant_scoreable_pick(picks_dir, date_str):
    from bts.picks import DailyPick, Pick, save_pick
    from bts.daily_decision import write_decision

    daily = DailyPick(
        date=date_str, run_time=f"{date_str}T15:00:00+00:00",
        pick=Pick(batter_name="Test Batter", batter_id=1, team="NYY",
                  lineup_position=1, pitcher_name="P", pitcher_id=2,
                  p_game_hit=0.8, flags=[], projected_lineup=False,
                  game_pk=111, game_time=f"{date_str}T23:10:00+00:00"),
        double_down=None, runner_up=None,
        notification_sent=True, notification_channel="bluesky_dm",
        notification_id="dm_x",
    )
    save_pick(daily, picks_dir)
    write_decision(date_str, picks_dir, action="single", source="test",
                   delivery_status="delivered", scoreable=True)


def test_check_results_skips_when_peer_scored_mid_flight(tmp_path, monkeypatch):
    """The daemon finishes scoring between check-results' pre-check and its
    streak update: the locked re-check must skip the second update."""
    from bts.cli import cli
    import bts.picks as picks_mod

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _plant_scoreable_pick(picks_dir, "2026-07-08")
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 5}))

    real_resolve = picks_mod.resolve_daily_slot_results

    def resolve_and_simulate_peer(daily, date):
        # Peer (the daemon) scores the same date while we're resolving.
        peer = picks_mod.load_pick(date, picks_dir)
        peer.result = "hit"
        peer.slot_results = {"pick": "hit"}
        picks_mod.save_pick(peer, picks_dir)
        streak = json.loads((picks_dir / "streak.json").read_text())
        streak["streak"] = 6  # peer applied +1
        (picks_dir / "streak.json").write_text(json.dumps(streak))
        return {"pick": "hit"}

    monkeypatch.setattr(picks_mod, "resolve_daily_slot_results",
                        resolve_and_simulate_peer)

    r = CliRunner().invoke(cli, ["check-results", "--date", "2026-07-08",
                                 "--picks-dir", str(picks_dir)])

    assert r.exit_code == 0, r.output
    streak = json.loads((picks_dir / "streak.json").read_text())
    assert streak["streak"] == 6, (
        f"streak must not be double-applied (got {streak['streak']}): {r.output}"
    )
    assert "lready" in r.output  # "Already resolved by another scorer"
