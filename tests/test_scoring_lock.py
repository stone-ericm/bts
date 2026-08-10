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

    r = CliRunner().invoke(cli, ["check-results", "--date", "2026-07-08", "--allow-stale-scoring",
                                 "--picks-dir", str(picks_dir)])

    assert r.exit_code == 0, r.output
    streak = json.loads((picks_dir / "streak.json").read_text())
    assert streak["streak"] == 6, (
        f"streak must not be double-applied (got {streak['streak']}): {r.output}"
    )
    assert "lready" in r.output  # "Already resolved by another scorer"


def test_save_nonterminal_result_refuses_to_clobber_terminal(tmp_path):
    """Review round 2 #1: the daemon's cap/unresolved/suspended writers must
    not overwrite a terminal result a peer scored while their stale object
    was in hand."""
    from bts.scheduler import save_nonterminal_result
    import bts.picks as picks_mod

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _plant_scoreable_pick(picks_dir, "2026-07-08")
    peer = picks_mod.load_pick("2026-07-08", picks_dir)
    peer.result = "hit"
    picks_mod.save_pick(peer, picks_dir)

    fresh = save_nonterminal_result("2026-07-08", picks_dir, "unresolved")

    assert fresh is not None and fresh.result == "hit"
    assert picks_mod.load_pick("2026-07-08", picks_dir).result == "hit"


def test_save_nonterminal_result_marks_when_not_terminal(tmp_path):
    from bts.scheduler import save_nonterminal_result
    import bts.picks as picks_mod

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _plant_scoreable_pick(picks_dir, "2026-07-08")

    fresh = save_nonterminal_result("2026-07-08", picks_dir, "suspended")

    assert fresh is not None and fresh.result == "suspended"
    assert picks_mod.load_pick("2026-07-08", picks_dir).result == "suspended"


def test_check_results_fails_closed_when_pick_vanishes(tmp_path, monkeypatch):
    """Review round 2 #6: fresh=None inside the lock must not resurrect and
    score the stale object."""
    from bts.cli import cli
    import bts.picks as picks_mod

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _plant_scoreable_pick(picks_dir, "2026-07-08")
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 5}))

    def resolve_and_delete(daily, date):
        (picks_dir / "2026-07-08.json").unlink()
        return {"pick": "hit"}

    monkeypatch.setattr(picks_mod, "resolve_daily_slot_results", resolve_and_delete)
    r = CliRunner().invoke(cli, ["check-results", "--date", "2026-07-08", "--allow-stale-scoring",
                                 "--picks-dir", str(picks_dir)])

    assert r.exit_code == 0, r.output
    assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 5
    assert not (picks_dir / "2026-07-08.json").exists(), "stale object must not be resurrected"


def test_check_results_adopts_fresh_metadata_before_scoring(tmp_path, monkeypatch):
    """Review round 2 #6: a concurrent metadata update (nonterminal) must not
    be clobbered by saving the stale pre-lock object."""
    from bts.cli import cli
    import bts.picks as picks_mod

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _plant_scoreable_pick(picks_dir, "2026-07-08")
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 5}))

    def resolve_and_touch_metadata(daily, date):
        peer = picks_mod.load_pick(date, picks_dir)
        peer.notification_id = "peer-updated-mid-flight"
        picks_mod.save_pick(peer, picks_dir)
        return {"pick": "hit"}

    monkeypatch.setattr(picks_mod, "resolve_daily_slot_results", resolve_and_touch_metadata)
    r = CliRunner().invoke(cli, ["check-results", "--date", "2026-07-08", "--allow-stale-scoring",
                                 "--picks-dir", str(picks_dir)])

    assert r.exit_code == 0, r.output
    scored = picks_mod.load_pick("2026-07-08", picks_dir)
    assert scored.result == "hit"
    assert scored.notification_id == "peer-updated-mid-flight"


def test_reconcile_results_saves_under_scoring_lock(tmp_path, monkeypatch):
    """Review round 2 #4: the 2am reconcile is a streak/pick writer and must
    hold the shared lock during its mutation phase."""
    import fcntl
    import bts.picks as picks_mod
    from datetime import date as date_cls, timedelta

    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    yesterday = (date_cls.today() - timedelta(days=1)).isoformat()
    _plant_scoreable_pick(picks_dir, yesterday)
    scored = picks_mod.load_pick(yesterday, picks_dir)
    scored.result = "hit"
    scored.slot_results = {"pick": "hit"}
    picks_mod.save_pick(scored, picks_dir)
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 5}))

    lock_held_during_save = []

    real_save = picks_mod.save_pick

    def probing_save(daily, pd):
        with open(Path(pd) / ".scoring.lock", "w") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_held_during_save.append(False)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                lock_held_during_save.append(True)
        return real_save(daily, pd)

    # correction path: boxscore now says miss
    monkeypatch.setattr(picks_mod, "resolve_daily_slot_results",
                        lambda daily, d: {"pick": "miss"})
    monkeypatch.setattr(picks_mod, "save_pick", probing_save)

    corrections = picks_mod.reconcile_results(picks_dir, lookback_days=2)

    assert corrections and corrections[0]["new_result"] == "miss"
    assert lock_held_during_save and all(lock_held_during_save), (
        "reconcile must hold scoring_lock while writing pick/streak state"
    )
