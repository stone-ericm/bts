"""Live-only: games whose submission cutoff has passed are not pick candidates.

Offline/backtest callers pass nothing and are unchanged (Codex review 2026-08-30 #7:
this is NOT a margin — only games that are already unenterable are excluded, so an
enterable pick is never filtered away). Lets a late cycle re-pick from later games
instead of re-selecting the batter the delivery guard just refused.
"""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def _preds():
    return pd.DataFrame([
        {"batter_name": "Kwan", "batter_id": 1, "team": "CLE", "game_pk": 100, "lineup": 1,
         "pitcher_name": "Lugo", "pitcher_id": 9, "p_game_hit": 0.76, "flags": "",
         "game_time": "2026-08-30T17:40:00Z"},
        {"batter_name": "McNeil", "batter_id": 2, "team": "ATH", "game_pk": 200, "lineup": 2,
         "pitcher_name": "Bassitt", "pitcher_id": 8, "p_game_hit": 0.74, "flags": "",
         "game_time": "2026-08-30T20:05:00Z"},
        {"batter_name": "Alvarez", "batter_id": 3, "team": "HOU", "game_pk": 300, "lineup": 2,
         "pitcher_name": "Pecko", "pitcher_id": 7, "p_game_hit": 0.70, "flags": "",
         "game_time": "2026-08-30T19:10:00Z"},
    ])


def test_games_past_cutoff_helper():
    from bts.scheduler import _games_past_cutoff
    games = [{"game_pk": 100, "game_time_et": "2026-08-30T13:40:00-04:00"},
             {"game_pk": 200, "game_time_et": "2026-08-30T16:05:00-04:00"},
             {"game_pk": 999, "game_time_et": None}]                       # malformed → ignored
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 13, 34, 59, tzinfo=ET)) == set()
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 13, 35, tzinfo=ET)) == {100}
    assert _games_past_cutoff(games, datetime(2026, 8, 30, 16, 0, tzinfo=ET)) == {100, 200}


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_select_pick_skips_unavailable_games(_st, tmp_path):
    from bts.strategy import select_pick
    sel = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                      allow_double=True, unavailable_game_pks={100})
    daily = sel.pick_result.daily
    assert daily.pick.batter_name == "McNeil"
    assert daily.double_down is None or daily.double_down.game_pk != 100
    assert daily.runner_up is None or daily.runner_up["batter_name"] != "Kwan"


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_select_pick_unchanged_without_the_argument(_st, tmp_path):
    from bts.strategy import select_pick
    sel = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                      allow_double=True)
    assert sel.pick_result.daily.pick.batter_name == "Kwan"


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_existing_pick_in_unavailable_game_is_replaced(_st, tmp_path):
    from bts.picks import save_pick
    from bts.strategy import select_pick
    first = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False)
    save_pick(first.pick_result.daily, tmp_path)
    again = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                        unavailable_game_pks={100})
    assert again.pick_result.daily.pick.batter_name == "McNeil"


def test_run_single_check_archives_undelivered_pick_past_cutoff(tmp_path):
    """Codex r2 F5: an undelivered preview whose committed game is past its cutoff
    used to classify as locked (game started) and strand the day; it must be
    archived so the cascade re-picks from later games."""
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import run_single_check
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1,
                pitcher_name="L", pitcher_id=2, p_game_hit=0.75, flags=[],
                projected_lineup=False, game_pk=100, game_time="2026-08-30T17:40:00Z")
    save_pick(DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None,
                        runner_up=None), tmp_path)
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".hb")}, "tiers": []}
    with patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick", return_value=(None, None, None)) as rap:
        run_single_check(date="2026-08-30", all_game_pks=[100, 200], confirmed_sides=set(),
                         config=config, early_lock_gap=0.03, unavailable_game_pks={100})
    rap.assert_called_once()                                  # cascade ran (no strand)
    assert list((tmp_path / "2026-08-30").glob("stale_pick_*.json"))
    assert not (tmp_path / "2026-08-30.json").exists()


def test_run_single_check_never_archives_a_committed_pick(tmp_path):
    """Rebase reconciliation with the 2026-08-14 'committed means immutable'
    fix: a scoreable decision on disk (e.g. private_locked — the pick FILE
    carries no delivery flags) must win over the past-cutoff archive."""
    import json as _json
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import run_single_check
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1,
                pitcher_name="L", pitcher_id=2, p_game_hit=0.75, flags=[],
                projected_lineup=False, game_pk=100, game_time="2026-08-30T17:40:00Z")
    save_pick(DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None,
                        runner_up=None), tmp_path)
    (tmp_path / "2026-08-30").mkdir()
    (tmp_path / "2026-08-30" / "decision.json").write_text(_json.dumps({
        "schema_version": "bts_daily_decision_v2", "date": "2026-08-30",
        "action": "single", "source": "mdp", "primary": None, "double_down": None,
        "second_candidate": None, "streak": 0, "saver_available": False,
        "state_source": None, "state_status": None, "allow_double": True,
        "contest_source_date": None, "delivery_status": "private_locked",
        "scoreable": True, "finalized_at": "2026-08-30T16:00:00Z"}))
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".hb")}, "tiers": []}
    with patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick") as rap:
        result = run_single_check(date="2026-08-30", all_game_pks=[100], confirmed_sides=set(),
                                  config=config, early_lock_gap=0.03,
                                  unavailable_game_pks={100})
    rap.assert_not_called()
    assert result["pick_result"].locked is True
    assert (tmp_path / "2026-08-30.json").exists()
    assert not list((tmp_path / "2026-08-30").glob("stale_pick_*.json"))


def test_run_single_check_never_archives_an_attempted_pick(tmp_path):
    """delivery_attempted=True is the crash-gap duplicate-delivery guard
    (2026-08-14); the past-cutoff archive must not erase it."""
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import run_single_check
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1,
                pitcher_name="L", pitcher_id=2, p_game_hit=0.75, flags=[],
                projected_lineup=False, game_pk=100, game_time="2026-08-30T17:40:00Z")
    save_pick(DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None,
                        runner_up=None, delivery_attempted=True), tmp_path)
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".hb")}, "tiers": []}
    with patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick") as rap:
        result = run_single_check(date="2026-08-30", all_game_pks=[100], confirmed_sides=set(),
                                  config=config, early_lock_gap=0.03,
                                  unavailable_game_pks={100})
    rap.assert_not_called()
    assert result["pick_result"].locked is True
    assert (tmp_path / "2026-08-30.json").exists()
    assert not list((tmp_path / "2026-08-30").glob("stale_pick_*.json"))


def test_run_single_check_keeps_delivered_pick_past_cutoff(tmp_path):
    """A DELIVERED pick whose game started stays locked — that is the committed day."""
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import run_single_check
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1,
                pitcher_name="L", pitcher_id=2, p_game_hit=0.75, flags=[],
                projected_lineup=False, game_pk=100, game_time="2026-08-30T17:40:00Z")
    save_pick(DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None,
                        runner_up=None, notification_sent=True, notification_id="m"), tmp_path)
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".hb")}, "tiers": []}
    with patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick") as rap:
        result = run_single_check(date="2026-08-30", all_game_pks=[100], confirmed_sides=set(),
                                  config=config, early_lock_gap=0.03,
                                  unavailable_game_pks={100})
    rap.assert_not_called()
    assert result["pick_result"].locked is True
    assert (tmp_path / "2026-08-30.json").exists()


@patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P", 300: "P"})
def test_all_games_unavailable_reports_no_eligible(_st, tmp_path):
    from bts.strategy import select_pick
    sel = select_pick(_preds(), "2026-08-30", tmp_path, streak=0, saver_available=False,
                      unavailable_game_pks={100, 200, 300})
    assert sel.pick_result is None and sel.no_pick_reason == "no_eligible"
