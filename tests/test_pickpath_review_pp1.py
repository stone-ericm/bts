"""Regression tests from the Codex review of the 2026-11-12/13 pick-path changes (review pp1, BLOCK round 1;
archived as docs/audit/2026-11-14-pickpath-codex-r1.md). Each was the reviewer's own counterexample probe and failed
before the fix.

- B1: a same-day reconcile between a scorer's crash and its retry must not withdraw the applied result while the
  applied-date marker stops the retry from restoring it.
- B2: a double whose second leg is postponed at the final fallback must still deliver the playable leg.
- N1: the cutoff is checked AFTER the delivery guard's schedule read, in every mode.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import nullcontext
import pytest

from bts import picks as p, scheduler as s
from bts.daily_decision import load_decision
from tests import test_incident_register_2026 as H
from tests.test_late_delivery_guard import _daily, _state, CONFIG, DATE, ET


def test_crash_reconcile_restart_keeps_applied_hit(monkeypatch, tmp_path):
    picks = tmp_path / 'picks'
    # Complete prior history, so replay can legitimately rebuild a streak of 1.
    p.save_pick(H._resolved('2026-06-09', 'hit', H.BATTER_A, H.GAME_A), picks)
    p.save_pick(H._delivered_daily(H._pick(H.BATTER_A, H.GAME_A)), picks)
    p.save_streak(1, picks, saver_available=True)
    H._route_http(monkeypatch, {H.GAME_A: H.HIT_FEED()})
    now = datetime(2026, 6, 10, 23, 0, tzinfo=ET)
    real_save = p.save_pick
    def die(*a, **k):
        raise H._ProcessDeath()
    monkeypatch.setattr(p, 'save_pick', die)
    try:
        H._check_results(monkeypatch, picks, tmp_path, now)
    except H._ProcessDeath:
        pass
    assert p.load_streak(picks) == 2
    assert p.load_applied_dates(picks) == [H.DATE]
    assert p.load_pick(H.DATE, picks).result is None
    monkeypatch.setattr(p, 'save_pick', real_save)
    p.reconcile_results(picks, clock=lambda: now)
    assert p.load_streak(picks) == 2, 'the replay withdrew a result whose marker blocks the retry'
    result = H._check_results(monkeypatch, picks, tmp_path, now)
    assert result.exit_code == 0, result.output
    assert (p.load_streak(picks), p.load_pick(H.DATE, picks).result) == (2, 'hit')


def test_private_cutoff_is_rechecked_after_new_schedule_request(tmp_path):
    cfg = json.loads(json.dumps(CONFIG))
    cfg['orchestrator']['picks_dir'] = str(tmp_path)
    cfg['scheduler']['pick_delivery'] = 'private'
    now = {'at': datetime(2026, 8, 30, 13, 34, 59, tzinfo=ET)}
    def delayed_status(date):
        now['at'] += timedelta(seconds=2)
        return {824393: {'abstract': 'P', 'detailed': 'Scheduled'}}
    state = _state()
    with patch.object(s, '_now_et', side_effect=lambda: now['at']), \
         patch.object(s, '_delivery_game_statuses', side_effect=delayed_status), \
         patch.object(s, '_trigger_live_forward_capture_on_lock'), \
         patch('bts.health.alert.dispatch_dm_for_health_alerts'):
        ok = s._deliver_and_lock_pick(_daily(), cfg, tmp_path, state, DATE, 'fallback')
    assert not ok and not state.pick_locked and load_decision(DATE, tmp_path) is None


@pytest.mark.parametrize('guard_enabled', [True, False])
def test_postponed_dd_does_not_erase_playable_primary(tmp_path, guard_enabled):
    # Use L03's real run_day + real run_single_check harness. Only modify its
    # first cascade to supply a two-game DD; subsequent cascades still fail.
    original = H._l03_selection
    def dd_selection():
        predictions, selection, tier = original()
        daily = selection.pick_result.daily
        daily.double_down = H._pick(H.BATTER_B, H.L03_GAME, game_time=daily.pick.game_time)
        daily.pick.game_pk = H.L03_GAME + 1  # healthy primary, postponed DD
        daily.pick.team = 'BOS'
        predictions.loc[0, 'game_pk'] = H.L03_GAME + 1
        predictions.loc[0, 'team'] = 'BOS'
        selection.action = 'double'
        return predictions, selection, tier
    original_statuses = p.get_game_statuses_detailed
    # H supplies schedule/status mocks internally for the postponed DD. Add a
    # second healthy primary to its schedule/status answers through wrappers.
    original_run_day = s.run_day
    def run_day(**kwargs):
        internal_schedule = s.fetch_schedule
        internal_statuses = p.get_game_statuses_detailed
        def schedule(date):
            games = internal_schedule(date)
            if games:
                healthy = dict(games[0], gamePk=H.L03_GAME + 1, teams={
                    'away': {'team': {'name': 'BOS'}}, 'home': {'team': {'name': 'NYY'}}})
                return games + [healthy]
            return games
        def statuses(date):
            return {**internal_statuses(date), H.L03_GAME + 1: {'abstract': 'P', 'detailed': 'Scheduled'}}
        with patch.object(s, 'fetch_schedule', side_effect=schedule), \
             patch.object(p, 'get_game_statuses_detailed', side_effect=statuses), \
             patch.object(s, '_delivery_game_statuses', side_effect=statuses), \
             patch('bts.health.alert.dispatch_dm_for_health_alerts'):
            return original_run_day(**kwargs)
    guard_control = nullcontext() if guard_enabled else patch.object(s, '_void_committed_games', return_value={})
    with guard_control, patch.object(H, '_l03_selection', side_effect=dd_selection), patch.object(s, 'run_day', side_effect=run_day):
        obs = H._run_l03_day(tmp_path / 'picks', postponed_at=H.L03_POSTPONED_AT)
    daily = obs['daily']
    assert daily is not None and p.pick_was_delivered(daily)
    if guard_enabled:                          # the playable primary alone; the postponed leg is gone
        assert daily.pick.game_pk == H.L03_GAME + 1 and daily.double_down is None
        assert load_decision(H.L03_DATE, tmp_path / 'picks')['action'] == 'single'
