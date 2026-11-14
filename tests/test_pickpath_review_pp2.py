"""Partial-void doubles keep their original slots; fixture-only delivery/restart regressions."""
import io
import json
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from bts import picks as p, scheduler as s
from bts.contest_state import ContestStateError
from bts.daily_decision import load_decision
from bts.health import late_delivery
from tests import test_incident_register_2026 as H
from tests.test_late_delivery_guard import _daily, _state, CONFIG, DATE, ET
from tests.test_scheduler_decision_record_integration import _game


def _config(picks):
    cfg = json.loads(json.dumps(CONFIG))
    cfg['orchestrator']['picks_dir'] = str(picks)
    cfg['orchestrator']['heartbeat_path'] = str(picks.parent / '.hb')
    cfg['scheduler']['lineup_check_offset_min'] = 60
    cfg['health_checks'] = {'enabled': False}
    return cfg


@pytest.mark.parametrize('mode', ['dm', 'public', 'private'])
@pytest.mark.parametrize('void_primary', [False, True])
def test_partial_void_double_is_delivered_without_reduction(tmp_path, capsys, mode, void_primary):
    daily = _daily(dd_utc=f'{DATE}T20:05:00Z')
    primary, double = daily.pick, daily.double_down
    slots = (asdict(primary), asdict(double))
    dead = primary if void_primary else double
    cfg = _config(tmp_path)
    cfg['scheduler']['pick_delivery'] = mode
    p.save_pick(daily, tmp_path)
    state = _state()
    real_save = p.save_pick
    saves = []

    def save(obj, target):
        saves.append((obj.delivery_attempted, p.pick_was_delivered(obj)))
        assert obj.pick is primary and obj.double_down is double
        assert (asdict(obj.pick), asdict(obj.double_down)) == slots
        return real_save(obj, target)

    with patch.object(s, '_now_et', return_value=datetime(2026, 8, 30, 13, 0, tzinfo=ET)), \
         patch.object(s, '_delivery_game_statuses', return_value={dead.game_pk: {'detailed': 'Postponed'}}), \
         patch.object(s, '_trigger_live_forward_capture_on_lock'), \
         patch('bts.contest_state.load_decision_streak_state', return_value=MagicMock(streak=0)), \
         patch('bts.health.alert.dispatch_dm_for_health_alerts') as alert, \
         patch('bts.dm.send_dm', return_value='msg-1') as dm, \
         patch('bts.posting.post_to_bluesky', return_value='uri-1') as post, \
         patch.object(p, 'save_pick', side_effect=save):
        assert s._deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, 'fallback')
    assert daily.pick is primary and daily.double_down is double
    saved = p.load_pick(DATE, tmp_path)
    assert (asdict(saved.pick), asdict(saved.double_down)) == slots
    assert not state.delivery_refusals
    alert.assert_not_called()
    assert (dm.call_count, post.call_count) == {'dm': (1, 0), 'public': (0, 1), 'private': (0, 0)}[mode]
    if mode != 'private':
        text = (dm if mode == 'dm' else post).call_args.args[-1]
        assert primary.batter_name in text and double.batter_name in text
        assert p.pick_was_delivered(saved)
    assert saves == ([(False, False)] if mode == 'private' else [(True, False), (True, True)])
    decision = load_decision(DATE, tmp_path)
    assert decision['action'] == 'double' and decision['scoreable']
    assert decision['primary']['batter_id'] == primary.batter_id
    assert decision['double_down']['batter_id'] == double.batter_id
    assert not list(tmp_path.glob('**/refused_delivery*'))
    assert late_delivery.check(tmp_path, today=datetime.fromisoformat(DATE).date()) == []
    lines = [line for line in capsys.readouterr().err.splitlines() if 'Postponed' in line]
    assert len(lines) == 1 and str(dead.game_pk) in lines[0]
    assert 'reduc' not in lines[0].lower()


@pytest.mark.parametrize('mode', ['dm', 'public'])
@pytest.mark.parametrize('void_primary', [False, True])
def test_caught_send_failure_keeps_double_and_has_no_reduction_health(tmp_path, mode, void_primary):
    daily = _daily(dd_utc=f'{DATE}T20:05:00Z')
    dead = daily.pick if void_primary else daily.double_down
    p.save_pick(daily, tmp_path)
    cfg = _config(tmp_path)
    cfg['scheduler']['pick_delivery'] = mode
    state = _state()
    with patch.object(s, '_now_et', return_value=datetime(2026, 8, 30, 13, 0, tzinfo=ET)), \
         patch.object(s, '_delivery_game_statuses', return_value={dead.game_pk: {'detailed': 'Cancelled'}}), \
         patch.object(s, '_trigger_live_forward_capture_on_lock'), \
         patch('bts.contest_state.load_decision_streak_state', return_value=MagicMock(streak=0)), \
         patch('bts.dm.send_dm', side_effect=OSError('offline')), \
         patch('bts.posting.post_to_bluesky', side_effect=OSError('offline')), \
         patch('bts.health.alert.dispatch_dm_for_health_alerts') as alert:
        assert not s._deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, 'fallback')
    saved = p.load_pick(DATE, tmp_path)
    assert saved.double_down is not None and not p.pick_was_delivered(saved) and not saved.delivery_attempted
    assert not state.delivery_refusals
    alert.assert_not_called()
    assert late_delivery.check(tmp_path, today=datetime.fromisoformat(DATE).date()) == []


@pytest.mark.parametrize('early_void', [False, True])
@pytest.mark.parametrize('mode', ['dm', 'public', 'private'])
@pytest.mark.parametrize('void_primary', [False, True])
@pytest.mark.parametrize('fault', ['state_save', 'delivery_save', 'contest_state', 'send'])
def test_cached_double_restart_matches_pre_guard_outcome(tmp_path, early_void, mode, void_primary, fault):
    """Compare real cache/archival/fallback outcomes with scheduler 4199360.

    A state-save death now targets the normal post-send/lock save: reduction's
    pre-send state save no longer exists. An early void slot retains its old cutoff.
    """
    if mode == 'private' and fault in {'contest_state', 'send'}:
        pytest.skip('private has neither contest-state fetch nor transport')
    picks = tmp_path / 'picks'
    cfg = _config(picks)
    cfg['scheduler']['pick_delivery'] = mode
    daily = _daily(dd_utc=f'{DATE}T23:10:00Z')
    primary_id, double_id = daily.pick.batter_id, daily.double_down.batter_id
    kept = daily.double_down if void_primary else daily.pick
    dead = daily.pick if void_primary else daily.double_down
    kept.game_time = f'{DATE}T23:10:00Z'
    dead.game_time = f'{DATE}T17:40:00Z' if early_void else f'{DATE}T23:10:00Z'
    p.save_pick(daily, picks)
    original = (picks / f'{DATE}.json').read_bytes()
    clock = {'now': datetime(2026, 8, 30, 13, 34, tzinfo=ET) if early_void else
                    datetime(2026, 8, 30, 18, 35, tzinfo=ET)}
    statuses = {kept.game_pk: {'abstract': 'P', 'detailed': 'Scheduled'},
                dead.game_pk: {'abstract': 'F', 'detailed': 'Postponed'}}
    games = [_game(kept.game_pk, '19:10', date=DATE),
             _game(dead.game_pk, '13:40' if early_void else '19:10', date=DATE)]
    games[1]['teams'] = {'away': {'team': {'name': 'BOS'}}, 'home': {'team': {'name': 'NYY'}}}

    def die(*args, **kwargs):
        raise H._ProcessDeath()

    def sleep(seconds, *args, **kwargs):
        clock['now'] += timedelta(seconds=seconds)

    with ExitStack() as stack:
        stack.enter_context(patch.object(s, '_now_et', side_effect=lambda: clock['now']))
        stack.enter_context(patch.object(s, '_delivery_game_statuses', return_value=statuses))
        stack.enter_context(patch.object(p, 'get_game_statuses_detailed', return_value=statuses))
        stack.enter_context(patch.object(s, 'fetch_schedule', side_effect=lambda d: games if d == DATE else []))
        stack.enter_context(patch.object(s, 'count_new_confirmations', return_value=0))
        stack.enter_context(patch.object(s, '_watchdog_ping_sleep', side_effect=sleep))
        stack.enter_context(patch.object(s, 'run_result_polling', return_value='final'))
        stack.enter_context(patch.object(s, '_trigger_live_forward_capture_on_lock'))
        stack.enter_context(patch.object(s, '_idle_until_next_wakeup'))
        stack.enter_context(patch('bts.orchestrator.run_and_pick', return_value=(None, None, 'local')))
        stack.enter_context(patch.object(s, 'run_and_pick', return_value=(None, None, 'local')))
        stack.enter_context(patch('bts.health.alert.dispatch_dm_for_health_alerts'))
        stack.enter_context(patch('bts.contest_state.load_decision_streak_state', return_value=MagicMock(streak=0)))
        stack.enter_context(patch.object(s, '_alert_contest_state_failure'))
        dm = stack.enter_context(patch('bts.dm.send_dm', return_value='msg-1'))
        post = stack.enter_context(patch('bts.posting.post_to_bluesky', return_value='uri-1'))
        if fault == 'contest_state':
            with patch('bts.contest_state.load_decision_streak_state', side_effect=ContestStateError('offline')):
                assert not s._deliver_and_lock_pick(daily, cfg, picks, _state(), DATE, 'lineup')
        elif fault == 'send':
            with patch('bts.dm.send_dm', side_effect=OSError('offline')) as failed_dm, \
                 patch('bts.posting.post_to_bluesky', side_effect=OSError('offline')) as failed_post:
                assert not s._deliver_and_lock_pick(daily, cfg, picks, _state(), DATE, 'lineup')
                assert failed_dm.call_count + failed_post.call_count == 1
        else:
            module, name = (s, 'save_state') if fault == 'state_save' else (p, 'save_pick')
            with patch.object(module, name, side_effect=die):
                with pytest.raises(H._ProcessDeath):
                    s._deliver_and_lock_pick(daily, cfg, picks, _state(), DATE, 'lineup')
            if fault == 'delivery_save':
                assert (picks / f'{DATE}.json').read_bytes() == original
        clock['now'] += timedelta(minutes=2 if early_void else 1)
        s.run_day(DATE, cfg)
    after = p.load_pick(DATE, picks)
    decision = load_decision(DATE, picks)
    # Measured on 4199360: an uncommitted early-slot double is lost at its old
    # cutoff; a post-send state-save death retains the delivered double instead.
    committed = not early_void or (fault == 'state_save' and mode != 'private')
    print('restart outcome', early_void, mode, void_primary, fault,
          'file', bool(after), 'decision', decision and decision['delivery_status'],
          'sends', dm.call_count + post.call_count)
    if committed:
        assert after is not None and after.pick.batter_id == primary_id and after.double_down.batter_id == double_id
        assert decision and decision['action'] == 'double' and decision['scoreable']
        assert (dm.call_count, post.call_count) == {'dm': (1, 0), 'public': (0, 1), 'private': (0, 0)}[mode]
    else:
        assert after is None and (decision is None or not decision['scoreable'])
        assert dm.call_count == post.call_count == 0


def test_corrected_status_after_pre_send_error_matches_pre_guard_loss(tmp_path):
    """The cached slot still has its old time when its schedule start is corrected."""
    picks = tmp_path / 'picks'
    cfg = _config(picks)
    daily = _daily(dd_utc=f'{DATE}T23:10:00Z')
    kept, dead = daily.double_down, daily.pick
    p.save_pick(daily, picks)
    state = _state()
    clock = {'now': datetime(2026, 8, 30, 13, 34, tzinfo=ET)}
    statuses = {kept.game_pk: {'abstract': 'P', 'detailed': 'Scheduled'},
                dead.game_pk: {'abstract': 'F', 'detailed': 'Postponed'}}
    games = [_game(kept.game_pk, '19:10', date=DATE), _game(dead.game_pk, '19:40', date=DATE)]
    games[1]['teams'] = {'away': {'team': {'name': 'BOS'}}, 'home': {'team': {'name': 'NYY'}}}

    def sleep(seconds, *args, **kwargs):
        clock['now'] += timedelta(seconds=seconds)

    with ExitStack() as stack:
        stack.enter_context(patch.object(s, '_now_et', side_effect=lambda: clock['now']))
        stack.enter_context(patch.object(s, '_delivery_game_statuses', side_effect=lambda d: statuses))
        stack.enter_context(patch.object(p, 'get_game_statuses_detailed', side_effect=lambda d: statuses))
        stack.enter_context(patch.object(s, 'fetch_schedule', side_effect=lambda d: games if d == DATE else []))
        stack.enter_context(patch.object(s, 'count_new_confirmations', return_value=0))
        stack.enter_context(patch.object(s, '_watchdog_ping_sleep', side_effect=sleep))
        stack.enter_context(patch.object(s, 'run_result_polling', return_value='final'))
        stack.enter_context(patch.object(s, '_trigger_live_forward_capture_on_lock'))
        stack.enter_context(patch.object(s, '_idle_until_next_wakeup'))
        stack.enter_context(patch('bts.orchestrator.run_and_pick', return_value=(None, None, 'local')))
        stack.enter_context(patch.object(s, 'run_and_pick', return_value=(None, None, 'local')))
        stack.enter_context(patch('bts.health.alert.dispatch_dm_for_health_alerts'))
        stack.enter_context(patch('bts.contest_state.load_decision_streak_state', return_value=MagicMock(streak=0)))
        stack.enter_context(patch.object(s, '_alert_contest_state_failure'))
        dm = stack.enter_context(patch('bts.dm.send_dm', return_value='msg-1'))
        with patch('bts.contest_state.load_decision_streak_state', side_effect=ContestStateError('offline')):
            assert not s._deliver_and_lock_pick(daily, cfg, picks, state, DATE, 'lineup')
        statuses[dead.game_pk] = {'abstract': 'P', 'detailed': 'Scheduled'}
        clock['now'] += timedelta(minutes=2)
        s.run_day(DATE, cfg)
    after = p.load_pick(DATE, picks)
    decision = load_decision(DATE, picks)
    print('corrected-status outcome', bool(after), decision, 'sends', dm.call_count)
    assert after is None and (decision is None or not decision['scoreable']) and dm.call_count == 0


@pytest.mark.parametrize('polling', [False, True])
@pytest.mark.parametrize('void_primary', [False, True])
@pytest.mark.parametrize('void_preview', [False, True], ids=['final_zero_ab', 'preview_no_roster'])
def test_delivered_hit_plus_void_double_adds_exactly_one(monkeypatch, tmp_path, void_primary, polling, void_preview):
    picks = tmp_path / 'picks'
    cfg = _config(picks)
    daily = p.DailyPick(date=H.DATE, run_time=H.DATE+'T21:00:00Z',
                       pick=H._pick(H.BATTER_A, H.GAME_A), double_down=H._pick(H.BATTER_B, H.GAME_B), runner_up=None)
    p.save_pick(H._resolved('2026-06-09', 'hit', H.BATTER_A, H.GAME_A), picks)
    p.save_streak(1, picks, saver_available=True)
    p.save_pick(daily, picks)
    kept = daily.double_down if void_primary else daily.pick
    dead = daily.pick if void_primary else daily.double_down
    with patch.object(s, '_now_et', return_value=datetime(2026, 6, 10, 18, 35, tzinfo=ET)), \
         patch.object(s, '_delivery_game_statuses', return_value={dead.game_pk: {'detailed': 'Postponed'}}), \
         patch.object(s, '_trigger_live_forward_capture_on_lock'), \
         patch('bts.health.alert.dispatch_dm_for_health_alerts'), \
         patch('bts.contest_state.load_decision_streak_state', return_value=MagicMock(streak=1)), \
         patch('bts.dm.send_dm', return_value='msg-1'):
        assert s._deliver_and_lock_pick(daily, cfg, picks, replace(_state(), date=H.DATE), H.DATE, 'fallback')
    assert p.load_pick(H.DATE, picks).double_down is not None
    hit_feed = H._feed(kept.batter_id, batting=H._batting(hits=1, at_bats=4))
    if void_preview:
        # No roster or batting line: only the schedule can resolve this Pass.
        void_feed = {'gameData': {'status': {'abstractGameCode': 'P', 'detailedState': 'Postponed'}},
                     'liveData': {}}
    else:
        void_feed = H._feed(dead.batter_id, batting=H._batting())
        void_feed['gameData']['status'] = {'abstractGameCode': 'F', 'detailedState': 'Postponed'}
    H._route_http(monkeypatch, {kept.game_pk: hit_feed, dead.game_pk: void_feed})
    if void_preview:
        feed_request = p.retry_urlopen

        def request(url, **kwargs):
            if '/api/v1/schedule' in url:
                games = [
                    {'gamePk': kept.game_pk, 'status': {'abstractGameCode': 'F', 'detailedState': 'Final'}},
                    {'gamePk': dead.game_pk, 'status': {'abstractGameCode': 'P', 'detailedState': 'Postponed'}},
                ]
                return io.BytesIO(json.dumps({'dates': [{'date': H.DATE, 'games': games}]}).encode())
            return feed_request(url, **kwargs)

        monkeypatch.setattr(p, 'retry_urlopen', request)
    if polling:
        with patch.object(s, 'retry_urlopen', p.retry_urlopen), \
             patch.object(s, '_now_et', return_value=H.AFTER_NORMAL_GAMES):
            # Real run_day passes the primary game even when that leg is void.
            assert s.run_result_polling(daily.pick.game_pk, H.DATE, picks) == 'final'
    else:
        result = H._check_results(monkeypatch, picks, tmp_path, H.AFTER_NORMAL_GAMES)
        assert result.exit_code == 0, result.output
    resolved = p.load_pick(H.DATE, picks)
    assert resolved.slot_results == ({'pick': 'void', 'double_down': 'hit'} if void_primary else
                                     {'pick': 'hit', 'double_down': 'void'})
    assert p.load_streak(picks) == 2 and p.streak_increment_for_resolved_hit(resolved) == 1
    p.reconcile_results(picks, clock=lambda: H.AFTER_NORMAL_GAMES)
    assert p.load_streak(picks) == 2
