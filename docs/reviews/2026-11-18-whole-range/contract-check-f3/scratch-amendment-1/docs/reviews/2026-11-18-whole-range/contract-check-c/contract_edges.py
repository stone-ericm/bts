"""Exercise the ambiguities in the draft on the checked-out tree."""

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
ET = ZoneInfo('America/New_York')


def health_clock(directory):
    from bts.health import contest_state as health
    from bts.contest_state import DATED_MANUAL_SCHEMA
    from tests.test_pick_chosen_on_cs5a import account_file

    end = datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    later = end + timedelta(minutes=16)

    def observation(when, streak, saver):
        day = when.date()
        account_file(directory, 'contest_streak.manual.json', {
            'schema_version': DATED_MANUAL_SCHEMA, 'season': day.year,
            'effective_date': day.isoformat(), 'active_streak': streak, 'best_streak': streak,
            'saver_state': saver, 'recorded_at': when.isoformat(), 'source': 'probe',
            'expires_at': datetime.combine(day, time.max, ET).isoformat(),
        })

    observation(end - timedelta(hours=1), 12, 'not_earned')
    old_day = health.check(directory, expected=True, now=end.astimezone(timezone.utc))
    assert len(old_day) == 1 and old_day[0].source == 'saver_state'
    assert '--date 2027-12-31' in old_day[0].message
    wall = health.check(directory, expected=True, now=later.astimezone(timezone.utc))
    assert any('--date 2028-01-01' in alert.message for alert in wall)
    observation(later, 0, 'used')
    assert health.check(directory, expected=True, now=later.astimezone(timezone.utc)) == []
    overwritten = health.check(directory, expected=True, now=end.astimezone(timezone.utc))
    assert any('expected but missing' in alert.message for alert in overwritten)
    assert any('season 2027 switch' in alert.message for alert in overwritten)
    print('C1: contradictory prior-day saver prints expired command; replacing the only manual file destroys prior-day selection')


def alert_receipt(directory):
    from bts.health import alert

    path = directory / 'status.json'
    day = datetime(2027, 12, 31, 23, 59, tzinfo=ET).date()
    unknown = alert.Alert('CRITICAL', 'late_delivery', 'DELIVERY OUTCOME UNKNOWN')
    late = alert.Alert('CRITICAL', 'late_delivery', 'Unenterable when delivered')
    other = alert.Alert('CRITICAL', 'other', 'Other incident')

    def dispatch(item):
        return alert.dispatch_dm_for_health_alerts([item], 'operator.example',
                                                   status_path=path, now_et_date=day)

    with patch.object(alert, 'send_dm', side_effect=['late-id', RuntimeError('fake failure'), None, 'unknown-id']) as send:
        assert dispatch(late) and not dispatch(unknown)
        assert dispatch(other) and json.loads(path.read_text())['status'] == 'failed'
        assert dispatch(unknown) and json.loads(path.read_text())['status'] == 'sent'
        assert not dispatch(unknown) and send.call_count == 3
    path.unlink()
    with patch.object(alert, 'send_dm', side_effect=['first-id', RuntimeError('fake failure'), 'second-id']):
        assert dispatch(unknown) and dispatch(other) and dispatch(unknown)
    print('C2: same-source late alert suppresses unknown; null reply suppresses retry; unrelated failure loses acknowledged dedup')


def replay(directory):
    from bts import picks
    from bts.daily_decision import decision_path, is_scoreable_commit, write_decision
    from tests.test_pick_chosen_on_cs5a import daily_pick

    past, today = '2027-07-15', '2027-07-16'
    daily = daily_pick()
    daily.result = 'hit'
    path = picks.save_pick(daily, directory)
    assert not is_scoreable_commit(past, directory, daily)
    assert picks._replay_season_streak(directory, 2027, today) == (1, True)
    write_decision(past, directory, action='single', source='test',
                   delivery_status='private_locked', scoreable=True)
    assert is_scoreable_commit(past, directory, daily)
    decision_path(past, directory).write_text('{}')
    assert not is_scoreable_commit(past, directory, daily)
    assert picks._replay_season_streak(directory, 2027, today) == (1, True)
    path.write_text('{"result":"miss"}')
    assert picks._replay_season_streak(directory, 2027, today) == (0, True)
    path.write_text('[]')
    try:
        picks._replay_season_streak(directory, 2027, today)
    except AttributeError:
        pass
    else:
        raise AssertionError('A non-object unexpectedly returned a replay result')
    daily.result = None
    picks.save_pick(daily, directory)
    for tag in (2026, None, 2027):
        doc = {'streak': 9, 'applied_dates': [past]}
        if tag is not None:
            doc['season'] = tag
        (directory / 'streak.json').write_text(json.dumps(doc))
        assert picks._replay_season_streak(directory, 2027, past) is None
        assert picks.load_applied_dates(directory, season=2027) == ([past] if tag == 2027 else [])
    print('C3: legacy terminal hit counts without flags; decision parser conflates damage and absence; non-object raises; scoped marker drops tagless control')


def refusal_keys(directory):
    from bts import contest_state, picks, scheduler
    from bts.health import alert
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, record, set_dated
    from tests.test_saved_pick_paths_cs5a import scheduler_state

    config = {'orchestrator': {'picks_dir': str(directory)}, 'bluesky': {'dm_recipient': 'operator.example'},
              'scheduler': {'pick_delivery': 'dm', 'contest_state_required': True}}
    set_dated(directory)
    daily = daily_pick()
    daily.chosen_on = record()
    picks.save_pick(daily, directory)
    path = directory / 'account_state/contest_streak.manual.json'
    status = directory.parent / 'health_state/health_dm_delivery_status.json'
    dispatch = alert.dispatch_dm_for_health_alerts

    def share_or_split(items, *args, **kwargs):
        if split and items[0].message.startswith(f'Saved pick for {DAY}'):
            items = [alert.Alert(items[0].level, items[0].source, items[0].message,
                                 incident_key='contest_state:saved_pick')]
        return dispatch(items, *args, now_et_date=NOW.date(), **kwargs)

    for split in (False, True):
        path.write_text('{corrupt correction')
        sent = Mock(return_value='alert-id')
        with patch.object(alert, 'dispatch_dm_for_health_alerts', share_or_split), \
             patch.object(alert, 'send_dm', sent), \
             patch.object(scheduler, '_now_et', return_value=NOW), \
             patch.object(scheduler, '_void_committed_games', return_value={}), \
             patch.object(scheduler, 'count_new_confirmations', return_value=2), \
             patch('bts.orchestrator.run_and_pick', side_effect=lambda *a, **k: (
                 contest_state.load_decision_streak_state(directory, now=NOW, require_contest_state=True))):
            result = scheduler.run_single_check(DAY, [101], set(), config, 0.03)
            assert result['pick_result'] is None and sent.call_count == 1
            set_dated(directory, streak=8)
            real_load = contest_state.load_decision_streak_state
            with patch.object(contest_state, 'load_decision_streak_state', side_effect=lambda *a, **k: real_load(*a, **k, now=NOW)):
                assert not scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, 'final fallback')
            assert sent.call_count == (2 if split else 1)
        status.unlink()
    print('C4: splitting the key double-pages one dated correction episode')


if __name__ == '__main__':
    def deny_network(*args, **kwargs):
        raise AssertionError('Network is forbidden during a contract probe')

    with patch.object(socket, 'socket', deny_network), patch.object(socket, 'create_connection', deny_network):
        for probe in (health_clock, alert_receipt, replay, refusal_keys):
            with TemporaryDirectory(dir=HERE) as scratch:
                probe(Path(scratch) / 'picks')
