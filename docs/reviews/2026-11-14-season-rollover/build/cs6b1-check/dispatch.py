import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from timeline import H, S, at, games
from atproto_client.models.chat.bsky.convo.defs import MessageInput
from bts import contest_state, dm, orchestrator, posting, scheduler
from bts.health import alert
from bts.season_contract import live_automatic_refusal


def deny_transport(*args, **kwargs):
    raise AssertionError('Unexpected boundary call')


with ExitStack() as stack:
    for module, attr in [(dm, 'send_dm'), (posting, 'post_to_bluesky'),
                         (orchestrator, 'run_cascade'), (scheduler, 'fetch_schedule')]:
        stack.enter_context(patch.object(module, attr, deny_transport))
    tmp = stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent))
    root = Path(tmp)
    date = at('2027-07-15', '10:00').date()
    now = at(date.isoformat(), '10:00')
    morning = root / 'contest_morning_check_dm_delivery_status.json'
    old_path = root / H
    stale_path = root / S
    old_path.write_text('legacy bytes')
    morning_alert = [alert.Alert('CRITICAL', 'contest_state', 'morning refusal', 'contest_state:morning')]
    stale_alert = [alert.Alert('WARN', 'contest_state', 'morning stale', 'contest_state:stale_reading')]
    bodies = []
    fail = True

    def capture(recipient, body):
        bodies.append(body)
        if fail:
            raise RuntimeError('offline simulated failure')
        return 'captured'

    with patch.object(alert, 'send_dm', capture):
        assert alert.dispatch_dm_for_health_alerts(morning_alert, 'offline.test',
                                                   status_path=morning, now_et_date=date)
        assert json.loads(morning.read_text())['status'] == 'failed'
        fail = False
        assert alert.dispatch_dm_for_health_alerts(morning_alert, 'offline.test',
                                                   status_path=morning, now_et_date=date)
        assert bodies[-1].startswith('⚠️ Previous health alert DM FAILED to send (1 CRITICAL)')
        assert not alert.dispatch_dm_for_health_alerts(morning_alert, 'offline.test',
                                                       status_path=morning, now_et_date=date)
        print('Critical first failure and restart retry:', bodies)
        print('saved keys:', json.loads(morning.read_text())['sent_sources'])
        assert old_path.read_text() == 'legacy bytes'

        fail = True
        assert alert.dispatch_dm_for_health_alerts([], 'offline.test', warn_attention=stale_alert,
                                                   status_path=stale_path, now_et_date=date)
        assert json.loads(stale_path.read_text())['status'] == 'failed'
        fail = False
        record = {'stale': True, 'reason': 'old', 'age_hours': 30,
                  'recorded_at': '2027-07-14T04:00:00-04:00'}
        with patch.object(scheduler, '_now_et', lambda: now):
            health_root = root / 'fixture'
            (health_root / 'health_state').mkdir(parents=True)
            actual_status = health_root / 'health_state' / S
            actual_status.write_text(stale_path.read_text())
            config = {'bluesky': {'dm_recipient': 'offline.test'}}
            scheduler._warn_stale_contest_reading(record, config, health_root / 'picks')
            assert bodies[-1].startswith('BTS health WARN attention:\n- [contest_state] The pick was delivered')
            count = len(bodies)
            scheduler._warn_stale_contest_reading(record, config, health_root / 'picks')
            assert len(bodies) == count
            print('Warn-only failure retried at real delivery-warning call:', bodies[-1])
            print('saved keys:', json.loads(actual_status.read_text())['sent_sources'])
        assert old_path.read_text() == 'legacy bytes'

    for label, path in [('normal', Path('data/picks')), ('deep', root.joinpath(*['x' * 250] * 8))]:
        error = live_automatic_refusal(path, now, None) if label == 'normal' else None
        if error is None:
            # Generate the same stock absent-switch refusal without requiring a deep directory to exist.
            with patch('bts.season_contract.switch_state') as switch:
                switch.return_value.state = 'absent'
                error = live_automatic_refusal(path, now, None)
        message = (f'Morning check for {date}: no pick can be made yet. {error} '
                   'Put this right before the first lineup check at 18:10 ET.')
        alert_body = alert.format_health_dm_body([alert.Alert('CRITICAL', 'contest_state', message)])
        try:
            MessageInput(text=alert_body)
            result = 'accepted by installed client length validation'
        except ValueError:
            result = 'rejected by installed client length validation'
        print(label, len(alert_body), len(alert_body.encode()), result)

    for label, start, starts, doubleheader in [
        ('night', '10:00:30', ('19:10', '19:40', '22:10'), False),
        ('getaway', '10:00:30', ('13:05', '13:10', '13:15'), False),
        ('thin', '10:00:30', ('13:05',), False), ('early', '05:10:30', ('06:10',), False),
        ('opening', '10:00:30', ('19:10',), False),
        ('doubleheader', '10:00:30', ('13:05', '18:40'), True),
    ]:
        slate = games('2027-03-24', starts)
        if doubleheader:
            slate[1]['teams'] = slate[0]['teams']
        now = at('2027-03-24', start)
        planned = scheduler.plan_run_times(slate, offset_min=60, now=now)
        print(label, 'production offset=60', [(r['time_et'].strftime('%H:%M'),
              int((r['time_et'] - now).total_seconds())) for r in planned], scheduler.detect_doubleheader_game2s(slate))

    evening = at('2027-03-23', '23:50')
    with patch.object(scheduler, '_now_et', lambda: evening), \
            patch.object(scheduler, 'fetch_schedule', lambda _: []):
        assert scheduler._next_day_wakeup('2027-03-23', {}) == at('2027-03-24', '10:00')
    later = at('2027-03-24', '05:11')
    with patch.object(scheduler, '_now_et', lambda: later), \
            patch.object(scheduler, 'fetch_schedule', lambda day: games(day, ('06:10',))):
        assert scheduler._next_day_wakeup('2027-03-23', {}) == at('2027-03-24', '05:12')
    print('empty lookahead: next day 10:00; late lookahead with real games: now + 1 minute')
