from contextlib import ExitStack, redirect_stderr
from datetime import date, datetime, timedelta
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import cli, contest_fetch, contest_state, dm, picks, saver_state, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick
from tests.test_stale_delivery_cs4c import envelope
from tests.test_morning_state_cs6b1 import warning


def now(day, time):
    return datetime.fromisoformat(f'{day}T{time}').replace(tzinfo=walk.ET)


def state(day):
    return scheduler.SchedulerState(day, 'pinned', [], [], [], False, None, None, None)


def invoke(path, *command, code=0):
    result, out = walk._invoke(*command, '--picks-dir', path)
    assert result == code, (command, result, out)
    print('CLI', ' '.join(map(str, command)), 'exit', result,
          '|', ' | '.join(line for line in out.strip().splitlines() if line))
    return out


def dated(path, day, streak=11, flag='used', code=0):
    return invoke(path, 'set-contest-streak', '--date', day, '--streak', streak,
                  '--best-streak', 14, '--saver', flag, code=code)


def live(path):
    return contest_state.load_decision_streak_state(path, require_contest_state=True)


def cfg(path):
    return walk._config(path) | {'bluesky': {'dm_recipient': 'capture.invalid'}}


DAY = '2027-07-15'
start = now(DAY, '10:00')
work = Path(__file__).resolve().parent
with TemporaryDirectory(dir=work) as temp, walk.offline(start), ExitStack() as stack:
    root = Path(temp)
    capture = []
    stack.enter_context(patch.object(alert, 'send_dm', side_effect=lambda _, body: capture.append(body) or 'alert-id'))
    stack.enter_context(patch.object(scheduler, '_trigger_live_forward_capture_on_lock'))
    stack.enter_context(patch.object(scheduler, '_delivery_game_statuses', return_value={}))
    for flag in ('active', 'not_earned', 'used', 'uninitialized'):
        path = root / flag / 'scheduler picks'
        path.mkdir(parents=True)
        assert walk.install_state(path, start.date(), walk.State('automatic', 12, 14, 'active', 0.9)) == []
        walk._set_clock(start)
        auto = path / 'account_state/contest_streak.json'
        stamp = '0001-01-01T00:00:00.' + '0' * 102 + '+00:00'
        auto.write_text(json.dumps(json.loads(auto.read_text()) | {'recorded_at': stamp}))
        if flag == 'uninitialized':
            (auto.parent / 'saver_state.json').unlink()
        else:
            invoke(path, 'saver-state', '--init', flag, '--force', '--season', 2027)
        record = contest_state.automatic_reading_record(live(path), start)
        detail = f'last accepted fetch {stamp}, about {record["age_hours"]:.2f} hours ago'
        scheduler._morning_state_check(cfg(path), path, DAY, start, start + timedelta(hours=4))
        assert capture[-1] == warning(detail)
        assert len(capture[-1]) == 895
        before = len(capture)
        with redirect_stderr(StringIO()) as journal:
            scheduler._warn_stale_contest_reading(record, cfg(path), path)
        assert len(capture) == before
        expected_log = '  STALE CONTEST READING — ' + envelope(detail=detail).split('- [contest_state] ', 1)[1]
        assert journal.getvalue().strip('\n') == expected_log
        print('C8/C9 morning text 895 characters:', capture[-1])
        invoke(path, 'season-contract', 'status')
        walk._set_clock(now(DAY, '10:05'))
        dated(path, DAY)
        assert live(path).observation_kind == 'dated' and live(path).streak == 11
        assert f'saver_state: {flag}' in invoke(path, 'saver-state')
        if flag in ('active', 'not_earned'):
            invoke(path, 'saver-state', '--use')
        elif flag == 'uninitialized':
            invoke(path, 'saver-state', '--init', 'used', '--season', 2027)
        assert 'saver_state: used' in invoke(path, 'saver-state')
        wrong = root / 'wrong' / 'picks'
        saved = os.getcwd()
        try:
            os.chdir(root)
            assert 'saver_state: used' in invoke(path, 'saver-state')
        finally:
            os.chdir(saved)
        assert 'uninitialized' in invoke(wrong, 'saver-state')
        assert live(path).saver_flag_state == 'used'
        walk._set_clock(now('2027-07-16', '10:00'))
        assert live(path).observation_kind == 'automatic' and not live(path).saver_available
        print('C5 on next automatic day', flag, '->', live(path).saver_flag_state)

    for phase in ('before_guard', 'during_send', 'after_delivery'):
        path = root / phase / 'picks'
        path.mkdir(parents=True)
        assert walk.install_state(path, start.date(), walk.State('automatic', 12, 14, 'active', 0.9)) == []
        clock = now(DAY, '18:35')
        walk._set_clock(clock)
        daily = daily_pick()
        daily.chosen_on = picks.chosen_on_record(live(path), required=True)
        if phase == 'before_guard':
            dated(path, DAY)
        delivery = []
        def send(_, body):
            delivery.append(body)
            assert picks.load_pick(DAY, path).delivery_attempted
            if phase == 'during_send':
                dated(path, DAY)
            return 'pick-id'
        with patch.object(dm, 'send_dm', side_effect=send):
            outcome = scheduler._deliver_and_lock_pick(daily, cfg(path), path, state(DAY), DAY, phase)
        assert outcome is (phase != 'before_guard')
        if outcome:
            assert delivery[0].endswith('Streak: 12')
            frozen = picks.load_pick(DAY, path)
            if phase == 'after_delivery':
                assert len(capture) and 'official' not in capture[-1]
                dated(path, DAY)
            assert live(path).streak == 11 and picks.load_pick(DAY, path) == frozen
            with patch.object(dm, 'send_dm', side_effect=AssertionError('already sent')):
                assert scheduler._deliver_and_lock_pick(frozen, cfg(path), path, state(DAY), DAY, 'restart')
        print('C9', phase, 'delivered=', outcome, 'transport=', delivery)

    path = root / 'midnight' / 'picks'
    path.mkdir(parents=True)
    assert walk.install_state(path, start.date(), walk.State('automatic', 12, 14, 'active', 0.9)) == []
    sent_day = now(DAY, '18:35')
    live_at = now('2027-07-16', '00:05')
    record = {'stale': True, 'reason': 'old',
              'recorded_at': '0001-01-01T00:00:00.' + '0' * 102 + '+00:00', 'age_hours': 17763800.0}
    walk._set_clock(sent_day)
    scheduler._warn_stale_contest_reading(record, cfg(path), path)
    expected = envelope(detail=f'last accepted fetch {record["recorded_at"]}, about 17763800.00 hours ago')
    assert capture[-1] == expected and len(expected) == 875
    print('C6 delivery text', len(expected), 'characters:', expected)
    walk._set_clock(live_at)
    dated(path, DAY, code=2)
    assert not (path / 'account_state/contest_streak.manual.json').exists()
    assert 'streak=12' in invoke(path, 'season-contract', 'status')
    print('00:05 prior-slate game live; 00:20 grading pending: no re-date, no new observation')
    walk._set_clock(now('2027-07-16', '09:00'))
    assert 'streak=12' in invoke(path, 'season-contract', 'status')
    dated(path, '2027-07-16', streak=13, flag='active')
    assert live(path).streak == 13 and live(path).saver_available

    path = root / 'year' / 'picks'
    walk._set_clock(now('2027-12-31', '23:55'))
    invoke(path, 'saver-state', '--init', 'active', '--season', 2027)
    walk._set_clock(now('2028-01-01', '00:05'))
    dated(path, '2027-12-31', code=2)
    assert 'uninitialized' in invoke(path, 'saver-state')
    assert 'season 2027' in invoke(path, 'saver-state', '--season', 2027)
    invoke(path, 'saver-state', '--use', code=1)

    for kind in ('cutoff', 'postponed'):
        path = root / kind / 'picks'
        daily = daily_pick()
        cutoff = picks.submission_cutoff_et(daily)
        walk._set_clock(cutoff)
        picks.save_pick(daily, path)
        expected = (f'LATE DELIVERY REFUSED: Steven Kwan would have been sent 0 min after the {cutoff:%H:%M} ET cutoff.'
                    if kind == 'cutoff' else f'DELIVERY REFUSED for {DAY}: game 101 is Postponed.')
        expected += (' This pick was withheld. A later check may produce another recommendation.'
                     ' Use a later official pick message only while the entry window is open.'
                     ' A manual entry can differ from a later recommendation.'
                     ' The entry check may report a mismatch if it verifies the entry in time.')
        statuses = {101: {'abstract': 'P', 'detailed': 'Postponed'}} if kind == 'postponed' else {}
        with patch.object(scheduler, '_delivery_game_statuses', return_value=statuses):
            assert not scheduler._deliver_and_lock_pick(daily, cfg(path), path, state(DAY), DAY, kind)
        assert capture[-1].endswith(expected), capture[-1]
        refusal = scheduler.load_state(DAY, path).delivery_refusals[0]
        assert (path / DAY / refusal['archive']).is_file() and picks.load_pick(DAY, path) is None
        print('C7', kind, capture[-1], refusal)
        invoke(path, 'check-pick-entered', '--now-et', f'{DAY}T19:04')

    path = root / 'future' / 'picks'
    path.mkdir(parents=True)
    assert walk.install_state(path, start.date(), walk.State('automatic', 12, 14, 'active', 0.9)) == []
    walk._set_clock(now(DAY, '08:45'))
    scheduler._morning_state_check(cfg(path), path, DAY, now(DAY, '08:45'), start)
    assert 'is in the future' in capture[-1]
    status = json.loads((path.parent / 'health_state/contest_streak_fetch_status.json').read_text())
    assert status['last_error'] is None
    print('B8a fetch diagnosis after clock correction:', status, capture[-1])
