from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from bts import cli, contest_state, dm, orchestrator, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_cli_integration import TestCheckPickEntered
from tests.test_pick_chosen_on_cs5a import daily_pick


def clock(day, time):
    return datetime.fromisoformat(f'{day}T{time}').replace(tzinfo=walk.ET)


def command(path, *args, code=0):
    rc, out = walk._invoke(*args, '--picks-dir', path)
    assert rc == code, (args, rc, out)
    return out


def dated(path, day, streak, saver):
    return command(path, 'set-contest-streak', '--date', day, '--streak', streak,
                   '--best-streak', 14, '--saver', saver)


def fresh_dir(root, label):
    path = root / label / 'picks'
    path.mkdir(parents=True)
    assert walk.install_state(path, start.date(), walk.State('automatic', 12, 14, 'active', .99)) == []
    return path


def state(day):
    return scheduler.SchedulerState(day, 'pinned', [], [], [], False, None, None, None)


def config(path):
    return walk._config(path) | {'bluesky': {'dm_recipient': 'capture.invalid'}}


DAY = '2027-07-15'
NEXT = '2027-07-16'
start = clock(DAY, '10:00')
work = Path(__file__).resolve().parent
with TemporaryDirectory(dir=work) as temp, walk.offline(start), ExitStack() as stack:
    root = Path(temp)
    stack.enter_context(patch.object(scheduler, '_trigger_live_forward_capture_on_lock'))
    stack.enter_context(patch.object(scheduler, '_delivery_game_statuses', return_value={}))
    captured = []
    stack.enter_context(patch.object(alert, 'send_dm', side_effect=lambda _, body: captured.append(body) or 'alert-id'))

    path = fresh_dir(root, 'reading')
    walk._set_clock(clock(DAY, '18:35'))
    dated(path, DAY, 3, 'used')
    status = command(path, 'season-contract', 'status')
    assert 'manual: dated date=2027-07-15 streak=3 best=14 saver=used in effect=True' in status
    assert 'automatic path alone: accepted (streak 12' in status
    assert 'saver: active; season: 2027' in status and 'streak=3; status=fresh; saver=used' in status
    print('B8a conflict for MLB streak 12, best 14, saver active: matches automatic/flag, differs from live dated.')
    print(status)
    state_read = contest_state.load_decision_streak_state(path, require_contest_state=True)
    assert state_read.streak == 3 and not state_read.saver_available

    frame = walk._predictions(start.date(), .999)
    statuses = {int(pk): {'abstract': 'F', 'detailed': 'Final'} for pk in frame.game_pk}
    choose = orchestrator.run_and_pick
    def end_of_selection(*args, **kwargs):
        result = choose(*args, **kwargs)
        assert result[1].decision is None
        dated(path, DAY, 12, 'active')
        return result
    with patch.object(orchestrator, 'run_and_pick', side_effect=end_of_selection), \
            patch.object(orchestrator, 'run_cascade', return_value=(frame, 'pinned')), \
            patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
            patch.object(scheduler, 'count_new_confirmations', return_value=3), \
            patch.object(scheduler, 'heartbeat_watchdog', side_effect=lambda *a, **kw: nullcontext()):
        result = scheduler.run_single_check(DAY, list(statuses), set(), config(path), .03)
    assert result['skip_summary']['streak'] == 12 and result['selection'].decision is None
    print('Only the read after the no-selection result saw 12:', scheduler.format_skip_dm(DAY, result['skip_summary']))

    path = fresh_dir(root, 'midnight')
    walk._set_clock(clock(DAY, '23:59:50'))
    daily = daily_pick()
    daily.pick.game_time = clock(NEXT, '00:35').isoformat()
    daily.chosen_on = picks.chosen_on_record(contest_state.load_decision_streak_state(
        path, require_contest_state=True), required=True)
    def send_across_midnight(*_):
        walk._set_clock(clock(NEXT, '00:00:05'))
        return 'pick-id'
    with patch.object(dm, 'send_dm', side_effect=send_across_midnight):
        assert scheduler._deliver_and_lock_pick(daily, config(path), path, state(DAY), DAY, 'late slate')
    assert daily.date == DAY and daily.delivered_at.startswith(NEXT)
    assert f'--date {NEXT}' in captured[-1] and f"today's ET date both equal {NEXT}" in captured[-1]
    print('Delivery crosses midnight. Pick slate:', daily.date, 'warning:', captured[-1])
    walk._set_clock(clock(NEXT, '00:38'))
    def advance_poll(*_):
        assert picks.load_pick(DAY, path).result is None
        print('Live and grading pending:', daily.date, daily.pick.game_time)
        walk._set_clock(clock(NEXT, '03:10'))
    with patch.object(scheduler, 'poll_game_result', side_effect=['live', 'final']), \
            patch.object(scheduler, '_check_hits_midgame', return_value=[None]), \
            patch.object(scheduler, '_poll_interval_sleep', side_effect=advance_poll), \
            patch.object(picks, 'get_game_statuses_detailed', return_value={}), \
            patch.object(picks, 'resolve_daily_slot_results', return_value={'pick': 'hit'}):
        assert scheduler.run_result_polling(daily.pick.game_pk, DAY, path) == 'final'
    assert picks.load_pick(DAY, path).result == 'hit'
    print('Local grading complete 03:10. MLB now synthetically exposes settled streak 13.')
    walk._set_clock(clock(NEXT, '10:00'))
    assert 'streak=12' in command(path, 'season-contract', 'status')
    assert 'in effect today' in dated(path, NEXT, 13, 'active')
    print('Replacement would permit the correction to 13 on the later slate date:', NEXT)

    for first_status in ('late', 'void'):
        path = fresh_dir(root, first_status)
        initial = daily_pick()
        cutoff = picks.submission_cutoff_et(initial)
        walk._set_clock(cutoff)
        picks.save_pick(initial, path)
        if first_status == 'void':
            statuses = {101: {'abstract': 'P', 'detailed': 'Postponed'}}
        else:
            statuses = {}
        with patch.object(scheduler, '_delivery_game_statuses', return_value=statuses):
            assert not scheduler._deliver_and_lock_pick(initial, config(path), path, state(DAY), DAY, first_status)
        replacement = daily_pick()
        replacement.pick.batter_name = 'Later Batter'
        replacement.pick.batter_id = 2
        replacement.pick.game_pk = 102
        replacement.pick.game_time = clock(DAY, '20:30').isoformat()
        replacement.chosen_on = picks.chosen_on_record(contest_state.load_decision_streak_state(
            path, require_contest_state=True), required=True)
        walk._set_clock(clock(DAY, '19:25'))
        later = []
        with patch.object(dm, 'send_dm', side_effect=lambda _, body: later.append(body) or 'pick-id'):
            assert scheduler._deliver_and_lock_pick(replacement, config(path), path, state(DAY), DAY, 'later check')
        h = TestCheckPickEntered()
        with pytest.MonkeyPatch.context() as mp:
            receipts = h._setup(mp, pending=[h._pending(101)], crosswalk={101: 1, 102: 2})
            mp.setattr(cli, '_fetch_rounds', lambda: {7: start.date()})
            mismatch = h._run(path, f'{DAY}T19:30')
            assert mismatch.exit_code == 1 and 'does NOT match' in receipts[-1][1]
            closed = h._run(path, f'{DAY}T20:25')
            assert closed.exit_code == 0 and 'outside window' in closed.output and len(receipts) == 1
        print('After', first_status, 'later recommendation:', later[0], '|', receipts[-1][1], '|', closed.output.strip())
