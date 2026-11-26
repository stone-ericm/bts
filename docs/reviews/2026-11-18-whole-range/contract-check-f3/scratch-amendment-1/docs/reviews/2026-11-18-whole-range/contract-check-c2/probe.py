"""Offline evidence for C2 on 269ae2b5. Run with the repository venv."""

from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVIEW = ROOT / 'docs/reviews/2026-11-18-whole-range'
sys.path[:0] = [str(ROOT / 'src'), str(REVIEW / 'slice-1-codex'),
                str(REVIEW / 'slice-1-carbon')]

import opening_edges as edges
import opening_rehearsal as opening
from bts import cli, contest_state, dm, picks, scheduler
from bts.daily_decision import decision_path, load_decision
from bts.health import alert, post_failure


def unknown(messages):
    return [body for body in messages if 'DELIVERY OUTCOME UNKNOWN' in body]


def rehearsal(directory, phase):
    season = opening.Season(directory)
    opening.write_json(season.account_dir / 'contest_streak.json', {
        'schema_version': 'bts_contest_streak_auto_v1', 'season': 2026,
        'active_streak': 17, 'best_streak': 26, 'saver_available': False,
        'source_date': opening.OLD_DAY, 'recorded_at': '2026-10-21T14:00:00Z',
        'source': 'manual_cli',
    })
    with season.run_offline():
        season.restore()
        result = edges.restart_at_send(season, phase)
        day = result['snapshot']['day']
        messages = [m['body'] for m in result['messages']]
        assert not unknown(messages)
        assert len(edges.picks_sent(season)) == int(phase in {'after_remote_accept', 'after_checkpoint'})
        if phase == 'after_checkpoint':
            assert result['next_morning']['state'] == 'entered'
        else:
            assert any('Delivery of this pick was NOT confirmed' in body for body in messages)
            assert result['snapshot']['decision']['scoreable']
            assert result['next_morning']['state'] == 'not_entered'
        print('rehearsal', phase, 'pick sends', len(edges.picks_sent(season)),
              'unknown DMs', len(unknown(messages)), 'decision',
              result['snapshot']['decision']['delivery_status'], 'next morning', result['next_morning']['state'])
        if phase == 'corrected_after_marker':
            pick = result['snapshot']['pick']
            assert pick['chosen_on']['streak'] == 1 and pick['double_down'] is not None
            control = opening.Season(directory / 'control')
            control.root.mkdir()
            control.write_config('dm', 'enter', True)
            with control.run_offline():
                corrected = edges.correction_between_reads(control)
            current = corrected['snapshot']['pick']
            assert current['chosen_on']['streak'] == 46 and current['double_down'] is None
            print('correction', 'attempted double at streak 1, current-state control single at 46',
                  'primary', pick['pick']['batter_name'], current['pick']['batter_name'])
        return result


def daily_for(day, batter='Example Batter'):
    game_time = opening.at(day, '19:00').astimezone(opening.timezone.utc).isoformat()
    primary = picks.Pick(batter, 1, 'EX', 1, 'Opponent', 2, .83, [], False, 101, game_time)
    return picks.DailyPick(day, opening.at(day, '18:15').isoformat(), primary, None, None)


def config_at(directory, mode='dm', recipient='operator.example'):
    return {'orchestrator': {'picks_dir': str(directory), 'heartbeat_path': str(directory.parent / 'beat')},
            'scheduler': {'pick_delivery': mode, 'entry_intent': 'research' if mode == 'private' else 'enter',
                          'contest_state_required': mode != 'private', 'live_forward_capture_on_lock': False},
            'bluesky': {'dm_recipient': recipient}, 'health_checks': {'enabled': False}}


def state_for(day):
    return scheduler.SchedulerState(day, opening.at(day, '18:15').isoformat(), [], [], [],
                                    False, None, None, None)


def direct(daily, config, *, expect=False):
    directory = Path(config['orchestrator']['picks_dir'])
    picks.save_pick(daily, directory)
    state = state_for(daily.date)
    assert scheduler._deliver_and_lock_pick(daily, config, directory, state, daily.date, 'probe') is expect
    return state


def render_cases(directory, corrected):
    day = '2027-03-25'
    for label, when in [
        ('never_left', opening.at(day, '18:20')),
        ('accepted_and_entered', opening.at(day, '18:20')),
        ('after_cutoff_before_pitch', opening.at(day, '18:56')),
        ('after_first_pitch', opening.at(day, '19:04')),
        ('after_midnight', opening.at('2027-03-26', '00:01')),
        ('new_year', opening.at('2028-01-01', '00:01')),
        ('private_fresh', opening.at(day, '18:20')),
        ('private_after_mode_change', opening.at(day, '18:20')),
    ]:
        slate = '2027-12-31' if label == 'new_year' else day
        daily = daily_for(slate)
        before_cutoff = when < picks.submission_cutoff_et(daily)
        entries = {slate: daily.pick.batter_id} if label == 'accepted_and_entered' else {}
        private = label.startswith('private')
        config = config_at(directory / label / 'picks', 'private' if private else 'dm')
        messages = Mock(return_value='health-ack')
        pick_send = Mock(return_value='pick-ack')
        clock = opening.Clock(when)
        with opening.sandbox(clock, directory), patch.object(alert, 'send_dm', messages), \
                patch.object(dm, 'send_dm', pick_send), patch.object(scheduler, '_void_committed_games', return_value={}):
            if label in {'never_left', 'accepted_and_entered', 'private_after_mode_change'}:
                class InterruptedPick(BaseException):
                    pass

                accepted = []

                def interrupt(recipient, body):
                    if label == 'accepted_and_entered':
                        accepted.append(body)
                    raise InterruptedPick

                with patch.object(dm, 'send_dm', side_effect=interrupt), \
                        patch.object(contest_state, 'load_decision_streak_state', return_value=Mock(streak=0)), \
                        patch.object(scheduler, '_contest_state_required', return_value=False):
                    try:
                        direct(daily, config_at(Path(config['orchestrator']['picks_dir'])))
                    except InterruptedPick:
                        pass
                    else:
                        raise AssertionError('did not reach pick send')
                assert bool(accepted) == bool(entries)
                daily = picks.load_pick(slate, Path(config['orchestrator']['picks_dir']))
            if label == 'private_fresh':
                assert direct(daily, config, expect=True).pick_locked
                assert not daily.delivery_attempted and not messages.called
            else:
                daily.delivery_attempted = True
                assert direct(daily, config).pick_locked
                assert len(unknown([messages.call_args.args[1]])) == 1
            pick_send.assert_not_called()
        print('U7', label, 'wall', when.isoformat(), 'before cutoff', before_cutoff,
              'MLB entries', entries, 'DM', repr(messages.call_args.args[1]) if messages.called else 'none')

    raw = corrected['boundary'][0]['pick']
    config = config_at(directory / 'corrected' / 'picks')
    pick_dir = Path(config['orchestrator']['picks_dir'])
    pick_dir.mkdir(parents=True)
    (pick_dir / f'{day}.json').write_text(json.dumps(raw))
    opening.write_json(pick_dir / 'account_state/contest_streak.manual.json',
                       corrected['snapshot']['account']['contest_streak.manual.json'])
    messages = Mock(return_value='health-ack')
    with opening.sandbox(opening.Clock(opening.at(day, '15:26')), directory), \
            patch.object(alert, 'send_dm', messages):
        choice = contest_state.load_decision_streak_state(pick_dir, require_contest_state=True)
        assert choice.streak == 46
        direct(picks.load_pick(day, pick_dir), config)
    print('U7 corrected', 'streak', choice.streak, 'DM', repr(messages.call_args.args[1]))


def receipt_cases(directory):
    day = '2027-03-25'
    now = opening.at(day, '18:55')
    current = opening.Clock(now)
    with opening.sandbox(current, directory):
        for order in [('unknown', 'cutoff'), ('cutoff', 'unknown')]:
            root = directory / order[0]
            dd = daily_for(day)
            dd.delivery_attempted = True
            other = daily_for(day, 'Other Batter')
            config = config_at(root / 'attempted')
            other_config = config_at(root / 'late')
            sent = Mock(return_value='health-ack')
            with patch.object(alert, 'send_dm', sent), patch.object(scheduler, '_void_committed_games', return_value={}):
                actions = {'unknown': lambda: direct(dd, config), 'cutoff': lambda: direct(other, other_config)}
                for route in order:
                    actions[route]()
                assert sent.call_count == 1
            assert order[0] == ('unknown' if 'OUTCOME UNKNOWN' in sent.call_args.args[1] else 'cutoff')
            print('U1 same shared health root', order, 'only sent', order[0])

        shared = directory / 'receipt/picks'
        path = shared.parent / 'health_state/health_dm_delivery_status.json'
        dd = daily_for(day)
        dd.delivery_attempted = True

        def call_unknown():
            return direct(dd, config_at(shared))

        def other_alert(now_day):
            return alert.dispatch_dm_for_health_alerts(
                [alert.Alert('CRITICAL', 'other', 'synthetic health digest')], 'operator.example',
                status_path=path, now_et_date=now_day)

        def result():
            return json.loads(path.read_text())

        for reply in (None, '', '   ', 12, RuntimeError('send failed')):
            sent = Mock(side_effect=[reply, 'ack'])
            with patch.object(alert, 'send_dm', sent):
                call_unknown()
                transport_status = result()['status']
                call_unknown()
            assert transport_status == ('failed' if isinstance(reply, Exception) else 'sent')
            assert sent.call_count == (2 if isinstance(reply, Exception) else 1)
            print('U4 reply', repr(reply), 'status', transport_status, 'transport calls', sent.call_count)
            path.unlink()

        for disturbance in ['failed health', 'other-slate EOD', 'midnight']:
            current.now = now
            second = RuntimeError('unrelated outage') if disturbance == 'failed health' else 'ack-2'
            sent = Mock(side_effect=['ack-1', second, 'ack-3'])
            with patch.object(alert, 'send_dm', sent):
                call_unknown()
                if disturbance == 'failed health':
                    assert other_alert(now.date()) and result()['status'] == 'failed'
                elif disturbance == 'other-slate EOD':
                    assert other_alert(now.date() - timedelta(days=1))
                else:
                    current.move(opening.at('2027-03-26', '00:01'))
                call_unknown()
            assert len(unknown([c.args[1] for c in sent.call_args_list])) == 2
            print('U1 repeat after', disturbance, 'receipt day', result()['sent_day'])
            path.unlink()

        for kind in ['directory', 'fifo', 'unreadable', 'array', 'null', 'truncated', 'link', 'blocked_parent']:
            subdir = directory / kind
            status = subdir / 'health_state/health_dm_delivery_status.json'
            status.parent.mkdir(parents=True)
            linked = subdir / 'shared-target.json'
            if kind == 'directory':
                status.mkdir()
            elif kind == 'fifo':
                os.mkfifo(status)
            elif kind in {'unreadable', 'array', 'null', 'truncated'}:
                status.write_text({'unreadable': '{}', 'array': '[]', 'null': 'null', 'truncated': '{'}[kind])
                if kind == 'unreadable':
                    status.chmod(0)
            elif kind == 'link':
                linked.write_text('[]')
                status.symlink_to(linked)
            else:
                status.parent.rmdir()
                status.parent.write_text('not a directory')
            sent = Mock(return_value='ack')
            with patch.object(alert, 'send_dm', sent):
                direct(dd, config_at(subdir / 'picks'))
            assert sent.call_count == 1
            if kind == 'link':
                assert json.loads(linked.read_text())['status'] == 'sent'
            print('receipt injury', kind, 'DM sent', sent.call_count, 'linked target changed', kind == 'link')

        for failure in ('dispatcher', 'formatter', 'status_write'):
            config = config_at(directory / failure / 'picks')
            sent = Mock(return_value='ack')
            target = {'dispatcher': 'dispatch_dm_for_health_alerts', 'formatter': 'format_health_dm_body',
                      'status_write': '_write_regular_file_text'}[failure]
            with patch.object(alert, 'send_dm', sent), patch.object(alert, target, side_effect=OSError('line one\nline two')):
                try:
                    direct(dd, config)
                except OSError:
                    assert failure != 'status_write'
                    escaped = True
                else:
                    assert failure == 'status_write'
                    escaped = False
            assert load_decision(day, Path(config['orchestrator']['picks_dir']))['scoreable']
            print('U5', failure, 'escaped', escaped, 'DM sent', sent.call_count)


def run_day_cases(root):
    for day, after in [('2027-03-25', '2027-03-26'), ('2027-12-31', '2028-01-01')]:
        directory = root / day
        directory.mkdir()
        clock = opening.Clock(opening.at(day, '18:20'))
        config = config_at(directory / 'picks')
        pick_dir = Path(config['orchestrator']['picks_dir'])
        dd = daily_for(day)
        sent = Mock(return_value='alert-ack')
        pick_sent = Mock(return_value=None)
        game = {'gamePk': 101, 'gameDate': dd.pick.game_time,
                'status': {'abstractGameCode': 'P', 'detailedState': 'Scheduled'},
                'teams': {'away': {'team': {'name': 'Away'}}, 'home': {'team': {'name': 'Home'}}}}
        poll = Mock(return_value='final')
        with opening.sandbox(clock, directory), patch.object(dm, 'send_dm', pick_sent), \
                patch.object(alert, 'send_dm', sent), patch.object(scheduler, 'fetch_schedule', side_effect=lambda d: [game] if d == day else []), \
                patch.object(scheduler, '_void_committed_games', return_value={}), \
                patch.object(scheduler, 'count_new_confirmations', return_value=0), \
                patch.object(scheduler, 'notify_watchdog'), patch.object(scheduler, 'notify_ready'), \
                patch.object(scheduler, '_idle_until_next_wakeup'), \
                patch.object(scheduler, '_watchdog_ping_sleep', clock.sleep), \
                patch.object(scheduler, 'run_result_polling', poll), \
                patch.object(scheduler, '_contest_state_required', return_value=False), \
                patch.object(contest_state, 'load_decision_streak_state', return_value=Mock(streak=0)), \
                patch('bts.orchestrator.run_and_pick', side_effect=AssertionError('cascade ran')):
            direct(dd, config, expect=True)
            recorded = picks.load_pick(day, pick_dir)
            assert recorded.delivery_attempted and not picks.pick_was_delivered(recorded)
            dec_path = decision_path(day, pick_dir)
            original = dec_path.read_bytes()
            assert load_decision(day, pick_dir)['delivery_status'] == 'delivered'
            scheduler.run_day(day, config)
            poll.assert_called_once()
            assert dec_path.read_bytes() == original and not sent.called
            recorded.result = 'hit'
            picks.save_pick(recorded, pick_dir)
            terminal_bytes = (pick_dir / f'{day}.json').read_bytes()
            # EOD post_failure is an independent CRITICAL for this same pick after 22:00.
            assert post_failure.check(pick_dir, today=opening.at(day, '22:00').date(), now=opening.at(day, '22:00'))[0].source == 'pick_delivery'
            clock.move(opening.at(after, '00:01'))
            assert cli._today_et() == after
            cli_config = opening.Season(directory)
            cli_config.write_config('dm', 'enter', True)
            restart = Mock(wraps=scheduler.run_day)
            with patch.object(scheduler, 'run_day', restart), \
                    patch('bts.orchestrator.load_config', return_value=config):
                opening.invoke('schedule', '--config', cli_config.config)
                opening.invoke('schedule', '--config', cli_config.config, '--date', day)
            assert [c.kwargs['date'] for c in restart.call_args_list] == [after, day]
            assert dec_path.read_bytes() == original
            assert (pick_dir / f'{day}.json').read_bytes() == terminal_bytes and not sent.called
            assert scheduler.load_state(day, pick_dir).committed_pick_written
            poll.assert_called_once()
            pick_sent.assert_called_once()
        print('U3', day, 'pick DM None -> scoreable delivered decision + unconfirmed marker;',
              'first restart polls; terminal restart skips polling; same decision bytes; no new DM;',
              'CLI default selected', after, 'explicit selected', day)


def main():
    import test_slice1_probes_v3 as carbon

    with TemporaryDirectory(dir=HERE) as tmp, redirect_stderr(StringIO()):
        root = Path(tmp)
        run = subprocess.run
        with patch.object(subprocess, 'run', side_effect=lambda *a, **k: run(*a, timeout=10, **k)):
            carbon.test_probe_2_midsend_restart_silently_locks_without_critical_dm(root)
        evidence = carbon.EVIDENCE['probe_2_midsend_restart']
        print('Carbon 2', 'DMs', evidence['dms_sent_all_day'], 'decision',
              evidence['decision_json']['delivery_status'], 'model streak', evidence['streak'])
        opening.REPO = ROOT
        for phase in ['before_remote_accept', 'after_remote_accept', 'corrected_after_marker', 'after_checkpoint']:
            directory = root / phase
            directory.mkdir()
            replay = rehearsal(directory, phase)
            if phase == 'corrected_after_marker':
                corrected = replay
        for name, probe in [('render', lambda p: render_cases(p, corrected)),
                            ('receipts', receipt_cases), ('run_day', run_day_cases)]:
            directory = root / name
            directory.mkdir()
            probe(directory)


if __name__ == '__main__':
    signal.alarm(45)
    main()
    signal.alarm(0)
