import logging
import os
import socket
import sys
import tempfile
from contextlib import nullcontext, redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo


def guard(event, args):
    if event.startswith('socket.'):
        raise AssertionError(f'Network forbidden: {event}')


def forbidden(*args, **kwargs):
    raise AssertionError(f'Unexpected external call: {args!r}')


sys.addaudithook(guard)
for name in ('getaddrinfo', 'getnameinfo', 'gethostbyname', 'gethostbyname_ex',
             'gethostbyaddr', 'create_connection'):
    setattr(socket, name, forbidden)
sys.path.insert(0, str(Path.cwd()))

from click.testing import CliRunner

from bts import cli, contest_state, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert, runner
from bts.health import contest_state as health_contest
from tests.test_pick_chosen_on_cs5a import daily_pick, set_automatic, set_dated


ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
H = 'health_dm_delivery_status.json'
S = 'contest_stale_reading_dm_delivery_status.json'


def at(day, time):
    return datetime.fromisoformat(f'{day}T{time}').replace(tzinfo=ET)


def games(day, starts):
    return [{
        'gamePk': 101+i, 'gameDate': at(day, start).astimezone(UTC).isoformat(),
        'status': {'abstractGameCode': 'P', 'detailedState': 'Scheduled'},
        'teams': {'away': {'team': {'name': f'Away {i}'}},
                  'home': {'team': {'name': f'Home {i}'}}},
    } for i, start in enumerate(starts)]


class Journal:
    def __init__(self, emit):
        self.emit = emit
        self.pending = ''

    def write(self, string):
        self.pending += string
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self.emit('stderr', line)
        return len(string)

    def flush(self):
        pass


def run(root, label, day, start='10:00', starts=('19:10', '19:40', '22:10'),
        seed=None, result_end=None, prior=None, supports_predictions=True):
    print(f'\n=== {label} ===')
    root.mkdir(exist_ok=True)
    directory = root / 'picks'
    now = at(day, start)
    initial = now
    next_day = (now + timedelta(days=1)).date().isoformat()
    slate = games(day, starts)
    next_slate = games(next_day, ('19:10',))
    deliveries = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def emit(kind, line):
        print(f'{now.isoformat(timespec="seconds")} {kind}: {line}')

    def sleep(seconds):
        nonlocal now
        now += timedelta(seconds=seconds)

    def fetch(date):
        emit('schedule', date)
        if date == day:
            return slate
        if date == next_day:
            return next_slate
        raise AssertionError(date)

    def health_dm(recipient, body):
        emit('health DM', (recipient, body))
        deliveries.append(('health', body))
        return 'captured-health'

    def pick_dm(recipient, body):
        emit('pick/skip DM', (recipient, body))
        deliveries.append(('pick', body))
        return 'captured-pick'

    def post(body, *args, **kwargs):
        emit('public post', body)
        deliveries.append(('post', body))
        return 'at://offline/post'

    def polling(*args, **kwargs):
        nonlocal now
        if result_end:
            now = result_end
        emit('results boundary', 'controlled final observation')
        return 'final'

    def cascade(*args, **kwargs):
        emit('prediction', day)
        if not supports_predictions:
            return None, None
        import pandas as pd

        return pd.DataFrame([{
            'batter_id': 680757, 'batter_name': 'Steven Kwan', 'team': 'CLE', 'lineup': 1,
            'pitcher_id': 543037, 'pitcher_name': 'Gerrit Cole', 'opp': 'NYY',
            'p_hit_pa': .24, 'p_game_hit': .99, 'flags': '', 'projected_lineup': False,
            'game_pk': 101, 'game_time': slate[0]['gameDate'],
        }]), 'offline'

    if seed == 'dated':
        set_dated(directory, now=initial.replace(hour=7), streak=7, best=9)
    elif seed in ('fresh', 'stale'):
        accepted = initial - timedelta(hours=20 if seed == 'stale' else 1)
        set_automatic(directory, now=accepted)
        from tests.test_pick_chosen_on_cs5a import account_file

        account_file(directory, 'saver_state.json', {'schema_version': 'bts_saver_state_v1',
                     'season': initial.year, 'state': 'used', 'updated_at': initial.isoformat()})
    elif seed == 'cached':
        assert day == '2027-07-15'
        picks.save_pick(daily_pick(), directory)
    if prior:
        # Simulate earlier processed-day coverage without spending a live DM.
        path = root / 'health_state' / H
        path.parent.mkdir(exist_ok=True)
        path.write_text(prior)

    statuses = {g['gamePk']: {'abstract': 'P', 'detailed': 'Scheduled', 'code': 'S'} for g in slate}
    config = {'tiers': [], 'orchestrator': {'picks_dir': str(directory),
              'heartbeat_path': str(root / 'heartbeat'), 'models_dir': str(root / 'models'),
              'data_dir': str(root / 'processed')}, 'bluesky': {'dm_recipient': 'offline.test'},
              'scheduler': {'contest_state_required': True, 'pick_delivery': 'dm',
                            'live_forward_capture_on_lock': False, 'entry_intent': 'enter'},
              'health_checks': {'enabled': True}}
    timestamped = Journal(emit)
    handler = logging.StreamHandler(timestamped)
    handler.setFormatter(logging.Formatter('%(name)s %(levelname)s %(message)s'))
    base_logger = logging.getLogger()
    targets = {
        scheduler: {'_now_et': lambda: now, 'datetime': Clock, 'fetch_schedule': fetch,
                    'check_confirmed_lineups': lambda *a: {}, 'retry_urlopen': forbidden,
                    '_watchdog_ping_sleep': sleep, 'run_result_polling': polling,
                    'heartbeat_watchdog': lambda *a, **k: nullcontext(),
                    'notify_ready': lambda: None, 'notify_watchdog': lambda: None},
        orchestrator: {'run_cascade': cascade},
        picks: {'get_game_statuses_detailed': lambda *a: statuses,
                'get_game_statuses': lambda *a: {pk: 'P' for pk in statuses}},
        strategy: {'get_game_statuses_detailed': lambda *a: statuses},
        contest_state: {'datetime': Clock}, health_contest: {'datetime': Clock},
        alert: {'send_dm': health_dm, 'datetime': Clock}, dm: {'send_dm': pick_dm},
        posting: {'post_to_bluesky': post, 'should_post_now': lambda: True},
        runner: {'read_systemd_nrestarts': lambda: None, 'get_self_pid': lambda: None},
    }
    from contextlib import ExitStack

    with ExitStack() as stack:
        for module, replacements in targets.items():
            for attr, value in replacements.items():
                stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch.object(base_logger, 'level', logging.INFO))
        stack.enter_context(patch.object(base_logger, 'handlers', [handler]))
        stack.enter_context(redirect_stderr(timestamped))
        scheduler.run_day(day, config)
    saved = {}
    for name in (H, S):
        path = root / 'health_state' / name
        if path.is_file():
            saved[name] = path.read_text()
            emit('status', (name, saved[name]))
    emit('counts', [(channel, len(body)) for channel, body in deliveries])
    return saved, now, deliveries


def launch(root, label, start, delay=0):
    print(f'\n=== {label} ===')
    pinned = start
    cfg = root / 'launch.toml'
    cfg.write_text('[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
    load = orchestrator.load_config

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return pinned.astimezone(tz) if tz else pinned.replace(tzinfo=None)

    def load_then_advance(path):
        nonlocal pinned
        pinned += timedelta(seconds=delay)
        return load(path)

    def capture(*, date, config, dry_run):
        print(f'run_day({date}) at {pinned.isoformat()} today={date == pinned.date().isoformat()}')

    with patch.object(cli, 'datetime', Clock), patch.object(scheduler, 'run_day', capture), \
            patch.object(orchestrator, 'load_config', load_then_advance):
        result = CliRunner().invoke(cli.cli, ['schedule', '--config', str(cfg)], catch_exceptions=False)
    print(start.isoformat(), result.exit_code, result.stdout.strip(), repr(result.stderr))
    assert result.exit_code == 0


def main():
    day = '2027-07-15'
    opening = '2027-03-24'
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        root = Path(tmp)
        run(root / 'missing', 'missing, no preview, real EOD', day)
        run(root / 'preview', 'missing, cached preview, real EOD', day, seed='cached')
        run(root / 'stale', 'covered, stale', day, seed='stale')
        run(root / 'dated', 'dated at 07:00', day, seed='dated')
        run(root / 'fresh', 'covered, fresh', day, seed='fresh')
        _, wake, _ = run(root / 'opening', 'previous evening is off', '2027-03-23',
                         start='22:30', starts=())
        launch(root, 'Opening Night restart after normal off-day idle', wake + timedelta(seconds=30))
        run(root / 'opening', 'Opening Night representative first pitch 20:10', opening,
            start='10:00:30', starts=('20:10',), supports_predictions=False)
        _, _, _ = run(root / 'midnight', 'previous slate EOD at 00:05 (dated expired)',
                      '2027-07-14', seed='dated', result_end=at(day, '00:05'))
        run(root / 'midnight', 'next day after the post-midnight EOD', day)
        launch(root, 'late evening', at('2027-03-23', '23:40'))
        launch(root, 'load crosses midnight', at('2027-03-23', '23:59:59'), delay=2)
        launch(root, 'after midnight', at(opening, '00:00:01'))
        for label, starts, doubleheader in [
            ('night', ('19:10', '19:40', '22:10'), False),
            ('getaway', ('13:05', '13:10', '13:15'), False),
            ('thin', ('13:05',), False), ('06:10', ('06:10',), False),
            ('doubleheader', ('13:05', '18:40'), True),
            ('opening', ('20:10',), False), ('thin exact :00', ('13:00',), False),
        ]:
            slate = games(opening, starts)
            if doubleheader:
                slate[1]['teams'] = slate[0]['teams']
            wake = scheduler.compute_wakeup_time(slate)
            begin = wake + timedelta(seconds=30)
            plan = scheduler.plan_run_times(slate, now=begin)
            print(label, wake.isoformat(), 'runs', [r['time_et'].strftime('%H:%M') for r in plan],
                  'lead seconds', (plan[0]['time_et'] - begin).total_seconds(),
                  'DH', scheduler.detect_doubleheader_game2s(slate))


if __name__ == '__main__':
    main()
