"""Pass captured schedule bytes through the parser and its production callers."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from io import BytesIO, StringIO
import json
from pathlib import Path
import runpy
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

from probe_pick import DAY, ROOT, offline


def outcome(call):
    try:
        return call()
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        return f'{type(error).__name__}: {error}'


def schedule_bodies():
    from bts import scheduler

    rows = [
        ('error object', b'{"message":"unavailable"}'), ('missing dates', b'{}'),
        ('null dates', b'{"dates":null}'), ('string dates', b'{"dates":""}'),
        ('object dates', b'{"dates":{}}'), ('non-object date', b'{"dates":[false]}'),
        ('missing games', b'{"dates":[{"date":"2027-03-24"}]}'),
        ('null games', b'{"dates":[{"games":null}]}'),
        ('non-object game', b'{"dates":[{"games":[{"gameType":"A"},42]}]}'),
        ('non-object top', b'[]'), ('invalid JSON', b'{'), ('empty bytes', b''),
        ('deep JSON', b'[' * 1500 + b']' * 1500),
        ('off day', b'{"totalItems":0,"totalGames":0,"dates":[]}'),
        ('empty games', b'{"dates":[{"games":[]}]}'),
        ('invalid NaN JSON', b'{"dates":[],"totalGames":NaN}'),
        ('invalid Infinity JSON', b'{"dates":[],"totalGames":Infinity}'),
        ('invalid negative Infinity JSON', b'{"dates":[],"totalGames":-Infinity}'),
        ('mixed and optional type', b'{"dates":[{"games":[{"gameType":"A","gamePk":4},'
                                    b'{"gameType":"E","gamePk":5},{"gameType":"R","gamePk":6},'
                                    b'{"gamePk":7}]}]}'),
        ('only non-regular', b'{"dates":[{"games":[{"gameType":"A","gamePk":4}]}]}'),
    ]
    with offline() as (world, patch, _):
        for name, body in rows:
            with patch.context() as scope:
                scope.setattr(scheduler, 'retry_urlopen', lambda *a, **k: BytesIO(body))
                print(name, repr(outcome(lambda: scheduler.fetch_schedule(world.day))))


def entrypoints():
    from bts import cli, dm, scheduler, util
    from bts.data import lineup_collect
    from bts.model import predict
    from click.testing import CliRunner

    labels = [
        ('error object', b'{"message":"unavailable"}'),
        ('missing games', b'{"dates":[{"date":"2027-03-24"}]}'),
        ('off day', b'{"dates":[]}'),
        ('invalid JSON constant', b'{"dates":[],"totalGames":NaN}'),
    ]
    for label, body in labels:
        with offline(hour=10, minute=0) as (world, patch, _):
            ordinary_output = StringIO()
            fetched, ticks = [], []

            def request(url, **kwargs):
                date, = parse_qs(urlsplit(url).query)['date']
                fetched.append(date)
                return BytesIO(body)

            patch.setattr(util, 'urlopen', request)
            patch.setattr(scheduler, 'retry_urlopen', util.retry_urlopen)
            patch.setattr(lineup_collect, 'run_collection_tick', lambda *a, **k: ticks.append('tick'))
            workspace = world.root / 'collect'
            workspace.mkdir()
            options = ['--date', world.day, '--config', str(world.config_path)]
            cmds = CliRunner()
            schedule = cmds.invoke(cli.cli, ['schedule', *options])
            day = cmds.invoke(cli.cli, ['day-outcome', *options, '--picks-dir', str(world.picks),
                                       '--dm-recipient', 'offline.invalid'])
            collector = cmds.invoke(cli.cli, ['data', 'collect-lineup-times', '--date', world.day,
                                             '--out-dir', str(workspace)])
            with redirect_stderr(ordinary_output):
                tomorrow = scheduler._next_day_wakeup(world.day, {})
            print(label, json.dumps({
                'scheduler_exit': schedule.exit_code,
                'traceback_end': str(schedule.exception),
                'scheduler_logged_off_day': 'No games scheduled' in schedule.stderr,
                'watchdog_exit': day.exit_code, 'watchdog': json.loads(day.stdout.splitlines()[0])['state'],
                'health_dms': world.sends('dm') + world.sends('health-dm'),
                'collector_exit': collector.exit_code, 'collector_error': str(collector.exception),
                'collection_ticks': ticks, 'JSONL': sorted(f.name for f in workspace.glob('*.jsonl')),
                'next_wake': tomorrow.isoformat(), 'next_wake_log': ordinary_output.getvalue().strip(),
                'requests': fetched,
            }))

    models = [('error', b'{"message":"unavailable"}'), ('off day', b'{"dates":[]}'),
              ('missing games', b'{"dates":[{}]}'), ('null dates', b'{"dates":null}')]
    with offline() as (world, patch, _):
        for label, body in models:
            patch.setattr(predict, 'urlopen', lambda *args, **kwargs: BytesIO(body))
            print('model', label, repr(outcome(lambda: predict._fetch_game_slots(world.day))))

    with offline() as (world, patch, _):
        tries, sleeps = [], []

        def bad_request(*args, **kwargs):
            tries.append(1)
            raise URLError('captured transport failure')

        patch.setattr(util, 'urlopen', bad_request)
        patch.setattr(util.time, 'sleep', sleeps.append)
        patch.setattr(scheduler, 'retry_urlopen', util.retry_urlopen)
        print('transport', outcome(lambda: scheduler.fetch_schedule(DAY)), len(tries), sleeps)

    with offline() as (world, patch, _):
        patch.setattr(scheduler, 'retry_urlopen', lambda *a, **k: BytesIO(b'{}'))
        cmds = CliRunner()
        options = ['--date', world.day, '--config', str(world.config_path),
                   '--picks-dir', str(world.picks), '--dm-recipient', 'offline.invalid']

        def broken_dm(*args, **kwargs):
            raise RuntimeError('captured failed DM')

        patch.setattr(dm, 'send_dm', broken_dm)
        day = cmds.invoke(cli.cli, ['day-outcome', *options])
        print('failed watchdog transport', day.exit_code, day.stdout.strip(), day.stderr.strip())
        world.config['scheduler']['entry_intent'] = 'invalid'
        world.write_config()
        day = cmds.invoke(cli.cli, ['day-outcome', *options])
        print('bad intent wins', day.exit_code, day.stdout.strip())


def delivery_statuses():
    from bts import picks, scheduler

    for case in ('error bytes', 'unreadable', 'known void', 'one void leg'):
        with offline() as (world, patch, _):
            with redirect_stdout(StringIO()):
                world.dated(0)
            refresh = scheduler._refresh_pick_at_fallback_decision(world.config, DAY, None, .03)
            assert refresh.daily.double_down
            state = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False,
                                             None, None, None)

            def statuses(url, **kwargs):
                assert 'hydrate' not in url
                if case == 'error bytes':
                    return BytesIO(b'{"message":"error"}')
                if case == 'unreadable':
                    return BytesIO(b'{')
                games = world.schedule[world.day]
                void = games if case == 'known void' else games[:1]
                for game in void:
                    game['status']['detailedState'] = 'Postponed'
                return world.http(url)

            patch.setattr(picks, 'retry_urlopen', statuses)
            delivered = scheduler._deliver_and_lock_pick(refresh.daily, world.config, world.picks,
                                                         state, DAY, 'fallback', selection=refresh.selection)
            print('delivery', case, delivered, bool(world.sends('dm')), bool(world.sends('health-dm')),
                  refresh.daily.double_down is not None)


def restarts():
    from bts.cli import cli
    from bts import scheduler
    from click.testing import CliRunner

    monitor = runpy.run_path(str(ROOT / 'scripts/check_heartbeat.py'))
    service = (ROOT / 'scripts/systemd/bts-scheduler.service').read_text()
    assert 'Restart=always' in service and 'RestartSec=30' in service
    with offline(hour=10, minute=0) as (world, patch, _):
        hits, attempts = [], []

        def malformed(url, **kwargs):
            hits.append(parse_qs(urlsplit(url).query)['date'][0])
            return BytesIO(b'{"message":"unavailable"}')

        patch.setattr(scheduler, 'retry_urlopen', malformed)
        origin = world.now
        for _ in range(4):
            result = CliRunner().invoke(cli, ['schedule', '--config', str(world.config_path)])
            attempts.append([result.exit_code, str(result.exception),
                             'Fetching schedule' in result.stderr])
            world.now += timedelta(seconds=30)
        print('service invocations', attempts, 'request dates', hits,
              'DMs', len(world.sends('dm') + world.sends('health-dm')))
        print('heartbeat churn', monitor['assess_churn']([{'ts': origin.isoformat(), 'n': 9}],
                                                        12, origin + timedelta(minutes=5))[:2])


if __name__ == '__main__':
    from bts import scheduler

    print('Scheduler source:', scheduler.__file__)
    schedule_bodies()
    entrypoints()
    delivery_statuses()
    restarts()
