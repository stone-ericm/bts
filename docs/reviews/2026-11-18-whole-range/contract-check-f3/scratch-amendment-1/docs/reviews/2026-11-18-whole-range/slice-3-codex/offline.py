"""Disposable filesystem, clock and transport boundaries for the review probes."""

from contextlib import contextmanager, nullcontext
import datetime as calendar
import importlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd
import pytest

from bts import cli, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert


ET = ZoneInfo('America/New_York')
REAL_DATETIME = calendar.datetime
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def instant(day, hour, minute=0, second=0):
    return REAL_DATETIME.fromisoformat(day).replace(hour=hour, minute=minute, second=second, tzinfo=ET)


def game(pk, at):
    return {'gamePk': pk, 'gameDate': at.isoformat(), 'gameType': 'R', 'doubleHeader': 'N',
            'teams': {'away': {'team': {'name': f'Away {pk}'}}, 'home': {'team': {'name': f'Home {pk}'}}},
            'status': {'abstractGameCode': 'P', 'detailedState': 'Scheduled', 'statusCode': 'S'}}


class Offline:
    def __init__(self, root, at):
        self.root = root
        self.now = at
        self.day = at.date().isoformat()
        self.picks = root / 'data' / 'picks'
        self.models = root / 'data' / 'models'
        self.picks.mkdir(parents=True)
        self.models.mkdir()
        self.account = self.picks / 'account_state'
        self.health = self.picks.parent / 'health_state'
        self.config_path = root / 'scheduler.toml'
        self.schedule = {self.day: [game(101, instant(self.day, 19, 10)),
                                    game(102, instant(self.day, 20, 10))]}
        self.events = []
        self.predictions = lambda: self.frame()
        self.config = {
            'orchestrator': {'picks_dir': str(self.picks), 'models_dir': str(self.models),
                             'heartbeat_path': str(root / 'heartbeat')},
            'scheduler': {'contest_state_required': True, 'pick_delivery': 'dm', 'entry_intent': 'enter',
                          'live_forward_capture_on_lock': False},
            'health_checks': {'enabled': False, 'contest_state_expected': True},
            'bluesky': {'dm_recipient': 'offline.invalid'},
            'tiers': [],
        }
        self.write_config()

    def write_config(self):
        lines = ['tiers = []']
        for name, settings in self.config.items():
            if name != 'tiers':
                lines.extend([f'[{name}]', *(f'{k} = {json.dumps(v)}' for k, v in settings.items())])
        self.config_path.write_text('\n'.join(lines))

    def call(self, *args, code=0, catch=False):
        result = CliRunner().invoke(cli.cli, list(args), catch_exceptions=catch)
        print(f'CLI {self.now.isoformat()} {" ".join(args[:2])}: {result.exit_code}\n{result.output}', end='')
        if result.exception and type(result.exception) is not SystemExit:
            print(type(result.exception).__name__, str(result.exception))
        assert result.exit_code == code, (args, result.output, result.exception)
        return result

    def dated(self, streak, *, best=None, day=None, saver='used', code=0):
        self.call('set-contest-streak', '--date', day or self.now.date().isoformat(),
                  '--streak', str(streak), '--best-streak', str(streak if best is None else best),
                  '--saver', saver, '--picks-dir', str(self.picks), code=code)

    def write_account(self, filename, value):
        self.account.mkdir(exist_ok=True)
        (self.account / filename).write_text(json.dumps(value))

    def automatic(self, streak, age_hours=0, source_date=None):
        year = self.now.year
        self.call('season-contract', 'enable', '--season', str(year), '--picks-dir', str(self.picks))
        switch = json.loads((self.account / 'season_contract.json').read_text())
        self.write_account('contest_streak.json', {
            'schema_version': 'bts_contest_streak_auto_v2', 'season': year,
            'activation_id': switch['activation_id'], 'active_streak': streak, 'best_streak': streak,
            'source_date': source_date or self.now.date().isoformat(),
            'recorded_at': (self.now - calendar.timedelta(hours=age_hours)).isoformat(),
        })

    def frame(self, probabilities=(0.99, 0.81), projected=(False, False)):
        rows = []
        for index, game_row in enumerate(self.schedule[self.day]):
            pk = game_row['gamePk']
            rows.append({'batter_id': pk + 500, 'batter_name': f'Synthetic Batter {pk}',
                         'team': 'ABC', 'lineup': 1, 'pitcher_id': 1000 + pk,
                         'pitcher_name': 'Synthetic Pitcher', 'opp': 'XYZ', 'p_hit_pa': 0.33,
                         'p_game_hit': probabilities[index], 'flags': 'PROJECTED' if projected[index] else '',
                         'projected_lineup': projected[index], 'game_pk': pk, 'game_time': game_row['gameDate']})
        return pd.DataFrame(rows)

    def http(self, url, **_):
        address = urlsplit(getattr(url, 'full_url', url))
        if address.path == '/api/v1/schedule':
            day, = parse_qs(address.query)['date']
            self.events.append(('schedule', self.now, day))
            games = self.schedule.get(day, [])
            response = {'dates': [{'games': games}] if games else []}
        else:
            raise AssertionError(f'Uncaptured HTTP {address.path}')
        return io.BytesIO(json.dumps(response).encode())

    def pipeline(self, *args, **kwargs):
        self.events.append(('predict', self.now, None))
        return self.predictions()

    def cascade(self, *args, **kwargs):
        return self.pipeline(), 'synthetic'

    def transport(self, kind, body):
        self.events.append((kind, self.now, body))
        print(f'TRANSPORT {kind} {self.now.isoformat()} {body!r}')
        return f'captured-{kind}'

    def advance(self, seconds, *args, **kwargs):
        assert 0 <= seconds <= 24 * 3600, seconds
        self.now += calendar.timedelta(seconds=seconds)

    def sends(self, kind):
        return [body for event, _, body in self.events if event == kind]


@contextmanager
def offline(at):
    # Imports precede clock replacement and no model module is imported at module scope.
    from bts.model import predict
    from bts.simulate import mdp, tail_policy

    with TemporaryDirectory(prefix='probe-', dir=HERE) as temporary, pytest.MonkeyPatch.context() as mp:
        w = Offline(Path(temporary), at)

        class Clock(REAL_DATETIME):
            @classmethod
            def now(cls, tz=None):
                return w.now.astimezone(tz) if tz else w.now.replace(tzinfo=None)

        for name in ('cli', 'scheduler', 'picks', 'posting', 'strategy', 'contest_state', 'season_contract',
                     'saver_state', 'daily_decision', 'heartbeat', 'health.contest_state', 'health.alert',
                     'reconcile_receipt', 'entry_receipt', 'slate'):
            module = importlib.import_module('bts.' + name)
            if getattr(module, 'datetime', None) is REAL_DATETIME:
                mp.setattr(module, 'datetime', Clock)
        mp.setattr(calendar, 'datetime', Clock)
        mp.setattr(strategy, '_mdp_cache', None)
        mp.chdir(w.root)
        # Strategy resolves the artifact defaults when functions are defined.
        # Link only the shipped policy artifacts into the disposable working directory.
        for policy_path in (mdp.DEFAULT_POLICY_PATH, tail_policy.DEFAULT_TAIL_POLICY_PATH):
            source = ROOT / policy_path
            if source.exists():
                (w.root / policy_path).symlink_to(source)
        for name in ('NOTIFY_SOCKET', 'INVOCATION_ID', 'BTS_REFRESH_ALWAYS'):
            mp.delenv(name, raising=False)

        def deny(*args, **kwargs):
            raise AssertionError('Sockets forbidden')

        for target in ('socket.socket.__init__', 'socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.socket.sendto', 'socket.getaddrinfo', 'socket.create_connection'):
            mp.setattr(target, deny)
        mp.setattr(scheduler, 'retry_urlopen', w.http)
        mp.setattr(picks, 'retry_urlopen', w.http)
        mp.setattr(scheduler, 'check_confirmed_lineups', lambda pks: {pk: {'home', 'away'} for pk in pks})
        mp.setattr(scheduler, '_watchdog_ping_sleep', w.advance)
        mp.setattr(scheduler, 'heartbeat_watchdog', lambda *args, **kwargs: nullcontext())
        mp.setattr(scheduler, '_idle_until_next_wakeup', lambda *args, **kwargs: None)
        mp.setattr(scheduler, 'run_result_polling', lambda *args, **kwargs: 'captured-poll')
        mp.setattr(orchestrator, 'run_cascade', w.cascade)
        mp.setattr(predict, 'run_pipeline', w.pipeline)
        mp.setattr(dm, 'send_dm', lambda _recipient, body: w.transport('dm', body))
        mp.setattr(alert, 'send_dm', lambda _recipient, body: w.transport('health-dm', body))
        mp.setattr(posting, 'post_to_bluesky', lambda body: w.transport('post', body))
        yield w, mp
