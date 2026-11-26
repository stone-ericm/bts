"""Offline, file-backed watchdog replays driven by arguments emitted by the installer."""
from contextlib import ExitStack
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import bts

ARTIFACTS = Path(__file__).resolve().parent
ROOT = ARTIFACTS.parents[1]
sys.path.insert(0, str(ROOT))
from bts import entry_receipt
from bts.daily_decision import write_decision
from bts.picks import save_pick
from tests.test_wr_f1_wrong_game import DAY, World, XW

ET = ZoneInfo('America/New_York')


def load_line(version):
    lines = json.loads((ARTIFACTS / 'generated.json').read_text())[f'{version}_enter']
    line, = [line for line in lines if 'bts check-pick-entered' in line]
    return line


def clock_slots(line, day=DAY):
    minute, hour, *_ = line.split()
    assert minute.startswith('*/')
    hours = set(range(24)) if hour == '*' else set(range(int(hour[:2]), int(hour[3:]) + 1))
    step = int(minute.split('/')[1])
    day = date.fromisoformat(day)
    from datetime import timedelta
    start = datetime.combine(day, datetime.min.time(), ET).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), ET).astimezone(timezone.utc)
    for offset in range(0, int((end - start).total_seconds() // 60), step):
        value = (start + timedelta(minutes=offset)).astimezone(ET)
        if value.hour in hours:
            yield value


def deny_network(*args, **kwargs):
    raise AssertionError('unexpected real network access')


def no_network():
    stack = ExitStack()
    for leaf in ('socket.getaddrinfo', 'socket.socket.connect', 'socket.socket.connect_ex', 'socket.socket.sendto'):
        stack.enter_context(patch(leaf, deny_network))
    return stack


class Replay(World):
    def __init__(self, root, version='f5', day=DAY):
        super().__init__(root, day)
        args = shlex.split(load_line(version), comments=True)
        self.args = args[args.index('check-pick-entered'):args.index('>>')]
        self.args[self.args.index('--picks-dir') + 1] = str(self.picks)
        self.username = self.args[self.args.index('--expected-username') + 1]
        self.recipient = self.args[self.args.index('--dm-recipient') + 1]

    def commit_at(self, pitch, *, at='05:10', game=101, batter=1010, dd=None, delivery='delivered'):
        def instant(hm):
            return datetime.fromisoformat(f'{self.day}T{hm}').replace(tzinfo=ET).isoformat()

        def leg(value):
            ident, game_pk, first = value
            return ident, game_pk, instant(first)

        daily = self.commit(game=game, pitch=instant(pitch), dd=leg(dd) if dd else None, delivery=None)
        daily.pick.batter_id, daily.pick.batter_name = batter, f'Batter {batter}'
        daily.run_time = instant(at)
        if delivery:
            daily.delivered_at = instant(at)
            daily.notification_sent = True
        save_pick(daily, self.picks)
        if delivery:
            assert write_decision(self.day, self.picks, action='double' if dd else 'single', source='mdp',
                                  delivery_status=delivery, scoreable=True, primary={'batter_name': daily.pick.batter_name},
                                  double_down={'batter_name': daily.double_down.batter_name} if dd else None,
                                  now=datetime.fromisoformat(instant(at)))
        return daily

    def invoke(self, at, *, pending=(), profile=None, crosswalk=None, dm='sent', elapsed=0,
               auth='ok', day=None, manual=False, window=None, persist=True, patches=None):
        day = self.day if day is None else day
        calls, sent, attempts, ticks = [], [], [], []
        marker_before = self.marker.read_bytes() if self.marker.is_file() else None

        def fetch(name, value):
            def run(*args, **kwargs):
                calls.append(name)
                if auth == name:
                    raise ValueError(f'synthetic {name} failure')
                return value
            return run

        def dm_send(recipient, msg):
            attempts.append(msg)
            if dm == 'failed':
                raise RuntimeError('synthetic send failure')
            if dm == 'interrupt':
                raise KeyboardInterrupt
            assert recipient == self.recipient
            sent.append(msg)

        def monotonic():
            ticks.append(None)
            return 1000 if len(ticks) == 1 else 1000 + elapsed

        session = SimpleNamespace(user_id='fixture', username='other' if auth == 'wrong-user' else self.username,
                                  xsid='fixture')
        transports = {
            'bts.leaderboard.auth.load_session_cookies': fetch('cookies', {}),
            'bts.leaderboard.auth.extract_uid': lambda _: 'fixture',
            'bts.leaderboard.auth.fetch_login_session': fetch('login', session),
            'bts.contest_fetch.fetch_profile': fetch('profile', profile or {'predictions': []}),
            'bts.contest_fetch.fetch_pending_predictions': fetch('pending', list(pending)),
            'bts.cli._fetch_rounds': fetch('rounds', {1: date.fromisoformat(day)}),
            'bts.cli._fetch_bts_to_mlb': fetch('players', XW if crosswalk is None else crosswalk),
            'bts.dm.send_dm': dm_send,
            'bts.picks._git_head_sha': lambda *args, **kwargs: 'review-fixed-revision',
            'time.monotonic': monotonic,
        }
        args = [arg for arg in self.args if not manual or arg != '--cron']
        if window is not None:
            args.extend(['--window-min', str(window)])
        args.extend(['--now-et', f'{day}T{at}'])
        with no_network() as stack:
            for target, fn in (transports | (patches or {})).items():
                stack.enter_context(patch(target, fn))
            if not persist:
                stack.enter_context(patch('bts.entry_receipt.receipt_io.publish'))
            result = self.runner.invoke(__import__('bts.cli', fromlist=['cli']).cli, args)
        marker_after = self.marker.read_bytes() if self.marker.is_file() else None
        receipts = entry_receipt.discover(self.picks, day) if persist else []
        found = [value for value in receipts if value['started_at'] ==
                 datetime.fromisoformat(f'{day}T{at}').replace(tzinfo=ET).isoformat()]
        return {'exit': result.exit_code, 'exception': type(result.exception).__name__ if result.exception else None,
                'stdout': result.stdout, 'stderr': result.stderr, 'calls': calls, 'marker': self.read_marker(),
                'marker_changed': marker_before != marker_after, 'sent': sent, 'attempts': attempts,
                'receipt': max(found, key=lambda value: value['attempt_id']) if found else None}

    def morning(self):
        messages = []
        from bts.cli import cli
        with no_network() as stack:
            stack.enter_context(patch('bts.scheduler.fetch_schedule', lambda _: [{'gamePk': 101}, {'gamePk': 102}]))
            stack.enter_context(patch('bts.dm.send_dm', lambda _, msg: messages.append(msg)))
            result = self.runner.invoke(cli, ['day-outcome', '--date', self.day, '--config', str(self.config),
                                             '--picks-dir', str(self.picks), '--expected-username', self.username,
                                             '--dm-recipient', self.recipient])
        assert result.exit_code in (0, 1) and result.stdout, (result.output, result.exception)
        return result.exit_code, json.loads(result.stdout), messages
