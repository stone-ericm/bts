"""Check base and scratch JSON, heads, send paths, and read-only behavior."""
from dataclasses import replace
from datetime import datetime
import datetime as calendar
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.codex-review/scratch-f4/src'), str(ROOT)]

from bts import day_outcome, dm, picks, scheduler, util
from bts.cli import cli
from bts.health import runner, streak_validation
from tests.test_day_outcome import DATE, _bound_receipt, _box, _publish_receipt, _write_decision

spec = importlib.util.spec_from_file_location('base_day_outcome', ROOT / 'src/bts/day_outcome.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
gather_scratch = day_outcome.gather


class Clock(datetime):
    when = datetime(2027, 4, 3, 8, 30, tzinfo=ZoneInfo('America/New_York'))

    @classmethod
    def now(cls, tz=None):
        return cls.when.astimezone(tz) if tz else cls.when.replace(tzinfo=None)


def run_outcome(p, cfg, games, *, gather=gather_scratch, date=DATE, send_failure=None):
    dms = []

    def schedule(_):
        if games is None:
            raise OSError('unavailable')
        return [{'gamePk': 900}] * games

    def send(to, body):
        dms.append((to, body))
        if send_failure is not None:
            raise send_failure
        return 'offline-confirmed'

    with patch.object(day_outcome, 'gather', gather), patch.object(calendar, 'datetime', Clock), \
         patch.object(scheduler, 'fetch_schedule', schedule), patch.object(dm, 'send_dm', send):
        result = CliRunner().invoke(cli, ['day-outcome', '--config', str(cfg), '--date', date,
                                          '--picks-dir', str(p), '--expected-username', 'stonehengee',
                                          '--dm-recipient', 'offline.test'])
    return result, dms


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    states = {
        'entered': (1, 'enter', {}, 'confirmed'),
        'entered_game_unverified': (1, 'enter', {}, 'batter'),
        'deliberate_skip': (1, 'enter', dict(action='skip', scoreable=False), None),
        'no_games': (0, 'enter', None, None),
        'research_commit': (1, 'research', dict(delivery_status='private_locked'), None),
        'delivered_under_research': (1, 'research', {}, None),
        'intent_unknown': (1, 'bogus', None, None),
        'schedule_unverifiable': (None, 'enter', None, None),
        'no_decision': (1, 'enter', None, None),
        'skip_not_deliberate': (1, 'enter', dict(action='skip', source='crash_guard', scoreable=False), None),
        'commit_not_scoreable': (1, 'enter', dict(delivery_status='failed', scoreable=False), None),
        'contradiction': (0, 'enter', {}, None),
        'entered_late': (1, 'enter', {}, 'late'),
        'not_entered': (1, 'enter', {}, 'absent'),
        'entry_unverifiable': (1, 'enter', {}, None),
    }
    for state, (games, intent, decision, receipt) in states.items():
        row = root / state
        p, cfg = _box(row, intent=intent, delivery='private' if intent == 'research' else 'dm')
        if decision is not None:
            _write_decision(p, **decision)
        if receipt is not None:
            record = _bound_receipt(p, before_cutoff=receipt != 'late', confirmed=receipt not in ('absent', 'batter'),
                                    reason='no_pick' if receipt == 'absent' else 'match')
            if receipt == 'batter':
                record['qualification']['slots'] = [{'state': 'unit_unverified'}]
            _publish_receipt(p, record)
        path = p / 'streak.json'
        path.write_text('{"season":2026,"streak":3,"applied_dates":null}')
        files = snapshot(row)
        good_base, good_base_dms = run_outcome(p, cfg, games, gather=base.gather)
        good, good_dms = run_outcome(p, cfg, games)
        good_payload = json.loads(good.stdout)
        assert good_payload.pop('streak_problem') is None
        assert json.dumps(good_payload) + '\n' == good_base.stdout
        assert (good.exit_code, good_dms, good.stderr) == (good_base.exit_code, good_base_dms, good_base.stderr)
        assert good_payload['state'] == state and snapshot(row) == files

        path.write_text('[]')
        files = snapshot(row)
        bad, bad_dms = run_outcome(p, cfg, games)
        assert bad.exit_code == 1 and len(bad_dms) == 1 and snapshot(row) == files
        assert json.loads(bad.stdout) == good_payload | {'ok': False, 'streak_problem': 'not a JSON object'}
        head = state.replace('_', ' ').upper() if not good_payload['ok'] else 'STREAK FILE UNREADABLE'
        assert bad_dms[0][1].startswith(f'BTS day check {DATE}: {head}. ')
        if state in ('entered', 'deliberate_skip', 'no_games', 'not_entered', 'no_decision'):
            print(state, bad_dms[0][1])
    print('Matched original JSON bytes minus new key, exit and DM with good file, and preserved all file contents:',
          ', '.join(states))

    p, cfg = _box(root / 'errors')
    path = p / 'streak.json'
    for label, content in [('oversized int', '{"streak":' + '9' * 5000 + '}'),
                           ('deep 2300', '[' * 2300 + '0' + ']' * 2300),
                           ('deep 6000', '[' * 6000 + '0' + ']' * 6000)]:
        path.write_text(content)
        original, original_dms = run_outcome(p, cfg, 1, gather=base.gather)
        after, after_dms = run_outcome(p, cfg, 1)
        try:
            picks.load_streak(p, season=2027)
        except Exception as exc:
            scorer_error = type(exc).__name__
        print(label, 'scorer:', scorer_error, 'base:', original.exit_code, original_dms,
              'scratch:', after.exit_code, type(after.exception).__name__, after.stdout, after.stderr, after_dms)
        assert original.exit_code == 1 and 'NO DECISION' in original_dms[0][1]
    path.write_text('[]')
    invalid_cfg, invalid_cfg_dms = run_outcome(p, cfg.with_name('missing.toml'), 1)
    original_cfg, original_cfg_dms = run_outcome(p, cfg.with_name('missing.toml'), 1, gather=base.gather)
    assert (invalid_cfg.exit_code, invalid_cfg.stderr, invalid_cfg_dms) == (2, original_cfg.stderr, original_cfg_dms)
    print('independent config failure with broken model file:', invalid_cfg_dms)
    failed, failed_dms = run_outcome(p, cfg, 1, send_failure=TimeoutError('possibly accepted'))
    assert failed.exit_code == 1 and len(failed_dms) == 1 and 'DM failed: TimeoutError' in failed.stderr
    print('unconfirmed send has one attempt and exit', failed.exit_code, failed.stderr.strip())

    # Simulate interruption before the replacement. A production scorer never truncates the live file in-place.
    path.write_text('{"season":2027,"streak":3}')
    try:
        with patch.object(util.os, 'replace', side_effect=InterruptedError('before replacement')):
            picks.update_streak([True], p, date=DATE)
    except InterruptedError:
        pass
    assert path.read_text() == '{"season":2027,"streak":3}'
    interrupted, _ = run_outcome(p, cfg, 1)
    print('interruption before replace, live read:', interrupted.stdout.strip())
    path.write_text('{')
    torn, torn_dms = run_outcome(p, cfg, 0)
    print('externally truncated file, scratch text:', torn_dms)
    path.write_text('[]')
    alerts = runner._safe_run('streak_validation', lambda: streak_validation.check(p))
    concurrent, concurrent_dms = run_outcome(p, cfg, 0)
    print('separate health CRITICAL and morning DM:', alerts[0].message, concurrent_dms)

    for day, wall, tag in [('2027-12-31', (2028, 1, 1), 2027), ('2028-03-23', (2028, 3, 24), 2027)]:
        Clock.when = datetime(*wall, 8, 30, tzinfo=ZoneInfo('America/New_York'))
        for value in ('3', '"3"'):
            path.write_text(f'{{"season":{tag},"streak":{value}}}')
            opening, messages = run_outcome(p, cfg, 0, date=day)
            print(day, Clock.when.isoformat(), 'last season count', value, 'actual new-season load',
                  picks.load_streak(p, season=wall[0]), 'morning JSON', opening.stdout.strip(), messages)
