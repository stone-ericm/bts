"""Rebuild BL-O3 with the archived offline schedule, grading and DM fixtures."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'docs/reviews/2026-11-18-whole-range/closing-test/codex-blind'))

from bts import entry_receipt, picks
from bts.health import runner, streak_validation
import offline
from midseason import REAL_GRADE, network_fixture
from offline import World, at
from opening import DAY2, OPEN

offline.ROOT = ROOT


def audit(bind_game):
    with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as world:
        path = world.directory / 'streak.json'
        path.write_text('[]')
        world.pending = [{'roundId': 1, 'playerId': 1}]
        if bind_game:
            import gzip

            unit_dir = world.directory.parent / 'leaderboard/static_snapshots/units'
            unit_dir.mkdir(parents=True)
            (unit_dir / '2027-03-24T220000Z.json.gz').write_bytes(gzip.compress(json.dumps({
                'captured_at': '2027-03-24T22:00:00Z',
                'units': [{'id': 700, 'roundId': 1, 'unitType': 'ROUND', 'feedId': 101,
                           'startDate': f'{OPEN}T04:00:00Z', 'endDate': f'{OPEN}T23:59:59Z'}],
            }).encode()))
            world.pending[0]['unitId'] = 700
        actual_health = runner.run_all_checks

        def record_health(*args, **kwargs):
            world.record('health started')
            return actual_health(*args, **kwargs)

        world.patch(runner, 'run_all_checks', record_health)
        world.patch(picks, 'resolve_daily_slot_results', REAL_GRADE)
        world.patch(picks, '_fetch_json', lambda url, **opts: network_fixture(world, url, **opts))
        world.dated()
        for minute in ('10:00', '10:01'):
            world.now = at(OPEN, minute)
            world.actions = [(at(OPEN, '18:45'), world.watchdog)]
            result = world.run(via_cli=True)
            assert isinstance(result.exception, AttributeError), (result.output, result.exception)
        world.now = at(DAY2, '01:00')
        result = world.invoke('check-results', '--date', OPEN, '--wait-deadline-et', '06:00')
        assert isinstance(result.exception, AttributeError), result.output
        for hm in ('02:00', '07:40'):
            world.now = at(DAY2, hm)
            result = world.invoke('reconcile')
            assert isinstance(result.exception, AttributeError), result.output
        assert not any(event[1] == 'health started' for event in world.events)
        for receipt_path in sorted(world.directory.parent.rglob('*.json')):
            if 'reconcile' in str(receipt_path):
                receipt = json.loads(receipt_path.read_text())
                print(receipt_path.relative_to(world.root), json.dumps(receipt, sort_keys=True))
        world.now = at(DAY2, '08:30')
        outcome = world.outcome(OPEN)
        print('bind_game:', bind_game, 'receipt slots:', [r.get('qualification') for r in
              entry_receipt.discover(world.directory, OPEN) if r.get('outcome') == 'observed'])
        assert outcome.exit_code == 0, outcome.output
        assert json.loads(outcome.stdout)['state'] == ('entered' if bind_game else 'entered_game_unverified')
        assert path.read_text() == '[]'
        assert picks.load_pick(OPEN, world.directory).result is None
        print('events:', json.dumps([e for e in world.events if e[1] in ('cli', 'send', 'health started')], indent=2))
        streak_alerts = runner._safe_run('streak_validation', lambda: streak_validation.check(world.directory))
        assert len(streak_alerts) == 1 and streak_alerts[0].level == 'CRITICAL'
        print('forced staggered health, streak check alone:', streak_alerts)
        if bind_game:
            alerts = actual_health(world.directory, world.models, 'offline.test', today=at(OPEN).date())
            print('forced full health CRITICAL:', [a.message for a in alerts if a.level == 'CRITICAL'])
            print('forced full health sends:', [e for e in world.events if e[1] == 'send'])


audit(False)
audit(True)
