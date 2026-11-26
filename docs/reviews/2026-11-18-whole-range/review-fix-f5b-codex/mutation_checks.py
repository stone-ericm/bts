import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = 'src/bts/cli.py'
ENT = 'src/bts/entry_receipt.py'
HEALTH = 'src/bts/health/pick_entry.py'
MORNING = 'src/bts/day_outcome.py'
INSTALLER = 'scripts/cron-setup-hetzner.sh'
TEST = 'tests/test_wr_f5b_watchdog.py'

LEG_CHECK = '        if slot_identities(prior.get("slots")) != slot_identities(early_slots):'
PITCH_CHECK = '\n'.join((
    '        elif [leg["game_time"] for leg in prior["slots"]] != [leg["game_time"] for leg in early_slots]:',
    '            unresolved = (prior.get("status") != "confirmed"',
    '                          and (prior.get("game_unresolved") is True or prior.get("reason") == "wrong_game"',
    '                               or "wrong_game" in prior.get("escalations", [])))',
    '            prior = {"game_unresolved": unresolved}',
))
CASES = {
    'early-writes-live': ((CLI, 'status_path = early_marker_path(picks, today)',
                          'status_path = picks.parent / "health_state" / "pick_entry_check.json"'),),
    'date-only': ((CLI, LEG_CHECK + '\n            prior = {}\n' + PITCH_CHECK + '\n', ''),),
    'no-writer-leg-binding': ((CLI, LEG_CHECK, '        if not prior:'),),
    'no-sealed-receipt-suppression': ((HEALTH, ' or _live_receipt_follows(picks_dir, et_date, observed)', ''),),
    'health-prefers-early': ((HEALTH, '    if marker.get("date") != today.isoformat():\n',
                             '    if (marker.get("date") == today.isoformat()\n'
                             '            and entry_receipt.read_early_marker(picks_dir, today.isoformat())):\n'
                             '        return _early_alerts(picks_dir, today, now)\n'
                             '    if marker.get("date") != today.isoformat():\n'),),
    'morning-ignores-base-slot': ((MORNING, ' or entry_receipt.has_original_slot(Date.fromisoformat(date), pitch)', ''),),
    'early-pick-no-1mib-cap': ((ENT, '1024 * 1024)', '64 * 1024 * 1024)'),),
    'early-receipt-flag-before-gate': ((CLI, '    early = cron and now.hour < 10\n',
                                         '    early = cron and now.hour < 10\n    rec.early_cron_run = early\n'),
                                        (CLI, '        rec.early_cron_run = True\n', '')),
    'no-installed-cron-flag': ((INSTALLER, 'check-pick-entered --cron --picks-dir',
                                 'check-pick-entered --picks-dir'),),
    'gate-starts-1015': ((ENT, 'range(10 * 60, 24 * 60, 15)', 'range(10 * 60 + 15, 24 * 60, 15)'),),
    'no-morning-leg-binding': ((MORNING, '        if entry_receipt.slot_identities(receipt["selection"].get("slots")) != identities:\n'
                                         '            continue\n', ''),),
    'no-health-leg-binding': ((HEALTH, '    if (entry_receipt.slot_identities(entry_receipt.selection_slots(daily))\n'
                                       '            != entry_receipt.slot_identities(marker["slots"])):\n'
                                       '        return []\n', ''),),
}


def check(case, scratch):
    with TemporaryDirectory(dir=scratch) as dirname:
        copy = Path(dirname)
        shutil.copytree(ROOT / 'src', copy / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        for name in (INSTALLER, TEST, 'tests/__init__.py', 'tests/conftest.py',
                     'tests/test_entry_intent.py', 'tests/test_wr_f1b_lock.py', 'tests/test_wr_f1_wrong_game.py',
                     'pyproject.toml'):
            dest = copy / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        (copy / 'docs').symlink_to(ROOT / 'docs', target_is_directory=True)
        for name, old, new in CASES.get(case, ()):
            target = copy / name
            data = target.read_text()
            assert data.count(old) == 1, (case, name, data.count(old))
            target.write_text(data.replace(old, new))
        env = os.environ | {'PYTHONPATH': str(copy / 'src')}
        now = subprocess.run([sys.executable, '-c', 'import bts; print(bts.__file__)'], cwd=copy, env=env,
                             check=True, capture_output=True, text=True, timeout=5)
        print(case, now.stdout.strip(), flush=True)
        assert Path(now.stdout.strip()).is_relative_to(copy / 'src')
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '-x', TEST],
                                cwd=copy, env=env, timeout=75, capture_output=True, text=True)
        print(result.stdout, result.stderr, sep='\n')
        assert result.returncode in (0, 1), result.returncode
        if case == 'control':
            assert result.returncode == 0
        elif result.returncode == 1:
            assert 'FAILED ' in result.stdout
            print(case, 'DETECTED')
        else:
            print(case, 'SURVIVED')


if __name__ == '__main__':
    check(sys.argv[1], Path(sys.argv[2]))
