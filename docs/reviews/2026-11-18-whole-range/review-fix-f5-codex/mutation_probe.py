"""Run the F5 tests against one isolated change from the brief."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CLI = 'src/bts/cli.py'
INSTALLER = 'scripts/cron-setup-hetzner.sh'
GATE = re.compile(r'    if cron and now.hour < 10:\n.*?^    status_path', re.S | re.M)
TEST = 'tests/test_wr_f5_watchdog_window.py'
CASES = {
    'drop-gate': (CLI, GATE, '    status_path'),
    'manual-too': (CLI, 'if cron and now.hour < 10:', 'if now.hour < 10:'),
    'drop-cron-flag': (INSTALLER, 'check-pick-entered --cron --picks-dir', 'check-pick-entered --picks-dir'),
    'include-ten': (CLI, 'now.hour < 10:', 'now.hour <= 10:'),
    'stop-2330': (CLI, 'range(0, 14 * 60, 15)', 'range(0, 13 * 60 + 31, 15)'),
    'start-1015': (CLI, 'range(0, 14 * 60, 15)', 'range(15, 14 * 60, 15)'),
    'inclusive-cutoff': (CLI, 'submit_cutoff_min < slot_lead', 'submit_cutoff_min <= slot_lead'),
}


def scratch_copy(copy):
    shutil.copytree(ROOT / 'src', copy / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for name in (INSTALLER, TEST, 'tests/__init__.py', 'tests/conftest.py', 'tests/test_entry_intent.py',
                 'tests/test_wr_f1_wrong_game.py', 'pyproject.toml'):
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (copy / 'docs').symlink_to(ROOT / 'docs', target_is_directory=True)


def main(case):
    with TemporaryDirectory() as directory:
        copy = Path(directory)
        scratch_copy(copy)
        if case != 'control':
            name, before, after = CASES[case]
            target = copy / name
            source = target.read_text()
            pattern = before if isinstance(before, re.Pattern) else re.compile(re.escape(before))
            changed, count = pattern.subn(after, source)
            assert count == 1 and changed != source
            target.write_text(changed)
        # The fixture's uv shim resets its environment. Putting this source at
        # the scratch checkout root lets its separate CLI process import it too.
        (copy / 'bts').symlink_to(copy / 'src/bts', target_is_directory=True)
        env = os.environ | {'PYTHONPATH': str(copy / 'src')}
        system = subprocess.run([sys.executable, '-c', 'import bts; print(bts.__file__)'], cwd=copy,
                                env=env, capture_output=True, text=True, check=True, timeout=5)
        print(case, system.stdout.strip(), flush=True)
        assert str(copy) in system.stdout
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', TEST],
                                cwd=copy, env=env, timeout=45)
        assert result.returncode in (0, 1)
        if case == 'control':
            assert result.returncode == 0
        print(case, 'SURVIVED' if result.returncode == 0 else 'DETECTED')


if __name__ == '__main__':
    main(sys.argv[1])
