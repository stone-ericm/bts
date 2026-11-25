import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-h-codex/probe_paired_base.py'


def run(source, fixed):
    env = os.environ | {'PYTHONPATH': os.pathsep.join((str(source), str(ROOT)))}
    args = [sys.executable, '-u', str(ORIGINAL)] + (['--fixed'] if fixed else [])
    captured = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
    print('round 2' if fixed else 'round 1', source)
    print(captured.stdout, end='')
    assert captured.returncode == 0, captured.stderr
    assert ('delivered single' if fixed else 'delivered double') in next(
        line for line in captured.stdout.splitlines() if line.startswith('scalar '))


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        archived = subprocess.run(['git', 'archive', '813935bc', 'src'], cwd=ROOT, capture_output=True,
                                  check=True, timeout=6).stdout
        subprocess.run(['tar', '-xf', '-', '-C', tmp], input=archived, check=True, timeout=6)
        run(Path(tmp) / 'src', False)
        run(ROOT / 'src', True)
