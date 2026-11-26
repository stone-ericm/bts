"""Check that each of the previously missed changes now fails its targeted test."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from review_support import ROOT

sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))
from mutation_checks import CASES


tests = {
    'health-prefers-early': ('round2_same_date_live_health_precedence_without_seal', 1),
    'early-pick-no-1mib-cap': ('round2_oversized_early_pick_is_unavailable', 1),
    'early-receipt-flag-before-gate': ('round2_local_gate_receipts_omit_early_flag', 2),
    'no-morning-leg-binding': ('round2_early_readers_bind_saved_legs[morning]', 1),
    'no-health-leg-binding': ('round2_early_readers_bind_saved_legs[health]', 1),
}
case, scratch = sys.argv[1:]
selector, expected = tests[case]
with TemporaryDirectory(dir=scratch) as folder:
    copy = Path(folder)
    shutil.copytree(ROOT / 'src', copy / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for filename, existing, replacement in CASES[case]:
        target = copy / filename
        code = target.read_text()
        assert code.count(existing) == 1
        target.write_text(code.replace(existing, replacement))
    env = os.environ | {'PYTHONPATH': str(copy / 'src')}
    command = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '--tb=short',
               f'tests/test_wr_f5b_watchdog.py::test_{selector}']
    result = subprocess.run(command, cwd=ROOT, env=env, timeout=10, capture_output=True, text=True)
    print(case, copy / 'src', result.stdout, result.stderr, sep='\n')
    assert result.returncode == 1 and result.stdout.count('FAILED tests/test_wr_f5b_watchdog.py::test_') == expected
