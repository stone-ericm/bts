import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
SOURCE = ROOT / 'src'
STRATEGY = 'bts/strategy.py'
TAIL = 'bts/simulate/tail_policy.py'
DECISION = 'bts/daily_decision.py'
CHECKS = (
    'tests/test_wr_h_longest.py', 'tests/simulate/test_tail_policy.py', 'tests/test_wr_g_guard.py',
    'tests/test_strategy.py', 'tests/test_decide_action.py', 'tests/test_scheduler_tail_objective.py',
    'tests/test_daily_decision_v3.py', 'tests/test_scheduler_decision_record.py',
    'tests/test_scheduler_decision_record_integration.py', 'tests/scripts/season_walk/',
)

LOAD_BLOCK = '''    if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:
        try:
            mdp = _load_mdp()
        except Exception as exc:
            mdp = {"base_error": f"{type(exc).__name__}: {exc}"}
    else:
        mdp = _load_mdp()
'''
EARLY_LOAD_BLOCK = '''    try:
        mdp = _load_mdp()
    except Exception as exc:
        if _regime(None, streak, date)[0] != OBJECTIVE_LONGEST:
            raise
        mdp = {"base_error": f"{type(exc).__name__}: {exc}"}
'''
MUTATIONS = {
    'double-first-ties': (TAIL, 'return ("single", "double", "skip")[values.index(gain)]',
        'return "double" if values[1] == gain else ("single", "double", "skip")[values.index(gain)]'),
    'swap-increments': (TAIL, 'p_hit * (1.0 + gain), p_both * (2.0 + gain)',
        'p_hit * (2.0 + gain), p_both * (1.0 + gain)'),
    'two-bins': (STRATEGY, 'if tail.n_bins != 1:', 'if tail.n_bins < 1:'),
    'zero-raw-dispatch': (TAIL, 'and days_remaining > 0:', 'and days_remaining >= 0:'),
    'skip-on-degradation': (STRATEGY, 'return "single", reason, None', 'return "skip", reason, None'),
    'drop-objective-enum': (DECISION, ', "longest_streak")', ')'),
    'none-bar': (STRATEGY, 'if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return 0.0',
        'if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return None'),
    'use-base-horizon': (STRATEGY, 'return OBJECTIVE_LONGEST, min(days, LONGEST_STREAK_HORIZON)',
        'return OBJECTIVE_LONGEST, min(days, (mdp or {}).get("season_length", LONGEST_STREAK_HORIZON))'),
    'load-before-dispatch': (STRATEGY, LOAD_BLOCK, EARLY_LOAD_BLOCK),
    'drop-int-horizon-type': (STRATEGY, 'not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1',
        'isinstance(horizon, bool) or horizon < 1'),
    'drop-bool-horizon-type': (STRATEGY, ' or isinstance(horizon, bool)', ''),
    'drop-positive-base-horizon': (STRATEGY, ' or horizon < 1', ''),
    'drop-list-type': (STRATEGY, '    if not isinstance(boundaries, list):\n'
        '        raise ValueError("base policy boundaries must be a list")\n', ''),
    'drop-array-type': (STRATEGY, 'not isinstance(table, np.ndarray) or ', ''),
    'drop-rank': (STRATEGY, ' or table.ndim != 4', ''),
    'drop-integer-dtype': (STRATEGY, '    if table.dtype.kind not in {"i", "u"}:\n'
        '        raise ValueError("base policy table must have an integer dtype")\n', ''),
    'drop-minimum-streaks': (STRATEGY, 'table.shape[0] < 57 or ', ''),
    'drop-day-axis': (STRATEGY, 'table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1)',
        'table.shape[2:] != (2, len(boundaries) + 1)'),
    'drop-saver-axis': (STRATEGY, 'table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1)',
        '(table.shape[1], table.shape[3]) != (horizon + 1, len(boundaries) + 1)'),
    'drop-bin-axis': (STRATEGY, 'table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1)',
        'table.shape[1:3] != (horizon + 1, 2)'),
    'drop-action-min': (STRATEGY, 'table.min() < 0 or ', ''),
    'drop-action-max': (STRATEGY, ' or table.max() > 2', ''),
}


def change(src, module, original, modified):
    path = src / module
    content = (SOURCE / module).read_text()
    assert content.count(original) == 1, (module, original)
    path.write_text(content.replace(original, modified))
    return path


def child(src, args, log, time_limit=18):
    env = os.environ | {'PYTHONPATH': os.pathsep.join((str(src), str(ROOT))),
                        'PYTHONDONTWRITEBYTECODE': '1'}
    with (REVIEW / log).open('w') as output:
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, stdout=output,
                              stderr=subprocess.STDOUT, timeout=time_limit)
    return proc.returncode


def test_changes(src, names, complete):
    for name in names:
        path = change(src, *MUTATIONS[name])
        try:
            launch = ('from pathlib import Path; import bts, pytest, sys; '
                      'assert Path(bts.__file__).resolve().is_relative_to(Path(sys.argv.pop(1))); '
                      'sys.exit(pytest.main(sys.argv[1:]))')
            suite = CHECKS if complete else CHECKS[:1]
            label = 'all' if complete else 'h'
            code = child(src, ['-c', launch, str(src), '-x', '-q', '-p', 'no:cacheprovider',
                               *suite, '-k', 'not legacy_parity'], f'mutation-{name}-{label}.log')
            assert code in (0, 1), (name, code)
            print(name, 'caught' if code else 'survived', label, flush=True)
        finally:
            shutil.copyfile(SOURCE / path.relative_to(src), path)


def boundary_checks(src):
    for boundary in (56, 58):
        change(src, TAIL, 'LONGEST_STREAK_FROM = 57', f'LONGEST_STREAK_FROM = {boundary}')
        assert child(src, [str(REVIEW / 'probe_rule.py')], f'boundary-{boundary}.log') == 0
        print('scratch boundary passed', boundary, flush=True)


def verify_fix(src):
    baseline = '    if not isinstance(boundaries, list):\n        raise ValueError("base policy boundaries must be a list")\n'
    change(src, STRATEGY, baseline, baseline +
           '    if any(not isinstance(edge, (int, float)) for edge in boundaries):\n'
           '        raise ValueError("base policy boundaries must be real numbers")\n')
    assert child(src, [str(REVIEW / 'probe_artifacts_followup.py'), '--paired'], 'scratch-boundary-content-fix.log') == 0
    print('scratch boundary-content fix passed', flush=True)


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        src = Path(tmp) / 'src'
        shutil.copytree(SOURCE, src, ignore=shutil.ignore_patterns('__pycache__'))
        match sys.argv[1]:
            case 'boundary':
                boundary_checks(src)
            case 'fix':
                verify_fix(src)
            case 'group1':
                test_changes(src, list(MUTATIONS)[:11], False)
            case 'group2':
                test_changes(src, list(MUTATIONS)[11:], False)
            case 'wider':
                test_changes(src, sys.argv[2:], True)
