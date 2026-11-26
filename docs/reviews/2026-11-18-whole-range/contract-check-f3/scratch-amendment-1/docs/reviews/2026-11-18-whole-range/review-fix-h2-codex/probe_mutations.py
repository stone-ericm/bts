import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
SOURCE = Path('src').resolve()
STATE, FETCH, CLI, PICKS, TAIL = ('bts/contest_state.py', 'bts/contest_fetch.py', 'bts/cli.py',
                                  'bts/picks.py', 'bts/simulate/tail_policy.py')
MUTATIONS = {
    's1-restore-57': (STATE, [('<= active <= MAX_SEASON_STREAK:', '<= active <= 57:'),
                              ('<= best <= MAX_SEASON_STREAK:', '<= best <= 57:')]),
    's2-restore-57': (STATE, [('streak > MAX_SEASON_STREAK\n', 'streak > 57\n'),
                              ('(best < streak or best > MAX_SEASON_STREAK)', '(best < streak or best > 57)')]),
    's3-restore-57': (STATE, [('best < contest.streak or best > MAX_SEASON_STREAK:', 'best < contest.streak or best > 57:')]),
    's4-restore-57': (FETCH, [('if best_streak > MAX_SEASON_STREAK:', 'if best_streak > 57:')]),
    's5-restore-57': (CLI, [('if best_streak > MAX_SEASON_STREAK:', 'if best_streak > 57:')]),
    's6-restore-57': (PICKS, [('and streak > MAX_SEASON_STREAK:', 'and streak > 57:'),
                              ('<= best <= MAX_SEASON_STREAK', '<= best <= 57')]),
    's4-cap-401': (FETCH, [('if best_streak > MAX_SEASON_STREAK:', 'if best_streak > MAX_SEASON_STREAK + 1:')]),
    's4-cap-399': (FETCH, [('if best_streak > MAX_SEASON_STREAK:', 'if best_streak > MAX_SEASON_STREAK - 1:')]),
    's2-no-active-cap': (STATE, [('streak > MAX_SEASON_STREAK\n        or (best is not None', '(best is not None')]),
    's4-no-cap': (FETCH, [('    if best_streak > MAX_SEASON_STREAK:\n'
                           '        raise ContestFetchError(f"seasonBestStreak must be <= {MAX_SEASON_STREAK}")\n', '')]),
    's4-no-order': (FETCH, [('    if best_streak < active_streak:\n'
                             '        raise ContestFetchError("seasonBestStreak must be >= activeStreak")\n', '')]),
    's6-dated-outer-cap-401': (PICKS, [('if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:',
                                       'if kind == "automatic" and streak > MAX_SEASON_STREAK or '\
                                       'kind == "dated" and streak > MAX_SEASON_STREAK + 1:')]),
    'policy-target-from-cap': (TAIL, [('TARGET = 57', 'from bts.contest_state import MAX_SEASON_STREAK\n\nTARGET = MAX_SEASON_STREAK')]),
    'contest-target-from-cap': (STATE, [('TARGET_STREAK = 57\nMAX_SEASON_STREAK = 400',
                                        'MAX_SEASON_STREAK = 400\nTARGET_STREAK = MAX_SEASON_STREAK')]),
}
GATE_GROUPS = [
    ['tests/test_wr_h2_caps.py', 'tests/test_contest_state_best_trust.py', 'tests/test_contest_state_cs1_review.py',
     'tests/test_contest_state_defensive_reads.py', 'tests/test_pick_chosen_on_cs5a.py', 'tests/test_saver_cs3.py',
     'tests/test_wr_c1_history.py'],
    ['tests/test_contest_dated_cs4a.py', 'tests/test_contest_dated_cs4a_review.py', 'tests/test_season_contract_cs4b1.py',
     'tests/test_season_contract_cs4b1_edges.py', 'tests/test_season_contract_cs4b2.py'],
    ['tests/test_wr_h_longest.py', 'tests/test_morning_state_cs6b1.py', 'tests/test_stale_delivery_cs4c.py'],
    ['tests/scripts/season_walk/'],
]


def test(mutation, args, scratch, suffix):
    log = REVIEW / f'mutation-{mutation}-{suffix}.log'
    environment = dict(os.environ, PYTHONPATH=f'{scratch}:{Path.cwd()}', PYTHONDONTWRITEBYTECODE='1')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-p', 'no:cacheprovider', '-q', '--tb=short', *args],
                                check=False, timeout=50, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert result.returncode in {0, 1} and (result.returncode == 0 or 'FAILED ' in text), (log, result.returncode)
    print(mutation, suffix, 'SURVIVED' if result.returncode == 0 else 'CAUGHT', text.splitlines()[-1], flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='+', choices=MUTATIONS)
    parser.add_argument('--group', type=int, choices=(1, 2, 3, 4))
    cli = parser.parse_args()
    scratch = REVIEW / 'mutation-worktree' / 'src'
    scratch.parent.mkdir(exist_ok=True)
    shutil.copytree(SOURCE, scratch, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for mutation in cli.cases:
        relative, edits = MUTATIONS[mutation]
        original = (SOURCE / relative).read_text()
        modified = original
        for old, new in edits:
            assert modified.count(old) == 1, (mutation, old)
            modified = modified.replace(old, new, 1)
        (scratch / relative).write_text(modified)
        try:
            args = GATE_GROUPS[cli.group - 1] if cli.group else ['tests/test_wr_h2_caps.py']
            test(mutation, args, scratch, f'group-{cli.group}' if cli.group else 'focused')
        finally:
            (scratch / relative).write_text(original)


if __name__ == '__main__':
    main()
