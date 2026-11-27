"""Compare the checklist bytes and file list between the two review commits."""

import subprocess


BASE, HEAD = '0b9dd2d2', '2883190b'
CHECKLIST = 'docs/ops/2027-season-start.md'
ARCHIVE = 'docs/reviews/2026-11-18-whole-range/review-fix-k1b-check/'
TEST = 'tests/test_wr_k1_r2.py'


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, timeout=5, check=True).stdout


before = git('show', f'{BASE}:{CHECKLIST}')
after = git('show', f'{HEAD}:{CHECKLIST}')
old = (b'`bts set-contest-streak --date <today> --streak N --best-streak B --saver S --picks-dir DIR` '
       b"with N, B and S read from MLB and DIR the scheduler's absolute picks directory, "
       b'the same directory used for status.')
report = git('show', f'{HEAD}:{ARCHIVE}report.md')
new = report.split('description with “'.encode(), 1)[1].split('”'.encode(), 1)[0]
assert before.count(old) == 1 and before.replace(old, new) == after
print('Checklist bytes match after replacing only the old daily command with the prior review’s exact wording.')

paths = [line.split('\t') for line in git('diff', '--name-status', BASE, HEAD).decode().splitlines()]
assert [(status, path) for status, path in paths if not path.startswith(ARCHIVE)] == [
    ('M', CHECKLIST), ('M', TEST),
]
assert sum(status == 'A' and path.startswith(ARCHIVE) for status, path in paths) == 13
assert not git('diff', '--check', BASE, HEAD)
print('Two changed files and 13 prior review additions:')
for status, path in paths:
    print(status, path)
