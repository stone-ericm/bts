"""Compare the committed checklist to the baseline with just the two approved replacements."""

from pathlib import Path
from subprocess import check_output


ROOT = Path(__file__).resolve().parents[2]
BASE = '4c301f7f'
HEAD = '0b9dd2d2'
CHECKLIST = 'docs/ops/2027-season-start.md'


def git(*args):
    return check_output(['git', *args], cwd=ROOT, timeout=5)


if __name__ == '__main__':
    paths = git('diff', '--name-only', BASE, HEAD, '--', 'docs/ops').decode().splitlines()
    assert paths == [CHECKLIST], paths
    assert not git('diff', '--name-only', BASE, HEAD, '--', 'src')
    base = git('show', f'{BASE}:{CHECKLIST}')
    head = git('show', f'{HEAD}:{CHECKLIST}')
    daily_old = (b'`bts set-contest-streak --date <today> --streak N --best-streak B --saver S` '
                 b'with N, B and S read from MLB.')
    daily_new = (b'`bts set-contest-streak --date <today> --streak N --best-streak B '
                 b'--saver S --picks-dir DIR` with N, B and S read from MLB and DIR the scheduler\'s '
                 b'absolute picks directory, the same directory used for status.')
    late_old = (b'If that check refused before choosing a pick, confirm this is still today\'s '
                b'slate in ET and that MLB has settled every prior entry or confirmed its absence '
                b'after its window closed. File the verified dated observation between checks '
                b'while another lineup check or fallback remains.')
    late_new = (b'If a lineup check or fallback is blocked by unusable contest state, confirm this '
                b'is still today\'s slate in ET and that MLB has settled every prior entry or '
                b'confirmed its absence after its window closed. If a later lineup check or '
                b'fallback refresh still has time to produce an enterable pick, file the verified '
                b'dated observation between runs.')
    row, = (line for line in base.splitlines() if line.startswith(b'| B8a |'))
    assert row.count(daily_old) == row.count(late_old) == 1
    assert base.replace(daily_old, daily_new, 1).replace(late_old, late_new, 1) == head
    assert (ROOT / CHECKLIST).read_bytes() == head
    print(f'{CHECKLIST}: only the two approved replacements in B8a, verified byte for byte')
    print('No other docs/ops path changed. No src changes.')
