"""Run the round-two scenarios with the command extracted from current B8a."""

from pathlib import Path
import re
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-k1-codex-r2'))

import check_operator_variants as review


def b8a_command():
    checklist = (ROOT / 'docs/ops/2027-season-start.md').read_text()
    row, = (line for line in checklist.splitlines() if line.startswith('| B8a |'))
    command, = re.findall(r'`(bts set-contest-streak --date <today>[^`]+)`', row)
    return shlex.split(command)[1:]


COMMAND = b8a_command()


def file_observation(day, directory, streak=57):
    assert directory.is_absolute(), directory
    values = {'<today>': day, 'N': str(streak), 'B': '60', 'S': 'used', 'DIR': str(directory)}
    command = [values.get(word, word) for word in COMMAND]
    code, output = review._invoke(*command)
    assert code == 0, output
    assert 'live decision now: accepted' in review.status(directory)
    if directory.name == 'live-picks':
        assert not review.snapshot(directory.parent / 'data/picks')
    return code, output


if __name__ == '__main__':
    review.file_observation = file_observation
    for case in ('custom-directory', 'later-check', 'in-loop-fallback'):
        print(f'Following current B8a in {case}', flush=True)
        review.run_day(case, repair=True)
