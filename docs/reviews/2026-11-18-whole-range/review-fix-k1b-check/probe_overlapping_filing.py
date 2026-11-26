"""Interleave two real CLI writes in the gap after the later refused check."""

import json

from pytest import MonkeyPatch

from replay_round_two import COMMAND, file_observation, review
from bts import util


def overlapping_filing(day, directory, streak=57):
    publish = util.atomic_write_text
    order = []

    def interleave(path, body):
        if not order:
            assert not path.exists()
            order.append('first reached publish')
            values = {'<today>': day, 'N': str(streak), 'B': '60', 'S': 'used', 'DIR': str(directory)}
            code, output = review._invoke(*(values.get(word, word) for word in COMMAND))
            assert code == 0 and 'Observation written' in output, output
            assert path.exists()
            order.append('second published')
            publish(path, body)
            order.append('first published')
        else:
            publish(path, body)

    with MonkeyPatch.context() as mp:
        mp.setattr(util, 'atomic_write_text', interleave)
        result = file_observation(day, directory, streak)
    assert order == ['first reached publish', 'second published', 'first published'], order
    snapshot = review.snapshot(directory / 'account_state')
    assert set(snapshot) == {'contest_streak.manual.json', 'contest_manual_requests.jsonl'}, snapshot
    requests = [json.loads(line) for line in snapshot['contest_manual_requests.jsonl'].splitlines()]
    assert len(requests) == 2
    row = json.loads(snapshot['contest_streak.manual.json'])
    for record in [row, *requests]:
        assert (record['effective_date'], record['active_streak'], record['best_streak'],
                record['saver_state']) == (day, 57, 60, 'used')
    print('two overlapping filings: two requests, one correct observation, no saver flag or streak write',
          flush=True)
    return result


if __name__ == '__main__':
    review.file_observation = overlapping_filing
    review.run_day('later-check', repair=True)
