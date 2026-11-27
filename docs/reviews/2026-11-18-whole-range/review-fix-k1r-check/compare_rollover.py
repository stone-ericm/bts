"""Compare every dispatch from the original rollover harness on both trees."""

import json
import sys
from collections import Counter
from pathlib import Path


def read_dispatches(log):
    dispatches = {}
    for line in log.read_text().splitlines():
        label, separator, payload = line.partition(' [')
        if not separator:
            label, separator, payload = line.partition(' {')
        if not separator or not label.startswith(('TRACE ', 'NEXT ', 'EOD ', 'DISTINCT ', 'DIGEST FIRST ')):
            continue
        if ' WIRE ' in label:
            continue
        values = json.loads(separator[-1] + payload)
        if isinstance(values, dict):
            values = [values]
        assert label not in dispatches, label
        for value in values:
            # Source line numbers shifted. The call path and the outcomes must agree.
            value['caller'] = value['caller'].rsplit(':', 1)[0]
        dispatches[label] = values
    return dispatches


head, final2 = (read_dispatches(Path(argument)) for argument in sys.argv[1:])
assert len(head) == len(final2) == 22, (len(head), len(final2))
assert head.keys() == final2.keys(), head.keys() ^ final2.keys()
counts = Counter()
for label, expected in final2.items():
    actual = head[label]
    assert actual == expected, (label, 'FINAL2', expected, 'HEAD', actual)
    outcomes = ['delivered' if row['wire'] else 'suppressed' for row in actual]
    assert all(row['attempt'] == row['wire'] for row in actual), label
    print(label, ', '.join(outcomes))
    counts.update(outcomes)
print(f'Matched {counts.total()} dispatches: {counts["delivered"]} delivered, {counts["suppressed"]} suppressed')
