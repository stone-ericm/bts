"""Adapt the first probe's fixture keyword to the existing write_decision API."""
from unittest.mock import patch

from probe_support import offline_probe
from shadow import compare, probe


def corrected(root, source):
    from bts import daily_decision

    write_decision = daily_decision.write_decision

    def write(*args, **kwargs):
        if 'rank1' in kwargs:
            kwargs['primary'] = kwargs.pop('rank1')
        return write_decision(*args, **kwargs)

    with patch.object(daily_decision, 'write_decision', write):
        probe(root, source)


if __name__ == '__main__':
    raise SystemExit(offline_probe(corrected, compare))
