import datetime as dt
import json
from unittest.mock import Mock, patch

from click.testing import CliRunner

from probe_support import freeze, offline_probe
from review_runner import dump
import shadow_rollover


def corrected(root, source):
    from bts import cli, daily_decision, skip_policy_shadow as policy
    from tests.test_skip_policy_shadow import _cand

    def complete_legacy_fixture(path, data):
        if data.get('schema_version') == 'bts_skip_policy_shadow_status_v1':
            data['shadow_band_hit_rate']['wilson_ci'] = None
        dump(path, data)

    with patch.object(shadow_rollover, 'dump', complete_legacy_fixture):
        shadow_rollover.probe(root, source)

    picks = root / 'corrupt-policy'
    old = '2026-09-26'
    now = dt.datetime(2027, 1, 1, 5, 1, tzinfo=dt.timezone.utc)
    daily_decision.write_decision(old, picks, action='skip', source='mdp', primary=_cand(),
                                  streak=10, delivery_status='not_applicable', scoreable=False,
                                  objective='reach57', now=now)
    dump(policy.decision_path(old, picks), [])
    output = root / 'corrupt-policy-status.json'
    clock = freeze(now)
    with patch.object(dt, 'datetime', clock), patch.object(policy, 'datetime', clock), \
         patch.object(policy, 'make_hit_checker', lambda: Mock(return_value=None)):
        result = CliRunner().invoke(cli.cli, ['skip-policy-shadow-update', '--picks-dir', str(picks),
                                             '--status-output', str(output)])
    dump(root / 'corrupt-policy-summary.json', {'exit': result.exit_code, 'exception': repr(result.exception),
                                               'stdout': result.stdout.replace(str(root), '<work>'),
                                               'status_written': output.exists()})


def compare(root):
    shadow_rollover.compare(root)
    a, b = [json.loads((root / label / 'corrupt-policy-summary.json').read_text()) for label in ('base', 'head')]
    assert a['exit'] == 1 and 'AttributeError' in a['exception'] and not a['status_written']
    assert b['exit'] == 0 and b['status_written']
    print('Corrupt policy object base/head:', a, b)


if __name__ == '__main__':
    raise SystemExit(offline_probe(corrected, compare))
