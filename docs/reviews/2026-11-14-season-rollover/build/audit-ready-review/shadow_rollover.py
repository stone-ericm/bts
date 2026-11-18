"""Confirm the default cron at midnight with actionable checkpoint and prunable state."""
import datetime as dt
from pathlib import Path
import json
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from probe_support import freeze, offline_probe
from review_runner import dump


def probe(root, source):
    from bts import cli, daily_decision, skip_policy_shadow as policy, web
    from tests.test_skip_policy_shadow import _cand

    folder = root / 'picks'

    def record(day, outcome, should_record=True):
        now = dt.datetime.fromisoformat(day + 'T23:00:00+00:00')
        daily_decision.write_decision(day, folder, action='skip', source='mdp', primary=_cand(),
                                      streak=10, delivery_status='not_applicable', scoreable=False,
                                      objective='reach57', now=now)
        if should_record:
            value = policy.record_skip_from_decision(day, folder, now=now)
            assert value
            dump(policy.decision_path(day, folder), value | {'shadow_pick_result': outcome})

    for i in range(30):
        record(str(dt.date(2026, 8, 1) + dt.timedelta(days=i)), 'hit')
    pending = '2026-12-31'
    prune_day = '2026-12-30'
    missed_record = '2026-12-29'
    record(pending, None)
    record(prune_day, None)
    daily_decision.write_decision(prune_day, folder, action='single', source='mdp',
                                  delivery_status='delivered', scoreable=True)
    record(missed_record, None, should_record=False)
    now = dt.datetime(2027, 1, 1, 5, 1, tzinfo=dt.timezone.utc)
    status_path = root / 'skip_policy_shadow_status.json'
    with pytest.MonkeyPatch.context() as patch:
        clock = freeze(now)
        patch.setattr(dt, 'datetime', clock)
        patch.setattr(policy, 'datetime', clock)
        patch.setattr(policy, 'make_hit_checker', lambda: Mock(return_value=None))
        patch.setattr(web, 'VALIDATION_DIR', root)
        result = CliRunner().invoke(cli.cli, ['skip-policy-shadow-update', '--picks-dir', str(folder),
                                             '--status-output', str(status_path)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert f'Pruned 1 superseded record(s): {prune_day}' in result.stdout
        assert f'Recorded 1 new skip decision(s): {missed_record}' in result.stdout
        assert not policy.decision_path(prune_day, folder).exists()
        assert policy.decision_path(missed_record, folder).exists()
        assert json.loads(policy.decision_path(pending, folder).read_text())['shadow_pick_result'] is None
        def read():
            status = web.load_skip_policy_shadow_status()
            result = CliRunner().invoke(cli.cli, ['skip-policy-shadow-status', '--status-file', str(status_path)])
            assert result.exit_code == 0
            return {'cli': result.stdout, 'dashboard': web.render_skip_policy_shadow_section(status)}

        data = {'nightly': result.stdout.replace(str(root), '<work>'), 'live': read()}
        original = status_path.read_bytes()
        dump(status_path, {'schema_version': 'bts_skip_policy_shadow_status_v1', 'generated_at': 'fixture',
                           'counts': {'divergent_days': 1, 'resolved_divergent': 1, 'pending': 0},
                           'shadow_band_hit_rate': {'verdict': 'above_breakeven', 'rate': 1.0,
                                                    'hits': 1, 'resolved': 1, 'breakeven_p': 0.744}})
        data['legacy'] = read()
        status_path.write_bytes(original)
    dump(root / 'summary.json', data)


def compare(root):
    a, b = [json.loads((root / label / 'summary.json').read_text()) for label in ('base', 'head')]
    assert a['legacy'] == b['legacy']
    assert 'above_breakeven' in a['nightly'] and 'insufficient_n' in b['nightly']
    assert '2 pending' in a['live']['cli'] and '0 pending' in b['live']['cli']
    assert not b['live']['dashboard']
    for label, data in [('base', a), ('head', b)]:
        print(label, json.dumps(data, indent=2))


if __name__ == '__main__':
    raise SystemExit(offline_probe(probe, compare))
