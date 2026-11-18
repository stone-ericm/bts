import datetime as dt
import json
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from probe_support import freeze, offline_probe
from review_runner import dump

UTC = dt.timezone.utc
OPTIONS = {'shadow-status': '--output', 'skip-policy-shadow-update': '--status-output',
           'check-results': '--shadow-status-output'}


def probe(root, source):
    from bts import cli, daily_decision, picks, shadow_eval, web
    from bts import skip_policy_shadow as policy
    from bts.health import result_resolution
    from tests.test_shadow_eval import _daily, _pick
    from tests.test_skip_policy_shadow import _cand

    def context(folder, day, result='hit', paired=True, version=shadow_eval.SHADOW_MODEL_NAME):
        if paired:
            picks.save_pick(_daily(day, _pick('Production', 1), result='hit'), folder)
        return picks.save_shadow_pick(_daily(day, _pick('Context', 2), result=result,
                                            shadow_model_version=version), folder)

    def record(folder, day, result='miss', *, contradict=False, defer=False):
        now = dt.datetime.fromisoformat(day + 'T23:00:00+00:00')
        daily_decision.write_decision(day, folder, action='skip', source='mdp', rank1=_cand(),
                                      streak=10, delivery_status='not_applicable', scoreable=False,
                                      objective='reach57', now=now)
        if defer:
            return
        assert policy.record_skip_from_decision(day, folder, now=now)
        handle = policy.decision_path(day, folder)
        dump(handle, json.loads(handle.read_text()) | {'shadow_pick_result': result})
        if contradict:
            daily_decision.write_decision(day, folder, action='single', source='mdp', delivery_status='delivered',
                                          scoreable=True, now=now)
        return handle

    def reconcile_to(result):
        return Mock(return_value=result)

    calls = {}

    def invoke(tag, instant, directory, command, args=()):
        clock = freeze(instant)
        checker = reconcile_to('hit' if tag.startswith('operations') else None)
        path = root / f'{tag}.json'
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dt, 'datetime', clock)
            for mod in (shadow_eval, policy, daily_decision, picks):
                patch.setattr(mod, 'datetime', clock)
            patch.setattr(cli, '_now_et', lambda: instant)
            patch.setattr(policy, 'make_hit_checker', lambda: checker)
            response = CliRunner().invoke(cli.cli, [command, '--picks-dir', str(directory),
                                                   OPTIONS[command], str(path), *args])
        wire = dict(exit=response.exit_code, stdout=response.stdout.replace(str(root), '<work>'),
                    stderr=response.stderr.replace(str(root), '<work>'),
                    exception=repr(response.exception) if response.exception else None,
                    checker_calls=checker.call_count)
        if path.exists():
            status = json.loads(path.read_text())
            wire['counts'] = status['counts']
            wire['dates'] = [row['date'] for row in status['rows']]
            if command == 'skip-policy-shadow-update':
                wire['band'] = status['shadow_band_hit_rate']
                patch_reader = pytest.MonkeyPatch()
                with patch_reader.context() as patch:
                    patch.setattr(web, 'VALIDATION_DIR', directory.parent / f'{tag}.validation')
                    dashboard_path = web.VALIDATION_DIR / 'skip_policy_shadow_status.json'
                    dump(dashboard_path, status)
                    wire['dashboard'] = web.render_skip_policy_shadow_section(web.load_skip_policy_shadow_status())
                wire['manual'] = CliRunner().invoke(cli.cli, ['skip-policy-shadow-status', '--status-file', str(path)]).stdout
            else:
                for key in ('cycle_state', 'coverage', 'action_items', 'quality_recorded'):
                    wire[key] = status[key]
        calls[tag] = wire
        return wire

    mixed = root / 'mixed'
    for day in ('2026-08-02', '2026-08-03', '2027-03-24'):
        context(mixed, day)
        record(mixed, day, 'hit')
    for tag, date in [('utc-next', '2027-01-01T02:00:00+00:00'), ('et-2359', '2027-01-01T04:59:00+00:00'),
                      ('et-0001', '2027-01-01T05:01:00+00:00')]:
        instant = dt.datetime.fromisoformat(date)
        invoke(f'{tag}-context', instant, mixed, 'shadow-status')
        invoke(f'{tag}-policy', instant, mixed, 'skip-policy-shadow-update', ['--no-reconcile'])
    now = dt.datetime(2027, 4, 1, 12, tzinfo=UTC)
    for year in (2026, 2027, 2031):
        invoke(f'explicit-{year}-context', now, mixed, 'shadow-status', ['--season', str(year)])
        invoke(f'explicit-{year}-policy', now, mixed, 'skip-policy-shadow-update', ['--season', str(year), '--no-reconcile'])
    for name, day in [('old', '2026-08-02'), ('new', '2027-03-24'), ('empty', '2031-04-01'), ('malformed', 'junk')]:
        invoke(f'{name}-results', now, mixed, 'check-results', ['--date', day])
        invoke(f'{name}-policy', now, mixed, 'skip-policy-shadow-update', ['--date', day, '--no-reconcile'])
    invoke('mismatch-policy', now, mixed, 'skip-policy-shadow-update',
           ['--date', '2026-08-02', '--season', '2027', '--no-reconcile'])
    invoke('malformed-override-policy', now, mixed, 'skip-policy-shadow-update',
           ['--date', 'junk', '--season', '2026', '--no-reconcile'])
    manifest = shadow_eval.build_shadow_backfill_manifest(mixed, n_bootstrap=0, hit_checker=lambda *_: 'hit')
    calls['backfill'] = [row['date'] for row in manifest['rows']]

    bad = root / 'bad-filenames'
    context(bad, '2027-xx')
    misdated = context(bad, '2027-03-25')
    dump(misdated, json.loads(misdated.read_text()) | {'date': '2026-09-20'})
    valid_policy = record(bad, '2027-03-25', 'hit')
    record_data = json.loads(valid_policy.read_text())
    dump(valid_policy, record_data | {'date': '2026-09-20'})
    dump(bad / '20270.policy_shadow.json', record_data | {'date': '20270'})
    dump(bad / '2027-xx.policy_shadow.json', record_data | {'date': '2027-xx'})
    invoke('bad-filenames-context', now, bad, 'shadow-status')
    invoke('bad-filenames-results', now, bad, 'check-results', ['--date', '2027-03-25'])
    invoke('bad-filenames-policy', now, bad, 'skip-policy-shadow-update', ['--no-reconcile'])

    incidents = root / 'incidents'
    context(incidents, '2026-08-30', None)
    context(incidents, '2026-08-31', 'hit', paired=False)
    context(incidents, '2026-12-30', None)
    context(incidents, '2026-12-31', None, version=None)
    for i in range(30):
        record(incidents, (dt.date(2026, 8, 1) + dt.timedelta(days=i)).isoformat())
    record(incidents, '2026-09-27', 'hit', contradict=True)
    record(incidents, '2026-12-31', None)
    record(incidents, '2026-12-30', None, contradict=True)
    record(incidents, '2027-01-01', None, defer=True)
    jan3 = dt.datetime(2027, 1, 3, 12, tzinfo=UTC)
    invoke('rollover-context', jan3, incidents, 'shadow-status')
    invoke('rollover-results', jan3, incidents, 'check-results', ['--date', '2027-01-02'])
    invoke('rollover-policy', jan3, incidents, 'skip-policy-shadow-update', ['--date', '2027-01-02'])
    feb = dt.datetime(2027, 2, 15, 12, tzinfo=UTC)
    invoke('outside-horizon-context', feb, incidents, 'shadow-status')
    for instant in (jan3, feb):
        calls[f'health-{instant.date()}'] = [a.__dict__ for a in result_resolution.check(incidents, today=instant.date())]

    ops = root / 'operations'
    record(ops, '2026-12-30', None, contradict=True)
    old = record(ops, '2026-12-31', None)
    record(ops, '2027-01-01', None, defer=True)
    invoke('operations-jan3', jan3, ops, 'skip-policy-shadow-update')
    calls['operations-side-effects'] = {'old_reconciled': json.loads(old.read_text())['shadow_pick_result'],
                                        'pruned': not policy.decision_path('2026-12-30', ops).exists(),
                                        'new_record': policy.decision_path('2027-01-01', ops).exists()}

    ready = root / 'ready'
    context(ready, '2026-09-27', 'hit')
    invoke('ready-context', now, ready, 'shadow-status', ['--min-days', '1'])
    corrupt = root / 'corrupt'
    corruption = context(corrupt, '2026-09-27')
    corruption.write_text('{malformed json')
    invoke('corrupt-context', now, corrupt, 'shadow-status')
    invoke('corrupt-results', now, corrupt, 'check-results', ['--date', '2027-03-24'])
    dump(root / 'summary.json', calls)


def compare(root):
    base, head = (json.loads((root / label / 'summary.json').read_text()) for label in ('base', 'head'))
    assert len(base) == len(head)
    for tag in base:
        before, after = base[tag], head[tag]
        if isinstance(before, dict) and 'exit' in before:
            print(tag, [(r['exit'], r.get('dates'), r.get('cycle_state')) for r in (before, after)])
        else:
            print(tag, repr(before), '=>', repr(after))
    for tag in ['rollover-context', 'rollover-results', 'rollover-policy', 'outside-horizon-context',
                'ready-context', 'corrupt-context', 'corrupt-results']:
        for label, data in [('base', base), ('head', head)]:
            wire = data[tag]
            print(label, tag, {k: wire[k] for k in ('stdout', 'stderr', 'exception', 'counts', 'coverage',
                                                   'action_items', 'manual', 'dashboard') if k in wire})


if __name__ == '__main__':
    raise SystemExit(offline_probe(probe, compare))
