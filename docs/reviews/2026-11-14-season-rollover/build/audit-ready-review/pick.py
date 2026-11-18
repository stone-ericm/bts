import datetime as dt
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from probe_support import freeze, offline_probe
from review_runner import ROOT, dump


def probe(root, source):
    from bts import cli, daily_decision, picks, shadow_eval
    from tests.test_pick_chosen_on_cs5a import DAY, NOW
    from tests.test_shadow_eval import _daily, _pick

    earlier = ROOT / 'docs/reviews/2026-11-14-season-rollover/build/batch-2026-11-18/review_probes.py'
    runpy.run_path(str(earlier))['pick_path'](root, 'base' if source != ROOT else 'head')
    original = root / 'picks'
    result = root / 'result'
    result.mkdir()
    grade = result / 'picks'
    grade.mkdir()
    prod = json.loads((root / 'pick.json').read_text())
    prod.update(result=None, slot_results=None)
    dump(grade / f'{DAY}.json', prod)
    marker = daily_decision.decision_path(DAY, grade)
    marker.parent.mkdir()
    marker.write_bytes((root / 'decision.json').read_bytes())
    picks.save_shadow_pick(_daily(DAY, _pick('Shadow probe', 8)), grade)
    picks.save_shadow_pick(_daily('2026-09-27', _pick('Historical', 9), result='miss'), grade)

    def result_probe(directory, day):
        args = ['check-results', '--date', day, '--picks-dir', str(directory),
                '--shadow-status-output', str(directory.parent / (directory.name + '.status.json'))]
        response = CliRunner().invoke(cli.cli, args)
        return dict(exit=response.exit_code, stdout=response.stdout.replace(str(root), '<work>'),
                    stderr=response.stderr.replace(str(root), '<work>'),
                    exception=repr(response.exception) if response.exception else None)

    cases = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, '_now_et', lambda: NOW)
        patch.setattr(picks, 'datetime', freeze(NOW))
        patch.setattr(shadow_eval, 'datetime', freeze(NOW))
        patch.setattr(picks, 'get_game_statuses_detailed', lambda *_: {})
        checker = Mock(side_effect=lambda *_, batter_name, **__: 'miss' if batter_name == 'Shadow probe' else 'hit')
        patch.setattr(picks, 'check_hit', checker)
        cases['after-run-day'] = result_probe(original, DAY)
        assert cases['after-run-day']['exit'] == 0 and 'Already resolved' in cases['after-run-day']['stdout']
        assert not checker.called
        cases['unsettled-replay'] = result_probe(grade, DAY)
        assert cases['unsettled-replay']['exit'] == 0, cases
        assert 'HIT!' in cases['unsettled-replay']['stdout'] and 'Shadow: Shadow probe — MISS' in cases['unsettled-replay']['stdout']
        for case in ('absent', 'shadow-only', 'terminal', 'uncommitted', 'scoreable'):
            folder = result / case
            folder.mkdir()
            invalid = 'bad-date'
            picks.save_shadow_pick(_daily(DAY, _pick('Current', 10), result='hit'), folder)
            if case == 'shadow-only':
                picks.save_shadow_pick(_daily(invalid, _pick('Shadow probe', 8)), folder)
            if case in ('terminal', 'uncommitted', 'scoreable'):
                d = _daily(invalid, _pick('Input', 11), result='hit' if case == 'terminal' else None)
                picks.save_pick(d, folder)
                if case == 'scoreable':
                    daily_decision.write_decision(invalid, folder, action='single', source='mdp',
                                                  scoreable=True, delivery_status='delivered', now=NOW)
            cases[case] = result_probe(folder, invalid)
        cases['hit_calls'] = checker.call_args_list
    dump(root / 'results.json', cases)


def compare(root):
    for name in ['pick.json', 'decision.json', 'score.json', 'sends.json', 'results.json']:
        a, b = (root / label / name for label in ('base', 'head'))
        assert a.read_bytes() == b.read_bytes(), name
        print(f'{name}: byte-identical')
    for name in ['picks/2027-07-15.json', *[str(path.relative_to(root / 'head' / 'result'))
                  for path in (root / 'head' / 'result' / 'picks').glob('*streak*.json')]]:
        a, b = (root / label / 'result' / name for label in ('base', 'head'))
        if a.exists():
            assert a.read_bytes() == b.read_bytes(), name
            print(f'CLI {name}: byte-identical')
    print((root / 'head' / 'results.json').read_text())


if __name__ == '__main__':
    raise SystemExit(offline_probe(probe, compare))
