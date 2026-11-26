from contextlib import redirect_stdout
import datetime as dt
import io
import json
from pathlib import Path

import pytest

from probe_support import freeze, offline_probe
from review_runner import dump


def probe(root, source):
    from bts import picks, shadow_eval
    from scripts.audit import p03_context_shadow_closeout as script
    from tests.test_shadow_eval import _daily, _pick

    assert Path(script.__file__).resolve().is_relative_to(source / 'scripts')
    snapshot = root / 'final-20260928'
    directory = snapshot / 'data/picks'
    raw = root / 'raw'
    for day, prod, shadow in [('2026-09-14', 'miss', 'hit'), ('2026-09-27', 'hit', 'miss')]:
        picks.save_pick(_daily(day, _pick('Prod', 1), result=prod), directory)
        picks.save_shadow_pick(_daily(day, _pick('Shadow', 2), result=shadow), directory)
    for game in (800001, 800002):
        dump(raw / '2026' / f'{game}.json', {'gameData': {'status': {'abstractGameCode': 'I'}}})
    (root / 'final-20260928.sha256').write_text('offline frozen fixture manifest\n')
    for year in (2026, 2027):
        output = root / f'{year}.json'
        stdout = io.StringIO()
        instant = dt.datetime(year, 11, 18, 17, tzinfo=dt.timezone.utc)
        with pytest.MonkeyPatch.context() as patch, redirect_stdout(stdout):
            clock = freeze(instant)
            patch.setattr(dt, 'datetime', clock)
            patch.setattr(shadow_eval, 'datetime', clock)
            assert script.main(['--snapshot', str(snapshot), '--raw-dir', str(raw), '--out', str(output)]) == 0
        (root / f'{year}.txt').write_text(stdout.getvalue())
        report = json.loads(output.read_text())
        assert report['recorded']['n_evaluable_days'] == 2
        assert report['recompute_crosscheck']['api_calls'] == 0


def compare(root):
    for suffix in ('json', 'txt'):
        outputs = [(root / label / f'{year}.{suffix}').read_text().replace(str(root / label), '<work>')
                   for label in ('base', 'head') for year in (2026, 2027)]
        assert len(set(outputs)) == 1, suffix
        print(f'2026 and 2027 on both revisions: {suffix} identical')
    print((root / 'head/2027.txt').read_text())


if __name__ == '__main__':
    raise SystemExit(offline_probe(probe, compare))
