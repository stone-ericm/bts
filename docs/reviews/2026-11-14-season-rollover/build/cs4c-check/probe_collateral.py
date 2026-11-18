"""Check affected existing assertions with an in-process last-field simulation."""

from dataclasses import make_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

import probe_main  # Denies DNS and socket operations before loading application code.

from pytest import MonkeyPatch
from bts import picks
from tests import test_pick_chosen_on_cs5a as encoding
from tests import test_saved_pick_paths_cs5a as paths


def checks(root):
    for kind in encoding.KINDS:
        yield kind, lambda tmp, _: encoding.test_round_trip_and_exact_prior_encoding(tmp, kind)
    yield 'legacy rewrite', lambda tmp, _: encoding.test_legacy_read_rewrite_and_projection(tmp)
    for mode in ['dm', 'public']:
        def archive(tmp, mp):
            world = paths.world.__wrapped__(tmp, mp)
            paths.test_marker_archive_and_grading_saves_preserve_chosen_on(world, mp, mode)

        yield mode, archive


if __name__ == '__main__':
    new_daily = make_dataclass('DailyPick', [('automatic_reading', dict | None, None)],
                              bases=(picks.DailyPick,))
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as outer:
        root = Path(outer)
        for simulated in (False, True):
            failures = []
            with MonkeyPatch.context() as outer_mp:
                if simulated:
                    outer_mp.setattr(picks, 'DailyPick', new_daily)
                for case, check in checks(root):
                    tmp = root / str(simulated) / case
                    tmp.mkdir(parents=True)
                    with MonkeyPatch.context() as mp:
                        try:
                            check(tmp, mp)
                        except AssertionError as exc:
                            site = traceback.extract_tb(exc.__traceback__)[-1]
                            assert simulated and site.lineno in {158, 590}, site
                            failures.append(case)
                            print('last-field assertion breaks:', case, site.name, site.lineno)
            assert len(failures) == (6 if simulated else 0)
