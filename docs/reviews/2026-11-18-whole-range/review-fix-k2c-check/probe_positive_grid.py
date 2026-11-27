import hashlib
import runpy
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path.cwd()
DOCS = ROOT / "docs/reviews/2026-11-18-whole-range"
sys.path.insert(0, str(DOCS / "review-fix-k2b-check"))
from run_tests import offline

sys.addaudithook(offline)
import bts
from bts import strategy

source = Path(sys.argv[1]).resolve()
assert Path(bts.__file__).resolve() == source / "bts" / "__init__.py"
print("source:", bts.__file__, flush=True)
fixture = runpy.run_path(str(DOCS / "contract-check-k2b/probe_policy.py"))
row, bar = fixture["row"], fixture["bar"]
artifacts = fixture["ARTIFACTS"]
boundaries = artifacts["boundaries"]
probabilities = sorted({0., .8, 1., *boundaries, *(edge - 1e-9 for edge in boundaries)})
print("artifacts:", artifacts["base_sha256"], artifacts["tail"].sha256)
digest = hashlib.sha256()
count = 0
end = date(2027, 9, 27)
for remaining in (1, 2, 3, 4, 7, 14, 30, 90, 180, 269):
    day = (end - timedelta(days=remaining)).isoformat()
    assert strategy._days_remaining(day) == remaining
    for streak in fixture["STREAKS"]:
        for saver in (False, True):
            displayed = bar(day, streak, saver, streak, "trusted")
            for p in probabilities:
                actual = row(day, streak, saver, p, streak, "trusted")
                digest.update(repr((day, streak, saver, p, actual, displayed)).encode())
                count += 1
            if streak + 2 * remaining < 57:
                stop = streak + 2 * remaining
                for best, status in ((stop - 1, "trusted"), (stop, "trusted"), (57, "untrusted"), (None, None)):
                    displayed = bar(day, streak, saver, best, status)
                    actual = row(day, streak, saver, .8, best, status)
                    digest.update(repr((day, streak, saver, actual, displayed)).encode())
                    count += 1
print(count, digest.hexdigest())
