import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from probe_support import DAY, bts, contest_state, health, observed, scratch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sweep():
    data = []
    with scratch() as directory:
        picks = directory / "picks"
        observed(picks, state="not_earned", streak=0)
        # A bounded outer subprocess catches any accidental read through the live saver loader.
        os.mkfifo(picks / "account_state/saver_state.json")
        for streak in range(contest_state.MAX_SEASON_STREAK + 1):
            observed(picks, state="not_earned", streak=streak)
            for mode in (None, DAY.date()):
                rows = health.check(picks, now=DAY, expected=True, historical_day=mode)
                assert len(rows) == int(10 <= streak <= 15), (streak, mode, rows)
                if rows:
                    assert rows[0].source == "saver_state" and rows[0].level == "WARN"
                    row = asdict(rows[0])
                    row["message"] = row["message"].replace(str(picks), "PICKS_DIR")
                    data.append((streak, bool(mode), row))
    return len(range(contest_state.MAX_SEASON_STREAK + 1)), hashlib.sha256(json.dumps(data).encode()).hexdigest()


if __name__ == "__main__" and sys.argv[1:] == ["child"]:
    print(f"bts.__file__ = {bts.__file__}")
    count, digest = sweep()
    print(f"{count} streaks, live and historical; warning digest {digest}")
elif __name__ == "__main__":
    results = []
    for source in (HERE / "base/src", REPO / "src"):
        run = subprocess.run([sys.executable, __file__, "child"], cwd=REPO,
                             env={**os.environ, "PYTHONPATH": f"{source}:{REPO}"},
                             timeout=10, capture_output=True, text=True, check=True)
        print(run.stdout, end="")
        results.append(run.stdout.splitlines()[-1])
    assert results[0] == results[1]
