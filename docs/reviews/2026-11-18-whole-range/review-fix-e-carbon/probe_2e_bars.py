"""Probe 2e: Compare `strategy.effective_pick_bar` across all 24,524 (streak, days, saver, best, trust)
combinations in both regimes (`reach57` and `emax_season_best`) between base 63669c42 and HEAD.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SRC = REPO / ".carbon-review/wr-e/base_63669c42/src"
HEAD_SRC = REPO / "src"

CODE = r"""
import json, sys
from datetime import date, timedelta
from pathlib import Path
import bts
from bts import strategy

expected = Path(sys.argv[1]).resolve()
actual = Path(bts.__file__).resolve()
assert actual.is_relative_to(expected)

mdp = strategy._load_mdp()
end = date(2027, 9, 27)
bars = []
regimes = set()
for days in range(0, 186):
    day_str = (end - timedelta(days=days)).isoformat()
    for streak in range(57):
        regime, _ = strategy._regime(mdp, streak, day_str)
        regimes.add(regime)
        bests = [(streak, "trusted")]
        if regime == strategy.OBJECTIVE_TAIL:
            bests.extend([(56, "trusted"), (56, "untrusted"), (None, None)])
        for saver in (False, True):
            for b_val, b_trust in bests:
                bars.append(strategy.effective_pick_bar(streak, day_str, saver, mdp, b_val, b_trust))
print(json.dumps({"file": str(actual), "regimes": sorted(regimes), "count": len(bars), "bars": bars}))
"""


def run(src: Path) -> dict:
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", CODE, str(src)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": f"{src}:{REPO}"},
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    b = run(BASE_SRC)
    h = run(HEAD_SRC)
    assert b["file"] != h["file"]
    assert b["regimes"] == ["emax_season_best", "reach57"] == h["regimes"]
    assert b["bars"] == h["bars"]
    print(f"effective_pick_bar identical across {h['count']} combinations in regimes {h['regimes']}.")


if __name__ == "__main__":
    main()
