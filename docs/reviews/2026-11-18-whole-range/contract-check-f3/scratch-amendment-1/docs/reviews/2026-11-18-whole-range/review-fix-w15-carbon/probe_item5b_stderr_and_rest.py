"""Probe 5b: inspect exact EPIPE stderr behavior (in-process completion vs interpreter finalize) + closed stderr + part 5(b) checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

from probe_item5_unusual_files import run_bounded, test_rejected_status_health_and_fd_leak, feed_str, VALID


def inspect_broken_and_closed_stderr(root: Path) -> None:
    print("=== 5b-1. Inspect real EPIPE pipe vs closed stderr vs feature-run completion ===")
    w = root / "epipe_detail"
    cache = w / "data/models/probable_pitcher_lookup.json"
    raw = w / "data/raw"
    (raw / "2025").mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    cache.write_text('{"100": null}')
    (raw / "2025/100.json").write_text(feed_str(VALID))

    res = run_bounded(["build", str(cache), str(raw), "broken_stderr"])
    print(f"  EPIPE child returncode={res.returncode}, stdout={res.stdout.strip()!r}, stderr={res.stderr.strip()!r}")
    # Notice: did compute._build_probable_pitcher_lookup complete and print stdout AND rewrite cache AND write status?
    assert json.loads(res.stdout) == {"100": VALID}
    assert json.loads(cache.read_text()) == {"100": VALID}
    st = list((w / "data/health_state/probable_pitcher_cache").glob("*.json"))
    assert len(st) == 1

    # Also test real closed sys.stderr (sys.stderr.close()) and real compute_all_features with broken EPIPE stderr (closing/restoring before exit vs leaving dirty buffer at Py_FinalizeEx)!
    code_closed = """
import json, os, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None
work = Path(sys.argv[1])
os.chdir(work)
compute._probable_pitcher_cache_path = work / "data/models/probable_pitcher_lookup.json"
mode = sys.argv[2]
if mode == "closed":
    sys.stderr.close()
elif mode == "devnull_ro":
    # FD opened O_RDONLY assigned to stderr -> EBADF on write!
    fd = os.open("/dev/null", os.O_RDONLY)
    sys.stderr = os.fdopen(fd, "r")
rows = json.loads((work / "rows.json").read_text())
df = pd.DataFrame(rows)
df["pa_id"] = range(len(df))
out = compute.compute_all_features(df)
print(json.dumps({"rows": len(out), "bullpen_col": "opp_bullpen_hr_30g" in out.columns}))
"""
    from probe_item1_parity_and_sequences import pa_row
    for mode in ("closed", "devnull_ro"):
        w2 = root / f"stderr_{mode}"
        (w2 / "data/models").mkdir(parents=True)
        (w2 / "data/raw/2025").mkdir(parents=True)
        (w2 / "data/models/probable_pitcher_lookup.json").write_text('{"100": null}')
        (w2 / "data/raw/2025/100.json").write_text(feed_str(VALID))
        (w2 / "rows.json").write_text(json.dumps([pa_row(100, "2025-04-01", 1, True)]))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(HEAD_SRC)
        cp = subprocess.run([sys.executable, "-c", code_closed, str(w2), mode], env=env, capture_output=True, text=True, timeout=15)
        print(f"  stderr mode={mode}: returncode={cp.returncode}, stdout={cp.stdout.strip()!r}")
        assert cp.returncode == 0
        assert json.loads(cp.stdout)["rows"] == 1


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe5b_") as tmp:
        root = Path(tmp)
        inspect_broken_and_closed_stderr(root)
        test_rejected_status_health_and_fd_leak(root)
    print("ALL PROBE 5B CHECKS PASSED")


if __name__ == "__main__":
    main()
