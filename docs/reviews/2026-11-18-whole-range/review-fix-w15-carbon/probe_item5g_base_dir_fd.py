"""Probe 5g: empirically compare open FD count on f02e6b9c (base) vs HEAD across 70 calls with a directory at probable_pitcher_lookup.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
BASE_SRC = REVIEW_DIR / "base_src" / "src"
HEAD_SRC = REPO_ROOT / "src"

from probe_item5_unusual_files import feed_str, VALID
from probe_item1_parity_and_sequences import pa_row


def main() -> None:
    code = """
import json, os, resource, sys
import pandas as pd
from pathlib import Path
import bts
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

work = Path(sys.argv[1])
os.chdir(work)
cache = work / "data/models/probable_pitcher_lookup.json"
compute._probable_pitcher_cache_path = cache
cache.mkdir(parents=True, exist_ok=True)
# Make cache non-empty directory so base's os.replace(tmp, dir) fails with OSError/IsADirectoryError or test scan-only (empty raw vs populated raw)
(cache / "keep").write_text("x")

def count_fds():
    n = 0
    for fd in range(3, 128):
        try:
            os.fstat(fd)
            n += 1
        except OSError:
            pass
    return n

fds_before = count_fds()
rows = json.loads((work / "rows.json").read_text())
vals = []
errs = 0
for _ in range(70):
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    try:
        out = compute.compute_all_features(df)
        v = out.iloc[-1]["opp_bullpen_hr_30g"]
        vals.append(None if pd.isna(v) else float(v))
    except OSError:
        # On base, os.replace(temp, non_empty_dir) raises IsADirectoryError/OSError after scanning,
        # while on empty raw_dir base doesn't even try os.replace!
        errs += 1
fds_after = count_fds()
print(json.dumps({
    "bts": bts.__file__,
    "fds_before": fds_before,
    "fds_after": fds_after,
    "leaked_fds": fds_after - fds_before,
    "num_0_5": vals.count(0.5),
    "num_nan": vals.count(None),
    "os_errors": errs,
}))
"""
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [pa_row(100 + i, d.strftime("%Y-%m-%d"), 99 if i < 10 else 1, True, hit=i % 2) for i, d in enumerate(dates)]

    with tempfile.TemporaryDirectory(prefix="probe5g_") as tmp:
        root = Path(tmp)
        for which, src in [("base", BASE_SRC), ("head", HEAD_SRC)]:
            w = root / which
            (w / "data/raw/2025").mkdir(parents=True)
            for i in range(11):
                (w / "data/raw/2025" / f"{100 + i}.json").write_text(feed_str(VALID))
            (w / "rows.json").write_text(json.dumps(rows))
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = str(src)
            cp = subprocess.run([sys.executable, "-c", code, str(w)], env=env, capture_output=True, text=True, timeout=25)
            assert cp.returncode == 0, (which, cp.stderr)
            print(f"{which}: {cp.stdout.strip()}")

        # Also compare directory at raw/2025/9999.json (with valid cache, where base completes 100% with 0 errors and 0 leaked FDs!)
        raw_dir_code = """
import json, os, resource, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

work = Path(sys.argv[1])
os.chdir(work)
compute._probable_pitcher_cache_path = work / "data/models/probable_pitcher_lookup.json"

def count_fds():
    return sum(1 for fd in range(3, 128) if _is_open(fd))
def _is_open(fd):
    try:
        os.fstat(fd); return True
    except OSError:
        return False

fds_before = count_fds()
rows = json.loads((work / "rows.json").read_text())
vals = []
for _ in range(70):
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    out = compute.compute_all_features(df)
    v = out.iloc[-1]["opp_bullpen_hr_30g"]
    vals.append(None if pd.isna(v) else float(v))
fds_after = count_fds()
print(json.dumps({"leaked_fds": fds_after - fds_before, "num_0_5": vals.count(0.5), "num_nan": vals.count(None)}))
"""
        for which, src in [("base", BASE_SRC), ("head", HEAD_SRC)]:
            w = root / f"rawdir_{which}"
            (w / "data/models").mkdir(parents=True)
            (w / "data/raw/2025").mkdir(parents=True)
            # Game 9999.json is a directory! Games 100..110 are valid raw feeds (not cached yet, or 109 uncached + status blocked? No: suppose 9999.json is a directory and 100..110 are raw feeds while cache is read-only or 100..109 cached + 9999.json dir!)
            # Even better: cache has {"999": null} (so 100..110 are scanned each time if status dir is blocked, OR cache is missing and 9999.json is a dir + 70 dir entries 200..269.json in raw/2025 in ONE SINGLE CALL!)
            for d_id in range(200, 265):
                (w / "data/raw/2025" / f"{d_id}.json").mkdir()
            for i in range(11):
                (w / "data/raw/2025" / f"{100 + i}.json").write_text(feed_str(VALID))
            # Wait: 200..264 sort AFTER 100..110! Put directory entries 10..74.json so they sort BEFORE 100..110!
            for d_id in range(10, 75):
                (w / "data/raw/2025" / f"{d_id}.json").mkdir()
            (w / "rows.json").write_text(json.dumps(rows))
            one_call_code = """
import json, os, resource, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
work = Path(sys.argv[1])
os.chdir(work)
compute._probable_pitcher_cache_path = work / "data/models/probable_pitcher_lookup.json"
df = pd.DataFrame(json.loads((work / "rows.json").read_text()))
df["pa_id"] = range(len(df))
try:
    out = compute.compute_all_features(df)
    v = out.iloc[-1]["opp_bullpen_hr_30g"]
    print(json.dumps({"status": "ok", "bullpen": None if pd.isna(v) else float(v)}))
except Exception as e:
    print(json.dumps({"status": "err", "exc": type(e).__name__, "msg": str(e)}))
"""
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = str(src)
            cp = subprocess.run([sys.executable, "-c", one_call_code, str(w)], env=env, capture_output=True, text=True, timeout=25)
            assert cp.returncode == 0, (which, cp.stderr)
            print(f"one_call_with_numeric_json_dirs ({which}): {cp.stdout.strip()}")


if __name__ == "__main__":
    main()
