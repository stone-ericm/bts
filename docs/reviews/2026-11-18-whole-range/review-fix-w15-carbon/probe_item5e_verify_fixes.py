"""Probe 5e: demonstrate silent live feature corruption from open_regular FD leak + verify smallest fixes for FD leak and EPIPE exit-120."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

from probe_item5_unusual_files import feed_str, VALID
from probe_item1_parity_and_sequences import pa_row


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe5e_") as root_s:
        root = Path(root_s)
        corrupt_code = """
import json, os, resource, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

work = Path(sys.argv[1])
os.chdir(work)
cache = work / "data/models/probable_pitcher_lookup.json"
compute._probable_pitcher_cache_path = cache

cache.mkdir(parents=True)
rows = json.loads((work / "rows.json").read_text())

results = []
for _ in range(70):
    df = pd.DataFrame(rows)
    df["pa_id"] = range(len(df))
    out = compute.compute_all_features(df)
    val = out.iloc[-1]["opp_bullpen_hr_30g"]
    results.append(None if pd.isna(val) else float(val))

print(json.dumps({"first_call": results[0], "last_call": results[-1], "num_0_5": results.count(0.5), "num_nan": results.count(None)}))
"""
        dates = pd.date_range("2025-04-01", periods=11)
        rows = [pa_row(100 + i, d.strftime("%Y-%m-%d"), 99 if i < 10 else 1, True, hit=i % 2) for i, d in enumerate(dates)]

        w = root / "scheduler_sim"
        (w / "data/raw/2025").mkdir(parents=True)
        for i in range(11):
            (w / "data/raw/2025" / f"{100 + i}.json").write_text(feed_str(VALID))
        (w / "rows.json").write_text(json.dumps(rows))

        env_head = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env_head["PYTHONPATH"] = str(HEAD_SRC)
        cp_head = subprocess.run([sys.executable, "-c", corrupt_code, str(w)], env=env_head, capture_output=True, text=True, timeout=25)
        assert cp_head.returncode == 0, cp_head.stderr
        sim_head = json.loads(cp_head.stdout)
        print(f"HEAD under repeated calls with dir at cache path: {sim_head}")
        assert sim_head["first_call"] == 0.5 and sim_head["last_call"] is None

        patched_src = root / "patched_src"
        shutil.copytree(HEAD_SRC / "bts", patched_src / "bts")

        status_py = patched_src / "bts/probable_pitcher_status.py"
        status_py.write_text(
            status_py.read_text().replace(
                '    with os.fdopen(os.open(path, flags), "rb") as stream:\n'
                '        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):\n'
                '            raise NonRegularFile(path)\n'
                '        yield stream',
                '    fd = os.open(path, flags)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise NonRegularFile(path)\n'
                '        with os.fdopen(fd, "rb", closefd=False) as stream:\n'
                '            yield stream\n'
                '    finally:\n'
                '        os.close(fd)',
            )
        )

        cache_py = patched_src / "bts/features/probable_pitcher_cache.py"
        cache_py.write_text(
            cache_py.read_text().replace(
                'def warn(message):\n'
                '    try:\n'
                '        print(f"WARNING: {message}", file=sys.stderr)\n'
                '    except Exception:\n'
                '        pass',
                'def warn(message):\n'
                '    try:\n'
                '        raw = getattr(getattr(sys.stderr, "buffer", None), "raw", None)\n'
                '        if isinstance(raw, io.FileIO):\n'
                '            sys.stderr.flush()\n'
                '            os.write(raw.fileno(), f"WARNING: {message}\\n".encode(sys.stderr.encoding or "utf-8", "replace"))\n'
                '        else:\n'
                '            print(f"WARNING: {message}", file=sys.stderr)\n'
                '    except Exception:\n'
                '        pass',
            )
        )

        w_fix = root / "scheduler_sim_fixed"
        shutil.copytree(w, w_fix)
        env_fix = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env_fix["PYTHONPATH"] = str(patched_src)
        cp_fix = subprocess.run([sys.executable, "-c", corrupt_code, str(w_fix)], env=env_fix, capture_output=True, text=True, timeout=25)
        assert cp_fix.returncode == 0, cp_fix.stderr
        sim_fix = json.loads(cp_fix.stdout)
        print(f"PATCHED under repeated calls with dir at cache path: {sim_fix}")
        assert sim_fix["num_0_5"] == 70 and sim_fix["num_nan"] == 0

        epipe_code = """
import json, os, sys
import pandas as pd
from pathlib import Path
from bts.features import compute, park_drag
park_drag.get_table = lambda: None
work = Path(sys.argv[1])
os.chdir(work)
compute._probable_pitcher_cache_path = work / "data/models/probable_pitcher_lookup.json"
df = pd.DataFrame(json.loads((work / "rows.json").read_text()))
df["pa_id"] = range(len(df))
out = compute.compute_all_features(df)
sys.stdout.write(f"FEATURES_OK rows={len(out)}\\n")
sys.stdout.flush()
"""
        w_ep = root / "epipe_fixed"
        (w_ep / "data/models").mkdir(parents=True)
        (w_ep / "data/raw/2025").mkdir(parents=True)
        (w_ep / "data/models/probable_pitcher_lookup.json").write_text('{"100": null}')
        (w_ep / "data/raw/2025/100.json").write_text(feed_str(VALID))
        (w_ep / "rows.json").write_text(json.dumps([pa_row(100, "2025-04-01", 1, True)]))
        r_err, w_err = os.pipe()
        os.close(r_err)
        try:
            cp_ep = subprocess.run([sys.executable, "-c", epipe_code, str(w_ep)], env=env_fix, stdout=subprocess.PIPE, stderr=w_err, text=True, timeout=15)
        finally:
            os.close(w_err)
        print(f"PATCHED broken-pipe stderr subprocess: returncode={cp_ep.returncode}, stdout={cp_ep.stdout.strip()!r}")
        assert cp_ep.returncode == 0 and cp_ep.stdout.strip() == "FEATURES_OK rows=1"
        print("ALL PROBE 5E FIX VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
