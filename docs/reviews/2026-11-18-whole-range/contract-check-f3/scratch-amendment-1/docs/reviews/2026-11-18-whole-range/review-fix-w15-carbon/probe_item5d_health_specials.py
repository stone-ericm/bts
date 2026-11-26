"""Probe 5d: complete health special-file bounded checks + verify smallest fixes for open_regular FD leak and warn EPIPE exit-120 on a scratch copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

from probe_item5_unusual_files import run_bounded, feed_str, VALID
from probe_item1_parity_and_sequences import pa_row


def check_5c3_fixed_mkdir(root: Path) -> None:
    print("=== 5d-1. Complete all rejected/, status dir, and health special-file bounded checks ===")
    for artifact in ("rejected", "status"):
        for kind in ("fifo", "file", "symlink_dir"):
            w = root / f"art_{artifact}_{kind}"
            cache = w / "data/models/probable_pitcher_lookup.json"
            raw = w / "data/raw"
            (raw / "2025").mkdir(parents=True)
            cache.parent.mkdir(parents=True)
            cache.write_text('{"100": null}')
            (raw / "2025/100.json").write_text(feed_str(VALID))
            folder = cache.parent / "rejected" if artifact == "rejected" else w / "data/health_state/probable_pitcher_cache"
            folder.parent.mkdir(parents=True, exist_ok=True)
            if kind == "fifo":
                os.mkfifo(folder)
            elif kind == "file":
                folder.write_text("blocker")
            elif kind == "symlink_dir":
                real_d = w / "real_dir"
                real_d.mkdir()
                folder.symlink_to(real_d)

            res = run_bounded(["build", str(cache), str(raw)])
            assert res.returncode == 0, (artifact, kind, res.stderr)
            assert json.loads(res.stdout) == {"100": VALID}
            assert cache.read_text() == '{"100": null}'
            assert len(res.stderr.strip().splitlines()) == 2

    for hkind in ("fifo_rec", "dir_rec", "sym_rec", "broken_sym_rec", "unreadable_rec", "fifo_sdir", "sym_sdir", "unreadable_sdir"):
        w = root / f"h_{hkind}"
        w.mkdir(parents=True)
        sdir = w / "probable_pitcher_cache"
        fname = "2026-12-31.invalid_entries.0123456789ab.json"
        try:
            if hkind == "fifo_sdir":
                os.mkfifo(sdir)
            elif hkind == "sym_sdir":
                tgt = w / "other"
                tgt.mkdir()
                sdir.symlink_to(tgt)
            elif hkind == "unreadable_sdir":
                sdir.mkdir()
                sdir.chmod(0o000)
            else:
                sdir.mkdir()
                rec = sdir / fname
                if hkind == "fifo_rec":
                    os.mkfifo(rec)
                elif hkind == "dir_rec":
                    rec.mkdir()
                elif hkind == "sym_rec":
                    tgt = w / "tgt.json"
                    tgt.write_text("{}")
                    rec.symlink_to(tgt)
                elif hkind == "broken_sym_rec":
                    rec.symlink_to(w / "missing.json")
                elif hkind == "unreadable_rec":
                    rec.write_text("{}")
                    rec.chmod(0o000)
            res = run_bounded(["health", str(sdir)])
            assert res.returncode == 0, (hkind, res.stderr)
            alerts = json.loads(res.stdout)
            assert len(alerts) == 1
            assert alerts[0][0] == "WARN" and "status_unreadable" in alerts[0][2]
        finally:
            if hkind == "unreadable_sdir":
                sdir.chmod(0o700)
            elif hkind == "unreadable_rec":
                (sdir / fname).chmod(0o600)
    print("PASS: All rejected/, status dir, and health special-file checks bounded and verified.")


def verify_silent_feature_corruption_under_fd_leak_and_smallest_fixes(root: Path) -> None:
    print("=== 5d-2. Demonstrate silent live feature corruption from open_regular FD leak + verify smallest fixes ===")
    # Show that in a long-lived process (like bts-scheduler) with a directory at probable_pitcher_lookup.json
    # (or directory entries in raw/), repeated calls leak FDs until scan() silently swallows EMFILE and drops valid feeds!
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

# Suppose probable_pitcher_lookup.json is a directory (special cache left in place by W15!),
# and data/raw/2025 holds valid feeds for games 100..110 (giving opp_bullpen_hr_30g == 0.5).
cache.mkdir(parents=True)
rows = json.loads((work / "rows.json").read_text())

results = []
for call_idx in range(70):
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
    cp_head = subprocess.run([sys.executable, "-c", corrupt_code, str(w)], env=env_head, capture_output=True, text=True, timeout=20)
    assert cp_head.returncode == 0, cp_head.stderr
    sim_head = json.loads(cp_head.stdout)
    print(f"  HEAD under repeated calls with dir at cache path: {sim_head}")
    assert sim_head["first_call"] == 0.5 and sim_head["last_call"] is None

    # Now test scratch copy with BOTH smallest fixes applied:
    # Fix 1 (probable_pitcher_status.py): close fd in finally when fstat/fdopen fails
    # Fix 2 (probable_pitcher_cache.py): flush + unbuffered FileIO write when stderr wraps FileIO so EPIPE leaves no dirty buffer
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
    cp_fix = subprocess.run([sys.executable, "-c", corrupt_code, str(w_fix)], env=env_fix, capture_output=True, text=True, timeout=20)
    assert cp_fix.returncode == 0, cp_fix.stderr
    sim_fix = json.loads(cp_fix.stdout)
    print(f"  PATCHED under repeated calls with dir at cache path: {sim_fix}")
    assert sim_fix["num_0_5"] == 70 and sim_fix["num_nan"] == 0

    # Also test EPIPE exit code on PATCHED!
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
    print(f"  PATCHED broken-pipe stderr subprocess: returncode={cp_ep.returncode}, stdout={cp_ep.stdout.strip()!r}")
    assert cp_ep.returncode == 0 and cp_ep.stdout.strip() == "FEATURES_OK rows=1"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe5d_") as tmp:
        root = Path(tmp)
        check_5c3_fixed_mkdir(root)
        verify_silent_feature_corruption_under_fd_leak_and_smallest_fixes(root)
    print("ALL PROBE 5D CHECKS PASSED")


if __name__ == "__main__":
    main()
