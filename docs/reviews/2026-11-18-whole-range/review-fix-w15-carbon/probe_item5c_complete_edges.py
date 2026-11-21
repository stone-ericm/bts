"""Probe 5c: isolate directory vs FIFO FD leak count, complete all 5(b) special-file checks, and demonstrate compute_all_features exit-120 on real broken-pipe stderr."""

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

from probe_item5_unusual_files import run_bounded, feed_str, VALID
from probe_item1_parity_and_sequences import pa_row


def check_fd_leak_by_kind(root: Path) -> None:
    print("=== 5c-1. Isolate FD leak: directory vs FIFO in open_regular ===")
    code = """
import json, os, sys
from pathlib import Path
from bts.probable_pitcher_status import open_regular

target = Path(sys.argv[1])
kind = sys.argv[2]
p = target / kind
if kind == "dir":
    p.mkdir()
else:
    os.mkfifo(p)

def count_fds():
    n = 0
    for fd in range(3, 512):
        try:
            os.fstat(fd)
            n += 1
        except OSError:
            pass
    return n

b = count_fds()
for _ in range(25):
    try:
        with open_regular(p):
            pass
    except OSError:
        pass
a = count_fds()
print(json.dumps({"kind": kind, "before": b, "after": a, "leaked": a - b}))
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    for kind in ("dir", "fifo"):
        w = root / f"fd_{kind}"
        w.mkdir()
        cp = subprocess.run([sys.executable, "-c", code, str(w), kind], env=env, capture_output=True, text=True, timeout=5)
        assert cp.returncode == 0, cp.stderr
        info = json.loads(cp.stdout)
        print(f"  open_regular({kind}) x 25 -> {info}")

    # Now demonstrate actual EMFILE crash when RLIMIT_NOFILE is reached via directory entries in data/raw/2025/ or repeated calls!
    emfile_code = """
import json, os, resource, sys
from pathlib import Path
from bts.features import compute

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

work = Path(sys.argv[1])
cache = work / "data/models/probable_pitcher_lookup.json"
raw = work / "data/raw"
(raw / "2025").mkdir(parents=True)
cache.parent.mkdir(parents=True)
cache.write_text('{"999": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}')
# Put 70 directories named <digits>.json inside data/raw/2025
for i in range(100, 170):
    (raw / "2025" / f"{i}.json").mkdir()

compute._probable_pitcher_cache_path = cache
compute._build_probable_pitcher_lookup(str(raw))
# Now try to open a completely normal file afterwards (as scheduler/compute_all_features does!)
probe_file = work / "normal.txt"
probe_file.write_text("ok")
print("AFTER_SCAN_OPEN_OK")
"""
    w_em = root / "emfile_demo"
    w_em.mkdir()
    cp_em = subprocess.run([sys.executable, "-c", emfile_code, str(w_em)], env=env, capture_output=True, text=True, timeout=5)
    print(f"  EMFILE demo returncode={cp_em.returncode}, stdout={cp_em.stdout.strip()!r}, stderr_last={cp_em.stderr.strip().splitlines()[-1] if cp_em.stderr.strip() else ''!r}")


def check_epipe_compute_all_features(root: Path) -> None:
    print("=== 5c-2. Real broken-pipe stderr during compute_all_features ===")
    code = """
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
    w = root / "epipe_feat"
    (w / "data/models").mkdir(parents=True)
    (w / "data/raw/2025").mkdir(parents=True)
    (w / "data/models/probable_pitcher_lookup.json").write_text('{"100": null}')
    (w / "data/raw/2025/100.json").write_text(feed_str(VALID))
    (w / "rows.json").write_text(json.dumps([pa_row(100, "2025-04-01", 1, True)]))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    r_err, w_err = os.pipe()
    os.close(r_err)  # Broken stderr pipe from the start, just like `cmd 2>&1 | closed_consumer`!
    try:
        cp = subprocess.run(
            [sys.executable, "-c", code, str(w)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=w_err,
            text=True,
            timeout=15,
        )
    finally:
        os.close(w_err)
    print(f"  Real broken-pipe stderr subprocess: returncode={cp.returncode}, stdout={cp.stdout.strip()!r}")


def check_remaining_5b(root: Path) -> None:
    print("=== 5c-3. Complete all rejected/, status dir, and health special-file checks ===")
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
            assert cache.read_text() == '{"100": null}', "active cache must stay untouched"
            assert len(res.stderr.strip().splitlines()) == 2

    for hkind in ("fifo_rec", "dir_rec", "sym_rec", "broken_sym_rec", "unreadable_rec", "fifo_sdir", "sym_sdir", "unreadable_sdir"):
        w = root / f"h_{hkind}"
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe5c_") as tmp:
        root = Path(tmp)
        check_fd_leak_by_kind(root)
        check_epipe_compute_all_features(root)
        check_remaining_5b(root)
    print("ALL PROBE 5C CHECKS COMPLETED")


if __name__ == "__main__":
    main()
