"""Verify 3B (UTF-8 vs ASCII stream encoding with relative paths), 3C (standalone logging route), and 4 (operator texts)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src"
BASE_SRC = HERE / "archives/base/src"
R2_SRC = HERE / "archives/r2/src"
TICK = chr(96)

ENC_CHILD = r"""
import os, sys
from pathlib import Path

os.chdir(sys.argv[1])
from bts.features import compute
Path("data/models").mkdir(parents=True)
Path("data/raw").mkdir(parents=True)
c_utf = Path("data/models/caf\u00e9_\u26a1.json")
c_utf.write_text("{")
compute._probable_pitcher_cache_path = c_utf
compute._build_probable_pitcher_lookup("data/raw")
surr = "surrogate_test:" + chr(0xDCFF) + "_end"
try:
    from bts.features import probable_pitcher_cache as ppc
    c_drop = Path(f"data/models/drop_caf\u00e9_{chr(96)}.json")
    c_drop.write_text('{"100": null}')
    compute._probable_pitcher_cache_path = c_drop
    compute._build_probable_pitcher_lookup("data/raw")
    ppc.warn(surr)
except ImportError:
    print(f"WARNING: {surr}", file=sys.stderr, flush=True)
"""


def run_enc(src_dir: Path, io_enc: str) -> subprocess.CompletedProcess:
    with TemporaryDirectory(dir=HERE, prefix="tmp-enc-") as tmp:
        script = Path(tmp) / "enc.py"
        script.write_text(ENC_CHILD, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script), tmp],
            env={**os.environ, "PYTHONPATH": str(src_dir), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": io_enc},
            capture_output=True,
            timeout=15,
            check=False,
        )


def main():
    print("=== 3B. warn() Stream Encoding Comparison (UTF-8 vs UTF-8:strict vs ASCII) ===")
    for io_enc in ("utf-8", "utf-8:strict", "ascii"):
        b_proc = run_enc(BASE_SRC, io_enc)
        r2_proc = run_enc(R2_SRC, io_enc)
        hd_proc = run_enc(REPO_SRC, io_enc)
        b_err, r2_err, hd_err = b_proc.stderr, r2_proc.stderr, hd_proc.stderr
        print(f"  PYTHONIOENCODING={io_enc} (rcs: base={b_proc.returncode}, r2={r2_proc.returncode}, HEAD={hd_proc.returncode}):")
        print(f"    base: {b_err.splitlines()}")
        print(f"    r2  : {r2_err.splitlines()}")
        print(f"    HEAD: {hd_err.splitlines()}")
        if io_enc == "utf-8":
            assert b_proc.returncode == r2_proc.returncode == hd_proc.returncode == 0
            assert b_err.splitlines()[0] == r2_err.splitlines()[0] == hd_err.splitlines()[0]
            assert b_err.splitlines()[1] == hd_err.splitlines()[2] == b"WARNING: surrogate_test:\\udcff_end"
            assert r2_err.splitlines()[2] == b"WARNING: surrogate_test:?_end"
        elif io_enc == "utf-8:strict":
            assert r2_proc.returncode == hd_proc.returncode == 0
            assert b_err.splitlines()[0] == r2_err.splitlines()[0] == hd_err.splitlines()[0]
            assert len(hd_err.splitlines()) == 2
        elif io_enc == "ascii":
            assert b_proc.returncode == r2_proc.returncode == hd_proc.returncode == 0
            assert b_err.splitlines()[0] == hd_err.splitlines()[0]
            assert b"caf\\xe9_\\u26a1.json" in hd_err.splitlines()[0]
            assert b"caf?_?.json" in r2_err.splitlines()[0]

    print("\n=== 3C. Standalone Process Logging Route (logging.lastResort -> stderr) ===")
    log_route_code = (
        "import json, sys; from datetime import date; from pathlib import Path; "
        "import bts.cli, bts.scheduler; from bts.health import probable_pitcher_cache as h; "
        "st = Path(sys.argv[1]) / 'data/health_state/probable_pitcher_cache'; st.mkdir(parents=True); "
        "(st / '2026-12-31.bad`file.json').write_text('{'); "
        "a, = h.check(st, date(2026, 12, 31)); print(a.message)"
    )
    with TemporaryDirectory(dir=HERE, prefix="tmp-logroute-") as tmp:
        lr = subprocess.run(
            [sys.executable, "-c", log_route_code, tmp],
            env={**os.environ, "PYTHONPATH": str(REPO_SRC), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=15, check=True,
        )
        assert "2026-12-31.bad\\x60file.json" in lr.stderr and TICK not in lr.stderr
        assert "details in this health run's log and in health_state/probable_pitcher_cache beside the picks directory." in lr.stdout
        print("  Standalone process with bts.cli + bts.scheduler imported: log.warning routes to stderr via lastResort, backticks escaped")

    print("\n=== 4. Exact Rendered Operator Texts Added/Changed in W15b ===")
    sys.path.insert(0, str(REPO_SRC))
    from bts.health import probable_pitcher_cache as health
    summary = health.summarize(date(2026, 12, 31), ["a" * 64] * 1000, {"invalid entries", "unreadable cache", "status unreadable"})
    print("  Longest summary (1000 obs, 3 kinds):", summary.message)
    print("  Longest summary len:", len(summary.message))
    print("\nALL 3B-4 PROBES COMPLETED.")


if __name__ == "__main__":
    main()
