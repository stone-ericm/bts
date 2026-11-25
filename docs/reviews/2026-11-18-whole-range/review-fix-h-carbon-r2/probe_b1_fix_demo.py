"""Demonstrate the smallest fix for B1 in a scratch copy of src/bts/strategy.py."""
from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


OLD_BLOCK = """    if not isinstance(boundaries, list):
        raise ValueError("base policy boundaries must be a list")
    if not isinstance(table, np.ndarray) or table.ndim != 4:"""

NEW_BLOCK = """    if (not isinstance(boundaries, list)
            or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not np.isfinite(b)
                   for b in boundaries)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("base policy boundaries must be finite strictly-increasing numbers")
    if not isinstance(table, np.ndarray) or np.ma.isMaskedArray(table) or table.ndim != 4:"""


def main() -> None:
    with tempfile.TemporaryDirectory(dir=HERE, prefix="scratch-b1-fix-") as tmp:
        sroot = Path(tmp)
        shutil.copytree(REPO / "src", sroot / "src")
        strat_file = sroot / "src/bts/strategy.py"
        txt = strat_file.read_text()
        assert OLD_BLOCK in txt
        strat_file.write_text(txt.replace(OLD_BLOCK, NEW_BLOCK, 1))

        code = f"""
import sys
from datetime import date
from pathlib import Path
import tempfile
import numpy as np

sys.path[:0] = [{str(sroot / 'src')!r}, {str(REPO)!r}, {str(HERE)!r}]
import bts
assert Path(bts.__file__).resolve() == Path({str(sroot / 'src/bts/__init__.py')!r}).resolve()

from bts import strategy
from bts.simulate import mdp
from probe_item2_base_and_h4 import _rebind_pair, _run_scheduler_delivery

with np.load(mdp.DEFAULT_POLICY_PATH) as f:
    shipped_base = dict(f)
shipped_bounds = shipped_base["boundaries"]
shipped_table = shipped_base["policy_table"]

with tempfile.TemporaryDirectory(dir={str(sroot)!r}, prefix="cases-") as ctmp:
    croot = Path(ctmp)
    bad_cases = {{
        "bounds_str_1d": shipped_base | {{"boundaries": np.array(["0.76", "0.78", "0.80", "0.82"])}},
        "bounds_2d_col": shipped_base | {{"boundaries": shipped_bounds.reshape(4, 1)}},
        "bounds_2d_empty_cols": shipped_base | {{"boundaries": np.zeros((4, 0), dtype=float)}},
        "bounds_complex_1d": shipped_base | {{"boundaries": np.array([0.76 + 0j, 0.78 + 0j, 0.80 + 0j, 0.82 + 0j])}},
        "bounds_bytes_1d": shipped_base | {{"boundaries": np.array([b"a", b"b", b"c", b"d"])}},
        "bounds_struct_1d": shipped_base | {{"boundaries": np.zeros(4, dtype=[("cut", "f8")])}},
        "bounds_nan": shipped_base | {{"boundaries": np.array([np.nan, 0.78, 0.80, 0.82])}},
        "bounds_inf": shipped_base | {{"boundaries": np.array([np.inf, np.inf, np.inf, np.inf])}},
        "bounds_unsorted": shipped_base | {{"boundaries": np.array([0.85, 0.75, 0.80, 0.82])}},
        "bounds_dup": shipped_base | {{"boundaries": np.array([0.80, 0.80, 0.81, 0.82])}},
        "bounds_bool_1d": shipped_base | {{"boundaries": np.array([False, True, False, True])}},
    }}
    for name, bdict in bad_cases.items():
        kdir = croot / name
        kdir.mkdir()
        kb, kt, _, _ = _rebind_pair(kdir, bdict)
        out = _run_scheduler_delivery(kdir, kb, kt, date(2027, 9, 26))
        assert out["action"] == "single" and out["double_down"] is None, (name, out)
        assert out["tail_sha"] is None and out["degraded_reason"].endswith("); forced single"), (name, out)
        assert "Batter 0" in out["dm"] and "Batter 1" not in out["dm"], (name, out)

    # MaskedArray in memory also forced single now
    shipped_ctx = strategy._load_mdp()
    raw_bad = shipped_table.copy()
    raw_bad[56, 1, 0, 4] = 99
    mask = np.zeros(shipped_table.shape, dtype=bool)
    mask[56, 1, 0, 4] = True
    masked_arr = np.ma.masked_array(raw_bad, mask=mask)
    m_dec = strategy.resolve_policy_decision(
        strategy.DecisionContext(0.99, 0.98, True, "2027-09-26", True, shipped_ctx | {{"policy_table": masked_arr}}, 57, "trusted"),
        57, False,
    )
    assert m_dec.action == "single" and m_dec.tail_sha256 is None and m_dec.degraded_reason.endswith("); forced single")

    # Healthy controls still pass (shipped, short table, 1-bin 0-boundary table)
    for hname, hdict, day, want_act in [
        ("shipped", shipped_base, date(2027, 9, 26), "double"),
        ("short_923", shipped_base | {{"policy_table": shipped_table[:, :2], "season_length": np.array(1)}}, date(2027, 9, 23), "single"),
        ("short_926", shipped_base | {{"policy_table": shipped_table[:, :2], "season_length": np.array(1)}}, date(2027, 9, 26), "double"),
        ("one_bin", shipped_base | {{"policy_table": shipped_table[:, :, :, :1], "boundaries": np.array([], dtype=float)}}, date(2027, 9, 26), "double"),
    ]:
        hdir = croot / hname
        hdir.mkdir()
        hb, ht, _, htail = _rebind_pair(hdir, hdict)
        hout = _run_scheduler_delivery(hdir, hb, ht, day)
        assert hout["action"] == want_act and hout["tail_sha"] == htail.sha256 and hout["degraded_reason"] is None, (hname, hout)

print("SCRATCH_FIX_VERIFIED")
"""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=60)
        assert "SCRATCH_FIX_VERIFIED" in proc.stdout
        print("OK: Scratch fix turned all 11 malformed-boundaries cases + MaskedArray into forced single while keeping all healthy controls green")


if __name__ == "__main__":
    main()
