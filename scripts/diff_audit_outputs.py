"""Diff two audit output dirs for bit-exactness (feature/strategy exps)
or near-exactness (model experiments, atol=1e-10)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Run-time metadata that's not model output — exclude from diff
SKIP_KEYS = {"timestamp"}

FEATURE_EXACT: set[str] = {
    # If we know the experiment must be bit-exact, list it here; atol=0 in that case
}


def _close(a, b, atol: float) -> tuple[bool, str]:
    if isinstance(a, dict):
        a_keys = set(a.keys()) - SKIP_KEYS
        b_keys = set(b.keys()) - SKIP_KEYS
        if a_keys != b_keys:
            return False, f"key mismatch: {a_keys ^ b_keys}"
        for k in a_keys:
            ok, msg = _close(a[k], b[k], atol)
            if not ok:
                return False, f"{k}: {msg}"
        return True, ""
    if isinstance(a, list):
        if len(a) != len(b):
            return False, f"length {len(a)} vs {len(b)}"
        for ai, bi in zip(a, b):
            ok, msg = _close(ai, bi, atol)
            if not ok:
                return False, msg
        return True, ""
    if isinstance(a, float) or isinstance(b, float):
        try:
            if math.isclose(float(a), float(b), abs_tol=atol):
                return True, ""
            return False, f"|{a} - {b}| > {atol}"
        except Exception:
            return False, f"non-numeric: {a} vs {b}"
    return (a == b, "equal" if a == b else f"{a} != {b}")


def main():
    a_root, b_root = Path(sys.argv[1]), Path(sys.argv[2])
    diffs = []
    exps_seen = 0
    for diff_path in sorted(a_root.glob("phase1/*/diff.json")):
        exp_name = diff_path.parent.name
        exps_seen += 1
        a = json.loads(diff_path.read_text())
        b_path = b_root / "phase1" / exp_name / "diff.json"
        if not b_path.exists():
            diffs.append((exp_name, "MISSING in run B"))
            continue
        b = json.loads(b_path.read_text())
        # atol: 1e-10 across the board — tight but allows LGBM/FP accumulation noise
        ok, msg = _close(a, b, atol=1e-10)
        if not ok:
            diffs.append((exp_name, msg))

    if diffs:
        print(f"DIFFS DETECTED in {len(diffs)}/{exps_seen} experiments:")
        for name, msg in diffs[:20]:
            print(f"  {name}: {msg}")
        sys.exit(1)
    print(f"All {exps_seen} experiments match within 1e-10")


if __name__ == "__main__":
    main()
