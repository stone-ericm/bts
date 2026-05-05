"""Compute BH/BY FDR-adjusted q-values over realized-picks Cut C.

SOTA tracker item #7 P0. p-value FDR baseline (NOT e-BH) per Codex bus
#225/#227. Reads the canonical realized-picks artifact, computes per-cell
Poisson-binomial p-values under H0=calibrated, applies BH and BY across the
Cut C family, emits a JSON artifact with methodology metadata + per-cell
q-values + a stdout table for CLI inspection.

Usage:
  uv run --extra model python scripts/run_realized_picks_fdr.py
  uv run --extra model python scripts/run_realized_picks_fdr.py \\
    --input data/validation/realized_picks_canonical_2026-05-05_p1.parquet \\
    --output data/validation/realized_picks_fdr_2026-05-05.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from bts.validate.fdr import (
    bh_qvalues,
    by_qvalues,
    cell_pvalues_from_artifact,
)


DEFAULT_INPUT = Path("data/validation/realized_picks_canonical_2026-05-05_p1.parquet")
DEFAULT_OUTPUT = Path("data/validation/realized_picks_fdr_2026-05-05.json")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="canonical realized-picks parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="output JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input does not exist: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.input)
    print(f"loaded {len(df)} rows from {args.input}")

    extracted = cell_pvalues_from_artifact(df)
    cells = extracted["cells"]
    m = extracted["m"]
    print(
        f"family size m = {m}; "
        f"excluded {extracted['excluded_pending_rows']} pending rows + "
        f"{extracted['excluded_na_rows']} NA-key rows"
    )

    if m == 0:
        print("no testable cells; emitting empty artifact", file=sys.stderr)
        q_bh = q_by = np.zeros(0, dtype=float)
    else:
        pvalues = np.array([c["p_two_sided"] for c in cells])
        q_bh = bh_qvalues(pvalues)
        q_by = by_qvalues(pvalues)

    for i, c in enumerate(cells):
        c["q_bh"] = float(q_bh[i])
        c["q_by"] = float(q_by[i])

    methodology = {
        "method": "p-value FDR baseline (BH + BY) over realized-picks Cut C",
        "p_value_construction": "Poisson-binomial under heterogeneous H0; "
        "two-sided cap = min(1, 2*min(p_lower, p_upper))",
        "tail_direction_convention": (
            "overconfidence iff p_lower < p_upper "
            "(observed hits LOW vs expected)"
        ),
        "bh_dependence_assumption": "PRDS",
        "by_dependence_assumption": "arbitrary (with c(m) harmonic penalty)",
        "family_scope": "cut_c_all_regimes_nonempty_non_na",
        "deploy_gate": None,
        "scipy_version": scipy.__version__,
        "git_head_sha": _git_head_sha(),
        "input_path": str(args.input),
        "input_file_sha256": _file_sha256(args.input),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "m": m,
        "excluded_pending_rows": extracted["excluded_pending_rows"],
        "excluded_na_rows": extracted["excluded_na_rows"],
        "notes": (
            "p-value FDR baseline only; does NOT close true e-BH or SAVI. "
            "Per Codex bus #225/#227: 1/p is not a valid universal p-to-e "
            "calibrator (Wang & Ramdas 2022). Valid e-values remain deferred."
        ),
    }
    payload = {"methodology": methodology, "cells": cells}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {args.output}")

    print()
    print("=" * 110)
    print("PER-CELL FDR TABLE — Cut C")
    print("=" * 110)
    print(
        f"{'regime':<28} {'slot':<13} {'env':<18} {'Q':<3} {'n':>3} {'hits':>4} "
        f"{'p2':>7} {'q_bh':>7} {'q_by':>7} {'tail':<16}"
    )
    print("-" * 110)
    cells_sorted = sorted(
        cells,
        key=lambda c: (c["regime"], c["slot"], not c["is_park_driven"], c["batter_skill_quartile"]),
    )
    for c in cells_sorted:
        env_label = "park_driven" if c["is_park_driven"] else "not_park_driven"
        print(
            f"{c['regime']:<28} {c['slot']:<13} {env_label:<18} "
            f"Q{c['batter_skill_quartile']:<2} {c['n']:>3} {c['hits']:>4} "
            f"{c['p_two_sided']:>7.4f} {c['q_bh']:>7.4f} {c['q_by']:>7.4f} "
            f"{c['tail_direction']:<16}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
