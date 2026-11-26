#!/usr/bin/env python3
"""v2_5_attribution.py — T4: nested factorial attribution for the v1→v2 verdict shift.

Given 6 verdict JSONs (one per ablation cell in the 3-bit design), computes a
nested factorial decomposition that attributes the improvement from v1 (cell 000)
to v2 (cell 111) to specific methodology changes.

Cell encoding (bit 2=A=params_mode, bit 1=B=rho_pair_mode, bit 0=C=policy_mode):
    000: pooled  + scalar  + global   (v1 baseline)
    010: pooled  + per-bin + global   (T3 ablation)
    001: pooled  + scalar  + per-fold (T3 ablation)
    011: pooled  + per-bin + per-fold (T3 ablation)
    101: fold-local + scalar + per-fold (T3 ablation)
    111: fold-local + per-bin + per-fold (v2 baseline)

The design is NESTED: A (params_mode) is only meaningful when C=per-fold (policy_mode).
This means 100 and 110 cells were never run, so a full symmetric Shapley decomposition
is not possible. Instead, a nested path-based attribution is used.

Usage:
    python scripts/v2_5_attribution.py [--v000-path PATH] [--v111-path PATH] \\
        [--cell-010-path PATH] [--cell-001-path PATH] \\
        [--cell-011-path PATH] [--cell-101-path PATH] \\
        [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path


# ---- Parsing ----------------------------------------------------------------

def parse_p57_string(s: str) -> tuple[float, float, float]:
    """Parse '0.0333 [0.0000, 0.1167]' into (point, ci_lo, ci_hi).

    Robust to '0.0333' (no CI) → returns (0.0333, nan, nan).
    """
    s = s.strip()
    m = re.fullmatch(
        r"([0-9.eE+\-]+)\s*\[([0-9.eE+\-]+),\s*([0-9.eE+\-]+)\]",
        s,
    )
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    # Try bare float (no CI brackets)
    try:
        return float(s), math.nan, math.nan
    except ValueError:
        raise ValueError(
            f"Cannot parse p57 string {s!r}. "
            "Expected format: '0.0333 [0.0000, 0.1167]' or '0.0333'."
        )


# ---- Core decomposition ------------------------------------------------------

def compute_attribution(verdicts: dict[str, float]) -> dict:
    """Compute nested factorial decomposition for the v1→v2 verdict shift.

    Parameters
    ----------
    verdicts:
        Dict mapping cell labels to point estimates.
        Required keys: '000', '010', '001', '011', '101', '111'.

    Returns
    -------
    dict with keys:
        - decomposition: the 5 attributed effects
        - sanity_check: whether the 4 main effects sum to total and the residual
        - interpretation_notes: plain-text explanations

    Design note: A (params_mode = fold-local vs pooled) is nested under C=per-fold
    because fold-local params only make sense when the policy uses per-fold MDP solves.
    The 100 and 110 cells (fold-local + global policy) were never run. This means
    the decomposition is NOT a symmetric Shapley average; instead it uses a
    path-based nested structure.

    The 4 main-path effects (global_B + policy_switch + A_given_per_fold +
    nested_AB_interaction) do NOT always sum to total in general. The residual
    is tracked explicitly. If it is zero (or near-zero), the path attribution is
    exact. If nonzero, B_given_per_fold is the 'missing' effect.
    """
    required = {"000", "010", "001", "011", "101", "111"}
    missing = required - set(verdicts.keys())
    if missing:
        raise ValueError(
            f"Missing required cell(s) from verdicts: {sorted(missing)}. "
            f"Provided: {sorted(verdicts.keys())}."
        )

    v000 = verdicts["000"]
    v010 = verdicts["010"]
    v001 = verdicts["001"]
    v011 = verdicts["011"]
    v101 = verdicts["101"]
    v111 = verdicts["111"]

    total = v111 - v000

    # B's effect in the global-policy branch (C=0)
    global_b_effect = v010 - v000

    # C's (policy switch) effect, averaged over B states
    # = avg(per-fold branch) - avg(global branch)
    policy_switch_effect = (v001 + v011) / 2.0 - (v000 + v010) / 2.0

    # A's (params_mode) effect within the per-fold branch, averaged over B
    # A is only defined for C=per-fold cells (001→101 and 011→111)
    a_effect_given_per_fold = 0.5 * ((v101 - v001) + (v111 - v011))

    # B's effect within the per-fold branch, averaged over A states
    b_effect_given_per_fold = 0.5 * ((v011 - v001) + (v111 - v101))

    # AB interaction within the per-fold branch
    nested_ab_interaction = v111 - v101 - v011 + v001

    # Sanity: do the 4 path effects (excluding b_given_per_fold) sum to total?
    # This holds when: -v000/2 + v001 + v010/2 - v011 - v101/2 + v111/2 = 0
    # (not guaranteed in general — tracked explicitly as residual)
    four_term_sum = (
        global_b_effect
        + policy_switch_effect
        + a_effect_given_per_fold
        + nested_ab_interaction
    )
    residual = four_term_sum - total
    effects_sum_to_total = abs(residual) < 1e-9

    decomposition = {
        "total_v1_to_v2": round(total, 6),
        "global_B_effect": round(global_b_effect, 6),
        "policy_switch_effect": round(policy_switch_effect, 6),
        "A_effect_given_per_fold": round(a_effect_given_per_fold, 6),
        "B_effect_given_per_fold": round(b_effect_given_per_fold, 6),
        "nested_AB_interaction": round(nested_ab_interaction, 6),
    }

    sanity_check = {
        "effects_sum_to_total": effects_sum_to_total,
        "residual": round(residual, 9),
        "note": (
            "4-term sum (global_B + policy_switch + A_given_per_fold + nested_AB) "
            "minus total. Near zero when the path attribution is exact; nonzero "
            "indicates that B_effect_given_per_fold absorbs the remainder."
        ),
    }

    interpretation_notes = [
        (
            "Design is NESTED: A (fold-local vs pooled params) is only meaningful "
            "under C=per-fold. Cells 100 and 110 were not run. "
            "This is NOT symmetric Shapley over 3 independent factors."
        ),
        (
            "Global B effect: how much switching rho_pair from scalar→per-bin contributes "
            "under global policy (C=0). Pure rho_pair effect in the simpler branch."
        ),
        (
            "Policy switch: how much switching from global→per-fold MDP policy adds, "
            "averaged over the two rho_pair settings (scalar and per-bin). "
            "This is the effect of fold-local MDP solves, independent of params_mode."
        ),
        (
            "A effect given per-fold: how much fold-local params (A=1) improve over "
            "pooled params (A=0) within the per-fold policy branch, averaged over B."
        ),
        (
            "B effect given per-fold: how much per-bin rho_pair (B=1) adds within "
            "the per-fold branch, averaged over A. Informational; not in the 4-term path sum."
        ),
        (
            "Nested AB interaction: whether A and B synergize or cancel within the "
            "per-fold branch. Formula: V111 - V101 - V011 + V001."
        ),
        (
            "4-term path sum (global_B + policy_switch + A_given_per_fold + nested_AB) "
            "does not equal total when B_effect_given_per_fold is nonzero. "
            "The residual captures this gap."
        ),
    ]

    return {
        "decomposition": decomposition,
        "sanity_check": sanity_check,
        "interpretation_notes": interpretation_notes,
    }


# ---- Formatting -------------------------------------------------------------

def format_table(attribution: dict) -> str:
    """Pretty-print the decomposition as a markdown table."""
    d = attribution["decomposition"]
    sc = attribution["sanity_check"]

    def pp(v: float) -> str:
        """Format as percentage-point string with sign."""
        pct = v * 100
        return f"{pct:+.2f}pp"

    rows = [
        ("Effect", "Value"),
        ("---", "---"),
        ("**Total v1→v2**", pp(d["total_v1_to_v2"])),
        ("Global B effect (rho_pair under global policy)", pp(d["global_B_effect"])),
        ("Policy switch (global→per-fold, avg over B)", pp(d["policy_switch_effect"])),
        ("A effect given per-fold (fold-local params)", pp(d["A_effect_given_per_fold"])),
        ("B effect given per-fold (per-bin rho, info only)", pp(d["B_effect_given_per_fold"])),
        ("Nested AB interaction (within per-fold branch)", pp(d["nested_AB_interaction"])),
        ("---", "---"),
        (
            "4-term sum residual",
            f"{sc['residual']:+.2e} "
            f"({'exact' if sc['effects_sum_to_total'] else 'not exact'})",
        ),
    ]

    col1_width = max(len(r[0]) for r in rows)
    col2_width = max(len(r[1]) for r in rows)

    lines = []
    for label, value in rows:
        lines.append(f"| {label:<{col1_width}} | {value:<{col2_width}} |")
    return "\n".join(lines)


# ---- I/O helpers ------------------------------------------------------------

def load_verdict_p57(path: Path) -> tuple[float, float, float]:
    """Load corrected_pipeline_p57 from a verdict JSON file.

    Returns (point, ci_lo, ci_hi) parsed from the string field.
    """
    data = json.loads(path.read_text())
    raw = data.get("corrected_pipeline_p57")
    if raw is None:
        raise KeyError(
            f"Field 'corrected_pipeline_p57' not found in {path}. "
            f"Available keys: {sorted(data.keys())}."
        )
    return parse_p57_string(str(raw))


def check_paths_exist(path_map: dict[str, Path]) -> None:
    """Raise FileNotFoundError if any required path is missing."""
    missing = {label: p for label, p in path_map.items() if not p.exists()}
    if missing:
        lines = "\n".join(f"  cell {label}: {p}" for label, p in sorted(missing.items()))
        raise FileNotFoundError(
            f"The following verdict file(s) are missing:\n{lines}\n"
            "T3 ablation runs must complete before running T4 attribution."
        )


# ---- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    today = date.today().isoformat()
    base = Path("data/validation")
    p = argparse.ArgumentParser(
        description=(
            "v2.5 T4: Compute nested factorial attribution for the v1→v2 "
            "falsification harness verdict shift."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--v000-path",
        default=str(base / "falsification_harness_2026-05-02.json"),
        help="Path to v1 baseline verdict JSON (cell 000).",
    )
    p.add_argument(
        "--v111-path",
        default=str(base / "falsification_harness_v2_2026-05-03.json"),
        help="Path to v2 baseline verdict JSON (cell 111).",
    )
    p.add_argument(
        "--cell-010-path",
        default=str(base / "falsification_harness_v2.5_cell010.json"),
        help="Path to T3 ablation verdict JSON for cell 010.",
    )
    p.add_argument(
        "--cell-001-path",
        default=str(base / "falsification_harness_v2.5_cell001.json"),
        help="Path to T3 ablation verdict JSON for cell 001.",
    )
    p.add_argument(
        "--cell-011-path",
        default=str(base / "falsification_harness_v2.5_cell011.json"),
        help="Path to T3 ablation verdict JSON for cell 011.",
    )
    p.add_argument(
        "--cell-101-path",
        default=str(base / "falsification_harness_v2.5_cell101.json"),
        help="Path to T3 ablation verdict JSON for cell 101.",
    )
    p.add_argument(
        "--out",
        default=str(base / f"v2_5_attribution_{today}.json"),
        help="Output JSON path for the attribution summary.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    path_map = {
        "000": Path(args.v000_path),
        "010": Path(args.cell_010_path),
        "001": Path(args.cell_001_path),
        "011": Path(args.cell_011_path),
        "101": Path(args.cell_101_path),
        "111": Path(args.v111_path),
    }

    check_paths_exist(path_map)

    # Load point estimates from each verdict file
    verdicts_raw: dict[str, tuple[float, float, float]] = {}
    for cell, path in path_map.items():
        verdicts_raw[cell] = load_verdict_p57(path)
        print(f"  cell {cell}: {verdicts_raw[cell][0]:.4f}  [{path.name}]")

    verdicts_points = {cell: t[0] for cell, t in verdicts_raw.items()}

    attribution = compute_attribution(verdicts_points)

    # Print table
    print()
    print("## v2.5 Nested Attribution Decomposition")
    print()
    print(format_table(attribution))
    print()

    sc = attribution["sanity_check"]
    status = "EXACT" if sc["effects_sum_to_total"] else f"RESIDUAL={sc['residual']:+.2e}"
    print(f"Sanity check: {status}")
    print()

    # Build output JSON
    today_str = date.today().isoformat()
    out_data = {
        "date": today_str,
        "verdicts": {
            cell: {
                "point": verdicts_raw[cell][0],
                "ci_lo": None if math.isnan(verdicts_raw[cell][1]) else verdicts_raw[cell][1],
                "ci_hi": None if math.isnan(verdicts_raw[cell][2]) else verdicts_raw[cell][2],
                "source": str(path_map[cell]),
            }
            for cell in ["000", "010", "001", "011", "101", "111"]
        },
        "decomposition": attribution["decomposition"],
        "sanity_check": attribution["sanity_check"],
        "interpretation_notes": attribution["interpretation_notes"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Attribution summary written to: {out_path}")


if __name__ == "__main__":
    main()
