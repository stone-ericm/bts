#!/usr/bin/env python3
"""Audit the experiment registry for monotone-equivalent feature pairs.

LightGBM trees are scale-invariant under monotone transformations: if feature
B is a strictly increasing function of feature A, models trained on A produce
identical splits to models trained on B. So an experiment that adds B (when A
is already in FEATURE_COLS) is a hidden duplicate that wastes audit compute.

Confirmed instance (2026-04-30): heat_index_linear and heat_index_squared
produced bit-identical per-seed deltas across all 4 splits in tonight's audit.
This script does a one-time name-pattern sweep + offers a verifier on demand.

Patterns detected:
  - Pairs where one name is the other plus a monotone-suffix (`_linear` ↔
    `_squared`, `_log` ↔ no suffix when the underlying is positive, `_z_score`,
    quantile transforms `_q<N>`).
  - Same-prefix pairs with different transformation labels.

Output:
  Prints a ranked list of suspect pairs with severity heuristic.
  Writes `data/validation/monotone_audit_<TODAY>.json` for follow-up.

Usage:
  uv run python scripts/audit_monotone_duplicates.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

# Suffixes that indicate monotone transforms when applied to non-negative bases
MONOTONE_SUFFIXES = {
    "_linear": "identity",
    "_squared": "x^2 (monotone for x>=0)",
    "_log": "log (monotone for x>0)",
    "_z_score": "linear standardization (always monotone)",
    "_quantile": "quantile/CDF (always monotone)",
    "_rank": "rank (always monotone)",
    "_zscored": "linear standardization",
}

# Name-stem extraction: strip any of the above suffixes
def stem(name: str) -> tuple[str, str]:
    """Returns (stem, transform_label) — strips one of the known suffixes."""
    for suffix, label in MONOTONE_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], label
    return name, "raw"


def quantile_pair(name: str) -> tuple[str, int] | None:
    """Detect _q<N> suffix. Returns (stem, percentile) or None."""
    import re
    m = re.match(r"^(.+)_q(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return None


def main():
    from bts.experiment.registry import load_all_experiments, EXPERIMENTS

    load_all_experiments()
    all_names = sorted(EXPERIMENTS.keys())

    print(f"Loaded {len(all_names)} candidate experiment/feature names")

    # Suspect pairs: same stem, different transform label
    by_stem: dict[str, list[tuple[str, str]]] = {}
    for name in all_names:
        s, label = stem(name)
        by_stem.setdefault(s, []).append((name, label))

    suspect_pairs = []
    for s, members in by_stem.items():
        if len(members) > 1:
            transforms = {m[1] for m in members}
            if len(transforms) > 1:
                suspect_pairs.append({
                    "stem": s,
                    "members": [m[0] for m in members],
                    "transforms": sorted(transforms),
                    "severity": "HIGH" if "x^2 (monotone for x>=0)" in transforms or "identity" in transforms else "MEDIUM",
                    "reasoning": "Same name stem with different monotone-transform suffixes; LightGBM tree-equivalent if applied to same base value",
                })

    # Quantile detection (separate pass)
    q_groups: dict[str, list[tuple[str, int]]] = {}
    for name in all_names:
        q = quantile_pair(name)
        if q:
            q_groups.setdefault(q[0], []).append((name, q[1]))
    for stem_name, members in q_groups.items():
        if len(members) > 1:
            suspect_pairs.append({
                "stem": stem_name,
                "members": [m[0] for m in sorted(members, key=lambda x: x[1])],
                "transforms": [f"q{m[1]}" for m in sorted(members, key=lambda x: x[1])],
                "severity": "MEDIUM",
                "reasoning": "Multiple quantile cutoffs over the same base feature — pairwise tree-equivalent if monotone",
            })

    # Known confirmed pair (by audit 2026-04-30)
    confirmed_pairs = [
        {"members": ["heat_index_linear", "heat_index_squared"],
         "evidence": "phase2 audit 2026-04-30: bit-identical per-seed deltas across 4 splits"},
    ]

    out = {
        "generated_at": str(date.today()),
        "n_candidates_scanned": len(all_names),
        "n_suspect_pairs": len(suspect_pairs),
        "confirmed_pairs": confirmed_pairs,
        "suspect_pairs": suspect_pairs,
        "next_steps": [
            "For each HIGH-severity pair, run scripts/validate_provider_determinism.py "
            "on phase2 audit dirs of both features at same seed set. Bit-identical = "
            "confirmed duplicate, unregister one.",
            "Add tree-equivalence verification to experiment registry CI: any new "
            "experiment whose feature_cols() output is a monotone function of an "
            "existing FEATURE_COL should fail registration.",
        ],
    }

    out_path = Path("/Users/stone/projects/bts/data/validation") / f"monotone_audit_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print(f"\nWrote {out_path}")
    print(f"\n=== Suspect pairs (n={len(suspect_pairs)}) ===")
    for p in suspect_pairs:
        members_str = " ↔ ".join(p["members"])
        print(f"  [{p['severity']}] {members_str}")
        print(f"      transforms: {', '.join(p['transforms'])}")
    print(f"\n=== Confirmed (audit-verified) ===")
    for p in confirmed_pairs:
        print(f"  • {' ↔ '.join(p['members'])}: {p['evidence']}")


if __name__ == "__main__":
    main()
