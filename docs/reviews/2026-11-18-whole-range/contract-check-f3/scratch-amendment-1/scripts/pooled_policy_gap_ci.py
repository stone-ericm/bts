#!/usr/bin/env python3
"""Artifact-level screen for pooled-policy A/B gaps.

This is a narrow zero-compute check over the existing pooled_policy_ab*.json
artifacts. It resamples paired seed-level gaps such as ``v_loo - v_prod``.
It does not resample day-level profiles and therefore is not a substitute for
a v2.6-style profile block bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("data/validation/pooled_policy_ab_24seed_consolidated.json")
DEFAULT_OUTPUT = Path("data/validation/pooled_policy_gap_ci_2026-05-06.json")


def _exact_sign_p_two_sided(n_positive: int, n_nonzero: int) -> float | None:
    """Two-sided exact sign-test p-value under p=0.5."""
    if n_nonzero == 0:
        return None
    k = min(n_positive, n_nonzero - n_positive)
    cdf = sum(math.comb(n_nonzero, i) for i in range(k + 1)) / (2 ** n_nonzero)
    return min(1.0, 2.0 * cdf)


def _exact_sign_p_one_sided_positive(n_positive: int, n_nonzero: int) -> float | None:
    """One-sided exact sign-test p-value for the directional claim gap > 0."""
    if n_nonzero == 0:
        return None
    tail = sum(math.comb(n_nonzero, i) for i in range(n_positive, n_nonzero + 1))
    return tail / (2 ** n_nonzero)


def _screen_verdict(ci_lower: float, ci_upper: float) -> str:
    """Asymmetric artifact-level verdict.

    The iid seed bootstrap is too narrow relative to a profile block-bootstrap.
    Therefore a CI that straddles zero is enough to falsify the screen, while a
    CI that excludes zero only leaves the positive screen standing.
    """
    if ci_lower <= 0.0 <= ci_upper:
        return "falsified_under_iid_seed_assumption"
    return "positive_screen_unchanged"


def summarize_gaps(
    rows: list[dict[str, Any]],
    *,
    variant_key: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired seed-level policy gaps for one artifact section."""
    gaps = np.array([float(row[variant_key]) - float(row["v_prod"]) for row in rows])
    if gaps.size == 0:
        raise ValueError("cannot summarize an empty row set")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, gaps.size, size=(n_bootstrap, gaps.size))
    bootstrap_means = gaps[sample_idx].mean(axis=1)
    ci_lower = float(np.quantile(bootstrap_means, 0.025))
    ci_upper = float(np.quantile(bootstrap_means, 0.975))

    n_positive = int(np.sum(gaps > 0))
    n_negative = int(np.sum(gaps < 0))
    n_zero = int(np.sum(gaps == 0))
    n_nonzero = int(gaps.size - n_zero)

    return {
        "n": int(gaps.size),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
        "se_gap": float(gaps.std(ddof=1) / math.sqrt(gaps.size)) if gaps.size > 1 else 0.0,
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "exact_sign_p_one_sided_positive": _exact_sign_p_one_sided_positive(n_positive, n_nonzero),
        "exact_sign_p_two_sided": _exact_sign_p_two_sided(n_positive, n_nonzero),
        "screen_verdict": _screen_verdict(ci_lower, ci_upper),
        "deployment_ready": False,
        "bootstrap": {
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "prob_mean_gt_zero": float(np.mean(bootstrap_means > 0)),
        },
        "gaps_by_seed": [
            {
                "seed": int(row["seed"]),
                "gap": float(float(row[variant_key]) - float(row["v_prod"])),
            }
            for row in rows
        ],
    }


def build_report(
    input_path: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    body = json.loads(input_path.read_text())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "pooled_policy_gap_ci_v1",
        "input_path": input_path.as_posix(),
        "estimand": "paired seed-level P(57) gap for candidate pooled-policy table(s) vs production policy table",
        "methodology": {
            "primary": "paired_seed_bootstrap_over_saved_policy_gaps",
            "unit": "seed",
            "iid_seed_assumption": True,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "is_profile_block_bootstrap": False,
            "limitation": (
                "This uses saved per-seed A/B values only. It does not "
                "recompute bins or policies over day-block resamples and "
                "therefore does not address day-level dependence."
            ),
            "interpretation": (
                "Because this iid seed bootstrap is narrower than a proper "
                "profile block-bootstrap, exclusion of zero leaves the positive "
                "screen standing but is not deployment evidence. A CI that "
                "straddles zero is enough to falsify the saved-gap screen."
            ),
        },
        "within_pool": summarize_gaps(
            body["within_pool"],
            variant_key="v_pool",
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "leave_one_out": summarize_gaps(
            body["leave_one_out"],
            variant_key="v_loo",
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ),
        "source_summary": {
            "within_pool_summary": body.get("within_pool_summary"),
            "leave_one_out_summary": body.get("leave_one_out_summary"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    report = build_report(args.input, n_bootstrap=args.n_bootstrap, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    loo = report["leave_one_out"]
    print(
        "LOO mean_gap={mean:.6f} ci=[{lo:.6f}, {hi:.6f}] sign_p={p:.3g}".format(
            mean=loo["mean_gap"],
            lo=loo["bootstrap"]["ci_lower"],
            hi=loo["bootstrap"]["ci_upper"],
            p=loo["exact_sign_p_two_sided"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
