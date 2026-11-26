#!/usr/bin/env python3
"""Screen whether existing artifacts bound determinism's role in pooled gaps.

This is intentionally an evidence-accounting script, not a training job. It
compares the saved pooled-policy gap screen against the deterministic n=100
baseline summary and records whether the available artifacts are sufficient to
bound provider/model nondeterminism in the seed-gap variance.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


DEFAULT_GAP_CI = Path("data/validation/pooled_policy_gap_ci_2026-05-06.json")
DEFAULT_DETERMINISTIC_BASELINE = Path("data/validation/baseline_n100_deterministic_2026-04-27.json")
DEFAULT_DETERMINISTIC_SCREEN = Path("data/validation/screen_pooled_n10_2026-04-28.json")
DEFAULT_OUTPUT = Path("data/validation/pooled_gap_determinism_bound_2026-05-06.json")


def summarize_gap_section(section: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-ready summary of one pooled-policy gap section."""
    gaps = [float(row["gap"]) for row in section["gaps_by_seed"]]
    if not gaps:
        raise ValueError("gap section must contain gaps_by_seed rows")

    positives = [gap for gap in gaps if gap > 0.0]
    negatives = [gap for gap in gaps if gap < 0.0]
    zeros = [gap for gap in gaps if gap == 0.0]
    gap_std = stdev(gaps) if len(gaps) > 1 else 0.0

    return {
        "n": len(gaps),
        "mean_gap": float(mean(gaps)),
        "std_gap": float(gap_std),
        "se_gap": float(gap_std / math.sqrt(len(gaps))) if gaps else 0.0,
        "min_gap": float(min(gaps)),
        "max_gap": float(max(gaps)),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_zero": len(zeros),
        "smallest_positive_gap": float(min(positives)) if positives else None,
        "sign_flip_margin": (
            float(min(positives))
            if len(positives) == len(gaps) and not negatives and not zeros
            else None
        ),
    }


def summarize_distribution_shift(
    deterministic_baseline: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Compare deterministic baseline summary against prior non-det summary."""
    metrics = deterministic_baseline["metrics"][metric]
    prior = deterministic_baseline["comparison_non_deterministic_prior"][metric]
    prior_std = float(prior["std"])
    delta = float(metrics["mean"]) - float(prior["mean"])
    return {
        "metric": metric,
        "deterministic_mean": float(metrics["mean"]),
        "deterministic_std": float(metrics["std"]),
        "deterministic_n": int(metrics["n"]),
        "prior_non_deterministic_mean": float(prior["mean"]),
        "prior_non_deterministic_std": prior_std,
        "mean_delta": float(delta),
        "mean_delta_abs": float(abs(delta)),
        "z_vs_prior_std": None if prior_std == 0.0 else float(delta / prior_std),
    }


def summarize_deterministic_feature_delta_screen(
    deterministic_screen: dict[str, Any],
    *,
    reference_std: float,
) -> dict[str, Any]:
    """Summarize deterministic-only p57 delta variation across feature screens."""
    rows = []
    for experiment, result in deterministic_screen["results"].items():
        delta = result["pooled"]["delta_p_57_mdp"]
        rows.append({
            "experiment": experiment,
            "n": int(delta["n"]),
            "mean": float(delta["mean"]),
            "std": float(delta["std"]),
            "se": float(delta["se"]),
            "t": float(delta["t"]),
        })
    stds = [row["std"] for row in rows]
    nonzero_stds = [std for std in stds if std > 0.0]
    ge_reference = [row for row in rows if row["std"] >= reference_std]
    top_by_std = sorted(rows, key=lambda row: row["std"], reverse=True)[:5]
    return {
        "corpus": deterministic_screen.get("corpus"),
        "flags": deterministic_screen.get("flags"),
        "n_seeds": int(deterministic_screen["n_seeds"]),
        "n_experiments": int(deterministic_screen["n_experiments"]),
        "metric": "per-experiment deterministic delta_p_57_mdp across seeds",
        "std_min": float(min(stds)),
        "std_min_nonzero": float(min(nonzero_stds)) if nonzero_stds else None,
        "std_median": float(median(stds)),
        "std_max": float(max(stds)),
        "n_experiments_std_ge_reference": len(ge_reference),
        "reference_std": float(reference_std),
        "top_by_std": top_by_std,
        "interpretation": (
            "This is a deterministic-only feature-delta proxy, not the pooled-policy A/B estimand. "
            "It shows the scale of seed variation that remains after the deterministic cutover."
        ),
    }


def build_report(
    gap_ci: dict[str, Any],
    deterministic_baseline: dict[str, Any],
    deterministic_screen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loo = summarize_gap_section(gap_ci["leave_one_out"])
    within = summarize_gap_section(gap_ci["within_pool"])
    p57_shift = summarize_distribution_shift(deterministic_baseline, "p_57_mdp")
    p1_shift = summarize_distribution_shift(deterministic_baseline, "p_at_1_avg")
    deterministic_proxy = (
        summarize_deterministic_feature_delta_screen(
            deterministic_screen,
            reference_std=float(loo["std_gap"]),
        )
        if deterministic_screen is not None else None
    )

    loo_std = float(loo["std_gap"])
    det_p57_std = float(p57_shift["deterministic_std"])
    return {
        "schema_version": "determinism_gap_bound_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pooled_policy_gap_ci": str(DEFAULT_GAP_CI),
            "deterministic_baseline": str(DEFAULT_DETERMINISTIC_BASELINE),
            "deterministic_feature_delta_screen": (
                str(DEFAULT_DETERMINISTIC_SCREEN)
                if deterministic_screen is not None else None
            ),
        },
        "methodology": {
            "estimand": "whether existing artifacts bound nondeterminism contribution to pooled-policy seed-gap variance",
            "direct_paired_bound_available": False,
            "direct_bound_missing_reason": [
                "No paired same-seed deterministic vs non-deterministic rerun artifact is available for the pooled-policy A/B gaps.",
                "The pooled-policy gap artifact does not embed deterministic training flags or provider determinism metadata.",
                "The deterministic n=100 baseline is a distribution-level screen on canonical baseline seeds, not the same paired seed-gap estimand as the pooled-policy A/B artifact.",
            ],
            "available_screen": "distribution-shift comparison between deterministic n=100 baseline and prior non-deterministic baseline summary",
        },
        "pooled_policy_gap": {
            "leave_one_out": loo,
            "within_pool": within,
        },
        "deterministic_baseline": {
            "corpus": deterministic_baseline.get("corpus"),
            "flags": deterministic_baseline.get("flags"),
            "seed_pool_size": deterministic_baseline.get("seed_pool_size"),
            "comparison_verdict": deterministic_baseline["comparison_non_deterministic_prior"].get("verdict"),
            "metric_shifts": {
                "p_57_mdp": p57_shift,
                "p_at_1_avg": p1_shift,
            },
        },
        "scale_comparison": {
            "loo_gap_std": loo_std,
            "deterministic_p57_mdp_seed_std": det_p57_std,
            "loo_gap_std_over_deterministic_p57_mdp_seed_std": (
                None if det_p57_std == 0.0 else float(loo_std / det_p57_std)
            ),
            "loo_mean_gap": float(loo["mean_gap"]),
            "abs_p57_distribution_mean_delta": float(p57_shift["mean_delta_abs"]),
            "loo_mean_gap_over_abs_p57_distribution_mean_delta": (
                None
                if p57_shift["mean_delta_abs"] == 0.0
                else float(abs(float(loo["mean_gap"])) / p57_shift["mean_delta_abs"])
            ),
        },
        "deterministic_feature_delta_proxy": deterministic_proxy,
        "verdict": {
            "status": "distribution_shift_not_detected_but_direct_bound_missing",
            "iid_seed_assumption_verdict": "not_evaluable_from_existing_artifacts",
            "c0_determinism_caveat_resolved": False,
            "pooled_gap_screen_status": "unchanged",
            "interpretation": (
                "Existing deterministic-baseline evidence does not show a distribution shift, and "
                "deterministic-only feature-delta screens show substantial seed variation remains "
                "after the deterministic cutover. These are not paired bounds on nondeterminism "
                "inside the pooled-policy seed gaps. "
                "The C0 positive screen remains standing, with the determinism/provenance caveat still open."
            ),
            "next_required_evidence": [
                "paired same-seed deterministic and non-deterministic reruns on the same policy-gap estimand",
                "embedded deterministic/provider metadata in raw pooled-policy profile artifacts",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-ci", type=Path, default=DEFAULT_GAP_CI)
    parser.add_argument("--deterministic-baseline", type=Path, default=DEFAULT_DETERMINISTIC_BASELINE)
    parser.add_argument("--deterministic-screen", type=Path, default=DEFAULT_DETERMINISTIC_SCREEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    gap_ci = json.loads(args.gap_ci.read_text())
    deterministic_baseline = json.loads(args.deterministic_baseline.read_text())
    deterministic_screen = json.loads(args.deterministic_screen.read_text())
    report = build_report(gap_ci, deterministic_baseline, deterministic_screen)
    report["inputs"]["pooled_policy_gap_ci"] = str(args.gap_ci)
    report["inputs"]["deterministic_baseline"] = str(args.deterministic_baseline)
    report["inputs"]["deterministic_feature_delta_screen"] = str(args.deterministic_screen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    loo = report["pooled_policy_gap"]["leave_one_out"]
    p57 = report["deterministic_baseline"]["metric_shifts"]["p_57_mdp"]
    print(
        "determinism gap bound: "
        f"direct_bound_available={report['methodology']['direct_paired_bound_available']} "
        f"loo_mean_gap={loo['mean_gap']:.6f} "
        f"loo_std_gap={loo['std_gap']:.6f} "
        f"p57_distribution_delta={p57['mean_delta']:.6g} "
        f"status={report['verdict']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
