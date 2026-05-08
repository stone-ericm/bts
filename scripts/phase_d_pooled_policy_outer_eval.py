#!/usr/bin/env python3
"""Phase D outer-evaluation for the 100-seed pooled-policy profile surface.

This script consumes the Phase C raw profile artifacts and enforces the
pre-registered split:

    selection seasons: 2021,2022,2023,2024
    outer evaluation: 2025

It builds the candidate pooled MDP policy only from the selection seasons, then
evaluates that fixed policy and the current production policy on each seed's
2025 outer-evaluation bins. It preserves provider tags for sensitivity checks
and writes a validation artifact only. It does not overwrite production models.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy
from bts.simulate.pooled_policy import (
    build_pooled_policy,
    compute_pooled_bins,
    evaluate_mdp_policy,
    parse_seed_from_path,
    split_by_phase_pooled,
)
from bts.validate.proper_scoring import compute_proper_scoring


DEFAULT_PROFILE_ROOTS = [
    Path("data/hetzner_results/phase_c_pooled_policy_profiles_2026-05-07"),
    Path("data/oci_results/phase_c_pooled_policy_profiles_2026-05-07"),
]
DEFAULT_PROD_POLICY_PATH = Path("data/models/mdp_policy.npz")
DEFAULT_OUTPUT = Path("data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json")

SCHEMA_VERSION = "phase_d_pooled_policy_outer_eval_v1"
ARTIFACT_ROLE = "raw_backtest_profile_surface"
SPLIT_MODE = "season_level_selection_outer_eval"
BACKTEST_RE = re.compile(r"backtest_(\d{4})\.parquet$")


@dataclass(frozen=True)
class SeedRecord:
    seed: int
    provider: str
    box: str
    region: str | None
    seed_dir: Path


def parse_seasons(raw: str) -> list[int]:
    seasons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seasons:
        raise ValueError("season list must not be empty")
    return seasons


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _season_from_path(path: Path) -> int:
    match = BACKTEST_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse season from {path}")
    return int(match.group(1))


def _validate_root_metadata(
    root: Path,
    *,
    selection_seasons: list[int],
    outer_eval_seasons: list[int],
) -> dict[str, Any]:
    meta_path = root / "audit_validation_split.json"
    if not meta_path.exists():
        raise ValueError(f"{root} missing audit_validation_split.json")
    meta = _load_json(meta_path)
    if meta.get("artifact_role") != ARTIFACT_ROLE:
        raise ValueError(f"{root} artifact_role is not {ARTIFACT_ROLE!r}")
    if meta.get("split_mode") != SPLIT_MODE:
        raise ValueError(f"{root} split_mode is not {SPLIT_MODE!r}")
    if meta.get("production_deploy_claim") is not False:
        raise ValueError(f"{root} must have production_deploy_claim=false")
    if meta.get("selection_seasons") != selection_seasons:
        raise ValueError(
            f"{root} selection_seasons {meta.get('selection_seasons')} "
            f"!= expected {selection_seasons}"
        )
    if meta.get("outer_eval_seasons") != outer_eval_seasons:
        raise ValueError(
            f"{root} outer_eval_seasons {meta.get('outer_eval_seasons')} "
            f"!= expected {outer_eval_seasons}"
        )
    driver = meta.get("audit_driver", {})
    if driver.get("run_kind") != "profiles":
        raise ValueError(f"{root} audit_driver.run_kind must be profiles")
    if driver.get("queue_mode") != "backtest":
        raise ValueError(f"{root} audit_driver.queue_mode must be backtest")
    return meta


def _record_from_seed_dir(seed_dir: Path) -> SeedRecord:
    meta_path = seed_dir / "audit_validation_split.json"
    if not meta_path.exists():
        raise ValueError(f"{seed_dir} missing audit_validation_split.json")
    meta = _load_json(meta_path)
    driver = meta.get("audit_driver", {})
    seed = parse_seed_from_path(seed_dir)
    return SeedRecord(
        seed=seed,
        provider=str(driver.get("provider") or ""),
        box=str(driver.get("box_name") or seed_dir.parent.name),
        region=driver.get("box_region"),
        seed_dir=seed_dir,
    )


def discover_seed_records(
    roots: list[Path],
    *,
    selection_seasons: list[int],
    outer_eval_seasons: list[int],
    expect_seeds: int | None = None,
) -> tuple[list[SeedRecord], list[dict[str, Any]]]:
    """Discover and validate Phase C per-seed profile directories."""
    root_metadata = []
    records: list[SeedRecord] = []
    required_seasons = set(selection_seasons) | set(outer_eval_seasons)

    for root in roots:
        root_metadata.append({
            "root": root.as_posix(),
            "metadata": _validate_root_metadata(
                root,
                selection_seasons=selection_seasons,
                outer_eval_seasons=outer_eval_seasons,
            ),
        })
        for seed_dir in sorted(root.glob("*/simulation_seed*")):
            if not seed_dir.is_dir():
                continue
            seasons = {_season_from_path(path) for path in seed_dir.glob("backtest_*.parquet")}
            missing = sorted(required_seasons.difference(seasons))
            if missing:
                raise ValueError(f"{seed_dir} missing required backtest seasons {missing}")
            record = _record_from_seed_dir(seed_dir)
            if not record.provider:
                raise ValueError(f"{seed_dir} metadata missing provider")
            records.append(record)

    records = sorted(records, key=lambda r: (r.provider, r.seed))
    seen: dict[int, SeedRecord] = {}
    for record in records:
        if record.seed in seen:
            raise ValueError(
                f"duplicate seed {record.seed}: {seen[record.seed].seed_dir} and {record.seed_dir}"
            )
        seen[record.seed] = record

    if expect_seeds is not None and len(records) != expect_seeds:
        raise ValueError(f"expected {expect_seeds} seeds, found {len(records)}")
    return records, root_metadata


def load_profiles_for_seasons(records: list[SeedRecord], seasons: list[int]) -> pd.DataFrame:
    frames = []
    for record in records:
        for season in seasons:
            path = record.seed_dir / f"backtest_{season}.parquet"
            if not path.exists():
                raise ValueError(f"{record.seed_dir} missing {path.name}")
            frame = pd.read_parquet(path)
            if "season" in frame.columns:
                observed = set(int(s) for s in frame["season"].dropna().unique())
                if observed and observed != {season}:
                    raise ValueError(f"{path} has season column {sorted(observed)}, expected {season}")
            else:
                frame["season"] = season
            frame["seed"] = record.seed
            frame["provider"] = record.provider
            frame["box"] = record.box
            frame["box_region"] = record.region
            frames.append(frame)
    if not frames:
        raise ValueError("no profile frames loaded")
    return pd.concat(frames, ignore_index=True)


def _bins_for_eval(
    profiles: pd.DataFrame,
    *,
    late_phase_days: int,
    n_bins: int,
) -> tuple[Any, Any | None, dict[str, Any]]:
    early_df, late_df = split_by_phase_pooled(profiles, late_phase_days)
    early_bins = compute_pooled_bins(early_df, n_bins=n_bins)
    late_bins = None
    late_status = "not_requested" if late_phase_days <= 0 else "unused"
    if late_phase_days > 0 and len(late_df) > 0:
        try:
            candidate = compute_pooled_bins(late_df, n_bins=n_bins)
            if len(candidate.bins) == n_bins:
                late_bins = candidate
                late_status = "used"
            else:
                late_status = f"fallback_early_only_{len(candidate.bins)}_late_bins"
        except (ValueError, IndexError) as exc:
            late_status = f"fallback_early_only_{type(exc).__name__}"

    diagnostics = {
        "n_rows": int(len(profiles)),
        "n_rank1_rows": int((profiles["rank"] == 1).sum()),
        "n_dates": int(profiles["date"].nunique()),
        "early_bin_count": int(len(early_bins.bins)),
        "late_bin_count": int(len(late_bins.bins)) if late_bins is not None else None,
        "late_bins_status": late_status,
    }
    return early_bins, late_bins, diagnostics


def _rank1_p_at_1(profiles: pd.DataFrame) -> float:
    rank1 = profiles[profiles["rank"] == 1]
    if len(rank1) == 0:
        return float("nan")
    return float(rank1["actual_hit"].mean())


def _quality_bins_summary(qb: Any | None) -> dict[str, Any] | None:
    if qb is None:
        return None
    return {
        "boundaries": [float(x) for x in qb.boundaries],
        "bins": [
            {
                "index": int(bin_.index),
                "p_range": [float(bin_.p_range[0]), float(bin_.p_range[1])],
                "p_hit": float(bin_.p_hit),
                "p_both": float(bin_.p_both),
                "frequency": float(bin_.frequency),
            }
            for bin_ in qb.bins
        ],
    }


def _action_counts(policy_table: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(policy_table, return_counts=True)
    names = {0: "skip", 1: "single", 2: "double"}
    return {names[int(value)]: int(count) for value, count in zip(values, counts)}


def evaluate_outer_policy_gap(
    records: list[SeedRecord],
    selection_profiles: pd.DataFrame,
    outer_profiles: pd.DataFrame,
    *,
    prod_policy_path: Path,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prod_table, prod_boundaries, prod_season_length = load_policy(prod_policy_path)
    selection_early_bins, selection_late_bins, selection_diagnostics = _bins_for_eval(
        selection_profiles,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    pooled_solution = build_pooled_policy(
        selection_profiles,
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    selection_prod_p57 = evaluate_mdp_policy(
        prod_table,
        selection_early_bins,
        season_length=season_length,
        late_bins=selection_late_bins,
        late_phase_days=late_phase_days,
    )
    selection_pooled_p57 = evaluate_mdp_policy(
        pooled_solution.policy_table,
        selection_early_bins,
        season_length=season_length,
        late_bins=selection_late_bins,
        late_phase_days=late_phase_days,
    )

    rows = []
    record_by_seed = {record.seed: record for record in records}
    for seed in sorted(record_by_seed):
        record = record_by_seed[seed]
        seed_outer = outer_profiles[outer_profiles["seed"] == seed].copy()
        early_bins, late_bins, diagnostics = _bins_for_eval(
            seed_outer,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        )
        v_prod = evaluate_mdp_policy(
            prod_table,
            early_bins,
            season_length=season_length,
            late_bins=late_bins,
            late_phase_days=late_phase_days,
        )
        v_pooled = evaluate_mdp_policy(
            pooled_solution.policy_table,
            early_bins,
            season_length=season_length,
            late_bins=late_bins,
            late_phase_days=late_phase_days,
        )
        rows.append({
            "seed": int(seed),
            "provider": record.provider,
            "box": record.box,
            "box_region": record.region,
            "v_prod": float(v_prod),
            "v_pooled": float(v_pooled),
            "gap": float(v_pooled - v_prod),
            "outer_p_at_1": _rank1_p_at_1(seed_outer),
            "eval_diagnostics": diagnostics,
        })

    policy_metadata = {
        "production_policy": {
            "path": prod_policy_path.as_posix(),
            "season_length": int(prod_season_length),
            "boundaries": [float(x) for x in prod_boundaries],
            "table_shape": list(prod_table.shape),
            "action_counts": _action_counts(prod_table),
        },
        "pooled_candidate": {
            "built_in_memory_only": True,
            "optimal_p57_on_selection_surface": float(pooled_solution.optimal_p57),
            "evaluated_p57_on_selection_surface": float(selection_pooled_p57),
            "selection_surface_gap_vs_production": float(selection_pooled_p57 - selection_prod_p57),
            "table_shape": list(pooled_solution.policy_table.shape),
            "action_counts": _action_counts(pooled_solution.policy_table),
            "selection_diagnostics": selection_diagnostics,
            "selection_early_bins": _quality_bins_summary(selection_early_bins),
            "selection_late_bins": _quality_bins_summary(selection_late_bins),
        },
        "selection_surface_baseline": {
            "production_p57": float(selection_prod_p57),
            "pooled_candidate_p57": float(selection_pooled_p57),
            "gap": float(selection_pooled_p57 - selection_prod_p57),
            "interpretation": (
                "The candidate improves the pre-registered selection surface but "
                "is judged by the disjoint 2025 outer-evaluation surface."
            ),
        },
    }
    return rows, policy_metadata


def _exact_sign_p_two_sided(n_positive: int, n_nonzero: int) -> float | None:
    if n_nonzero == 0:
        return None
    k = min(n_positive, n_nonzero - n_positive)
    cdf = sum(math.comb(n_nonzero, i) for i in range(k + 1)) / (2 ** n_nonzero)
    return min(1.0, 2.0 * cdf)


def _gap_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["gap"]) for row in rows], dtype=float)


def summarize_policy_gaps(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    stratify_by_provider: bool,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")

    gaps = _gap_array(rows)
    rng = np.random.default_rng(seed)
    if stratify_by_provider:
        groups = []
        for provider in sorted({str(row["provider"]) for row in rows}):
            group = _gap_array([row for row in rows if row["provider"] == provider])
            groups.append(group)
        reps = []
        for group in groups:
            idx = rng.integers(0, group.size, size=(n_bootstrap, group.size))
            reps.append(group[idx])
        bootstrap_means = np.concatenate(reps, axis=1).mean(axis=1)
        bootstrap_kind = "provider_stratified_seed_bootstrap"
    else:
        idx = rng.integers(0, gaps.size, size=(n_bootstrap, gaps.size))
        bootstrap_means = gaps[idx].mean(axis=1)
        bootstrap_kind = "iid_seed_bootstrap"

    n_positive = int(np.sum(gaps > 0))
    n_negative = int(np.sum(gaps < 0))
    n_zero = int(np.sum(gaps == 0))
    n_nonzero = int(gaps.size - n_zero)

    return {
        "n": int(gaps.size),
        "mean_prod": float(np.mean([row["v_prod"] for row in rows])),
        "mean_pooled": float(np.mean([row["v_pooled"] for row in rows])),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
        "se_gap": float(gaps.std(ddof=1) / math.sqrt(gaps.size)) if gaps.size > 1 else 0.0,
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "exact_sign_p_two_sided": _exact_sign_p_two_sided(n_positive, n_nonzero),
        "bootstrap": {
            "kind": bootstrap_kind,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
            "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
            "prob_mean_gt_zero": float(np.mean(bootstrap_means > 0)),
        },
    }


def summarize_p_at_1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["outer_p_at_1"]) for row in rows], dtype=float)
    return {
        "n": int(values.size),
        "mean_seed_outer_p_at_1": float(values.mean()),
        "std_seed_outer_p_at_1": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "gap_candidate_vs_prod": None,
        "gap_not_applicable_reason": (
            "The Phase D candidate changes only the MDP policy table. It does not "
            "change the rank-1 probability model, so raw outer-surface P@1 is a "
            "shared calibration diagnostic rather than a candidate-vs-production gap."
        ),
    }


def provider_summaries(rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    out = {}
    for i, provider in enumerate(sorted({str(row["provider"]) for row in rows})):
        provider_rows = [row for row in rows if row["provider"] == provider]
        summary = summarize_policy_gaps(
            provider_rows,
            n_bootstrap=n_bootstrap,
            seed=seed + 101 + i,
            stratify_by_provider=False,
        )
        summary["p_at_1"] = summarize_p_at_1(provider_rows)
        out[provider] = summary
    return out


def derive_verdict(
    overall: dict[str, Any],
    providers: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if overall["mean_gap"] <= 0:
        return {
            "verdict": "falsified",
            "production_deploy_ready": False,
            "reasons": ["pooled-policy gap is non-positive on the 2025 outer-evaluation surface"],
        }

    if overall["bootstrap"]["ci_lower"] <= 0:
        reasons.append("primary 100-seed bootstrap interval overlaps zero")

    provider_gaps = {
        provider: summary["mean_gap"]
        for provider, summary in providers.items()
    }
    non_positive_providers = [
        provider for provider, gap in provider_gaps.items() if gap <= 0
    ]
    if non_positive_providers:
        reasons.append(
            "provider-tagged diagnostics are not uniformly positive: "
            + ", ".join(non_positive_providers)
        )

    if reasons:
        return {
            "verdict": "inconclusive",
            "production_deploy_ready": False,
            "reasons": reasons,
        }

    return {
        "verdict": "survives_outer_eval",
        "production_deploy_ready": False,
        "reasons": [
            "outer-evaluation gap is positive",
            "primary provider-stratified seed-bootstrap interval excludes zero",
            "provider-tagged mean gaps are all positive",
        ],
    }


def proper_scoring_summary(outer_profiles: pd.DataFrame) -> dict[str, Any]:
    scoring = {
        "overall": compute_proper_scoring(outer_profiles),
        "providers": {},
        "interpretation": (
            "Proper scoring is reported on the shared 2025 outer profile surface. "
            "It is a calibration/falsification diagnostic for the probability model, "
            "not a direct P(57) policy-value gap."
        ),
    }
    for provider, group in outer_profiles.groupby("provider"):
        scoring["providers"][str(provider)] = compute_proper_scoring(group)
    return scoring


def build_report(
    *,
    roots: list[Path],
    prod_policy_path: Path,
    selection_seasons: list[int],
    outer_eval_seasons: list[int],
    expect_seeds: int | None,
    n_bootstrap: int,
    seed: int,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> dict[str, Any]:
    records, root_metadata = discover_seed_records(
        roots,
        selection_seasons=selection_seasons,
        outer_eval_seasons=outer_eval_seasons,
        expect_seeds=expect_seeds,
    )
    selection_profiles = load_profiles_for_seasons(records, selection_seasons)
    outer_profiles = load_profiles_for_seasons(records, outer_eval_seasons)
    rows, policy_metadata = evaluate_outer_policy_gap(
        records,
        selection_profiles,
        outer_profiles,
        prod_policy_path=prod_policy_path,
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    overall_stratified = summarize_policy_gaps(
        rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
        stratify_by_provider=True,
    )
    overall_iid = summarize_policy_gaps(
        rows,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
        stratify_by_provider=False,
    )
    providers = provider_summaries(rows, n_bootstrap=n_bootstrap, seed=seed)
    verdict = derive_verdict(overall_stratified, providers)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "phase_d_outer_eval_validation",
        "production_deploy_claim": False,
        "inputs": {
            "profile_roots": [root.as_posix() for root in roots],
            "prod_policy_path": prod_policy_path.as_posix(),
            "root_metadata": root_metadata,
        },
        "split": {
            "mode": SPLIT_MODE,
            "selection_seasons": selection_seasons,
            "outer_eval_seasons": outer_eval_seasons,
        },
        "methodology": {
            "candidate_policy": (
                "pooled MDP policy built from all Phase C seeds on selection seasons only"
            ),
            "comparison": (
                "fixed production policy table vs fixed pooled candidate policy table, "
                "evaluated on each seed's 2025 outer-evaluation bins"
            ),
            "primary_uncertainty": "provider_stratified_seed_bootstrap_preserving_provider_counts",
            "secondary_uncertainty": "iid_seed_bootstrap",
            "season_length": season_length,
            "late_phase_days": late_phase_days,
            "n_bins": n_bins,
        },
        "seed_pool": {
            "n": len(records),
            "providers": {
                provider: int(sum(record.provider == provider for record in records))
                for provider in sorted({record.provider for record in records})
            },
            "seeds": [record.seed for record in records],
        },
        "policy_metadata": policy_metadata,
        "p57_outer_eval": {
            "overall": overall_stratified,
            "overall_iid_seed_bootstrap": overall_iid,
            "providers": providers,
            "rows": rows,
        },
        "p_at_1_outer_eval": {
            "overall": summarize_p_at_1(rows),
            "providers": {
                provider: summary["p_at_1"]
                for provider, summary in providers.items()
            },
        },
        "proper_scoring_outer_eval": proper_scoring_summary(outer_profiles),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", action="append", type=Path, dest="profile_roots",
                    help="Phase C profile root; repeat for multiple providers")
    ap.add_argument("--prod-policy", type=Path, default=DEFAULT_PROD_POLICY_PATH)
    ap.add_argument("--selection-seasons", default="2021,2022,2023,2024")
    ap.add_argument("--outer-eval-seasons", default="2025")
    ap.add_argument("--expect-seeds", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--season-length", type=int, default=180)
    ap.add_argument("--late-phase-days", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    roots = args.profile_roots if args.profile_roots else DEFAULT_PROFILE_ROOTS
    report = build_report(
        roots=roots,
        prod_policy_path=args.prod_policy,
        selection_seasons=parse_seasons(args.selection_seasons),
        outer_eval_seasons=parse_seasons(args.outer_eval_seasons),
        expect_seeds=args.expect_seeds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        season_length=args.season_length,
        late_phase_days=args.late_phase_days,
        n_bins=args.n_bins,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    overall = report["p57_outer_eval"]["overall"]
    verdict = report["verdict"]
    print(
        "wrote {path} | n={n} mean_gap={gap:+.6f} ci=[{lo:+.6f}, {hi:+.6f}] "
        "verdict={verdict}".format(
            path=args.out,
            n=overall["n"],
            gap=overall["mean_gap"],
            lo=overall["bootstrap"]["ci_lower"],
            hi=overall["bootstrap"]["ci_upper"],
            verdict=verdict["verdict"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
