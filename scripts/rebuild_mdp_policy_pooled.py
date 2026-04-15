#!/usr/bin/env python3
"""Rebuild MDP policy from pooled-seed profile parquets (Option 7).

Thin CLI wrapper around bts.simulate.pooled_policy. See that module's
docstring (and memory/project_bts_2026_04_15_audit_state.md) for the
full rationale.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/rebuild_mdp_policy_pooled.py \\
        --seed-dirs data/hetzner_results/pooled_bins_run/*/simulation_seed* \\
        --out data/models/mdp_policy_pooled_v1.npz \\
        --late-phase-days 30 \\
        --season-length 180
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bts.simulate.pooled_policy import (
    build_pooled_policy,
    compute_pooled_bins,
    load_pooled_profiles,
    split_by_phase_pooled,
)
from bts.simulate.quality_bins import QualityBins


def _print_bins(label: str, qb: QualityBins) -> None:
    print(f"  {label} ({len(qb.bins)} bins):")
    for b in qb.bins:
        print(f"    Q{b.index + 1} [{b.p_range[0]:.3f}-{b.p_range[1]:.3f}]: "
              f"P(hit)={b.p_hit:.3%}  P(both)={b.p_both:.3%}  "
              f"freq={b.frequency:.2%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dirs", type=Path, nargs="+", required=True,
                    help="Per-seed directories each containing backtest_*.parquet")
    ap.add_argument("--out", type=Path,
                    default=Path("data/models/mdp_policy_pooled_v1.npz"),
                    help="Output .npz policy path")
    ap.add_argument("--season-length", type=int, default=180)
    ap.add_argument("--late-phase-days", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading pooled profiles from {len(args.seed_dirs)} seed dirs...")
    profiles = load_pooled_profiles(args.seed_dirs)
    n_seeds = profiles["seed"].nunique()
    n_dates = profiles["date"].nunique()
    print(f"  rows={len(profiles):,} seeds={n_seeds} unique_calendar_dates={n_dates}")
    print(f"  seeds: {sorted(profiles['seed'].unique())}")

    print(f"\nBuilding pooled MDP policy "
          f"(season_length={args.season_length}, late_phase_days={args.late_phase_days}, "
          f"n_bins={args.n_bins})...")
    sol = build_pooled_policy(
        profiles,
        season_length=args.season_length,
        late_phase_days=args.late_phase_days,
        n_bins=args.n_bins,
    )

    if args.late_phase_days > 0:
        early_df, late_df = split_by_phase_pooled(profiles, args.late_phase_days)
        if len(late_df) > 0:
            _print_bins("pooled early bins", compute_pooled_bins(early_df, args.n_bins))
            _print_bins("pooled late bins", compute_pooled_bins(late_df, args.n_bins))
        else:
            _print_bins("pooled bins (single phase)", sol.quality_bins)
    else:
        _print_bins("pooled bins (single phase)", sol.quality_bins)

    print(f"\n  Optimal pooled P(57) = {sol.optimal_p57:.4%}")

    print(f"\nSaving to {args.out}...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sol.save(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
