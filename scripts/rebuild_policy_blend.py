#!/usr/bin/env python3
"""Rebuild MDP policy using blend_walk_forward (experiment-runner code path).

Sibling of scripts/rebuild_policy.py, which uses walk_forward_backtest from
scripts/arch_eval.py. This variant uses bts.simulate.backtest_blend.blend_walk_forward,
which is the OTHER walk-forward implementation in the codebase. The experiment
runner (bts experiment screen) uses this path; rebuild_policy.py uses the
arch_eval path.

The two implementations are similar in intent but not byte-identical, and
the seed=42 outlier audit last night was measured on blend_walk_forward.
Cross-checking the pooled-policy A/B result on THIS path rules out
shared-code-path bias in the original finding.

Output: writes backtest_{season}.parquet to data/simulation/, same format
as rebuild_policy.py. The per-seed Hetzner/Vultr driver then moves this
directory to data/simulation_seed$SEED before running the next seed.

Usage:
    BTS_LGBM_RANDOM_STATE=42 UV_CACHE_DIR=/tmp/uv-cache \\
        uv run python scripts/rebuild_policy_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from bts.simulate.backtest_blend import blend_walk_forward, save_profiles

sys.path.insert(0, "scripts")

from phase7_same_game_double import (  # type: ignore
    add_game_pk_to_profiles,
    apply_different_game_rule,
    build_game_pk_lookup,
)

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def load_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load all PA parquets and compute features.

    Mirrors scripts/arch_eval.py::load_data but is self-contained so this
    script doesn't need arch_eval (which carries the walk_forward_backtest
    implementation we're deliberately NOT using here).
    """
    from bts.features.compute import compute_all_features

    proc = Path(data_dir)
    parquets = sorted(proc.glob("pa_*.parquet"))
    if not parquets:
        raise RuntimeError(
            f"No pa_*.parquet files found in {data_dir}. Run 'bts data build' first."
        )
    dfs = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} PAs from {len(parquets)} parquets", file=sys.stderr)

    df = compute_all_features(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    df = load_data()

    all_profiles = []
    for season in TEST_SEASONS:
        print(f"Running blend_walk_forward for season {season}...", file=sys.stderr)
        profiles = blend_walk_forward(df, season)
        profiles["season"] = season
        all_profiles.append(profiles)

    combined = pd.concat(all_profiles, ignore_index=True)

    # Match rebuild_policy.py's post-processing: attach game_pk and apply
    # the different-game rule, so the top-1/top-2 ranking semantics are
    # identical to the walk_forward_backtest path. Otherwise the pooled
    # bins from the two paths would diverge structurally and the
    # cross-check would be meaningless.
    proc = Path("data/processed")
    pa_dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    pa_data = pd.concat(pa_dfs, ignore_index=True)
    gp_lookup = build_game_pk_lookup(pa_data)
    combined = add_game_pk_to_profiles(combined, gp_lookup)
    combined = apply_different_game_rule(combined)

    out = Path("data/simulation")
    out.mkdir(parents=True, exist_ok=True)
    for season in TEST_SEASONS:
        season_df = combined[combined["season"] == season]
        save_profiles(season_df, season, out)
        print(f"  saved {out}/backtest_{season}.parquet  ({len(season_df)} rows)",
              file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
