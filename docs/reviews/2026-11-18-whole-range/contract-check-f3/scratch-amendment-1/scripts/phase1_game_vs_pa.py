"""Phase 1: Game-level vs PA-level modeling comparison.

Runs walk-forward backtests comparing:
  - pa_level: PA-level training with live-aligned aggregation (current production)
  - game_level: Game-level training with lineup_position as an extra feature

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase1_game_vs_pa.py
"""

import sys

# scripts/ is not a package — add it to sys.path before importing arch_eval
sys.path.insert(0, "scripts")

from arch_eval import load_data, walk_forward_backtest, compute_metrics, print_comparison
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS

# Game-level blend: same 12 variants but each feature list includes lineup_position.
# lineup_position is available after _aggregate_to_game_level (first value per game).
GAME_BLEND_CONFIGS = [
    ("baseline",    FEATURE_COLS + ["lineup_position"]),
    ("barrel",      FEATURE_COLS + ["lineup_position", "batter_barrel_rate_30g"]),
    ("hard_hit",    FEATURE_COLS + ["lineup_position", "batter_hard_hit_rate_30g"]),
    ("sweet_spot",  FEATURE_COLS + ["lineup_position", "batter_sweet_spot_rate_30g"]),
    ("avg_ev",      FEATURE_COLS + ["lineup_position", "batter_avg_ev_30g"]),
    ("velo",        FEATURE_COLS + ["lineup_position", "pitcher_avg_velo_30g"]),
    ("spin",        FEATURE_COLS + ["lineup_position", "pitcher_avg_spin_30g"]),
    ("extension",   FEATURE_COLS + ["lineup_position", "pitcher_avg_extension_30g"]),
    ("break",       FEATURE_COLS + ["lineup_position", "pitcher_break_total_30g"]),
    ("velo_faced",  FEATURE_COLS + ["lineup_position", "batter_avg_velo_faced_30g"]),
    ("best_two",    FEATURE_COLS + ["lineup_position", "batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
    ("all_statcast", FEATURE_COLS + ["lineup_position"] + STATCAST_COLS),
]

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def main():
    print("Loading data...", file=sys.stderr)
    df = load_data()

    print("\nRunning PA-level backtest...", file=sys.stderr)
    pa_profiles = walk_forward_backtest(
        df, TEST_SEASONS, BLEND_CONFIGS, LGB_PARAMS, game_level=False
    )

    print("\nRunning game-level backtest...", file=sys.stderr)
    game_profiles = walk_forward_backtest(
        df, TEST_SEASONS, GAME_BLEND_CONFIGS, LGB_PARAMS, game_level=True
    )

    print("\nComputing metrics...", file=sys.stderr)
    pa_metrics = compute_metrics(pa_profiles)
    game_metrics = compute_metrics(game_profiles)

    results = {
        "pa_level": pa_metrics,
        "game_level": game_metrics,
    }

    print_comparison(results, label="Phase 1: Game-level vs PA-level")

    # Quality bin detail for both variants
    def _print_bins(label, metrics):
        print(f"\n=== Quality Bin Detail: {label} ===")
        qbins = metrics["early_bins"]
        print(f"  Early bins ({len(qbins.bins)} bins):")
        for b in qbins.bins:
            print(
                f"    bin={b.index} p=[{b.p_range[0]:.3f}, {b.p_range[1]:.3f}]"
                f"  p_hit={b.p_hit:.3f}  p_both={b.p_both:.3f}  freq={b.frequency:.3f}"
            )
        if metrics["late_bins"] is not None:
            print(f"  Late bins (September, {len(metrics['late_bins'].bins)} bins):")
            for b in metrics["late_bins"].bins:
                print(
                    f"    bin={b.index} p=[{b.p_range[0]:.3f}, {b.p_range[1]:.3f}]"
                    f"  p_hit={b.p_hit:.3f}  p_both={b.p_both:.3f}  freq={b.frequency:.3f}"
                )

    _print_bins("PA-level", pa_metrics)
    _print_bins("game-level", game_metrics)


if __name__ == "__main__":
    main()
