"""Phase 3 (densest bucket ablation) + Phase 4 (alt-params blend).

Runs sequentially. Phase 3 reuses the PA-level backtest from Phase 1's
harness, applies densest bucket post-hoc. Phase 4 adds a 13th blend member.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase3_and_4.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import (
    load_data, walk_forward_backtest, compute_metrics, print_comparison,
)
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS, _is_barrel

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
OVERRIDE_THRESHOLD = 0.78


# ── Game time lookup ──────────────────────────────────────────────────

def build_game_time_lookup(raw_dir: str = "data/raw") -> dict[int, str]:
    """Build game_pk → game_time (UTC ISO) lookup from raw JSON feeds."""
    lookup = {}
    raw = Path(raw_dir)
    for season_dir in sorted(raw.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in season_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                pk = int(f.stem)
                gt = d.get("gameData", {}).get("datetime", {}).get("dateTime")
                if gt:
                    lookup[pk] = gt
            except Exception:
                continue
    print(f"Built game_time lookup: {len(lookup)} games", file=sys.stderr)
    return lookup


def add_game_time_to_profiles(profiles: pd.DataFrame, gt_lookup: dict) -> pd.DataFrame:
    """Add game_time column to profiles from lookup. Requires game_pk in profiles."""
    # profiles don't have game_pk directly, but we can join through the backtest
    # Actually, the harness doesn't carry game_pk to profiles. We need a different approach.
    # The profiles have (date, batter_id) — we need to know which game_pk each batter was in.
    # For densest bucket, we need game_time per batter-day. Since we're testing post-hoc,
    # we'll need to rebuild this mapping from the PA data.
    return profiles


# ── Densest bucket ────────────────────────────────────────────────────

def classify_et_hour(game_time_str):
    """Convert UTC game time to ET hour."""
    try:
        utc = datetime.fromisoformat(str(game_time_str).replace("Z", "+00:00"))
        return (utc - timedelta(hours=4)).hour
    except Exception:
        return 18


def apply_densest_bucket_to_day(day_df: pd.DataFrame) -> pd.DataFrame:
    """Apply densest bucket filter to one day's profiles."""
    if "game_time" not in day_df.columns or day_df["game_time"].isna().all():
        return day_df

    df = day_df.copy()
    df["_et_hour"] = df["game_time"].apply(classify_et_hour)

    early = df[df["_et_hour"] < 16]
    prime = df[(df["_et_hour"] >= 16) & (df["_et_hour"] < 20)]
    west = df[df["_et_hour"] >= 20]

    buckets = {"early": early, "prime": prime, "west": west}
    densest_name = max(buckets, key=lambda k: len(buckets[k]))

    if len(df) == 0:
        return df

    top = df.iloc[0]
    top_hour = top["_et_hour"]
    top_window = "early" if top_hour < 16 else ("prime" if top_hour < 20 else "west")

    if top_window == densest_name:
        filtered = buckets[densest_name]
    elif top["p_game_hit"] > OVERRIDE_THRESHOLD:
        filtered = df
    else:
        filtered = buckets[densest_name]

    if len(filtered) == 0:
        filtered = df

    filtered = filtered.sort_values("p_game_hit", ascending=False).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)
    return filtered


# ── Phase 3: Densest bucket ablation ─────────────────────────────────

def run_phase3(df: pd.DataFrame, gt_lookup: dict):
    """Test densest bucket with and without filtering."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("PHASE 3: Densest Bucket Ablation", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # We need game_time in profiles. The harness carries it as None since
    # it's not in parquets. We need to enrich the PA data with game_time
    # BEFORE running the backtest, so it flows through.
    df_with_gt = df.copy()
    df_with_gt["game_time"] = df_with_gt["game_pk"].map(gt_lookup)
    has_gt = df_with_gt["game_time"].notna().sum()
    print(f"  Enriched {has_gt:,}/{len(df_with_gt):,} PAs with game_time", file=sys.stderr)

    # Run backtest once with game_time
    profiles = walk_forward_backtest(
        df_with_gt, TEST_SEASONS, BLEND_CONFIGS, LGB_PARAMS, game_level=False,
    )

    results = {}

    # Without densest bucket (raw blend ranking) — this is the baseline
    results["no_bucket"] = compute_metrics(profiles)

    # With densest bucket — apply post-hoc
    filtered_days = []
    for date, day_df in profiles.groupby("date"):
        filtered = apply_densest_bucket_to_day(day_df)
        filtered_days.append(filtered)
    filtered_profiles = pd.concat(filtered_days, ignore_index=True)
    results["with_bucket"] = compute_metrics(filtered_profiles)

    print_comparison(results, "Phase 3: Densest Bucket Ablation")
    return results


# ── Phase 4: Alt-params blend member ─────────────────────────────────

ALT_PLATOON_THRESHOLD = 40
ALT_GB_MIN_PERIODS = 15
ALT_VENUE_MIN_PERIODS = 30
ALT_STATCAST_MIN_PERIODS = 3


def compute_alt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add alt-params feature columns alongside originals."""
    print("  Computing alt-params features...", file=sys.stderr)

    # 1. platoon_hr_alt (threshold=40)
    platoon_key = df["batter_id"].astype(str) + "_" + df["pitch_hand"].fillna("U")
    date_platoon = df.assign(platoon_key=platoon_key).groupby(
        ["platoon_key", "batter_id", "pitch_hand", "date"]
    ).agg(ph_hits=("is_hit", "sum"), ph_pas=("is_hit", "count")).reset_index()
    date_platoon = date_platoon.sort_values(["platoon_key", "date"])
    date_platoon["cum_ph_hits"] = date_platoon.groupby("platoon_key")["ph_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    date_platoon["cum_ph_pas"] = date_platoon.groupby("platoon_key")["ph_pas"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    date_platoon["platoon_hr_alt"] = np.where(
        date_platoon["cum_ph_pas"] >= ALT_PLATOON_THRESHOLD,
        date_platoon["cum_ph_hits"] / date_platoon["cum_ph_pas"],
        np.nan,
    )
    df = df.merge(
        date_platoon[["batter_id", "pitch_hand", "date", "platoon_hr_alt"]].drop_duplicates(
            subset=["batter_id", "pitch_hand", "date"]
        ),
        on=["batter_id", "pitch_hand", "date"], how="left",
    )

    # 2. batter_gb_hit_rate_alt (min_periods=15)
    df["_is_gb"] = df["launch_angle"].notna() & (df["launch_angle"] < 10)
    df["_gb_hit"] = np.where(df["_is_gb"], df["is_hit"], np.nan)
    date_gb = df.groupby(["batter_id", "date"])["_gb_hit"].agg(
        ["sum", "count"]
    ).reset_index().sort_values(["batter_id", "date"])
    date_gb.columns = ["batter_id", "date", "gb_hits", "gb_count"]
    date_gb["cum_gb_hits"] = date_gb.groupby("batter_id")["gb_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=ALT_GB_MIN_PERIODS).sum()
    )
    date_gb["cum_gb_count"] = date_gb.groupby("batter_id")["gb_count"].transform(
        lambda x: x.shift(1).expanding(min_periods=ALT_GB_MIN_PERIODS).sum()
    )
    date_gb["batter_gb_hit_rate_alt"] = np.where(
        date_gb["cum_gb_count"] > 0,
        date_gb["cum_gb_hits"] / date_gb["cum_gb_count"],
        np.nan,
    )
    df = df.merge(
        date_gb[["batter_id", "date", "batter_gb_hit_rate_alt"]].drop_duplicates(
            subset=["batter_id", "date"]
        ),
        on=["batter_id", "date"], how="left",
    )
    df.drop(columns=["_is_gb", "_gb_hit"], inplace=True)

    # 3. park_factor_alt (venue min_periods=30)
    venue_dates = df.groupby(["venue_id", "date"]).agg(
        v_hits=("is_hit", "sum"), v_pas=("is_hit", "count"),
    ).reset_index().sort_values(["venue_id", "date"])
    venue_dates["venue_hr"] = venue_dates["v_hits"] / venue_dates["v_pas"]
    venue_dates["venue_expanding_hr"] = venue_dates.groupby("venue_id")["venue_hr"].transform(
        lambda x: x.shift(1).expanding(min_periods=ALT_VENUE_MIN_PERIODS).mean()
    )
    league_daily = df.groupby("date").agg(
        league_hits=("is_hit", "sum"), league_pas=("is_hit", "count"),
    ).reset_index().sort_values("date")
    league_daily["league_hr"] = league_daily["league_hits"] / league_daily["league_pas"]
    league_daily["league_expanding_hr"] = league_daily["league_hr"].shift(1).expanding(min_periods=30).mean()
    venue_dates = venue_dates.merge(league_daily[["date", "league_expanding_hr"]], on="date", how="left")
    venue_dates["park_factor_alt"] = np.where(
        venue_dates["league_expanding_hr"].notna() & venue_dates["venue_expanding_hr"].notna(),
        venue_dates["venue_expanding_hr"] / venue_dates["league_expanding_hr"],
        np.nan,
    )
    df = df.merge(
        venue_dates[["venue_id", "date", "park_factor_alt"]].drop_duplicates(
            subset=["venue_id", "date"]
        ),
        on=["venue_id", "date"], how="left",
    )

    # 4. Batter Statcast alt (min_periods=3)
    df["_is_barrel"] = df.apply(lambda r: _is_barrel(r["launch_speed"], r["launch_angle"]), axis=1)
    df["_is_hard_hit"] = df["launch_speed"].notna() & (df["launch_speed"] >= 95)
    df["_is_sweet_spot"] = df["launch_angle"].notna() & (df["launch_angle"] >= 8) & (df["launch_angle"] <= 32)
    df["_has_bb"] = df["launch_speed"].notna()
    date_batted = df.groupby(["batter_id", "date"]).agg(
        barrels=("_is_barrel", "sum"),
        hard_hits=("_is_hard_hit", "sum"),
        sweet_spots=("_is_sweet_spot", "sum"),
        batted_balls=("_has_bb", "sum"),
        avg_ev=("launch_speed", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
    ).reset_index().sort_values(["batter_id", "date"])
    for rate_col, num_col in [("barrel_rate", "barrels"), ("hard_hit_rate", "hard_hits"),
                               ("sweet_spot_rate", "sweet_spots")]:
        date_batted[rate_col] = np.where(
            date_batted["batted_balls"] > 0,
            date_batted[num_col] / date_batted["batted_balls"],
            np.nan,
        )
    alt_statcast_cols = []
    for src, dest in [("barrel_rate", "batter_barrel_rate_30g_alt"),
                      ("hard_hit_rate", "batter_hard_hit_rate_30g_alt"),
                      ("sweet_spot_rate", "batter_sweet_spot_rate_30g_alt"),
                      ("avg_ev", "batter_avg_ev_30g_alt")]:
        date_batted[dest] = date_batted.groupby("batter_id")[src].transform(
            lambda x: x.shift(1).rolling(30, min_periods=ALT_STATCAST_MIN_PERIODS).mean()
        )
        alt_statcast_cols.append(dest)
    df = df.merge(
        date_batted[["batter_id", "date"] + alt_statcast_cols].drop_duplicates(
            subset=["batter_id", "date"]
        ),
        on=["batter_id", "date"], how="left",
    )
    df.drop(columns=["_is_barrel", "_is_hard_hit", "_is_sweet_spot", "_has_bb"], inplace=True)

    print(f"  Alt features added: platoon_hr_alt, batter_gb_hit_rate_alt, "
          f"park_factor_alt, 4 Statcast alts", file=sys.stderr)
    return df


def build_blend_configs_13() -> list[tuple]:
    """Build 13-model blend: standard 12 + alt-params baseline."""
    alt_col_map = {
        "platoon_hr": "platoon_hr_alt",
        "batter_gb_hit_rate": "batter_gb_hit_rate_alt",
        "park_factor": "park_factor_alt",
        "batter_barrel_rate_30g": "batter_barrel_rate_30g_alt",
        "batter_hard_hit_rate_30g": "batter_hard_hit_rate_30g_alt",
        "batter_sweet_spot_rate_30g": "batter_sweet_spot_rate_30g_alt",
        "batter_avg_ev_30g": "batter_avg_ev_30g_alt",
    }
    alt_base = [alt_col_map.get(c, c) for c in FEATURE_COLS]
    configs = list(BLEND_CONFIGS) + [("alt_params", alt_base)]
    return configs


def run_phase4(df: pd.DataFrame):
    """Test 13-model blend vs 12-model baseline."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("PHASE 4: Alt-Params Blend Member", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    df = compute_alt_features(df)

    configs_12 = list(BLEND_CONFIGS)
    configs_13 = build_blend_configs_13()

    results = {}

    print("\n  Running 12-model baseline...", file=sys.stderr)
    profiles_12 = walk_forward_backtest(
        df, TEST_SEASONS, configs_12, LGB_PARAMS, game_level=False,
    )
    results["12_model"] = compute_metrics(profiles_12)

    print("\n  Running 13-model blend...", file=sys.stderr)
    profiles_13 = walk_forward_backtest(
        df, TEST_SEASONS, configs_13, LGB_PARAMS, game_level=False,
    )
    results["13_model"] = compute_metrics(profiles_13)

    print_comparison(results, "Phase 4: Alt-Params Blend Member (12 vs 13 models)")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    df = load_data()
    gt_lookup = build_game_time_lookup()

    phase3_results = run_phase3(df, gt_lookup)
    phase4_results = run_phase4(df)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nPhase 3 (densest bucket):")
    nb = phase3_results["no_bucket"]
    wb = phase3_results["with_bucket"]
    print(f"  Without: P@1={nb['p_at_1']['avg']:.4f}, MDP P(57)={nb['mdp_p57']:.6f}")
    print(f"  With:    P@1={wb['p_at_1']['avg']:.4f}, MDP P(57)={wb['mdp_p57']:.6f}")
    print(f"  Verdict: {'REMOVE bucket' if nb['mdp_p57'] >= wb['mdp_p57'] else 'KEEP bucket'}")

    print(f"\nPhase 4 (alt-params blend):")
    m12 = phase4_results["12_model"]
    m13 = phase4_results["13_model"]
    print(f"  12-model: P@1={m12['p_at_1']['avg']:.4f}, MDP P(57)={m12['mdp_p57']:.6f}")
    print(f"  13-model: P@1={m13['p_at_1']['avg']:.4f}, MDP P(57)={m13['mdp_p57']:.6f}")
    print(f"  Verdict: {'ADOPT 13-model' if m13['mdp_p57'] > m12['mdp_p57'] else 'KEEP 12-model'}")


if __name__ == "__main__":
    main()
