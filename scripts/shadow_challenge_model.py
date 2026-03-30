"""Shadow model: monitor when ABS challenge features have enough signal to help P@1.

Computes expanding challenge features from 2026 data and checks whether
adding them to the model improves P@1 on a recent validation window.

Run periodically (weekly or monthly) to track feature maturation.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shadow_challenge_model.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from bts.features.compute import compute_all_features, FEATURE_COLS, TRAIN_START_YEAR

BASELINE_COLS = [
    "batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g",
    "batter_whiff_60g", "batter_count_tendency_30g", "batter_gb_hit_rate",
    "platoon_hr", "pitcher_hr_30g", "pitcher_entropy_30g",
    "weather_temp", "park_factor", "days_rest",
]

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)


def compute_challenge_features(df):
    """Compute expanding challenge features from ABS data.

    Features:
    - batter_challenge_edge: net favorable overturns per PA (expanding)
    - opposing_catcher_challenge_edge: net favorable overturns for opposing catcher (expanding)
    - pitcher_challenge_exposure: net unfavorable overturns against pitcher (expanding)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    has_challenges = df["challenge_player_id"].notna().sum()
    if has_challenges == 0:
        print(f"  No challenge data found. ABS features not available yet.")
        df["batter_challenge_edge"] = np.nan
        df["pitcher_challenge_exposure"] = np.nan
        return df

    print(f"  {has_challenges} PAs with challenge data")

    # For each PA with a challenge, determine the "value" to the batter:
    # +1 if overturned in batter's favor (batter challenged and won, OR opposing challenge failed)
    # -1 if overturned against batter (opposing challenge succeeded, OR batter challenge failed)
    # 0 if no challenge
    def _batter_challenge_value(row):
        if pd.isna(row["challenge_overturned"]):
            return 0
        batter_side = row["challenge_team_batting"]
        overturned = row["challenge_overturned"]
        if batter_side and overturned:
            return 1  # Batter challenged, call overturned in batter's favor
        elif batter_side and not overturned:
            return 0  # Batter challenged, lost — no count change
        elif not batter_side and overturned:
            return -1  # Fielding team challenged, overturned against batter
        else:
            return 0  # Fielding team challenged, lost — no count change

    df["challenge_value"] = df.apply(_batter_challenge_value, axis=1)

    # Batter challenge edge: expanding sum of challenge values / total PAs
    batter_dates = df.groupby(["batter_id", "date"]).agg(
        cv_sum=("challenge_value", "sum"),
        n_pas=("challenge_value", "count"),
    ).reset_index().sort_values(["batter_id", "date"])

    batter_dates["cum_cv"] = batter_dates.groupby("batter_id")["cv_sum"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    batter_dates["cum_pas"] = batter_dates.groupby("batter_id")["n_pas"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    batter_dates["batter_challenge_edge"] = np.where(
        batter_dates["cum_pas"] > 0,
        batter_dates["cum_cv"] / batter_dates["cum_pas"],
        np.nan,
    )

    # Pitcher challenge exposure: inverse — how often do challenges go against this pitcher
    pitcher_dates = df.groupby(["pitcher_id", "date"]).agg(
        cv_sum=("challenge_value", "sum"),
        n_pas=("challenge_value", "count"),
    ).reset_index().sort_values(["pitcher_id", "date"])

    pitcher_dates["cum_cv"] = pitcher_dates.groupby("pitcher_id")["cv_sum"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    pitcher_dates["cum_pas"] = pitcher_dates.groupby("pitcher_id")["n_pas"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    # Positive = challenges help batters against this pitcher
    pitcher_dates["pitcher_challenge_exposure"] = np.where(
        pitcher_dates["cum_pas"] > 0,
        pitcher_dates["cum_cv"] / pitcher_dates["cum_pas"],
        np.nan,
    )

    # Merge back
    df = df.merge(
        batter_dates[["batter_id", "date", "batter_challenge_edge"]]
        .drop_duplicates(subset=["batter_id", "date"]),
        on=["batter_id", "date"], how="left",
    )
    df = df.merge(
        pitcher_dates[["pitcher_id", "date", "pitcher_challenge_exposure"]]
        .drop_duplicates(subset=["pitcher_id", "date"]),
        on=["pitcher_id", "date"], how="left",
    )

    return df


def main():
    print("Loading data...", flush=True)
    proc = Path("data/processed")
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  {len(df):,} PAs", flush=True)

    print("Computing base features...", flush=True)
    df = compute_all_features(df)

    print("\nComputing challenge features...", flush=True)
    df = compute_challenge_features(df)

    # Check if we have enough challenge data
    has_edge = df["batter_challenge_edge"].notna().sum()
    if has_edge == 0:
        print("\n  No challenge features available yet.")
        print("  ABS started 2026-03-25. Pull and build 2026 data first:")
        print("    bts data pull --start 2026-03-25 --end <today>")
        print("    bts data build --seasons 2026")
        return

    print(f"\n  {has_edge:,} PAs with challenge edge values")

    # Report challenge feature stats
    print(f"\n  Challenge feature coverage:")
    for col in ["batter_challenge_edge", "pitcher_challenge_exposure"]:
        nn = df[col].notna().sum()
        if nn > 0:
            vals = df[col].dropna()
            print(f"    {col}: {nn:,} non-null, mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Quick P@1 check on recent data (if enough 2026 games)
    df_2026 = df[df["season"] == 2026]
    if len(df_2026) < 1000:
        print(f"\n  Only {len(df_2026)} PAs in 2026 — too early to test P@1.")
        print(f"  Need ~30+ game days for meaningful comparison.")
        print(f"  Check back in a few weeks.")
        return

    n_days = df_2026["date"].nunique()
    print(f"\n  2026 data: {len(df_2026):,} PAs across {n_days} days")

    if n_days < 14:
        print(f"  Need at least 14 days for validation. Check back later.")
        return

    # Train on pre-2026 + early 2026, test on recent 2026
    challenge_cols = BASELINE_COLS + ["batter_challenge_edge", "pitcher_challenge_exposure"]
    test_dates = sorted(df_2026["date"].unique())
    split = test_dates[len(test_dates) // 2]

    train = df[df["date"] < split]
    test = df[(df["date"] >= split) & (df["season"] == 2026)]

    # Baseline model
    tX = train[BASELINE_COLS]
    ty = train["is_hit"]
    valid = tX.notna().any(axis=1)
    base_model = lgb.LGBMClassifier(**LGB_PARAMS)
    base_model.fit(tX[valid], ty[valid])

    # Challenge model
    tX_c = train[challenge_cols]
    valid_c = tX_c.notna().any(axis=1)
    challenge_model = lgb.LGBMClassifier(**LGB_PARAMS)
    challenge_model.fit(tX_c[valid_c], ty[valid_c])

    # Evaluate
    test_dates_list = sorted(test["date"].unique())
    base_hits = []
    chal_hits = []

    for day in test_dates_list:
        day_data = test[test["date"] == day].copy()

        for model, cols, results in [
            (base_model, BASELINE_COLS, base_hits),
            (challenge_model, challenge_cols, chal_hits),
        ]:
            day_data["p_hit"] = model.predict_proba(day_data[cols])[:, 1]
            game = day_data.groupby(["batter_id", "game_pk"]).agg(
                p_game=("p_hit", lambda x: 1 - np.prod(1 - x.values)),
                actual=("is_hit", "max"),
            ).reset_index()
            top = game.nlargest(1, "p_game").iloc[0]
            results.append(int(top["actual"]))

    base_p1 = np.mean(base_hits)
    chal_p1 = np.mean(chal_hits)
    delta = chal_p1 - base_p1

    print(f"\n  {'='*50}")
    print(f"  SHADOW MODEL REPORT")
    print(f"  {'='*50}")
    print(f"  Test period: {len(test_dates_list)} days")
    print(f"  Baseline P@1:  {base_p1:.1%}")
    print(f"  Challenge P@1: {chal_p1:.1%}")
    print(f"  Delta:         {delta:+.1%}")

    if delta > 0.02:
        print(f"\n  ** SIGNAL DETECTED: Challenge features improving P@1 by {delta:+.1%}")
        print(f"  ** Consider promoting to main model.")
    elif delta > 0:
        print(f"\n  Weak positive signal ({delta:+.1%}). Keep monitoring.")
    else:
        print(f"\n  No improvement yet. Feature needs more data to mature.")


if __name__ == "__main__":
    main()
