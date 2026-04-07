# Architecture Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and align BTS architecture across five sequential phases, using MDP P(57) as the primary metric.

**Architecture:** Each phase is a standalone experiment script that loads PA parquets, computes features, runs a 5-season walk-forward backtest, computes quality bins, and solves the MDP. Phases are sequential — each depends on the prior phase's outcome. No production code changes until results are in.

**Tech Stack:** Python 3.12, LightGBM, pandas, numpy. Existing `bts.simulate` and `bts.features` modules.

---

### Task 1: Shared Evaluation Harness

Build a reusable evaluation function so all phases use identical metrics. Every subsequent task imports from this file.

**Files:**
- Create: `scripts/arch_eval.py`
- Test: manual — run with `--smoke` flag on 1 season

- [ ] **Step 1: Write the evaluation harness**

```python
"""Shared evaluation harness for architecture alignment experiments.

Each phase imports compute_metrics() which runs:
  walk-forward backtest → quality bins → MDP solve → metrics dict

Usage:
    from arch_eval import load_data, compute_metrics, print_comparison
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LATE_PHASE_DAYS = 30  # Sept-only bins (matches deployed policy)


def load_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load all PA parquets and compute features."""
    from bts.features.compute import compute_all_features

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    if not dfs:
        raise RuntimeError("No parquet files found. Run 'bts data build' first.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Computing features on {len(df):,} PAs...", file=sys.stderr)
    df = compute_all_features(df)
    return df


def walk_forward_backtest(
    df: pd.DataFrame,
    test_seasons: list[int],
    blend_configs: list[tuple],
    lgb_params: dict,
    game_level: bool = False,
    retrain_every: int = 7,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run blend walk-forward and return profile DataFrames.

    If game_level=True, aggregates to batter-game rows before training
    and predicts P(game_hit) directly. Otherwise uses PA-level predictions
    with per-model game aggregation (live-aligned averaging order).

    Returns DataFrame with [date, rank, batter_id, p_game_hit, actual_hit, game_time].
    """
    import lightgbm as lgb
    from bts.features.compute import TRAIN_START_YEAR

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if game_level:
        # Aggregate to one row per batter-game
        # Keep first value of date-level features (identical across PAs)
        feature_cols = set()
        for _, cols in blend_configs:
            feature_cols.update(cols)

        agg_dict = {
            "is_hit": "max",  # target: did batter get >=1 hit
            "season": "first",
            "lineup_position": "first",
            "game_time": "first",
        }
        for col in feature_cols:
            if col in df.columns and col not in agg_dict:
                agg_dict[col] = "first"

        df = df.groupby(["batter_id", "game_pk", "date"]).agg(agg_dict).reset_index()

    all_profiles = []

    for test_season in test_seasons:
        season_data = df[df["season"] == test_season]
        test_start = season_data["date"].min()
        train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)].copy()
        test_data = season_data.copy()
        test_dates = sorted(test_data["date"].unique())

        print(f"Blend walk-forward ({test_season}): {len(test_dates)} test days, "
              f"train pool: {len(train_pool):,} rows, "
              f"{len(blend_configs)} models", file=sys.stderr)

        blend = None
        for i, day in enumerate(test_dates):
            day_data = test_data[test_data["date"] == day].copy()

            # Retrain periodically
            if blend is None or (i % retrain_every == 0):
                available = pd.concat([train_pool, test_data[test_data["date"] < day]])
                train_y = available["is_hit"]

                blend = {}
                for name, cols in blend_configs:
                    train_X = available[cols]
                    mask = train_X.notna().any(axis=1)
                    model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                    model.fit(train_X[mask], train_y[mask])
                    blend[name] = (model, cols)

                if (i + 1) % 30 == 0 or i == 0:
                    print(f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()}) "
                          f"— retrained on {len(available):,} rows", file=sys.stderr)

            if game_level:
                # Game-level: each model predicts P(game_hit) directly, average
                game_scores = {}
                for name, (model, cols) in blend.items():
                    pred_X = day_data[cols]
                    valid = pred_X.notna().any(axis=1)
                    if valid.any():
                        preds = model.predict_proba(pred_X[valid])[:, 1]
                        for idx, p in zip(day_data.index[valid], preds):
                            bid = day_data.at[idx, "batter_id"]
                            gpk = day_data.at[idx, "game_pk"]
                            key = (bid, gpk)
                            game_scores.setdefault(key, []).append(float(p))

                game_preds = day_data[["batter_id", "game_pk", "is_hit"]].copy()
                game_preds = game_preds.rename(columns={"is_hit": "actual_hit"})
                game_preds["p_game_hit"] = game_preds.apply(
                    lambda r: np.nanmean(game_scores.get((r["batter_id"], r["game_pk"]), [np.nan])),
                    axis=1,
                )
                if "game_time" in day_data.columns:
                    game_preds["game_time"] = day_data["game_time"].values
            else:
                # PA-level: per-model game aggregation, then average (live-aligned)
                model_game_preds = {}
                for name, (model, cols) in blend.items():
                    pred_X = day_data[cols]
                    valid = pred_X.notna().any(axis=1)
                    probs = pd.Series(np.nan, index=day_data.index)
                    if valid.any():
                        probs[valid] = model.predict_proba(pred_X[valid])[:, 1]
                    day_data[f"_p_{name}"] = probs

                    # Aggregate this model to game level
                    model_game = day_data.groupby(["batter_id", "game_pk"]).agg(
                        **{f"p_{name}": (f"_p_{name}", lambda x: 1 - np.prod(1 - x.values))}
                    ).reset_index()
                    model_game_preds[name] = model_game.set_index(["batter_id", "game_pk"])[f"p_{name}"]

                # Average across models at game level
                game_preds = day_data.groupby(["batter_id", "game_pk"]).agg(
                    actual_hit=("is_hit", "max"),
                ).reset_index()

                blend_df = pd.DataFrame(model_game_preds)
                game_preds["p_game_hit"] = game_preds.apply(
                    lambda r: blend_df.loc[(r["batter_id"], r["game_pk"])].mean()
                    if (r["batter_id"], r["game_pk"]) in blend_df.index else np.nan,
                    axis=1,
                )

                # Carry game_time for densest bucket testing
                if "game_time" in day_data.columns:
                    gt = day_data.groupby(["batter_id", "game_pk"])["game_time"].first().reset_index()
                    game_preds = game_preds.merge(gt, on=["batter_id", "game_pk"], how="left")

            # Rank and take top N
            game_preds = game_preds.dropna(subset=["p_game_hit"])
            game_preds = game_preds.nlargest(top_n, "p_game_hit").reset_index(drop=True)
            game_preds["rank"] = range(1, len(game_preds) + 1)
            game_preds["date"] = pd.Timestamp(day).date()

            profile_cols = ["date", "rank", "batter_id", "p_game_hit", "actual_hit"]
            if "game_time" in game_preds.columns:
                profile_cols.append("game_time")
            all_profiles.append(game_preds[profile_cols])

            # Clean up temp columns
            for name in blend:
                col = f"_p_{name}"
                if col in day_data.columns:
                    day_data.drop(columns=[col], inplace=True)

    result = pd.concat(all_profiles, ignore_index=True)
    print(f"  Done: {len(result)} profile rows", file=sys.stderr)
    return result


def compute_metrics(profiles: pd.DataFrame, season_length: int = 180) -> dict:
    """Compute all metrics from backtest profiles: P@1, quality bins, MDP P(57)."""
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.mdp import solve_mdp

    profiles = profiles.copy()
    profiles["date"] = pd.to_datetime(profiles["date"])

    rank1 = profiles[profiles["rank"] == 1]

    # P@1 by season
    p_at_1 = {}
    for season_start_year in sorted(rank1["date"].dt.year.unique()):
        season_mask = rank1["date"].dt.year == season_start_year
        p_at_1[season_start_year] = float(rank1[season_mask]["actual_hit"].mean())
    p_at_1["avg"] = float(rank1["actual_hit"].mean())

    # Quality bins — early (all) and late (September)
    early_bins = compute_bins(profiles)
    late_profiles = profiles[profiles["date"].dt.month == 9]
    late_bins = compute_bins(late_profiles) if len(late_profiles) >= 50 else None

    # MDP solve
    solution = solve_mdp(early_bins, season_length=season_length,
                         late_bins=late_bins, late_phase_days=LATE_PHASE_DAYS)

    # Longest replay streak
    hits = rank1.sort_values("date")["actual_hit"].values
    max_streak = cur = 0
    for h in hits:
        cur = cur + 1 if h else 0
        max_streak = max(max_streak, cur)

    return {
        "p_at_1": p_at_1,
        "mdp_p57": solution.optimal_p57,
        "early_bins": early_bins,
        "late_bins": late_bins,
        "longest_replay": max_streak,
        "mean_p_hit": float(rank1["p_game_hit"].mean()),
    }


def print_comparison(results: dict[str, dict], label: str = ""):
    """Print formatted comparison table."""
    if label:
        print(f"\n{'='*80}")
        print(f"  {label}")

    # Determine which seasons are present
    sample = next(iter(results.values()))
    seasons = [k for k in sample["p_at_1"] if k != "avg"]

    header = f"{'':>15}"
    for s in seasons:
        header += f" | {s:>8}"
    header += f" | {'Avg P@1':>8} | {'MDP P(57)':>10} | {'Replay':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, r in results.items():
        row = f"{name:>15}"
        for s in seasons:
            row += f" | {r['p_at_1'].get(s, 0):>8.4f}"
        row += f" | {r['p_at_1']['avg']:>8.4f} | {r['mdp_p57']:>10.6f} | {r['longest_replay']:>7}"
        print(row)
    print("=" * len(header))

    # Delta from first entry
    names = list(results.keys())
    if len(names) >= 2:
        base = results[names[0]]
        for name in names[1:]:
            r = results[name]
            dp = r["p_at_1"]["avg"] - base["p_at_1"]["avg"]
            dm = r["mdp_p57"] - base["mdp_p57"]
            ratio = r["mdp_p57"] / base["mdp_p57"] if base["mdp_p57"] > 0 else 0
            print(f"\n  {name} vs {names[0]}:")
            print(f"    P@1 avg: {dp:+.4f}")
            print(f"    MDP P(57): {dm:+.6f} ({ratio:.2f}x)")
```

- [ ] **Step 2: Smoke test the harness**

Run on a single season to verify it works:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "
from scripts.arch_eval import load_data, walk_forward_backtest, compute_metrics
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS

df = load_data()
profiles = walk_forward_backtest(df, [2025], BLEND_CONFIGS, LGB_PARAMS)
metrics = compute_metrics(profiles)
print(f'P@1 2025: {metrics[\"p_at_1\"][2025]:.4f}')
print(f'MDP P(57): {metrics[\"mdp_p57\"]:.6f}')
"
```

Expected: runs in ~25 min, produces P@1 and MDP P(57) values.

- [ ] **Step 3: Commit**

```bash
git add scripts/arch_eval.py
git commit -m "feat: shared evaluation harness for architecture alignment experiments"
```

---

### Task 2: Phase 1 — Game-Level vs PA-Level Experiment

**Files:**
- Create: `scripts/phase1_game_vs_pa.py`
- Read: `scripts/arch_eval.py`, `src/bts/features/compute.py`

- [ ] **Step 1: Write the experiment script**

```python
"""Phase 1: Game-level vs PA-level modeling comparison.

Tests whether predicting P(game_hit) directly performs comparably to
the current PA-level P(hit|PA) → aggregation approach.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase1_game_vs_pa.py
"""

import sys
sys.path.insert(0, "scripts")

from arch_eval import (
    load_data, walk_forward_backtest, compute_metrics, print_comparison,
)
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Game-level blend configs: same structure but add lineup_position
GAME_FEATURE_COLS = FEATURE_COLS + ["lineup_position"]
GAME_BLEND_CONFIGS = [
    ("baseline", GAME_FEATURE_COLS),
    ("barrel", GAME_FEATURE_COLS + ["batter_barrel_rate_30g"]),
    ("hard_hit", GAME_FEATURE_COLS + ["batter_hard_hit_rate_30g"]),
    ("sweet_spot", GAME_FEATURE_COLS + ["batter_sweet_spot_rate_30g"]),
    ("avg_ev", GAME_FEATURE_COLS + ["batter_avg_ev_30g"]),
    ("velo", GAME_FEATURE_COLS + ["pitcher_avg_velo_30g"]),
    ("spin", GAME_FEATURE_COLS + ["pitcher_avg_spin_30g"]),
    ("extension", GAME_FEATURE_COLS + ["pitcher_avg_extension_30g"]),
    ("break", GAME_FEATURE_COLS + ["pitcher_break_total_30g"]),
    ("velo_faced", GAME_FEATURE_COLS + ["batter_avg_velo_faced_30g"]),
    ("best_two", GAME_FEATURE_COLS + ["batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
    ("all_statcast", GAME_FEATURE_COLS + STATCAST_COLS),
]


def main():
    df = load_data()

    results = {}

    # --- PA-level baseline (live-aligned averaging) ---
    print(f"\n{'='*60}", file=sys.stderr)
    print("PA-LEVEL (live-aligned averaging order)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    pa_profiles = walk_forward_backtest(
        df, TEST_SEASONS, BLEND_CONFIGS, LGB_PARAMS, game_level=False,
    )
    results["pa_level"] = compute_metrics(pa_profiles)

    # --- Game-level ---
    print(f"\n{'='*60}", file=sys.stderr)
    print("GAME-LEVEL (lineup_position as feature)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    game_profiles = walk_forward_backtest(
        df, TEST_SEASONS, GAME_BLEND_CONFIGS, LGB_PARAMS, game_level=True,
    )
    results["game_level"] = compute_metrics(game_profiles)

    print_comparison(results, "Phase 1: Game-Level vs PA-Level")

    # Quality bin detail
    print("\nQuality bin detail (early phase):")
    print(f"  {'Bin':>4} | {'PA P(hit)':>10} {'Game P(hit)':>12} | {'PA freq':>8} {'Game freq':>10}")
    print(f"  {'-'*60}")
    pa_bins = results["pa_level"]["early_bins"].bins
    game_bins = results["game_level"]["early_bins"].bins
    for pb, gb in zip(pa_bins, game_bins):
        print(f"  Q{pb.index+1:>3} | {pb.p_hit:>10.4f} {gb.p_hit:>12.4f} "
              f"| {pb.frequency:>8.3f} {gb.frequency:>10.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase1_game_vs_pa.py 2>&1 | tee data/validation/phase1_results.txt
```

Expected: ~100 min (2 variants × 5 seasons × ~10 min each). Prints comparison table with P@1 per season, MDP P(57), quality bins.

- [ ] **Step 3: Evaluate results and decide**

Read the output. Apply decision criteria from spec:
- If game-level P(57) >= PA-level P(57): adopt game-level, skip Phase 2
- If PA-level P(57) > game-level by >10% relative: keep PA-level, proceed to Phase 2

- [ ] **Step 4: Commit results**

```bash
git add scripts/phase1_game_vs_pa.py data/validation/phase1_results.txt
git commit -m "experiment: Phase 1 game-level vs PA-level modeling results"
```

---

### Task 3: Phase 2 — Align Backtest Aggregation (Conditional)

**Skip this task entirely if game-level won in Phase 1.** The live-aligned averaging is already built into `arch_eval.py`'s `walk_forward_backtest(game_level=False)` path.

**Files:**
- Read: `scripts/arch_eval.py` (the alignment is already implemented in the harness)

- [ ] **Step 1: Run 5-season baseline with aligned aggregation**

If PA-level won Phase 1, the aligned aggregation result was already computed in Phase 1 as `pa_level`. Compare it against the current deployed baseline from `data/validation/scorecard_baseline.json`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "
import json
with open('data/validation/scorecard_baseline.json') as f:
    baseline = json.load(f)
print(f'Deployed baseline P@1: {baseline[\"precision\"][\"1\"]:.4f}')
print(f'Deployed baseline MDP P(57): {baseline[\"p_57_mdp\"]:.6f}')
print()
print('Compare with Phase 1 pa_level results above.')
print('The difference is the effect of live-aligned averaging order.')
"
```

- [ ] **Step 2: Document the new baseline**

If the aligned aggregation changes the baseline, note the delta and update `data/validation/scorecard_baseline.json` with the corrected values.

- [ ] **Step 3: Commit**

```bash
git add data/validation/scorecard_baseline.json
git commit -m "fix: update baseline scorecard with live-aligned aggregation"
```

---

### Task 4: Phase 3 — Densest Bucket Ablation

**Files:**
- Create: `scripts/phase3_densest_bucket.py`
- Read: `scripts/arch_eval.py`, `src/bts/strategy.py:111-138`

- [ ] **Step 1: Write the densest bucket ablation script**

```python
"""Phase 3: Test whether densest bucket still helps with the blend.

Compares P(57) with and without densest bucket filtering.
Requires game_time in profiles (added by arch_eval harness).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase3_densest_bucket.py
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import (
    load_data, walk_forward_backtest, compute_metrics, print_comparison,
)
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Use whichever modeling level won Phase 1.
# Set this based on Phase 1 results:
GAME_LEVEL = False  # Change to True if game-level won

OVERRIDE_THRESHOLD = 0.78


def classify_et_hour(game_time_str):
    """Convert UTC game time to ET hour."""
    try:
        utc = datetime.fromisoformat(str(game_time_str).replace("Z", "+00:00"))
        return (utc - timedelta(hours=4)).hour
    except Exception:
        return 18


def apply_densest_bucket(day_profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply densest bucket filter to a single day's top-10 profiles."""
    if "game_time" not in day_profiles.columns or day_profiles["game_time"].isna().all():
        return day_profiles

    df = day_profiles.copy()
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
        return df  # fallback to unfiltered if bucket is empty

    # Re-rank within filtered set
    filtered = filtered.sort_values("p_game_hit", ascending=False).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)
    return filtered


def apply_densest_to_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply densest bucket day by day, return re-ranked profiles."""
    result = []
    for date, day_df in profiles.groupby("date"):
        filtered = apply_densest_bucket(day_df)
        result.append(filtered)
    return pd.concat(result, ignore_index=True)


def main():
    df = load_data()

    # Determine blend configs based on Phase 1 outcome
    if GAME_LEVEL:
        game_feature_cols = FEATURE_COLS + ["lineup_position"]
        blend_configs = [
            ("baseline", game_feature_cols),
            ("barrel", game_feature_cols + ["batter_barrel_rate_30g"]),
            ("hard_hit", game_feature_cols + ["batter_hard_hit_rate_30g"]),
            ("sweet_spot", game_feature_cols + ["batter_sweet_spot_rate_30g"]),
            ("avg_ev", game_feature_cols + ["batter_avg_ev_30g"]),
            ("velo", game_feature_cols + ["pitcher_avg_velo_30g"]),
            ("spin", game_feature_cols + ["pitcher_avg_spin_30g"]),
            ("extension", game_feature_cols + ["pitcher_avg_extension_30g"]),
            ("break", game_feature_cols + ["pitcher_break_total_30g"]),
            ("velo_faced", game_feature_cols + ["batter_avg_velo_faced_30g"]),
            ("best_two", game_feature_cols + ["batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
            ("all_statcast", game_feature_cols + STATCAST_COLS),
        ]
    else:
        blend_configs = BLEND_CONFIGS

    # Run backtest once — get profiles with game_time
    print(f"\n{'='*60}", file=sys.stderr)
    print("Running backtest for densest bucket ablation...", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    profiles = walk_forward_backtest(
        df, TEST_SEASONS, blend_configs, LGB_PARAMS, game_level=GAME_LEVEL,
    )

    results = {}

    # Without densest bucket (raw blend ranking)
    results["no_bucket"] = compute_metrics(profiles)

    # With densest bucket
    filtered_profiles = apply_densest_to_profiles(profiles)
    results["with_bucket"] = compute_metrics(filtered_profiles)

    print_comparison(results, "Phase 3: Densest Bucket Ablation")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase3_densest_bucket.py 2>&1 | tee data/validation/phase3_results.txt
```

Expected: ~50 min (one backtest run, two evaluations). One run because both variants use the same profiles — the densest bucket is applied post-hoc.

- [ ] **Step 3: Evaluate and commit**

```bash
git add scripts/phase3_densest_bucket.py data/validation/phase3_results.txt
git commit -m "experiment: Phase 3 densest bucket ablation results"
```

---

### Task 5: Phase 4 — Alt-Params Blend Member

**Files:**
- Create: `scripts/phase4_alt_params_blend.py`
- Read: `scripts/arch_eval.py`, `src/bts/features/compute.py`

- [ ] **Step 1: Write the alt-params blend experiment**

```python
"""Phase 4: Test 13-model blend with alt-params feature variant.

Adds a blend member trained on features computed with:
  platoon threshold=40, gb min_periods=15,
  venue min_periods=30, statcast min_periods=3

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase4_alt_params_blend.py
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import (
    load_data, walk_forward_backtest, compute_metrics, print_comparison,
)
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS, _is_barrel

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Set based on Phase 1 outcome:
GAME_LEVEL = False  # Change to True if game-level won

# Alt parameter values
ALT_PLATOON_THRESHOLD = 40
ALT_GB_MIN_PERIODS = 15
ALT_VENUE_MIN_PERIODS = 30
ALT_STATCAST_MIN_PERIODS = 3


def compute_alt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add alt-params feature columns alongside originals."""

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

    return df


def build_alt_blend_configs(game_level: bool) -> list[tuple]:
    """Build 13-model blend configs with alt-params member."""
    base_cols = FEATURE_COLS + (["lineup_position"] if game_level else [])

    # Alt feature column mapping
    alt_col_map = {
        "platoon_hr": "platoon_hr_alt",
        "batter_gb_hit_rate": "batter_gb_hit_rate_alt",
        "park_factor": "park_factor_alt",
        "batter_barrel_rate_30g": "batter_barrel_rate_30g_alt",
        "batter_hard_hit_rate_30g": "batter_hard_hit_rate_30g_alt",
        "batter_sweet_spot_rate_30g": "batter_sweet_spot_rate_30g_alt",
        "batter_avg_ev_30g": "batter_avg_ev_30g_alt",
    }

    alt_base = [alt_col_map.get(c, c) for c in base_cols]

    # Start with the standard 12
    if game_level:
        configs = [
            ("baseline", base_cols),
            ("barrel", base_cols + ["batter_barrel_rate_30g"]),
            ("hard_hit", base_cols + ["batter_hard_hit_rate_30g"]),
            ("sweet_spot", base_cols + ["batter_sweet_spot_rate_30g"]),
            ("avg_ev", base_cols + ["batter_avg_ev_30g"]),
            ("velo", base_cols + ["pitcher_avg_velo_30g"]),
            ("spin", base_cols + ["pitcher_avg_spin_30g"]),
            ("extension", base_cols + ["pitcher_avg_extension_30g"]),
            ("break", base_cols + ["pitcher_break_total_30g"]),
            ("velo_faced", base_cols + ["batter_avg_velo_faced_30g"]),
            ("best_two", base_cols + ["batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
            ("all_statcast", base_cols + STATCAST_COLS),
        ]
    else:
        configs = list(BLEND_CONFIGS)

    # Add 13th: alt-params baseline
    configs.append(("alt_params", alt_base))

    return configs


def main():
    df = load_data()

    print("Computing alt-params features...", file=sys.stderr)
    df = compute_alt_features(df)

    # Build 12-model and 13-model blend configs
    base_configs = build_alt_blend_configs(GAME_LEVEL)
    configs_12 = base_configs[:-1]  # standard 12
    configs_13 = base_configs       # 12 + alt_params

    results = {}

    # 12-model baseline
    print(f"\n{'='*60}", file=sys.stderr)
    print("12-MODEL BLEND (baseline)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    profiles_12 = walk_forward_backtest(
        df, TEST_SEASONS, configs_12, LGB_PARAMS, game_level=GAME_LEVEL,
    )
    results["12_model"] = compute_metrics(profiles_12)

    # 13-model blend
    print(f"\n{'='*60}", file=sys.stderr)
    print("13-MODEL BLEND (+ alt_params)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    profiles_13 = walk_forward_backtest(
        df, TEST_SEASONS, configs_13, LGB_PARAMS, game_level=GAME_LEVEL,
    )
    results["13_model"] = compute_metrics(profiles_13)

    print_comparison(results, "Phase 4: Alt-Params Blend Member")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase4_alt_params_blend.py 2>&1 | tee data/validation/phase4_results.txt
```

Expected: ~100 min (2 variants × 5 seasons). Prints comparison table.

- [ ] **Step 3: Evaluate and commit**

```bash
git add scripts/phase4_alt_params_blend.py data/validation/phase4_results.txt
git commit -m "experiment: Phase 4 alt-params blend member results"
```

---

### Task 6: Phase 5 — Team Bullpen Composite (Deferred)

**Depends on Phases 1-4 being resolved.** Not implemented now — design only.

**Files:**
- Create: `scripts/phase5_bullpen_composite.py` (skeleton with TODO)

- [ ] **Step 1: Write skeleton with design notes**

```python
"""Phase 5: Team bullpen composite feature.

DEFERRED — implement after Phases 1-4 are resolved.

Design:
- Compute rolling team bullpen composite from reliever PAs:
  - Identify relievers: pitchers who are NOT the game's starter
  - For each team+date, compute rolling 30-day avg of:
    pitcher_hr, pitcher_entropy, Statcast features
    across all relievers who pitched for that team
  - Shift by 1 day to prevent leakage

- If game-level model: add as opp_bullpen_hr_30g feature
- If PA-level model: replace league-avg in starter/reliever split

- Key question: how to identify relievers in the data?
  Option A: pitcher_id != probable_pitcher_id (need to join schedule)
  Option B: PAs where lineup_position > 0 and pitcher changed (from play data)
  Option C: Use _is_opener flag + pitch count heuristic

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase5_bullpen_composite.py
"""

# Implementation deferred until Phases 1-4 complete.
# See docs/superpowers/specs/2026-04-07-architecture-alignment-design.md
```

- [ ] **Step 2: Commit skeleton**

```bash
git add scripts/phase5_bullpen_composite.py
git commit -m "docs: Phase 5 bullpen composite skeleton (deferred)"
```

---

### Task 7: Apply Winning Changes to Production

**Only after all phases are evaluated and decisions made.**

- [ ] **Step 1: Summarize results**

Create `data/validation/architecture_alignment_summary.md` with:
- Phase 1 outcome: game-level or PA-level
- Phase 2 outcome: aggregation change (if applicable)
- Phase 3 outcome: keep or remove densest bucket
- Phase 4 outcome: 12 or 13 model blend
- Combined P(57) vs original baseline

- [ ] **Step 2: Apply changes to production code**

Based on outcomes, modify:
- `src/bts/simulate/backtest_blend.py` — modeling level + aggregation
- `src/bts/features/compute.py` — alt feature columns (if Phase 4 won)
- `src/bts/model/predict.py` — BLEND_CONFIGS, FEATURE_COLS, game-level predict
- `src/bts/strategy.py` — remove densest bucket (if Phase 3 says remove)

- [ ] **Step 3: Re-run full scorecard and MDP solver**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard --diff data/validation/scorecard_baseline.json
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy.npz
```

- [ ] **Step 4: Deploy to Pi5**

```bash
git push origin main
ssh stonehengee@pi5.local "cd ~/projects/bts && git pull -q origin main"
```

- [ ] **Step 5: Commit all production changes**

```bash
git add -A
git commit -m "feat: apply architecture alignment results — [summary of changes]"
```
