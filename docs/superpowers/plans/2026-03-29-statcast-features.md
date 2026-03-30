# Statcast Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Statcast data from existing game feeds, build 9 new rolling features, and validate which improve P@1.

**Architecture:** Add 9 new columns to the PA schema (3 hitData, 6 pitchData lists). Compute 9 new rolling features in compute.py using the same temporal guard pattern as existing features. Rebuild parquets, then A/B backtest.

**Tech Stack:** Python 3.12, pandas, LightGBM, pyarrow (existing stack — no new deps)

---

### Task 1: Add Statcast fields to schema

**Files:**
- Modify: `src/bts/data/schema.py`
- Test: `tests/data/test_schema.py`

- [ ] **Step 1: Write failing test for new columns**

Add to `tests/data/test_schema.py`:

```python
def test_pa_columns_has_statcast_fields():
    from bts.data.schema import PA_COLUMNS
    statcast_fields = [
        "trajectory", "hardness", "total_distance",
        "pitch_speeds", "pitch_spin_rates", "pitch_extensions",
        "pitch_break_vertical", "pitch_break_horizontal",
        "pitch_end_speeds",
    ]
    for field in statcast_fields:
        assert field in PA_COLUMNS, f"Missing Statcast field: {field}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_schema.py::test_pa_columns_has_statcast_fields -v`
Expected: FAIL with "Missing Statcast field: trajectory"

- [ ] **Step 3: Add new columns to PA_COLUMNS**

In `src/bts/data/schema.py`, add after `"launch_angle",`:

```python
    "trajectory",
    "hardness",
    "total_distance",
    "pitch_speeds",
    "pitch_end_speeds",
    "pitch_spin_rates",
    "pitch_extensions",
    "pitch_break_vertical",
    "pitch_break_horizontal",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_schema.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/schema.py tests/data/test_schema.py
git commit -m "schema: add Statcast fields to PA_COLUMNS"
```

---

### Task 2: Extract Statcast fields in parse_game_feed

**Files:**
- Modify: `src/bts/data/build.py:83-146`
- Modify: `tests/conftest.py` (add Statcast data to fixture)
- Test: `tests/data/test_build.py`

- [ ] **Step 1: Write failing tests for Statcast extraction**

Add to `tests/data/test_build.py`:

```python
def test_parse_game_feed_statcast_hit_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["trajectory"] == "line_drive"
    assert single["hardness"] == "hard"
    assert single["total_distance"] == 310.0
    # Strikeout has no hitData
    strikeout = rows[1]
    assert strikeout["trajectory"] is None
    assert strikeout["hardness"] is None
    assert strikeout["total_distance"] is None


def test_parse_game_feed_statcast_pitch_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["pitch_speeds"] == [93.5, 85.2, 84.1, 94.0]
    assert single["pitch_spin_rates"] == [2400, 2700, 1800, 2350]
    assert single["pitch_extensions"] == [6.3, 6.1, 6.2, 6.4]
    assert single["pitch_break_vertical"] == [-15.0, -32.0, -28.0, -14.0]
    assert single["pitch_break_horizontal"] == [8.0, -2.0, -12.0, 9.0]
    assert single["pitch_end_speeds"] == [85.0, 78.0, 76.0, 86.0]
    assert len(single["pitch_speeds"]) == 4
```

- [ ] **Step 2: Update sample_game_feed fixture with Statcast data**

In `tests/conftest.py`, update the pitchData sections to include Statcast fields. For the first play (single), update each pitch event to add `startSpeed`, `endSpeed`, `extension`, and `breaks`. Also add `hitData` fields to the last pitch event.

First pitch (FF):
```python
"pitchData": {
    "coordinates": {"pX": -0.5, "pZ": 2.8},
    "strikeZoneTop": 3.4,
    "strikeZoneBottom": 1.7,
    "startSpeed": 93.5,
    "endSpeed": 85.0,
    "extension": 6.3,
    "breaks": {
        "spinRate": 2400,
        "breakVertical": -15.0,
        "breakHorizontal": 8.0,
    },
},
```

Second pitch (SL):
```python
"pitchData": {
    "coordinates": {"pX": 0.3, "pZ": 2.1},
    "strikeZoneTop": 3.4,
    "strikeZoneBottom": 1.7,
    "startSpeed": 85.2,
    "endSpeed": 78.0,
    "extension": 6.1,
    "breaks": {
        "spinRate": 2700,
        "breakVertical": -32.0,
        "breakHorizontal": -2.0,
    },
},
```

Third pitch (CH):
```python
"pitchData": {
    "coordinates": {"pX": 0.1, "pZ": 1.9},
    "strikeZoneTop": 3.4,
    "strikeZoneBottom": 1.7,
    "startSpeed": 84.1,
    "endSpeed": 76.0,
    "extension": 6.2,
    "breaks": {
        "spinRate": 1800,
        "breakVertical": -28.0,
        "breakHorizontal": -12.0,
    },
},
```

Fourth pitch (FF, in play):
```python
"pitchData": {
    "coordinates": {"pX": -0.2, "pZ": 2.5},
    "strikeZoneTop": 3.4,
    "strikeZoneBottom": 1.7,
    "startSpeed": 94.0,
    "endSpeed": 86.0,
    "extension": 6.4,
    "breaks": {
        "spinRate": 2350,
        "breakVertical": -14.0,
        "breakHorizontal": 9.0,
    },
},
"hitData": {
    "launchSpeed": 98.3,
    "launchAngle": 12.0,
    "trajectory": "line_drive",
    "hardness": "hard",
    "totalDistance": 310.0,
},
```

For the second play (strikeout), add the same pitchData fields to each pitch event:

First pitch (FF): `"startSpeed": 95.1, "endSpeed": 87.0, "extension": 6.5, "breaks": {"spinRate": 2300, "breakVertical": -13.0, "breakHorizontal": 7.0}`

Second pitch (SL): `"startSpeed": 86.0, "endSpeed": 79.0, "extension": 6.2, "breaks": {"spinRate": 2650, "breakVertical": -30.0, "breakHorizontal": -3.0}`

Third pitch (CH): `"startSpeed": 85.5, "endSpeed": 77.0, "extension": 6.3, "breaks": {"spinRate": 1750, "breakVertical": -27.0, "breakHorizontal": -11.0}`

- [ ] **Step 3: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_build.py::test_parse_game_feed_statcast_hit_data tests/data/test_build.py::test_parse_game_feed_statcast_pitch_data -v`
Expected: FAIL (fields not in output dict)

- [ ] **Step 4: Update parse_game_feed to extract Statcast fields**

In `src/bts/data/build.py`, modify the pitch loop (lines 83-111) to collect new fields:

```python
        pitch_types = []
        pitch_calls = []
        pitch_px = []
        pitch_pz = []
        pitch_speeds = []
        pitch_end_speeds = []
        pitch_spin_rates = []
        pitch_extensions = []
        pitch_break_vertical = []
        pitch_break_horizontal = []
        sz_top = None
        sz_bottom = None
        launch_speed = None
        launch_angle = None
        trajectory = None
        hardness = None
        total_distance = None

        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            details = event.get("details", {})
            pitch_data = event.get("pitchData", {})
            coords = pitch_data.get("coordinates", {})
            breaks = pitch_data.get("breaks", {})

            pitch_types.append(details.get("type", {}).get("code", "UN"))
            pitch_calls.append(details.get("call", {}).get("code", ""))
            pitch_px.append(coords.get("pX"))
            pitch_pz.append(coords.get("pZ"))
            pitch_speeds.append(pitch_data.get("startSpeed"))
            pitch_end_speeds.append(pitch_data.get("endSpeed"))
            pitch_spin_rates.append(breaks.get("spinRate"))
            pitch_extensions.append(pitch_data.get("extension"))
            pitch_break_vertical.append(breaks.get("breakVertical"))
            pitch_break_horizontal.append(breaks.get("breakHorizontal"))

            sz_top = pitch_data.get("strikeZoneTop", sz_top)
            sz_bottom = pitch_data.get("strikeZoneBottom", sz_bottom)

            hit_data = event.get("hitData")
            if hit_data:
                launch_speed = hit_data.get("launchSpeed")
                launch_angle = hit_data.get("launchAngle")
                trajectory = hit_data.get("trajectory")
                hardness = hit_data.get("hardness")
                total_distance = hit_data.get("totalDistance")
```

And add to the row dict (after `"launch_angle": launch_angle,`):

```python
            "trajectory": trajectory,
            "hardness": hardness,
            "total_distance": total_distance,
            "pitch_speeds": pitch_speeds,
            "pitch_end_speeds": pitch_end_speeds,
            "pitch_spin_rates": pitch_spin_rates,
            "pitch_extensions": pitch_extensions,
            "pitch_break_vertical": pitch_break_vertical,
            "pitch_break_horizontal": pitch_break_horizontal,
```

- [ ] **Step 5: Run all build tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_build.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/bts/data/build.py tests/conftest.py tests/data/test_build.py
git commit -m "feat: extract Statcast fields from game feeds"
```

---

### Task 3: Rebuild all parquets

**Files:**
- No code changes — CLI command only.

- [ ] **Step 1: Rebuild all seasons**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts data build --seasons 2019,2020,2021,2022,2023,2024,2025
```

- [ ] **Step 2: Verify new columns exist**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/pa_2024.parquet')
for col in ['trajectory', 'hardness', 'total_distance', 'pitch_speeds', 'pitch_spin_rates']:
    print(f'{col}: {df[col].notna().sum()}/{len(df)} non-null')
"
```

Expected: trajectory/hardness/total_distance populated for ~68% of PAs (same as launch_speed). pitch_speeds populated for nearly all PAs.

- [ ] **Step 3: Verify row counts match**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "
import pandas as pd
from pathlib import Path
total = 0
for p in sorted(Path('data/processed').glob('pa_*.parquet')):
    df = pd.read_parquet(p)
    print(f'{p.stem}: {len(df):,}')
    total += len(df)
print(f'Total: {total:,}')
"
```

Expected: ~1,519,326 total PAs (same as before rebuild).

---

### Task 4: Add Statcast rolling features to compute.py

**Files:**
- Modify: `src/bts/features/compute.py`

- [ ] **Step 1: Add barrel classification helper**

Add before `compute_all_features`:

```python
def _is_barrel(ev, la):
    """MLB barrel classification from exit velocity + launch angle."""
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    bonus = (min(ev, 116) - 98) * 2
    la_min = max(8, 26 - bonus)
    la_max = min(50, 30 + bonus)
    return la_min <= la <= la_max


def _mean_of_list(lst):
    """Mean of a list, ignoring None values. Returns NaN if empty."""
    if not isinstance(lst, (list, np.ndarray)):
        return np.nan
    vals = [v for v in lst if v is not None]
    return np.mean(vals) if vals else np.nan


def _total_break(vert_list, horiz_list):
    """Mean total break (sqrt(vert^2 + horiz^2)) from per-pitch lists."""
    if not isinstance(vert_list, (list, np.ndarray)):
        return np.nan
    total = []
    for v, h in zip(vert_list, horiz_list):
        if v is not None and h is not None:
            total.append(np.sqrt(v**2 + h**2))
    return np.mean(total) if total else np.nan
```

- [ ] **Step 2: Add batter batted ball features inside compute_all_features**

Add after the `batter_gb_hit_rate` section (after line 103), before the platoon section:

```python
    # --- Batter Statcast batted ball features (date-level) ---
    df["is_barrel"] = df.apply(lambda r: _is_barrel(r["launch_speed"], r["launch_angle"]), axis=1)
    df["is_hard_hit"] = df["launch_speed"].notna() & (df["launch_speed"] >= 95)
    df["is_sweet_spot"] = df["launch_angle"].notna() & (df["launch_angle"] >= 8) & (df["launch_angle"] <= 32)
    df["has_batted_ball"] = df["launch_speed"].notna()

    date_batted = df.groupby(["batter_id", "date"]).agg(
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        sweet_spots=("is_sweet_spot", "sum"),
        batted_balls=("has_batted_ball", "sum"),
        avg_ev=("launch_speed", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
    ).reset_index().sort_values(["batter_id", "date"])

    date_batted["barrel_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["barrels"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["hard_hit_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["hard_hits"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["sweet_spot_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["sweet_spots"] / date_batted["batted_balls"],
        np.nan,
    )

    date_batted["batter_barrel_rate_30g"] = date_batted.groupby("batter_id")["barrel_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_hard_hit_rate_30g"] = date_batted.groupby("batter_id")["hard_hit_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_sweet_spot_rate_30g"] = date_batted.groupby("batter_id")["sweet_spot_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_avg_ev_30g"] = date_batted.groupby("batter_id")["avg_ev"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )

    batter_dates = batter_dates.merge(
        date_batted[["batter_id", "date", "batter_barrel_rate_30g", "batter_hard_hit_rate_30g",
                      "batter_sweet_spot_rate_30g", "batter_avg_ev_30g"]],
        on=["batter_id", "date"], how="left",
    )
```

- [ ] **Step 3: Add pitcher Statcast features inside compute_all_features**

Add after the pitcher entropy section (after line 147):

```python
    # --- Pitcher Statcast features (date-level) ---
    df["pa_avg_velo"] = df["pitch_speeds"].apply(_mean_of_list)
    df["pa_avg_spin"] = df["pitch_spin_rates"].apply(_mean_of_list)
    df["pa_avg_extension"] = df["pitch_extensions"].apply(_mean_of_list)
    df["pa_total_break"] = df.apply(
        lambda r: _total_break(r.get("pitch_break_vertical", []), r.get("pitch_break_horizontal", [])), axis=1
    )

    date_pitch_stats = df.groupby(["pitcher_id", "date"]).agg(
        avg_velo=("pa_avg_velo", "mean"),
        avg_spin=("pa_avg_spin", "mean"),
        avg_extension=("pa_avg_extension", "mean"),
        avg_break=("pa_total_break", "mean"),
    ).reset_index().sort_values(["pitcher_id", "date"])

    date_pitch_stats["pitcher_avg_velo_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_velo"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_avg_spin_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_spin"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_avg_extension_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_extension"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_break_total_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_break"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
```

- [ ] **Step 4: Add batter velo-faced feature**

Add after the pitcher Statcast section:

```python
    # --- Batter: average pitch velocity faced (date-level) ---
    date_velo_faced = df.groupby(["batter_id", "date"])["pa_avg_velo"].mean().reset_index()
    date_velo_faced.columns = ["batter_id", "date", "avg_velo_faced"]
    date_velo_faced = date_velo_faced.sort_values(["batter_id", "date"])
    date_velo_faced["batter_avg_velo_faced_30g"] = date_velo_faced.groupby("batter_id")["avg_velo_faced"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
```

- [ ] **Step 5: Add merge statements for new features**

After the pitcher entropy merge (line ~194), add:

```python
    # Pitcher Statcast
    df = df.merge(
        date_pitch_stats[["pitcher_id", "date", "pitcher_avg_velo_30g", "pitcher_avg_spin_30g",
                           "pitcher_avg_extension_30g", "pitcher_break_total_30g"]]
        .drop_duplicates(subset=["pitcher_id", "date"]),
        on=["pitcher_id", "date"], how="left",
    )

    # Batter velo faced
    df = df.merge(
        date_velo_faced[["batter_id", "date", "batter_avg_velo_faced_30g"]]
        .drop_duplicates(subset=["batter_id", "date"]),
        on=["batter_id", "date"], how="left",
    )
```

- [ ] **Step 6: Update FEATURE_COLS**

Replace the FEATURE_COLS list with:

```python
FEATURE_COLS = [
    "batter_hr_7g",
    "batter_hr_30g",
    "batter_hr_60g",
    "batter_hr_120g",
    "batter_whiff_60g",
    "batter_count_tendency_30g",
    "batter_gb_hit_rate",
    "batter_barrel_rate_30g",
    "batter_hard_hit_rate_30g",
    "batter_sweet_spot_rate_30g",
    "batter_avg_ev_30g",
    "platoon_hr",
    "pitcher_hr_30g",
    "pitcher_entropy_30g",
    "pitcher_avg_velo_30g",
    "pitcher_avg_spin_30g",
    "pitcher_avg_extension_30g",
    "pitcher_break_total_30g",
    "batter_avg_velo_faced_30g",
    "weather_temp",
    "park_factor",
    "days_rest",
]
```

- [ ] **Step 7: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/bts/features/compute.py
git commit -m "feat: add 9 Statcast rolling features (barrel, EV, velo, spin, break)"
```

---

### Task 5: Validation — A/B backtest

**Files:**
- Create: `scripts/validate_statcast_features.py`

- [ ] **Step 1: Write validation script**

```python
"""A/B backtest: 13 baseline features vs 22 features (baseline + 9 Statcast).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_statcast_features.py
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

PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)

def walk_forward_p1(df, test_season, feature_cols, retrain_every=14):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start = df[df["season"] == test_season]["date"].min()
    train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)]
    test_data = df[(df["date"] >= test_start) & (df["season"] == test_season)]
    test_dates = sorted(test_data["date"].unique())

    model = None
    daily_hits = []
    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()
        if model is None or (i % retrain_every == 0):
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            tX = available[feature_cols]
            ty = available["is_hit"]
            valid = tX.notna().any(axis=1)
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(tX[valid], ty[valid])

        day_data["p_hit"] = model.predict_proba(day_data[feature_cols])[:, 1]
        game = day_data.groupby(["batter_id", "game_pk"]).agg(
            p_game=("p_hit", lambda x: 1 - np.prod(1 - x.values)),
            actual=("is_hit", "max"),
        ).reset_index()
        top = game.nlargest(1, "p_game").iloc[0]
        daily_hits.append(int(top["actual"]))

    p1 = np.mean(daily_hits)
    importance = dict(zip(feature_cols, model.feature_importances_))
    return p1, importance

def main():
    print("Loading data...")
    dfs = [pd.read_parquet(p) for p in sorted(Path("data/processed").glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  {len(df):,} PAs")

    print("Computing features...")
    df = compute_all_features(df)

    configs = [
        ("Baseline (13 features)", BASELINE_COLS),
        ("All Statcast (22 features)", FEATURE_COLS),
    ]

    for name, cols in configs:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        for season in [2024, 2025]:
            p1, imp = walk_forward_p1(df, season, cols)
            print(f"  {season} P@1: {p1:.1%}")
            if season == 2025:
                print(f"  Feature importance (top 10):")
                for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {k}: {v}")

    # Individual feature ablation — only if all-Statcast beats baseline
    print(f"\n{'='*50}")
    print("  Individual feature ablation")
    print(f"{'='*50}")
    new_features = [c for c in FEATURE_COLS if c not in BASELINE_COLS]
    for feat in new_features:
        cols_without = [c for c in FEATURE_COLS if c != feat]
        print(f"\n  Without {feat}:")
        for season in [2024, 2025]:
            p1, _ = walk_forward_p1(df, season, cols_without)
            print(f"    {season} P@1: {p1:.1%}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_statcast_features.py
```

Expected: Results showing P@1 for baseline vs all-Statcast on 2024 and 2025, plus per-feature ablation.

- [ ] **Step 3: Analyze results and decide**

- If all-Statcast improves P@1 on BOTH seasons: keep all, run ablation to trim noise features
- If mixed: check ablation to find which individual features help
- If worse: revert FEATURE_COLS to baseline 13

- [ ] **Step 4: Update FEATURE_COLS to final set**

Based on results, update `FEATURE_COLS` in `compute.py` to include only features that earned their spot.

- [ ] **Step 5: Commit final state**

```bash
git add src/bts/features/compute.py scripts/validate_statcast_features.py
git commit -m "feat: add validated Statcast features (N of 9 survived ablation)"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update ARCHITECTURE.md features table**

Add new features to the feature table, mark any dropped features as rejected.

- [ ] **Step 2: Update CLAUDE.md if needed**

Add any new safety rules (e.g., barrel classification formula source).

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md CLAUDE.md
git commit -m "docs: update architecture with Statcast feature results"
```
