# Context Stack Shadow Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 context features to `compute_all_features()`, wire a shadow prediction path through the Fly scheduler that runs alongside production, and provide a CLI report to compare shadow vs production picks over time.

**Architecture:** Features are always computed but only used by the shadow blend (controlled via `feature_cols_override` param on `run_pipeline`). Shadow picks saved to `{date}.shadow.json`. Production path is unchanged.

**Tech Stack:** Python 3.12, pandas, LightGBM, Click CLI

---

### Task 1: Add `CONTEXT_COLS` and 4 context features to `compute_all_features()`

**Files:**
- Modify: `src/bts/features/compute.py:390-506` (after bullpen block, before merge section)

- [ ] **Step 1: Write tests for the 4 context features**

Create `tests/test_context_features.py`:

```python
"""Tests for context_stack features added to compute_all_features."""

import numpy as np
import pandas as pd
import pytest

from bts.features.compute import compute_all_features, CONTEXT_COLS


def _make_pa_df(n=20):
    """Minimal PA DataFrame with columns needed for context features."""
    np.random.seed(42)
    dates = pd.date_range("2024-06-01", periods=5, freq="D")
    rows = []
    for i in range(n):
        d = dates[i % len(dates)]
        rows.append({
            "date": d, "season": 2024, "game_pk": 700000 + (i % 5),
            "batter_id": 100 + (i % 4), "pitcher_id": 200 + (i % 3),
            "is_hit": int(np.random.random() > 0.7), "is_home": i % 2 == 0,
            "venue_id": 1, "pitch_hand": "R",
            "final_count_balls": 2, "final_count_strikes": 1,
            "launch_speed": 90.0 + np.random.random() * 20,
            "launch_angle": 10.0 + np.random.random() * 20,
            "pitch_calls": ["B", "S", "X"], "pitch_types": ["FF", "SL"],
            "pitch_speeds": [92.0, 85.0], "pitch_spin_rates": [2200, 2500],
            "pitch_extensions": [6.2, 6.1],
            "pitch_break_vertical": [-15.0, -30.0],
            "pitch_break_horizontal": [8.0, 3.0],
            "pitch_px": [0.3, -0.8], "pitch_pz": [2.5, 3.1],
            "sz_top": 3.4, "sz_bottom": 1.6,
            "weather_temp": 72, "weather_wind_dir": "Out To CF",
            "weather_wind_speed": 12.0, "roof_type": "Open",
            "hp_umpire_id": 300 + (i % 2),
            "hardness": np.random.choice(["hard", "medium", "soft", None]),
        })
    return pd.DataFrame(rows)


class TestContextCols:
    def test_context_cols_has_4_entries(self):
        assert len(CONTEXT_COLS) == 4

    def test_context_cols_names(self):
        assert CONTEXT_COLS == [
            "ump_hr_30g", "wind_out_cf",
            "batter_hard_contact_30g", "is_indoor",
        ]


class TestUmpireHitRate:
    def test_column_present_after_compute(self):
        df = compute_all_features(_make_pa_df(50))
        assert "ump_hr_30g" in df.columns

    def test_values_between_0_and_1(self):
        df = compute_all_features(_make_pa_df(50))
        valid = df["ump_hr_30g"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0


class TestWindVector:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df())
        assert "wind_out_cf" in df.columns

    def test_out_to_cf_positive(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "Out To CF"
        df["weather_wind_speed"] = 10.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] > 0).all()

    def test_in_from_cf_negative(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "In From CF"
        df["weather_wind_speed"] = 10.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] < 0).all()

    def test_calm_is_zero(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "Calm"
        df["weather_wind_speed"] = 0.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] == 0).all()


class TestBatterHardContact:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df(50))
        assert "batter_hard_contact_30g" in df.columns

    def test_values_between_0_and_1(self):
        df = compute_all_features(_make_pa_df(50))
        valid = df["batter_hard_contact_30g"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0


class TestIsIndoor:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df())
        assert "is_indoor" in df.columns

    def test_open_is_zero(self):
        df = _make_pa_df()
        df["roof_type"] = "Open"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 0).all()

    def test_dome_is_one(self):
        df = _make_pa_df()
        df["roof_type"] = "Dome"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 1).all()

    def test_retractable_is_one(self):
        df = _make_pa_df()
        df["roof_type"] = "Retractable"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_context_features.py -v`
Expected: FAIL — `CONTEXT_COLS` not defined, columns missing from output.

- [ ] **Step 3: Add CONTEXT_COLS constant and 4 feature computations to compute.py**

In `src/bts/features/compute.py`, add after the bullpen block (after line 439 `df["opp_bullpen_hr_30g"] = np.nan`) and before the merge section (line 441 `# === Merge everything back to PA level ===`):

```python
    # --- Context features (4) — always computed, used by shadow model ---

    # Umpire hit rate: rolling 30-day hit rate per home-plate umpire
    if "hp_umpire_id" in df.columns:
        ump_daily = df.groupby(["hp_umpire_id", "date"]).agg(
            ump_hits=("is_hit", "sum"), ump_pas=("is_hit", "count"),
        ).reset_index().sort_values(["hp_umpire_id", "date"])
        ump_daily["ump_hr"] = ump_daily["ump_hits"] / ump_daily["ump_pas"]
        ump_daily["ump_hr_30g"] = (
            ump_daily.groupby("hp_umpire_id")["ump_hr"]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
        )
        df = df.merge(
            ump_daily[["hp_umpire_id", "date", "ump_hr_30g"]]
            .drop_duplicates(subset=["hp_umpire_id", "date"]),
            on=["hp_umpire_id", "date"], how="left",
        )
    else:
        df["ump_hr_30g"] = np.nan

    # Wind vector: signed scalar (positive = blowing out to CF, helps hitters)
    if "weather_wind_dir" in df.columns and "weather_wind_speed" in df.columns:
        direction = df["weather_wind_dir"].astype(str).str.lower()
        speed = pd.to_numeric(df["weather_wind_speed"], errors="coerce").fillna(0)
        direction_score = np.where(
            direction.str.contains("out to cf|out to center"), 1.0,
            np.where(
                direction.str.contains("in from cf|in from center"), -1.0,
                np.where(
                    direction.str.contains("out to lf|out to l f|out to rf|out to r f"), 0.5,
                    np.where(
                        direction.str.contains("in from lf|in from rf"), -0.5,
                        0.0,
                    ),
                ),
            ),
        )
        df["wind_out_cf"] = direction_score * speed
    else:
        df["wind_out_cf"] = np.nan

    # Batter hard-contact rate: rolling 30-day from categorical hardness column
    if "hardness" in df.columns:
        is_hard = (df["hardness"].astype(str).str.lower() == "hard").astype(float)
        is_hard = is_hard.where(df["hardness"].notna(), np.nan)
        df["_is_hard"] = is_hard
        df = df.sort_values(["batter_id", "date"])
        df["batter_hard_contact_30g"] = (
            df.groupby("batter_id")["_is_hard"]
            .rolling(window=120, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
        df.drop(columns=["_is_hard"], inplace=True)
    else:
        df["batter_hard_contact_30g"] = np.nan

    # Indoor flag: binary for dome/closed/retractable roofs
    if "roof_type" in df.columns:
        rt = df["roof_type"].astype(str).str.lower()
        df["is_indoor"] = rt.isin(["dome", "closed", "retractable"]).astype(int)
    else:
        df["is_indoor"] = 0
```

Then add the `CONTEXT_COLS` constant after `FEATURE_COLS` (after line 528):

```python
# Context features (4) — computed alongside baseline features but only used
# by the shadow model (via feature_cols_override). Graduates to FEATURE_COLS
# after 30-day shadow validation.
CONTEXT_COLS = [
    "ump_hr_30g",
    "wind_out_cf",
    "batter_hard_contact_30g",
    "is_indoor",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_context_features.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ --tb=short -q`
Expected: All 437+ tests pass (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/bts/features/compute.py tests/test_context_features.py
git commit -m "feat: add 4 context features (ump_hr, wind, hardness, indoor) to compute_all_features

Always computed but only used by shadow model via CONTEXT_COLS.
Ported from experiment/features.py implementations."
```

---

### Task 2: Add `feature_cols_override` to `run_pipeline()` and `train_blend()`

**Files:**
- Modify: `src/bts/model/predict.py:12,18-31,59-75,615-674`

- [ ] **Step 1: Write tests for feature_cols_override**

Create `tests/test_shadow_pipeline.py`:

```python
"""Tests for shadow model pipeline (feature_cols_override)."""

import pytest
from unittest.mock import patch, MagicMock

from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
from bts.model.predict import _build_blend_configs


class TestBuildBlendConfigs:
    def test_default_uses_feature_cols(self):
        configs = _build_blend_configs()
        base_name, base_cols = configs[0]
        assert base_name == "baseline"
        assert base_cols == FEATURE_COLS

    def test_override_replaces_base(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        base_name, base_cols = configs[0]
        assert base_name == "baseline"
        assert base_cols == override

    def test_override_preserves_statcast_boltons(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        barrel_name, barrel_cols = configs[1]
        assert barrel_name == "barrel"
        assert "batter_barrel_rate_30g" in barrel_cols
        # base features are the override, not FEATURE_COLS
        for col in CONTEXT_COLS:
            assert col in barrel_cols

    def test_override_keeps_12_configs(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        assert len(configs) == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_pipeline.py -v`
Expected: FAIL — `_build_blend_configs` not defined

- [ ] **Step 3: Refactor BLEND_CONFIGS into a function and add feature_cols_override**

In `src/bts/model/predict.py`, replace the module-level `BLEND_CONFIGS` list (lines 18-31) with a function:

```python
def _build_blend_configs(base_feature_cols: list[str] | None = None) -> list[tuple[str, list[str]]]:
    """Build the 12-model blend config list.

    When base_feature_cols is None, uses FEATURE_COLS (production).
    When provided (e.g., FEATURE_COLS + CONTEXT_COLS), substitutes
    the base for shadow model training.
    """
    base = base_feature_cols or FEATURE_COLS
    return [
        ("baseline", base),
        ("barrel", base + ["batter_barrel_rate_30g"]),
        ("hard_hit", base + ["batter_hard_hit_rate_30g"]),
        ("sweet_spot", base + ["batter_sweet_spot_rate_30g"]),
        ("avg_ev", base + ["batter_avg_ev_30g"]),
        ("velo", base + ["pitcher_avg_velo_30g"]),
        ("spin", base + ["pitcher_avg_spin_30g"]),
        ("extension", base + ["pitcher_avg_extension_30g"]),
        ("break", base + ["pitcher_break_total_30g"]),
        ("velo_faced", base + ["batter_avg_velo_faced_30g"]),
        ("best_two", base + ["batter_sweet_spot_rate_30g", "pitcher_avg_extension_30g"]),
        ("all_statcast", base + STATCAST_COLS),
    ]


# Module-level constant for backward compat (used by train_model, coercion loop)
BLEND_CONFIGS = _build_blend_configs()
```

Update `train_blend()` (line 59) to accept an optional `base_feature_cols`:

```python
def train_blend(df: pd.DataFrame, base_feature_cols: list[str] | None = None) -> dict:
    """Train 12-model blend on historical PA data (2019+).

    Returns dict of {name: (model, feature_cols)} for each blend variant.
    When base_feature_cols is provided, uses it instead of FEATURE_COLS
    as the base for all blend configs (shadow model).
    """
    configs = _build_blend_configs(base_feature_cols)
    train = df[df["season"] >= TRAIN_START_YEAR]
    train_y = train["is_hit"]
    blend = {}

    for name, cols in configs:
        train_X = train[cols]
        mask = train_X.notna().any(axis=1)
        model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
        model.fit(train_X[mask], train_y[mask])
        blend[name] = (model, cols)

    return blend
```

Update `run_pipeline()` signature (line 615) to add `feature_cols_override`:

```python
def run_pipeline(
    date: str,
    data_dir: str = "data/processed",
    check_openers: bool = True,
    cached_blend: dict | None = None,
    save_blend_path=None,
    refresh_data: bool = True,
    feature_cols_override: list[str] | None = None,
) -> pd.DataFrame:
```

And update the coercion block (line 654) and training call (line 672) inside `run_pipeline()`:

```python
    # Coercion: cover base + statcast + any override cols
    base_cols = feature_cols_override or FEATURE_COLS
    all_feature_cols = set(base_cols) | set(STATCAST_COLS)
    # ... rest of coercion unchanged ...

    if cached_blend:
        model = cached_blend.pop("_model")
        blend = cached_blend
    else:
        model = train_model(df)
        blend = train_blend(df, base_feature_cols=feature_cols_override)
        # ... rest unchanged ...
```

Also update the single-model prediction block later in `run_pipeline()` where it uses `FEATURE_COLS` directly for the single-model fallback prediction (around line 514 in the predict section):

```python
    feat_df = pred_df[base_cols]  # was: pred_df[FEATURE_COLS]
```

- [ ] **Step 4: Update the import in compute.py**

In `src/bts/model/predict.py` line 12, add `CONTEXT_COLS` to the import:

```python
from bts.features.compute import compute_all_features, FEATURE_COLS, CONTEXT_COLS, STATCAST_COLS, TRAIN_START_YEAR
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_pipeline.py tests/ --tb=short -q`
Expected: All pass (new + existing)

- [ ] **Step 6: Commit**

```bash
git add src/bts/model/predict.py tests/test_shadow_pipeline.py
git commit -m "feat: add feature_cols_override to run_pipeline for shadow model

Refactors BLEND_CONFIGS into _build_blend_configs() function that
accepts optional base_feature_cols. When provided, all 12 blend
variants use the extended feature set. Production path unchanged."
```

---

### Task 3: Add `save_shadow_pick()` to picks.py

**Files:**
- Modify: `src/bts/picks.py:61-66`

- [ ] **Step 1: Write tests for shadow pick save/load**

Add to `tests/test_picks.py` (or create `tests/test_shadow_picks.py`):

```python
"""Tests for shadow pick save/load."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from bts.picks import DailyPick, Pick, save_shadow_pick, load_shadow_pick


def _make_daily(date="2026-04-10"):
    pick = Pick(
        batter_name="Luis Arraez", batter_id=650333, team="SF",
        lineup_position=2, pitcher_name="Shane Baz", pitcher_id=669358,
        p_game_hit=0.767, flags=[], projected_lineup=False,
        game_pk=824858, game_time="2026-04-10T23:15:00Z", pitcher_team="BAL",
    )
    return DailyPick(date=date, run_time="2026-04-10T22:32:41Z", pick=pick)


class TestSaveShadowPick:
    def test_saves_to_shadow_json(self, tmp_path):
        daily = _make_daily()
        path = save_shadow_pick(daily, tmp_path)
        assert path == tmp_path / "2026-04-10.shadow.json"
        assert path.exists()

    def test_content_matches_daily(self, tmp_path):
        daily = _make_daily()
        save_shadow_pick(daily, tmp_path)
        data = json.loads((tmp_path / "2026-04-10.shadow.json").read_text())
        assert data["pick"]["batter_name"] == "Luis Arraez"
        assert data["pick"]["p_game_hit"] == pytest.approx(0.767, abs=0.001)


class TestLoadShadowPick:
    def test_load_existing(self, tmp_path):
        daily = _make_daily()
        save_shadow_pick(daily, tmp_path)
        loaded = load_shadow_pick("2026-04-10", tmp_path)
        assert loaded is not None
        assert loaded.pick.batter_name == "Luis Arraez"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_shadow_pick("2026-04-10", tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_picks.py -v`
Expected: FAIL — `save_shadow_pick` not defined

- [ ] **Step 3: Implement save_shadow_pick and load_shadow_pick**

Add to `src/bts/picks.py` after `save_pick()` (after line 66):

```python
def save_shadow_pick(daily: DailyPick, picks_dir: Path) -> Path:
    """Save shadow model pick to {date}.shadow.json."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / f"{daily.date}.shadow.json"
    path.write_text(json.dumps(asdict(daily), indent=2))
    return path


def load_shadow_pick(date: str, picks_dir: Path) -> DailyPick | None:
    """Load shadow model pick. Returns None if not found."""
    path = picks_dir / f"{date}.shadow.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["pick"].setdefault("pitcher_team", None)
    pick = Pick(**data["pick"])
    dd = Pick(**data["double_down"]) if data.get("double_down") else None
    return DailyPick(
        date=data["date"], run_time=data["run_time"], pick=pick,
        double_down=dd, runner_up=data.get("runner_up"),
        bluesky_posted=False, bluesky_uri=None, result=data.get("result"),
    )
```

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_picks.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/bts/picks.py tests/test_shadow_picks.py
git commit -m "feat: add save_shadow_pick/load_shadow_pick for context_stack shadow model"
```

---

### Task 4: Wire shadow prediction into scheduler

**Files:**
- Modify: `src/bts/scheduler.py:580-645`
- Modify: `src/bts/orchestrator.py:76-106`

- [ ] **Step 1: Write test for shadow prediction in scheduler**

Add to `tests/test_scheduler.py` (find existing scheduler test file):

```python
"""Test shadow model integration in scheduler."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from bts.scheduler import _run_shadow_prediction


class TestRunShadowPrediction:
    def test_saves_shadow_pick(self, tmp_path):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.date = "2026-04-10"
        mock_result.daily.pick.batter_name = "Luis Arraez"
        mock_result.daily.pick.p_game_hit = 0.767

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick") as mock_save:
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
            mock_save.assert_called_once()

    def test_logs_agreement(self, tmp_path, capsys):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.pick.batter_name = "Luis Arraez"
        mock_result.daily.pick.team = "SF"
        mock_result.daily.pick.p_game_hit = 0.767

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick"):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
        captured = capsys.readouterr()
        assert "AGREES" in captured.err

    def test_logs_disagreement(self, tmp_path, capsys):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.pick.batter_name = "Steven Kwan"
        mock_result.daily.pick.team = "CLE"
        mock_result.daily.pick.p_game_hit = 0.720

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick"):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
        captured = capsys.readouterr()
        assert "DISAGREES" in captured.err

    def test_failure_does_not_raise(self, tmp_path):
        with patch("bts.scheduler.predict_local_shadow", side_effect=RuntimeError("boom")):
            # Should not raise — shadow failures are logged, not propagated
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py::TestRunShadowPrediction -v`
Expected: FAIL — `_run_shadow_prediction` not defined

- [ ] **Step 3: Add predict_local_shadow to orchestrator.py**

Add to `src/bts/orchestrator.py` after `predict_local()` (after line 106):

```python
def predict_local_shadow(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
) -> pd.DataFrame | None:
    """Run shadow predictions locally with context_stack features.

    Same as predict_local but uses FEATURE_COLS + CONTEXT_COLS.
    Gets its own model cache (blend_{date}_shadow.pkl).
    """
    from bts.model.predict import run_pipeline, load_blend, save_blend
    from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
    from pathlib import Path

    shadow_cols = FEATURE_COLS + CONTEXT_COLS
    models_path = Path(models_dir)
    cache_path = models_path / f"blend_{date}_shadow.pkl"
    cached_blend = None
    if cache_path.exists():
        print(f"  [shadow] Loading cached shadow model from {cache_path}", file=sys.stderr)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
            refresh_data=False,  # data already refreshed by production run
            feature_cols_override=shadow_cols,
        )
        return predictions
    except Exception as e:
        print(f"  [shadow] Shadow prediction failed: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 4: Add _run_shadow_prediction to scheduler.py**

Add to `src/bts/scheduler.py` after the imports section:

```python
def _run_shadow_prediction(config: dict, date: str, production_pick_name: str) -> None:
    """Run shadow model prediction and save result. Never raises."""
    from bts.orchestrator import predict_local_shadow
    from bts.picks import save_shadow_pick, load_streak
    from bts.strategy import select_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    try:
        predictions = predict_local_shadow(date)
        if predictions is None or predictions.empty:
            print("  [SHADOW MODEL] No predictions returned.", file=sys.stderr)
            return

        streak = load_streak(picks_dir)
        result = select_pick(predictions, date, picks_dir, streak=streak)
        if result is None or result.daily is None:
            print("  [SHADOW MODEL] Skip (below threshold).", file=sys.stderr)
            return

        save_shadow_pick(result.daily, picks_dir)
        shadow_name = result.daily.pick.batter_name
        shadow_team = result.daily.pick.team
        shadow_p = result.daily.pick.p_game_hit
        agreed = shadow_name == production_pick_name
        tag = "AGREES" if agreed else f"DISAGREES (prod: {production_pick_name})"
        print(f"  [SHADOW MODEL] {shadow_name} ({shadow_team}) "
              f"{shadow_p:.1%} — {tag}", file=sys.stderr)
    except Exception as e:
        print(f"  [SHADOW MODEL] Failed: {e}", file=sys.stderr)
```

- [ ] **Step 5: Wire into the scheduler main loop**

In `src/bts/scheduler.py`, in the main scheduler loop, read the `shadow_model` config (near line 505 where `shadow_mode` is read):

```python
    shadow_model_enabled = sched_config.get("shadow_model", False)
    if shadow_model_enabled:
        print("  [SHADOW MODEL] Context stack shadow model enabled.", file=sys.stderr)
```

Then after a production pick is saved (after line 621, inside the `if result["should_post"]` block, after either shadow-mode save or Bluesky post), add:

```python
        # Run shadow model if enabled (after production pick is resolved)
        if shadow_model_enabled and result.get("pick_result") and result["pick_result"].daily:
            prod_name = result["pick_result"].daily.pick.batter_name
            _run_shadow_prediction(config, date, prod_name)
```

- [ ] **Step 6: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler.py tests/ --tb=short -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/bts/scheduler.py src/bts/orchestrator.py tests/test_scheduler.py
git commit -m "feat: wire shadow model into scheduler

Runs context_stack prediction after production pick is saved.
Controlled by shadow_model=true in [scheduler] config.
Logs AGREES/DISAGREES, saves to {date}.shadow.json."
```

---

### Task 5: Add `bts shadow-report` CLI command

**Files:**
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Write test for shadow-report**

Create `tests/test_shadow_report.py`:

```python
"""Tests for bts shadow-report CLI command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bts.cli import cli


def _write_pick_pair(picks_dir: Path, date: str,
                     prod_name: str, prod_p: float, prod_result: str | None,
                     shadow_name: str, shadow_p: float, shadow_result: str | None):
    """Write matching production + shadow pick files."""
    for suffix, name, p, result in [
        (".json", prod_name, prod_p, prod_result),
        (".shadow.json", shadow_name, shadow_p, shadow_result),
    ]:
        data = {
            "date": date,
            "run_time": f"{date}T22:00:00Z",
            "pick": {
                "batter_name": name, "batter_id": 100, "team": "SF",
                "lineup_position": 1, "pitcher_name": "Pitcher", "pitcher_id": 200,
                "p_game_hit": p, "flags": [], "projected_lineup": False,
                "game_pk": 800000, "game_time": f"{date}T23:00:00Z", "pitcher_team": "BAL",
            },
            "double_down": None,
            "runner_up": None,
            "bluesky_posted": False,
            "bluesky_uri": None,
            "result": result,
        }
        (picks_dir / f"{date}{suffix}").write_text(json.dumps(data))


class TestShadowReport:
    def test_no_shadow_picks(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No shadow pick pairs found" in result.output

    def test_agreement_rate(self, tmp_path):
        _write_pick_pair(tmp_path, "2026-04-01", "Arraez", 0.77, "hit", "Arraez", 0.76, None)
        _write_pick_pair(tmp_path, "2026-04-02", "Kwan", 0.72, "hit", "Marte", 0.73, None)
        _write_pick_pair(tmp_path, "2026-04-03", "Arraez", 0.75, "miss", "Arraez", 0.74, None)
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Agreement" in result.output
        assert "66.7%" in result.output  # 2 of 3 agree

    def test_shows_disagreement_detail(self, tmp_path):
        _write_pick_pair(tmp_path, "2026-04-01", "Arraez", 0.77, "hit", "Marte", 0.76, None)
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert "Arraez" in result.output
        assert "Marte" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_report.py -v`
Expected: FAIL — `shadow-report` command not found

- [ ] **Step 3: Implement shadow-report CLI command**

Add to `src/bts/cli.py`:

```python
@cli.command(name="shadow-report")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
def shadow_report(picks_dir: str):
    """Compare shadow model picks against production picks.

    Reads {date}.json and {date}.shadow.json pairs from the picks directory.
    Reports agreement rate, disagreement details, and P@1 for both models.
    """
    import json as _json
    from pathlib import Path

    picks_path = Path(picks_dir)
    shadow_files = sorted(picks_path.glob("*.shadow.json"))

    if not shadow_files:
        click.echo("No shadow pick pairs found.")
        return

    pairs = []
    for sf in shadow_files:
        date = sf.name.replace(".shadow.json", "")
        prod_file = picks_path / f"{date}.json"
        if not prod_file.exists():
            continue
        prod = _json.loads(prod_file.read_text())
        shadow = _json.loads(sf.read_text())
        pairs.append((date, prod, shadow))

    if not pairs:
        click.echo("No shadow pick pairs found (shadow files exist but no matching production files).")
        return

    agrees = 0
    disagrees = []
    prod_hits = 0
    shadow_hits = 0
    resolved = 0

    for date, prod, shadow in pairs:
        prod_name = prod["pick"]["batter_name"]
        shadow_name = shadow["pick"]["batter_name"]
        prod_result = prod.get("result")

        if prod_name == shadow_name:
            agrees += 1
        else:
            disagrees.append((date, prod_name, prod.get("pick", {}).get("p_game_hit"),
                              shadow_name, shadow.get("pick", {}).get("p_game_hit"),
                              prod_result))

        if prod_result in ("hit", "miss"):
            resolved += 1
            if prod_result == "hit":
                prod_hits += 1
            # For shadow: check if the shadow pick actually hit
            # (we'd need to look up the shadow batter's result, but
            # we only track production result in the pick file).
            # For now, if they agreed, same result. If not, unknown.
            if prod_name == shadow_name:
                if prod_result == "hit":
                    shadow_hits += 1

    total = len(pairs)
    pct = agrees / total * 100

    click.echo(f"Shadow Model Report ({total} days, {30 - total} remaining to threshold)")
    click.echo(f"{'='*60}")
    click.echo(f"Agreement rate: {agrees}/{total} ({pct:.1f}%)")
    if resolved > 0:
        click.echo(f"Production P@1: {prod_hits}/{resolved} ({prod_hits/resolved*100:.1f}%)")
    click.echo()

    if disagrees:
        click.echo(f"Disagreements ({len(disagrees)} days):")
        click.echo(f"{'Date':<12} {'Production':<20} {'Shadow':<20} {'Result'}")
        click.echo(f"{'-'*12} {'-'*20} {'-'*20} {'-'*8}")
        for date, pn, pp, sn, sp, res in disagrees:
            pp_str = f"{pp:.1%}" if pp else "?"
            sp_str = f"{sp:.1%}" if sp else "?"
            res_str = res or "pending"
            click.echo(f"{date:<12} {pn:<15} {pp_str:<4}  {sn:<15} {sp_str:<4}  {res_str}")
```

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_report.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ --tb=short -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/bts/cli.py tests/test_shadow_report.py
git commit -m "feat: add bts shadow-report command for context_stack comparison"
```

---

### Task 6: Enable shadow_model on Fly and deploy

**Files:**
- No code changes — config + deploy

- [ ] **Step 1: Update orchestrator.toml on Fly volume**

```bash
fly ssh console -a bts-mlb -C "/bin/sh -c 'grep -q shadow_model /data/orchestrator.toml || sed -i \"/shadow_mode/a shadow_model = true\" /data/orchestrator.toml && grep shadow /data/orchestrator.toml'"
```

Expected output includes both `shadow_mode = true` and `shadow_model = true`.

- [ ] **Step 2: Run full test suite one final time**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ --tb=short -q`
Expected: All pass

- [ ] **Step 3: Push and deploy**

```bash
git push origin main
fly deploy -a bts-mlb --remote-only
```

- [ ] **Step 4: Verify shadow model runs on Fly**

After deploy, wait for the scheduler to run predictions, then check logs:

```bash
fly logs -a bts-mlb --no-tail | grep "SHADOW MODEL"
```

Expected: `[SHADOW MODEL] <batter> (<team>) <pct> — AGREES` or `DISAGREES`

Also verify the shadow pick file exists:

```bash
fly ssh console -a bts-mlb -C "ls /data/picks/*.shadow.json"
```

- [ ] **Step 5: Commit any final fixes, update memory**

Update `project_bts.md` memory with shadow model status.
