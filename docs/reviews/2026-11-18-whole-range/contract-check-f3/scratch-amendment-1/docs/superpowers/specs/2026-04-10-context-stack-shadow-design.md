# Context Stack Shadow Model — Design Spec

## Overview

A shadow prediction model that adds 4 context features to the existing 15-feature baseline. The shadow model runs daily alongside production on Fly, saves its pick to a separate file, but does NOT post to Bluesky or affect the live streak. After 30 days, compare picks to decide whether to promote to production.

## The 4 Context Features

All derived from columns already in PA parquet (no new data sources):

| Feature | Source Column(s) | Type | Computation |
|---|---|---|---|
| `ump_hr_30g` | `hp_umpire_id` (100%) | Rolling | 30-day hit rate per HP umpire, shift(1) |
| `wind_out_cf` | `weather_wind_dir`, `weather_wind_speed` (100%) | Per-game | Signed wind vector: direction score (-1 to +1) x speed |
| `batter_hard_contact_30g` | `hardness` (68%) | Rolling | 30-day hard-contact rate per batter, 120-PA window, min_periods=10, shift(1) |
| `is_indoor` | `roof_type` (100%) | Per-game | Binary: 1 if dome/closed/retractable |

## Backtest Results

Tested on 2023+2024+2025 (551 days, walk-forward with 7-day retraining):

| Metric | Baseline | Context Stack | Delta |
|---|---|---|---|
| P@1 2023 | 88.5% | 85.2% | -3.3pp |
| P@1 2024 | 84.9% | 81.6% | -3.2pp |
| P@1 2025 | 85.9% | 88.6% | +2.7pp |
| Mean max streak | 29.16 | 31.11 | +1.95 |
| P99 max streak | 56 | 62 | +6 |
| Exact P(57) | 1.63% | 1.83% | +0.20pp |

Lag pattern: context features hurt when training data is immature (2023, 2024) but help once the model has 2+ years of context patterns (2025). With 3 years of training patterns for 2026, the effect should be stronger. Shadow mode validates this hypothesis on live data.

## Architecture

### Feature Integration

Add the 4 features to `compute_all_features()` in `src/bts/features/compute.py`, after the bullpen feature block. Port from the existing experiment implementations in `src/bts/experiment/features.py`:

- `UmpireHitRateExperiment.modify_features()` -> umpire rolling rate
- `WindVectorExperiment.modify_features()` -> vectorized wind parsing
- `BatterHardnessRateExperiment.modify_features()` -> hardness rolling rate
- `RoofTypeExperiment.modify_features()` -> roof binary flag

New constant in compute.py:

```python
CONTEXT_COLS = [
    "ump_hr_30g",
    "wind_out_cf",
    "batter_hard_contact_30g",
    "is_indoor",
]
```

`FEATURE_COLS` stays at 15. The shadow blend uses `FEATURE_COLS + CONTEXT_COLS` (19 features). Features are always computed for all PAs (cost is ~4 seconds of additional groupby/vectorized work) but only used by the shadow blend.

### Shadow Prediction Path

`run_pipeline()` gains an optional parameter:

```python
def run_pipeline(
    date, data_dir="data/processed", ...,
    feature_cols_override: list[str] | None = None,
) -> pd.DataFrame:
```

When `None` (default), trains the 12-model blend on `FEATURE_COLS` — production behavior unchanged. When set to `FEATURE_COLS + CONTEXT_COLS`, the blend trains on the 19-feature set.

When `feature_cols_override` is provided, `train_blend()` substitutes it for `FEATURE_COLS` in each `BLEND_CONFIGS` entry. For example, `("baseline", FEATURE_COLS)` becomes `("baseline", feature_cols_override)`, and `("barrel", FEATURE_COLS + ["batter_barrel_rate_30g"])` becomes `("barrel", feature_cols_override + ["batter_barrel_rate_30g"])`. The Statcast bolt-on pattern stays the same — only the base feature set changes.

Shadow model gets its own cache key (`blend_{date}_shadow.pkl`) to avoid colliding with the production cache.

### Scheduler Integration

The scheduler calls shadow prediction inside `run_single_check`, after the production prediction succeeds. Controlled by `shadow_model = true` in `[scheduler]` section of orchestrator.toml (defaults to `false`).

```
lineup check fires
  +-- run_pipeline(date)                              -> production predictions
  |   +-- select_pick() -> save to {date}.json         -> pick/post decision
  |
  +-- run_pipeline(date, feature_cols_override=...)   -> shadow predictions
      +-- save to {date}.shadow.json                   -> comparison only
```

If the shadow prediction fails, it logs the error and continues. Never blocks production.

The shadow run reuses the same refreshed data (no second API pull) and the same parquets. The shadow result is logged:

```
[SHADOW MODEL] Luis Arraez (SF) 76.7% — AGREES
[SHADOW MODEL] Steven Kwan (CLE) 72.0% — DISAGREES (prod: Ketel Marte)
```

### Shadow Pick File

`{picks_dir}/{date}.shadow.json` — same schema as the production pick file (`pick`, `double_down`, `runner_up`, `result`). This makes comparison trivial.

### Comparison CLI

New `bts shadow-report` command that reads all `*.shadow.json` / `*.json` pairs from the picks directory and prints:

- Agreement rate: how often #1 pick matches
- Disagreement days: which pick each model chose + which actually hit
- Running P@1 for both models
- Days analyzed / days remaining until 30-day threshold

No automated switching.

### Performance

First daily prediction doubles from ~15 min to ~25-30 min (two 12-model blends to train on 2 vCPUs). Subsequent same-day checks use cached blends and run in ~5 min total. The 15-minute health check grace period already covers this.

## Files Changed

| File | Change |
|---|---|
| `src/bts/features/compute.py` | Add 4 context features + `CONTEXT_COLS` constant |
| `src/bts/model/predict.py` | Add `feature_cols_override` param to `run_pipeline()`, dynamic `BLEND_CONFIGS` |
| `src/bts/scheduler.py` | Shadow prediction call after production, `shadow_model` config |
| `src/bts/picks.py` | `save_shadow_pick()` function (writes `{date}.shadow.json`) |
| `src/bts/cli.py` | New `bts shadow-report` command |
| Tests | Feature computation tests, shadow pick save/load, shadow-report output |

## Success Criteria for Promotion

After 30 days of shadow data:

1. **Agreement rate > 80%** — context model's #1 matches baseline's #1 most of the time
2. **Context-specific wins** — on disagreement days, context model's pick hits more often
3. **No regressions** — shadow P@1 is not worse than production P@1

If all met, promote by moving `CONTEXT_COLS` into `FEATURE_COLS` and removing the shadow path.

## Known Risks

- LightGBM nondeterminism: single-seed runs have ~2pp variance. The +2.7pp 2025 improvement is within noise for individual runs, but the consistent directional signal across 4 independent features strengthens confidence.
- Doubling prediction time on first daily run. Mitigated by model caching on subsequent checks.
