# BTS Conformal Lower Bounds — Design Spec

**Date**: 2026-05-01 (PM session)
**Status**: Approved (brainstorm) → pending implementation plan
**Author**: Eric Stone (with Claude assist)
**Source brainstorm**: 2026-05-01 evening session

## 1. Motivation

Today's investigation produced two findings that motivate this work:

1. **The current model is consistently UNDER-confident at high predicted-p**, with the gap widening 2024 → 2025 → 2026 (chi-square p=0.011 across the [0.75, 0.80) bucket; -6.6pp / -4.4pp / -10.9pp respectively). See `project_bts_strategic_gaps_2026_04_30.md` gap #1 (reframed) and `project_bts_skill_vs_park_calibration_2026_05_01.md`.

2. **Park × position dominates batter-skill in driving predictions, faithfully** (per sensitivity analysis on Beck's row: park_factor contributes 4× more than all batter-skill features combined, AND the resulting calibration is honest in aggregate). The aesthetic concern about high-park-low-skill picks does not predict realized rate.

The legitimate path forward for **variance-aware picks at high streak** is conformal lower bounds (Tier 5.3 of the feature queue), NOT per-PA model retraining or feature reweighting. This spec designs the v1 implementation.

## 2. Goals & non-goals

**Goals**:
- Produce per-pick lower bounds on P(hit) at three coverage levels (95% / 90% / 80%)
- Use state-of-the-art weighted conformal prediction with covariate-shift correction (Tibshirani et al. 2019), addressing the documented 2024→2026 distribution shift
- Store bounds alongside picks for future MDP integration (deferred to v1.5)
- Validate before shipping using K-fold cross-validation with proper time-series embargo

**Non-goals (v1)**:
- v1 does NOT change MDP behavior — the bounds are stored, not consumed by decisions
- v1 does NOT implement online streaming conformal updates
- v1 does NOT include MDP integration design (deferred to its own brainstorm at v1.5)
- v1 does NOT extend to multi-class predictions (we're binary)

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────────┐
│  src/bts/model/conformal.py  (NEW)                               │
│                                                                    │
│  WeightedMondrianConformalCalibrator                             │
│    fit_calibrator(calibration_pairs, alphas, lr_classifier)      │
│    apply(predicted_p) → lower_bound                              │
│  BucketWilsonCalibrator                                          │
│    fit_calibrator(calibration_pairs, alphas)                     │
│    apply(predicted_p) → lower_bound                              │
│  fit_lr_classifier(cal_features, target_features) → LightGBM    │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/bts/orchestrator.py  (MODIFY — predict_local extension)     │
│                                                                    │
│  After run_pipeline produces predictions:                        │
│    if BTS_USE_CONFORMAL=1 (default ON for v1):                   │
│      load latest calibrator from data/conformal/                 │
│      compute 6 lower bounds per row × per alpha                  │
│      attach as p_game_hit_lower_{conformal,wilson}_{95,90,80}    │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/bts/picks.py  (MODIFY — Pick dataclass extension)           │
│                                                                    │
│  6 new fields, all Optional[float] = None                        │
│  (preserves backward-compat with old pick files)                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  scripts/refit_conformal_calibrator.py  (NEW — daily cron)       │
│                                                                    │
│  Fits LR classifier + both calibrators                           │
│  Persists to data/conformal/{calibrator,wilson_calibrator,       │
│       lr_classifier}_{YYYY-MM-DD}.pkl                            │
│  Logs validation metrics to validation_log.jsonl                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  scripts/validate_conformal.py  (NEW — validation gate)          │
│                                                                    │
│  K-fold cross-validation with 7-day embargo across 2025+2026     │
│  Computes 3-gate decision matrix (marginal × bucketed × tight-   │
│       ness) per (method × α). Per-cell SHIP/NO-SHIP independently.│
│  Output: data/validation/conformal_validation_{date}.json        │
└──────────────────────────────────────────────────────────────────┘
```

**Architectural commitments**:
- **Decoupled module** — `bts.model.conformal` is pure: takes pairs, returns calibrators. No I/O, no model awareness, no scheduler awareness.
- **Default ON for v1** because it's a passive feature column (no behavior change). v1.5 will gate MDP integration on a separate flag `BTS_USE_CONFORMAL_MDP=1`.
- **Three alphas computed** (0.05 / 0.10 / 0.20) — costs ~milliseconds extra and lets v1.5 choose streak-dependent thresholds without re-deploy.
- **Both methods stored** (weighted Mondrian conformal AND bucket Wilson) — Wilson serves as sanity check; conformal is the primary technique.
- **Daily refresh** of calibrators by `scripts/refit_conformal_calibrator.py`, invoked from the same systemd path that runs the daily blend retrain. The script: (a) loads latest 2025+2026 backtest profiles, (b) fits LR classifier, (c) fits both calibrators, (d) serializes dated artifacts to `data/conformal/` (using project-standard joblib for LightGBM models, matching existing `data/models/blend_*.pkl` pattern), (e) appends a row to `validation_log.jsonl`. ~1-3 minute fit cost.
- **Validation log** is a longitudinal append-only JSONL for drift detection.

## 4. Calibration procedure (state-of-the-art)

The state-of-the-art for our specific problem (binary classification, covariate shift between calibration and test data, distribution-free coverage on the latent probability) is **Weighted Conformal Prediction** (Tibshirani et al. 2019, "Conformal Prediction Under Covariate Shift") with class-conditional Mondrian binning and a likelihood-ratio-reweighted quantile.

This addresses what naive split conformal misses: today's chi-square showed 2026 differs significantly from 2024+2025 (p=0.011). Naive split conformal assumes calibration and test data are exchangeable — they aren't. Weighted conformal corrects via reweighting.

### Algorithm

**Step 1 — Estimate likelihood ratio** (covariate-shift correction)

Train a LightGBM binary classifier `ŵ(x)` predicting `P(year = 2026 | x)` using the same 16 FEATURE_COLS. Calibration rows span 2025+2026 backtest (~3,650 rows). The classifier output gives:
```
LR(x_i) = ŵ(x_i) / (1 − ŵ(x_i))     # density ratio
```
This is the standard density-ratio estimation trick (Sugiyama et al. 2012). For each calibration row, `w_i = LR(x_i)` reweights it toward the test distribution.

**Step 2 — Non-conformity scores**

```
s_i = predicted_p_i − actual_hit_i           (signed residual; Tibshirani et al. 2019 standard)
```

**Step 3 — Mondrian bin-conditional weighted quantiles**

Partition by predicted-p bucket `B_k = [0.025·k, 0.025·(k+1))`. For each bucket with effective sample size `n_k_eff = ∑_{i∈B_k} w_i ≥ 50`:
```
q_k(α) = smallest s_(j) such that
    (∑_{i: s_i ≤ s_(j), i∈B_k} w_i) / (∑_{i∈B_k} w_i) ≥ ⌈(n_k + 1)(1 − α)⌉ / (n_k + 1)
```
The `⌈(n+1)(1−α)⌉/(n+1)` is the standard finite-sample correction from split conformal.

**Step 4 — Per-prediction lower bound**

For new prediction `p_new` in bucket `B_k`:
```
L_conformal(p_new, α) = max(0, min(p_new, p_new − q_k(α)))
```
Clamped to `[0, p_new]` (lower bound can't exceed point estimate or go negative).

For sparse buckets (n_k_eff < 50), fall back to marginal weighted quantile across all calibration data.

**Step 5 — Bucket Wilson sanity check (parallel)**

For each bucket `B_k` with n_k ≥ 30, compute one-sided Wilson lower bound on the realized hit rate:
```
L_wilson(B_k, α) = Wilson_one_sided_lower(hits_k, n_k, α)
```
Stored independently. Serves as sanity check: weighted-conformal bounds should track Wilson bounds when LR weights are near-uniform. Divergence signals issues with the LR step.

### Calibrator objects

```python
@dataclass
class WeightedMondrianConformalCalibrator:
    alphas: list[float]                            # [0.05, 0.10, 0.20]
    bucket_quantiles: dict[float, list[float]]     # {bucket_low: [q_05, q_10, q_20]}
    marginal_quantiles: list[float]                # fallback for sparse buckets
    bucket_width: float                            # 0.025
    n_calibration: int
    n_effective_per_bucket: dict[float, float]     # for diagnostics
    lr_classifier: object                          # serialized LightGBM model
    lr_weights_summary: dict                       # mean/median/IQR for sanity

@dataclass
class BucketWilsonCalibrator:
    bucket_lower: dict[float, list[float]]         # {bucket_low: [wilson_05, wilson_10, wilson_20]}
    bucket_n: dict[float, int]
    bucket_hit_rate: dict[float, float]
```

### Why this is state-of-the-art

| Naive split conformal | Weighted Mondrian conformal (this spec) |
|----------------------|----------------------------------------|
| Assumes exchangeability between cal and test | Explicitly handles covariate shift via LR weights |
| Bucket-conditional but not feature-conditional | Both bucket-conditional AND feature-weighted via LR |
| Bounds collapse trivially at high α for binary | Mondrian + LR weighting preserves usable bounds at α=0.05 in many buckets |
| Single calibration set, no shift adjustment | Adapts to evolving test distribution as 2026 data accumulates |

### Source citations

- Tibshirani, Foygel Barber, Candes, Ramdas. "Conformal Prediction Under Covariate Shift." NeurIPS 2019. ([arxiv:1904.06019](https://arxiv.org/abs/1904.06019)) — foundational weighted-conformal paper
- Sesia & Candès 2020 — handles binary outcomes specifically
- Sugiyama, Suzuki, Kanamori 2012 — density ratio estimation
- Jonkers et al. 2024 — recent extension under covariate shift
- Wilson 1927 — original Wilson confidence interval (used as sanity check)
- Vovk 2013 — finite-sample corrections in split conformal
- Lopez de Prado 2018 — combinatorial purged CV for time-series validation

## 5. Storage format + refresh policy

### Pick record extension

```python
@dataclass
class Pick:
    # ... existing 12 fields ...
    p_game_hit_lower_conformal_95: float | None = None
    p_game_hit_lower_conformal_90: float | None = None
    p_game_hit_lower_conformal_80: float | None = None
    p_game_hit_lower_wilson_95: float | None = None
    p_game_hit_lower_wilson_90: float | None = None
    p_game_hit_lower_wilson_80: float | None = None
```

All optional with `None` default. Backward-compatible with existing pick JSON files. Populated when `BTS_USE_CONFORMAL=1` (default ON for v1).

### Calibration cache

```
data/conformal/
  ├── calibrator_{YYYY-MM-DD}.pkl
  ├── wilson_calibrator_{YYYY-MM-DD}.pkl
  ├── lr_classifier_{YYYY-MM-DD}.pkl
  └── validation_log.jsonl
```

Dated filenames preserve ~30 days of history for retrospective analysis (existing `data/.cleanup_policy` cron prunes).

### Refresh policy

- **Daily**: alongside blend retrain (~1-3 minutes). Cheap.
- **Forced**: `BTS_FORCE_CONFORMAL_REFRESH=1` (deploy-hook can set).
- **Drift trigger**: 7-day rolling marginal coverage in validation_log diverges >5pp from claimed → force refresh.

### Validation log entry shape

Each fit appends one line to `data/conformal/validation_log.jsonl`:

```json
{
  "fit_date": "2026-05-01",
  "n_calibration": 3650,
  "n_effective_after_lr_weighting": 2987,
  "lr_weight_summary": {"mean": 1.0, "median": 0.92, "p95": 1.84, "p5": 0.42},
  "alphas": [0.05, 0.10, 0.20],
  "marginal_coverage": [0.943, 0.897, 0.804],
  "bucket_coverage": {
    "[0.700, 0.725)": [0.91, 0.86, 0.79],
    "[0.725, 0.750)": [0.94, 0.89, 0.81]
  },
  "median_interval_width": [0.18, 0.12, 0.07]
}
```

## 6. Validation gate (D triple test, applied to both methods)

Per the 2026-04-16 isotonic lesson (analytical positive but MC bootstrap negative → REJECTED): bootstrap-grounded validation, not point estimates.

### `scripts/validate_conformal.py`

**Step 1 — K-fold CV with 7-day embargo**

K=5 folds. Each fold is a contiguous date range (NOT random row-shuffling), preserving temporal structure. Embargo: drop rows from train fold within 7 days of any held-out date (prevents leakage from short-horizon residual correlation; standard time-series CV practice per Lopez de Prado 2018). Held-out dates are stratified to ensure each fold covers representative bucket distribution.

**Step 2 — Per-fold metrics** (per method × α)

```
marginal_coverage = mean(actual_hit ≥ L) on held-out fold
per_bucket_coverage[B_k] = mean(actual_hit ≥ L) within each bucket (n_k ≥ 30)
median_interval_width = median(predicted_p - L)
```

**Step 3 — MC bootstrap aggregation**

Pool held-out predictions across folds. 1000 bootstrap resamples. 95% CI on each statistic.

**Step 4 — Decision rule (per method × α independently)**

| Gate | Threshold | Why |
|------|-----------|-----|
| **Marginal coverage** | Bootstrap 95% CI of `coverage` includes claimed `(1−α)` ± 2pp | The fundamental conformal guarantee |
| **Bucketed coverage** | At most 10% of populated buckets (n_k ≥ 30) have coverage outside `(1−α) ± 5pp` | Catches conditional miscoverage that marginal hides |
| **Tightness** | Bootstrap 95% CI of `median_width` upper bound: < `0.20` at α=0.10, < `0.10` at α=0.20, no requirement at α=0.05 | At α=0.05, binary-outcome conformal can legitimately be trivial; informativeness threshold relaxed accordingly |

### Per-method × per-α shipping policy

Each (method, α) ships independently. If only some pass, only those are populated in pick records. Fields that fail validation get `None`.

This avoids "all-or-nothing" rigidity; allows partial deployment as data improves.

### Decision matrix output

```json
{
  "weighted_mondrian_conformal": {
    "0.05": {"marginal": "PASS", "bucketed": "PASS", "tightness": "FAIL", "ship": false},
    "0.10": {"marginal": "PASS", "bucketed": "PASS", "tightness": "PASS", "ship": true},
    "0.20": {"marginal": "PASS", "bucketed": "PASS", "tightness": "PASS", "ship": true}
  },
  "bucket_wilson": {
    "0.05": {...}, "0.10": {...}, "0.20": {...}
  }
}
```

## 7. Phasing

### v1 (this design)

- Compute and store 6 lower-bound fields in pick records
- Calibrators refresh daily alongside blend retrain
- `BTS_USE_CONFORMAL=1` env var (default ON for v1, set in bts-hetzner `.env` at first deploy)
- Validation log accumulates daily; drift watchdog triggers re-fit
- **No change to MDP behavior**

### v1.5 (separate brainstorm, gated on v1 success)

- Wire conformal bounds to MDP decisions at high streak
- Specific design questions deferred: bin-substitution vs veto-rule vs hybrid
- Streak threshold (e.g., ≥47, or continuous schedule) requires its own backtest validation
- Prerequisite: ≥30 days of v1 data + clean MDP-with-bounds vs MDP-without backtest comparison

### v2 (later, gated on v1.5 success)

- **False Coverage Rate (FCR) control** across (method × α × bucket) matrix (Benjamini & Yekutieli 2005)
- **Anytime-valid sequential testing** for streaming validation (Howard et al. 2021)
- **Online conformal updates** during the season
- **Locally-adaptive widths** via kernel smoothing (Lei et al. 2018)
- **Beta-conformal hybrid** if Wilson dominates conformal in v1 validation

## 8. Out of scope (explicit, prevents creep)

- **MDP integration**: any production decision use of bounds. v1 is data-only.
- **Multi-class extensions**: BTS is binary.
- **Counterfactual explanations**: "what feature change lifts this bound?"
- **TreeSHAP-on-bounds**: per-feature attribution belongs in broader interpretability upgrade.
- **Conformalized quantile regression** (Romano et al. 2019): would require a quantile model.
- **Per-batter conformal personalization**: separate calibrator per batter is over-calibrated for our n.

## 9. Validation gate for shipping v1

**This design is NOT deployed until**:

1. `scripts/validate_conformal.py` runs and produces a non-empty "ship" set in the decision matrix
2. At least one (method, α) combination passes all three gates
3. A 7-day soak in production with `BTS_USE_CONFORMAL=1`, monitoring `validation_log.jsonl` for coverage drift
4. `validate_conformal.py` re-runs weekly during soak; failure to maintain coverage triggers automatic flag-flip back to OFF

If validation produces empty "ship" set, the spec is rejected and we revisit (likely an LR-classifier or calibration-data issue).

## 10. Notes from the SOTA audit (for cross-reference)

This design is the **first concrete output** of the project-wide SOTA audit (`project_bts_state_of_art_audit_2026_05_01.md`). It exemplifies the audit's intended pattern: pull state-of-the-art techniques forward (Weighted Conformal Prediction, Mondrian binning, density-ratio reweighting) rather than defaulting to "what's in the codebase."

The other 9 audit areas (CVaR-MDP, Beta calibration, TreeSHAP, always-valid sequential testing, CPCV, drift detection, FDR, online conformal, transformer features, Bayesian ensemble) are tracked separately and will get their own brainstorms when prioritized.
