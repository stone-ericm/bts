My recommendation: do **not** make pure covered-only training the only primary regime. It fixes the sentinel learning problem, but it creates the confound you named: the production baseline is weaker than it should be, so swing lift can be overstated.

Use a **full-history production prior + covered-era swing layer**.

**1. Training Window**
Run this as the primary design:

- Train a production-only prior model on `2019-01-01` through `2024-06-30`.
- Use **no swing features, no swing coverage indicators** in that prior.
- Generate out-of-fold prior predictions for covered-era training rows.
- Train every Stage-1 arm only on covered-era rows, but with the same frozen production prior available to every arm.

So candidate lift is measured as:

```text
production full-history prior
+ covered-era calibration / residual layer
+ optional swing feature
```

not as:

```text
weakened covered-only production model
+ swing feature
```

I would still run the pure covered-only model as a diagnostic, but not as the decision metric.

I would not use “full 2019-2024H1 with swing NaN plus coverage indicator” as primary. That tests whether LightGBM can discover a sparse late-era feature inside a mostly pre-coverage matrix. You already proved that failure mode.

**2. Rolling Warmup**
Do not let 30g/60g swing windows silently reach into pre-coverage space.

For the primary run:

- Score rows: require rolling swing window coverage to be fully or near-fully covered. For 2024 H2 this should mostly be true.
- Training rows: either start after 60-game swing warmup, or enforce a row-level coverage threshold.

Concrete rule:

```text
include row if:
  game_date between 2023-07-01 and 2024-06-30
  swing_feed_available = true
  rolling_60g_swing_coverage >= 0.90
```

If that starts the effective training set around late August or September 2023, accept it. Clean coverage matters more than squeezing in early rows that recreate the same ambiguity.

**3. Gross Sentinel Expectation**
Yes, the gross same-day whiff sentinel should now explode.

Not “barely positive.” It should be:

```text
every seed positive
week-block p very small
delta far above practical null, preferably multiple times larger
```

If it still does not inflate, the conclusion is not “swing features are useless.” The conclusion is: **the harness still cannot learn or evaluate an obvious leak. Candidate results are uninterpretable.**

Next move if it fails:

1. Score raw sentinel alone by daily rank-AUC.
2. Score `production_prior + sentinel` with a simple logistic/calibration model.
3. Train a single-feature LightGBM on sentinel only.
4. Add an oracle canary: the label or same-day target-derived rank as a feature.

Interpretation:

```text
oracle fails      -> evaluator / label / AUC direction bug
oracle passes, raw sentinel fails -> sentinel is not actually aligned with target
raw passes, model fails -> model training / feature ingestion / missing handling bug
model passes, paired delta fails -> baseline pairing / daily aggregation bug
```

**4. Power**
With 88 score days, 10 seeds is not the real limit. The independent unit is the score day or week block, not the seed.

After coverage is fixed, this may be adequate for a large sentinel and maybe for stable `+0.005` effects, but it is still marginal for candidate screening if the null band remains around `0.003-0.005`.

I would run:

```text
30 seeds, not 10
week-blocked sign permutation remains primary
daily paired deltas remain primary
```

Then add a powered screen sensitivity using all covered 2024 via time folds if you want more power without touching 2025:

```text
Fold A:
  prior train: 2019-2023
  swing layer train: 2023 covered warm rows
  score: 2024 H1 covered rows

Fold B:
  prior train: 2019-2024 H1
  swing layer train: 2023 covered warm rows + 2024 H1
  score: 2024 H2
```

If the canary passes but candidates are still hovering around `+0.004` to `+0.006`, I would spend 2025 as a pre-registered replication screen and reserve 2026 ABS as final confirmation. If 2025 must remain untouched, then accept that Stage-1 may still be underpowered for small effects.

**5. Exact Construction**
Pseudocode-level design:

```python
T_full = rows[
    (date >= "2019-01-01") &
    (date <= "2024-06-30")
]

T_cov = rows[
    (date >= "2023-07-01") &
    (date <= "2024-06-30") &
    swing_feed_available &
    (rolling_60g_swing_coverage >= 0.90)
]

E = rows[
    (date >= "2024-07-01") &
    (date <= "2024-regular-season-end") &
    swing_feed_available &
    (rolling_60g_swing_coverage >= 0.90)
]
```

For each seed:

```python
# production prior
for week_block in T_cov.week:
    prior_model_oof = fit_lgbm(
        train=T_full excluding week_block,
        features=production_features_only,
        seed=seed,
    )
    T_cov["prod_prior"] = prior_model_oof.predict(T_cov rows in week_block)

prior_model_final = fit_lgbm(
    train=T_full,
    features=production_features_only,
    seed=seed,
)
E["prod_prior"] = prior_model_final.predict(E)
```

Then all arms use identical `T_cov`, identical `E`, identical labels, identical weights, identical seeds.

```python
base_features = [
    "prod_prior",
    "prod_prior_daily_rank",
]
```

Baseline arm:

```python
baseline = fit_lgbm(
    train=T_cov,
    features=base_features,
    seed=seed,
)
```

Candidate arm:

```python
candidate = fit_lgbm(
    train=T_cov,
    features=base_features + [
        candidate_feature,
        candidate_feature_present,
        candidate_window_coverage,
    ],
    seed=seed,
)
```

Gross sentinel:

```python
gross = fit_lgbm(
    train=T_cov,
    features=base_features + [
        same_day_whiff_count_zero_filled,
        same_day_whiff_present,
    ],
    seed=seed,
)
```

Nulls:

```python
ctl_permuted = base_features + [
    permute_within_date_or_date_role_bucket(candidate_feature),
    candidate_feature_present,
    candidate_window_coverage,
]

ctl_mask_only = base_features + [
    candidate_feature_present,
    candidate_window_coverage,
]
```

Evaluate all arms on the same `E` rows:

```python
daily_delta = daily_rank_auc(arm_pred, y) - daily_rank_auc(baseline_pred, y)
seed_avg_delta = mean_over_seeds(daily_delta)
primary_delta = mean_over_days(seed_avg_delta)
p_value = week_blocked_sign_permutation(seed_avg_delta)
```

**Exact Config I’d Run**
- Production prior train: `2019-01-01` through `2024-06-30`, production features only.
- Swing-layer train: covered/warmed rows from `2023-07-01` through `2024-06-30`, with `rolling_60g_swing_coverage >= 0.90`.
- Score: covered/warmed `2024 H2` regular-season rows only.
- Seeds: `30`.
- Primary stat: seed-averaged paired daily rank-AUC delta.
- Inference: week-blocked sign permutation.
- Required canary gate:
  - gross sentinel every seed positive,
  - gross sentinel delta > `5x practical_null_band`,
  - gross sentinel one-sided `p <= 0.001`.
- Practical null band:
  - max absolute delta from `ctl_permuted`, `ctl_mask_only`, and other placebo arms.
- Candidate pass:
  - delta > practical null band,
  - preferably delta >= `+0.005`,
  - one-sided week-block `p <= 0.05`,
  - not explained by mask-only control.

Confidence: high that this resolves the **harness truth**. If the gross sentinel passes and candidates remain null, I would believe the Stage-1 swing candidates are not producing a detectable 2024 H2 lift under this metric. Confidence is lower that 88 days alone resolves marginal `+0.005` candidates; for that, I would add the folded all-2024 sensitivity or spend 2025 as a replication screen after the canary passes.