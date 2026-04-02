# Item 8: Contact Quality Composite Investigation

**Date:** 2026-04-02
**Status:** DONE
**Verdict:** REJECT — composite is statistically worse than the best individual component

## Question

Does a single composite contact-quality feature (z-scored average of barrel rate,
hard-hit rate, sweet-spot rate, and avg exit velocity) capture interaction effects
that the individual blend members miss?

r/beatthestreak user shefBoiRDee uses xBA, xwOBA, and hard-hit rate as a weighted
composite. We proxy this with the 4 batter Statcast batted-ball features already
in the 12-model blend.

## Setup

- PA data: `data/processed/pa_{season}.parquet` (2021–2025, 903,180 PAs)
- All features use date-level shift(1) temporal guard (provably leak-free)
- Composite = standardize each component (z-score across all PAs), then average
  available z-scores; NaN if fewer than 2 components present
- 784,017 PAs (86.8%) have all 4 components available

## Findings

### Component correlation matrix

Computed on 784,017 PAs where all 4 components are present:

|            | barrel | hard_hit | sweet_spot | avg_ev |
|------------|--------|----------|------------|--------|
| barrel     | 1.000  | 0.704    | 0.312      | 0.656  |
| hard_hit   | 0.704  | 1.000    | 0.168      | 0.846  |
| sweet_spot | 0.312  | 0.168    | 1.000      | 0.203  |
| avg_ev     | 0.656  | 0.846    | 0.203      | 1.000  |

Pairwise summary: min=0.168, **avg=0.481**, max=0.846

The structure is notable:
- `hard_hit` and `avg_ev` are highly correlated (r=0.846) — both are EV-based, just
  different thresholds
- `barrel` correlates moderately with both EV features (0.656–0.704)
- `sweet_spot` is the outlier — it's angle-based and relatively independent (r=0.168–0.312)

Average pairwise r=0.481 falls in the "moderate" range (0.3–0.7), suggesting the
4 components are not purely redundant. But the composite fails anyway (see below).

### Correlation with is_hit (point-biserial r)

| Feature                  | r       | p-value    | n       |
|--------------------------|---------|------------|---------|
| batter_barrel_rate_30g   | +0.0013 | 0.248      | 784,017 |
| batter_hard_hit_rate_30g | +0.0091 | 5.8e-16 *  | 784,017 |
| batter_sweet_spot_rate_30g | +0.0039 | 6.0e-04 * | 784,017 |
| batter_avg_ev_30g        | +0.0133 | 3.9e-32 *  | 784,017 |
| **contact_composite**    | +0.0088 | 4.9e-15 *  | 784,017 |

The composite (r=+0.0088) is **significantly worse** than the best individual
component (`batter_avg_ev_30g`, r=+0.0133). Fisher z-test: z=−2.808, p=0.005.

The composite is dragged down by `batter_barrel_rate_30g` (r=+0.0013, not
statistically significant). Averaging a non-signal feature with the signal features
dilutes the signal.

### Per-season stability

| Season | barrel  | hard_hit | sweet_spot | avg_ev  | composite |
|--------|---------|----------|------------|---------|-----------|
| 2021   | +0.0014 | +0.0101  | +0.0046    | +0.0177 | +0.0110   |
| 2022   | +0.0027 | +0.0125  | +0.0049    | +0.0123 | +0.0105   |
| 2023   | −0.0019 | +0.0049  | +0.0013    | +0.0116 | +0.0050   |
| 2024   | +0.0042 | +0.0112  | +0.0052    | +0.0134 | +0.0108   |
| 2025   | −0.0002 | +0.0067  | +0.0032    | +0.0111 | +0.0066   |

`avg_ev` leads 4 of 5 seasons. The composite never leads. Barrel rate goes negative
in 2023 and 2025, dragging the composite down.

## Why the composite underperforms

1. **Barrel rate is noise at PA level.** r=+0.0013 (p=0.25) — not statistically
   significant. Averaging it in dilutes the signal from the other components.

2. **EV features dominate the signal.** `avg_ev` and `hard_hit` (both EV-based)
   carry most of the contact-quality signal. Including angle-based features
   (barrel, sweet_spot) adds noise relative to the EV signal.

3. **Equal weighting is wrong.** Z-scoring and averaging assigns equal weight to
   each component, but the signal is clearly concentrated in `avg_ev`. A
   weighted composite would need regularized weights — and at that point, LightGBM
   already does this implicitly in each blend model.

4. **The blend already handles this optimally.** The 12-model blend lets each
   Statcast feature compete independently. LightGBM weights features by their
   actual predictive value within each model. A hand-crafted composite can only
   do worse than the model's own learned weighting.

## Relationship to existing blend

The 12 blend models each use `FEATURE_COLS + one Statcast feature`. The 4 contact
quality components are already 4 separate blend members:
- Model 9: baseline + `batter_barrel_rate_30g`
- Model 10: baseline + `batter_hard_hit_rate_30g`
- Model 11: baseline + `batter_sweet_spot_rate_30g`
- Model 12: baseline + `batter_avg_ev_30g`

These 4 models already vote collectively on each pick. Adding a 13th model using
the composite would introduce a correlated vote (r=0.481 average between composite
and each component) without adding signal — pure noise dilution.

## Verdict

**REJECT. Do not add composite as a 13th blend model.**

The composite is statistically worse than the best individual component (avg_ev).
The equal-weighting assumption is incorrect (barrel rate contributes noise), and
the blend already handles multi-component voting optimally through independent models.
No backtest needed.

Architecture note: barrel rate is the weakest contact-quality signal at PA level
(not statistically significant). The blend model using barrel rate may itself be
noise — worth monitoring but not acting on absent a dedicated ablation test.
