# Probability Scale PA-Basis Investigation (2026-05-24)

**Status:** diagnostic memo only. No production behavior change.

## Summary

The current production probability scale is materially lower than the
2021-2025 backtest probability scale used to build the deployed MDP quality
bins. The primary explanation is a PA-basis mismatch, not a same-scale 2026
distribution shift.

Historical backtest profiles aggregate each batter-game over the realized
number of PA rows in the game. Production aggregates over pre-game estimated
lineup PAs. That is the only information available at pick time, but it means
the deployed MDP bins were built from a hindsight-PA-expanded game probability
surface.

This reframes the current MDP-bin collapse: Gate B should rebuild and evaluate
policy bins on a production-consistent estimated-PA basis before treating the
problem as a calibration-map or model-retraining problem. Gate A calibration
remains a separate track.

## Code Path Check

Production uses lineup-position PA estimates and a game-level independence
product:

- `src/bts/model/predict.py:657-674` maps lineup slots to `est_pas`, splits
  starter and reliever PAs, and computes
  `p_game_hit = 1 - (1 - p_hit_vs_starter) ** starter_pas *
  (1 - p_hit_vs_reliever) ** reliever_pas`.
- Blend ranking uses the same game-level aggregation shape at
  `src/bts/model/predict.py:694-699`.

The 2021-2025 backtest profile path uses actual PA rows:

- `src/bts/simulate/backtest_blend.py:682-687` computes
  `p_game_hit = 1 - prod(1 - p_hit_blend)` and stores `n_pas` as the actual
  PA-row count.

The product-form aggregation means a per-game implied PA probability

`p_pa = 1 - (1 - p_game_hit) ** (1 / n_pas)`

is a fair first-order normalization for measuring the PA-volume effect. It is
not a replacement for a proper production-consistent backtest because production
uses starter/reliever splits and the historical PA probabilities can vary
within a batter-game.

## Evidence

Deployed policy boundaries from `data/models/mdp_policy.npz`:

`[0.795979, 0.811491, 0.825247, 0.840740]`

Backtest rank-1 raw `p_game_hit` from `data/simulation/backtest_*.parquet`:

| Measure | Value |
| --- | ---: |
| n | 912 |
| min | 0.732903 |
| p20 | 0.794352 |
| median | 0.816654 |
| mean | 0.818333 |
| p80 | 0.841115 |
| max | 0.918326 |

Backtest rank-1 actual PA volume:

| n_pas | days |
| ---: | ---: |
| 4 | 8 |
| 5 | 472 |
| 6 | 402 |
| 7 | 28 |
| 8 | 2 |

Mean actual `n_pas` for backtest rank-1 is `5.500`.

Production canonical primary picks from `data/picks/YYYY-MM-DD.json` on prod,
2026-03-29 through 2026-05-25:

| Measure | Value |
| --- | ---: |
| n | 58 |
| min | 0.690137 |
| p20 | 0.723274 |
| median | 0.750392 |
| mean | 0.747920 |
| p80 | 0.777767 |
| max | 0.791806 |

Production primary estimated PA basis:

| lineup | days |
| ---: | ---: |
| 1 | 41 |
| 2 | 12 |
| 3 | 4 |
| 5 | 1 |

Mean estimated PAs for production primary picks is `4.429`.

If each backtest rank-1 game probability is inverted to an implied PA
probability and re-aggregated at `4.429` PAs, the backtest rank-1 distribution
becomes:

| Measure | Value |
| --- | ---: |
| n | 912 |
| min | 0.666783 |
| p20 | 0.721178 |
| median | 0.748074 |
| mean | 0.749201 |
| p80 | 0.775935 |
| max | 0.850768 |

That almost exactly matches the observed production primary-pick scale
(`p20=0.723274`, `median=0.750392`, `mean=0.747920`). Matching the distribution
shape, not just the mean, is the strongest evidence that PA-basis mismatch is
the main driver.

The same pattern appears for rank-2 / double-down scale:

| Surface | p20 | median | mean |
| --- | ---: | ---: | ---: |
| Backtest rank-2 raw | 0.782038 | 0.801985 | 0.802643 |
| Backtest rank-2 normalized to 4.422 PAs | 0.711103 | 0.739631 | 0.737574 |
| Production double-down | 0.703235 | 0.729671 | 0.723361 |

## Interpretation

The deployed MDP bins are on a hindsight actual-PA probability basis. Current
production picks are on a forecast estimated-PA basis. Because going from about
5.5 realized PAs to about 4.4 estimated PAs lowers `P(>=1 hit)` by roughly the
same amount as the observed policy-bin collapse, the bin mismatch should not be
fixed by blindly calibrating current probabilities upward.

This also means historical P(57) estimates that consumed actual-PA-expanded
profile probabilities should be treated as optimistic for live decision making:
they used PA volume that is not available at pick time.

## Decision

Do not change live pick selection, calibration, or the deployed MDP policy from
this memo alone.

Update the Gate B framing from "raw probability re-bin" to
"production-PA-consistent re-bin". A valid Gate B candidate should use one of:

1. Historical profiles generated with a pre-game PA estimator comparable to the
   production lineup-position estimator.
2. A frozen transformation from actual-PA backtest profiles onto the production
   PA basis, used only as an exploratory screen before a proper walk-forward
   policy-file evaluation.

The candidate policy still needs the existing P(57) gate before any deployment.
Gate A calibration should continue to require separate leakage-free,
proper-score evidence.

## Reproduction Notes

Backtest evidence came from:

`UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z uv run --locked python ...`

reading `data/simulation/backtest_2021.parquet` through
`data/simulation/backtest_2025.parquet`.

Production evidence came from prod canonical date pick JSON files only:

`/home/bts/projects/bts/data/picks/YYYY-MM-DD.json`

The production sample intentionally excludes archived or non-canonical pick
JSON files.
