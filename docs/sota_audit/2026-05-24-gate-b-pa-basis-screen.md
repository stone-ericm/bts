# Gate B PA-Basis Re-bin Screen (2026-05-24)

**Status:** measurement-only. No production behavior change.

## Question

The PA-basis investigation found that the deployed MDP bins were built on
actual-PA-expanded historical game probabilities, while production sees
estimated-PA game probabilities at pick time.

This screen asks a narrower follow-up question:

If historical backtest profiles are transformed onto a production-like PA
basis, does re-binning and re-solving the MDP show enough point signal to
justify the heavier walk-forward policy-file harness?

## Method

The new script `scripts/measure_pa_basis_rebin_gate.py` loads
`data/simulation/backtest_*.parquet` and transforms rank-1/rank-2
`p_game_hit` values from actual observed PA count to target production-like PA
count:

1. Infer a first-order PA probability:
   `p_pa = 1 - (1 - p_game_hit) ** (1 / n_pas)`.
2. Re-aggregate to target PA volume:
   `p_game_hit_pa_basis = 1 - (1 - p_pa) ** target_pas`.
3. Rebuild equal-frequency bins on transformed rank-1 probabilities.
4. Solve the same reachability MDP on those transformed bins.
5. Compare against the deployed policy table projected onto the transformed
   bin manifold by representative primary probability.

Default target PA volumes come from current production canonical pick JSONs:

- rank 1 / primary: `4.4293103448`
- rank 2 / double-down: `4.4222222222`

This is an exploratory transform, not a deployable Gate B. The preferred Gate B
path remains regenerating historical profiles with a real pre-game PA estimator
comparable to production.

The reported P(57) values below are in-sample and optimistic: the same
transformed 2021-2025 rows are used to fit bins, solve the MDP, and evaluate
the resulting policy table. They are only an investment screen for the heavier
walk-forward harness.

## Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/measure_pa_basis_rebin_gate.py \
  --output /tmp/pa_basis_rebin_gate_2026-05-24.json \
  --date 2026-05-24 \
  --bootstrap-reps 200
```

The transformed backtest surface loaded `912` rank-1 days. The screen found a
positive point P(57) signal across all evaluated bin counts. The CI column is
a rank-pair-day bootstrap screen that refits bins and the MDP inside each
resample; it is not a leakage-free validation interval.

| n_bins | min bin n | projected deployed baseline P(57) | PA-basis re-solve P(57) | gap | bootstrap gap 95% CI |
|---:|---:|---:|---:|---:|---:|
| 2 | 456 | `0.002567` | `0.010479` | `+0.007912` | `[+0.004070, +0.042987]` |
| 3 | 304 | `0.002491` | `0.012201` | `+0.009711` | `[+0.005494, +0.052312]` |
| 4 | 228 | `0.002438` | `0.010827` | `+0.008389` | `[+0.006441, +0.063029]` |
| 5 | 182 | `0.004758` | `0.017748` | `+0.012990` | `[+0.006742, +0.067782]` |

The screen decision is:

`SCREEN_SIGNAL_REQUIRES_POLICY_FILE_BACKTEST`

## Interpretation

This does not justify swapping `data/models/mdp_policy.npz`.

It does justify the next Gate B evidence task: a production-PA-consistent
walk-forward policy-file evaluation. The current point screen says the
PA-basis correction can materially change the MDP action surface, but the
candidate must still meet or beat the deployed baseline under a leakage-safe
multi-season evaluation before any runtime policy change is considered.

## Caveats

- The transform assumes the product aggregation form and compresses each
  historical batter-game into an implied average PA probability. Production
  splits starter and reliever PA probabilities, so this is only a first-order
  approximation.
- The deployed baseline is projected onto transformed bins by representative
  primary probability. That is exact only when transformed candidate bins do
  not cross deployed policy boundaries.
- The preferred next step is still a proper historical re-generation with a
  pre-game PA estimator, not relying on this actual-to-target PA transform as a
  final policy artifact.
- The bootstrap interval is scoped to this same in-sample screen. It resamples
  transformed rank-pair days and refits bins/MDP inside each replicate; it does
  not replace a walk-forward or profile-regeneration validation.
