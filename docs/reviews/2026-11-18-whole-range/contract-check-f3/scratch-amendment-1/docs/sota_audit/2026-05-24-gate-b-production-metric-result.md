# Gate B Production-Metric Re-baseline Result (2026-05-24)

**Status:** measurement-only. No production behavior change, no policy artifact
write, and no deploy claim.

## Question

This run measures the pre-registered boundary-only Gate B lever from
`docs/sota_audit/2026-05-24-gate-b-production-metric-prereg.md`.

Both arms keep the deployed `data/models/mdp_policy.npz` action table fixed.
Only the probability boundaries change:

- `CURRENT`: deployed action table plus deployed saved boundaries.
- `CANDIDATE`: deployed action table plus estimated-PA boundaries.

The measurement has two claims:

1. mechanism: does the boundary-only candidate clear the production MDP
   bin-collapse mechanism on 2026 production picks?
2. outcome quality: does the boundary-only candidate improve exact
   streak-ladder outcomes on held-out estimated-PA folds?

## Command

Estimated-PA profile inputs were verified at
`/tmp/bts_gate_b_estimated_pa_profiles` with `p_game_hit_basis=estimated_pa`
for all profile rows. Production pick JSON was copied read-only from
`/home/bts/projects/bts/data/picks` on the production host to
`/tmp/bts_prod_picks_2026-05-24`.

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/gate_b_production_metric_rebaseline.py \
  --profiles-dir /tmp/bts_gate_b_estimated_pa_profiles \
  --picks-dir /tmp/bts_prod_picks_2026-05-24 \
  --prod-policy-path data/models/mdp_policy.npz \
  --output /tmp/gate_b_production_metric_rebaseline_2026-05-24.json \
  --date 2026-05-24 \
  --pretty
```

The output artifact reports `production_deploy_claim=false` and
`writes_policy_artifact=false`.

## Mechanism Result

Decision:

`MECHANISM_DOWNGRADED_SCALE_PARITY_DIVERGENCE`

The boundary-only candidate clears the primary-pick bin-collapse alert
mechanically, but the production probability scale is materially lower than
the historical estimated-PA backtest scale used to derive candidate boundaries.
Per the pre-registration, that scale mismatch downgrades the mechanism result.

### Scale Parity

| Distribution | n | mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| historical estimated-PA rank-1 | `912` | `0.779394` | `0.750903` | `0.776991` | `0.811070` |
| 2026 production primary picks | `57` | `0.748849` | `0.715214` | `0.750684` | `0.781366` |
| production - historical |  | `-0.030546` | `-0.035689` | `-0.026308` | `-0.029704` |

The pre-set materiality threshold was `0.03` absolute mean or median delta, or
`0.05` absolute anchor-quantile delta. The mean delta crosses the threshold.

This is the important mechanism finding: the estimated-PA backtest boundary
surface does not transfer cleanly onto the current production-live probability
surface.

### Bin Occupancy

Recent primary picks over the default 21-pick window:

| Arm | bin counts | dominant fraction | alert |
|---|---|---:|---|
| `CURRENT` | `[21, 0, 0, 0, 0]` | `1.000` | yes |
| `CANDIDATE` | `[14, 2, 3, 2, 0]` | `0.667` | no |

The candidate primary occupancy clears the existing `mdp_policy_alignment`
threshold because no single bin contains `>= 0.80` of recent primary picks.

Double-down occupancy remains weak under candidate boundaries:

| Arm | bin counts | dominant fraction |
|---|---|---:|
| `CURRENT` | `[21, 0, 0, 0, 0]` | `1.000` |
| `CANDIDATE` | `[20, 0, 1, 0, 0]` | `0.952` |

### Decision Diffs

Only one state-known production decision changed:

| Date | current | candidate | p_game_hit | pre-streak | result |
|---|---|---|---:|---:|---|
| 2026-04-03 | `single` | `double` | `0.763046` | `6` | `miss` |

This is enough to show operational effect, but not enough to justify a swap.
The scale-parity downgrade is the governing mechanism conclusion.

## Outcome-Quality Result

Decision:

`OUTCOME_MIXED_HEADLINE_FAILS_STABILITY_BAR`

The headline `E[max streak]` gap is positive on average, but it does not clear
the pre-registered stability bar and has two negative folds.

| Metric | active | current mean | mean gap | std gap | negative folds |
|---|---|---:|---:|---:|---:|
| `E[max streak]` | yes | `17.469192` | `+1.395567` | `1.864230` | `2 / 4` |
| `P(reach >= 10)` | yes | `0.957079` | `+0.018954` | `0.024430` | `0 / 4` |
| `P(reach >= 20)` | yes | `0.286889` | `+0.100440` | `0.169535` | `2 / 4` |
| `P(reach >= 30)` | yes | `0.022462` | `+0.018700` | `0.016988` | `0 / 4` |
| `P(reach >= 40)` | no | `0.001716` | `+0.002484` | `0.001293` | `0 / 4` |

`P(reach >= 40)` is demoted to diagnostic because at least one current-arm
fold is below the `1e-3` floor guard.

Fold-level headline gaps:

| Holdout | current `E[max]` | candidate `E[max]` | gap |
|---:|---:|---:|---:|
| 2022 | `17.523402` | `20.831737` | `+3.308335` |
| 2023 | `18.166916` | `18.076531` | `-0.090385` |
| 2024 | `19.983033` | `19.670495` | `-0.312538` |
| 2025 | `14.203419` | `16.880274` | `+2.676855` |

The positive average is driven by 2022 and 2025. It does not exceed the
fold-to-fold standard deviation, and it fails the no-more-than-one-negative-fold
rule for active outcome metrics.

## Interpretation

This result does not advance a boundary-only policy artifact.

The mechanism branch shows that correcting the boundaries would clear the
primary bin-collapse alert on recent production picks, but the live production
probability scale is lower than the historical estimated-PA backtest scale.
That means the candidate boundary surface is not yet proven to be the right
production-scale surface.

The concrete scale diagnosis is about a three-point downward shift:
production-live `p_game_hit` averages `0.748849` versus `0.779394` on the
backtest-reconstructed estimated-PA rank-1 surface. Because the historical
estimated-PA bin boundaries are only a few points apart, this shift can move
production picks across bins. Any future boundary derivation must either fit
directly on production-live `p_game_hit` or first reconcile the backtest
reconstruction to that live scale.

The outcome branch is directionally interesting but not stable. The exact
streak ladder shows improvements in several rungs, but the headline metric
fails the pre-registered stability bar and sign-flips across folds.

Net decision:

- do not swap `data/models/mdp_policy.npz`;
- do not generate a replacement policy artifact from this result;
- do not deploy;
- keep Gate B closed as `NO SWAP` unless a new production-scale boundary
  derivation is pre-registered and tested.

## Verification

Focused tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked pytest tests/scripts/test_gate_b_production_metric_rebaseline.py -q
```

Result: `2 passed`.

Measurement rerun after the floor-guard implementation was tightened to demote
any support rung with a below-`1e-3` current-arm fold.
