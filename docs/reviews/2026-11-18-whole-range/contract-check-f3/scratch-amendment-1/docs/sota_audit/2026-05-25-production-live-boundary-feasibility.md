# Production-Live Boundary Feasibility (2026-05-25)

**Status:** inventory and feasibility only. No boundary derivation, no
reconciliation map, no policy artifact write, and no deploy claim.

## Purpose

Gate B closed with `NO SWAP` because the estimated-PA backtest boundary surface
did not transfer cleanly to production-live `p_game_hit`. This slice asks a
narrower question:

Can we currently derive or reconcile MDP boundaries from the production-live
probability scale?

The target distribution is the probability the MDP acts on: production primary
rank-1 `p_game_hit`, with double-down rank-2 probabilities inventoried
separately. This is not a full candidate-slate study and not a policy-swap
attempt.

## Command

Production pick JSON was copied read-only from the production host into
`/tmp/bts_prod_picks_2026-05-24`. Historical estimated-PA profile rows were read
from `/tmp/bts_gate_b_estimated_pa_profiles`.

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/production_live_boundary_feasibility.py \
  --picks-dir /tmp/bts_prod_picks_2026-05-24 \
  --historical-profiles-dir /tmp/bts_gate_b_estimated_pa_profiles \
  --output /tmp/production_live_boundary_feasibility_2026-05-25.json \
  --date 2026-05-25 \
  --pretty
```

The output artifact reports:

- `production_deploy_claim=false`
- `writes_policy_artifact=false`
- `derives_boundaries=false`
- `builds_reconciliation_map=false`

## Inventory

Authoritative production pick JSON:

| Surface | n | date range | mean | q10 | q50 | q90 |
|---|---:|---|---:|---:|---:|---:|
| primary rank-1 | `58` | 2026-03-29 to 2026-05-25 | `0.748464` | `0.715626` | `0.750392` | `0.781297` |
| double-down rank-2 | `54` | 2026-03-29 to 2026-05-25 | `0.723764` | `0.689369` | `0.729671` | `0.750048` |

Lineup evolution logs contain more rows, but they are repeated saved
primary/double-down observations, not a full ranked slate:

| Surface | rank-1 n | rank-2 n | date range | role |
|---|---:|---:|---|---|
| lineup evolution | `113` | `113` | 2026-05-01 to 2026-05-25 | audit trail only |

No full ranked production-live candidate slate is available from the copied
`data/picks` snapshot. The available production-live surface is selected
primary/double-down decisions, not the full daily candidate distribution.

## Provenance Windows

Pick provenance is incomplete before 2026-05-05.

| Field | rank-1 present | rank-1 missing | present fraction |
|---|---:|---:|---:|
| `model_git_sha` | `21` | `37` | `0.362` |
| `model_pickle_sha256` | `21` | `37` | `0.362` |
| `policy_npz_sha256` | `21` | `37` | `0.362` |

Best non-null policy-hash window:

| Window | rank-1 n | rank-2 n | policy hash |
|---|---:|---:|---|
| 2026-05-05 to 2026-05-25 | `21` | `21` | `66d154717ae5...` |

Best strict `model_git_sha + policy_npz_sha256` window:

| Window | rank-1 n | rank-2 n | git | policy |
|---|---:|---:|---|---|
| 2026-05-05 to 2026-05-08 | `4` | `4` | `a3bc4d346056...` | `66d154717ae5...` |

The strict git window is too strict to be a scale-stability proof because daily
repo commits can be docs or tooling changes. But the opposite problem is also
true: pick JSON does not persist environment-level feature configuration, so a
policy-hash window alone cannot prove model/feature-scale stability. This is
why the feasibility conclusion treats the 21-pick policy-hash window as the
best available live-scale support, not as validated training support.

## Scale Parity

The production-live probability scale remains materially below the historical
estimated-PA backtest surface.

All production rank-1 picks versus historical estimated-PA rank-1:

| Distribution | n | mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| production-live rank-1 | `58` | `0.748464` | `0.715626` | `0.750392` | `0.781297` |
| historical estimated-PA rank-1 | `912` | `0.779394` | `0.750903` | `0.776991` | `0.811070` |
| live - historical |  | `-0.030931` | `-0.035277` | `-0.026599` | `-0.029772` |

Recent non-null policy-hash window versus historical estimated-PA rank-1:

| Distribution | n | mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| production-live rank-1 | `21` | `0.747199` | `0.723034` | `0.750100` | `0.781091` |
| historical estimated-PA rank-1 | `912` | `0.779394` | `0.750903` | `0.776991` | `0.811070` |
| live - historical |  | `-0.032195` | `-0.027869` | `-0.026891` | `-0.029978` |

The pre-set materiality rule flags `>= 0.03` absolute mean or median delta, or
`>= 0.05` absolute anchor-quantile delta. Both all-picks and recent-policy
windows cross the mean-delta threshold.

## Feasibility Verdict

Decision:

`NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N`

Direct live-boundary derivation is not currently feasible. The run used a
minimum support rule of `250` rank-1 points to fit boundaries plus `100` rank-1
points for holdout, or `350` live rank-1 points total. The best non-null
policy-hash window has only `21` rank-1 points.

Backtest-to-live reconciliation is also not ready to build. The run used a
minimum exploratory reconciliation support rule of `50` policy-stable rank-1
points. The best available policy-hash window has `21`.

The useful fork is therefore:

1. `WAIT_FOR_MORE_LIVE_N`: accumulate more production-live points before any
   derivation or reconciliation attempt.
2. `PREREGISTER_RECONCILIATION`: once enough live support exists, test whether
   the backtest-to-live gap is a stable transform that can be applied to the
   multi-season historical estimated-PA surface.

Direct live derivation remains the slower path because production gives roughly
one primary point per day and no multi-season holdout. Reconciliation is the
more plausible future path, but it still needs a separate pre-registration and
more live-scale support than exists now.

Instrumentation recommendation: future pick artifacts should persist a compact
feature-environment fingerprint for scale-affecting settings such as
`BTS_PITCHER_HR_30G_MIN_PERIODS`, `BTS_ROOKIE_GATE_K`, deterministic LightGBM
mode, and related model-scale flags. A complete live-scale fingerprint would be
`model_pickle_sha256 + feature_env_hash`, which is stronger than the current
proxy fields and would make future stability-window inventory definitive.

## Non-goals

This result does not:

- derive production-live boundaries;
- fit a backtest-to-live transform;
- generate or modify `data/models/mdp_policy.npz`;
- change production pick selection;
- deploy anything.

Any future artifact still requires a separate pre-registration, leakage audit,
nuclear test, reversible artifact generation, and explicit deploy gate.

## Verification

Focused tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked pytest tests/scripts/test_production_live_boundary_feasibility.py -q
```

Result: `2 passed`.
