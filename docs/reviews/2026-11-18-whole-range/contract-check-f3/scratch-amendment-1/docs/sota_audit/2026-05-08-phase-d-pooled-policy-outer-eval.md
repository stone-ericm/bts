# Phase D Pooled-Policy Outer Evaluation

- **Generated**: 2026-05-08
- **Status**: `phase_d_outer_eval_falsified`
- **Artifact**: `data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json`
- **Generator**: `scripts/phase_d_pooled_policy_outer_eval.py`
- **Production deploy claim**: `false`

## Verdict

The pre-registered pooled-policy candidate is falsified on the disjoint 2025
outer-evaluation surface.

The candidate improves the 2021-2024 selection surface, but it fails the
holdout year decisively:

| Surface | Production P(57) | Pooled candidate P(57) | Gap |
| --- | ---: | ---: | ---: |
| Selection, 2021-2024 | `0.067049` | `0.096378` | `+0.029329` |
| Outer eval, 2025 | `0.127678` | `0.064691` | `-0.062987` |

Primary uncertainty is a provider-stratified seed bootstrap over the 48 Hetzner
and 52 OCI seeds:

- `n=100`
- mean P(57) gap: `-0.062987`
- 95% bootstrap CI: `[-0.065250, -0.060757]`
- positive seeds: `0/100`
- exact sign-test two-sided p-value: `1.578e-30`

Provider-tagged sensitivity agrees with the pooled result:

| Provider | Seeds | Mean gap |
| --- | ---: | ---: |
| Hetzner | `48` | `-0.062095` |
| OCI | `52` | `-0.063810` |

This is not deploy-ready, and it is not inconclusive. Under the pre-registered
verdict ladder, the candidate is `falsified` because the pooled-policy gap is
non-positive on the outer-evaluation surface.

## Method

The script enforces the Phase C split metadata before loading any profiles:

- selection seasons: `2021,2022,2023,2024`
- outer-evaluation seasons: `2025`
- providers: Hetzner and OCI
- expected seed count: `100`
- required artifact role: `raw_backtest_profile_surface`
- required split mode: `season_level_selection_outer_eval`
- required production deploy claim: `false`

It builds the pooled candidate policy in memory only from the selection
profiles, then evaluates both the current production policy table and the fixed
pooled candidate policy table on each seed's 2025 outer-evaluation bins.

The production reference in this artifact is the shipped local policy file
`data/models/mdp_policy.npz`. The Phase D script does not rebuild a per-seed
production policy and does not retrain the production reference during the
run. The comparison is therefore:

```text
fixed shipped production policy table
vs
pooled candidate policy table built from 2021-2024 Phase C profiles
```

Both fixed policy tables are evaluated on the same 2025 outer-evaluation
profile bins.

P@1 is included as a shared 2025 outer-surface diagnostic. It is not reported
as a candidate-vs-production policy gap because this candidate changes only the
MDP policy table, not the rank-1 probability model.

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_phase_d_pooled_policy_outer_eval.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase_d_pooled_policy_outer_eval.py
```

Verification result:

```text
5 passed
wrote data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json | n=100 mean_gap=-0.062987 ci=[-0.065250, -0.060757] verdict=falsified
```

## Implication

The older pooled-policy positives remain useful as a candidate-generation
screen, but they do not survive the stricter Phase C/D protocol. The immediate
next step should not be deployment or a larger pooled-policy confirmation run.
It should be a post-mortem on why the selection-surface improvement reverses on
2025, then a new candidate-selection pass that remains blind to a future outer
evaluation target.

One read-only follow-up diagnostic supports that interpretation: on pooled 2025
outer bins, the 2021-2024 selection policy evaluates at `0.060968`, production
at `0.125353`, and an outer-only hindsight policy would evaluate at `0.357596`.
The outer-only policy is not deployable evidence because it is trained on the
holdout year, but it shows that 2025 is a materially different policy manifold
rather than just random seed noise.

## Relation To Earlier C0 Screen

Claude requested a sanity check before treating the result as the cycle's
scientific verdict, because the old C0 artifact reported a positive
leave-one-out gap:

- `data/validation/pooled_policy_ab_24seed_consolidated.json`
- C0 leave-one-out mean gap: `+0.019290`
- positive seeds: `24/24`

That C0 artifact and this Phase D artifact answer different questions. C0 uses
the older full-season/leave-one-seed-out pooled-policy screen. Phase D uses the
pre-registered temporal split: build from `2021-2024`, evaluate on `2025`.

As a cross-check, the Phase D temporal-split method was applied read-only to the
old 24-seed raw surface (`pooled_bins_run` plus `pooled_bins_run_trackc`):

| Old 24-seed surface under Phase D split | Production P(57) | Pooled candidate P(57) | Gap |
| --- | ---: | ---: | ---: |
| Selection, 2021-2024 | `0.098066` | `0.109665` | `+0.011599` |
| Outer eval, 2025 | `0.055207` | `0.023139` | `-0.032068` |

Old-surface temporal-split result:

- `n=24`
- mean P(57) gap: `-0.032068`
- 95% iid seed-bootstrap CI: `[-0.039162, -0.025890]`
- positive seeds: `0/24`
- exact sign-test two-sided p-value: `1.192e-07`

This means the C0-to-Phase-D flip is primarily an estimand change: the older
C0 screen nominated the pooled-policy candidate, but it did not test the
deployment-relevant temporal outer-evaluation claim. Under the temporal split,
both the old 24-seed surface and the new deterministic 100-seed surface reject
the pooled-policy candidate.

Post-mortem and next-candidate plan:
`docs/sota_audit/2026-05-08-pooled-policy-postmortem-next-candidates.md`.
