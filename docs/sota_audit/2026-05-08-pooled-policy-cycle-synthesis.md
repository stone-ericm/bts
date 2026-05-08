# Pooled-Policy Cycle Synthesis

- **Generated**: 2026-05-08
- **Status**: `cycle_closed_no_deployable_candidate`
- **Production deploy claim**: `false`
- **Primary artifacts**:
  - `data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json`
  - `data/validation/phase_d_pooled_policy_postmortem_2026-05-08.json`
  - `data/validation/rolling_origin_policy_candidate_screen_2026-05-08.json`
  - `data/validation/state_segment_policy_candidate_screen_2026-05-08.json`
- **Provenance**:
  - `data/hetzner_results/phase_c_pooled_policy_profiles_2026-05-07/`
  - `data/oci_results/phase_c_pooled_policy_profiles_2026-05-07/`

## Bottom Line

This cycle did not produce a deployable BTS policy candidate.

The original pooled-policy signal did not survive a real temporal split. Local
follow-ups found useful diagnostics and a few small segment-level leads, but the
final FDR-controlled segment screen landed in the pre-committed E3 state:
too many survivors with fold heterogeneity. That is not a clean freeze for a
fresh lockbox candidate and not a deployment claim.

The correct next move is to stop candidate generation on the consumed
2021-2025 surface. Any future policy candidate needs a new pre-registration
that fixes the candidate, the family-control rule, and the fresh evaluation
target before looking at new outcomes.

## Infrastructure Recap

The lasting deliverable is not the pooled-policy candidate. It is the audit
infrastructure built and exercised during the PR #26 through Phase D cycle:

- SOTA closeout methodology and documentation for the real split audit path.
- Split-aware experiment-runner and audit-driver plumbing for provider,
  determinism, run-kind, queue-mode, and seed provenance.
- Hetzner + OCI Phase C profile generation with per-seed
  `audit_validation_split.json` metadata recording provider, box, region,
  run kind, queue mode, and determinism intent.
- OCI inclusion-rule verification across Phase B canaries and the Phase C
  production-sized run.
- `audit_attach` recovery tooling for `--run-kind profiles` and exact
  `--only-box` recovery, locally tested but not invoked because Phase C
  completed cleanly inside the 12-hour cap.
- Profile-block bootstrap work deferred but unblocked for a future audit cycle.

Cycle cloud spend was approximately `$30-35` of the `$1000` authorized cap:
about `$0.32` for the initial OCI canary, about `$0.80` for the multi-AD OCI
scaling canary, roughly `$5-8` for Hetzner Phase C, and roughly `$22-25` for
OCI Phase C r2.

## What Was Proven

### Phase C Surface

The Phase C 100-seed Hetzner + OCI profile surface completed cleanly:

- Hetzner: 48 seeds
- OCI: 52 seeds
- Combined profile seasons: `2021-2025`
- Provider tags preserved for downstream sensitivity checks
- No provider artifact was found in the Phase D result: Hetzner and OCI gaps
  agreed closely.

This means the subsequent falsification is not plausibly explained by one cloud
provider behaving differently.

### Phase D Temporal Split

The split-aware outer evaluation tested the deploy-relevant question:

- Build pooled candidate from selection seasons `2021-2024`
- Evaluate fixed shipped production and pooled candidate on outer season `2025`
- Preserve seed/provider pairing

Result:

| Metric | Value |
| --- | ---: |
| Production mean P(57), 2025 | `0.127678` |
| Pooled candidate mean P(57), 2025 | `0.064691` |
| Mean gap, pooled minus production | `-0.062987` |
| Provider-stratified bootstrap CI | `[-0.065250, -0.060757]` |
| Positive seeds | `0/100` |
| Hetzner mean gap | `-0.062095` |
| OCI mean gap | `-0.063810` |

The plain pooled-policy candidate is falsified under the temporal split.

Claude's methodology objection was resolved before treating this as the cycle
result:

- production was the shipped `data/models/mdp_policy.npz`, not rebuilt per seed;
- pooled policy construction used only `2021-2024`;
- evaluation used only `2025`;
- the old 24-seed surface is also negative under the same temporal split.

The earlier positive C0 result from PR #31 reported a `+1.929pp` paired-seed
gap on the 24-seed pool with `24/24` positive seeds and
`exact_sign_p=1.19e-7`. That result and this Phase D result answer different
questions. Phase D is the stricter deployment-relevant estimand.

## What Was Diagnosed

The post-mortem found a real 2025 policy-manifold shift:

| Policy reference | Selection, 2021-2024 | Outer, 2025 | Deployable? |
| --- | ---: | ---: | --- |
| Shipped production policy | `0.067049` | `0.125353` | current reference |
| Selection pooled candidate | `0.096378` | `0.060968` | no, falsified |
| Full 2021-2025 pooled policy | `0.080851` | `0.290015` | no, uses outer year |
| 2025-only hindsight oracle | `0.058768` | `0.357596` | no, uses outer year |

The post-mortem aggregates differ slightly from Phase D's outer-eval aggregates
because the scripts aggregate the outer surface differently. Both confirm the
same direction by wide margins and do not change the falsification verdict.

The hindsight rows show that 2025 admits high-value policy behavior on its own
manifold, but they are not deployment evidence. They explain why the surface is
interesting, not what should be deployed.

## Candidate Screens

### Rolling-Origin Recency Screen

The first rolling-origin screen tested cumulative, last-season, and exponential
decay pooled candidates. All were negative overall versus the fixed production
reference:

| Candidate | Overall mean gap | Positive seed-folds |
| --- | ---: | ---: |
| `cumulative_pooled` | `-0.040012` | `116/400` |
| `decay_half_life_2` | `-0.041537` | `130/400` |
| `decay_half_life_1` | `-0.043682` | `132/400` |
| `last_season_pooled` | `-0.078078` | `87/400` |

### Production-Anchored Hybrid Screen

The hybrid extension tested full, late-only, and early-only production-anchored
variants. Overall rolling-origin remained negative. The best overall candidate
was still below production:

| Candidate | Overall mean gap | 95% bootstrap CI | Positive seed-folds |
| --- | ---: | ---: | ---: |
| `prod_early_cumulative_late` | `-0.016812` | `[-0.020424, -0.013204]` | `192/400` |

The consumed 2025 fold did show a post-hoc early-only signal:

| Candidate | 2025 mean gap | 95% bootstrap CI | Positive seeds |
| --- | ---: | ---: | ---: |
| `cumulative_early_prod_late` | `+0.038044` | `[+0.031571, +0.044627]` | `88/100` |

That signal justified one final segment screen. It did not justify deployment.

### State-Segment FDR Screen

The final screen used a fixed 45-cell family:

- days remaining: `1-30`, `31-90`, `91-180`
- streaks: `0-9`, `10-29`, `30-56`
- quality bins: `Q1`, `Q2`, `Q3`, `Q4`, `Q5`

It applied one-sided positive bootstrap p-values and BH/BY across the family.

Result:

| Metric | Value |
| --- | ---: |
| Tested hypotheses | `45` |
| BH survivors at q<=0.05 | `9` |
| BY survivors at q<=0.05 | `9` |
| Stop-state | `E3_over_survival_revisit_family_control_before_conclusions` |

Several survivors were not temporally stable. Only three had positive mean gaps
in all four folds:

| Candidate | Overall mean gap | Worst fold mean gap |
| --- | ---: | ---: |
| `cumulative__mid_d31_90_s10_29_q5` | `+0.000999` | `+0.000269` |
| `cumulative__early_d91_180_s0_9_q5` | `+0.000702` | `+0.000007` |
| `cumulative__mid_d31_90_s0_9_q5` | `+0.000241` | `+0.000014` |

Those are research leads only. The registered stop rule says 4+ FDR survivors
requires revisiting family control before conclusions. It does not allow a
post-hoc freeze of the three stable-looking rows.

## Cycle Decision

The cycle closes as:

> `cycle_closed_no_deployable_candidate`

This means:

1. Do not deploy the pooled policy.
2. Do not deploy a broad early-only hybrid.
3. Do not deploy a segment patch from the consumed 2021-2025 surface.
4. Do not run another candidate-generation slice on 2021-2025.
5. Preserve the diagnostics as inputs to a future pre-registration.

## Future Work

A future fresh-lockbox audit can still be worthwhile, but only after a new
pre-registration fixes:

- the exact candidate policy construction;
- the tested family and FDR rule;
- the fresh evaluation target;
- the cloud budget and provider split;
- the acceptance/rejection thresholds.

Given the current evidence, the next audit should not be framed as confirmation
of the pooled-policy cycle. It should be framed as a new cycle with a new,
pre-registered candidate and no reuse of 2021-2025 as fresh evidence.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/scripts/test_state_segment_policy_candidate_screen.py \
  tests/scripts/test_rolling_origin_policy_candidate_screen.py \
  tests/scripts/test_phase_d_pooled_policy_postmortem.py \
  tests/scripts/test_phase_d_pooled_policy_outer_eval.py \
  tests/scripts/test_pooled_policy_gap_ci.py \
  tests/simulate/test_pooled_policy.py

git diff --check

/Users/stone/agent-room/bin/agent-bts-review diff-check
```
