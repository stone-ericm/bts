# State-Segment Policy Candidate Screen

- **Generated**: 2026-05-08
- **Status**: `diagnostic_complete_e3_over_survival`
- **Artifact**: `data/validation/state_segment_policy_candidate_screen_2026-05-08.json`
- **Generator**: `scripts/state_segment_policy_candidate_screen.py`
- **Production deploy claim**: `false`

## Purpose

The Phase D temporal split falsified wholesale pooled-policy replacement. The
first rolling-origin screen then showed a post-hoc 2025 lift for replacing
early states while preserving production late states, but the broad family still
failed across rolling-origin folds.

This screen was the final candidate-generation pass on the consumed 2021-2025
Phase C surface. Its job was not to approve a deploy. Its job was to decide
whether any coarse state segments should be frozen for a fresh lockbox or
whether the cycle should move to synthesis.

## Method

The screen is production-anchored. For each candidate, the shipped production
policy is kept everywhere except one pre-specified state segment. Inside that
segment, actions are copied from a train-fold cumulative pooled policy.

Inputs and folds:

- Profile surface: Phase C 100-seed Hetzner + OCI surface
- Reference: shipped `data/models/mdp_policy.npz`
- Folds: train prior seasons, hold out next season for `2022`, `2023`, `2024`,
  and `2025`
- Candidate base policy: cumulative pooled
- Bootstrap: 5,000 iid seed-fold resamples for reported mean-gap intervals and
  one-sided positive bootstrap p-values
- FDR: BH and BY over the 45 tested segment cells

Pre-specified segment grid:

- Days remaining: `1-30`, `31-90`, `91-180`
- Streaks: `0-9`, `10-29`, `30-56`
- Quality bins: `Q1`, `Q2`, `Q3`, `Q4`, `Q5`

That yields `m=45` production-anchored segment hypotheses.

## FDR Result

The screen found `9/45` positive segments surviving BH at `q<=0.05` and
`9/45` surviving BY at `q<=0.05`.

Per the pre-committed stop rule, this is **E3**:
`E3_over_survival_revisit_family_control_before_conclusions`.

The result is not a deployment claim and not a clean freeze of one to three
segments. It says the family/control design is not decisive enough for candidate
selection on this consumed surface.

FDR survivors:

| Candidate | Mean gap | p one-sided | q_BH | q_BY |
| --- | ---: | ---: | ---: | ---: |
| `cumulative__late_d1_30_s30_56_q2` | `+0.004288` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__late_d1_30_s30_56_q3` | `+0.001972` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__mid_d31_90_s10_29_q5` | `+0.000999` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__early_d91_180_s10_29_q5` | `+0.000877` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__early_d91_180_s0_9_q5` | `+0.000702` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__mid_d31_90_s0_9_q5` | `+0.000241` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__late_d1_30_s10_29_q2` | `+0.000112` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__mid_d31_90_s0_9_q1` | `+0.000042` | `0.000200` | `0.001125` | `0.004943` |
| `cumulative__late_d1_30_s10_29_q3` | `+0.000083` | `0.000600` | `0.002999` | `0.013182` |

## Temporal Stability

Several FDR survivors are not temporally stable. For example:

- `cumulative__late_d1_30_s30_56_q2` is `+0.017302` in holdout `2022`, but
  slightly negative in `2023`, `2024`, and `2025`.
- `cumulative__late_d1_30_s30_56_q3` is positive in `2022` and `2023`, but
  negative in `2024` and `2025`.
- `cumulative__late_d1_30_s10_29_q3` is positive in `2022` and `2023`, but
  negative in `2024` and `2025`.

Only three segments have positive mean gaps in all four folds:

| Candidate | Overall mean gap | Worst fold mean gap |
| --- | ---: | ---: |
| `cumulative__mid_d31_90_s10_29_q5` | `+0.000999` | `+0.000269` |
| `cumulative__early_d91_180_s0_9_q5` | `+0.000702` | `+0.000007` |
| `cumulative__mid_d31_90_s0_9_q5` | `+0.000241` | `+0.000014` |

That stability filter is useful for synthesis, but it was not the registered
family-control rule for this pass. It should not be retrofitted into a deploy
claim after seeing the screen.

## Interpretation

The honest conclusion is not "deploy a segment patch." It is:

1. Whole-policy pooled replacement failed the temporal split.
2. Simple recency and broad early/late hybrids failed as rolling-origin
   deployment candidates.
3. The fixed 45-cell segment family produced too many BH/BY survivors on the
   consumed surface, with clear fold heterogeneity.
4. The candidate-generation sub-cycle should stop here and move to synthesis.

If a future fresh-lockbox audit is designed from this evidence, it should be
designed before looking at new outcomes and should pre-register both the
candidate policy and the family-control rule. The consumed 2021-2025 surface is
now too mined for another candidate-generation pass.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/scripts/test_state_segment_policy_candidate_screen.py

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/state_segment_policy_candidate_screen.py
```
