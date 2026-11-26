# Pooled-Policy Post-Mortem And Next Candidates

- **Generated**: 2026-05-08
- **Status**: `postmortem_complete_next_candidate_plan`
- **Artifact**: `data/validation/phase_d_pooled_policy_postmortem_2026-05-08.json`
- **Generator**: `scripts/phase_d_pooled_policy_postmortem.py`
- **Production deploy claim**: `false`

## Verdict

Plain pooled-policy is not the next deployable BTS lever. The temporal split
audit did what it was supposed to do: it separated a seed-axis positive screen
from a calendar-time deployment claim, and the candidate failed the latter.

The failure mode is not provider noise. It is a temporal policy-manifold shift:
the 2021-2024 pooled policy improves its selection manifold, but it carries the
wrong action shape into 2025.

## Diagnostic Summary

Policy values on the pooled 100-seed surfaces:

| Policy reference | Selection, 2021-2024 | Outer, 2025 | Deployable? |
| --- | ---: | ---: | --- |
| Shipped production policy | `0.067049` | `0.125353` | current reference |
| Selection pooled candidate | `0.096378` | `0.060968` | no, falsified |
| Full 2021-2025 pooled policy | `0.080851` | `0.290015` | no, uses outer year |
| 2025-only hindsight oracle | `0.058768` | `0.357596` | no, uses outer year |

The hindsight rows are not deployment evidence. They show that 2025 admits a
high-value policy on its own manifold, while the 2021-2024 pooled policy is the
wrong fit for that manifold.

## Surface Shift

Raw rank-pair outcomes changed meaningfully between the selection and outer
surfaces:

| Metric | 2021-2024 selection | 2025 outer | Outer minus selection |
| --- | ---: | ---: | ---: |
| Rank-1 mean predicted hit probability | `0.819074` | `0.818382` | `-0.000693` |
| Rank-1 actual hit rate | `0.865440` | `0.886685` | `+0.021245` |
| Rank-1 mean-p minus actual | `-0.046365` | `-0.068303` | `-0.021938` |
| Rank-2 actual hit rate | `0.842157` | `0.843043` | `+0.000887` |
| Rank-1/rank-2 both-hit rate | `0.727555` | `0.741467` | `+0.013912` |

The early phase broadly improved in 2025, especially in lower bins. The late
phase shifted non-monotonically:

- early Q1 rank-1 hit rate rose from `0.765461` to `0.851948`
- early Q3 both-hit rate rose from `0.725822` to `0.795130`
- late Q1 rank-1 hit rate fell from `0.818333` to `0.680000`
- late Q2 both-hit rate fell from `0.648750` to `0.463333`
- late Q5 rank-1 hit rate fell from `0.965417` to `0.666667`

That late-phase inversion is the clearest post-mortem signal. The issue is not
that pooling is noisy; it is that static pooling across older seasons can learn
a phase/bin manifold that no longer describes the future season.

## Action Shape

Compared with production, the selection pooled candidate is materially less
single-heavy and more skip/double-heavy:

| Policy | Skip states | Single states | Double states |
| --- | ---: | ---: | ---: |
| Production | `54,244` | `29,069` | `19,287` |
| Selection pooled candidate | `59,240` | `20,889` | `22,471` |
| 2025 hindsight oracle | `51,014` | `27,957` | `23,629` |

The selection candidate differs from production in `13.63%` of decision states.
The largest changes are:

- production `single` -> pooled `double`: `5,398` states
- production `single` -> pooled `skip`: `5,207` states
- production `double` -> pooled `single`: `1,784` states

The 2025 hindsight oracle is still double-heavier than production, but it is not
as single-starved as the 2021-2024 selection candidate. This points toward a
drift-aware policy-shape problem, not a generic "more doubles" or "more skips"
rule.

## Next Candidate Plan

Do not launch another large pooled-policy confirmation run. The next work should
be local, diagnostic, and nested before any cloud spend:

1. **Build a rolling-origin policy evaluator.**
   Evaluate policy candidates on folds such as train `2021` -> holdout `2022`,
   train `2021-2022` -> holdout `2023`, train `2021-2023` -> holdout `2024`,
   and train `2021-2024` -> holdout `2025`. This makes temporal generalization
   the selection criterion instead of seed-axis LOO.

2. **Test recency-aware bin manifolds.**
   Candidate family: last-season-only bins, exponentially decayed season weights,
   recent-season-heavy shrinkage toward the shipped production table, and
   phase-specific recency weights. The gating metric is outer-fold P(57) versus
   shipped production, not in-sample optimal P(57).

3. **Treat late-phase bins as a separate failure mode.**
   The 2025 late bins are the sharpest shift. Candidate family should include
   late-phase-specific shrinkage, late-phase monotonicity checks, and a fallback
   that refuses to trust a high-p top late bin when recent evidence contradicts
   it.

4. **Re-run DR-MDP only on the temporal candidate manifold.**
   The previous DR-MDP screen did not justify solver changes from the canonical
   surface. If recency-aware binning creates a plausible candidate, use
   `scripts/dr_mdp_gap_measure.py` as a robustness screen before any solver
   implementation work.

5. **Reserve a fresh lockbox or live-forward target.**
   The 2025 outer year has now been consumed for candidate diagnosis. Any new
   candidate can use this post-mortem for generation, but deployment claims need
   a fresh lockbox/live-forward evaluation target and explicit go-ahead.

## Immediate Implementation Slice

The next code slice should be a local evaluator, not cloud orchestration:

- input: existing Phase C 100-seed raw profile surface
- output: `data/validation/rolling_origin_policy_candidate_screen_<date>.json`
- candidates: production, cumulative pooled, last-season pooled,
  exponential-decay pooled, and late-phase shrinkage variants
- folds: rolling-origin seasons, with train seasons strictly before holdout
- gates: paired seed-level P(57) gap, provider-tagged sensitivity when using
  Phase C roots, proper-scoring diagnostics for rank-1 and late-phase bins, and
  no deploy-ready verdict from this artifact alone

Only after that local screen identifies a temporally robust candidate should we
consider cloud re-runs or a new lockbox audit.

## First Rolling-Origin Screen

The immediate local evaluator was implemented as
`scripts/rolling_origin_policy_candidate_screen.py`.

Artifact:
`data/validation/rolling_origin_policy_candidate_screen_2026-05-08.json`.

Candidate family:

- `cumulative_pooled`
- `last_season_pooled`
- `decay_half_life_1`
- `decay_half_life_2`
- production-anchored late-only hybrids such as `prod_early_cumulative_late`
- production-anchored early-only hybrids such as `cumulative_early_prod_late`

Overall result across rolling-origin folds and 100 seeds:

| Candidate | Mean gap vs fixed production reference | Positive seed-folds |
| --- | ---: | ---: |
| `prod_early_cumulative_late` | `-0.016812` | `192/400` |
| `prod_early_decay_hl2_late` | `-0.019412` | `188/400` |
| `cumulative_early_prod_late` | `-0.020114` | `188/400` |
| `cumulative_pooled` | `-0.040012` | `116/400` |
| `decay_half_life_2` | `-0.041537` | `130/400` |
| `decay_half_life_1` | `-0.043682` | `132/400` |
| `last_season_pooled` | `-0.078078` | `87/400` |

Fold detail:

| Holdout | Best simple candidate | Best mean gap | Interpretation |
| --- | --- | ---: | --- |
| `2022` | `cumulative_pooled` / `last_season_pooled` | `-0.096437` | fixed production reference is very strong on this fold |
| `2023` | `decay_half_life_1` | `-0.001862` | near tie, no robust win |
| `2024` | `last_season_pooled` | `+0.005081` | simple recency helps here |
| `2025` | `cumulative_early_prod_late` | `+0.038044` | early-only cumulative pooled is positive post hoc |

This is a screen only. The shipped production reference is today's fixed
production policy, not a leak-free historical baseline for the early folds.
Even with that caveat, the result is useful:

- simple recency weighting and last-season-only pooling do not solve the 2025
  failure;
- replacing only late-phase production states is harmful on 2025;
- keeping production late states and replacing only early states with the
  cumulative pooled candidate is strongly positive on consumed 2025:
  mean gap `+0.038044`, 95% seed-bootstrap CI `[+0.031571, +0.044627]`,
  positive seeds `88/100`.

The last point is post-hoc candidate-generation evidence, not deployment
evidence. It changes the next hypothesis: production late-phase behavior may be
doing useful damage control, while early-phase production may leave value on the
table relative to a cumulative pooled policy.

Updated next-candidate direction:

1. Treat **production-anchored early-phase replacement** as the primary target,
   with production late-phase states preserved by default.
2. Treat late-phase bin behavior as a risk boundary: do not replace late states
   unless a future rolling-origin screen shows stable evidence.
3. Compare production versus 2025 hindsight-oracle action deltas to identify
   state regions where production is already right and where the oracle differs.
4. Design a production-anchored, state-segment candidate rather than a wholesale
   pooled-policy replacement.
5. Keep rolling-origin screening as the local selection gate before any cloud
   run or lockbox claim.

## Action-Delta Spot Check

A read-only state-grid comparison between production, the selection pooled
candidate, and the 2025 hindsight oracle gives one more constraint on the next
candidate. The oracle is not deployable, but its differences from production
show where a future candidate might look.

Largest production-to-2025-oracle changes by state segment:

- early days `91-180`, streak `31-45`, Q4: production `single` -> oracle
  `skip` in `2,111` states
- mid days `31-90`, streak `16-30`, Q4: production `single` -> oracle
  `double` in `1,295` states
- mid days `31-90`, streak `16-30`, Q2: production `single` -> oracle
  `double` in `1,193` states
- mid days `31-90`, streak `16-30`, Q1: production `skip` -> oracle
  `single` in `1,040` states

The selection pooled candidate shares some of those broad directions, but it is
more aggressive in the wrong places: its largest change is early days
`91-180`, streak `31-45`, Q4, production `single` -> pooled `skip` in `2,348`
states, and it is still far worse on 2025. A production-anchored successor
should therefore not blindly copy oracle deltas. It should restrict any action
change to segments that survive rolling-origin evidence.

## State-Segment Candidate Screen

The state-segment follow-up was implemented as
`scripts/state_segment_policy_candidate_screen.py`.

Artifact:
`data/validation/state_segment_policy_candidate_screen_2026-05-08.json`.

Memo:
`docs/sota_audit/2026-05-08-state-segment-policy-candidate-screen.md`.

The screen tested 72 production-anchored candidates: cumulative or decay-HL2
candidate actions copied into one coarse non-late segment, with shipped
production preserved everywhere else.

The FDR-controlled result is not a clean candidate freeze. The fixed
45-segment family produced `9/45` BH survivors and `9/45` BY survivors at
`q<=0.05`, which triggers the pre-committed E3 stop condition:
`E3_over_survival_revisit_family_control_before_conclusions`.

Several survivors are fold-heterogeneous. Only three have positive mean gaps in
all four folds:

- `cumulative__mid_d31_90_s10_29_q5`: overall mean gap `+0.000999`, worst fold
  `+0.000269`
- `cumulative__early_d91_180_s0_9_q5`: overall mean gap `+0.000702`, worst fold
  `+0.000007`
- `cumulative__mid_d31_90_s0_9_q5`: overall mean gap `+0.000241`, worst fold
  `+0.000014`

Updated candidate direction: stop candidate generation on the consumed
2021-2025 surface. Move to cycle synthesis and, if a future fresh-lockbox audit
is designed, pre-register both the candidate and the family-control rule before
looking at new outcomes.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/scripts/test_phase_d_pooled_policy_postmortem.py \
  tests/scripts/test_phase_d_pooled_policy_outer_eval.py

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/phase_d_pooled_policy_postmortem.py

UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/scripts/test_rolling_origin_policy_candidate_screen.py

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/rolling_origin_policy_candidate_screen.py

UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/scripts/test_state_segment_policy_candidate_screen.py

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/state_segment_policy_candidate_screen.py
```
