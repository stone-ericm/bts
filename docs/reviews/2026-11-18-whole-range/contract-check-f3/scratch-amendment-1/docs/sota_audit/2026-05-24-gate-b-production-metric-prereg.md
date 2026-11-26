# Gate B Production-Metric Re-baseline Pre-registration (2026-05-24)

**Status:** pre-registration only. No production behavior change, no policy
artifact write, no script, no deploy claim.

## Purpose

The Gate B fair-comparator re-baseline isolated the useful signal: the deployed
MDP boundary scale is out of domain for production-style estimated-PA
probabilities, while re-solving the action table added mixed and sign-flipping
value.

This pre-registration defines the next measurement before any new result is
computed. The goal is to test the boundary-scale lever under production-facing
metrics, without changing the deployed action table and without relying on
floor-level P(57) alone.

## Comparator

The comparison is boundary-only.

`CURRENT`:

- deployed `data/models/mdp_policy.npz` action table;
- deployed saved boundaries from that artifact;
- production-style estimated-PA probabilities classified through those deployed
  boundaries.

`CANDIDATE`:

- the same deployed `data/models/mdp_policy.npz` action table;
- production-scale estimated-PA boundaries fit without holdout leakage;
- the same production-style estimated-PA probabilities classified through the
  corrected boundaries.

No measurement in this slice may re-solve the action table. Re-solving remains
out of scope unless a separate pre-registration reopens it.

## Claims

The re-baseline must keep two claims separate.

### Mechanism Claim

This is necessary but not sufficient for a policy swap.

Question:

Does the boundary-only candidate make the decision layer discriminate on the
production pick stream, compared with the current bottom-collapsed boundary
projection?

Mechanism evidence may be computed on canonical 2026 production picks and
scheduler snapshots.

Report:

- recent bin occupancy under `CURRENT` versus `CANDIDATE`;
- whether `mdp_policy_alignment` would clear under candidate boundaries;
- action counts by arm: `skip`, `single`, `double`;
- changed decision days;
- changed decision direction, including `skip -> pick`, `pick -> skip`,
  `single -> double`, and `double -> single`;
- examples of changed days for auditability.

Pass condition:

- candidate primary-pick bin occupancy must not trigger the existing
  `mdp_policy_alignment` dominance rule: with the default 21 recent production
  picks and minimum 14-pick support, no single candidate bin may contain
  `>= 0.80` of primary picks; and
- the candidate must change at least one production decision in a way that is
  attributable only to the boundary scale.

A passing mechanism result means only that the boundary fix has operational
effect. It does not justify a policy artifact swap.

### Outcome-Quality Claim

This is the sufficient evidence branch for advancing toward a reversible
policy artifact candidate.

Question:

When both arms keep the deployed action table fixed, does the boundary-only
candidate improve streak outcomes on held-out estimated-PA profile seasons?

Primary evaluation surface:

- use the same estimated-PA profile basis as Gate B;
- use expanding-origin folds with prior seasons for boundary fitting and the
  next season as holdout;
- require `p_game_hit_basis == "estimated_pa"` on all input profiles;
- reject inputs that write production artifacts or set
  `production_deploy_claim=true`.

Primary outcome metrics:

- headline metric: `E[max streak length]`;
- support ladder: `P(reach streak >= 10)` within the season;
- support ladder: `P(reach streak >= 20)` within the season;
- support ladder: `P(reach streak >= 30)` within the season;
- support ladder: `P(reach streak >= 40)` within the season.

These metrics replace floor-level P(57) as the primary outcome-quality ladder.
P(57) may still be reported as a diagnostic because the MDP was optimized for
that target, but it must not be the sole gate.

Floor guard:

- if the `CURRENT` arm value for a support-ladder rung is `< 1e-3`, that rung
  must be demoted to diagnostic for the decision rule;
- the report must state which rungs remain active after the floor guard;
- a demoted rung may not be used to support or reject the candidate.

The ladder rungs are correlated views of the same policy behavior. They are not
independent confirmations. `E[max streak length]` is the headline metric
because it is less floor-sensitive than tail reach probabilities.

Positive screen condition:

- the headline `E[max streak length]` metric has a positive aggregate mean gap
  (`CANDIDATE - CURRENT`);
- the headline aggregate mean gap either exceeds the fold-to-fold standard
  deviation of the gap or has a bootstrap confidence interval with lower bound
  greater than zero;
- every active non-floor support-ladder rung has a positive aggregate mean gap
  (`CANDIDATE - CURRENT`);
- no active outcome metric is negative in more than one holdout fold; and
- the result is not driven solely by one outlier season, as shown by fold-level
  reporting.

Mixed screen condition:

- the headline metric has a positive aggregate mean gap but fails the
  fold-stability or bootstrap-CI bar;
- aggregate means are positive for some active support-ladder rungs but not
  all; or
- any active outcome metric has two or more negative folds.

Reject condition:

- the headline metric has a negative aggregate mean gap; or
- any active non-floor support-ladder rung has a negative aggregate mean gap.

Only a positive screen may advance the boundary-only candidate to the next
artifact-design step. Even then, it remains non-deploy evidence until the full
production gate passes.

## 2026 Realized Pick Outcomes

Resolved 2026 production picks are descriptive only for this slice.

They may be reported to show how changed decisions would have resolved where
the necessary slot-level outcomes are available, but they must not override the
multi-season outcome-quality gate.

No directional realized-outcome claim may be made unless all of the following
are true before looking at the result:

- at least 30 resolved changed-decision days are available;
- at least 10 resolved changed days move from pick to skip or skip to pick;
- double-down comparisons have slot-level outcomes for every changed slot;
- uncertainty is reported with an interval, not just a point estimate.

If those support thresholds are not met, 2026 realized outcomes remain
corroborating examples and failure analysis only.

## Non-goals

This work does not:

- change `data/models/mdp_policy.npz`;
- generate a replacement policy artifact;
- re-solve the action table;
- change production pick selection, posting, or DM delivery;
- deploy anything;
- claim full production replay validity.

The decision layer is evaluated on a fixed selected-pick stream. This measures
the MDP decision layer, not upstream candidate generation or lineup selection.

## Required Follow-on Gate

A positive production-metric re-baseline still does not authorize deploy.

Any future production policy change must separately include:

1. leakage audit;
2. nuclear test;
3. reversible artifact generation with the prior artifact preserved;
4. explicit before/after health expectations, including MDP alignment;
5. deploy-gated rollout with rollback instructions; and
6. live production verification after deploy.

Until those steps pass, the only allowable conclusion is whether the
boundary-only candidate deserves the next artifact-design step.
