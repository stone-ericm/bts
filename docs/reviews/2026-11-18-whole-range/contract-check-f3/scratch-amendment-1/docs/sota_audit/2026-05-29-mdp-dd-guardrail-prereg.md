# MDP Double-Down Guardrail Pre-registration (2026-05-29)

**Status:** pre-registration only. No production behavior change, no policy
artifact write, no script result, and no deploy claim.

## Purpose

The 2026-05-28 live aggressiveness audit found a concrete production symptom:
the deployed MDP probability boundaries do not distinguish the current live
pick stream. All 61 production primary picks through 2026-05-28 map to Q0, and
the saved policy chooses `double` for all 61 at low-streak states 0 and 4 with
the streak saver available. Thirty of those 61 independent pair probabilities
are below `0.55`; the recent 21-pick window has 9 such days.

This memo freezes the next measurement before any behavior change. The question
is whether a reversible double-down floor guardrail improves streak outcomes
enough to justify a later, config-gated implementation.

## Non-goals

This work does not:

- change `data/models/mdp_policy.npz`;
- re-solve the MDP;
- change probability boundaries;
- change primary pick ranking or lock timing;
- change Bluesky posting or DM delivery;
- deploy anything; or
- use 2026 production outcomes to tune the guardrail floor.

The guardrail is a live-path patch for the current bottom-collapsed probability
regime, not a replacement for Gate B reconciliation.

## Candidate Definition

`CURRENT` is the deployed strategy:

- pick the same primary candidate as production would pick;
- use the deployed MDP action table and deployed boundaries;
- if the MDP action is `double`, use the first eligible double-down candidate
  from a different game;
- otherwise keep the MDP action unchanged.

`GUARDRAIL(floor)` is an overlay on `CURRENT`:

- first compute the exact `CURRENT` action;
- if `CURRENT` is not `double`, make no change;
- if no eligible different-game double-down candidate exists, make no change;
- compute `p_both = primary_p * double_down_p`;
- if `p_both < floor`, downgrade the final action from `double` to `single`;
- otherwise keep `double`.

Frozen floor sweep:

- `0.40`
- `0.50`
- `0.55`
- `0.60`

The production Q0 collapse is motivation and diagnostic context only. Q0 must
not be part of the guardrail trigger unless a later amendment proves the
backtest probability scale reproduces the production scale. Triggering on Q0 in
production but not in backtest would test a different intervention from the one
deployed. The comparable trigger is the direct pair probability floor.

## Required Data Surface

The primary backtest requires one daily ranked candidate surface per evaluated
season with at least:

- `date`;
- `season`;
- `rank`;
- `p_game_hit`;
- `actual_hit`;
- a stable game identifier such as `game_pk`; and
- enough rows per day to choose the first eligible different-game double-down
  candidate.

The existing `data/simulation/backtest_2021.parquet` through
`data/simulation/backtest_2025.parquet` files are useful diagnostics but are
not sufficient for the primary deployment gate because they lack `game_pk`.
Those files may support a `rank2_proxy` diagnostic only. A result that cannot
enforce the different-game rule is not allowed to advance to implementation.

Any rebuilt profile surface must be frozen before result inspection:

- exact generation command;
- git SHA;
- input data directory;
- profile output path;
- profile schema;
- row counts by season and date;
- null counts for required fields; and
- SHA-256 for each profile parquet.

## Trigger-Overlap Representativeness Gate

The schema gate above is necessary but not sufficient. The backtest surface
must also enter the same trigger regime as production often enough to test the
guardrail. This is a representativeness gate, not a statistical-power
calculation: the primary evaluator is deterministic dynamic programming, so the
question is whether the intervention is exercised across enough distinct
season contexts to generalize.

For each floor, report on the primary surface:

- total double-eligible days;
- total double-eligible days with `p_both < floor`;
- fraction of double-eligible days with `p_both < floor`;
- number of seasons with at least five trigger days; and
- the same trigger counts for the diagnostic rank-2 proxy if the primary
  surface is unavailable.

A floor has sufficient trigger overlap only if it has at least 30 primary
trigger days and at least three seasons with five or more trigger days. If this
condition fails, the floor is labeled `UNDERPOWERED_TRIGGER_OVERLAP`, not
`NO_EFFECT`. That distinction matters: `NO_EFFECT` means the guardrail would
rarely fire in the evaluated regime, while `UNDERPOWERED_TRIGGER_OVERLAP` means
the evaluated regime does not test the production symptom.

Known diagnostic context before any new result: the existing rank-2 proxy
surface has much higher pair probabilities than production. Across
`data/simulation/backtest_2021.parquet` through `backtest_2025.parquet`, the
proxy `p_both` mean is about `0.657`; only `3 / 912` days are below `0.55` and
`65 / 912` are below `0.60`. Production through 2026-05-28 has mean `p_both`
about `0.544`. This is why the primary gate must report trigger overlap before
any metric interpretation.

## Evaluation Design

Evaluate seasons `2021, 2022, 2023, 2024, 2025` if the required surface exists
for all five seasons. If any season is missing the required fields, the primary
result is invalid and the run is diagnostic only.

For each season and each arm (`CURRENT`, `GUARDRAIL(0.40)`,
`GUARDRAIL(0.50)`, `GUARDRAIL(0.55)`, `GUARDRAIL(0.60)`):

1. Use the same primary ranked candidate stream.
2. Use the same first eligible different-game double-down candidate stream.
3. Evaluate the deployed MDP action at each state using the deployed action
   table, deployed boundaries, days remaining, streak, and saver state.
4. Apply the candidate overlay only after the deployed MDP action is known.
5. Recompute the full streak trajectory distribution, not a per-day swap count.

The preferred evaluator is exact dynamic programming over the historical daily
row stream. For each day, the evaluator applies the state-dependent action and
uses the row-level probabilities for transitions:

- `skip`: streak and saver state hold;
- `single`: hit probability is `primary_p`;
- `double`: hit probability is `primary_p * double_down_p`;
- saver rules match production and the existing MDP state model.

The evaluator must reuse the same transition kernel as
`bts.simulate.mdp.solve_mdp` instead of re-deriving similar logic by hand. In
particular, saver protection is active only when `10 <= streak <= 15`, and it
protects a missed `double` the same way it protects a missed `single`: the
streak holds and the saver is consumed. This requirement should be covered by
unit tests before a sweep is run.

Because `solve_mdp` currently keeps the transition logic inline, the evaluator
implementation should first extract a shared single-day transition helper and
make both `solve_mdp` and the guardrail evaluator call it. A test must assert
that the evaluator transition matches the solver transition across sampled
`(streak, saver, action, p)` states.

For reach target `k`, compute `P(reach >= k)` by carrying state mass through
the ordered season as first passage into the absorbing set `{streak >= k}`.
This must count a `double` jump that crosses a threshold without landing
exactly on it, for example `8 -> 10` crosses `k = 9`. Compute capped
`E[max streak]` as the layer-cake sum of `P(reach >= k)` for `k = 1..57`. This
avoids Monte Carlo noise and preserves the path-dependent effect of downgrading
doubles to singles.

The evaluator test suite must include an explicit first-passage case where a
`double` jump crosses a reach threshold, such as `8 -> 10` counting as
`P(reach >= 9)`.

If an exact row-stream evaluator does not exist, implement it first with unit
tests before running the sweep. A Monte Carlo implementation may be reported as
a robustness check, but it is not the primary decision surface unless this memo
is amended before results are computed.

## Metrics

Primary metric:

- `E[max streak]`

Support ladder:

- `P(reach >= 10)`
- `P(reach >= 20)`
- `P(reach >= 30)`

Diagnostics:

- `P(reach >= 40)`
- `P(reach >= 57)`
- count of changed decisions by season and floor;
- changed-decision directions, which should be only `double -> single`;
- distribution of `p_both` among changed and unchanged double calls;
- production Q-bin distribution among changed days, reported only as context;
- deployed policy Q0 `p_both` transition value versus production `p_both`
  distribution;
- realized replay longest streak by season, using `actual_hit`, diagnostic only.

p_both is computed as `primary_p * double_down_p`, which assumes independence.
That is consistent across `CURRENT` and `GUARDRAIL`, so it should not bias the
arm gap, but the floor is an independent-pair proxy rather than a true joint
probability floor.

Floor guard:

- if `CURRENT` has `P(reach >= k) < 1e-3` for a support-ladder rung in any
  evaluated season, that rung is demoted to diagnostic for that season;
- demoted rungs cannot support or reject a floor;
- `E[max streak]` is always active.

## Decision Rule

A floor is `INVALID_PRIMARY_SURFACE` if the primary surface lacks required
fields, cannot enforce different-game double-down, or fails schema validation.

A floor is `UNDERPOWERED_TRIGGER_OVERLAP` if it lacks the trigger support
defined above. In that case, the calibrated historical backtest cannot provide
a benefit proof for the production collapsed-regime patch. It can only provide
a no-harm screen if the floor still has enough changed decisions to evaluate
harm.

A floor is `NO_EFFECT` only if the primary surface is valid, trigger overlap is
sufficient, and the guardrail changes fewer than 10 total decisions or changes
decisions in fewer than two seasons.

For floors with sufficient trigger overlap, a positive benefit screen requires
all of the following:

- aggregate mean gap in `E[max streak]` is positive
  (`GUARDRAIL - CURRENT`);
- `E[max streak]` is negative in no more than one evaluated season;
- every active support-ladder rung has non-negative aggregate mean gap;
- no active support-ladder rung is negative in more than one evaluated season;
- no season has a catastrophic regression, defined as
  `E[max streak] <= CURRENT - 0.25`; and
- the positive aggregate result is not driven solely by one season, shown by
  per-season reporting.

A floor is `MIXED` if:

- `E[max streak]` aggregate mean is positive but has two or more negative
  seasons;
- an active support-ladder rung has negative aggregate mean but the headline
  metric is positive; or
- the only positive floor is near the no-effect support boundary.

A floor is `REJECT` if:

- aggregate mean gap in `E[max streak]` is negative; or
- any season has a catastrophic regression.

A floor is `NO_HARM_SCREEN` if it has underpowered trigger overlap and all of
the following hold:

- it changes at least 10 total decisions and changes decisions in at least two
  seasons;
- no catastrophic regression;
- aggregate mean gap in `E[max streak]` is at least `-0.10`;
- `E[max streak]` negative in no more than one evaluated season; and
- no active support-ladder rung has aggregate mean gap below `-0.005` or a
  relative gap below `-20%` versus `CURRENT`.

The no-harm pathway does not prove benefit. It says only that the guardrail did
not materially damage the calibrated historical regime. If production-trigger
evidence remains active, a no-harm result may justify a later default-off
implementation PR for explicit human review, but it does not authorize enabling
the guardrail automatically.

If multiple floors have positive benefit screen status, the only floor that may
advance to implementation design is the smallest passing floor. This
pre-registered tie-break minimizes live behavior change and prevents selecting
the most aggressive floor after seeing the result.

If no floor has either a positive benefit screen or an underpowered no-harm
screen, no guardrail implementation should be built from this evidence.

The principled benefit test is a regime-matched backtest that maps historical
probabilities onto the current production probability scale before applying the
floor. That test is blocked until Gate B reconciliation has enough
complete-fingerprint production support to estimate the map. Until then, the
calibrated historical test must not be described as a full benefit proof for
the collapsed production regime.

## Result Interpretation

A positive benefit screen or no-harm screen does not authorize deploy. It only
permits a later implementation PR for a default-off, config-gated guardrail.

Any implementation PR must include:

- config flag defaulting off;
- floor value defaulting to the smallest positive-benefit passing floor from
  this sweep;
- if the only advancing evidence is `NO_HARM_SCREEN`, no floor value may be
  enabled without a separate explicit human decision that acknowledges the
  backtest did not prove benefit in the production collapsed regime;
- persisted decision metadata on every pick:
  - pre-decision streak;
  - saver availability;
  - days remaining;
  - raw MDP action;
  - raw MDP quality bin;
  - primary probability;
  - double-down probability;
  - `p_both`;
  - guardrail floor;
  - final action; and
  - override reason;
- unit tests for no-op, downgrade, missing double-down, and metadata cases;
- review-before-merge; and
- paired deploy with live verification if the flag is ever enabled.

## Expiration Rule

This guardrail is explicitly a collapsed-regime patch. It must be re-evaluated
or removed when Gate B reconciliation completes, when production picks no
longer collapse into Q0 under the active probability scale, or when a corrected
policy artifact is deployed.

The implementation, if built, should therefore include an operational reminder
or health-note path tying the enabled guardrail to Gate B reconciliation status.

## Required Reporting Template

The result artifact must report:

- input profile paths and SHA-256 values;
- profile schema validation result;
- exact git SHA and command;
- current deployed policy SHA or artifact hash;
- backtest-versus-production `p_both` distribution comparison;
- trigger-overlap table by floor;
- floor sweep table;
- per-season metric table;
- changed-decision support table;
- diagnostic Q-bin table;
- realized replay diagnostic table;
- final decision label for each floor:
  - `NO_EFFECT`;
  - `UNDERPOWERED_TRIGGER_OVERLAP`;
  - `NO_HARM_SCREEN`;
  - `POSITIVE_SCREEN`;
  - `MIXED`;
  - `REJECT`; or
  - `INVALID_PRIMARY_SURFACE`; and
- the pre-registered smallest-passing-floor selection, or `none`.

No result should be interpreted without the schema validation and changed-
decision support tables.
