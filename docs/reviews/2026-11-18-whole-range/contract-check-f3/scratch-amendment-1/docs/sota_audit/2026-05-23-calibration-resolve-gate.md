# Calibration / MDP Re-solve Gate (2026-05-23)

**Status:** diagnostic and gating memo only. No production behavior change.

## Why this memo exists

The 2026-05-23 C+F preflight found two live production facts that look related
but must not be collapsed into one unvalidated fix:

1. Production pick probabilities are modestly overconfident.
2. The deployed MDP policy no longer distinguishes current production pick
   probabilities: all recent picks map to Q0 against the saved policy
   boundaries.

PR #109 added `mdp_policy_alignment` as a WARN-only health guardrail for the
second fact. This memo records the evidence and the gate for any future
behavioral change.

## Current live evidence

Live prod at merge/deploy SHA `783986e`:

- Current MDP policy boundaries: `[0.795979, 0.811491, 0.825247, 0.840740]`.
- All 57 production primary pick days in the checked range mapped to Q0.
- Recent 21 production primary picks: `21/21` Q0, `21/21` below the lowest
  boundary, p range `0.690-0.788`.
- Recent 21 double-down picks: `21/21` Q0, p range `0.686-0.780`.

Resolved production-pick calibration check, 90-day lookback ending
2026-05-23:

- `n=158` resolved pick samples (primary plus double-down slots).
- Raw mean p: `0.7336`.
- Realized hit rate: `0.6519`.
- 5-fold isotonic Brier, raw: `0.235632`.
- 5-fold isotonic Brier, calibrated: `0.236824`.
- Brier improvement: `-0.001192`.

Resolved double-down pair check:

- `n=49` resolved double-down pairs.
- Mean independent p_both: `0.5441`.
- Realized both-hit rate: `0.4490`.
- Last 14 pairs: mean p_both `0.5469`, realized both-hit `0.2857`; this is
  directionally concerning but too small for a behavior change by itself.

## Decision now

Do not enable `BTS_USE_CALIBRATION`.

Do not swap or regenerate the production `mdp_policy.npz`.

The isotonic map fails the immediate proper-score check: out-of-fold Brier is
slightly worse than raw probabilities. The current evidence supports continued
monitoring and better validation tooling, not a live calibration or policy
change.

## Gate A: probability calibration

Question: should production apply a post-hoc calibration map before strategy
selection and MDP lookup?

Required gate:

1. Calibration data must be leakage-free. The map must be fit on temporally
   clean, out-of-sample predictions.
2. The target must be explicit. The MDP consumes game-level `p_game_hit`; a
   PA-level fit may be explored for volume, but it must be verified against the
   game-level probability that the policy actually sees.
3. Minimum resolved sample count: `n >= 200` pick-slot outcomes before any
   ship/no-ship verdict.
4. Proper-score decision: cross-fitted calibrated probabilities must improve
   Brier score, and the bootstrap 95% CI on Brier improvement must exclude zero.
5. If the calibration map requires model retraining or feature changes, run the
   leakage audit and nuclear test before interpreting any validation result.

If Gate A fails, calibration stays off even if nominal probabilities look
overconfident.

## Gate B: MDP policy re-binning / re-solve

Question: should the MDP policy be re-binned and re-solved on the current raw
probability distribution, even without a shipped calibration map?

This is independently testable. The Q0 collapse is a policy-scale mismatch,
not automatically proof that probability calibration should ship.

Required gate:

1. Build a candidate policy artifact separate from `data/models/mdp_policy.npz`.
2. Regenerate quality bins on the candidate probability distribution and solve
   the same reachability MDP. The solver objective remains P(57).
3. Evaluate the candidate policy against the current baseline with the
   multi-season walk-forward / policy-file harness. The candidate must meet or
   beat the current baseline P(57); a "more honest" or "better aligned" policy
   that lowers P(57) does not ship.
4. Keep the current policy artifact reversible until the candidate clears the
   P(57) gate and gets explicit deployment approval.

If Gate B clears before Gate A, a raw-distribution policy re-bin can be
considered as a policy-only change. If it fails, the calibration/re-solve
coupling remains the safer default.

## Staging plan

1. Validation tooling only: make `scripts/validate_calibration.py` re-runnable
   against current picks, PA data, dates, lookbacks, and output paths.
2. Calibration evidence PR: when `n >= 200`, run Gate A and record the JSON
   artifact. No production behavior in this step.
3. Policy evidence PR: evaluate Gate B using a separate candidate `.npz` and
   policy-file P(57) harness. No production behavior in this step.
4. Wiring PR only if the relevant evidence gate clears: apply the calibration
   map and/or swap the policy artifact with a documented rollback path.

PR #109's `mdp_policy_alignment` WARN should auto-clear only after the deployed
policy boundaries again discriminate recent production pick probabilities.
