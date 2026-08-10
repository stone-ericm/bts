# 2026-08-09 — Boundary-only shadow measurement (pre-registration)

**Status: PRE-REGISTERED — committed before any census execution.** Measurement
only. No production change, no live knob, no promotion path is defined here;
any promotion discussion requires its own pre-registration (gate-B precedent:
boundary-scale correction carried most apparent value, but the historical
boundary artifact failed production transfer — 2026-05-24 docs — and nothing
here reopens that decision).

Provenance: authorized by the r3-corrected tripwire doc
(`docs/audit/2026-08-09-dd-tripwire-recompute.md`): the quality-bin collapse
is chronic since May (estimated-PA vs served-p scale mismatch; all-Q0
occupancy), possibly deepened by the 2026 ball-regime churn. Question: **if
ONLY the bin boundaries were rescaled to the currently-served p distribution
(action table untouched), which realized 2026 decisions would have changed —
and where in state space do the changes live?**

## Artifact under test

`B*` = equal-frequency quintile boundaries (quantiles .2/.4/.6/.8 — the same
scheme `compute_bins` used for the deployed set) over the served 2026
primary-p distribution. Two pre-registered variants, both reported:

- **B*_a**: `primary.p_game_hit` from all `bts_daily_decision_v1` records
  (2026-06-23 → 2026-08-09, the full decision-record era; n≈44).
- **B*_b**: rank-1 `p_game_hit` from all 2026 pick files (season-to-date;
  n≈130; includes pre-decision-era and preview-only days).

Disagreement between the two boundary sets is itself a reported result
(estimator stability under provenance choice). Both are snapshots of a
shifting regime; the harness parameterizes the window so re-runs are trivial,
and the snapshot dates + input hashes are fingerprinted in the artifact.

**Construction uses p values only — no outcomes.** The census consumes
outcomes solely for state reconstruction (streak/saver), which precedes each
day's decision and is not a look at the counterfactual's result.

## Census design (M1)

Era: 2026-06-23 → 2026-08-09, days with a `decision.json` whose
`source == "mdp"` (non-mdp days counted + listed, excluded from the diff
census).

Per-day state: `(streak_before, days_remaining, saver, primary_p)`.
- `streak_before`: replayed from season start using graded slot results +
  the saver rule (consume on first miss at streak 10–15). **Parity gate A:**
  replayed streak must equal `decision.json`'s `streak` field on every
  record where it is populated; any mismatch halts the census (state
  reconstruction bug, not data).
- `saver`: from the same replay.
- `days_remaining`: `SEASON_END_DATE − date` exactly as
  `strategy._mdp_action_from` computes it.
- `primary_p`: `decision.json` `primary.p_game_hit`.

**Parity gate B:** `lookup_action(table, deployed_boundaries, state)` must
reproduce the recorded deployed action on every mdp-source day (skip days
included). Any mismatch halts the census. Only after both gates pass do
counterfactual lookups run: `lookup_action(table, B*, state)` for each
variant.

Outputs (JSON artifact + audit doc):
- boundary sets (deployed, B*_a, B*_b) + input-window hashes + policy npz
  sha + git sha;
- per-day rows: state, deployed action, B*_a action, B*_b action;
- diff counts by streak band (0–2 / 3–7 / ≥8) and by direction
  (skip→single/double, double→single, etc.);
- retro outcome annotation on diff days (what the realized slot results
  imply the counterfactual would have scored) — **explicitly labeled
  descriptive/non-verdict**: n will be small, outcomes are single-trajectory,
  and this pre-registration draws no EV conclusion from them.

## Stopping rule / follow-up gate

- If diff days < 8 across both variants: the census stands alone as the
  result; no evaluation follow-up is designed from this spec.
- If diff days ≥ 8 in either variant: a follow-up *evaluation* design
  (realized-sequence replay per the 7/13 standard; iid DP values remain
  disallowed for policy comparison) gets its own pre-registration before
  anything is computed.

## Guardrails

- Harness is a versioned script (`scripts/audit/`), TDD'd; tests pin: the
  quintile computation against a hand-built frame, the streak replay against
  a constructed season (incl. saver consumption + void handling), both
  parity gates (positive + failure cases), and determinism (no
  `date.today()` defaults — explicit `today=`/window args per the repo's
  date-relative-test gotcha).
- No new dependencies; read-only over `data/picks`; artifact written under
  `data/validation/` with the standard fingerprint block.
- The census result, whatever it shows, changes nothing in production and
  does not by itself justify a boundary swap (see status paragraph).

## Non-goals

MDP re-solve; threshold changes; DD/`p_both` transition re-estimation; any
touch of `FEATURE_COLS`, the skip rule, or delivery; promotion criteria.
