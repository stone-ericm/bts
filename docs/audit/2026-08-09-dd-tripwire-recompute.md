# 2026-08-09 — DD-band leg-gap: exploratory interim look (n=57) + bin-collapse disposition

**Status: exploratory interim read. The pre-registered tripwire (7/13 doc: at
season n≈80–90 DD legs, ≥10pp shortfall → rerun
`scripts/audit/dd_p_policy_value_sensitivity.py` + take the re-solve question
seriously) was NOT evaluated tonight — n=57 is outside its window.** This look
was user-requested after the 8/09 `mdp_policy_alignment` quality-bin-collapse
WARN. One Codex adversarial round (r3, same night) reviewed the first draft;
its corrections are incorporated and listed in the review trail below.

**Commitment device (binding):** the formal read happens at the first
measurement snapshot with n≥80 season DD legs (~early-mid September at
current cadence); the ≥10pp threshold and the escalation action are unchanged
regardless of tonight's numbers; the formal read cannot be canceled or
delayed; no further interim reads before it. Tonight's tails are descriptive
only.

## Numbers (measurement = 7/12 recipe; per-slot grading, `slot_results`
authoritative, plus the 7/12 builder's legacy-single fallback for pre-slot
files; exact tails = Poisson-binomial)

| slice | legs | realized | stated | gap | exact tail |
|---|---|---|---|---|---|
| DD legs, season | 39/57 | .684 | .740 | **−5.6pp** | P(≤39)=.207 |
| DD legs, thru 7/12 (anchor ✓ doc) | 25/42 | .595 | .734 | −13.9pp | .035 |
| DD legs, thru 7/27 | 33/51 | .647 | .739 | −9.2pp | .094 |
| DD legs since 7/13 | 14/15 | .933 | .757 | +17.7pp | P(≥14)=.089 |
| primaries, thru 7/12 (anchor ✓ doc) | 51/65 | .785 | .767 | +1.8pp | — |
| primaries, season | 64/83 | .771 | .770 | +0.07pp | .549 |

Monthly DD: May 16/28 (.571), June 7/7, July 10/16 (.625), Aug 6/6.

Fidelity: all 57 DD rows match the `build_slot_dataset.py` semantics
row-for-row (r3 direct diff), and both anchors reproduce the 7/12 doc
exactly. (The first draft's primaries used slot_results-only inclusion,
dropping four legacy single-pick days — 60/79/−1.2pp was wrong; corrected
above.)

## What tonight's read does and does not establish

- The season point estimate has narrowed: −13.9pp (7/12) → −9.2pp (7/27) →
  −5.6pp, driven by post-7/12 legs running 14/15. That window's endpoint is
  tonight's unplanned look, so the 14/15 emphasis is partially post-selected
  — the favorable mirror of the cold-streak framing the 7/12 doc retired.
- **A true ≥10pp shortfall is NOT excluded**: −5.6pp carries a null SE of
  ~5.8pp (95% ≈ [−16.9, +5.8]); three hit→miss flips would read −10.8pp.
  The tail .207 says the observed count is unsurprising under stated-p
  calibration; it does not bound the underlying shortfall.
- "Regressed because luck" is unidentified: luck, the ball-regime mix
  (juiced spring → partial reversion since late July, stated p's shifting),
  and selection/composition changes are confounded. Defensible wording: the
  subsequent outcomes moved the season point estimate toward zero.
- The 7/12 backtest reference (+0.98pp, clustered SE ~1.1pp) is CONTEXT, not
  a shared-null test — the 7/13 doc itself records that the backtest and
  live gaps are overlapping but distinct estimands (estimated_pa conditions
  on realized participation; serving-path effects absent).
- Primaries: "no primary underperformance detected," not "calibrated" —
  n=83 argmax-selected picks at .771 realized vs .770 stated is
  neutral-to-favorable (≈+2.1pp above the serving-realistic rank-1 ≈.75
  benchmark), and too small to certify calibration.

**Net: no production change tonight, and the formal checkpoint stands
unchanged. That is the whole result.**

## Quality-bin collapse: corrected attribution + disposition

The first draft attributed the 8/09 `mdp_policy_alignment` WARN (17/21 recent
primaries below the lowest boundary, .796) to the ball-regime shift. **Wrong
timeline:** the 2026-05-23 calibration-resolve gate memo already records all
57 checked production primary days mapping to Q0 at the same boundaries
(`[.796, .811, .825, .841]`) — the collapse is a chronic
estimated-PA-vs-actual-PA scale mismatch documented in May. The deadening may
be deepening it; it did not cause it.

Mechanics corrections (r3): production digitizes ONLY `primary_p`
(`strategy.decide_action` → `mdp.lookup_action`); the "16/16 DD candidates in
Q0" line is monitor telemetry, not lost production discrimination (a shifted
DD distribution can still make learned `p_both` transitions wrong — different
mechanism). Action regret from collapse is state-dependent, not implied by
occupancy: at the 8/09 end state (streak 0, ~50 days, saver available) the
deployed table doubles in every bin — zero immediate action cost; at streak
≥8 Q0 skips while higher bins double, where discrimination loss has teeth.
The skip-policy shadow is the live evidence stream there (10/12 resolved
divergent days hit; still below its pre-registered n=30 verdict gate — EV
sign unresolved).

Disposition:

1. **No production re-solve, no live boundary knob.** The repo's own gate-B
   history stands: most apparent re-solve benefit came from boundary-scale
   correction, but the historical boundary artifact failed production
   transfer and "does not advance a boundary-only policy artifact"
   (2026-05-24 docs). Nothing tonight overturns that.
2. **Cheap boundary-only MEASUREMENT is justified now** (first draft wrongly
   deferred all work): a current-scale quantile/estimated-PA-rescaled
   boundary set as a shadow artifact, evaluated by actual-state action-diff
   replay on realized sequences (patterns: gate-B fair comparator + the 7/13
   realized-replay standard). Measurement, not deployment; own
   pre-registration before any promotion talk.
3. **Re-solve readiness is gated on DIRECT signals, not park_drag** (first
   draft wrongly named the drag table the completion signal; it measures
   pitch-flight Cd only, is COR-blind, and its ranking screen was null):
   stability of the served primary-p distribution + model/feature
   fingerprints, and sufficient current-scale outcome support. park_drag is
   an explanatory covariate.

## Review trail (r3, same night)

Draft-1 errors corrected here: "tripwire does not fire" (invalid — rule not
evaluated at n=57); primaries 60/79 (missing legacy-single fallback; and the
draft failed to flag that its own primaries anchor, 47/61, mismatched the
7/12 doc's 51/65); "gap no longer approaches 10pp" (SE ~5.8pp — 10pp not
excluded); "if luck, it regresses" as established (unidentified); backtest
used as shared-null reference (distinct estimand per 7/13); ball-regime
attribution of bin collapse (May precedent); park_drag as re-solve gate
(COR-blind); "skips at ≥8 sub-.796" (state-dependent shorthand); "primaries
calibrated" (overclaim). Verified against: `build_slot_dataset.py` fallback,
`strategy.py`/`simulate/mdp.py` digitize path, 2026-05-23 gate memo,
2026-05-24 gate-B docs; all corrected numbers independently recomputed.

Reproduction: `tripwire.py` (+ fallback variant) in the 8/09 session
scratchpad; 7/12 builder semantics.
