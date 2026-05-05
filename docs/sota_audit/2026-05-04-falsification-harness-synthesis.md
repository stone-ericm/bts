# Falsification harness — methodology synthesis

**Date**: 2026-05-04
**Branch**: `feature/sota-synthesis-memo`
**Predecessors**: [v2 comparison memo](./2026-05-02-harness-v2-comparison.md) · [v2.5/v2.6 attribution memo](./2026-05-03-harness-v2.5-attribution.md)
**Tracker**: [`docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md`](../superpowers/specs/2026-05-01-bts-sota-audit-tracker.md)
**Scope**: docs-only synthesis. No source edits. No data regeneration. No production-deploy claim.

> This memo connects the seven validation-methodology PRs merged 2026-05-04 (and yesterday's v2.5/v2.6 attribution work) into a single picture: **what infrastructure exists, what it has and has not measured, and what we are not yet entitled to claim.**

## Headline

The validation stack now has five composable pieces — proper scoring (#12), purged-blocked CV with lockbox (#5), binary-y conformal gate (#11), policy-value evaluation (#13), and CE-IS rare-event MC (#14). **All five are infrastructure shipments.** None of #11 / #13 / #14 has been used to authorize a production deploy; #11's real-data smoke during PR review returned an empty `ship_set` (all 6 method×alpha cells FAIL → verdict `NO_PRODUCTION_DEPLOY`); #13's V_pi shows ~70× across-fold spread under universal `SPARSE_HOLDOUT_SUPPORT`, consistent with fold instability under sparse support, not evidence for a single stable expected value; #14's fixed-window estimand is not a season-level P57 and is not comparable to #13.

The v2.6 block-bootstrap CI (in the `corrected_audit_pipeline` path that the harness already uses) further showed that v1's `HEADLINE_BROKEN` gate-class transition was a percentile-CI artifact at the half-headline=0.04085 threshold — point-estimate attribution to per-bin `rho_pair` and per-fold MDP solve survives, but the gate-class story collapses.

**Action implied**: keep production policy as-is. The next priority is a research-design choice (see "Next priorities," last section), not a deploy.

## Methodology shipped (this audit cycle)

| Area | PR | Module | CLI command | Output schema |
|---|---|---|---|---|
| #12 Proper scoring | [#9](https://github.com/stone-ericm/bts/pull/9) | `src/bts/validate/proper_scoring.py` | `bts validate proper-scoring` | log loss, Brier, Murphy reliability/resolution/uncertainty, top-bin calibration |
| #5 Validation contract | [#10](https://github.com/stone-ericm/bts/pull/10) | `src/bts/validate/splits.py` + scorecard `--manifest` | `bts validate split-manifest`, `bts validate scorecard --manifest` | manifest with `LockboxSpec` + per-fold rolling-origin `FoldSpec`; per-fold scorecard with `manifest_metadata` |
| #11 Conformal gate v2 | [#11](https://github.com/stone-ericm/bts/pull/11) | `src/bts/validate/conformal_gate.py` + `src/bts/model/conformal.py` | `bts validate conformal-gate` | `conformal_validation_v2` with method×alpha matrix, `ship_set`, `PRODUCTION_DEPLOY_READY` / `NO_PRODUCTION_DEPLOY` |
| #13 Policy-value eval | [#13](https://github.com/stone-ericm/bts/pull/13) | `src/bts/validate/ope_eval.py` | `bts validate policy-value` | `policy_value_eval_v1` with per-fold V_pi, terminal-MC replay cross-check, `verdict_flag` ∈ {`OK`, `SPARSE_HOLDOUT_SUPPORT`} |
| #14 CE-IS rare-event MC | [#14](https://github.com/stone-ericm/bts/pull/14) (design) + [#15](https://github.com/stone-ericm/bts/pull/15) (impl) | `src/bts/validate/rare_event_mc_eval.py` | `bts validate rare-event-ce-is` | `rare_event_ce_is_v1` with per-fold fixed-window estimate, ESS / max_weight_share / log_weight_variance |

**Schema convergence across the cycle (with the actual variation):** the four manifest-bound validators (#5 scorecard, #11, #13, #14) all carry `lockbox_held_out=true` + `manifest_metadata`. The three CV-evaluation outputs (#5 scorecard, #13, #14) additionally carry `aggregate_deferred=true` to make the cross-fold-CI deferral explicit; **#11 lacks `aggregate_deferred`** because its verdict is per-cell pass/fail rather than a deferred aggregate. **#12 proper scoring is descriptive only** — it computes scores on profile rows and does not consume a manifest, so it has neither field. Aggregate fold CIs are deliberately deferred for the three CV-eval outputs to a future P1.5+ cycle.

## Production claims (what the harness has and has not concluded)

**The harness has produced the following empirical findings.** Each is methodology-conditional, not a production-deploy authorization.

### #11 Conformal gate v2 — real-data smoke ran, returned `NO_PRODUCTION_DEPLOY`
- **What exists**: bucket-Wilson lower-bound validity test + median-bound-width tightness gate, composed over the #5 manifest with lockbox held out. Two methods × three alphas = 6 method×alpha cells.
- **What we have measured**: real-data smoke executed during PR #11 review on the default #5 manifest (lockbox 2025-08-30..2025-09-28). All 6 method×alpha cells (`bucket_wilson` × {0.05, 0.1, 0.2}, `weighted_mondrian_conformal` × {0.05, 0.1, 0.2}) returned cell-level verdict `FAIL`. Top-level result: `ship_set=[]`, verdict `NO_PRODUCTION_DEPLOY`.
- **What is NOT a durable artifact**: the smoke JSONs (`/tmp/sota11_gate_codex.json`, `/tmp/sota11_gate_pr11_codex.json`) were /tmp-only review smokes; no validation artifact was committed under `data/validation/` as a published gate run.
- **What we cannot claim**: that any (`method`, `alpha`) combination passed. The smoke gives `NO_PRODUCTION_DEPLOY` on the methods/alphas tested; a `PRODUCTION_DEPLOY_READY` verdict would require a non-empty `ship_set`, which the current smoke did not produce.

### #13 V_pi — fold-level instability under universal sparse support
- **What exists**: per-fold target-policy solve on fold-train, evaluation against fold-holdout bins via `evaluate_mdp_policy`, and a terminal-MC replay cross-check.
- **What we have measured**: real-data smoke at PR #13 produced model-based V_pi values across the 5 folds of approximately {0.087, 0.099, 0.055, 0.0014, 0.014} — a ~70× spread. Cross-check `V_replay = 0.0` on every fold because the corresponding trajectory replay produced **zero terminal successes** under the target policy in any fold. `sparse_support` flag fires universally.
- **What we cannot claim**: that V_pi is a stable estimate of a single season-horizon policy value at this support level. The 70× across-fold spread, with universal `SPARSE_HOLDOUT_SUPPORT` and zero terminal successes in replay, is consistent with **fold instability under support limitations** — a property of the estimator and the data sparsity, not necessarily a property of the policy. V_pi at this support level is informative as a diagnostic of fold-level estimate stability, not as a deployable expected value.

### #14 CE-IS fixed-window — diagnostics pass; estimand is not season P57
- **What exists**: black-box wrapper around `estimate_p57_with_ceis` that fits theta on fold-train (incurred and discarded train estimate), evaluates on fold-holdout with `n_rounds=0`.
- **What we have measured**: real-data smoke produced per-fold fixed-window estimates in 5e-5 to 1.4e-4 with all five folds within the diagnostic thresholds (`verdict_flag = OK` requires ESS ≥ 1000 and max_weight_share ≤ 0.1; thresholds are diagnostic flags, not a production gate):

  | fold | estimate    | CI                       | ESS   | MaxWS | flag |
  | ---- | ----------- | ------------------------ | ----- | ----- | ---- |
  |    0 | 5.319e-05   | [2.76e-05, 8.33e-05]     |  4256 | 0.004 |   OK |
  |    1 | 1.430e-04   | [8.69e-05, 2.10e-04]     |  5552 | 0.003 |   OK |
  |    2 | 1.119e-04   | [4.91e-05, 1.84e-04]     |  7944 | 0.001 |   OK |
  |    3 | 6.642e-05   | [2.42e-05, 1.20e-04]     |  8955 | 0.001 |   OK |
  |    4 | 9.138e-05   | [4.00e-05, 1.53e-04]     | 10474 | 0.001 |   OK |

- **What we cannot claim**: that 5e-5 to 1.4e-4 is a "real-data P(57) estimate." The estimand is **P(max consecutive rank-1 hits ≥ streak_threshold) over the ordered fold-holdout date sequence under independent Bernoulli rank-1 hits, horizon = `n_holdout_dates` per fold** (≈30 dates per fold). It is not a season-horizon P57 and **is not comparable to #13's V_pi**.

### v2.6 block-bootstrap reconciliation (in the `corrected_audit_pipeline` path)
- **What exists** (shipped in PR #8 prior to this cycle): profile-level paired hierarchical block-bootstrap (Politis–Romano stationary, expected_block_length=7, n=500) on `corrected_pipeline_p57`.
- **What we have measured**: under block-bootstrap CI, all six v2.5 ablation cells gate `HEADLINE_REDUCED`. The v1 `HEADLINE_BROKEN` classification was a percentile-CI artifact at the half-headline=0.04085 threshold (ci_upper 0.0375 was 0.7pp below threshold on a CI whose grid resolution is 0.83pp = 1/120).
- **Threshold-specificity (per the v2.5/v2.6 addendum)**: this artifact claim pertains to half-headline=0.04085. For higher thresholds (e.g. 0.08), additional v2.5-percentile cells that currently gate BROKEN (010/001/101 at `v25_hi=0.0792`) would similarly flip to REDUCED under v2.6 block-bootstrap. The direction is: more cells collapse to REDUCED at higher thresholds, not fewer.
- **What survives from v2.5**: point-estimate attribution. Single-mode B (per-bin `rho_pair`) and single-mode C (per-fold MDP solve) each shift +1.67pp from the cell-000 baseline; combined B+C shift +2.50pp via one extra 2023 fold success. Change A (fold-local params) has no observable effect at metric resolution and is also an order-of-magnitude slower in this runner — keep A as leakage-hygiene methodology for final audits.
- **What collapses**: the v1→v2 "gate-class transition" framing. Under the adopted block-bootstrap CI, no transition occurs at this threshold.

## Interpretation — guardrails

These are constraints on the language used when drawing conclusions from the harness. They are not findings; they are limits on inference.

1. **#13 V_pi and #14 CE-IS measure different estimands.**
   - V_pi is a **season-horizon policy value** under target-policy execution against held-out trajectory replay; its support is the trajectories the data actually contains.
   - The #14 fixed-window CE-IS estimate is a **fold-window event probability** under the assumed independent-Bernoulli rank-1 model; its support is parametric (the elite-tilted theta_0 simulator).
   - **Numerical comparison between V_pi values and #14 fixed-window estimates is illegitimate.** They have different units (per-trajectory expected value vs. probability of a streak event in a 30-day window) and different generative assumptions.

2. **#13 replay is support-limited.**
   - The `sparse_support` flag fires when minimum-bin holdout count drops below the configured threshold. On the real-data run, it fired on every fold.
   - The 70× spread is in **model-based V_pi**, not replay; cross-check `V_replay = 0.0` on every fold because no terminal successes occurred in any fold's trajectory replay. The spread is consistent with fold instability under sparse support, not with a single stable underlying value.
   - V_pi at this support level is informative as a **diagnostic of fold-level estimate stability**, not as a deployable expected value.

3. **#11 conformal gate smoke returned `NO_PRODUCTION_DEPLOY`.**
   - The gate machinery (bucket-Wilson LB validity + tightness) is shipped and tested. Real-data smoke during PR #11 review covered 6 method×alpha cells and produced `ship_set=[]`, verdict `NO_PRODUCTION_DEPLOY`.
   - The smoke JSONs were /tmp-only review artifacts; no validation file was committed as a durable gate-run record.
   - We can support: "the smoke on the methods and alphas tested produced an empty `ship_set`." We cannot support: "any method×alpha is `PRODUCTION_DEPLOY_READY`" — that requires a non-empty `ship_set`, which the smoke did not produce.

4. **v2.6 attribution survives, but only at fold-resolution.**
   - The cell-000-baseline → cell-111 path (+2.50pp) is decomposed into B (+1.67pp single-mode), C (+1.67pp single-mode), B+C synergy (+2023 fold success worth +0.83pp).
   - These are point-estimate attributions at 1/120 metric resolution. Mechanism-level claims ("B and C are substitutable") would require trajectory-ID and policy-action inspection, which has not been done.
   - The v2.5 fold-vector observation extends to v2.6 — cells {010, 001, 101} share fold vector `[0, 0, 2/24, 0, 1/24]`; cells {011, 111} share `[0, 0, 3/24, 0, 1/24]`. Same seasons succeed with same counts.

5. **No production-deploy authorization in this stack.**
   - The four manifest-bound validators (#5 scorecard, #11, #13, #14) hold the lockbox out; the three CV-eval outputs (#5 scorecard, #13, #14) explicitly defer aggregate CIs. Neither the lockbox certification run nor aggregate fold CIs have been produced.
   - The harness can falsify; it has not been used to license a deploy. Treat any future "ship X based on the harness" claim as requiring an explicit lockbox certification + aggregate-CI step that does not yet exist.

## Cross-cutting observations

### The schema convention is converging, with the cycle's actual variation
The four manifest-bound validators (#5 scorecard, #11, #13, #14) carry `lockbox_held_out` + `manifest_metadata`. The three CV-eval outputs that defer aggregate uncertainty (#5 scorecard, #13, #14) additionally carry `aggregate_deferred=true`; #11 doesn't, because its verdict is per-cell pass/fail rather than a deferred aggregate. #12 is descriptive only — it computes scores on profile rows and is not manifest-bound. A future P1.5+ aggregate-CI cycle can target a shared codepath for the three CV-eval outputs that carry `aggregate_deferred=true`, rather than per-method machinery — the convention is consistent enough across PRs #10/#13/#15 that the aggregate work is no longer per-method.

### The Codex review trail is the audit record
The bus thread (#9 → #15 PR series, message IDs ~70–142) is now the durable methodology trail. Each PR carried 2–3 critique rounds: scope claim → first review (catch contract gaps) → checkpoint → second review (catch contract gaps surfaced in real code) → final review (catch invariant violations) → sign-off. **Catching invariant violations through call-graph tracing — not just through tests — is the high-value pattern.** Specific examples this cycle: the duplicate-rank-1-date guard in #14 P0/P1 (would have silently extended the estimator horizon under multi-seed input); the partial-current-season manifest-universe pitfall in #5 (would have leaked 2026 partial dates into 2025-lockbox fold holdouts); the elite-set vs. LR-weighted MLE discrepancy in the #14 design memo (the v1 source code did not match the textbook CE description).

### The v2.6 reconciliation is a methodology lesson
The collapsed gate-class transition is the second instance in this audit cycle (after the v2.5 attribution refuting the "fold-local fix" framing) where a methodology change was announced as a verdict-mover and was then refuted under closer review. The lesson, copy-edited from the v2.6 addendum: **when a methodology change produces a verdict-gate transition, default to two robustness checks before announcing — (1) is the point-estimate attribution actually driven by the change being announced; (2) is the gate-class transition robust to alternative CI methods at the same threshold.** Skipping either check leads to overclaim under review.

## Next priorities — recommendation tree (NOT a decision)

Codex's #142 named three candidate next directions. This memo does not choose; it lays out the conditions under which each becomes the right pick.

### Option A — SOTA area #15 dependence: full out-of-fold residual modeling

(The "#15" here is the SOTA tracker area, not PR #15 from this cycle — the latter is the CE-IS implementation discussed above.)

**Pick this if**: the harness is judged ready to extend the dependence work from "two-knob mean correction in the corrected_audit_pipeline" to "full out-of-fold modeling of within-game PA correlation and cross-game pair-residual structure" with proper per-fold parameter estimation that does not contaminate the held-out fold.
**Conditions**: requires (a) the dependence parameters (`rho_PA`, `tau`, `rho_pair_per_bin`) refit per fold without contamination from held-out data, (b) acceptance that v2.6's "A is below detection" is a point-estimate observation, not a license to drop A, (c) tolerance for the runtime cost (~3.5× slower per cell on the current Mac runner).
**Rejects if**: the dependence layer's marginal value on `corrected_pipeline_p57` is below 1/120 = 0.83pp at current resolution, and we do not have a sharper metric.

### Option B — #5 P1.5 aggregate CIs: cross-fold uncertainty for the validation stack

**Pick this if**: the harness's deferred `aggregate_deferred=true` flag becomes the binding constraint on producing publishable verdicts. Currently each CV-eval method (#5 scorecard, #13, #14) produces per-fold results without an across-fold CI; an across-fold CI would let those surfaces produce single bounds usable for gate decisions.
**Conditions**: requires (a) decision on aggregate-CI methodology (block-bootstrap on profile rows? hierarchical bootstrap on fold point-estimates? Politis–Romano with what block length?), (b) shared codepath that all five methods can target, (c) cross-validation against the v2.6 block-bootstrap result on `corrected_pipeline_p57` (the only existing aggregate CI in the codebase).
**Rejects if**: per-fold flagging (`SPARSE_HOLDOUT_SUPPORT` for #13, ESS / max_weight_share for #14, ship_set for #11) already provides the actionable signal and aggregate CIs would not change a deploy decision.

### Option C — Data-side multi-seed regeneration

**Pick this if**: the binding constraint is the data, not the methodology. Currently `data/simulation/backtest_*.parquet` carries one rank-1 row per date (single-seed); the harness has the duplicate-date guard in #14 because multi-seed semantics were deferred. A multi-seed regeneration would unlock (a) reduction of the seed-variance noise floor that the v2.5 / v2.6 attribution surfaces ran into, (b) tighter per-fold V_pi estimates by averaging across seeded trajectories, (c) richer #14 input (with explicit aggregation semantics).
**Conditions**: requires (a) decision on aggregation semantics (do duplicate-date rank-1 rows extend the estimator horizon, get averaged into a single per-day p_game profile, or define a different estimand entirely?), (b) compute budget for the regeneration run, (c) validation that the regenerated data reproduces the existing single-seed verdict at seed=0.
**Rejects if**: the existing single-seed corpus is judged sufficient for the next round of harness work, and seed-variance noise can be addressed methodologically (P1.5+ aggregate CIs) rather than data-side.

### Strategic-question reframe (per tracker 2026-05-04 status update)

The tracker's current framing — that the binding strategic question is "is current-model top-of-slate under-confidence real and exploitable" rather than "distribution-shift remediation" — is **not** addressed by any of A/B/C above. None of the three options moves the harness toward measuring top-of-slate calibration on production picks. **If under-confidence/top-of-slate leverage is the binding strategic question, the right next move is #12 phase 2 (the existing tracker's "next active item") on production realized picks, not any of the harness extensions.**

This is the choice the next session needs to make: harness-extension (A/B/C) vs. strategic-leverage (#12 phase 2 on realized picks).

## Process lessons (for the next audit cycle)

1. **Two-stage review with call-graph tracing has earned its keep.** The four invariant violations caught by Codex this cycle (duplicate-date guard, manifest-universe leak, elite-set CE description, exact_p57 oracle parameters) would not have been caught by tests alone — they were contract-level. Default to call-graph-traced review on every methodology PR going forward.

2. **Schema-as-record is durable where it applies.** Where validators are manifest-bound (#5 scorecard, #11, #13, #14), `manifest_metadata` + `lockbox_held_out` are now part of the durable JSON record; the CV-eval outputs that defer cross-fold uncertainty (#5 scorecard, #13, #14) additionally carry `aggregate_deferred=true`; method-specific simplification blocks (e.g. `v1_simplifications` in #14) document where v1 trades off rigor for reach. The convention is now strong enough across these surfaces that archeology of past runs survives even when the conversation context does not. Continue this convention; do not retroactively force the descriptive #12 surface or the per-cell-pass/fail #11 surface into a shape they were not designed for.

3. **Defer aggregate CIs explicitly.** Every CV-eval output that deferred aggregate uncertainty (#5 scorecard, #13, #14) shipped `aggregate_deferred=true` rather than silently per-fold-only. This made the methodology limit visible in the output rather than buried in code. Continue this convention for future CV-eval surfaces.

4. **Methodology-vs-production claims must be separated explicitly.** This memo's structure (Methodology shipped vs. Production claims) is the discipline that the v2 / v2.5 / v2.6 cycle had to learn the hard way (twice). Default to this separation in every future synthesis memo.

5. **The Codex bus thread is the methodology trail.** Searchable, threaded, externally durable. Continue threading PR-cycle reviews through it; it survives compaction.
