# W1.2 Benchmark bridge — design (season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, sections W1.2 and ground rule 1.3.
**Status:** rev 2, with Codex design review r1's verbatim edits applied (r1 BLOCK, F1–F9; `.codex-review/w12-bridge/codex-findings-design-r1.md`). Round 2 is the last (Eric's pace rule: at most two rounds for an analysis deliverable, then freeze with the limits stated).

## 1. Question
The README's historical backtest headline (top-1 about 86%) and the 2026 live result (about 68–72%) are not comparable estimates (rule 1.3). This bridge compares scores and reranked primaries within the reconstructable 2026 slate window. It diagnoses mechanisms behind the information-set mismatch; it does not reproduce the historical headline or assign a fraction of the headline-to-live gap to each step. Mixed transitions remain descriptive rather than causal decompositions.

The surfaces:
- **A, legacy actual-PA:** probabilities compounded over the realized PA rows;
- **B, estimated-PA:** realized lineup and starter, a lineup-slot PA count, a reliever context;
- **C, at-cutoff reconstruction:** the served lineup and starter inputs and the as-of lookups;
- **D, archived production:** the scores actually served.

A partly unexplained gap is an acceptable result. Nothing here builds a point-in-time platform (the July decision stands).

## 2. Window and shared candidates
- **Dates:** inventory the archived `bts_slate_v1` files dated 2026-06-11 through 2026-09-27; 105 is the provisional census, to be verified. Publish the denominator of all 2026 regular-season ET game dates and the missing-date fraction. The live-forward top-10 is frozen-code rescoring and is not substituted for served scores.
- **Shared candidate set** = each date's D slate rows (`batter_id`, `game_pk`). Every surface is scored on these rows. A row a surface cannot score (for example, a batter who never batted, absent from A) is a reported coverage loss per surface, never silently dropped.
- **Selection consistency:** use `decision.json.primary` where an authoritative decision exists; otherwise use the pick-file primary. Record conflicts between sources rather than choosing a favorable match. Match normalized `(batter_id, game_pk)` and the selected probability after applying the slate writer's exact pandas JSON serialization. This establishes selection consistency only; it does not bind the full slate to a prediction attempt or final lock. Record existing attempt/time witnesses when available, otherwise provenance is unknown. The primary selected-date analysis uses selection-consistent dates, with all available dates as a separately labelled sensitivity. Research-only/no-selection dates are a separate stratum, and stale selections do not supply their model provenance.

## 3. Surfaces (all on the shared rows)
| Surface | Definition | Built how |
|---|---|---|
| **D, served** | the slate row's `p_game_hit` as written | read |
| **C-served, archived-coefficient reconstruction** | Load the archived `blend_<date>.pkl` only when its sha256 matches the selected pick's top-level `model_pickle_sha256` and its origin is consistent with the archived tier/provenance. Extract `_model` as the single model and pass the remaining members as `blend`. Check historical code, feature/env and calibration provenance. Build ordered lookups and opener history from the declared pre-date frame. Preserve the slate's lineup, pitcher, projected state and candidate identity; declare the source for every additional slot field, including pitcher hand and weather. Final-source reconstruction is not proof of historical input availability. Missing binding/provenance is reported, never replaced silently by a retrained model. | Inject slots into `predict()` without network or `run_pipeline()` refresh |
| **C-frozen, fixed coefficients** | Train the single model and blend once on the declared 2019–2025 training pool, with deterministic settings fixed before importing training configuration; use the same reconstruction adapter and inputs as C-served wherever their provenance permits. | Same scoring path |
| **B26, estimated-PA (oracle)** | `blend_walk_forward` over 2026, mode `estimated_pa`, with `top_n` set to keep every batter | one deterministic run |
| **A26, actual-PA (oracle)** | the same run in mode `actual_pa` | one deterministic run |
| **A26-count, geometric count normalization (oracle)** | For `N_actual > 0`, `1 − (1 − A26) ** (N_est / N_actual)`. N_est uses the realized lineup slot and the existing lineup-to-PA map, default 4.0 for a missing slot. This holds the geometric per-PA no-hit rate fixed and equals A26 when the counts agree. | Derived from A26 game score, `n_pas` and the realized lineup slot; no PA log required |

Anything that uses the realized participation, starter or PA count (A26, A26-count, B26) is an **oracle diagnostic**, never an achievable comparator.

A26/B26 use one pinned daily-learning recipe, including code, feature/env configuration, seed, model set, training start and seven-test-date retraining calendar. Run the original full 2026 calendar and restrict reported outputs afterward. Record/reuse identical fitted models for their paired comparison. C-served is the archived production sequence, not automatically a frozen recipe; report recoverable recipe strata. A26-count→B26 changes matchup/context and ensemble aggregation conventions. B26→C-frozen changes inputs and coefficients/training cadence. C-frozen→C-served changes fitted models and may also change recipe/calibration. These are composite contrasts unless the other components are explicitly held fixed. A same-artifact B-frozen/C-frozen rescoring is optional; without it, the isolated input effect is unresolved. Neither recipe is a pristine 2026 holdout.

**The reproduction test (C-served vs D).** Compare every reconstructable row on selection-consistent dates, using a predeclared absolute tolerance of 1e-9 for serialized game scores and separately reporting the maximum/quantiles of residuals and winner agreement. Score parity establishes numerical consistency on those rows; it does not prove attempt identity, historical availability, selection parity or causal attribution. Report missing artifact/provenance separately from numeric residuals. Residual explanations may include recipe/env/calibration, opener/PA-split, pitcher-hand or game context, lookup/source revision, and serialization. Call a cause verified only when supported by an archived live witness or a controlled substitution; otherwise label it inferred or unexplained. Projected/confirmed state alone is not a verified numeric cause. If C-served fails to reproduce D, attribution across that boundary remains unresolved; even successful parity does not isolate the composite B-vs-C transitions.

## 4. Outcomes
Resolve `(date, batter_id, game_pk)` through `read_pa_for_bts_scoring`, excluding resumed PA. Labels are hit, no_hit, no_pa, or unknown. Assert no_pa only when the relevant game's PA source is complete; missing/unreadable/incomplete sources are unknown. Verify suspended-game exclusion support or mark the affected result unknown. Hit/no_hit are baseball-event labels, not BTS settlement labels. Report selected no_pa separately without treating it as a miss, and do not describe it as the exhaustive BTS Pass category. Rank before joining outcomes and never replace a winner because it is no_pa or unknown.

## 5. Eligibility and decisions
Use a shared reconstructed eligibility mask for primary forecast metrics and reranking. Recover scheduled time and detailed candidate status at the observation time where existing evidence permits, including unavailable/warmup/cutoff exclusions and the applicable serving era. When only final raw start time and slate `written_at` exist, label `scheduled_start > written_at + 5 minutes` a scheduled-time surrogate; report eligibility as unverified and show that surrogate diagnostic separately. All-row scoring is a separate diagnostic that can include already unavailable games. Preserve the archived D primary and action alongside each surface's diagnostic argmax; count skip, research-only and unknown actions. Compare primaries only; DD selection and policy trajectories remain out of scope. Break exact score ties by archived D row order across surfaces.

## 6. Metrics
- **Per surface, on the shared hit/no_hit rows:**
  - top-1 hit rate (rank-1, eligible; void counted separately);
  - within-slate discrimination: a per-date tie-aware rank AUC (`bts.health.slate_auc._rank_auc`), date-weighted mean;
  - Brier and log loss (`bts.validate.proper_scoring`);
  - the mean stated−realized residual;
  - a reliability table.
- **Per adjacent comparison** (A26→A26-count→B26→C-frozen→C-served→D, plus C-frozen vs C-served):
  - same-batter paired score change: mean, median, absolute deviation and correlation;
  - reranked rank-1 changes and their count;
  - discordant-outcome days;
  - paired differences in top-1 and in Brier, with a **date-block bootstrap** 95% interval (10,000 resamples, fixed seed).
- Coverage and missing fraction per comparison.
- Descriptive only: no hypothesis test chooses anything, and no candidate is nominated from this read.

For each pair, freeze the identical candidate pool with finite scores in both arms and the shared eligibility rule before outcome inspection. Report intersection/union overlap and exclusions by surface and reason. Keep native-surface winners and coverage separately; ranking on a common reduced pool is a conditional diagnostic, not the archived production action. AUC and proper scores use the same known hit/no_hit rows in both arms. Paired top-1 differences use the same dates where both previously selected winners have known hit/no_hit labels; report unilateral no_pa/unknown exclusions and never reselect. Publish row and date denominators for every table.

AUC is the equal-date mean of tie-aware within-date AUCs; omit and count dates lacking either class. Brier, log loss and mean stated−realized residual are candidate-row-weighted; paired differences use the identical rows. Resample whole eligible dates jointly for both arms, with replacement, 10,000 times at a recorded fixed seed, preserving repeated-date multiplicities; recompute the declared statistics and take the 2.5/97.5 percentiles. Reliability tables use fixed bins on [0,1] and omit the helper's candidate-level Wilson intervals. Report effective dates and discordant-outcome counts. These intervals are conditional on the frozen artifacts/support and exchangeable date clusters; they do not include cross-date dependence, fitting uncertainty or selection bias. No equivalence or causal attribution follows from a narrow interval. Adjacent contrasts have the composite meanings stated in §3, and changing pairwise support prevents adding their effects into a single decomposition.

## 7. Prior results carried (not re-derived)
- **May PA basis:** rank-1 realized PA 5.500 vs estimated 4.429 (`docs/sota_audit/2026-05-24-probability-scale-pa-basis.md`).
- **M3 v3:** 39 discordant days, −0.67 pp, CI [−3.4, +2.0], HOLD (`docs/audit/2026-06-11-m3-serving-staleness.md`).
- **June 29 PA tilt:** paired 146 vs 143, p = 0.91; median top-pick p: 0.817 actual / 0.778 estimated / 0.765 live (`docs/audit/2026-06-29-skip-threshold-and-discrimination.md`).
- **May gate-B:** `OUTCOME_MIXED_HEADLINE_FAILS_STABILITY_BAR`, and a B-vs-D scale divergence of 0.7794 vs 0.7488 (`docs/sota_audit/2026-05-24-gate-b-production-metric-result.md`).

## 8. Exposure
X-21 must be predeclared and pushed before any outcome-bearing bridge execution or inspection, including A/B generation, fitting, PA logs, joins and metric tables. It records the analysis recipe, full 2026 PA training/participation basis, registered slate dates/candidate identities, outcome definitions, support/eligibility rules, metrics and permitted descriptive outputs. Whole-season walk-forward is needed for the original fitting calendar, but disclosed diagnostic outputs are restricted to registered rows/dates; broader outputs require explicit prior scope. X-01/X-09 cover production selected-slot reads, not all newly examined slate-candidate outcomes. X-12 permits the research-only W1.2 diagnostic stratum. This is a new registered candidate-level descriptive exposure under D3, not a claim that all these outcomes were already consumed. Any model/policy candidate motivated by it requires its own registration and prospective 2027 validation; this bridge supplies no 2026 candidate test. Results may inform W1.3 and the decision memo within those limits.

## 9. Build and run
- **Code:** `scripts/audit/benchmark_bridge/` (pure functions plus a CLI), test-first with synthetic slates, feeds and pickles.
- **Run order:** Before the transient analysis unit runs, verify X-21's published scope and freeze/hash its input manifest. Then generate the original-calendar A/B outputs, train C-frozen, score the declared rows, resolve outcomes and produce metrics. Use only owned caches and outputs under the run directory; isolate `_build_probable_pitcher_lookup` and external-artifact dependencies from mutable production caches. Do not refresh feeds or invoke live serving. Tests cover JSON probability precision, same-selection/different-slate ambiguity, `_model` extraction, count identity at N_est=N_actual, differing coefficients, opener/context/provenance residuals with and without witnesses, reliever-only participation, void/unknown winners, shared comparison masks, eligibility surrogates and repeated-date bootstrap weights. The memo reports reproduction first, then labelled contrasts, coverage/denominators, actual-action counts and the missing fraction.
- **Where:** on the box, in a transient `systemd-run --user` unit; output under `data/validation/w12_bridge/<sha>-<run>/`. Memo `docs/audit/<date>-benchmark-bridge.md`.

## 10. Limits stated up front
- **Window:** 6/11–9/27 only, a late-season subset, so it is not the season.
- **Game-level slot fields** come from post-game feeds.
- **The slate's `written_at`** is the last write, not necessarily the lock.
- **Unbound slates** are excluded from the primary analysis.
- **The served model** is used only where its sha binds.
- **A26 and B26** retrain every 7 days on all prior data; that is their legacy definition, not the served cadence.
- Missing attempt identity, live input/status witnesses or recipe provenance remains missing. Final-source reconstruction and scheduled-time-surrogate metrics are diagnostics rather than demonstrated point-in-time forecasts. Composite transitions do not identify an isolated information-set effect, and pairwise support changes prevent telescoping the contrasts. The 105-date count and reconstructable-artifact coverage are verified at run time. The historical README/live headline remains noncomparable context.
