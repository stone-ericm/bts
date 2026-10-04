# W1.2 Benchmark bridge — design (season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, sections W1.2 and ground rule 1.3.
**Status:** draft for Codex design review. Analysis deliverable: at most two review rounds, then freeze with the limits stated (Eric, 2026-10-03).

## 1. Question
The README's backtest headline (top-1 about 86%) and the 2026 live result (about 68–72%) are not comparable estimates (rule 1.3). The bridge measures, on the same candidates, how much of the gap each step explains:
- **A, legacy actual-PA:** probabilities compounded over the realized PA rows;
- **B, estimated-PA:** realized lineup and starter, a lineup-slot PA count, a reliever context;
- **C, at-cutoff reconstruction:** the served lineup and starter inputs and the as-of lookups;
- **D, archived production:** the scores actually served.

A partly unexplained gap is an acceptable result. Nothing here builds a point-in-time platform (the July decision stands).

## 2. Window and shared candidates
- **Dates:** the 105 dates with a served full slate (`data/picks/slates/<date>.json`, `bts_slate_v1`, written by `bts.slate` from 2026-06-11, through 2026-09-27).
  - Every other 2026 game date is reported as the missing fraction. Before 6/11 no full served slate exists; the live-forward top-10 is a frozen-code rescoring, not served scores, and is not used.
- **Shared candidate set** = each date's D slate rows (`batter_id`, `game_pk`). Every surface is scored on these rows. A row a surface cannot score (for example, a batter who never batted, absent from A) is a reported coverage loss per surface, never silently dropped.
- **Binding.** A slate is **bound** when that date's selection (pick file, or `decision.json` from 6/23) equals one of its rows on `batter_id`, `game_pk` and `p_game_hit`. The slate is last-write-wins, so an unbound slate may come from a later failed attempt. The primary analysis uses bound dates; all dates are a sensitivity.

## 3. Surfaces (all on the shared rows)
| Surface | Definition | Built how |
|---|---|---|
| **D, served** | the slate row's `p_game_hit` as written | read |
| **C-served, at cutoff, daily-learning recipe as served** | that date's served model `data/models/blend_<date>.pkl`, used only when its sha256 equals the pick's `provenance.model_pickle_sha256`. Lookups come from `_build_feature_lookups(df_feat[date < D])`, which reproduces serving, as M3 verified. Slot inputs come from the D slate rows (`lineup`, `pitcher_id`, `projected`, batter/team). Game-level slot fields (venue, opponent, roof, umpire, weather) come from the local raw feed. | `predict()` scoring with `bts.model.predict._fetch_game_slots` replaced by the injected slots (no network) |
| **C-frozen, at cutoff, frozen coefficients** | the same inputs, with one blend trained once through 2025 (`BTS_LGBM_DETERMINISTIC=1`) | same |
| **B26, estimated-PA (oracle)** | `blend_walk_forward` over 2026, mode `estimated_pa`, with `top_n` set to keep every batter | one deterministic run |
| **A26, actual-PA (oracle)** | the same run in mode `actual_pa`, with the per-PA log on | one deterministic run |
| **A26-count, count-only (oracle)** | `1 − (1 − p̄)^{N_est}`: p̄ = mean per-PA probability over the realized rows (from A26's PA log); N_est = the lineup-slot PA estimate | derived |

Anything that uses the realized participation, starter or PA count (A26, A26-count, B26) is an **oracle diagnostic**, never an achievable comparator. Neither recipe is a pristine 2026 holdout (rule 1.2).

**The reproduction test (C-served vs D).** On bound rows, C-served should reproduce D up to floating-point noise. Each residual row is classified:
- the projected-vs-confirmed lineup state;
- a lookup difference;
- a game-level slot field: a post-game raw value against the value live at serve time;
- unexplained.

A reproduction failure is reported, not hidden. If C-served cannot reproduce D, then B-vs-C differences are not attributable to inputs.

## 4. Outcomes (one definition for all surfaces)
- Per `(date, batter_id, game_pk)`, from `read_pa_for_bts_scoring` (the resumed portion excluded): **hit** (at least one hit), **no_hit** (at least one PA and no hit), or **no_pa**.
- Proper scores and within-slate discrimination use hit and no_hit rows.
- A rank-1 row with no_pa is reported separately as void, never as a miss. The BTS pass rules are W1.5's L01, not this read.

## 5. Decisions and eligibility
- **Rank-1 per surface:** the argmax among rows **eligible at the slate's `written_at`**, meaning the game's scheduled first pitch more than 5 minutes after `written_at`, with times from the raw feed's scheduled start.
- Only the primary top-1 is compared. The double-down and second-candidate rules are out of scope.

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

## 7. Prior results carried (not re-derived)
- **May PA basis:** rank-1 realized PA 5.500 vs estimated 4.429 (`docs/sota_audit/2026-05-24-probability-scale-pa-basis.md`).
- **M3 v3:** 39 discordant days, −0.67 pp, CI [−3.4, +2.0], HOLD (`docs/audit/2026-06-11-m3-serving-staleness.md`).
- **June 29 PA tilt:** paired 146 vs 143, p = 0.91; median top-pick p: 0.817 actual / 0.778 estimated / 0.765 live (`docs/audit/2026-06-29-skip-threshold-and-discrimination.md`).
- **May gate-B:** `OUTCOME_MIXED_HEADLINE_FAILS_STABILITY_BAR`, and a B-vs-D scale divergence of 0.7794 vs 0.7488 (`docs/sota_audit/2026-05-24-gate-b-production-metric-result.md`).

## 8. Exposure (before any outcome join)
A new exposure-register row, **X-21**, is predeclared and pushed before the outcome join runs. It covers:
- the per-candidate 2026 outcomes on the slate dates, for exactly the metrics in §6;
- descriptive use;
- results feed W1.3 and the decision memo.

No W4 candidate may cite the bridge as its motivation without its own registration. D3 = RESERVE is respected: these dates' production outcomes were consumed by X-01 and X-09, and the 9/19–9/27 research captures are X-12 ("W1.2 diagnostic use only"). This read adds comparisons, not new outcome dates.

## 9. Build and run
- **Code:** `scripts/audit/benchmark_bridge/` (pure functions plus a CLI), tested test-first with synthetic slates, feeds and pickles. The slot injection and binding are unit-tested; the reproduction classifier gets a planted mismatch per class.
- **Run:** on the box, in a transient `systemd-run --user` unit; output under `data/validation/w12_bridge/<sha>-<run>/`. The unit runs:
  1. the A26/B26 walk-forward (deterministic);
  2. C-frozen training;
  3. scoring;
  4. the outcome join, **only after X-21 is pushed**;
  5. the metric tables.
- **Memo:** `docs/audit/<date>-benchmark-bridge.md`, reporting the reproduction test first, then the A→D table with intervals and the missing fraction.

## 10. Limits stated up front
- **Window:** 6/11–9/27 only, a late-season subset, so it is not the season.
- **Game-level slot fields** come from post-game feeds.
- **The slate's `written_at`** is the last write, not necessarily the lock.
- **Unbound slates** are excluded from the primary analysis.
- **The served model** is used only where its sha binds.
- **A26 and B26** retrain every 7 days on all prior data; that is their legacy definition, not the served cadence.
