# W2.3 MLB forecast benchmark — design (season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, W2 product 3 (gates i–vi) and W4 rank 1.
**Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).

## 1. Question
Does MLB's own "% chance to hit" (`probabilityStarter`, published in the BTS app's `most_selected_players` sheet) carry information about the outcome beyond our served probability, on the same candidates? The answer is descriptive. The plan forbids a blend or gate search on this window: at most one combination rule is nominated afterwards as W4 rank 1, to be validated on later untouched dates (2027).

## 2. Data (inventoried 2026-10-04: schema and dates only)
- **MLB forecasts:** `data/leaderboard/static_snapshots/most_selected_players/<UTC stamp>.json[.gz]`.
  - 217 content-deduped captures, 2026-07-04 → 2026-09-27, taken every 30 minutes and stored only when bytes change.
  - Rows `{roundId, playerId, probabilityStarter, numberSelections}`, covering only the most-selected players (about 55 rows per capture across today's and tomorrow's rounds).
  - The mappings `roundId → date` (rounds sheet), `playerId → batter_id` (players sheet, `feedId`) and `unit → game_pk` (units sheet) come from the same capture directory, via the W1.1 ledger parsers (`scripts/audit/season_ledger/sources/static.py`).
- **Our served probability:** the served slates `data/picks/slates/<date>.json` (`bts_slate_v1`), with the frozen W1.2 bridge rules: selection consistency (§2) and verified or surrogate eligibility (§5).
- **Outcomes:** the W1.2 definition (§4) — baseball-event hit / no_hit / no_pa / unknown, resumed portion excluded, no_pa only from a complete source.

## 3. Gates (the plan's i–vi)
1. **As-of join.**
   - Key: `(date, batter_id, game_pk, captured_at, probabilityStarter)`.
   - For each slate, use the last capture at or before the slate's `written_at`.
   - `roundId` fixes the date, so today's and tomorrow's rounds are never conflated.
   - A batter with two games that day (a doubleheader) maps through the unit (`feedId` = game). If the unit is unresolved, the row is ambiguous and excluded, and counted.
   - Rows MLB lists that are not in our slate, and slate rows MLB does not list, are coverage, reported both ways.
2. **Target semantics.** No published definition exists; the FAQ says only "our unique prediction model". Two targets are compared by calibration on shared rows:
   - **T1:** P(hit) unconditionally, with no_pa counted as no hit;
   - **T2:** P(hit | at least one PA), excluding no_pa.

   Both are reported. If neither fits clearly better (the date-block bootstrap interval of the calibration-in-the-large difference contains 0), the read is **association only**, not a certified forecaster. Our served p targets T2-like game probabilities, which is a stated asymmetry.
3. **Freshness.** Report the distribution of each forecast's age at the slate's `written_at`. Captures are content-deduped, so an unchanged sheet leaves no capture, and age is an upper bound. No fetch log is retained; stated.
4. **`numberSelections`** is global popularity. It is reported descriptively and never used as a weight or a forecast.
5. **Scoring on identical shared candidates, date-weighted.** For MLB p and our served p on the same rows:
   - Brier, log loss and calibration-in-the-large, by target T1 and T2;
   - within-slate discrimination: the equal-date mean rank AUC over shared rows, dates lacking a class counted;
   - shared-set top-1: each forecaster's argmax among the shared eligible rows, ranked before outcomes, voids never replaced;
   - **disagreement-only outcomes:** dates where the two argmaxes differ, with each side's hit rate;
   - coverage and denominators.
   - Intervals: whole-date bootstrap, 10,000 resamples, fixed seed (as W1.2 §6).
6. **Residual information, descriptive.** A single logistic regression of the outcome (T2 rows) on `logit(p_ours)` and `logit(p_MLB)`, i.e. a forecast-encompassing check. Report the MLB coefficient with a date-block bootstrap interval, plus the in-sample log-loss change.
   - No model selection, no weights fitted for use, no gate.
   - The combination rule for W4 rank 1 is chosen afterwards from the literature (W3 §5: beta-transformed or logit pool), not from these numbers.

## 4. Exposure
A new exposure-register row, **X-23**, is predeclared and pushed before any outcome-bearing execution. It covers:
- the candidate-level outcomes of the shared rows, for exactly the §3 metrics;
- descriptive use only.

The slate candidates' outcomes are also registered under X-21 (W1.2), but comparing them with MLB forecasts is a new read. Under D3 = RESERVE, no candidate is tested on 2026. W4 rank 1 validates on 2027.

## 5. Build and run
- **Code:** `scripts/audit/mlb_benchmark/`, written test-first. It reuses `scripts/audit/benchmark_bridge/core.py` (selection consistency, eligibility, outcomes, AUC, bootstrap) and the ledger's static parsers.
- **Run:** on the box, in a transient unit, into `data/validation/w23_mlb_benchmark/<sha>-<run>/`.
- **Memo:** `docs/sota_audit/<date>-field-mlb-forecast-benchmark.md`, reporting coverage and freshness first, then the target-semantics result, then the scores.

## 6. Limits stated up front
- Only the most-selected players have MLB forecasts, a popularity-selected subset.
- The window is 7/04–9/27.
- Forecast age is an upper bound.
- The target semantics may be unresolvable.
- Shared-set top-1 is not the production decision.
- The encompassing check is in-sample and descriptive.
