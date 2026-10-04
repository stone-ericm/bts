# W2.3 MLB's own forecast against ours: descriptive benchmark (season wrap)

**Date:** 2026-10-04. **Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Design:** `docs/superpowers/specs/2026-10-04-mlb-forecast-benchmark-design.md` rev 3, FROZEN. **Code:** `scripts/audit/mlb_benchmark/` at `e66bfe0`, FROZEN after two Codex code rounds (`docs/audit/2026-10-04-w13-w23-code-codex-r{1,2}.md`). **Exposure:** X-23 (`b68098d`), published before any outcome-bearing step; the runner refuses to run unless its pinned X-23 commit `b68098d` is an ancestor of the code commit and the checkout's register carries the X-23 row.
**Run:** `data/validation/w23_mlb_benchmark/e66bfe0-20261004T170614Z/` (transient unit `w23-mlb`, 13:06–13:31 ET, clean exit, 0 error lines), copied to `data/hetzner_results/season_wrap_outputs/w23/`.
- **Hashes:** `results.json` `58ed9ea7…`, `joined.parquet` `e25050de…`, `manifest.json` `a14d6dad…`.
- **Our side:** read only through the accepted W1.2 run `4e4bf5a-20261004T060725Z` (`ACCEPTED.json`, technical acceptance before any W1.3/W2.3 result was read).

**Use:** descriptive only, under D3 = RESERVE. Every result is an association with an undefined target (§3). Nothing here is a blend, a gate, a weight or a tested 2026 candidate.

## 1. Coverage and exclusions (reported before any score)
- **Dates.** W1.2 registered 105 dates (6/11–9/27).
  - The 23 dates 6/11–7/03 come before the forecast captures began, so they are excluded as `before_capture_window`.
  - The other **82 dates, 7/04–9/27**, all have a qualifying forecast sheet, a resolvable round, rows for that round and complete, conflict-free lookups: 0 unit conflicts, 0 player conflicts, units complete on all 82.
- **Forecast sheets.** 217 stored `most_selected_players` sheets, none schema-invalid. Lookup sheets used: 82 each of rounds, players and units, and 82 MLB schedules from the W1.1 bundle.
- **Forecast rows for the target rounds:** 902.
  - 18 had an invalid `probabilityStarter` and were excluded, not repaired.
  - There were 0 duplicate players, 0 unmapped players, 0 batter conflicts and 0 invalid identities.
  - That leaves **884 usable forecasts**.
- **The join to our served slates (20,622 slate rows):**

| Link status | Rows | Meaning |
|---|---|---|
| `not_listed` | 19,798 | Slate-only: the batter is not on MLB's most-selected list |
| `inferred_unique_game` | **808** | Linked: MLB's sheets give the batter exactly one game that date, and it is the slate row's game |
| `multi_or_no_game` | 16 | Listed, but zero or several games that date; excluded, never split or chosen |
| `multiplicity_unknown`, `game_mismatch` | 0 | — |

  - 67 listed batters were not on our served slate at all (MLB-only).
  - Every link is **inferred**, never witnessed: forecast rows carry no unit id.
- **Shared pool.** All 808 linked rows had valid probabilities in both arms (0 invalid shared scores). By bridge stratum:

| Stratum | All rows | Verified-eligible | Surrogate | Dates |
|---|---|---|---|---|
| Selection-consistent | 630 | **541** | 541 | 63 |
| Inconsistent | 28 | 19 | 19 | 3 (7/04, 7/28, 8/21) |
| No selection (incl. the X-12 research-only dates 9/19–9/27) | 150 | 41 | 41 | 16 (15 with eligible rows) |

  - In every stratum the surrogate pool is exactly the verified pool on these linked rows: no row is surrogate-only. The two tables are therefore identical, and only one is reported.
  - **The headline stratum** is selection-consistent × verified-eligible, which is W1.2's primary. It has **541 rows on 63 dates**, a median of 9 per date (range 1–11).
  - **The pool is MLB's popularity-selected subset:** about 9 of our roughly 250 served candidates a day.

## 2. Observation recency (an observation measure, not forecast age)
- **Timing of the two sides** (headline pool, 541 rows):
  - the selected forecast sheet's run-start stamp was at about **08:00 ET** for at least three quarters of rows; 11 rows used a sheet stamped late on the previous ET date;
  - our slate's last write (`written_at`) was at a median 17:00 ET (range 11:00–21:00).
- **Gap from sheet stamp to `written_at`** (808 linked rows): median **564 minutes** (≈ 9.4 h); interquartile range 407–638; range 168–1,160.
- **Unchanged time** (808 linked rows). The row's probability had been observed unchanged for 0 minutes before its sheet in at least three quarters of rows (it first appeared in that sheet); the maximum was 1,440.
- **What these numbers can't tell us:**
  - Stamps are run-start stamps, not receipt times.
  - No later stored change exists before `written_at`. Content deduplication without a per-fetch log cannot distinguish unchanged successful fetches from failed ones, so no freshness is assumed for the gap.
  - **Reasoning, not a measurement:** on most dates our served probability was written hours after the MLB sheet we compared it with, so on timing the comparison does not obviously favour MLB.
- **`numberSelections`** (global popularity, never a weight or a forecast), over the linked rows: minimum 8, quartiles 124.75 / 175 / 272.25, maximum 599.

## 3. Target semantics: unresolved
No independent documentation or archival evidence in hand defines what `probabilityStarter` conditions on: starting, at least one plate appearance, or at least one at-bat. Both declared sensitivities are reported, and neither is selected by fit:
- **T1:** a known no_pa counts as no hit. The headline pool has 541 rows: 342 hit, 189 no_hit, 10 no_pa.
- **T2:** the no_pa rows are dropped, leaving 531 rows.

The two populations differ by 10 rows and give the same direction on every score. A calibration difference between them identifies neither the target nor the better forecaster. **W4 rank 1's target and availability kill condition stays open.**

## 4. Scores on the headline stratum (541 rows, 63 dates)
- **How the scores are computed:**
  - each score is an equal-date mean of within-date means;
  - differences are MLB − ours, so negative favours MLB on Brier and log loss;
  - brackets are 95% percentile intervals from 10,000 joint whole-date draws (seed 20261004) over 63 effective dates, with 0 failed draws;
  - AUC is computed within each date, with single-class dates omitted (T1: 3, T2: 4).

| Score | Target | Ours | MLB | MLB − ours |
|---|---|---|---|---|
| Brier | T1 | 0.2397 | 0.2343 | **−0.0053 [−0.0114, −0.0003]** |
| Brier | T2 | 0.2350 | 0.2302 | −0.0049 [−0.0110, +0.0003] |
| Log loss | T1 | 0.6752 | 0.6614 | **−0.0137 [−0.0292, −0.0009]** |
| Log loss | T2 | 0.6654 | 0.6528 | −0.0126 [−0.0284, +0.0004] |
| Stated − realized | T1 | +0.082 [0.039, 0.129] | +0.052 [0.009, 0.097] | −0.031 [−0.034, −0.027] |
| Stated − realized | T2 | +0.070 [0.025, 0.118] | +0.039 [−0.006, 0.087] | −0.031 [−0.035, −0.027] |
| Within-date AUC | T1 | 0.570 [0.511, 0.629] | 0.588 [0.529, 0.646] | +0.018 [−0.045, +0.083] |
| Within-date AUC | T2 | 0.567 [0.507, 0.627] | 0.579 [0.520, 0.639] | +0.012 [−0.053, +0.077] |

- **Proper scores.** MLB's forecast scored slightly better on this pool. The T1 intervals exclude zero by a small margin; the T2 intervals just include it. The magnitudes are small: about 0.005 Brier and 0.013 log loss.
- **Calibration in the large.** Both forecasters stated more than was realized on these popular candidates, ours by more.
  - MLB's stated probabilities averaged about 3 pp below ours on the same rows (row-weighted mean 0.678 against 0.706).
  - The difference interval is narrow because it does not depend on outcomes: the realized term cancels.
  - This does not certify either forecaster or identify the target (§3).
- **Ranking within a day.** Both are weak, and the AUC difference interval spans zero. The two forecasts correlate only moderately on this pool (row correlation 0.39).

**Shared-set top-1.** Each forecaster's highest-probability candidate within the shared pool, ranked before outcomes, voids never replaced. Both winners were target-known on all 63 dates, so T1 and T2 are identical.

| | Dates | Ours | MLB | MLB − ours |
|---|---|---|---|---|
| All common dates | 63 | 42 / 63 = 66.7% | 46 / 63 = 73.0% | +6.3 pp [−6.3, +19.0] |
| Dates the two picked different batters | 49 | 34 / 49 = 69.4% | 38 / 49 = 77.6% | +8.2 pp [−7.8, +24.4] |

Both intervals include zero. **This top-1 is not the production decision:** the pool holds only MLB's most-selected players, and our production pick, our double-down and the skip rule all operate on the full slate.

**Sensitivities, the other tables** (T1 differences, MLB − ours):
- **Selection-consistent, all rows** (630 rows, 63 dates):
  - Brier −0.0037 [−0.0068, −0.0009], log loss −0.0095 [−0.0171, −0.0024]; T2 is similar and also excludes zero;
  - AUC +0.015 [−0.037, +0.067];
  - top-1 68.3% vs 76.2%, +7.9 pp [−3.2, +20.6].
- **Inconsistent** (3 dates) and **no selection** (15–16 dates): too small to read.
  - Every proper-score difference interval includes zero, and several AUC, disagreement and encompassing intervals are unavailable because of failed draws (single-class or degenerate resamples), counted in `results.json`.
  - Inconsistent, all rows, gives a within-date AUC below 0.5 for both forecasters on 28 rows. That is a 3-date artifact, not a finding.

## 5. Residual information (gate 6, in-sample and descriptive)
- **The fits.** On the T2 headline population (531 rows, equal-date weights, unpenalized MLE, 0 boundary rows, 0 failed bootstrap refits):
  - ours alone, recalibrated: intercept +0.157, slope **0.464** on logit(ours);
  - with logit(MLB) added: ours 0.012, **MLB 1.861 [0.220, 3.633]**.
- **The log-loss change** is −0.0048 [−0.0173, −0.0001] in-sample.
  - Nested maximum-likelihood fits can never be worse in-sample, so this change is ≤ 0 by construction. That its interval stays below zero is **not** evidence of a gain.
  - The MLB coefficient's interval excluding zero is the descriptive association: conditional on our probability, MLB's carried information about the outcome on this pool, and in the joint fit ours added almost nothing beyond MLB's.
- **What it is not.** It is an in-sample association on a popularity-selected pool with an undefined target. It supplies no weights and establishes no prospective gain.
- **Our slope.** The ours-only slope below 1 says our served probabilities spread more widely than outcomes support on this pool. That is consistent with W1.3's separate flag on D's calibration (E5), though on a different population.

## 6. What this means for W4 rank 1 (reasoning only, not a measurement)
- **The case for rank 1.** The descriptive picture is consistent with MLB's own forecast holding information ours lacks, at least on the popular candidates it covers. That keeps rank 1 alive as a 2027 candidate.
- **What blocks it:**
  - **Coverage.** MLB publishes a forecast only for its most-selected players, about 9 of our ~250 daily candidates. A blend could change probabilities only for those players, and so only among them.
  - **Target.** What `probabilityStarter` predicts is undefined (§3).
  - **Availability.** The sheet we compared was usually the 08:00 ET one, and receipt times are not logged. Whether a pre-lock value is reliably available at our decision time is unmeasured.
- **What a 2027 test needs** (for the D2/D4 decisions, not done here):
  - one combination rule, fixed in advance and taken from the literature rather than fitted on these dates, with its own registration;
  - a 2027 capture of `most_selected_players` with a per-fetch receipt log, not only content-deduplicated sheets;
  - untouched 2027 validation, conditional on independently resolving the target and availability.

Nothing is nominated here; the plan allows at most one rule, named afterwards.

## Limits
- **Popularity-selected subset.** The pool is about 9 popular candidates a day, chosen by public popularity: not our slate, not our pick, not a random sample.
- **Inferred links.** Every link is inferred from a unique-game relationship, never witnessed. Eligibility follows the frozen W1.2 bridge, and verified eligibility is not upgraded by this join.
- **Window and exposure.** 7/04–9/27 only (82 dates; 63 in the headline stratum). The outcomes overlap X-21, and the dates overlap production-exposed X-01/X-09 and the X-12 research-only stratum.
- **Timing.** Observation recency is not forecast-generation age, and stamps are not receipt times (§2).
- **Target semantics** are unresolved; T1 and T2 are sensitivities, not an answer (§3).
- **Intervals** assume exchangeable dates. They exclude cross-date dependence, selection, link uncertainty and historical-availability uncertainty.
- **Gate 6** is in-sample and descriptive; its log-loss change cannot be positive by construction.
- **Shared-set top-1** is not the production decision.
