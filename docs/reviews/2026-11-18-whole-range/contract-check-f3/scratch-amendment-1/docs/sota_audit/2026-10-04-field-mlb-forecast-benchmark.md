# W2.3 MLB's own forecast against ours: descriptive benchmark (season wrap)

**Date:** 2026-10-04. **Status:** FROZEN 2026-10-04 after Codex memo r1 SIGN WITH EDITS (edits 1–6 applied verbatim by script; review archived at `docs/audit/2026-10-04-w23-memo-codex-r1.md`), plus one author factual addition in §6 (the all-player field), marked there. Pace rule: no further review rounds.
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
| Frozen `no_selection` (includes the X-12 window; separate research-only results unavailable) | 150 | 41 | 41 | 16 (15 with eligible rows) |

  - In every stratum the surrogate pool is exactly the verified pool on these linked rows: no row is surrogate-only. The two tables are therefore identical, and only one is reported.
  - **The headline stratum** is selection-consistent × verified-eligible, which is W1.2's primary. It has **541 rows on 63 dates**, a median of 9 per date (range 1–11).
  - **The pool is MLB's popularity-selected subset:** about 9 of our roughly 250 served candidates a day.

## 2. Observation recency (an observation measure, not forecast age)
- **Timing of the two sides** (headline pool, 541 rows):
  - The author-reported `joined.parquet` timing summary places the selected forecast sheet's run-start stamp in the **08:00–08:59 ET hour** for at least three quarters of headline rows; 11 rows use a sheet stamped on the previous ET date.
  - The reported `written_at` ET-hour quantiles (minimum / 25th / median / 75th / maximum) are **11 / 13 / 17 / 18 / 21**. These are hour bins, not exact clock times. `written_at` is the slate's last write, not a verified probability-refresh or lock time.
- **Gap from sheet stamp to `written_at`** (808 linked rows): median **564 minutes** (≈ 9.4 h); interquartile range 407–638; range 168–1,160.
- **Recorded unchanged span** (808 linked rows): the 75th percentile is 0 minutes and the maximum is approximately 1,440 minutes. Zero means no earlier consecutive stored sheet extends the run of the same player/round probability value; it does not establish the player's first appearance or the probability's generation time. Gaps without retained fetch history remain unresolved.
- **What these numbers can't tell us:**
  - Stamps are run-start stamps, not receipt times.
  - No later stored change exists before `written_at`. Content deduplication without a per-fetch log cannot distinguish unchanged successful fetches from failed ones, so no freshness is assumed for the gap.
  - **Timing interpretation remains unresolved:** the earlier sheet stamp and later slate write do not identify which forecaster had a more favorable information set. Neither timestamp verifies forecast generation, a probability refresh, reliable pre-lock receipt or the actual decision/lock time.
- **`numberSelections`** (global popularity, never a weight or a forecast), over the linked rows: minimum 8, quartiles 124.75 / 175 / 272.25, maximum 599.

## 3. Target semantics: unresolved
No independent documentation or archival evidence in hand defines what `probabilityStarter` conditions on: starting, at least one plate appearance, or at least one at-bat. Both declared sensitivities are reported, and neither is selected by fit:
- **T1:** a known no_pa counts as no hit. The headline pool has 541 rows: 342 hit, 189 no_hit, 10 no_pa.
- **T2:** the no_pa rows are dropped, leaving 531 rows.

The headline target populations differ by 10 rows, and the MLB-minus-ours point differences have the same direction across the displayed headline scores under T1 and T2. This does not identify the target, establish equivalence between the sensitivities or certify the better forecaster; other strata need not share that direction. **W4 rank 1's target and availability kill condition stays open.**

## 4. Scores on the headline stratum (541 rows, 63 dates)
- **How the scores are computed:**
  - each score is an equal-date mean of within-date means;
  - differences are MLB − ours, so negative favours MLB on Brier and log loss;
  - brackets are pointwise, unadjusted 95% percentile intervals from 10,000 joint whole-date draws at seed 20261004, with zero failed draws on the headline results;
  - proper scores use 63 effective dates under both targets, with 541 T1 rows and 531 T2 rows. The bootstrap samples the 63 date records jointly. Within-date AUC omits single-class dates/copies: its base support is 60 dates under T1 (3 omitted) and 59 under T2 (4 omitted).

| Score | Target | Ours | MLB | MLB − ours |
|---|---|---|---|---|
| Brier | T1 | 0.2397 | 0.2343 | **−0.0053 [−0.0114, −0.0003]** |
| Brier | T2 | 0.2350 | 0.2302 | −0.0049 [−0.0110, +0.0003] |
| Log loss | T1 | 0.6752 | 0.6614 | **−0.0137 [−0.0292, −0.0009]** |
| Log loss | T2 | 0.6654 | 0.6528 | −0.0126 [−0.0284, +0.0004] |
| Stated − realized | T1 | +0.082 [0.039, 0.128] | +0.052 [0.009, 0.097] | −0.030 [−0.034, −0.027] |
| Stated − realized | T2 | +0.070 [0.025, 0.118] | +0.039 [−0.006, 0.087] | −0.031 [−0.035, −0.027] |
| Within-date AUC | T1 | 0.570 [0.510, 0.629] | 0.588 [0.529, 0.646] | +0.018 [−0.045, +0.083] |
| Within-date AUC | T2 | 0.567 [0.507, 0.627] | 0.579 [0.520, 0.639] | +0.012 [−0.053, +0.077] |

- **Proper-score point comparison:** MLB has lower Brier and log-loss point estimates under both target sensitivities on this shared population. The pointwise, unadjusted T1 difference intervals exclude zero by a small margin; T2's include zero. These are conditional target-sensitivity descriptions, not a general or prospective superiority finding, and no practical-effect threshold was registered for this descriptive read.
- **Calibration-in-the-large descriptions:** both headline residual point estimates are positive, with ours larger. MLB's T2 residual interval includes zero; neither forecaster's calibration is certified. These equal-date descriptions do not establish uniform error or a validated correction.
  - The author-reported row-weighted mean probabilities from `joined.parquet` are approximately 0.7064 for ours and 0.6781 for MLB. These weights differ from the equal-date residual aggregation.
  - On each matched target population, the realized term cancels exactly in the residual difference. Its interval describes the difference in stated probabilities; it is not by itself a comparison of predictive accuracy or evidence defining the target.
- **Within-date discrimination:** AUC point estimates are approximately 0.57 for ours and 0.59 for MLB. Both difference intervals include zero, leaving the comparison undetermined. The author-reported row correlation is 0.389 on the headline pool; it is a descriptive probability correlation, not evidence of independent predictive gain.

**Shared-set top-1.** Each forecaster's highest-probability candidate within the shared pool, ranked before outcomes, voids never replaced. Both winners were target-known on all 63 dates, so T1 and T2 are identical.

| | Dates | Ours | MLB | MLB − ours |
|---|---|---|---|---|
| All common dates | 63 | 42 / 63 = 66.7% | 46 / 63 = 73.0% | +6.3 pp [−6.3, +19.0] |
| Dates the two picked different batters | 49 | 34 / 49 = 69.4% | 38 / 49 = 77.6% | +8.2 pp [−7.8, +24.4] |

Both intervals include zero. **This top-1 is not the production decision:** the pool holds only MLB's most-selected players, and our production pick, our double-down and the skip rule all operate on the full slate.

**Sensitivities, the other tables** (T1 differences, MLB − ours):
- **Selection-consistent, all rows** (63 dates; 630 T1 rows, 620 T2 rows):
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
  - Both valid unpenalized nested MLE fits optimize and are evaluated with the same equal-date weights on the same T2 rows. The full model contains the ours-only solution with the MLB coefficient set to zero, so its in-sample log loss cannot increase apart from numerical precision. The negative training-loss interval therefore does not establish a prospective gain.
  - The positive MLB-coefficient interval is a pointwise descriptive conditional association under the specified additive-logit model, beyond its ours-only recalibration term. The full-model ours coefficient of 0.012 is only a point estimate: no coefficient interval or MLB-only comparison was computed, so the reverse question of what ours adds beyond MLB is unavailable.
- **What it is not.** This is an in-sample association on a popularity-selected pool with unresolved target semantics. It supplies no deployed weights, identifies no cause and establishes no prospective gain.
- **Our slope:** the ours-only point fit has slope 0.464 and compresses logit(ours) on this T2 population. No interval for that slope was recorded, so a departure from 1 is not established here. W1.3 uses a different candidate population and row weighting; its D slope interval includes 1 and its calibration flag comes from the intercept. These are separate diagnostics, not replication of the same distortion.

## 6. What this means for W4 rank 1 (reasoning only, not a measurement)
- **Conditional hypothesis:** the positive MLB coefficient is compatible with additional outcome association under the specified gate-6 model on the popular candidates it covers. The existing W4 rank 1 slot remains conditional; this does not establish a usable gain, select a candidate for execution or pass its target/availability gate.
- **Open constraints:**
  - **Coverage:** the headline shared pool has an author-reported median nine candidates per date. A rule can directly adjust scores only where an admissible MLB forecast exists, but those changes can alter rankings, selections, DD partners or skip decisions across the full slate. Unlisted and unavailable forecasts require a frozen fallback; their scores are not inferred from popularity.
  - **All-player field (author addition after review r1; not read):** MLB's `players` sheet also carries `probabilityStarter` for every player, but with no round id, so which game a value refers to is unresolved. The frozen design and X-23 exclude it. Whether it can be bound to a game is a question for any 2027 registration (`docs/ops/2027-season-start.md` §D).
  - **Target:** independent evidence must define what `probabilityStarter` predicts and establish the intended comparison; fit under T1/T2 does not resolve it.
  - **Availability and identity:** the retained run-start stamps do not witness reliable pre-lock receipt. Same-game linkage, eligibility and decision-time availability must be established for any prospective use; this retrospective inferred link is not upgraded to witnessed linkage.
- **A later 2027 test would require** (for the D2/D4 decisions; not done or authorized here):
  - at most one literature-based combination rule, fixed before validation, with its own registration covering the baseline, as-of inputs, valid/missing/invalid or unavailable forecast handling, temporal split, primary metric, practical-effect threshold, family control, guardrails and stopping/disposition rules;
  - a registered 2027 capture with per-fetch success/failure and receipt records bound to the response bytes, including unchanged successful responses; such records witness retrieval availability, not the provider's model-generation age;
  - independently resolved target and availability, then untouched 2027 evaluation of the frozen baseline versus the resulting forecast rule across its declared full-slate support, including the fallback and coverage limits, followed by independent acceptance. Shared-set top-1 or an in-sample coefficient is not the production-value gate.

Nothing is nominated here. Later selection and approved computation remain separate steps; no blend, gate or weight search on these 2026 dates is authorized.

## Limits
- **Popularity-selected subset.** The pool is about 9 popular candidates a day, chosen by public popularity: not our slate, not our pick, not a random sample.
- **Inferred links.** Every link is inferred from a unique-game relationship, never witnessed. Eligibility follows the frozen W1.2 bridge, and verified eligibility is not upgraded by this join.
- **Window and exposure.** 7/04–9/27 on 82 registered dates, with 63 in the headline stratum. Outcomes overlap X-21; dates overlap production-exposed X-01/X-09 and the X-12 research-only window. `results.json` reports aggregate frozen `no_selection` strata, not a separate X-12 research-only partition. Its separate coverage/outcome/score account is unavailable, so the aggregate no-selection results must not be read as that cohort's estimate or as proof of research-only provenance for every date.
- **Timing.** Observation recency is not forecast-generation age, and stamps are not receipt times (§2).
- **Target semantics** are unresolved; T1 and T2 are sensitivities, not an answer (§3).
- **Intervals** are pointwise and unadjusted, with no simultaneous error-rate guarantee across targets, scores or strata. They condition on archived support and assume exchangeable dates, excluding cross-date dependence, selection, link uncertainty and historical-availability uncertainty. Failed-draw nominal intervals remain unavailable; successful-draw conditional endpoints do not supply findings. Zero inclusion is undetermined, not equivalence or absence of an effect.
- **Gate 6** is in-sample and descriptive. For valid nested MLE fits evaluated on the same rows with the same equal-date weights, training log loss cannot increase apart from numerical precision; its reduction is not a prospective-gain test. Reverse incremental contribution and the ours-only slope interval were not computed.
- **Shared-set top-1** is not the production decision.
