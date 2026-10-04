# W1.3 distinguishing the explanations: results memo

**Date:** 2026-10-04. **Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Design:** `docs/superpowers/specs/2026-10-04-w13-distinguishing-tests-design.md` rev 3, FROZEN. It was written before any W1.2 output existed. **Exposure:** X-29 (`8c47fb1`), published before any computation.
**Run:** `data/validation/w13_tests/e66bfe0-20261004T164027Z/` (code `e66bfe0`, frozen after two Codex code rounds), copied to `data/hetzner_results/season_wrap_outputs/w13/`. It reads the accepted W1.2 run `4e4bf5a-20261004T060725Z` through its verified `ACCEPTED.json`. The files are `results.json` (sha256 `a05e2aa3…`), `per_test.parquet` (`3f0398cb…`) and `manifest.json` (`1b97140e…`).

**How to read it.**
- Every flag is a **pointwise, unadjusted exploratory diagnostic** on one measured component. It is not a cause, an achievable lift, an equivalence, or a share of the headline-to-live gap.
- Zero inside an interval means **undetermined**, not "no effect".
- Each 95% interval comes from 10,000 whole-date draws at seed 20261004. Every interval reported here had all 10,000 draws defined.
- Sensitivity strata are in `results.json` and carry no flags.
- Descriptive only, under D3 = RESERVE.

**Primary support:** W1.2's selection-consistent × verified stratum, 63 dates and 12,726 candidate rows (7,345 hit, 4,784 no_hit, 597 no_pa).

**Selection evidence.** The decision and pick bytes were verified against the W1.2 manifest; none were missing or unmanifested, and no decision contradicted its pick. The saved per-date reconciliation shows four discrepancies (7/01, 7/04, 7/28, 8/21). On each, the authoritative decision is a skip. W1.2's frozen run had fallen back to the stale pick and labelled the date `inconsistent`, so none of the four is in the primary stratum.

## E2: numerical reconstruction (serving drift / stale inputs)
- **Flag: `unexplained_reconstruction_gap`.** Of the 12,726 primary rows, 3,321 reproduce exactly from final feeds, 2,304 only with weather blanked, and 7,101 (55.8%) are unexplained.
- **The residuals are small** (W1.2 memo §2: unexplained median 0.0017, maximum 0.075).
- **Served artifacts:** 81 dates scored, 24 `sha_unbound`.
- **This flags a reconstruction gap, not measured live drift.** Final-feed game fields, source revisions or reconstruction limits can all produce it. Historical feature age, lineup and pitcher changes, and causal staleness are not testable here. M3 (X-03) is carried, not rerun.

## E1: benchmark optimism
- **Flag: consistent (the restricted optimism component).** G(A26, B26), the actual-PA walk-forward's top-1 rate minus the estimated-PA one on their common pool, is **+17.5 pp [+7.9, +28.6]** over 63 dates, with 13 discordant dates.
- **Rank-1 picks by surface:**
  - A26's rank-1 picks went to batters who got more plate appearances: mean 5.46 scoring PA, 96.8% with five or more.
  - D's averaged 4.54 PA (54.0% with five or more).
  - The 62 genuinely selected primaries averaged 4.53 PA.
- **The rest of the chain** (W1.2 memo §4) shows no later link with an interval excluding zero except the oracle step. The C-served → D interval touches zero.
- **Oracle diagnostic.** No numerical attribution to the README's headline follows.

## E3: PA-opportunity forecast error
- **Flag: `oracle_count_consistent`.** G(A26, A26_count) = **+19.0 pp [+9.5, +30.2]**. Normalizing to the expected PA count removes the gain.
- **The composite step from A26_count to B26 is −1.6 pp [−6.3, +3.2].** It mixes context and ensemble changes, so it is undetermined.
- The scoring and A26 PA counts agreed on every common row.
- **The plan's at-lock PA forecast-error test is not testable here:** no at-lock PA forecast is archived. A positive oracle reduction shows that realized-count information carries the walk-forward's optimism. It is never an achievable improvement.

## E4: conditional hit-model drift
- **Flag: undetermined.** On numerically reproduced support (first half 23 dates and 2,105 rows; second half 40 dates and 3,520 rows), C-frozen's second-half minus first-half residual is **+0.005 [−0.032, +0.043]**.
- **The full-primary descriptions are also undetermined:**
  - C-frozen residual +0.004 [−0.019, +0.028];
  - D residual +0.004 [−0.019, +0.028];
  - D AUC +0.006 [−0.018, +0.030];
  - C-frozen AUC +0.002 [−0.023, +0.027].
- **One evolving season,** final-source context and changing composition prevent identifying conditional hit-model drift. PA-level evidence is unavailable.

## E5: calibration without ranking loss
- **Calibration flag: flagged for every surface.**
  - D: intercept −0.144 [−0.248, −0.042], slope 1.016 [0.857, 1.179]. The intercept excludes 0; the slope covers 1.
  - B26: intercept −0.123 [−0.221, −0.026].
  - C-frozen: −0.141 [−0.243, −0.040].
  - C-served: −0.145 [−0.249, −0.043].
  - A26 alone also has a slope off 1 (1.304 [1.220, 1.390]).
  - **Read together:** a uniform overprediction (stated probabilities too high by about 3 points; W1.2 §4) with no clear slope distortion on the serving surfaces.
- **Ranking: undetermined.** D−B26 equal-date AUC is −0.003 [−0.007, +0.001].
- **The joint explanation is undetermined:** no material-loss or equivalence criterion was predeclared. These same-row fits are calibration diagnostics, not held-out corrections. W1.2 carries the original-score Brier and log loss.

## E6: ball regime
- **Flag: undetermined.** Spearman ρ between registered-slate as-of drag and D's all-candidate per-date residual is **−0.05 [−0.31, +0.22]**, over 62 complete dates (1 incomplete; no conflicting drag keys).
- **Companions (no flag):** C-frozen ρ −0.05; D rank-1 residual ρ −0.23.
- **The full, independently dated regime test is not testable from these inputs.** This narrow association neither supports nor weakens a ball regime.

## E7: selection / composition (descriptive, no label)
- **Discrimination is within slates, not across days.** D's within-date AUC is 0.566 [0.555, 0.577] and its pooled AUC 0.565 [0.555, 0.576]; they are nearly identical.
- **Rank-1 picks overpredict more than candidates overall:** stated minus realized is +0.060 at rank 1 (63 known dates) against +0.031 across all candidates.
- **Residual by recommendation action** (all registered dates, verified pool):
  - double: +0.031 (55 dates);
  - single: +0.029 (7 dates);
  - skip: +0.023 (10 dates);
  - skip inside the 9/19–9/27 window: +0.035 (9 dates). That window is a date range, not evidence of research-only provenance.
  - Action is the recommendation, not contest entry.
- **Within the primary stratum:**
  - projected-lineup rows: +0.053 (3,006 rows);
  - confirmed rows: +0.026 (9,720);
  - slates above the median size: +0.047 (21 dates);
  - slates at or below it: +0.023 (42 dates).
  - Mean valid D is about 0.63–0.64 in every stratum.

## E8: sampling variation
- **Flag: null-compatible.** The 62 genuinely selected primaries in the primary stratum with known outcomes produced **44 hits against 48.0 expected** at their served probabilities. The independent-Bernoulli 95% envelope is [41, 54].
- **Support:** 25 skip dates, 3 with no selection, and 15 selected primaries outside the primary stratum.
- **All-registered sensitivity (no flag):** 59 hits against 59.9 expected, envelope [52, 67].
- **This assumes calibrated served p and independence across dates.** E5 finds the served p overstated, so the expected count is itself high. Compatibility does not mean luck explains the gap.
- **Carried context (X-09):** Wilson intervals on the brief's counts are 60.0–75.2% for 96/141 and 63.4–80.0% for 79/109. They are not this window.

## The eight explanations together

| # | Explanation | Measured component | Flag | Full explanation |
|---|---|---|---|---|
| E1 | Benchmark optimism | G(A26,B26) +17.5 pp [+7.9, +28.6] | consistent | oracle; composite chain; no attribution |
| E2 | Serving drift / stale inputs | 55.8% unexplained reconstruction, small residuals | unexplained reconstruction gap | live drift not measured; feature age not testable |
| E3 | PA-opportunity forecast error | G(A26,A26_count) +19.0 pp [+9.5, +30.2] | oracle count consistent | at-lock PA forecast not testable |
| E4 | Conditional hit-model drift | reproduced-support residual contrast +0.005 [−0.032, +0.043] | undetermined | not identified |
| E5 | Calibration without ranking loss | D intercept −0.144 [−0.248, −0.042]; slope covers 1; AUC difference covers 0 | calibration flagged; ranking undetermined | joint undetermined |
| E6 | Ball regime | drag ρ −0.05 [−0.31, +0.22] | undetermined | full test not testable |
| E7 | Selection / composition | within ≈ pooled AUC; rank-1 overpredicts more | descriptive | — |
| E8 | Sampling variation | 44 vs 48.0 expected, envelope [41, 54] | null-compatible | not a luck verdict |

**What this adds for the decision memo, as reasoning only:**
- The README-style 86% sits on realized-count information (E1/E3).
- The serving recipe's stated probabilities run about 3 points high at the slate level, and about 6 at rank 1 (E5/E7).
- Its within-slate discrimination is modest and stable across the window (E4/E7).
- Nothing here measures drift, regime or luck as a cause.

The calibration offset is the clearest component. A 2027 calibration map is W4 rank 4a: identity versus one regularized intercept map, gated on held-out proper scores. Under D3 that validates on 2027 dates only.

## Limits
- **Window:** 63 primary dates (from 105 registered served dates since 6/11). Intervals are wide, and several flags rest on few dates.
- **No overall error rate.** Many components and strata were examined, and the flags carry no overall error-rate guarantee.
- **W1.2's limits carry over:**
  - final-feed game fields;
  - the weather class as a sensitivity only;
  - one seed with deterministic training;
  - fits that used all 2026 PA before each retrain date.
- **The drag export** is source-date-causal by construction, not proof of what was available live. Its daily producer re-ran after the season, and the bytes used are hashed in the run manifest.
- **Exchangeable dates assumed.** The date-block intervals assume exchangeable dates and exclude cross-date dependence and selection effects.
