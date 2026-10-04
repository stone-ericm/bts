# W1.3 distinguishing the explanations: results memo

**Date:** 2026-10-04. **Status:** FROZEN: Codex memo r1 SIGN WITH EDITS (`docs/audit/2026-10-04-w13-memo-codex-r1.md`), edits 1–10 applied verbatim by script.
**Design:** `docs/superpowers/specs/2026-10-04-w13-distinguishing-tests-design.md` rev 3, FROZEN. It was written before any W1.2 output existed. **Exposure:** X-29 (`8c47fb1`), published before any computation.
**Run:** `data/validation/w13_tests/e66bfe0-20261004T164027Z/` (code `e66bfe0`, frozen after two Codex code rounds), copied to `data/hetzner_results/season_wrap_outputs/w13/`. It reads the accepted W1.2 run `4e4bf5a-20261004T060725Z` through its verified `ACCEPTED.json`. The files are `results.json` (sha256 `a05e2aa3…`), `per_test.parquet` (`3f0398cb…`) and `manifest.json` (`1b97140e…`).

**How to read it.**
- Every flag is a **pointwise, unadjusted exploratory diagnostic** on one measured component. It is not a cause, an achievable lift, an equivalence, or a share of the headline-to-live gap.
- Zero inside an interval means **undetermined**, not "no effect".
- Newly computed 95% bootstrap intervals use 10,000 whole-date draws at seed 20261004, and every saved W1.3 bootstrap interval has all draws defined. Carried W1.2 chain intervals retain that memo's recording limits. E8's 95% envelope instead uses 10,000 independent-Bernoulli vectors at the same seed; the carried X-09 Wilson intervals are analytic.
- Sensitivity strata are in `results.json` and carry no flags.
- Descriptive only, under D3 = RESERVE.

**Primary support:** W1.2's selection-consistent × verified stratum, 63 dates and 12,726 candidate rows (7,345 hit, 4,784 no_hit, 597 no_pa).

**Selection evidence.** The runner reports verifying consumed decision and pick bytes against the W1.2 manifest, with no expected-but-missing or present-but-unmanifested files reported. Sources are authoritative decisions on 92 dates, pick fallback on 10, and no selection source on 3. It recorded zero differing primary `(batter_id, game_pk)` keys where both decision and pick supplied one; that check does not certify agreement on action, probability, DD or delivery. The saved per-date reconciliation shows four source discrepancies (7/01, 7/04, 7/28, 8/21). On each, the authoritative decision is a skip with no genuine selected primary. W1.2's frozen metadata instead records source `pick` and state `inconsistent`, so none is in the primary stratum. The accepted W1.2 cohort is preserved; numerical selection consistency is not delivery or contest-entry evidence.

## E2: numerical reconstruction (serving drift / stale inputs)
- **Flag: `unexplained_reconstruction_gap`.** Of the 12,726 primary rows, 3,321 reproduce exactly from final feeds, 2,304 only with weather blanked, and 7,101 (55.8%) are unexplained.
- **Primary numerical magnitudes:** across all 12,726 finite reconstructed primary rows, serialized C-served minus D has signed p01/median/p99 −0.04938 / 0 / +0.03611; absolute median/p90/maximum are 0.00115 / 0.01878 / 0.09447. These are all-class magnitudes, including the original unblanked C-served discrepancy on weather-sensitivity rows. Unexplained-only primary quantiles are not supplied. W1.2's unexplained median 0.0017 and maximum 0.075 describe its broader selection-consistent population, not this primary unexplained subset. No materiality threshold was registered.
- **Primary reconstruction classes by composition** (counts; no stratum flags):

| Stratum | Exact final feed | Weather-blank sensitivity | Unexplained | Total | Unexplained share |
|---|---:|---:|---:|---:|---:|
| July | 1,080 | 774 | 2,322 | 4,176 | 55.6% |
| August | 1,247 | 756 | 3,037 | 5,040 | 60.3% |
| September | 994 | 774 | 1,742 | 3,510 | 49.6% |
| Projected | 101 | 2,304 | 601 | 3,006 | 20.0% |
| Confirmed | 3,220 | 0 | 6,500 | 9,720 | 66.9% |

Month and lineup are separate partitions. Weather blanking is a numerical sensitivity, not verified historical weather provenance or a cause.
- **Served artifacts:** 81 dates scored, 24 `sha_unbound`.
- **This flags a reconstruction gap, not measured live drift.** Final-feed game fields, source revisions or reconstruction limits can all produce it. Historical feature age, lineup and pitcher changes, and causal staleness are not testable here. M3 (X-03) is carried, not rerun.

## E1: benchmark optimism
- **Flag: consistent (the restricted optimism component).** G(A26, B26), the actual-PA walk-forward's top-1 rate minus the estimated-PA one's, is **+17.5 pp [+7.9, +28.6]** on 11,706 common finite-score candidate rows and 63 common-known winner dates, with 13 discordant dates and no unilateral unknown/no_pa winner exclusions. Both surfaces are oracle diagnostics; the 12,726-row primary pool is larger than this paired common pool.
- **Rank-1 picks by surface:**
  - A26's rank-1 picks went to batters who got more plate appearances: mean 5.46 scoring PA, 96.8% with five or more.
  - D's averaged 4.54 PA (54.0% with five or more).
  - The 62 genuinely selected primaries averaged 4.53 PA.
- **The carried top-1 chain** (W1.2 memo §4) has no later top-1 interval excluding zero. The C-served → D top-1 interval touches zero. This statement concerns top-1 changes, not every score metric, and does not establish equivalence or historical input parity.
- **Oracle diagnostic.** No numerical attribution to the README's headline follows.

## E3: PA-opportunity forecast error
- **Flag: `oracle_count_consistent`.** G(A26, A26_count) = **+19.0 pp [+9.5, +30.2]** on 12,129 common finite-score rows and 63 common-known winner dates (14 discordant). Count normalization lowers the diagnostic top-1 rate; the mean row-level A26 minus A26_count score change is instead −0.0125 on those common rows. These are different statistics, not a uniform probability reduction.
- **The composite step from A26_count to B26 is −1.6 pp [−6.3, +3.2]** on 11,706 common finite-score rows and 63 common-known winner dates (3 discordant). Both comparisons have no unilateral unknown/no_pa winner exclusions. The composite mixes context and ensemble changes; its zero-covering interval is undetermined and does not prove the remaining difference absent.
- The scoring and A26 PA counts agreed on every common row.
- **The plan's at-lock PA forecast-error test is not testable here:** no at-lock PA forecast is archived. Within this window, the actual-PA walk-forward's advantage over count-normalized scoring rests on realized-count information. That is an oracle contrast: never an achievable improvement, and it assigns no share of the historical headline-to-live gap.

## E4: conditional hit-model drift
- **Flag: undetermined.** On numerically reproduced support (first half 23 dates and 2,105 rows; second half 40 dates and 3,520 rows), C-frozen's second-half minus first-half residual is **+0.005 [−0.032, +0.043]**.
- **The full-primary descriptions are also undetermined:**
  - C-frozen residual +0.004 [−0.019, +0.028];
  - D residual +0.004 [−0.019, +0.028];
  - D AUC +0.006 [−0.018, +0.030];
  - C-frozen AUC +0.002 [−0.023, +0.027].
- **One evolving season,** final-source context and changing composition prevent identifying conditional hit-model drift. PA-level evidence is unavailable.
- **Split and support qualification:** the frozen registered split is 6/11–8/06 (53 dates) versus 8/07–9/27 (52 dates). Reproduced support has 23 and 40 effective dates with known supported rows. Its 2,105 first-half rows comprise 1,259 exact and 846 weather-sensitivity matches; its 3,520 second-half rows comprise 2,062 exact and 1,458 weather-sensitivity matches. These row totals precede outcome filtering. Known-row counts and excluded candidate-row counts by half were not serialized and remain unavailable. The reproduced-support flag is conditional on numerical reconstruction, including inferred weather sensitivity, and does not establish historical serving parity.

## E5: calibration without ranking loss
- **Fit support:** six valid candidate-row-weighted logistic fits, each on 63 dates. B26 uses 11,706 known finite-score rows; each other surface uses 12,129. Boundary, low-clipping and high-clipping counts are zero for every base fit. Bootstrap parameters are refitted in each draw; these diagnostics are not deployed calibration maps.
- **Calibration flag: flagged for every surface.**
  - D: intercept −0.144 [−0.248, −0.042], slope 1.016 [0.857, 1.179]. The intercept excludes 0; the slope covers 1.
  - B26: intercept −0.123 [−0.221, −0.026].
  - C-frozen: −0.141 [−0.243, −0.040].
  - C-served: −0.145 [−0.249, −0.043].
  - A26 alone also has a slope off 1 (1.304 [1.220, 1.390]).
  - **Read together:** the negative recalibration intercepts flag departures from identity under this logistic diagnostic. D's candidate-row-weighted stated-minus-realized point residual is +0.031 on 12,129 known rows. This does not establish a uniform probability offset across candidates or dates. The D, C-frozen and C-served slope intervals include 1, so slope distortion remains undetermined rather than absent.
- **Ranking: undetermined.** D−B26 equal-date AUC is −0.003 [−0.007, +0.001] on 63 dates and 11,706 common known finite-score rows. B26 is an oracle comparator. Zero inclusion does not establish preserved ranking or equivalence.
- **The joint explanation is undetermined:** no material-loss or equivalence criterion was predeclared. These same-row fits are calibration diagnostics, not held-out corrections. W1.2 carries the original-score Brier and log loss.

## E6: ball regime
- **Flag: undetermined.** Spearman ρ between registered-slate as-of drag and D's all-candidate per-date residual is **−0.05 [−0.31, +0.22]**, over 62 complete dates (1 incomplete; no conflicting drag keys).
- **Companions (no flag):** C-frozen ρ −0.05; D rank-1 residual ρ −0.23.
- **Candidate support:** 12,726 common finite D/C-frozen rows, including 12,129 known outcomes before the drag-completeness restriction. Each reported correlation uses 62 complete dates; candidate-row and known-row totals after dropping the incomplete date were not serialized and remain unavailable.
- **The full, independently dated regime test is not testable from these inputs.** This narrow association neither supports nor weakens a ball regime.

## E7: selection / composition (descriptive, no label)
- **AUC point descriptions:** D's equal-date within-slate AUC is 0.566 [0.555, 0.577] over 63 dates; pooled AUC is 0.565 [0.555, 0.576] on 12,129 known finite-D rows. The point estimates are close. This does not establish equivalence or locate discrimination exclusively within slates; pooled AUC includes cross-date comparisons.
- **Residual point descriptions:** stated minus realized is +0.060 at the native D rank-1 winner (63 known dates), versus the candidate-row-weighted +0.031 across 12,129 known finite-D rows. These are different weights and populations; no significance or causal selection-effect claim is attached.
- **Recommendation-action strata:** all registered dates with verified-pool rows, including unknown action separately. These cells cover 82 dates, 14,094 rows and 13,457 known outcomes. Action is the frozen recommendation, not contest entry. The 9/19–9/27 label is a date range, not evidence of research-only provenance.

| Action / window | Dates | Total rows | Known rows | hit / no_hit / no_pa | Mean valid D | Known-row residual |
|---|---:|---:|---:|---|---:|---:|
| double | 55 | 11,070 | 10,550 | 6,396 / 4,154 / 520 | 0.636 | +0.031 |
| single | 7 | 1,494 | 1,431 | 868 / 563 / 63 | 0.634 | +0.029 |
| skip outside 9/19–9/27 | 10 | 756 | 735 | 455 / 280 / 21 | 0.640 | +0.023 |
| skip inside 9/19–9/27 | 9 | 612 | 593 | 348 / 245 / 19 | 0.623 | +0.035 |
| unknown | 1 | 162 | 148 | 81 / 67 / 14 | 0.630 | +0.084 |

**Primary composition strata:** the size threshold is the frozen median 270 rows over the 105 registered slates.

| Stratum | Dates | Total rows | Known rows | hit / no_hit / no_pa | Mean valid D | Known-row residual |
|---|---:|---:|---:|---|---:|---:|
| projected | 53 | 3,006 | 2,465 | 1,441 / 1,024 / 541 | 0.633 | +0.053 |
| confirmed | 63 | 9,720 | 9,664 | 5,904 / 3,760 / 56 | 0.637 | +0.026 |
| slate size >270 | 21 | 4,230 | 4,046 | 2,380 / 1,666 / 184 | 0.635 | +0.047 |
| slate size ≤270 | 42 | 8,496 | 8,083 | 4,965 / 3,118 / 413 | 0.636 | +0.023 |

All D scores in these displayed strata are valid; missing and invalid D counts are zero, as are unknown-outcome counts. Mean-p includes valid no_pa rows; residuals use only known hit/no_hit rows and are candidate-row-weighted. Projected/confirmed dates overlap; lineup and size are separate row partitions. These remain composition descriptions without automatic flags, significance comparisons or causal attribution.

## E8: sampling variation
- **Flag: null-compatible.** The 62 genuinely selected primaries in the primary stratum with known outcomes produced **44 hits against 48.0 expected** at their served probabilities. The independent-Bernoulli 95% envelope is [41, 54].
- **Support:** 25 skip dates, 3 with no selection, and 15 selected primaries outside the primary stratum.
- **All-registered sensitivity (no flag):** 77 genuinely selected known primaries, 59 hits against 59.9 expected, envelope [52, 67].
- **Conditional null only:** this assumes calibrated served probabilities and conditional independence across dates; neither assumption is certified here. The 48.0 expected count is the unadjusted sum of the selected served probabilities under that null. E5 fits the same window's candidate outcomes and does not establish calibration or an adjusted expected count for these 62 selected primaries; calibration circularity remains a limit. Compatibility does not mean luck explains the gap.
- **Carried context (X-09):** Wilson intervals on the brief's counts are 60.0–75.2% for 96/141 and 63.4–80.0% for 79/109. They are not this window.

## The eight explanations together

| # | Explanation | Measured component | Flag | Full explanation |
|---|---|---|---|---|
| E1 | Benchmark optimism | G(A26,B26) +17.5 pp [+7.9, +28.6] | consistent | oracle; composite chain; no attribution |
| E2 | Serving drift / stale inputs | 55.8% unexplained; all-primary absolute median 0.00115, maximum 0.09447 | unexplained reconstruction gap | live drift not measured; feature age not testable |
| E3 | PA-opportunity forecast error | G(A26,A26_count) +19.0 pp [+9.5, +30.2] | oracle count consistent | at-lock PA forecast not testable |
| E4 | Conditional hit-model drift | reproduced-support residual contrast +0.005 [−0.032, +0.043] | undetermined | not identified |
| E5 | Calibration without ranking loss | D intercept −0.144 [−0.248, −0.042]; slope covers 1; AUC difference covers 0 | calibration flagged; ranking undetermined | joint undetermined |
| E6 | Ball regime | drag ρ −0.05 [−0.31, +0.22] | undetermined | full test not testable |
| E7 | Selection / composition | within AUC 0.566; pooled 0.565; rank-1 residual +0.060 versus candidate +0.031 | descriptive | no equivalence or causal selection claim |
| E8 | Sampling variation | 44 vs 48.0 expected, envelope [41, 54] | null-compatible | not a luck verdict |

**What this adds for the decision memo, as reasoning only:**
- The largest displayed top-1 contrasts in this window involve realized-count oracle information (E1/E3). They do not reproduce the README's historical 86.2% or assign a share of the headline-to-live gap. The pointwise exploratory diagnostics cannot rank the eight explanations.
- D's stated-minus-realized point residual is +0.031 on 12,129 known candidate rows and +0.060 at its 63 native rank-1 winners. These conditional population descriptions do not establish a uniform or stable calibration error.
- Within-date AUC is 0.566 on 63 dates. E4's half contrasts are undetermined; they do not establish stability across the window. The close within-date and pooled AUC point estimates do not establish equivalence.
- Nothing here identifies drift, regime, selection or luck as a cause, or measures an achievable improvement.

The existing conditional W4 rank 4a slot specifies identity versus one regularized intercept map, with a slope challenger only with support and a held-out proper-score improvement gate. This read neither nominates nor tests a map, selects that slot for execution, nor validates a correction. Any candidate selected later needs its own registration, approved computation and independent acceptance; under D3, validation of an idea motivated here uses 2027 dates only.

## Limits
- **Window:** 63 primary dates (from 105 registered served dates since 6/11). Intervals are wide, and several flags rest on few dates.
- **No overall error rate.** Many components and strata were examined, and the flags carry no overall error-rate guarantee.
- **W1.2's limits carry over:**
  - final-feed game fields;
  - the weather class as a sensitivity only;
  - one seed with deterministic training;
  - fits that used all 2026 PA before each retrain date.
- **The drag export** is source-date-causal by construction, not proof of what was available live. Its daily producer re-ran after the season, and the bytes used are hashed in the run manifest.
- **Conditional date-block intervals:** they condition on frozen artifacts and scoreable support and assume exchangeable dates. They exclude fitting-history uncertainty, selection bias and cross-date dependence, including seasonal trends and overlapping drag windows. E5's within-window diagnostic refits do not include uncertainty from retraining the original forecasting models.
