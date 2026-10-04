# W1.2 benchmark bridge: results memo

**Status:** rev 2, FROZEN: Codex memo r1 SIGN WITH EDITS (`docs/audit/2026-10-04-benchmark-bridge-memo-codex-r1.md`), edits E1–E6 applied verbatim by script. Analytical values come from the accepted run's `summary.json`; operational statements are acceptance-record attestations where identified.
**Design:** `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md` rev 3, FROZEN. **Exposure:** X-21 (`e09a7b7`), predeclared before any outcome-bearing execution.
**Use:** descriptive only, under D3 = RESERVE. Nothing here nominates or tests a candidate.

## 1. The run
- **Accepted run:** `data/validation/w12_bridge/4e4bf5a-20261004T060725Z/`, code `4e4bf5a`. The acceptance record attests completion at 2026-10-04 05:07:45 ET and technical acceptance before result reads for W1.3/W2.3. Its criteria concern completion, full-run scope, the two pre-output code fixes, PA input consistency and prior X-21 publication; none depends on favorable analytical results.
- **Verified copied output:** the copied `summary.json` matches the acceptance record's SHA256 `2373a77110c31124df392b8d193f62da7ba3fe7f8a2b97eb2902b0cdd97545b0`. It records false `skip_ab`/`skip_frozen` flags, 105 registered dates and a season-game-date denominator of 184, with missing fraction 0.429. The receipt also lists `table.parquet` (`accb6d48…`) and `manifest.json` (`dae86f08…`); those files were not independently inspected in this memo review.
- **Pre-output fixes:** the receipt attests that earlier attempts produced no output and that the accepted code includes slim feeds (`24129fd`) and reading plain plus gzipped unit captures (`4e4bf5a`). Source inspection confirms those changes. The accepted summary reports 14,094 verified-pool candidate rows; that coverage count alone does not certify every reconstructed eligibility edge.
- **Input and execution limits:** the receipt attests that the 2026 PA input retained SHA256 prefix `06264f2f…` across the 03:00 rebuild. The underlying logs and start-time input pins were not provided here. The executed runner writes its manifest at the end, so these copies alone do not establish a full input manifest frozen before execution. They also do not record the actual CLI resample count or fully bind the process environment. Technical acceptance is separate from numerical reproduction and complete historical provenance; no general claim that operational changes left every input and result unaffected follows from this review.

## 2. Reproduction (C-served vs D), first
On the 77 selection-consistent dates, 19,050 rows have both a C-served and a D score:

| Class | Rows | Share |
|---|---|---|
| `exact_final_feed` (within 1e-9) | 7,532 | 39.5% |
| `inferred_weather_absent_at_serve` | 2,808 | 14.7% |
| `unexplained` | 8,710 | 45.7% |

- **Absolute score discrepancies:** among unexplained rows, the median is 0.0017, the 90th percentile 0.0069 and the maximum 0.075. Across all 19,050 reconstructed rows, the absolute discrepancy's 99th percentile is 0.049. These are magnitudes, not signed residuals or verified historical causes.
- **Reproduction differs by reconstructed pool.** In the runner's verified-status plus scheduled-time pool at the slate's last written_at, 7,101 of 12,726 reconstructed rows (55.8%) are unexplained, compared with 8,710 of 19,050 (45.7%) overall. The summary does not separate reproduction classes by contemporaneous playing status, so it does not establish that already-underway games reproduce exactly or identify why the rates differ. Weather blanking establishes numerical sensitivity only; historical serving values and residual causes remain unverified.
- **Served artifacts:** C-served scored 81 of 105 dates; 24 are `sha_unbound`, meaning no matching archived-model hash binding. Their recorded actions are 21 skips and three unknown. The status alone does not distinguish an absent binding from a mismatching hash.

## 3. Coverage and denominators
- **Registered dates:** 105 served slates (6/11 → 9/27) of 184 season game dates.
- **Selection states:** 77 selection-consistent, 4 inconsistent, 24 no selection.
- **Candidate rows:** 26,399; pools: verified 14,094, surrogate 16,956, all 26,399.
- **Primary stratum** (selection-consistent × verified): 63 dates with a rank-1 in every surface.
- The A26/B26 walk-forwards ran over all 184 season test days; only registered dates and candidates are disclosed (X-21).
- **Recommendation actions:** 56 double, 11 single, 25 skip and 13 unknown (`nan` in the summary). These are recommendations, not proof of delivery or contest entry. X-12 identifies 9/19–9/27 as research-only forecast captures; that provenance does not turn a diagnostic argmax into a genuine selection.
- **Selection and eligibility limits:** numerical selection consistency does not bind the full slate to a prediction attempt or final lock. The runner's primary pool combines reconstructed unit status with a scheduled-time cutoff at the slate's last written_at. Surrogate and all-row pools remain separate diagnostics; missing historical witnesses remain missing.

## 4. Surfaces and the chain
**Primary stratum: selection-consistent × verified, 63 dates.**

| Surface | Scored / known rows | Top-1 hits | AUC (equal-date) | Brier | Log loss | Stated − realized |
|---|---|---|---|---|---|---|
| A26: actual-PA walk-forward (oracle) | 12,129 / 12,129 | 53/63 = 84.1% | 0.651 | 0.221 | 0.633 | +0.013 |
| A26_count: geometric count normalization (oracle) | 12,129 / 12,129 | 41/63 = 65.1% | 0.572 | 0.235 | 0.664 | +0.025 |
| B26: estimated-PA walk-forward (oracle) | 11,706 / 11,706 | 42/63 = 66.7% | 0.568 | 0.234 | 0.661 | +0.024 |
| C_frozen: fixed coefficients, final-source reconstruction | 12,726 / 12,129 | 44/63 = 69.8% | 0.567 | 0.237 | 0.666 | +0.031 |
| C_served: archived coefficients, final-source reconstruction | 12,726 / 12,129 | 43/63 = 68.3% | 0.567 | 0.237 | 0.666 | +0.031 |
| D: served scores, diagnostic argmax | 12,726 / 12,129 | 45/63 = 71.4% | 0.566 | 0.237 | 0.666 | +0.031 |

These are native-surface diagnostic winners, selected before outcomes, not archived delivered/entered primaries. All six have 63 known winners and no no_pa/unknown winners. AUC uses 63 dates in each surface, with no single-class dates omitted; Brier, log loss and stated−realized residuals are row-weighted on each surface's known hit/no_hit rows. Different scoreable candidate pools limit comparisons between native table entries.

**The chain** (later minus earlier; each pair reranked on its own identical finite-score/eligible candidate pool; date-block 95% intervals):

| Link | Top-1 change | Discordant (a hit only / b hit only) | Rank-1 changed dates | Brier change |
|---|---|---|---|---|
| A26 → A26_count | −19.0 pp [−30.2, −9.5] | 13 / 1 | 41 | +0.0142 [+0.0128, +0.0155] |
| A26_count → B26 | +1.6 pp [−3.2, +6.3] | 1 / 2 | 12 | +0.0002 [−0.0000, +0.0004] |
| B26 → C_frozen | +3.2 pp [−4.8, +11.1] | 2 / 4 | 19 | +0.0006 [+0.0001, +0.0012] |
| C_frozen → C_served | −1.6 pp [−6.3, +3.2] | 2 / 1 | 13 | +0.0001 [−0.0001, +0.0002] |
| C_served → D | +3.2 pp [0.0, +7.9] | 0 / 2 | 5 | +0.0001 [−0.0001, +0.0003] |

All five paired top-1 comparisons use 63 common-known dates, with no unilateral no_pa/unknown date exclusions. In table order, their common candidate counts are 12,129; 11,706; 11,706; 12,726; 12,726, and their paired Brier known-row counts are 12,129; 11,706; 11,706; 12,129; 12,129. Relative to the 12,726-row primary pool, A26/A26_count lack 597 scores and B26 lacks 1,020; C_frozen/C_served/D have full score coverage. Thus paired changes need not equal differences between native-surface table entries. Later transitions change context/aggregation, fitted coefficients/training cadence or recipe/calibration; they are composite contrasts. Changing pairwise support prevents adding the links into a decomposition.

**Reading it (descriptive, under D3):**
- **The largest displayed adjacent top-1 change is A26 → A26_count:** −19.0 pp [−30.2, −9.5] on this primary diagnostic support. A26_count geometrically normalizes each A26 probability from its realized PA count to an expected count based on the realized lineup slot. Its native AUC is 0.572 versus A26's 0.651. A26, A26_count and B26 are oracle diagnostics; none is an achievable serving comparator.
- **The historical headline is context only.** The README reports 86.2% average P@1 for 2024–2025. A26's 84.1% here reuses the actual-PA method on restricted 2026 served-slate candidates; it does not reproduce that historical evaluation, its original recipe/pool, or the full-season delivered/contest population. This bridge assigns no fraction of the historical-headline/live gap to any link.
- **The later native estimates are numerically similar:** B26, C_frozen, C_served and D span 66.7–71.4% top-1 and 0.566–0.568 AUC. B26 still uses realized information, and C uses final-source reconstruction. The later paired top-1 intervals contain zero, but that does not establish equivalence or absence of a material difference; B26 → C_frozen spans −4.8 to +11.1 pp.
- **Winner agreement and score reproduction are different:** C_served → D changes the diagnostic argmax on 5 of 63 dates, while 55.8% of reconstructed primary-pool rows remain unexplained at the numerical tolerance. Neither result establishes historical input parity or attempt identity.
- **Row-weighted residual estimates:** B26/C_frozen/C_served/D have stated−realized point residuals from +0.024 to +0.031 on their respective known-row populations; A26's oracle estimate is +0.013. These are conditional descriptions, not isolated effects or demonstrated correction opportunities.
- **All-row sensitivities also show a negative A26 → A26_count contrast.** Selection-consistent × all rows, 77 dates: native A26 84.4%, A26_count 67.5%, B26 71.4%, C_frozen 72.7%, D 72.7%; paired A26 → A26_count −16.9 pp [−27.3, −7.8]. All dates × all rows, 105 dates: native A26 81.0% and D 69.5%; paired A26 → A26_count −11.4 pp [−20.0, −3.8]. These remain their own conditional pools, not substitutes for primary forecast quality or an attribution to the historical gap.
- **Nothing here separates drift, calibration, ball regime, selection or sampling.** W1.3's component diagnostics were predeclared under X-29 before these outputs existed; this memo requests no additional tests.

## 5. Prior results carried (not re-derived)
From the plan: the May PA-basis memo; M3's null; the June 29 PA-tilt; the May Gate B headline.

## 6. Limits
- **A window, not the season.** The run reports 105 registered served-slate dates from 6/11 through 9/27 against a season-game-date denominator of 184, so its reported missing fraction is 43%. The primary diagnostic has 63 dates. The shown top-1 intervals concern paired changes; the summary provides no native-surface hit-rate intervals, and the paired widths are not uniformly ±10 points.
- **Final-source fields and historical provenance.** C uses archived final feeds for game-level fields whose historical serving values are unverified. Weather blanking is numerical sensitivity, not a verified cause. Neither numerical consistency nor the model hash establishes attempt identity, lock-time eligibility or historical availability.
- **One registered learning recipe**, at seed 42 with `BTS_LGBM_DETERMINISTIC=1`. A26/B26 use prior 2026 PA on the original retraining calendar and are not a pristine 2026 holdout. These copied outputs do not fully record the executed process environment.
- **Conditional date-block intervals.** They condition on fitted artifacts and scoreable support and assume exchangeable dates, excluding fitting uncertainty, cross-date dependence and selection effects. The source fixes bootstrap seed 20261004; the actual CLI resample count is not recorded in the copied summary/receipt. Zero inclusion is inconclusive, not equivalence, and the composite contrasts do not identify causes or achievable lift.
- **Descriptive only (D3 = RESERVE).** No candidate is nominated or tested; any idea motivated here needs its own registration and 2027 validation.
