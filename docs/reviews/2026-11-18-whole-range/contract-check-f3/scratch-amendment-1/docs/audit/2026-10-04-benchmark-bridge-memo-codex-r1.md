## Verdict

**SIGN WITH EDITS.** The copied analytical values support a bounded descriptive memo. Withdraw the historical-gap attribution, label all oracle surfaces correctly, distinguish native and paired populations, qualify the pre-game/operational readings, and make the two rounding corrections below. No new experiment, W1.3-type test, acquisition, real-run repair or additional outcome read is requested.

Reviewed the memo at requested `db050475c8adc4aa592b8c03524781c8ae978840`, frozen design rev 3, both design reviews, X-21, and runner/core/report source **at executed `4e4bf5aa33070f7c1dd76c6644eeac4af1b886fe`**, not a later runner. Copied `summary.json` has full SHA256 **`2373a77110c31124df392b8d193f62da7ba3fe7f8a2b97eb2902b0cdd97545b0`**, exactly matching the copied acceptance record. The table/manifest hashes are receipt attestations: those files were not provided or opened.

`check_numbers_r1.py` / `numeric-checks-r1.json` record **139 numerical cell/identity checks**, including every displayed surface value, chain point/interval/discordance/count, reproduction value, inventory/pool count and sensitivity rate. There are exactly two rounding mismatches. The other numerical-looking claims that cannot be checked from these copies are explicitly inventoried as provenance/framing limits. No statistical outputs were recomputed from candidate data.

The memo/design/reviews and X-21 row remain the requested versions. The shared checkout advanced during review, including unrelated README/exposure-register changes; `reviewed-inputs-r1.json` pins the versions consulted and executed source objects. Final verification is in `verification-r1.json`. All reads stayed within this checkout; no production `data/`, SSH, network, `gh`, loaders, tracked edits, commits or pushes were used. This review accepts neither unseen operational evidence nor a broader historical headline reproduction.

## Findings

**F1 — P1: The leading interpretation restores the historical-gap attribution explicitly withdrawn by the frozen design.** Memo `:59`; frozen design §1 and §6; design reviews r1 F9/r2 F9.

“The headline's optimism sits almost entirely in the first link” assigns a share of the historical-headline/live gap to this bridge. A26 uses the actual-PA method on restricted 2026 D candidates; it does not reproduce the original headline's recipe, historical population or original evaluation. D's diagnostic argmax is not the full-season delivered/contest result. The README table at the requested commit labels its **86.2% average as 2024–2025**, not 2021–2025 as the memo says. Neither historical number is independently re-derived here.

Within the primary diagnostic, A26→A26_count is indeed the largest displayed adjacent top-1 change: −19.0 pp [−30.2, −9.5]. That supported statement is sufficient. Different pairwise scoreable pools and composite later transitions prevent using the chain to explain a fraction of the headline gap. Calling the paragraph “descriptive” does not make that attribution valid. The same attribution was copied concurrently into README's “What the backtest numbers below measure”; this memo's correction must not be cited as support for that uncorrected wording.

**F2 — P1: B26 remains an oracle, and the similar estimates/zero-containing intervals do not establish equivalence or serving parity.** Memo `:42–46`, `:60–62`; frozen design §3 and §6.

B26 uses realized participation, lineup/starter and matchup information even though its PA count is estimated. The frozen design labels **A26, A26_count and B26** oracle diagnostics. Calling B26 one of “every serving-realistic surface” contradicts that rule. C uses final-source reconstruction, with historical availability unproved. D's displayed top-1 is a reranked diagnostic on the reconstructed pool, not an actual entered/delivered pick rate.

The reported native ranges, 66.7–71.4% and AUC 0.566–0.568, are numerically correct. The later paired top-1 intervals contain zero, but include material effects; for example B26→C_frozen spans −4.8 to +11.1 pp. That does not establish equality or absence of an effect. C-served→D's five changed argmax dates support winner agreement, not numerical reproduction: **7,101/12,726 primary reconstruction rows remain unexplained**. Retain the row-weighted residual estimates, explicitly as conditional descriptives on each surface's known rows.

**F3 — P2: Two displayed scores are rounded incorrectly; row denominators and interval meaning are incomplete.** Memo `:39–56`, `:70`.

| Cell | Copied value | Correct direct three-decimal display | Memo |
|---|---|---|---|
| A26_count Brier | 0.23548730308802293 | **0.235** | 0.236 |
| D log loss | 0.666496424590056 | **0.666** | 0.667 |

All other displayed primary surface cells and all five chain signs/intervals/counts agree at the displayed precision. “Later minus earlier” and “stated minus realized” are correctly stated. The discordant counts independently reproduce every paired top-1 difference. The −0.0000 Brier endpoint is a rounded negative number, not an exact-zero bound.

The native-surface proper-score denominators differ: **12,129 known rows** for A26/A26_count/C_frozen/C_served/D, but **11,706** for B26. C/D have 12,726 scored candidates, including 597 outside the known hit/no_hit population. Paired candidate counts are **12,129; 11,706; 11,706; 12,726; 12,726**; paired Brier uses **12,129; 11,706; 11,706; 12,129; 12,129** known rows. All five paired top-1 denominators are 63. These distinctions explain why A26_count→B26 has paired Brier **+0.0002072** although subtracting the native table values gives approximately **−0.0011511**. Both are correct for their different populations. Publish these denominators rather than imply the native table is the paired common set.

“Top-1 intervals are about ±10 points wide” is not supported as a statement about the surface rates: the summary supplies intervals for **paired changes**, not native top-1 rates, and their widths differ. Replace it with that precise limitation.

**F4 — P2: The pre-game paragraph turns a pool-level association into a game-status/history explanation.** Memo `:25–27`.

Verified-pool unexplained **7,101/12,726 = 55.8%** versus overall **8,710/19,050 = 45.7%** is correct. The summary contains reproduction classes by pool, not classes by contemporaneous playing status. The complement of `pool_verified` can contain unknown status, cutoff exclusions and other cases; it is not simply “games already under way.” Even the complement has unexplained rows. The copied evidence does not establish that underway games reproduce exactly or that final-feed changes were the likely historical cause. The weather-blank comparison is numerical sensitivity, as the design already requires.

The reported quantiles are also correct, but they are **absolute score discrepancies**, not signed residuals. Name that explicitly. `sha_unbound` means no matching model-hash binding; the executed branch allows absent binding or a mismatching hash, not just an absent pick file. Copied metadata establishes that 21 of the 24 unbound dates have skip actions and three have unknown actions.

**F5 — P2: The technical acceptance criteria are outcome-independent, but the memo overstates what the receipt proves.** Memo `:3`, `:8–14`; copied `ACCEPTED.json`; frozen design §9.

The criteria name completion, full-run flags/inventory, code fixes, input consistency and prior exposure. None requires a favorable rate, effect sign, interval or reproduction percentage. Those are sound structural acceptance conditions; **no outcome-dependent acceptance rule was found**. The copied summary corroborates code/run identity, false smoke flags and reported inventory/coverage. The source defines the 184-date denominator from distinct 2026 dates in the loaded PA feature frame; an independent season-calendar completeness witness was not provided. The receipt attests completion at 09:07:45 UTC (05:07:45 ET), no earlier output, and PA byte identity across the rebuild. It does not independently verify those events.

The detailed 11G/13G caps, feed sizes/counts, 187/2,335 captures, prior 1,854 eligibility rows, pause times/6.1 GB, exact rebuild time and backup-copy claim are not established by the two copied artifacts. An OOM priority setting also does not by itself prove the universal “before any production process” guarantee. “Results/computation unaffected” exceeds the evidence provided. These operational statements need their own attribution or removal; they are not new analytical results.

Moreover, technical acceptance is not a certificate of every design condition. The executed runner writes its manifest after the results. The copied receipt does not establish that a full input manifest was frozen **before execution**, as §9 requires. The source fixes bootstrap seed 20261004, but the CLI permits `--n-resamples`; its actual value is absent from the copied summary/receipt. The registered learning recipe is seed 42/deterministic, but these copies do not fully bind the executed process environment. These are explicit verification limits, not evidence that the run actually used different parameters or that a prior pin did not exist. Do not add an acceptance threshold selected from the observed results, rerun diagnostics, or request W1.3 tests to fill these gaps.

**Fidelity otherwise verified:** reproduction is the first analytical section; the metrics disclosed are within X-21's registered candidate/date scope; all-row sensitivities are distinguishable; no candidate nomination occurs; the oracle warning, sign convention, resumed-PA/baseball-event target and conditional date-block design are present in the executed source/design. The memo should also report the available recommendation-action counts and explicitly retain the research-only/no-selection distinction. None of the frozen-design limits is reopened as a new build requirement.

## Verbatim edits

**E1 — Replace the status line and the body of §1.** Keep the existing section heading.

```markdown
**Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated). Analytical values below come from the accepted run's `summary.json`; operational statements are acceptance-record attestations where identified.
```

```markdown
- **Accepted run:** `data/validation/w12_bridge/4e4bf5a-20261004T060725Z/`, code `4e4bf5a`. The acceptance record attests completion at 2026-10-04 05:07:45 ET and technical acceptance before result reads for W1.3/W2.3. Its criteria concern completion, full-run scope, the two pre-output code fixes, PA input consistency and prior X-21 publication; none depends on favorable analytical results.
- **Verified copied output:** the copied `summary.json` matches the acceptance record's SHA256 `2373a77110c31124df392b8d193f62da7ba3fe7f8a2b97eb2902b0cdd97545b0`. It records false `skip_ab`/`skip_frozen` flags, 105 registered dates and a season-game-date denominator of 184, with missing fraction 0.429. The receipt also lists `table.parquet` (`accb6d48…`) and `manifest.json` (`dae86f08…`); those files were not independently inspected in this memo review.
- **Pre-output fixes:** the receipt attests that earlier attempts produced no output and that the accepted code includes slim feeds (`24129fd`) and reading plain plus gzipped unit captures (`4e4bf5a`). Source inspection confirms those changes. The accepted summary reports 14,094 verified-pool candidate rows; that coverage count alone does not certify every reconstructed eligibility edge.
- **Input and execution limits:** the receipt attests that the 2026 PA input retained SHA256 prefix `06264f2f…` across the 03:00 rebuild. The underlying logs and start-time input pins were not provided here. The executed runner writes its manifest at the end, so these copies alone do not establish a full input manifest frozen before execution. They also do not record the actual CLI resample count or fully bind the process environment. Technical acceptance is separate from numerical reproduction and complete historical provenance; no general claim that operational changes left every input and result unaffected follows from this review.
```

**E2 — Replace §2's three bullets following the reproduction-class table.** Keep the table unchanged.

```markdown
- **Absolute score discrepancies:** among unexplained rows, the median is 0.0017, the 90th percentile 0.0069 and the maximum 0.075. Across all 19,050 reconstructed rows, the absolute discrepancy's 99th percentile is 0.049. These are magnitudes, not signed residuals or verified historical causes.
- **Reproduction differs by reconstructed pool.** In the runner's verified-status plus scheduled-time pool at the slate's last written_at, 7,101 of 12,726 reconstructed rows (55.8%) are unexplained, compared with 8,710 of 19,050 (45.7%) overall. The summary does not separate reproduction classes by contemporaneous playing status, so it does not establish that already-underway games reproduce exactly or identify why the rates differ. Weather blanking establishes numerical sensitivity only; historical serving values and residual causes remain unverified.
- **Served artifacts:** C-served scored 81 of 105 dates; 24 are `sha_unbound`, meaning no matching archived-model hash binding. Their recorded actions are 21 skips and three unknown. The status alone does not distinguish an absent binding from a mismatching hash.
```

**E3 — Add these bullets to §3.** Retain the correctly stated inventory, selection-state and pool counts.

```markdown
- **Recommendation actions:** 56 double, 11 single, 25 skip and 13 unknown (`nan` in the summary). These are recommendations, not proof of delivery or contest entry. X-12 identifies 9/19–9/27 as research-only forecast captures; that provenance does not turn a diagnostic argmax into a genuine selection.
- **Selection and eligibility limits:** numerical selection consistency does not bind the full slate to a prediction attempt or final lock. The runner's primary pool combines reconstructed unit status with a scheduled-time cutoff at the slate's last written_at. Surrogate and all-row pools remain separate diagnostics; missing historical witnesses remain missing.
```

**E4 — Replace §4's surface table and the sentence immediately before the chain.** The two rounding corrections and oracle labels are included.

```markdown
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
```

Add immediately after the unchanged chain table:

```markdown
All five paired top-1 comparisons use 63 common-known dates, with no unilateral no_pa/unknown date exclusions. In table order, their common candidate counts are 12,129; 11,706; 11,706; 12,726; 12,726, and their paired Brier known-row counts are 12,129; 11,706; 11,706; 12,129; 12,129. Relative to the 12,726-row primary pool, A26/A26_count lack 597 scores and B26 lacks 1,020; C_frozen/C_served/D have full score coverage. Thus paired changes need not equal differences between native-surface table entries. Later transitions change context/aggregation, fitted coefficients/training cadence or recipe/calibration; they are composite contrasts. Changing pairwise support prevents adding the links into a decomposition.
```

**E5 — Replace all bullets under “Reading it (descriptive, under D3)”.**

```markdown
- **The largest displayed adjacent top-1 change is A26 → A26_count:** −19.0 pp [−30.2, −9.5] on this primary diagnostic support. A26_count geometrically normalizes each A26 probability from its realized PA count to an expected count based on the realized lineup slot. Its native AUC is 0.572 versus A26's 0.651. A26, A26_count and B26 are oracle diagnostics; none is an achievable serving comparator.
- **The historical headline is context only.** The README reports 86.2% average P@1 for 2024–2025. A26's 84.1% here reuses the actual-PA method on restricted 2026 served-slate candidates; it does not reproduce that historical evaluation, its original recipe/pool, or the full-season delivered/contest population. This bridge assigns no fraction of the historical-headline/live gap to any link.
- **The later native estimates are numerically similar:** B26, C_frozen, C_served and D span 66.7–71.4% top-1 and 0.566–0.568 AUC. B26 still uses realized information, and C uses final-source reconstruction. The later paired top-1 intervals contain zero, but that does not establish equivalence or absence of a material difference; B26 → C_frozen spans −4.8 to +11.1 pp.
- **Winner agreement and score reproduction are different:** C_served → D changes the diagnostic argmax on 5 of 63 dates, while 55.8% of reconstructed primary-pool rows remain unexplained at the numerical tolerance. Neither result establishes historical input parity or attempt identity.
- **Row-weighted residual estimates:** B26/C_frozen/C_served/D have stated−realized point residuals from +0.024 to +0.031 on their respective known-row populations; A26's oracle estimate is +0.013. These are conditional descriptions, not isolated effects or demonstrated correction opportunities.
- **All-row sensitivities also show a negative A26 → A26_count contrast.** Selection-consistent × all rows, 77 dates: native A26 84.4%, A26_count 67.5%, B26 71.4%, C_frozen 72.7%, D 72.7%; paired A26 → A26_count −16.9 pp [−27.3, −7.8]. All dates × all rows, 105 dates: native A26 81.0% and D 69.5%; paired A26 → A26_count −11.4 pp [−20.0, −3.8]. These remain their own conditional pools, not substitutes for primary forecast quality or an attribution to the historical gap.
- **Nothing here separates drift, calibration, ball regime, selection or sampling.** W1.3's component diagnostics were predeclared under X-29 before these outputs existed; this memo requests no additional tests.
```

**E6 — Replace §6's window, final-feed, learning-recipe and interval bullets.** Keep its D3 bullet.

```markdown
- **A window, not the season.** The run reports 105 registered served-slate dates from 6/11 through 9/27 against a season-game-date denominator of 184, so its reported missing fraction is 43%. The primary diagnostic has 63 dates. The shown top-1 intervals concern paired changes; the summary provides no native-surface hit-rate intervals, and the paired widths are not uniformly ±10 points.
- **Final-source fields and historical provenance.** C uses archived final feeds for game-level fields whose historical serving values are unverified. Weather blanking is numerical sensitivity, not a verified cause. Neither numerical consistency nor the model hash establishes attempt identity, lock-time eligibility or historical availability.
- **One registered learning recipe**, at seed 42 with `BTS_LGBM_DETERMINISTIC=1`. A26/B26 use prior 2026 PA on the original retraining calendar and are not a pristine 2026 holdout. These copied outputs do not fully record the executed process environment.
- **Conditional date-block intervals.** They condition on fitted artifacts and scoreable support and assume exchangeable dates, excluding fitting uncertainty, cross-date dependence and selection effects. The source fixes bootstrap seed 20261004; the actual CLI resample count is not recorded in the copied summary/receipt. Zero inclusion is inconclusive, not equivalence, and the composite contrasts do not identify causes or achievable lift.
```

**Propagation note:** the concurrently added README paragraph repeats F1/F2. If that linked summary is retained, use this replacement for its actual-PA bullet; it is not an extension of this review to the other README results.

```markdown
- **The reported 86.2% P@1 is an actual-PA walk-forward headline**, averaged over 2024–2025. The W1.2 bridge reuses that method on restricted 2026 served-slate candidates: A26 is 84.1%, and geometric count normalization is 65.1% on the primary diagnostic support. Both are oracle diagnostics. The bridge does not reproduce the historical headline or delivered live population, and it assigns no fraction of their gap to realized-count information. See `docs/audit/2026-10-04-benchmark-bridge.md` for populations and limits.
```
