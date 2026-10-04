# W1.3 Distinguishing the explanations: pre-declared tests (design, season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, W1.3 (the eight-row explanation table).
**Status:** draft for Codex design review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Written before any W1.2 output was produced or read.** The W1.2 full run was in progress when this was drafted. No table, summary, metric or log line carrying an outcome had been opened. Exposure row X-29 must be published before any W1.3 computation.
**Use:** descriptive only, under D3 = RESERVE. Each explanation gets a predeclared label from the rules in §4. No explanation is "proved", no candidate is tested, and nothing here changes production.

## 1. Question
Why did the 2026 live record (W1.1 / X-09: primaries 96/141) fall short of the README headline (86%, an actual-PA walk-forward)? The plan lists eight explanations that may overlap. Each one is classified as consistent with the evidence, weakened by it, or undetermined. Several can hold at once, and the labels are not a decomposition summing to the gap.

## 2. Inputs (all already produced; nothing new is acquired)
- **The accepted W1.2 run:** `data/validation/w12_bridge/<sha>-<run>/` (`table.parquet`, `summary.json`, `manifest.json`), the one the W1.2 memo accepts. Its registered dates (105 served slates, 2026-06-11 → 09-27), candidates, surfaces (A26, A26_count, B26, C_frozen, C_served, D), outcomes, pools and reproduction classes are used as frozen. W1.3 never re-scores a surface.
- **Served primary:** the decision-first primary of each date, re-derived with W1.2's own `decision_primary` / `primary_of` and `selection_consistency` from the hashed decision and pick files of W1.2's manifest.
- **Venue:** `gameData.venue.id` from the archived final feed of each registered game (`data/raw/2026/<game_pk>.json`), as W1.2 already reads it.
- **Ball regime series:** the external as-of park-drag table `data/external/park_drag/park_drag_export.csv` (`venue_id`, `date`, `park_drag_delta`), with its manifest hash recorded. It is produced outside this repo from four-seam pitch-flight physics, and each (venue, date) value uses only that venue's games strictly before the date (`src/bts/features/park_drag.py` docstring). It is therefore dated independently of our outcomes and contains no hit outcomes. Sign, from the producer code (`src/bts/features/park_drag_producer.py` `build_export`): `park_drag_delta` = (the venue's rolling 15-date drag coefficient − its own anchor, the expanding mean at its 10th season date) × a pitch-count shrink weight, as of strictly before the date. Positive means more drag than that venue's early-season level. The expected hit direction (more drag → fewer hits than stated, since drag shortens fly balls) is physical reasoning, not a measurement; drag mostly affects extra-base contact, so a weak or null hit-level association is not evidence against a ball regime.

## 3. Common rules
- **Strata:** the primary stratum is W1.2's `selection_consistent` cohort × `pool_verified`. Sensitivities are `all_dates` and `pool_surrogate` / `pool_all`, reported beside the primary and never substituted for it.
- **Outcomes:** known = W1.2's hit / no_hit. no_pa and unknown are counted, never scored as misses. A rank-1 that is no_pa or unknown is never replaced.
- **Rank-1:** each surface's argmax per date within the stratum, W1.2's `_winners` rule (ties by archived row order), chosen before labels.
- **Intervals:** W1.2's `date_block_bootstrap`: whole dates resampled with replacement, 10,000 draws, seed 20261004, 95% percentile intervals. Paired contrasts resample both arms jointly. Each drawn copy of a date is its own block. Failed or undefined draws are counted and reported, never dropped silently.
- **Half split (E4 only):** the median registered date of the frozen W1.2 date list. The split is fixed by the date list alone, not by outcomes.
- **Every reported number** carries its date, row and known-row denominators.

## 4. The eight tests and their labels
Labels: **consistent** (the evidence the plan names points the way the explanation predicts, with the interval excluding no-effect), **weakened** (the plan's "what weakens it" condition holds), **undetermined** (neither). An explanation whose required evidence is missing is **not testable here**, with the reason.

| # | Explanation | Computation (primary stratum) | Consistent if | Weakened if |
|---|---|---|---|---|
| E1 | Benchmark optimism | W1.2's chain of paired rank-1 hit-rate differences A26→A26_count→B26→C_frozen→C_served→D; plus the realized-PA distribution (W1.2 `n_pa`: mean, share ≥5) of each surface's known rank-1 picks | the A26→B26 rank-1 difference interval lies above 0 | the A26→B26 interval includes 0 while the C_served→D interval lies above 0 |
| E2 | Serving drift / stale inputs | W1.2's reproduction classes (exact / weather-absent sensitivity / unexplained) and residual quantiles; unexplained share by month and by projected vs confirmed lineup | any unexplained reproduction class persists beyond 1e-9 on ≥5% of reproducible rows | reproduction holds within tolerance after the weather-blank class (with M3's carried null, X-03) |
| E3 | PA-opportunity forecast error | A26 vs A26_count (count normalization isolates the realized-count contribution); A26_count vs B26; realized-PA distribution of rank-1 picks by surface | the A26→A26_count rank-1 difference interval lies above 0 (labelled **oracle**: realized-count information, not achievable lift) | that interval includes 0 |
| E4 | Conditional hit-model drift | equal-date mean stated−realized residual and equal-date AUC under C_frozen (fixed 2019–2025 coefficients), second half minus first half; the same contrast under B26 and D for comparison | the C_frozen residual contrast interval excludes 0 and E2 is not consistent | the C_frozen residual contrast interval includes 0 |
| E5 | Calibration without ranking loss | per surface: logistic recalibration of outcome on logit(p) (intercept and slope, unpenalized, logits clipped to [1e-15, 1−1e-15] with boundary counts) on known rows, plus equal-date AUC; the paired D−B26 AUC difference | D's intercept or slope interval excludes identity (0, 1) and the D−B26 AUC interval includes 0 | D's intercept and slope intervals both include identity |
| E6 | Ball regime | per date: league as-of drag = mean `park_drag_delta` over that date's registered venues; Spearman correlation across dates with (a) the all-known-candidate mean stated−realized residual under D and (b) the same under C_frozen; selected-player (D rank-1) residual reported beside them | the (a) correlation interval lies above 0 (more drag → stated above realized; direction by reasoning, §2) | the (a) interval includes 0 or lies below 0 |
| E7 | Selection / composition | D: equal-date within-slate AUC vs pooled across-date AUC; stated−realized residual of rank-1 vs all candidates; strata: decision action (pick vs skip dates), projected vs confirmed lineup, slate size above vs at-or-below the median row count | descriptive only; no label (the plan names no single decisive pattern) | — |
| E8 | Sampling variation | matched-probability null for the served primary on known-outcome dates: 10,000 Bernoulli draws at each primary's served p (seed 20261004); where the realized hit count falls; Wilson intervals on the brief's numbers (plan W1.3) | the realized count lies within the null's central 95% | the realized count lies outside it |

Rules that hold for every row:
- **E3, E1 and E6 use oracle or hindsight information** where stated (realized PA counts, the realized lineup slot). Their "consistent" label never means an achievable improvement.
- **E4 runs on one evolving season.** A drift label cannot separate model drift from composition or regime changes.
- **E6 uses a date-level aggregate of an external series.** A correlation is association, not a regime effect. Dated change-points are not used: none was independently dated inside the window before outcomes (the plan's 5/24 change-point precedes it), and choosing one now would risk the plan's own warning about change-points picked to fit our misses.
- **E8 conditions on the served p** being calibrated, so it is circular for calibration questions. "Within the null" does not mean luck, and "outside it" does not by itself mean drift.
- **A label never upgrades a sensitivity stratum.** If the primary stratum and a sensitivity disagree, both are reported and the primary's label stands.

## 5. Outputs
- **Run:** `data/validation/w13_tests/<sha>-<run>/` with `results.json`, `per_test.parquet` and `manifest.json` (hashes of the W1.2 run files, decision/pick files, feeds used and the drag table).
- **Memo:** `docs/audit/<date>-w13-distinguishing-tests.md`, in the order E2 (reproduction first), E1, E3, E4, E5, E6, E7, E8, then the eight labels in one table with their limits. The memo feeds the decision memo; it nominates nothing.

## 6. Exposure
**X-29** must be published before any computation:
- It names this design at its frozen commit, the W1.2 run it reads, the eight computations and their strata, and the drag table.
- It discloses the reuse of X-21's labels and outputs (which permits W1.3 only "within the design's limits") and X-03 (M3), X-09 (the brief's numbers) and X-02 (the earlier park-drag screen on 2026 dates through about 7/06).
- The new outputs beyond X-21's metric list are: realized-PA distributions of rank-1 picks, half-split contrasts, recalibration intercept and slope, drag co-movement, within- vs across-date AUC and the composition strata, and the matched-probability null.
- Outputs stay within W1.2's registered dates and candidates.
- Under D3, no candidate is nominated or tested. An idea it motivates needs its own registration and 2027 prospective validation.

## 7. Build
- **Code:** `scripts/audit/w13_tests/`, test-first with synthetic tables. It reuses `scripts/audit/benchmark_bridge/{core,report}.py` and reads only the inputs in §2.
- **Run:** a transient unit on the box after the W1.2 memo accepts its run, one job at a time with the OOM guard.

## 8. Limits stated up front
- **The window is the W1.2 window:** 105 served dates from 6/11, of a 2026 season of about 180 game dates. Surfaces outside it are not disclosed.
- **Intervals at about 105 dates are wide.** Most labels are expected to be undetermined, and that is an acceptable result.
- **Reproduction-based reasoning inherits W1.2's limits:** final-feed game fields, the weather-blank sensitivity class, and no proof of attempt identity or historical availability.
- **The drag series is an external product** whose internals this repo does not certify beyond its as-of construction.
- **The plan's lineup/pitcher change and feature-age evidence** (E2) are not stored historically beyond the slate's projected flag. Those parts are not testable here.
- **At-lock expected PA is not in the served slates**, so the at-lock vs realized PA comparison of E3 is limited to what B26's estimated-PA surface implies.
