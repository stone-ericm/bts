# #87 Leaderboard mechanism mining: registered result (season wrap W2.4)

**Date:** 2026-10-04. **Status:** FROZEN 2026-10-04 after Codex memo r1 SIGN WITH EDITS (C-E1–C-E3 applied verbatim by script; review archived at `docs/audit/2026-10-04-field-87-memos-codex-r1.md`). Pace rule: no further review rounds.
**Protocol:** `docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md`, as amended by `docs/sota_audit/2026-10-04-mechanism-mining-amendment.md` rev 5 (FROZEN). **Code:** `scripts/audit/mining87/` final `dfa2bb4`, FROZEN. **Exposure:** X-22 (`6425eda`), published before any outcome-bearing execution.
**Run:** `data/validation/mining87/9640703-20261004T171403Z-64e7d1e6/`, copied to `data/hetzner_results/season_wrap_outputs/mining87/`.
- **Mode:** registered, complete (`COMPLETE.json`).
- **Inputs:** the copied completion receipt attests that consumed inputs matched the manifest (`f56917bc…`).
- **Registration:** fingerprint `31e7c791…`.
- **Pins and outputs:** the runner writes execution pins before outcome-bearing loaders and emits the final run directory, completion flag and mode on stdout.

`research_only=true`, `production_deploy_claim=false`, `no_policy_edit_supported=true`. This is a retrospective post-hoc analysis of a previously exposed window (X-10); it is not an untouched holdout.

## 1. Coverage and denominators
- **Production:** 154 selection rows in the window 3/26–7/03.
  - **145 locked units** (82 primary, 63 double-down). 9 selections were excluded for unconfirmed commitment.
  - **Settlement:** 122 resolved by contest grade; 23 unknown (no linked contest slot), kept as units but outside the outcome estimands.
  - **Days without units:** 6 skip days, 3 unfinalized and 3 unobserved.
- **Public picks:** 2,042 user files, 102 of them empty.
  - 2,809,077 rows read; 27,139 outside the window; none after the capture cutoff (end of 7/04 ET).
  - 184,001 user-slot observations; 99 without a valid batter id.
  - Captures span 5/01–7/04.
- **Fixed cohort:** the `active_streak` tab of `leaderboard_snapshots/2026-07-04.parquet`, membership sha256 `0ce1968c…`. Its 22,800 rows hold 21,710 usernames: the deep 7/04 board, which is the historical script's rule, every username in the tab. Of those, **778** have a public pick file; there were no sanitization collisions.
- **All-tracked:** every pick file. Both cohorts form a consensus for all 145 units.
- **Consensus settlement:** fixed cohort 142 resolved, 1 unknown and 2 void; all-tracked 143, 1 and 1.
- **Ranked surfaces:** 91 dates were considered and **0 admitted**: 23 had no independent witness and 68 no served slate. Variables 13–14 are therefore `missing_surface` throughout.
- **Production context** is unavailable (no audited source), so variables 6, 7 and 10–12 take their `missing` / `indoor_or_missing` bins (amendment A5).

## 2. Primary estimand: consensus top-N coverage
**Unavailable:** no admitted ranked surface. Missing surfaces are not off-top-N failures.

**The protocol's fallback, same-slot agreement:** the fixed-cohort consensus picked the same batter as our locked production pick in **6 of 122** resolved slots (4.9%): 4/70 primary and 2/52 double-down. All-tracked gives the same counts. This is a production-decision diagnostic, not model coverage.

## 3. Decomposition cells, FDR and the five nomination conditions

- **Primary stream:** all 94 observed nonempty cross-product cells are retained: 48 fixed-cohort and 46 all-tracked. Every cell and its counts/conditions is in `cells_primary.parquet` and `report.json` under `streams.primary.cells`.
- **Testability:** none reaches 15 resolved disagreements; the maximum cell support is 13 fixed-cohort and 12 all-tracked. The realized testable BH/BY family is empty. No cell has an available p/q; there are no statistical candidates or nominations. Forty fixed cells have positive but sparse disagreement support; eight have zero resolved disagreements.
- **Five conditions, separately, over all 48 fixed-cohort cells:** sparse point effects and directions remain descriptive. All-tracked cells supply comparisons and are not nomination candidates.

| Condition | Reported status |
|---|---|
| c1: at least 30 resolved disagreements | 48 fail |
| c2: consensus-minus-production lift at least 0.05 | 12 pass, 28 fail, 8 unknown |
| c3: BH q ≤ 0.10 | 48 not testable; q unavailable |
| c4: all-tracked direction not contradictory | 26 pass, 10 fail, 12 unknown |
| c5: stated mechanism with evidenced lock-available variables | 48 unknown: no mechanism records supplied |

- **Tie-excluded sensitivity:** one flagged unit is removed per cohort, leaving 144 units per cohort. The inventory still has 94 nonempty cells and zero testable cells. The five-condition status counts above are unchanged. No primary cell passes c1-c4, so the candidate sensitivity-comparison list is empty; this is not a claim that the two support inventories are identical.
- **Outcome state:** `power_limited_no_testable_cells`.
- **Nomination:** none. **No cell met all five conditions, so there is no actionable mechanism for this cohort and window under the stated information limits.** Independently, condition 5 was unavailable by the frozen input choice, so this execution could not have nominated even with stronger outcome support.
- **Information limits:** all 145 fixed units lack an admitted surface; 23 have unresolved production settlement; 3 have unresolved consensus settlement; 40 fixed cells have sparse positive disagreement support. These are separate limits, not measured negative effects.

**This does not disprove signal** in a larger or better-instrumented sample. It is a power and instrumentation limit. Sparse descriptive lift/direction passages do not establish a mechanism.

## 4. Secondary estimands (descriptive)
**Paired outcomes, consensus minus production.** Unit-weighted means; circular geometric date-block bootstrap with expected block 7, 2,000 replicates, seed 20260510, blocks over ordered observed dates.

| Cohort | Population | Units / dates | Production | Consensus | Mean difference [95%] |
|---|---|---|---|---|---|
| Fixed | all jointly resolved | 119 / 72 | 70.6% | 74.8% | +4.2 pp [−5.4, +12.8] |
| Fixed | resolved disagreements | 113 / 70 | 69.0% | 73.5% | +4.4 pp [−5.7, +13.4] |
| All-tracked | all jointly resolved | 120 / 73 | 70.8% | 76.7% | +5.8 pp [−5.0, +15.2] |
| All-tracked | resolved disagreements | 114 / 71 | 69.3% | 75.4% | +6.1 pp [−5.2, +15.8] |

Every interval includes zero. **The consensus outcomes are public pick logs, not evidence that the consensus was visible before lock**, so no copying claim follows.
- **Conditional miscalibration:** unavailable (no admitted surface bound to the consensus outcome target).
- **Consensus concentration:** this summary includes every public-consensus date-slot in the registered window, whether production-matched or not. The fixed cohort has 100 slot-1 and 100 slot-2 consensuses. Median selected-batter vote share is 20.2% for slot 1 (median 488 public users) and 16.5% for the legal slot-2 choice (median 377.5 public users). Concentration may indicate a publicly obvious batter class or a stale blind spot; it is not a success metric.
- **Unproven served-slate diagnostic:** outside the registered primary, decomposition/FDR family and nomination stream, on 15 resolved production date-slots whose dates are selection-consistent but failed admission, the fixed-cohort consensus batter appeared at served rank 1 in 2/15 slots (13.3%) and in the top 10 in 7/15 (46.7%). These are slot-weighted diagnostics, not proportions of distinct dates or admitted top-N coverage.

## Limits (amendment A4/A5)
- **Exposure and selection:** post-hoc mining of a previously exposed window (X-10); the cohort is chosen retrospectively from the 7/04 snapshot, with survivorship and right-truncation.
- **No pre-lock claim:** public pick logs are behaviour observations, not pre-lock proof.
- **Admission:** served slates are not admitted without an independent witness.
- **Context:** five context axes have no audited source.
- **Test assumption:** the exact one-sided sign test (the amended all-cell extension) assumes sign-exchangeability.
- **Half a season:** the daily corpus ends at 7/04.
- **For W4 rank 7:** nothing is nominated, so there is nothing to fund under D4 from this run.
