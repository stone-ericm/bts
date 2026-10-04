# #87 Leaderboard mechanism mining: registered result (season wrap W2.4)

**Date:** 2026-10-04. **Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Protocol:** `docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md`, as amended by `docs/sota_audit/2026-10-04-mechanism-mining-amendment.md` rev 5 (FROZEN). **Code:** `scripts/audit/mining87/` final `dfa2bb4`, FROZEN. **Exposure:** X-22 (`6425eda`), published before any outcome-bearing execution.
**Run:** `data/validation/mining87/9640703-20261004T171403Z-64e7d1e6/`, copied to `data/hetzner_results/season_wrap_outputs/mining87/`.
- **Mode:** registered, complete (`COMPLETE.json`).
- **Inputs:** consumed inputs verified against the manifest (`f56917bc…`).
- **Registration:** fingerprint `31e7c791…`.
- **Pins and outputs:** the pins were persisted before any loader; stdout carried only the run directory.

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
- **Primary stream:** 94 nonempty cross-product cells (48 fixed-cohort, 46 all-tracked).
  - **None reaches the n ≥ 15 resolved-disagreement testability floor**, so there are 0 testable cells. No BH/BY family exists, nothing survives BH or BY, there are no statistical candidates, and every condition c1–c5 has empty support.
  - 40 fixed-cohort cells have sparse support.
  - Every cell, with its counts, is in `cells_primary.parquet`.
  - **Outcome state: `power_limited_no_testable_cells`.**
- **The tie-excluded sensitivity** has the same 94 cells and 0 testable, so there is nothing to compare.
- **Nomination:** none. **No cell met all five conditions, so there is no actionable mechanism for this cohort and window.** Independently, condition 5 was unavailable by the frozen input choice (no mechanism records), so this execution could not have nominated anyway.
- **Information limits:**
  - all 145 fixed units lack an admitted surface;
  - 23 have production settlement unresolved;
  - 3 have consensus settlement unresolved;
  - 40 fixed cells are sparse.

  **This does not disprove signal** in a larger or better-instrumented sample. It is a power and instrumentation limit, not a measured negative effect.

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
- **Consensus concentration:** among the fixed cohort's 100 primary slots with a consensus, the modal batter took a median 20% of the votes (median 488 public users per slot); for slot 2 it was 16% (median 378). The concentration may indicate a publicly obvious batter class or a stale blind spot. It is not a success metric.
- **Unproven served-slate diagnostic** (outside the registered family; these dates failed admission): on 15 selection-consistent dates, the fixed-cohort consensus batter was our served rank-1 on 13% of dates and in our top 10 on 47%.

## Limits (amendment A4/A5)
- **Exposure and selection:** post-hoc mining of a previously exposed window (X-10); the cohort is chosen retrospectively from the 7/04 snapshot, with survivorship and right-truncation.
- **No pre-lock claim:** public pick logs are behaviour observations, not pre-lock proof.
- **Admission:** served slates are not admitted without an independent witness.
- **Context:** five context axes have no audited source.
- **Test assumption:** the exact one-sided sign test (the amended all-cell extension) assumes sign-exchangeability.
- **Half a season:** the daily corpus ends at 7/04.
- **For W4 rank 7:** nothing is nominated, so there is nothing to fund under D4 from this run.
