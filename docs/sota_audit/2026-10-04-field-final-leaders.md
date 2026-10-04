# W2.1 Final-leader case series and the 2026 board census

**Date:** 2026-10-04. **Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Design:** `docs/superpowers/specs/2026-10-04-field-products-design.md` rev 2, FROZEN. **Code:** `scripts/audit/field_products/` at `793e987`, FROZEN after two Codex code rounds. **Exposure:** X-24 (`b972cd9`), published before any outcome-bearing step.
**Run:** `data/validation/w21_w22_field/327fcd6-20261004T171153Z/` (gate pin `327fcd6`), copied to `data/hetzner_results/season_wrap_outputs/w21_w22/`. It froze 1,366 source files before parsing any outcome.
**Use:** descriptive only, under D3 = RESERVE. Nothing here estimates the value of copying, doubling or skipping.

## 1. The board census (9/27 final grab)
- **Census status: valid.** Every integrity and coverage check passed:
  - the retained board status;
  - raw page hashes and page set;
  - an exhausted, conflict-free recomputed walk;
  - recomputed population equal to the reported participant count;
  - unique ids;
  - output equal to the raw rows;
  - the qualified rows conflict-free and equal to the reported count.

  N = **121,852** entrants; no season-best value is missing; listing floor 1.
- **The field's season best** (C-01: the 9/27 walk was parsed with tab-semantic `season_best_streak`; C-04/X-17: the 9/28 post-cutoff pass was identical for every user, cited rather than remeasured):

| Reached | Entrants |
|---|---|
| ≥ 20 | 2,186 |
| ≥ 30 | 49 |
| ≥ 40 | 0 |
| Maximum | **39** (one entrant; no tie) |

  These are census counts: each known count equals its upper bound because no value is missing. The full distribution is in `w21_distribution.parquet`. A census has no sampling interval, so none is reported.
- **Our position:** matched by stable user id 50311, with our season best of **18** reconciled to the declared 18.
  - 117,273 entrants are below, 1,553 equal and 3,026 above.
  - That gives the percentile interval **[96.2, 97.5]**, using the convention [100·below/N, 100·(below+equal)/N].
  - The board's stored rank is 3,027; that is its own tie semantics, not a percentile convention.

**The Top Streak prize, for D1:** a listing as of the capture is not an awarded prize. The board's maximum was 39, so under the rules (W3) the $10,000 Top Streak prize, which requires at least 20, would go to that entrant. Our 18 was below the floor.

## 2. Final-leader case series (Cohort A: the top 150 of the final board)
**Survivor-selected:** Cohort A is chosen on the final board, so every statistic here describes winners after the fact. It estimates nothing about the field.

- **Coverage.** All 150 profiles have a verified raw final-grab response, usable history, a validated slot-set witness for every round (18,168 rounds, 32,042 slots), no revisions, conflicts or deleted legs, and 5 rounds incomplete (missing primary slot).
- **Pick days and doubling:** 18,168 pick rounds across the 150 users. Of the 18,163 complete rounds, 13,874 were double-downs (**76.4%**). Per-user values are in `w21_case_series_A.parquet`.
- **Slot labels:** 22,253 hit, 8,927 not_hit and 862 void. `void` is a settled Pass under the W1.1 normalization, never a graded hit or miss.
- **Runs:**
  - Board season best is kept separate from run reconstruction.
  - Exact run start and end dates are unavailable for 144 of the 150, because no entered-round completeness witness exists.
  - 6 have no settled round reporting their best.
  - Only observed-segment lower bounds are reported, in `w21_runs_A.parquet`.
- **Composition:** team and home/away are **unknown** for all 32,042 slots. Stored context comes from capture-time lookups, and no independent historical pick-time witness exists. Lineup slot is not stored.

## Limits
- **Census scope.** The census is as of the retained 9/27 capture.
- **Cohort A** is survivor-selected; its histories come from one end-of-season grab, whose depth is whatever the API returned.
- **Pick logs** are observations of public behaviour, not proof of pre-lock timing.
- **Unavailable quantities.** Exact streak maxima, run dates and historical composition are unavailable, not zero.
- **Descriptive only:** no W4 selection, causal effect, skill rank or equivalence.
