# W2.1 / W2.2 field products — design (season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, W2 (identifiability table; products 1–2) and owner decision D5 (150/150 profile split).
**Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Use:** descriptive only. No causal value of copying, doubling or skipping is estimated (the plan cuts those).

## Data (inventoried 2026-10-04; names and counts only)
- **The final board:** `data/leaderboard/final_grab_20260927/`.
  - `leaderboard_snapshots/2026-09-27_all_season_full.parquet`: 121,852 rows, equal to the reported participant count, so a census.
  - The 9/28 post-cutoff pass was identical for every user (register X-17).
  - The C-01 fix makes `season_best_streak` tab-semantic on `all_season`.
- **Profiles:** `season_stats/2026-09-27.parquet` (300) and `user_picks/` (300 id-keyed files).
  - Cohort A: 150 from the top of the final board.
  - Cohort B: 150 from the frozen 2026-05-01 early cohort (`docs/audit/2026-09-22-early-cohort-2026-05-01.json`).
  - Membership comes from `cohort.json` and `identity.json` (the grab's own records).
- **The daily corpus** (`data/leaderboard/user_picks/`, pick dates 3/25 → 7/03) holds Cohort B's picks before the final grab.
- **Our production:** the W1.1 ledger (contest-graded slots; `committed_evidenced` selections).

## W2.1 Final-leader case series and threshold counts
1. **Whole field (the census):**
   - the distribution of season best;
   - the counts reaching 20, 30 and 40;
   - the maximum and the size of the tie at the maximum;
   - **our season best (18) and its percentile and rank**, with ties as reported.

   Rank and tie semantics follow the board's own `rank` field. Users with season best 0 are included (the post-season board lists everyone).
2. **Final-leader case series (Cohort A):** per user:
   - season best and its dates (start and end of the best run, from `user_picks`);
   - the number of pick days;
   - double-down frequency, as DD-day count over pick-day count;
   - missing pick dates, counted as **missing, never as skips** (the plan's rule).

   Plus descriptive pick composition: home/away, lineup-slot availability, and team concentration.
3. **Labels:** a capture-as-of listing is not an awarded prize. Survivor selection is stated for every Cohort A statistic.

## W2.2 As-of cohort comparison (Cohort B)
1. **Cohort:** the 150 B users, defined by the 2026-05-01 snapshot before their later outcomes were known. The manifest is frozen and its hash recorded.
2. **Follow-up (primary):** 2026-05-01 → 2026-07-03, from the daily corpus, with an attrition table showing which users have picks after each month boundary.
   - Per user: graded picks, slot hit rate, and longest streak inside the window.
   - The cohort's pooled slot hit rate, clustered by date.
3. **Final-survivor extension:** 7/04 → 9/27, from the final grab, **labelled separately and never appended to the primary curve** (the plan's rule).
4. **Comparison with our production over the same dates:** our contest-graded slot hit rate (from the ledger) beside the cohort's pooled rate. Both use date-block bootstrap intervals (10,000 resamples, fixed seed), resampling whole dates.
   - Different denominators: Cohort B users pick on days we skip, and the reverse. So the comparison runs both on all dates and on the **shared-date** subset, the dates on which both made a graded pick.
   - Not causal: the cohort picks its own batters.
5. **Outcome labels:** the contest's own per-slot result (`hit` / `not_hit`). `pending` or missing labels are unknown, never misses.

## Exposure
New rows are predeclared and pushed before any outcome-bearing execution:
- **X-24 (W2.1):** the census distribution, our percentile, and the Cohort A case series;
- **X-25 (W2.2):** Cohort B's follow-up and the comparison with our production.

X-10 (the 7/03 me-vs-leaderboard read) and X-14 (the first descriptive read of the 9/22 board) are disclosed as overlapping prior exposure. Under D3 these are descriptions, not tests; nothing here selects a W4 candidate.

## Build and run
- **Code:** `scripts/audit/field_products/`, written test-first with synthetic boards and pick files.
- **Run:** on the box, in a transient unit, into `data/validation/w21_w22_field/<sha>-<run>/`.
- **Memos:** `docs/sota_audit/<date>-field-final-leaders.md` and `docs/sota_audit/<date>-field-cohort-comparison.md`.

## Limits stated up front
- **Cohort A** is survivor-selected.
- **Cohort B's** 7/04–9/27 data come from one end-of-season grab, so a profile's history depth is whatever the API returned (recorded by W0.6).
- **Pick logs** are observations of public behaviour, not proof of pre-lock timing.
- **Ties** follow the board's semantics.
- **Hit-rate comparisons** are descriptive and cluster by date; they are not tests.
