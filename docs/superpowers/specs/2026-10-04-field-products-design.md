# W2.1 / W2.2 field products — design (season wrap)

**Date:** 2026-10-04.
**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md`, W2 (identifiability table; products 1–2) and owner decision D5 (150/150 profile split).
**Status:** rev 2: Codex design r1 BLOCK (`docs/audit/2026-10-04-w2-designs-codex-r1.md`) with its verbatim edits B-E1–B-E7 applied by script; round 2 is the last.
**Use:** descriptive only. No causal value of copying, doubling or skipping is estimated (the plan cuts those).

## Data (inventoried 2026-10-04; names and counts only)
- **The final board:** `data/leaderboard/final_grab_20260927/`.
  - `leaderboard_snapshots/2026-09-27_all_season_full.parquet`: 121,852 rows, equal to the reported participant count, so a census.
  - The 9/28 post-cutoff pass was identical for every user (register X-17).
  - The C-01 fix makes `season_best_streak` tab-semantic on `all_season`.
- **Profiles:** `season_stats/2026-09-27.parquet` (300) and `user_picks/` (300 id-keyed files).
  - Cohort A: 150 from the top of the final board.
  - Allocation B: see the early-manifest definition below (B-E1).
  - Membership comes from `cohort.json` and `identity.json` (the grab's own records).
- **The daily corpus** (`data/leaderboard/user_picks/`, pick dates 3/25 → 7/03) holds Cohort B's picks before the final grab.
- **Our production:** the W1.1 ledger (contest-graded slots; `committed_evidenced` selections).

## Data contract
Freeze the input/source manifest and resolve user identity before outcomes. Final-grab ID-keyed files use cohort.json and identity.json; daily sanitized-username files do not contain user_id in their pick rows. Attribute daily observations only where existing capture-time identity evidence uniquely binds the observation/batch to an E member. Ambiguous names, collisions or ownership changes are quarantined and counted, not assigned from current usernames.

Read appended observations without latest_per_pick_date. Analytical identity is (user_id, round_id, pick_number), retaining unit/player identity and source/capture provenance. Resolve revisions by a declared observation order, record changed identities and equal-time conflicts, and preserve both DD slots. Prefer a complete auditable round snapshot; where stored evidence cannot establish snapshot completeness or deletion, mark DD/streak summaries incomplete rather than silently retaining an old deleted leg. A later pending/unknown observation is not replaced by an older settled label without an explicitly evidenced settlement rule. Only exact hit/not_hit labels enter graded-slot rates; pending, missing, Pass and other labels are separately counted. Never infer settlement from at_bats/hits.

Observed pick days are distinct user/round dates. DD frequency is complete observed two-slot rounds divided by complete observed pick rounds; incomplete rounds and unobserved calendar dates are reported separately. Unknown slot 2 is not a known single-pick day. All rates report user, date, round and slot denominators and coverage.

## W2.1 Final-leader case series and threshold counts
1. **Whole field (the census):**
   - the distribution of season best;
   - the counts reaching 20, 30 and 40;
   - the maximum and the size of the tie at the maximum;
   - **our season best (18) and its percentile and rank**, with ties as reported.

   Rank and tie semantics follow the board's own `rank` field. Users with season best 0 are included (the post-season board lists everyone).

   Verify the retained board status, unique IDs, raw/output hashes, exhausted conflict-free pagination, stable valid participant count and valid identical updatedAt on every page before calling the board a census. Equal row and participant counts alone are insufficient. Report the listing floor and missing season-best values; no absent/null best is imputed zero. If completeness fails, threshold counts are observed lower bounds and entrant-wide percentiles are unavailable. With a complete board, report N, counts strictly below/equal/above our verified season best, and its percentile interval [100·below/N, 100·(below+equal)/N]. Report our stored board rank only after matching our stable user ID; do not equate that rank with a percentile convention. Cite C-01's historical tab correction and C-04/X-17's final-board equality evidence. The census distribution itself receives no sampling-bootstrap interval.
2. **Final-leader case series (Cohort A):** per user, the number of pick days and double-down frequency under the data contract; missing pick dates counted as missing, never as skips.

   Board/profile season best is kept separate from run reconstruction. Report all recoverable attaining runs and their start/end dates only when retained round-level settlement, transition/entry evidence and boundary history support them. Otherwise report dates unavailable and any defensible observed-segment lower bound, with coverage. For May 1–July 3, streak summaries exclude streak carried into the window and require an explicit boundary/transition rule. max(streak_after) is not the longest streak constructed inside that window. Null-coerced streaks, missing rounds, mixed DD results, Pass and saver ambiguity are not fabricated resets, skips or uninterrupted continuations. If existing evidence cannot resolve them, the exact metric is unavailable; this product does not build a new contest-state acquisition system.

   Report home/away and team concentration only on rows with independently witnessed historical pick-time context, and give coverage/unknown counts. Current final-grab player-team lookup values are not historical team evidence. Lineup-slot availability is omitted because it is not stored in the pick schema; this build introduces no new context-enrichment pipeline.

3. **Labels:** a capture-as-of listing is not an awarded prize. Survivor selection is stated for every Cohort A statistic.

## W2.2 As-of cohort comparison (early manifest E)
**Early manifest E:** the 310 distinct user IDs in the frozen May 1 four-tab manifest, in its recorded order; hash and exact snapshot/tab definition are recorded. **Allocation B:** the grab's first 150 E IDs outside final-rank allocation A. B, E_in_A, E_unfetched and B_shortfall are acquisition labels, not an outcome-independent analytical cohort.

**Primary analytical cohort:** all members of E, including E_in_A and members without usable logs. Membership is never conditioned on final rank, capture success or later activity. The fixed follow-up is May 1–July 3 using the daily corpus. Report per-user usable graded-slot counts and hit rates, the pooled graded-slot rate, and recoverable within-window streak summaries under the history rule below. Keep every member in the availability table, separating capture/history availability from observed activity and unknown inactivity; missing observations do not establish that someone stopped playing or skipped.

**Final-backfill extension:** July 4–September 27 from usable histories of E∩(A∪B), labelled separately and never appended to the primary curve. Report E coverage, E_in_A, budget omissions, fetch/parse failures, no-history responses and history depth. This restricted support is partly determined by final outcomes through allocation A/B and is not an unselected continuation of E. The original D5 capture allocation is unchanged; no new requests are required.

Compare E's usable contest-graded slots with our ledger slots during May 1–July 3. Our denominator requires committed_evidenced selection, uniquely confirmed contest linkage and an exact hit/not_hit slot grade; preserve both DD legs and count exclusions. Use the contest result, not local or current-feed regrading. Report hits/graded slots as a pooled slot ratio, which weights prolific users and DD days more and is not a mean-user skill estimate.

Report each arm's all-observed-date ratio within that fixed window, then both arms' ratios on a single frozen shared-date subset: dates with at least one usable graded slot in each arm. Publish user/date/slot counts and coverage for each table. Jointly resample whole dates for both arms, retaining every selected date's rows and every repeated draw, 10,000 times at seed 20261004; recompute ratios and any descriptive paired difference and report 95% percentile intervals. For the all-date table use the union of observed dates and retain each arm's own denominator; for the shared table use the common dates. Report empty/failed statistics as unavailable. Intervals assume exchangeable date clusters and do not account for cross-date dependence, observation selection or missing histories. No causal value, preregistered test, whole-field skill rank or equivalence follows.

## Exposure
Before outcome-bearing inspection or execution, including identity/revision work that inspects results, joins, summary preparation and resampling, verify published X-24/X-25 and freeze/hash the source, membership and code manifests. X-24 names the board distribution, own rank/percentile and survivor-selected A case series; it discloses X-14 plus X-15/X-17's capture/equality-only history and C-01/C-04. X-25 names all E members, May 1–July 3 follow-up, the restricted final-backfill extension, unknown/attrition handling and the production comparison; it discloses X-10 and applicable X-01/X-09 production reads. X-19 authorized ledger counts only, so it supplies no rate-analysis permission. Disclose X-21 if its labels/outputs are reused, without treating it as permission for this different contest-slot analysis. Under D3 both products remain retrospective descriptions, not preregistered tests; neither selects a W4 candidate.

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
