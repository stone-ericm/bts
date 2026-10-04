# C1 rank 3 registration: a pregame plate-appearance count distribution against the fixed slot table

**Status:** design rev 2, 2026-10-04: Codex trio design r1 **BLOCK**; B-E1–B-E5 and X-E1 applied verbatim by script (review `docs/audit/2026-10-04-c1-trio-design-codex-r1.md`); for design round 2 of 2, then freeze or defer. The 2027 shadow archive is production code: reviewed until SIGN, shipped only with Eric's D7 approval.
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`).
**Plan row 3:**
- The same PA models and slates, comparing the fixed slot-count baseline against one pregame count-distribution / starter-allocation candidate, scored on proper scores fitted earlier and tested later.
- **Kill if** the candidate needs actual-exposure inputs; or the hits-per-PA target changes inconsistently; or there is no residual beyond the June PA-tilt null.

**Exposure:** row X-34, published before any 2027 outcome is read. Fitting uses only 2021–2025 (no 2026 outcome; D3).

## In plain words
- **How it works today.** Production turns "chance of a hit per plate appearance" into "chance of a hit in the game" with a fixed table: about 4.5 trips to the plate for the leadoff hitter down to 3.6 for the ninth. The first 2.5 are against the starter.
- **The candidate.** It uses a fitted distribution of how many trips a batter in that slot gets, and how many come against this particular starter, estimated from 2021–2025 using only what is known before the game.
- **The test.** In 2027 both versions are computed side by side at every pre-lock run (only the current one is served), then scored on how accurate they were.

## 1. Target (ruling R1: fixes the kill condition on target consistency)
- **The target is P(hit | N ≥ 1):** the chance of at least one hit given the batter gets a plate appearance. BTS grades a game with no plate appearance as a Pass (void), and 4a and W2.3's T2 exclude no_pa the same way.
- **Both arms** are evaluated on known hit / no_hit rows; no_pa and unknown are excluded and counted.
- **PA rows** follow the same definition as q (the PA parquet's `PA_ENDING_EVENTS`), with the resumed portion excluded (`read_pa_for_bts_scoring`).

## 2. Arms
- **Baseline B, production unchanged** (`src/bts/model/predict.py` 730–746):
  - slot table {1: 4.5, 2: 4.3, …, 9: 3.6}, 4.0 for a missing slot;
  - `starter_pas = 2.5`, the rest against relievers; openers all against relievers;
  - p = 1 − (1−q_s)^{N_s} (1−q_r)^{N_r}.
Candidate count: for certified historical starting batters, count scoring-definition PAs with resumed portions excluded, condition on N≥1, and fit the slot×home/away empirical table with add-one smoothing. The forecast explicitly caps N at 8: category 8 contains N≥8, and its formula uses 8. Report the historical overflow count; do not drop overflow games or renormalize them out. This is a declared capped-count approximation.

BF is the mean completed-start batters faced from the latest five available starts on strictly earlier official dates. Same-date starts never enter. Per-start BF uses complete PA-event counts for the certified starter, including legitimate resumed PA; this workload input is distinct from the resumed-excluded contest count target. History is built from certified 2021–2025 sources and completed prior-date 2027 sources; no 2026 count/outcome fitting is introduced. If fewer than five prior starts exist, use all available prior starts; if none exist, use the fixed 2021–2025 league median per-start BF. Missing identity, invalid BF or invalid/missing slot causes candidate p to equal baseline p and is flagged/countable, not an outcome-dependent exclusion.

For slot k, N_s(n)=min(n,max(0,floor((BF−k)/9)+1)); openers use N_s=0. Candidate p=1−Σ_n P(n)(1−q_s)^N_s(n)(1−q_r)^(n−N_s(n)). Both arms use the same per-model starter q, the same single-model reliever q shared across blend models, and production's active-model/NaN/single-model-fallback conventions. Archive those inputs at the same immutable run snapshot. BF calculation and its availability witness are new shadow functionality, not an assertion that BF is already computed in production.

**One candidate.** No second count specification is tried in C1.

## 3. Admissible inputs
Prospective inputs are only the run-known projected/confirmed slot, home/away, probable starter identity, lagged BF and opener flag. Realized slot/starter/N or actual-starter-matchup participation never selects a forecast or prospective pool. Historical N≥1 conditioning constructs the declared conditional training target; excluding no_pa from scoring constructs that target's evaluation support. Neither authorizes retrospective selection of prospective forecast rows.

## 4. Fitting the count table on 2021–2025 (frozen before 2027)
The count build requires the 2021–2025 scoring PA parquets plus authoritative starting-lineup, opposing-starting-pitcher and chronological PA-order metadata for every included game, bound to source bytes in the pre-build manifest. The current PA schema alone does not contain those witnesses. First-seen batter/pitcher is not treated as certified starting identity; no first-inning test is claimed from a schema without innings.

Before fitting, verify metadata/PA game and side identities, starting slots, chronology, substitutions and PA-event completeness. Quarantine every unresolved or contradictory game before aggregation and count it; never retain a bad game merely because failures total ≤1%. Stop if quarantined games exceed 1% of the eligible source games, or if authoritative metadata is unavailable. With unavailable metadata the build is deferred. New public-feed acquisition requires a separately recorded scope decision and is not authorized by this registration. Producer-version/order provenance is recorded; a cyclic slot pattern alone is not an order certificate.
- **Outputs:**
  - the count table (slot × home/away → distribution over N);
  - each starter's per-start batters faced, giving lagged BF without leakage (date-level `shift(1)`);
  - an artifact with sha256 and a manifest of input hashes.
- **Leakage:** the count table is a fixed historical aggregate applied to 2027 only; lagged BF is computed with date-level `shift(1)`. `scripts/leakage_audit.py` is unaffected (no PA-model feature changes); this is stated, not run.

## 5. The 2027 shadow archive

With production-code SIGN and Eric's D7 approval, archive complete ordered run snapshots append-only under data/picks/runs/<date>.jsonl from the first registered date. Each snapshot carries schema/run identity, prediction/input-availability/write timestamps, date, ordered (batter_id,game_pk) pool, run-known game time and pregame-status evidence, slot/state, home/away, probable starter, opener, lagged-BF source/hash, count-artifact hash, serving recipe/environment/model hashes, active-model identities, each model's starter q, shared reliever q, baseline p, candidate p and fallback reasons. Run completion and terminal production-decision/commit events have stable identity links. Partial writes are not complete snapshots.

Primary and ranking comparisons use one common complete run per date: the latest run whose inputs and complete archive were available strictly before both the first terminal production commit, if one occurred, and the earliest run-known submission cutoff of the registered date's game pool. On an evidenced no-commit/skip date use that common earliest cutoff. If the commit boundary or required eligibility witness is unavailable, exclude/count the date; do not infer no commit from a missing file. Do not stitch a different run for each candidate or substitute post-run realized times/starters.

Choose the run/pool without outcomes and before inspecting candidate success. A missing or invalid candidate computation on an otherwise valid baseline row uses baseline p as the candidate fallback, with a counted reason; it does not remove the row or select an older run. Both arms rank the same eligible pool before outcome exclusions and preserve baseline row order for ties. Unknown/no_pa winners are never replaced.

Shadow inputs are copied without modifying predictions, shared model/cache state or decision state. Shadow failure, storage failure or slow work must not block the active pick/delivery path; late/incomplete shadow output is counted as unavailable. No additional acquisition is made inside the shadow.

Before code review, fixed-clock red/green fixtures compare shadow on/off production slate, pick and decision bytes, selected identities/action/partners, eligibility/state/defer results, delivery and transport traces, and cutoff behavior. Cover ordinary, opener, missing-input, skip/private, cached-fallback, restart and shadow-error/slow paths; include a decision-path mutation that leaves pick/slate bytes unchanged and must fail. Verify the hand formula, same-date history exclusion, complete-run selection, actual-commit-before-cutoff and exact-cutoff exclusions. Production review remains until SIGN; deployment still needs Eric's D7 approval and the normal deploy/canary gates.

## 6. Split, metrics and thresholds (fixed before any 2027 outcome)
- **Fit:** 2021–2025, fixed now.
- **Test:** 2027 contest dates 1–90, one analysis after date 90 is graded.
- **Primary:** the equal-date mean log-loss difference, C minus B, on the as-of rows (all served candidates with known outcomes), with a 95% whole-date bootstrap (10,000 draws, seed 20270102).
- **Dispositions:**

| Disposition | Condition |
|---|---|
| **Positive** | difference ≤ **−0.001** with the upper bound < 0, **and** the guardrail holds |
| **Negative** | difference ≥ 0 |
| **Inconclusive** | anything else |

Fixed calendar numbering, settled-source completeness, invalid-probability handling and rank-before-outcome rules follow 4a's registration. Test dates are 1–90, with no extension; freeze and analyze once at 08:00 ET after date 90. Require at least 75 primary-scoreable dates and 75 known baseline rank-1 dates or record inconclusive (insufficient support). Use the same baseline rank-1 identities for the regression guardrail in both arms; arm-specific winners are a separate descriptive comparison on common-known dates.

Bootstrap the precomputed paired date-loss differences directly, preserving draw multiplicity: 10,000 draws, seed 20270102, 2.5th/97.5th percentile primary interval. The 95th-percentile upper bound of baseline-rank-1 loss differences must be ≤0.005 nats (10,000 draws, seed 20270104). Keep the −0.001 primary bar and registered dispositions. Report all calendar/run/eligibility/fallback/outcome exclusions; no data-dependent window or threshold changes.

Precision rationale is assumptions only: at 90 independent paired dates, MDE ≈2.802·σ_date/√90 for normal two-sided 5%/80% sign detection. Assumed SDs 0.005/0.010/0.020 imply approximately 0.00148/0.00295/0.00591 nats. These are not measured power, do not certify the joint practical-threshold/guardrail gate, and exclude cross-date dependence. The window and thresholds stay fixed if precision is insufficient.

- **Secondary, descriptive:** Brier; within-date AUC; stated − realized for each arm; how often the rank-1 batter differs between arms, and each arm's hit rate on those dates.
The June PA-tilt null is carried as context, not rerun or overturned by an all-candidate proper-score gain. A positive primary result is labelled positive (forecast scores only). It does not establish a residual ranking/decision lever beyond that null, recoverable oracle lift or improved play. That plan kill condition remains unresolved unless a separately frozen downstream test provides candidate-specific evidence; C1 does not search another count specification to close it. A null primary receives the registered negative/inconclusive disposition and is not retried.

The formula assumes independent PA hit trials conditional on N and the deterministic starter allocation, with the archived q values applying across those N values. The actual PA models do not directly fit that conditional hazard; because hits can extend innings and change N, this is a forecast approximation to be tested, not an exact identity for their marginal q. Jensen's lowering result applies to a common one-rate comparison with equal means, not generally to this two-rate candidate against the production table. Count means, allocations and q interactions can also change ranks. No achieved PA-mechanism lift is inferred from a probability-level shift.

## 7. If positive
Before any production use, require independent acceptance of the declared forecast comparison and a separately registered applicable state/eligibility/downstream-value replay with regression and calibration/dependence stress, followed by Eric's D7 approval. This candidate is not a common scalar monotone map and has no presumed behavior-preserving boundary-remapping path. Any score/ranking/policy effect is assessed as a named change, without combining it with 4a or 4b in one comparison.

## 8. Compute
- **Count-table build on the box:** through the launcher (`c1-r3-build`), **declared 6 CPU-hours.** It reads 5 seasons of parquet.
- **Evaluation after date 90:** declared 0.5 CPU-hours.
- **The shadow:** adds one sum over at most 8 terms per candidate per model, which is negligible.

## Limits
- **Historical fit:** the count table is fitted on 2021–2025, with no 2026 data, and may not transfer to 2027 rules.
- **Starter split:** a deterministic batting-order cutoff, not a fitted hazard.
- **Lineup state:** projected lineups had many more no_pa rows in 2026 (541 / 3,006 against 56 / 9,720 confirmed, W1.3). The candidate does not model the probability of not starting; that remains excluded by the target.
- **Window:** one 90-day window in one season.

## Freeze manifest and cross-design rules (trio review X-E1)
Before each authorized fitting/evaluation/diagnostic run, publish the appropriate exposure row and an outcome-free freeze manifest: reviewed registration/code commit and hashes; old serving recipe, model-training/retraining schedule, active blend and aggregation/fallback definitions; configuration/environment including calibration/deterministic/seed flags; fixed calendar/as-of/eligibility rules; and count/identity/reader artifacts as applicable. Future forecast/model/input hashes are recorded with each immutable capture. Pin/hash the exact consumed fitting/outcome/input bytes at the declared freeze, and parse those same bytes. A missing pin, mismatched hash, unsupported schema or unregistered recipe change refuses acceptance; naming a directory is not a pin. Ordinary model retraining under the frozen schedule is allowed and its artifact hashes are recorded. A recipe change does not trigger a post-result refit, window reset or silent pooling; it is reported and the affected study is inconclusive pending a separately approved prospective registration.

X-32 covers 4a's fit and test; X-34 covers rank 3's historical fitting and prospective evaluation; X-33 covers only the outcome-free broader-field/identity diagnostic. The index distinguishes X-33 from receipt-only capture. These rows are published before their respective reads, with no 2026 candidate outcome test. All new artifacts/readers preserve both plain and gzipped static input support and the declared missingness/fallback rules.

Each forecast study has one fixed primary comparison and its declared regression gate. Their shared dates create dependent evidence, not replication; secondary metrics are descriptive and cannot select another map/count specification or a combination. Keep 4a, rank 3 and 4b comparisons separate under D2. No fitted 4a output is supplied to rank 3 and no rank-3 output is supplied to 4a. Independent result acceptance and Eric's D7 approval precede any named production change; applicable policy replay and D1 trade approval remain separate. The approved C1 launcher, cumulative caps, sleep-window/production-safety and calendar stops apply. An unresolved rank-1 403/429 stop pauses C1 until Eric's recorded resumption and keeps capture separately disabled until his recorded reset.
