# C1 rank 3 registration: a pregame plate-appearance count distribution against the fixed slot table

**Status:** design rev 1, 2026-10-04, for Codex design review (at most 2 rounds, then freeze). The 2027 shadow archive is production code: reviewed until SIGN, and shipped only with Eric's D7 approval.
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
- **Candidate C, one fixed specification:**
  - **Count:** P(N = n | slot, is_home), n = 1…8. The empirical distribution of plate appearances for *starting* batters in 2021–2025, with add-one smoothing, conditioned on N ≥ 1.
  - **Starter split:** the opposing starter's batters faced, BF = the mean of his last 5 starts' batters faced (date-level `shift(1)`; no prior start falls back to the 2021–2025 league median).
    - The batter in slot k faces the starter in his j-th trip if 9(j−1) + k ≤ BF, so N_s = min(N, ⌊(BF − k)/9⌋ + 1) for BF ≥ k, else 0.
    - Openers: all against relievers, as in production.
  - **p** = 1 − Σ_n P(n) (1−q_s)^{N_s(n)} (1−q_r)^{n − N_s(n)}.
- **Shared inputs:** both arms use the same q_s and q_r per blend model, and the blend averages game-level p per model exactly as production does.

**One candidate.** No second count specification is tried in C1.

## 3. Admissible inputs (ruling R2: answers the actual-exposure kill condition)
- **Allowed:** lineup slot *as known at the run* (projected or confirmed), home/away, the probable starter's identity and lagged batters faced, the opener flag. All are already in production's run-time state.
- **Inadmissible:** realized slot, realized starter, realized N, and any filter on batter-games with N ≥ 1 or a starter matchup (the `estimated_pa` backtest's conditioning). The candidate is evaluated only on forecasts archived before lock (§5).

## 4. Fitting the count table on 2021–2025 (frozen before 2027)
- **Source:** the PA parquets `data/processed/pa_{2021..2025}.parquet` (on the box). The raw 2021–2025 feeds are not on the box or the Mac (checked 10/04).
- **Derivations** (the parquet is written in play order per game, `build.parse_game_feed` iterating `allPlays`):
  - the starting batter for each (game, side, slot) = the first batter seen in that slot;
  - each side's starting pitcher = the pitcher of the first PA against that side.
- **Order check first:** row order is verified before use. In every game, each side's first-PA pitcher must be one pitcher, and slot 1–9 first appearances must occur in batting order in the first inning's PAs.
  - If the check fails for more than 1% of games, the build stops.
  - Re-acquiring raw feeds from the public MLB API (about 12,000 requests) would then need a recorded scope decision; it is not authorized here.
- **Outputs:**
  - the count table (slot × home/away → distribution over N);
  - each starter's per-start batters faced, giving lagged BF without leakage (date-level `shift(1)`);
  - an artifact with sha256 and a manifest of input hashes.
- **Leakage:** the count table is a fixed historical aggregate applied to 2027 only; lagged BF is computed with date-level `shift(1)`. `scripts/leakage_audit.py` is unaffected (no PA-model feature changes); this is stated, not run.

## 5. The 2027 shadow archive (production change, D7)
- **What it writes:** at every production run, an append-only per-run record `data/picks/runs/<date>.jsonl`. For every slate candidate it holds `written_at`, game time, slot (projected / confirmed), the starter used, BF, the opener flag, baseline p and candidate p.
- **What it must not do:** the served p, the slate file (`save_slate` keeps last-write-wins) and every decision stay byte-identical. A test asserts identical slate and pick bytes with the shadow on and off.
- **The as-of rule:** for each candidate, the evaluated row is the last run written before first pitch − 5 minutes (`picks.SUBMISSION_CUTOFF_MIN`). Runs after first pitch are excluded (they may carry the actual starter).
- **Tests first, then review:** a formula fixture (hand-computed p for a known distribution, BF and slot), an opener fixture, a byte-identity test and a cutoff-selection test. Then deploy-gating review until SIGN, Eric's D7 approval, and a deploy before the first 2027 contest date.

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

- **Guardrail:** rank-1 rows' log loss is not worse by more than 0.005.
- **Secondary, descriptive:** Brier; within-date AUC; stated − realized for each arm; how often the rank-1 batter differs between arms, and each arm's hit rate on those dates.
- **The June PA-tilt null** (rank-1 AUC of estimated PAs 0.516; re-ranking by estimated PAs p = 0.91) is reported as context. A null primary here is consistent with it and is recorded as negative or inconclusive, not retried.

**Rationale (reasoning only).**
- The candidate lowers p relative to the plug-in at the same mean (Jensen), which overlaps 4a's intercept shift. The two are separate studies and are never combined in one comparison.
- Ranking changes only where the count spread differs between candidates (slot, projected vs confirmed, opener, short-starter matchups).

## 7. If positive
- **What it establishes:** a better forecast on the test window, and only that.
- **Before it changes play:** the same two paths as 4a, put to Eric under D7: (i) consistent boundary remapping, or (ii) a policy change with the 4b-style replay gate.

## 8. Compute
- **Count-table build on the box:** through the launcher (`c1-r3-build`), **declared 6 CPU-hours.** It reads 5 seasons of parquet.
- **Evaluation after date 90:** declared 0.5 CPU-hours.
- **The shadow:** adds one sum over at most 8 terms per candidate per model, which is negligible.

## Limits
- **Historical fit:** the count table is fitted on 2021–2025, with no 2026 data, and may not transfer to 2027 rules.
- **Starter split:** a deterministic batting-order cutoff, not a fitted hazard.
- **Lineup state:** projected lineups had many more no_pa rows in 2026 (541 / 3,006 against 56 / 9,720 confirmed, W1.3). The candidate does not model the probability of not starting; that remains excluded by the target.
- **Window:** one 90-day window in one season.
