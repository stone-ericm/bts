# Season 2026 canonical ledger (W1.1) — design

**Date:** 2026-09-28 · **Plan item:** W1.1 (`docs/superpowers/plans/2026-09-14-season-wrap-plan.md` §W1.1, approved 9/22) · **Brief:** confirmed by Eric 2026-09-28 · **Status:** draft → Codex design review → Eric's spec review → implementation plan (writing-plans) → build.

## 1. Purpose
One audit-grade table of every decision the system made in the 2026 season, so the wrap's reads and memos (W1.2 bridge, W1.3 tests, W1.5 incidents, W2 memos, the 2027 decision memo) query one fact base instead of re-deriving from raw files each time. Built once, deterministically, from frozen inputs; not a nightly production job.

**Success:** (a) the ledger exists with explicit conflict and missingness flags; (b) a reconciliation table explains, row by row, the gap between the naive file tally (191 primaries / 157 DD legs, 9/14) and the 9/11 scorecard (141 primaries / 82 DD legs), with unexplained residue listed rather than forced; (c) fixtures cover the hard cases in §8; (d) rebuilding from the same frozen inputs yields byte-identical ledger and reconciliation files (the evidence manifest alone carries a `built_at`).

## 2. Scope
**In:** every season date 2026-03-25 → 2026-09-27; decision streams `production` (primary, double-down, and a row for each no-pick day), `shadow_context` (v1 and v2 context-stack shadow picks), `policy_shadow` (skip-policy shadow records) and `research_d8` (the 9/19–9/27 research-only top-1).
**Out:** the field data from the 9/22 and 9/27 grabs (W2); the decision-weighted live-forward candidate stream (P-04 closed as insufficient; W1.2 reads it directly if needed); re-running any model; changing any production file.

## 3. Inputs (all read-only)
| Source | Where (frozen W0.7 snapshot `final-20260928/` unless noted) | Coverage facts (9/28 inventory) |
|---|---|---|
| Pick files | `data/picks/<date>.json` | 156 files, 3/29 → 9/18; per-slot `slot_results` on 122; `model_git_sha`/`policy_npz_sha256` on 122; `feature_env_hash` on 99 (from 5/26); `notification_sent` on 104; `delivered_at` on 20; `policy_decision`/`tail_policy_sha256` on 16 |
| Decision files | `data/picks/<date>/decision.json` | 92 files: v1 from 6/23, v2 from 8/09, v3 from 9/03. Absence before 6/23 is not a skip |
| Scheduler state | `data/picks/<date>/scheduler_state.json` | 167 files: lock time, committed pick, skip candidate, delivery refusals, fallback refreshes |
| Shadow / policy shadow | `data/picks/<date>.shadow.json`, `<date>.policy_shadow.json` | 118 and 16 files |
| Slate snapshots | `data/picks/slates/<date>.json` | 105 files from 6/11: the full selectable slate (~270 rows) with stated p per batter |
| Contest ledger | `data/picks/account_state/contest_ledger.jsonl` | 411 records 6/17 → 9/28; the latest record lists every graded entered round (136 rounds, round ids 827 → 995) with per-slot `playerId`, `result`, `hits`, `atBats` |
| Saver / streak state | `account_state/saver_transitions.jsonl`, `saver_state.json`, `contest_streak*.json`, `data/picks/streak.json` + the three late-May repair archives | saver changes with timestamp and source |
| Research D8 | `data/validation/decision_weighted_lgbm_v0_live_forward_research/<date>/` | 9/19 → 9/27 |
| Journals | `journal_bts-scheduler_retained.txt` | from 5/11: supporting evidence for delivery and lock times only |
| Deploy timeline | `deploy_history.txt` (reflog), `config/` | recipe-epoch boundaries |
| BTS static lookups | `data/leaderboard/final_grab_20260927/raw/static/{001_rounds,002_players}.json.gz` | round id ↔ date; BTS `playerId` ↔ MLB `feedId` |
| Pre-cutoff game feeds | **box** `data/raw/2026/<gamePk>.json` (NOT in the snapshot) | first pulled 03:00 ET the day after each game; before the 08:00 cutoff for 250 of 279 pick slots; 27 early-season games cached 4/11 (after their cutoffs); 2 missing |
| MLB final record | statsapi, fetched once at build time | today's official box score per game (includes late re-scorings) |

**Freezing the two non-snapshot inputs.** The builder copies every raw feed it grades from, and every MLB final-record response it fetches, into an evidence bundle `data/hetzner_results/season_2026_ledger_evidence/` (inside the archive backup set) with a sha256 manifest. Later builds read the bundle, never the live directories, so the ledger stays reproducible.

## 4. Architecture — per-source observations → one reconciled view
**Approaches considered.**
- (A) *Extend `scripts/canonicalize_realized_picks.py`.* Rejected: it grades from the processed PA frame, i.e. MLB's current record, which repeats the C-03 error for BTS outcomes; its scope (calibration rows, regime labels) differs.
- (B) *One flat builder straight to the final table.* Simpler, but a disagreement between sources disappears into whichever value wins.
- (C) **Recommended — the plan's shape:** each source is parsed into its own normalized observation table (one row per source fact, carrying `source_path` + `source_sha256`); a reconciler joins them into the ledger with an explicit precedence per field and flags every disagreement and gap. Parsers are small and testable in isolation; precedence lives in one place.

**Units.** `sources/*.py` (one parser per source; pure functions over bytes → rows), `reconcile.py` (keys, precedence, flags), `grade.py` (BTS-rule grading of a slot from one feed; reuses `picks.grade_pick_in_feed`), `evidence.py` (copy + manifest), `build_season_ledger.py` (CLI). Kept under `scripts/audit/season_ledger/` (audit tooling, not `src/bts`).

## 5. Ledger schema (one row per date × stream × slot)
**Keys:** `date` (ET contest date) · `round_id` · `stream` · `slot` (`primary` | `double_down` | `none` for a no-pick row) · `decision_revision` (decision.json `finalized_at`, else pick-file `run_time`) · `objective` (`reach57` | `emax_season_best` | `absent`).

**Pick:** `batter_id`, `batter_name`, `team`, `game_pk`, `game_number`, `lineup_position`, `projected_lineup`, `pitcher_id`, `p_stated` (served `p_game_hit`), `slate_ref` (slate file + sha256 when one exists).

**Decision:** `action` (single / double / skip), `action_source` (mdp / heuristic / forced / tail), policy hashes (`policy_npz_sha256`, `tail_policy_sha256`), `streak_before_decision` and its `state_source`/`state_status`, `effective_best`, `degraded_reason`.

**Timeline:** `predicted_at`, `locked_at`, `delivery_attempted`, `delivery_confirmed`, `delivery_status`, `game_start`, `entered` (contest holds a graded slot for this round and batter).

**Outcomes — three columns, never merged:**
- `bts_result` (`hit` | `no_hit` | `pass` | `pending` | `unknown`) with `bts_result_source`: **contest** for entered slots (the contest's own grading; its round label `void` means a miss at streak 0 and is not a missing entry); otherwise **pre_cutoff_feed** (the cached feed if pulled before date + 1, 08:00 ET, graded by BTS rules: a hit, else an at-bat or sacrifice fly = no_hit, else pass; suspended games on pre-suspension plate appearances only); otherwise `post_cutoff_feed` or `none`, flagged.
- `mlb_final_result` from the MLB final record, same grading rules.
- `local_recorded_result` from the pick file (`slot_results`, or the legacy day result where no slot results exist).
- Flags: `rescored_after_cutoff` (bts ≠ mlb_final), `local_disagrees` (local ≠ bts), `legacy_day_fallback`.

**Contest state:** `contest_streak_before`, `contest_streak_after`, `contest_round_result`, `saver_available_before/after` (contest where present, else saver transitions, flagged).

**Recipe epoch:** `epoch_id` built from the deployed code SHA (reflog timeline), `feature_env_hash` (from 5/26), policy hashes and determinism settings; rows the fingerprints cannot place are `epoch_unknown`, never silently the current recipe. The canonicalizer's `post_bpm` label is carried as one extra column, not as the epoch.

**Flags (booleans + a reason string):** `conflict_*` for each field pair that disagrees (e.g. pick file vs decision.json action; contest batter vs our slot batter) and `missing_*` for each absent input (no decision.json before 6/23; no slot_results; no delivery evidence; no pre-cutoff feed; no contest observation).

## 6. Precedence rules (reconciler)
1. **Entered production slots:** the contest record decides `bts_result`, `entered` and contest streaks. A contest slot is matched to our slot by batter (BTS `playerId` → MLB `feedId` → `batter_id`), never by the contest's slot `number`, which does not follow our primary/double-down order.
2. **Not-entered slots:** pre-cutoff feed grading; if the only feed was pulled after the cutoff, `bts_result_source = post_cutoff_feed` with a flag.
3. **Action / objective:** decision.json where it exists (6/23 →), else pick-file `policy_decision`, else inferred from pick-file shape (DD present → double) and flagged `inferred`.
4. **Delivery:** decision.json `delivery_status`, else pick-file fields by era (`notification_sent`, `delivered_at`, `bluesky_posted`), else journal lines; `delivery_confirmed` needs a success signal, not an attempt.
5. **Local results never override** the contest or a pre-cutoff feed; they are kept for the `local_disagrees` flag (C-03).

## 7. Reconciliation table (acceptance)
For every primary and double-down slot in the pick files: included or excluded under (a) the 9/11 scorecard recipe (primaries with a hit/miss result, 3/29 → 9/10, as recorded in memory `bts_index`) and (b) the 9/14 naive tally recipe, each with a reason code (e.g. undelivered preview graded locally, private day, skip day with a preview file, void, unresolved, outside window). Date windows are matched explicitly. If a recipe cannot be recovered exactly, the table says so and shows the closest reconstruction; it never forces the published number. Output: `season_2026_reconciliation.parquet` + a markdown summary.

## 8. Fixtures (tests)
Undelivered-but-graded preview · preview overwritten by a later commit · DD leg voided (postponed game) · doubleheader (two game_pks, same batter date) · private_locked day · tail-period skip row · **C-03** (local miss vs contest hit) · contest `void` = miss at streak 0 · legacy DD without slot_results · pre-cutoff vs post-cutoff cached feed · suspended game graded on pre-suspension plate appearances · contest slot order differing from ours · a row the epoch fingerprints cannot place. All fixtures are synthetic files built in the test; no test reads the real snapshot.

## 9. Outputs
`data/validation/season_2026_ledger.parquet`, `data/validation/season_2026_reconciliation.parquet` (+ `.md` summary), the evidence bundle in §3, `scripts/audit/season_ledger/` + `scripts/audit/build_season_ledger.py`, `tests/scripts/test_build_season_ledger.py` (and per-unit tests), a memo `docs/audit/<date>-season-ledger.md`, and a W1.1 row update in the wrap index. The ledger itself is an outcome fact base: reading it for any analysis goes through the exposure register as usual. **Exposure note:** the 9/28 source inventory that scoped this spec counted result labels per source (pick-file results, contest round labels) — no rates, comparisons or model reads — recorded as register X-18; this spec cites coverage counts only.

## 10. Risks and open points
- **The 9/14 naive recipe may be unrecoverable** (191 > 156 current pick files, so it must have counted other files); §7 handles this by reporting, not forcing.
- **BTS player-id coverage:** the 9/27 static players lookup must map every entered slot; any unmapped slot is flagged, not guessed by name.
- **Early-season feeds** (3/31 → 4/10 cached 4/11) cannot give a pre-cutoff `bts_result` for not-entered slots; entered ones use the contest.
- **Two pick slots have no cached feed**; they get `mlb_final_result` only.
- **Journals** start 5/11 and their line formats drift; they are supporting evidence, never the only source for a field.
- **Determinism:** fixed ordering, no wall-clock values in outputs except a single `built_at` in the manifest; the byte-identical rebuild check runs in the test suite on a fixture tree.
