# Season 2026 Ledger — Phase 1 Implementation Plan (rev 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 canonical ledger of the 2026 season's production decisions from a sealed evidence bundle:
- lossless, typed source observations, persisted with their raw records
- qualified contest evidence matched to specific games
- explicit row kinds
- an honest reconciliation

**Architecture:** `acquire` runs on the box. It copies every input from the frozen W0.7 snapshot into a sealed bundle with a sha256 manifest; its only network access is the MLB schedule fetch. `compile` verifies the bundle and runs offline:
1. Per-source parsers.
2. Accounting and an independent structural census, run before any ledger row exists, which proves every record is accounted for exactly once (the anti-join).
3. Contest slot history and game matching.
4. Day and row rules.
5. Outcomes and eligibility.
6. Recipe reconciliation over the full membership universe.
7. Output-phase checks (typed references, dispositions, recipe links, contest-slot identity, season days), then deterministic parquet and markdown outputs written into a newly reserved directory.

Everything lives under `scripts/audit/season_ledger/`; the package imports no production module.

**Tech Stack:** Python 3.12, stdlib (`json`, `gzip`, `hashlib`, `re`, `urllib`, `zoneinfo`, `platform`, `inspect`), `pyarrow` 23 (already a dependency), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-28-season-ledger-design.md` (v4, approved on 2026-09-28 by Codex design r4, Claude and Eric).

## Global Constraints
- **Scope:** Phase 1 covers the production stream only (spec §2). It excludes shadow v1/v2, the skip-policy shadow, D8, cached-feed grading, MLB's current record, recipe epochs and slate binding.
- **No production changes:** new code only under `scripts/audit/season_ledger/`, `scripts/audit/build_season_ledger.py` and `tests/scripts/season_ledger/`.
- **Self-contained package:** it imports nothing from `bts`. The two decision helpers it needs are vendored, and a test pins them to production (Codex plan r1 #13).
- **Network:** `compile` performs no network access. The only network call in `acquire` is `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=<D>&gameType=R&hydrate=team`, once per season date, paced at 0.5 s.
- **Authoritative outcome:** the only authoritative BTS outcome is a qualified contest slot grade whose match is `evidenced` or `inferred` (spec §6–§7).
- **Never inferred** (spec §5–§6, §9): entry absence, history `complete`, saver availability, and eligibility from an outcome. Phase 1 therefore emits:
  - `entry_status ∈ {confirmed, unknown}`
  - `history_status ∈ {known_incomplete, unknown}`
  - `saver_available_before = null`
  - `game_eligibility = unknown` unless there is timed evidence from before lock (I6)
- **Recipe rules are hypotheses** (spec §8):
  - Task 9's rule table and everything its evaluator reads are fixed now and fingerprinted as `rules_fingerprint() = 5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f`. That covers the source of every repo-local function reachable from the recipe roots (predicates, universe, windows, labels, the JSON decoder) and every configuration value those functions read (regexes with their flags, the time zone, tables).
  - The fingerprint is recorded in the exposure register before the real run (Task 13 Step 1), and the build must reproduce it.
  - The rules must not be edited after the first real run.
- **P-01 is untouched:** the completed P-01 read and `scripts/audit/build_slot_dataset.py` are not modified.
- **Tests:** synthetic files only. No test reads the snapshot, the box or the network. Gzip fixtures pin `mtime=0`.
- **Commands:** every `uv` command uses `UV_CACHE_DIR=/tmp/uv-cache`, and pytest runs with `TZ=America/New_York`.
- **Output locations:**
  - Each build reserves a new directory with `mkdir` and refuses one that already exists, even an empty one. On the box that is `data/validation/season_2026_ledger/<code-sha>-<run-id>/` (gitignored).
  - Only a directory holding `ACCEPTED.json`, written after the twin-build comparison (Task 13 Step 4), is a published build. Failed attempts stay where they are.
  - The evidence bundle goes to `data/hetzner_results/season_2026_ledger_evidence/v1/`, inside the `archive` restic set. A later acquisition is `v2`, never a rewrite.
- **Git:** work on `main` (audit scripts; `main` does not deploy), with one commit per task. Commit messages end with the session's attribution lines.
- **Box:**
  - Nothing is deployed.
  - The reviewed code reaches the box as a `git archive` of a named commit under `/tmp/ledger_code_<sha>`.
  - No service restarts.
  - Jobs run as `systemd-run --user` units through one runner. It prints `RUN=<id> START` first and `RUN=<id> EXIT=<status>` last, and exits with that status. Completion is read back from the log by run id.
- **Exposure:** exposure-register row X-19 is written and committed before any real data is compiled (Task 13 Step 1).

## Verified source facts (structure-only probe of `final-20260928/`, 2026-09-28; key names, types and counts, no values)
- **`data/picks/` files by name pattern:**
  - Production files: 156 `DATE.json` (primary `game_pk`/`game_time` null only on 3/29–3/30), 92 `DATE/decision.json`, 167 `DATE/scheduler_state.json`, 27 `DATE/deferred_fallback_<YYYYmmddTHHMMSS-0400>.json`, 124 `lineup_evolution_DATE.jsonl`.
  - Shadow and slate files: 118 `DATE.shadow.json` (has `pick`), 16 `DATE.policy_shadow.json` (no `pick`), 28 `backup_shadow_DATE/DATE.shadow.json`, 105 `slates/DATE.json` (no `pick`).
  - Archives and repairs: 1 `archive/DATE.json.postponed` (has `pick`; no writer in the code, so it was a manual move); `archive_actual_streak_repair_<stamp>[_missed_N]/` holds 4 `DATE.json` + 4 `DATE.json.before` + `streak.before.json`; `archive_replay_restore_<stamp>_post_contest_state_deploy/{README.txt, streak.before.json}`.
  - State and markers: 14 AppleDouble `._*` files, `streak.json`, `.nrestarts_checkpoint`.
  - `account_state/`: `contest_ledger.jsonl` (411 lines), `saver_transitions.jsonl`, `contest_streak.json`, 3 `contest_streak.manual.json.*` variants, `saver_state.json`.
- **Time formats:**
  - `run_time`: `+00:00`, with and without microseconds.
  - `delivered_at`, `pick_locked_at`, `deferred_at`: ET offset with microseconds.
  - `finalized_at`, `recorded_at`: `Z` with microseconds.
  - `game_time`: `Z` without microseconds.
- **Contest ledger:**
  - Every line has `recorded_at`, `active_streak`, `best_streak`, `source_date` and `predictions`. Each line carries the account's full season of rounds so far (70 rounds on the first line, rising to 136 distinct roundIds 827–995).
  - Round keys: `roundId`, `result`, `streak`, `streakIncrease`, `roundPredictions`. Slot keys: `number`, `unitId`, `playerId`, `result`, `hits`, `atBats`.
  - 5 rounds and 18 slots with null `result` all sit in the latest round of their line.
  - 3 slots have null `playerId` (lines 89, 329 and 353).
- **BTS static captures** (`data/leaderboard/static_snapshots/<feed>/<YYYYmmddTHHMMSSZ>.json[.gz]`, plus a `.last_sha256` marker per feed):
  - rounds: 173 files, 1 MB. `{"rounds": [{id, date, status, contestId}]}`, 188 rounds 823→1010, one per date 3/25→9/28, no id gaps.
  - units: 2,335 files, 270 MB. `{"units": [{id, feedId, roundId, status, startDateTime, lockDateTime, awaySquadId, homeSquadId, lineups, …}]}`. The status vocabulary is `scheduled`/`playing`/`complete`/`postponed`, and the last capture is an empty list.
  - players: 3,773 files, 1.5 GB. `{"players": [{id, feedId, squadId, name, …}]}` (2,930 players).
  - The 9/27 grab's `raw/static/` holds `001_rounds`, `002_players` (143 KB), `003_units` (empty) and `004_squads` `.json.gz`.
- **Snapshot root:** holds `cron.log` and `journal_bts-scheduler_retained.txt`. Snapshot pick files keep their live mtimes.
- **Box code:** the box's `src/bts/daily_decision.py` is byte-identical to main's (sha256 `f04217f5…`).
- **MLB schedule:** the endpoint with `hydrate=team` gives `teams.{away,home}.team.abbreviation`, identical to the live feed's `gameData.teams.*.abbreviation`, which is the source of `Pick.team`. It lists `Cancelled`/`Postponed` games (e.g. 9/27 BAL@NYY `C Cancelled`).

## Revision 2 — how this answers Codex plan r1 (`docs/audit/2026-09-28-season-ledger-codex-plan-r1.md`, BLOCK)
*This table is kept as history, with rev 2's names; the Revision 3 table below records what replaced them (e.g. `required_locators` and `census_problems`, and a reserved output directory).*

| # | Finding | Change |
|---|---|---|
| 1 | Omitted occurrence inside a file not detected | Occurrences carry their real locators. Task 9 adds an independent structural census (`expected_locators`, `census_gaps`), run by `check_invariants` in every build: an omitted leaf or phantom locator fails the build. Also added: a closed disposition vocabulary and referential integrity for every ledger `*_obs_id`. Tests delete or add an occurrence inside a double-down file and a multi-line contest file. |
| 2 | Observations not lossless or persisted | Every parsed row keeps nested absent-vs-null (`pick.pitcher_id`), type mismatches, full provenance (`policy_decision_json`, `feature_env_json`, `feature_env_schema_version`, `shadow_model_version`, pitcher fields) and its raw record (`record_raw_json`, per line/round/slot for the contest). Scheduler state keeps `final_skip_candidate_json`, `delivery_refusals_json` and `fallback_refreshes_json`. The occurrences table now persists `fields_json` and `record_raw_json` for every emitted occurrence. |
| 3 | I3 bound an unnamed commit flag to a preview | The flag is never commit evidence. It is reported as `scheduler_commit_flag`, and the preview stays `unconfirmed`. |
| 4 | Selection-set conflict resolved per slot | I14: a decision and a pick file are compared as whole selection sets. Any difference makes every decision row `unresolved`, attaches no file facts, counts the file's delivery as other-selection evidence, and marks every file slot `unresolved_pick_file_view`. |
| 5 | I5 treated a retained change as missing | `known_incomplete` only when a lineup-evolution entry names a (slot, batter, game) that no retained full record names. |
| 6 | I2 invented a streak across a dropped round | The previous entered round is taken from all qualified lines. If it is absent from the selected line, the value is null. |
| 7 | Incomplete schedule could manufacture a unique game | `parse_schedule` quarantines date entries without games and games without `gamePk` or both abbreviations. A date with any quarantine is `incomplete` and blocks inference (`team_schedule_incomplete`). |
| 8 | Eligibility chronology | The latest unit status before lock decides. A refusal needs a valid time before lock, and the latest one counts. |
| 9 | Reconciliation semantics | Labels follow spec §8: the scorecard is always `hypothesis`, and the tally is `hypothesis` only if a rule fits both totals, else `unrecoverable`; no `partial`/`exact`. Fit is its own axis. The universe is every pick-object record under `picks/`, with an exclusion reason per rule. Rows carry the occurrence id, disposition, canonical selection and canonical outcome, `historical_membership = unknown`, the suggestive mtime and the freeze time. |
| 10 | Missing candidate; weak freeze | Added G4, which counts every slot object (ungraded included): T1–T24. The whole table and its predicates are fingerprinted, and a truth table pins the gradings. The equal-totals fixture now has different memberships. |
| 11 | Scalar types; partial outputs | Typed-field policy (I13). All tables are built before any write. A non-empty output directory is refused. |
| 12 | Gzip fixture flake | Builders pin `mtime=0` (asserted), and one fixture byte set is reused. |
| 13 | Mixed code identity on the box | The package imports nothing from `bts`: the decision helpers are vendored and tested for equivalence. `build.json` records Python, pyarrow, environment-lock and rules-fingerprint identity. |
| 14 | Task 13 gates | X-19 goes first. Code goes to a unique `/tmp/ledger_code_<sha>`, and every job logs `EXIT=`. Compile-twice comparison fails on any file-set or byte difference and checks the fingerprint. Backup runs under `pipefail` + `UV_CACHE_DIR` with a restic membership check. The retry path recompiles the sealed bundle into new directories. |
| 15 | I4 wording | I4 now states the precedence: a positive bound signal wins, and false/null apply only without one. |
| — | Incoherent fixture facts | Round 977 is now hit+void, streak 11, +1. |

## Revision 3 — how this answers Codex plan r2 (`docs/audit/2026-09-28-season-ledger-codex-plan-r2.md`, BLOCK)
*Kept as history; the Revision 4 table below records what replaced its census, membership and fingerprint items.*

| # | Finding | Change |
|---|---|---|
| 1 | The census accepted dropped records and did not enforce exactly-once coverage | `required_locators(kind, data)` derives every record from the raw bytes alone: files, contest lines, rounds and slots, items, dates and games, metadata records as well as leaves. An empty parse no longer exempts a file; `no_records` is allowed only where the census finds no record. `census_problems` requires each record exactly once, emitted directly or covered by exactly one quarantined ancestor. It reports foreign source paths, phantom locators, rows emitted off a record, exclusions over records, and double cover. `check_sources` runs right after accounting, before any ledger row exists. `check_invariants` runs after the outputs are built: each `*_obs_id` must name an emitted occurrence of the right kind, each canonical disposition must be referenced (a canonical selection exactly once), every recipe row must link to an occurrence, and contest-slot identities, selection links, row ids and season days are checked. Codex's four probes are acceptance tests. |
| 2 | Raw values lost for slotted rounds and static items | Every contest round is its own occurrence (`line=N/round=i`) with its own raw record, slotted or not. Static rows keep the raw values of every field they normalize. Every quarantined record that parsed keeps its parsed value; only undecodable bytes and path-level refusals are left to the sealed bundle. Tests read wrong-typed values back from the compiled occurrences table. |
| 3 | I13 applied inconsistently; an invalid identity could be committed | Integers are bounded to int64 (`is_int`). A single/double decision whose chosen candidate lacks an integer `batter_id` is quarantined (`decision_candidate_missing_batter_id`), and enum fields are type-checked before membership tests. Lineup-evolution slots need an integer batter. G1/G2 count only string labels, so a wrong-typed `result` is never counted and never crashes a recipe. A wrong-typed contest grade reads `unknown` (`slot_result_state = type_mismatch`, raw grade kept), distinct from a source null (`matched_ungraded`). |
| 4 | Quarantine turned a conflicting double-down file into a matching single | `pick_file_state` records the slots the raw file holds and whether every one parsed. Only a complete file naming the decision's exact set agrees (I14); a partial file leaves every decision row `unresolved`. With no decision, a file with no usable slot is `pick_file_unparseable`, never an absent file. |
| 5 | Reconciliation IDs did not resolve for part of the universe | Recipe rows name their record with `recipe_slot_key` and link through `source_obs_id` to the emitted occurrence that accounts for it: the slot, or the excluded or quarantined file that holds it. The link carries its state, reason and disposition. Canonical selection and outcome attach only through a production canonical row. `check_invariants` checks every link after the join. |
| 6 | The fingerprint omitted the predicates | `rules_fingerprint()` hashes the rule data plus the exact source of every recipe function and of the JSON decoder they read through (`inspect.getsource`); the new value is `eadce48f…`. A comment-only edit to the decoder now fails the fingerprint test. Codex's G1-`suspended` mutant now changes the fingerprint and fails a truth table that covers every label shape (`suspended`, non-strings, absent slots). |
| 7 | The runner exited 0 after a failure; monitoring could miss a fast exit | One runner with an EXIT trap prints `RUN=<id> START` … `RUN=<id> EXIT=<status>` and exits with that status. The wait polls the log for that run's `EXIT` line (bounded), then reads the whole run back, so an early failure cannot be missed. |
| 8 | Refusal rule differed between text and code | `_eligibility` returns `unknown` without a known lock, and a refusal needs a non-empty reason and a valid time before lock. I6 now states the binding: a refusal archive binds to the (date, slot, batter, game) it names, never to an attempt. |
| 9 | No verified-publication boundary | The compiler reserves its output directory with `mkdir`; an existing directory is refused, even an empty one. Each attempt writes `<sha>-<run-id>`, and `ACCEPTED.json` is written only after the file-set, byte and fingerprint checks pass. |
| 10 | The restic check was not bound to this backup | Backup mode records its start time and requires exactly one archive snapshot since then. It compares that snapshot's manifest with the sealed one by sha256, and requires every manifest member to be present in it (`restic ls <id> --recursive --json`). |

## Revision 4 — how this answers Codex plan r3 (`docs/audit/2026-09-28-season-ledger-codex-plan-r3.md`, BLOCK; the last planned plan round)
| # | Finding | Change |
|---|---|---|
| 1 | File-record omissions and forged occurrence ids passed both checks | `required_locators` returns the true record set: `{"file"}` for a file that is itself one record (a decision, a scheduler state, a pick file without slots) or whose container cannot be read, and the empty set only for a readable empty container. `census_problems` checks every bundle path: a declared-missing input, an excluded path and an empty container each need exactly one `file` row in that state with its reason, and records may only be emitted or quarantined. `check_sources` recomputes every occurrence's id from (path, locator, content hash), and checks its hash and its kind. Codex's four probes are acceptance tests. |
| 2 | Malformed game identities collapsed into equal nulls | A present `game_pk` that is not an integer quarantines the pick slot, the decision (for a chosen candidate) and the lineup-evolution slot. A null stays a supported unrecorded game. Codex's end-to-end false agreement now compiles to `decision_unusable`. |
| 3 | The membership check proved existence only | `check_invariants` recomputes membership independently of the join. Every rule must list every universe slot exactly once (`membership_slots`), its included rows must add up to the reported totals, and every row must link to the occurrence that accounts for its record, with the same state, reason, disposition and canonical link. |
| 4 | Regex flags escaped the freeze | `recipe_closure()` walks everything reachable from `evaluate_rules` and `recipe_labels`: repo-local function sources (imports included), compiled regexes with their flags, the time zone, and tables. Modules and outside callables are bound by name, and any other kind fails closed. `rules_fingerprint()` hashes the rule table plus the closure: `5e9d74f2…`. A test pins the closure's names; Codex's IGNORECASE mutant, a time-zone change and a decoder comment each change the digest. |
| 5 | Quarantined evidence was lost in the other day branches | New I15. `day_rows` takes the date's `unusable` kinds, and `pick_file_state` keeps the file-level delivery fields. An unusable decision gives `decision_unusable`, with the pick file kept as an unresolved view. A skip is established only beside a readable, undelivered pick file (or none) and a usable scheduler state. Quarantined evidence alone is `unusable_evidence_only`, never `unobserved_day`, and `known_incomplete` is withheld while a record that might hold the version is unusable. |
| 6 | A parsed JSON null lost its raw value | `quarantine` takes a sentinel default: a parsed null is kept as `"null"`, and only bytes that never decoded carry no raw value. |
| 7 | Runner edge cases | The EXIT trap is installed before the run-id check. `.env` is sourced as production's cron does it (no nounset), with its status checked. Run ids carry a random suffix. The read-back uses non-interactive ssh with connect and liveness timeouts and a two-hour deadline, and `NO EXIT LINE` is checked against the unit's status before it is called a stop. |
| 8 | `ACCEPTED.json` could be written partially | The receipt is written to a temporary file, synced, then renamed. Only a parseable receipt naming the run and its six files publishes a build. |

## Interpretations pinned by this plan (each is a reading of the spec; Codex reviews them)
- **I1 Qualification:**
  - Identity fields (`roundId`, `unitId`, `playerId`) must be non-null integers.
  - `recorded_at` must parse with an offset.
  - `result` keys must be present; null means in progress, which reads `matched_ungraded`.
  - A line with any slot lacking identity is quarantined whole. This affects the 3 playerless lines; their rounds repeat in neighbouring lines, and the census covers the nested slots.
- **I2 `streak_before`:** the previous entered round is the greatest roundId below this one that any qualified line reports. Its streak counts only when that round is present in the same line; otherwise null.
- **I3 Commit flag:** `committed_pick_written` names no selection and is never commit evidence. It is reported as `scheduler_commit_flag`.
- **I4 Delivery:** a decision `delivered`, then a positive pick-side signal (delivered_at, a DM with an id, a public post with a URI), confirm delivery. Only without either does `private_locked` read `false` and `locked_unconfirmed` null. `delivery_evidence_conflict` marks private/lock status beside a positive pick-side signal.
- **I5 History:** `known_incomplete` only when a lineup-evolution entry names a (slot, batter, game) that no retained full record for the date names (pick file, scheduler archive, manual or repair version). That version's content is gone. A retained archive is an observation, not missing content.
- **I6 Eligibility:**
  - Evidence counts only when it is timed before a known lock (`locked_at`). With no known lock, eligibility is `unknown`.
  - `postponed_evidenced` requires an `evidenced` match whose latest BTS units capture stamped before lock shows `postponed`.
  - `refused_evidenced` requires a `refused_delivery` archive naming the selection's (date, slot, batter, game), with a non-empty reason and a valid time before lock (the latest counts). The archive does not say which delivery attempt it belongs to, so the claim is about that selection, never about an attempt.
  - Anything else is `unknown`. The MLB schedule is fetched after the season, so it never evidences eligibility.
- **I7 Evidenced game, no same-game selection:** the slot becomes `contest_only` (`unit_capture_other_game`), and the batter's local selection that day reads `match_ambiguous`.
- **I8 One selection, two contest slots:** both are demoted to `ambiguous` (`multiple_contest_slots_for_selection`), and no grade transfers.
- **I9 Recipes:**
  - The 9/11 scorecard rerun is `hypothesis` whatever its fit.
  - The 9/14 tally is `hypothesis` if a pre-declared rule fits both totals, else `unrecoverable`.
  - Fit (`both`/`primaries_only`/`legs_only`/`none`) is reported separately.
  - `exact`/`partial` need independent per-record evidence, so they are never emitted.
  - The mtime comparison is suggestive only.
- **I10 Players lookup:** built from the 9/27 grab plus the first and last static players capture. An unmapped player reads `player_unknown`, and the memo reports how many.
- **I11 Graded:** `bts_outcome_status` comes from the typed slot result:
  - `graded` for a string grade.
  - `matched_ungraded` for a source null (in progress).
  - `unknown` for a present value of the wrong type. Its raw value is kept in `contest_slot_grade_raw`, and it is never presented as a source null.
  - Unknown string labels normalize to `UNKNOWN` and are never compared.
- **I12 Outputs:**
  - `…_occurrences.parquet` persists every occurrence: locator, obs_id, state, reason, disposition, file hash, `fields_json` and `record_raw_json`. The raw record is the record itself, except that a static item keeps the raw values of the fields it normalizes; the rest of the item stays in the sealed bundle.
  - `…_reconciliation.parquet` is recipe membership over the full universe. Each row names its record (`recipe_slot_key`) and links to the emitted occurrence that accounts for it (`source_obs_id`, with its state, reason and disposition).
  - The build fails unless the source census passes before any row exists and the output checks pass after the outputs are built.
- **I13 Typed-field policy:**
  - A field whose raw value has the wrong type becomes null in the typed column and is listed in `type_mismatch_fields`; the raw value survives in `record_raw_json`. Integers must fit int64, and a bool is never an integer.
  - Identity fields that cannot be typed quarantine the record: a pick slot's or lineup-evolution slot's batter and game, a single/double decision's chosen batters and games, a contest line's round, unit and player ids, and static item and game ids. A null `game_pk` is a supported unrecorded game (the 3/29–3/30 files); a present value that is not an integer never reads as null. Enum fields are type-checked before any membership test.
  - A quarantined record that parsed keeps its parsed value as `record_raw_json` (a parsed null as `"null"`). Only undecodable bytes and path-level refusals are left to the sealed bundle.
  - Absent keys are listed in `absent_fields` at every depth (`pick.pitcher_id`).
- **I14 Selection sets:** a decision and the surviving pick file agree only when the file is complete and names the identical set of (slot, batter, game). Only then are the file's facts attached.
  - Complete means every slot the file's raw structure holds parsed into a usable row (`pick_file_state`). A quarantined leg therefore blocks agreement, and every decision row stays `unresolved`.
  - With no decision, a file with no usable slot is `pick_file_unparseable`, never an absent file.
- **I15 Unusable evidence** (Codex plan r3 #5): day evidence that is present but quarantined is never read as absent.
  - An unusable decision makes the day `unfinalized_day` / `decision_unusable`; a surviving pick file stays an unresolved view.
  - A skip decision is a `skip_day` only if no pick file survives, or it is readable and shows no positive delivery signal, and the scheduler state is usable (or absent). A delivered file makes it `skip_decision_with_commit_evidence`, an unreadable one `skip_decision_with_unreadable_pick_file`, and an unusable state `skip_decision_with_unusable_state`.
  - A day whose only evidence is quarantined is `unfinalized_day` / `unusable_evidence_only`.
  - `known_incomplete` is withheld (history `unknown`) while a record that might hold the version is unusable: a quarantined archive for the date, or an incomplete pick file.

## Review Focus
1. AppleDouble `._*` files, runtime markers and `.before`/`.postponed` versions in the picks tree must be routed or excluded with a reason. They are never parsed as production picks, and no stray file crashes the build. *(Task 9 routing tests)*
2. The 3/29–3/30 pick files have null `game_pk`/`game_time`. Their `selection_id` carries `None`, `game_time` is null, and matching reads ambiguous `selection_game_pk_unrecorded`; nothing crashes. *(Task 2, Task 6 tests)*
3. Contest lines recorded mid-round, with null round `result`/`streak` or a null slot `result`, qualify and read `matched_ungraded`. A playerless slot quarantines its line; a wrong-typed stat is nulled and flagged. *(Task 4 tests)*
4. The end-of-season units capture holds an empty `units` list, and plain `.json` sits beside `.json.gz`. Both parse without rows or false conflicts, and a corrupt gzip is quarantined. *(Task 5 tests)*
5. Mixed time formats (`Z`, `+00:00`, ET `-04:00`, with and without microseconds) must end in one fixed-precision UTC form so that string order is time order. A naive time becomes null and is never read in the host zone. *(Task 1 test)*

---

## File Structure
| File | Responsibility |
|---|---|
| `scripts/audit/season_ledger/__init__.py` | package marker, `BUILDER_VERSION` |
| `scripts/audit/season_ledger/ids.py` | sha256, occurrence ids, `Parsed`, typed-field policy, JSON/gzip loading, UTC normalization |
| `scripts/audit/season_ledger/bundle.py` | manifest writing (acquire) and verification (compile) |
| `scripts/audit/season_ledger/io.py` | build-then-write deterministic parquet |
| `scripts/audit/season_ledger/sources/__init__.py` | subpackage marker |
| `scripts/audit/season_ledger/sources/pick_files.py` | O1 pick files, O2 scheduler archives, manual and repair versions |
| `scripts/audit/season_ledger/sources/day_records.py` | O3 decisions (vendored helpers), O4 scheduler state, O5 lineup evolution |
| `scripts/audit/season_ledger/sources/contest_ledger.py` | O6 contest ledger (lossless), saver-transition attempts |
| `scripts/audit/season_ledger/sources/static.py` | O7 rounds / players / units captures, validated MLB schedules |
| `scripts/audit/season_ledger/contest.py` | slot history, streak chain, lookups, game matching |
| `scripts/audit/season_ledger/outcomes.py` | outcome normalization, slot comparison, single-pick derivation |
| `scripts/audit/season_ledger/rows.py` | day status, row kinds, commit / history status, delivery predicate |
| `scripts/audit/season_ledger/reconcile.py` | routing, accounting, census, invariants, recipe rules |
| `scripts/audit/season_ledger/compile.py` | offline pipeline and output schemas |
| `scripts/audit/season_ledger/acquire.py` | acquisition into a sealed bundle |
| `scripts/audit/build_season_ledger.py` | CLI (`acquire`, `compile`) |
| `tests/scripts/season_ledger/__init__.py`, `builders.py` | synthetic source builders (production file shapes) |
| `tests/scripts/season_ledger/test_*.py` | one test file per module |

---

### Task 1: Package, ids, typed-field policy, bundle, deterministic writer

**Files:**
- Create: `scripts/audit/season_ledger/__init__.py`, `ids.py`, `bundle.py`, `io.py`, `sources/__init__.py`
- Create: `tests/scripts/season_ledger/__init__.py` (empty), `tests/scripts/season_ledger/builders.py`, `tests/scripts/season_ledger/test_bundle_io.py`

**Interfaces:**
- **Produces (`ids.py`):**
  - `ids.UTC_FORMAT`
  - `ids.sha256_hex(data) -> str`
  - `ids.obs_id(rel_path, locator, content_sha256) -> str`
  - `ids.Parsed(rows, quarantined)`
  - `ids.quarantine(rel_path, locator, reason, raw=<omitted>) -> dict`, whose `record_raw_json` holds the record's parsed value (a parsed null is `"null"`); `raw` is omitted only for bytes that never decoded
  - `ids.is_int(v) -> bool`: an int64-range integer, never a bool
  - `ids.typed(value, kind) -> (value | None, mismatch: bool)`, where kind ∈ int/float/bool/str
  - `ids.take(record, {name: kind}, prefix="") -> (values, absent_names, mismatched_names)`
  - `ids.joined(names) -> str | None`
  - `ids.canonical_json(obj) -> str`
  - `ids.load_json_bytes(data)`, which raises `ValueError("bad_gzip:…"|"invalid_json:…")`
  - `ids.utc_iso(raw) -> str | None`
  - `ids.stamp_to_utc(name) -> str | None`
- **Produces (`bundle.py`):**
  - `bundle.BundleEntry`
  - `bundle.write_manifest(root, entries, *, acquired_at_utc, builder_version, source_root=None) -> Path`
  - `bundle.open_bundle(root) -> (manifest, {rel_path: bytes | None})`, sorted by path
  - `bundle.BundleError`
- **Produces (`io.py`):**
  - `io.build_table(rows, schema, *, sort_keys, name) -> pa.Table`
  - `io.write_table(table, path) -> Path`
- **Produces (`builders.py`):**
  - `builders.dumps(obj) -> bytes`
  - `builders.gz(data) -> bytes`, gzip with mtime 0
  - `builders.seal_bundle(root, files, missing=(), mtimes=None)`

- [ ] **Step 1: Write the builders module and the failing tests**

`tests/scripts/season_ledger/builders.py`:
```python
"""Synthetic source builders for the season-ledger tests. They write the shapes production writes
(bts.picks.save_pick, bts.daily_decision.write_decision, scheduler save_state, the CLI contest-ledger
append); no test reads real data. Gzip output is pinned (mtime=0) so fixtures are byte-stable."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit.season_ledger.bundle import BundleEntry, write_manifest
from scripts.audit.season_ledger.ids import sha256_hex


def dumps(obj) -> bytes:
    return json.dumps(obj).encode()


def gz(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def seal_bundle(root, files: dict[str, bytes], missing=(), mtimes: dict[str, str] | None = None) -> None:
    root = Path(root)
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_mtime_utc=(mtimes or {}).get(rel)))
    entries += [BundleEntry(rel_path=rel, status="missing", note="not found at acquisition") for rel in missing]
    write_manifest(root, entries, acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="test")
```

`tests/scripts/season_ledger/test_bundle_io.py`:
```python
import json

import pyarrow as pa
import pytest

from scripts.audit.season_ledger.bundle import BundleEntry, BundleError, open_bundle, write_manifest
from scripts.audit.season_ledger.ids import (load_json_bytes, obs_id, quarantine, sha256_hex, stamp_to_utc, take, typed,
                                             utc_iso)
from scripts.audit.season_ledger.io import build_table, write_table
from tests.scripts.season_ledger.builders import gz, seal_bundle


def test_declared_missing_input_is_accepted(tmp_path):
    seal_bundle(tmp_path, {"picks/2026-05-01.json": b"{}"}, missing=["schedules/2026-05-01.json"])
    _, files = open_bundle(tmp_path)
    assert files == {"picks/2026-05-01.json": b"{}", "schedules/2026-05-01.json": None}


def test_declared_present_but_absent_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").unlink()
    with pytest.raises(BundleError, match="declared present but absent"):
        open_bundle(tmp_path)


def test_changed_file_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").write_bytes(b'{"x": 1}')
    with pytest.raises(BundleError, match="hash mismatch"):
        open_bundle(tmp_path)


def test_unsafe_path_is_refused(tmp_path):
    with pytest.raises(BundleError, match="unsafe"):
        write_manifest(tmp_path, [BundleEntry(rel_path="../x.json", status="missing")],
                       acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="t")


def test_manifest_order_does_not_change_what_compile_reads(tmp_path):
    seal_bundle(tmp_path, {"b.json": b"2", "a.json": b"1"})
    manifest = tmp_path / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()
    manifest.write_text(json.dumps(doc))
    _, files = open_bundle(tmp_path)
    assert list(files) == ["a.json", "b.json"]


def test_duplicate_bytes_at_two_paths_are_two_occurrences():
    digest = sha256_hex(b"same bytes")
    assert obs_id("picks/a.json", "file", digest) != obs_id("picks/archive/a.json", "file", digest)


def test_gzip_and_plain_json_both_load():
    assert load_json_bytes(b'{"a": 1}') == {"a": 1}
    assert load_json_bytes(gz(b'{"a": 1}')) == {"a": 1}
    assert gz(b"x")[4:8] == b"\x00\x00\x00\x00"     # pinned gzip mtime: fixtures are byte-stable (Codex plan r1 #12)
    with pytest.raises(ValueError, match="bad_gzip"):
        load_json_bytes(b"\x1f\x8b" + b"not really gzip")
    with pytest.raises(ValueError, match="invalid_json"):
        load_json_bytes(b'{"a": ')


def test_times_normalize_to_fixed_precision_utc():
    # Review Focus 5: every source format in the snapshot, plus a naive value.
    assert utc_iso("2026-08-20T17:00:00.123456-04:00") == "2026-08-20T21:00:00.123456Z"
    assert utc_iso("2026-05-01T23:05:00Z") == "2026-05-01T23:05:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00+00:00") == "2026-05-01T15:00:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00") is None
    assert utc_iso(None) is None and utc_iso("garbage") is None
    assert utc_iso("2026-05-01T15:00:00.5Z") < utc_iso("2026-05-01T15:00:01Z")   # string order == time order


def test_capture_stamp_to_utc():
    assert stamp_to_utc("20260704T030011Z.json.gz") == "2026-07-04T03:00:11.000000Z"
    assert stamp_to_utc("001_rounds.json.gz") is None


def test_typed_policy_keeps_null_absent_and_wrong_type_distinct():
    values, absent, bad = take({"hits": "1", "atBats": None, "p": 1},
                               {"hits": "int", "atBats": "int", "p": "float", "n": "int"}, prefix="s.")
    assert values == {"hits": None, "atBats": None, "p": 1.0, "n": None}
    assert (absent, bad) == (["s.n"], ["s.hits"])
    assert typed(True, "int") == (None, True) and typed(3, "int") == (3, False) and typed(None, "str") == (None, False)


SCHEMA = pa.schema([("obs_id", pa.string()), ("n", pa.int64()), ("flag", pa.bool_())])


def test_quarantine_keeps_a_parsed_null_distinct_from_no_value():
    # Codex plan r3 #6: a record that parsed as JSON null keeps "null"; only undecodable bytes carry no raw value.
    assert quarantine("picks/x.json", "file", "no_pick_object", raw=None)["record_raw_json"] == "null"
    assert quarantine("picks/x.json", "file", "invalid_json:JSONDecodeError")["record_raw_json"] is None


def test_write_table_is_byte_identical_for_reordered_input(tmp_path):
    rows = [{"obs_id": "b", "n": 2, "flag": None}, {"obs_id": "a", "n": None, "flag": True}]
    write_table(build_table(rows, SCHEMA, sort_keys=["obs_id"], name="t"), tmp_path / "x.parquet")
    write_table(build_table(list(reversed(rows)), SCHEMA, sort_keys=["obs_id"], name="t"), tmp_path / "y.parquet")
    assert (tmp_path / "x.parquet").read_bytes() == (tmp_path / "y.parquet").read_bytes()


def test_build_table_refuses_duplicate_sort_keys_and_unknown_columns():
    with pytest.raises(ValueError, match="duplicate sort keys"):
        build_table([{"obs_id": "a"}, {"obs_id": "a"}], SCHEMA, sort_keys=["obs_id"], name="t")
    with pytest.raises(ValueError, match="not in schema"):
        build_table([{"obs_id": "a", "typo": 1}], SCHEMA, sort_keys=["obs_id"], name="t")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_bundle_io.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'scripts.audit.season_ledger'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/__init__.py`:
```python
"""Season 2026 canonical ledger, Phase 1 (spec docs/superpowers/specs/2026-09-28-season-ledger-design.md)."""
BUILDER_VERSION = "season-ledger-phase1/3"
```

`scripts/audit/season_ledger/sources/__init__.py`:
```python
"""Per-source parsers: bytes in; typed, lossless rows out."""
```

`scripts/audit/season_ledger/ids.py`:
```python
"""Occurrence identity, the typed-field policy and small shared helpers."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_STAMP = re.compile(r"(\d{8}T\d{6})Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def obs_id(rel_path: str, locator: str, content_sha256: str) -> str:
    """Identity of one source occurrence: path + locator + content hash, so byte-identical files at
    two paths remain two occurrences (spec §4)."""
    return hashlib.sha256(f"{rel_path}\x00{locator}\x00{content_sha256}".encode()).hexdigest()[:24]


@dataclass
class Parsed:
    rows: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)


_NO_RAW = object()     # `raw` omitted: the record never decoded (a parsed JSON null is passed as None)


def quarantine(rel_path: str, locator: str, reason: str, raw=_NO_RAW) -> dict:
    """A quarantined occurrence. Every quarantined record that parsed passes its own parsed value as `raw`
    (kept as JSON in the occurrence table; a parsed null is "null"). Only undecodable bytes (bad gzip, invalid
    JSON, not UTF-8) and path-level refusals omit it; they stay in the sealed bundle, pinned by the manifest."""
    return {"source_path": rel_path, "locator": locator, "reason": reason,
            "record_raw_json": None if raw is _NO_RAW else canonical_json(raw)}


def is_int(value) -> bool:
    """An integer that fits the int64 output columns (bools are not integers here)."""
    return isinstance(value, int) and not isinstance(value, bool) and -(2 ** 63) <= value < 2 ** 63


_KINDS = {"int": is_int, "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "bool": lambda v: isinstance(v, bool), "str": lambda v: isinstance(v, str)}


def typed(value, kind: str) -> tuple[object, bool]:
    """Typed-field policy (Interpretation I13): (value, False) when it fits `kind` (a float field accepts
    an int and returns a float); (None, False) for None; (None, True) for a present value of the wrong
    type. The raw value always survives in the record's raw JSON."""
    if value is None:
        return None, False
    if not _KINDS[kind](value):
        return None, True
    return (float(value) if kind == "float" else value), False


def take(record: dict, fields: dict[str, str], prefix: str = "") -> tuple[dict, list[str], list[str]]:
    """Typed values for `fields` ({name: kind}), plus the names that were absent and the names whose value
    had the wrong type (both qualified with `prefix`, e.g. 'pick.')."""
    values, absent, mismatched = {}, [], []
    for name, kind in fields.items():
        if name not in record:
            absent.append(prefix + name)
        value, bad = typed(record.get(name), kind)
        if bad:
            mismatched.append(prefix + name)
        values[name] = value
    return values, absent, mismatched


def joined(names) -> str | None:
    return ",".join(sorted(names)) or None


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json_bytes(data: bytes):
    """Parse JSON from plain or gzip bytes; raise ValueError('bad_gzip:…' / 'invalid_json:…')."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"bad_gzip:{exc.__class__.__name__}") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_json:{exc.__class__.__name__}") from exc


def utc_iso(raw) -> str | None:
    """ISO-8601 with an offset → fixed-precision UTC ('YYYY-MM-DDTHH:MM:SS.ffffffZ'), so string order is
    time order. None, non-strings, unparseable and naive values → None: a naive time is never read in
    the host's zone."""
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if t.tzinfo is None or t.utcoffset() is None:
        return None
    return t.astimezone(timezone.utc).strftime(UTC_FORMAT)


def stamp_to_utc(name: str) -> str | None:
    """Static-capture file names carry a UTC stamp 'YYYYmmddTHHMMSSZ'."""
    m = _STAMP.search(name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).strftime(UTC_FORMAT)
```

`scripts/audit/season_ledger/bundle.py`:
```python
"""Sealed evidence bundle: manifest writing (acquire) and verification (compile). Spec §3."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .ids import sha256_hex

MANIFEST_NAME = "manifest.json"
BUNDLE_SCHEMA = "bts_season_ledger_bundle_v1"


class BundleError(Exception):
    pass


@dataclass(frozen=True)
class BundleEntry:
    rel_path: str
    status: str                       # "present" | "missing"
    sha256: str | None = None
    size: int | None = None
    source_path: str | None = None    # relative to the acquisition source root, or the fetch URL
    source_mtime_utc: str | None = None
    note: str | None = None


def check_rel_path(rel: str) -> str:
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or p.as_posix() != rel:
        raise BundleError(f"unsafe bundle path: {rel!r}")
    return rel


def write_manifest(root: Path, entries: list[BundleEntry], *, acquired_at_utc: str, builder_version: str,
                   source_root: str | None = None) -> Path:
    for e in entries:
        check_rel_path(e.rel_path)
        if e.status not in ("present", "missing"):
            raise BundleError(f"bad status {e.status!r} for {e.rel_path}")
    body = {"schema": BUNDLE_SCHEMA, "acquired_at_utc": acquired_at_utc, "builder_version": builder_version,
            "source_root": source_root,
            "entries": sorted((asdict(e) for e in entries), key=lambda d: d["rel_path"])}
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return path


def open_bundle(root: Path) -> tuple[dict, dict[str, bytes | None]]:
    """Verify every declared entry; return (manifest, {rel_path: bytes | None}) sorted by path. A declared
    `missing` entry is valid evidence of absence; a declared-present file that is absent or changed is
    refused. Files the manifest does not declare are ignored."""
    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"unreadable manifest: {exc}") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unexpected bundle schema {manifest.get('schema')!r}")
    files: dict[str, bytes | None] = {}
    for e in sorted(manifest["entries"], key=lambda e: e["rel_path"]):
        rel = check_rel_path(e["rel_path"])
        if rel in files:
            raise BundleError(f"duplicate manifest entry {rel}")
        if e["status"] == "missing":
            files[rel] = None
            continue
        path = root / rel
        if not path.is_file():
            raise BundleError(f"declared present but absent: {rel}")
        data = path.read_bytes()
        if sha256_hex(data) != e["sha256"]:
            raise BundleError(f"hash mismatch: {rel}")
        files[rel] = data
    return manifest, files
```

`scripts/audit/season_ledger/io.py`:
```python
"""Deterministic parquet output (spec §3: identical inputs → identical bytes). Tables are built — and so
type-checked — before anything is written."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _sort_key(row: dict, keys: list[str]) -> tuple:
    return tuple("" if row.get(k) is None else str(row.get(k)) for k in keys)


def build_table(rows: list[dict], schema: pa.Schema, *, sort_keys: list[str], name: str) -> pa.Table:
    names = set(schema.names)
    for row in rows:
        extra = set(row) - names
        if extra:
            raise ValueError(f"columns not in schema for {name}: {sorted(extra)}")
    ordered = sorted(rows, key=lambda r: _sort_key(r, sort_keys))
    keys = [_sort_key(r, sort_keys) for r in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate sort keys in {name}; sort on a unique column")
    return pa.Table.from_pydict({f.name: [r.get(f.name) for r in ordered] for f in schema}, schema=schema)


def write_table(table: pa.Table, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=False, write_statistics=False)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_bundle_io.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger tests/scripts/season_ledger
git commit -m "feat(ledger): sealed bundle, occurrence ids, typed-field policy, UTC normalization, deterministic parquet (W1.1 task 1)"
```

---

### Task 2: O1 pick files, O2 archives, manual and repair versions

**Files:**
- Create: `scripts/audit/season_ledger/sources/pick_files.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `pick_json`)
- Test: `tests/scripts/season_ledger/test_pick_files.py`

**Interfaces:**
- **Consumes:** Task 1 `ids.*`.
- **Produces:**
  - `parse_pick_file(rel_path, data, *, kind="pick_file") -> Parsed`, where kind ∈ {pick_file, manual_archive, repair_archive}
  - `parse_archive(rel_path, data) -> Parsed`
  - `pick_file_state(data, parsed) -> {"slots": frozenset, "complete": bool, "file_fields": dict | None}`: the slots the raw file holds, whether every one of them parsed (I14), and its typed file-level fields, delivery signals included (None when the file is not a readable object; I15)
- **Every row has:**
  - Identity: `obs_id, locator ("slot=primary"|"slot=double_down"), source_kind, source_path, content_sha256, slot, day_result_raw, slot_result_raw, has_double_down`.
  - Typed pick fields: `batter_id, batter_name, team, game_pk, game_time, lineup_position, projected_lineup, pitcher_id, pitcher_name, pitcher_team, p_game_hit`.
  - Typed file fields: `date, run_time, bluesky_posted, bluesky_uri, notification_sent, notification_id, notification_channel, delivery_attempted, delivered_at, model_git_sha, model_pickle_sha256, policy_npz_sha256, feature_env_schema_version, feature_env_hash, tail_policy_sha256, shadow_model_version`.
  - Nested JSON: `slot_results_json, policy_decision_json, feature_env_json, pick_policy_objective`.
  - Archive metadata: `archive_prefix, archive_reason, archived_at`, which are None except on scheduler archives.
  - Presence and raw: `absent_fields, type_mismatch_fields, record_raw_json`.
- **Quarantine (I13):** a file that is not JSON or has no `pick` object is quarantined at `file`. A slot that is not an object, whose `batter_id` is not an int64 integer, or whose present `game_pk` is not one (`slot_bad_game_pk`) is quarantined at its own locator; a null `game_pk` is an unrecorded game. Each keeps its parsed value.

- [ ] **Step 1: Append the builder** to `tests/scripts/season_ledger/builders.py`:

```python
_PICK = {"batter_name": "Ada Batter", "batter_id": 101, "team": "TB", "lineup_position": 1,
         "pitcher_name": "P", "pitcher_id": 900, "p_game_hit": 0.78, "flags": [],
         "projected_lineup": False, "game_pk": 5001, "game_time": "2026-05-01T23:05:00Z", "pitcher_team": "BOS"}
_DD = {**_PICK, "batter_name": "Dee Leg", "batter_id": 202, "team": "NYY", "game_pk": 5002, "lineup_position": 2}


def pick_json(date: str, *, primary: dict | None = None, dd: dict | None = None, **file_fields) -> bytes:
    """A pick file as save_pick writes it (asdict(DailyPick)) with only the given file-level fields;
    pass dd={} for the default double-down."""
    doc = {"date": date, "run_time": f"{date}T15:00:00Z", "pick": {**_PICK, **(primary or {})},
           "double_down": None if dd is None else {**_DD, **dd}, "runner_up": None}
    doc.update(file_fields)
    return dumps(doc)
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_pick_files.py`:

```python
import json

from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file, pick_file_state
from tests.scripts.season_ledger.builders import dumps, pick_json


def test_double_down_file_yields_two_slot_rows_with_raw_slot_results():
    data = pick_json("2026-08-20", dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                     notification_sent=True, notification_id="dm-1")
    parsed = parse_pick_file("picks/2026-08-20.json", data)
    assert parsed.quarantined == []
    rows = {r["slot"]: r for r in parsed.rows}
    assert (rows["primary"]["locator"], rows["double_down"]["locator"]) == ("slot=primary", "slot=double_down")
    assert rows["primary"]["slot_result_raw"] == "miss" and rows["double_down"]["slot_result_raw"] == "hit"
    assert rows["primary"]["day_result_raw"] == "miss" and rows["primary"]["has_double_down"] is True
    assert rows["double_down"]["batter_id"] == 202 and rows["double_down"]["game_pk"] == 5002
    assert rows["primary"]["obs_id"] != rows["double_down"]["obs_id"]


def test_early_file_records_absent_fields_at_every_depth_and_never_defaults_delivery():
    # Review Focus 2 and spec §4 losslessness: the 3/29–3/30 shape — null game, no later fields.
    data = dumps({"date": "2026-03-29", "run_time": "2026-03-29T15:00:00+00:00", "result": "hit",
                  "bluesky_posted": False, "bluesky_uri": None, "runner_up": None, "double_down": None,
                  "pick": {"batter_id": 101, "batter_name": "Ada Batter", "team": "TB", "game_pk": None,
                           "game_time": None, "lineup_position": 3}})
    (row,) = parse_pick_file("picks/2026-03-29.json", data).rows
    assert (row["game_pk"], row["game_time"], row["bluesky_posted"]) == (None, None, False)
    assert row["notification_sent"] is None and row["slot_result_raw"] is None
    absent = set(row["absent_fields"].split(","))
    assert {"notification_sent", "slot_results", "delivered_at", "model_git_sha", "pick.pitcher_id"} <= absent
    assert "bluesky_posted" not in absent and "pick.game_pk" not in absent     # present as null, not absent


def test_provenance_and_the_raw_record_survive():
    policy = {"objective": "emax_season_best", "best_streak": 18}
    data = pick_json("2026-09-05", policy_decision=policy, feature_env={"BTS_ROOKIE_GATE_K": "20"},
                     feature_env_schema_version="v1", runner_up={"batter_name": "R", "p_game_hit": 0.7})
    (row,) = parse_pick_file("picks/2026-09-05.json", data).rows
    assert row["pick_policy_objective"] == "emax_season_best" and json.loads(row["policy_decision_json"]) == policy
    assert json.loads(row["feature_env_json"]) == {"BTS_ROOKIE_GATE_K": "20"}
    assert row["feature_env_schema_version"] == "v1" and row["pitcher_name"] == "P"
    assert json.loads(row["record_raw_json"]) == json.loads(data)


def test_wrong_types_are_nulled_and_flagged_and_a_slot_needs_its_batter_id():
    data = pick_json("2026-05-01", primary={"lineup_position": "1"}, notification_sent="yes",
                     dd={"batter_id": "202"})
    parsed = parse_pick_file("picks/2026-05-01.json", data)
    (row,) = parsed.rows
    assert row["lineup_position"] is None and row["notification_sent"] is None
    assert set(row["type_mismatch_fields"].split(",")) == {"pick.lineup_position", "notification_sent"}
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])["batter_id"]) == (
        "slot=double_down", "slot_missing_batter_id", "202")


def test_a_malformed_game_pk_quarantines_the_slot_but_a_source_null_does_not():
    # Codex plan r3 #2: a present game identity that cannot be typed must never read as an unrecorded game.
    parsed = parse_pick_file("picks/2026-05-01.json", pick_json("2026-05-01", primary={"game_pk": "bad"}, dd={}))
    assert [r["slot"] for r in parsed.rows] == ["double_down"]
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])["game_pk"]) == (
        "slot=primary", "slot_bad_game_pk", "bad")
    early = parse_pick_file("picks/2026-03-29.json", pick_json("2026-03-29", primary={"game_pk": None, "game_time": None}))
    assert early.quarantined == [] and early.rows[0]["game_pk"] is None


def test_pick_file_state_keeps_file_level_delivery_facts_when_no_slot_parses():
    # Codex plan r3 #5: the file's own delivery fields stay readable when every slot is quarantined.
    data = pick_json("2026-05-01", primary={"batter_id": "bad"}, notification_sent=True, notification_id="dm-1")
    state = pick_file_state(data, parse_pick_file("picks/2026-05-01.json", data))
    assert (state["slots"], state["complete"]) == (frozenset({"primary"}), False)
    assert (state["file_fields"]["notification_sent"], state["file_fields"]["notification_id"]) == (True, "dm-1")
    unreadable = b'{"date": "2026-05-01", "pick": {'
    assert pick_file_state(unreadable, parse_pick_file("picks/2026-05-01.json", unreadable))["file_fields"] is None


def test_non_json_and_truncated_files_are_quarantined():
    for data in (b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X", b'{"date": "2026-05-01", "pick": {'):
        parsed = parse_pick_file("picks/2026-05-01.json", data)
        assert parsed.rows == [] and parsed.quarantined[0]["reason"].startswith("invalid_json")


def test_file_without_pick_object_is_quarantined_with_its_raw_document():
    parsed = parse_pick_file("picks/2026-05-01.json", dumps({"date": "2026-05-01"}))
    assert parsed.quarantined == [{"source_path": "picks/2026-05-01.json", "locator": "file", "reason": "no_pick_object",
                                   "record_raw_json": '{"date":"2026-05-01"}'}]
    assert parse_pick_file("picks/2026-05-01.json", b"null").quarantined[0]["record_raw_json"] == "null"


def test_archive_rows_carry_prefix_reason_and_time():
    data = pick_json("2026-08-30", deferred_fallback={"reason": "gap_blocked", "deferred_at": "2026-08-30T12:01:00-04:00"})
    (row,) = parse_archive("picks/2026-08-30/deferred_fallback_20260830T120100-0400.json", data).rows
    assert (row["source_kind"], row["archive_prefix"], row["archive_reason"], row["archived_at"]) == (
        "archive", "deferred_fallback", "gap_blocked", "2026-08-30T12:01:00-04:00")


def test_repair_and_manual_versions_parse_as_pick_versions():
    before = parse_pick_file("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json.before",
                             pick_json("2026-05-24", result="miss"), kind="repair_archive").rows[0]
    postponed = parse_pick_file("picks/archive/2026-04-11.json.postponed", pick_json("2026-04-11"),
                                kind="manual_archive").rows[0]
    assert (before["source_kind"], before["archive_prefix"], before["day_result_raw"]) == ("repair_archive", None, "miss")
    assert (postponed["source_kind"], postponed["batter_id"]) == ("manual_archive", 101)


def test_unrecognized_archive_name_is_quarantined():
    parsed = parse_archive("picks/2026-08-30/something_else.json", pick_json("2026-08-30"))
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "unrecognized_archive_name"
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_pick_files.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.sources.pick_files'`.

- [ ] **Step 4: Write the implementation** — `scripts/audit/season_ledger/sources/pick_files.py`:

```python
"""O1 pick files (`picks/<date>.json`), O2 scheduler archives (`picks/<date>/<prefix>_<stamp>.json`) and the
manual / streak-repair versions of pick files (spec §4). Typed values follow Interpretation I13, absent
keys are listed at every depth, and each row keeps the file's raw JSON, so nothing written is lost."""
from __future__ import annotations

import re

from ..ids import Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex, take, typed

SLOTS = (("primary", "pick"), ("double_down", "double_down"))   # ledger slot, JSON key (= slot_results key)
PICK_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "game_pk": "int", "game_time": "str",
               "lineup_position": "int", "projected_lineup": "bool", "pitcher_id": "int", "pitcher_name": "str",
               "pitcher_team": "str", "p_game_hit": "float"}
FILE_FIELDS = {"date": "str", "run_time": "str", "result": "str", "bluesky_posted": "bool", "bluesky_uri": "str",
               "notification_sent": "bool", "notification_id": "str", "notification_channel": "str",
               "delivery_attempted": "bool", "delivered_at": "str", "model_git_sha": "str",
               "model_pickle_sha256": "str", "policy_npz_sha256": "str", "feature_env_schema_version": "str",
               "feature_env_hash": "str", "tail_policy_sha256": "str", "shadow_model_version": "str"}
NESTED_FIELDS = ("slot_results", "policy_decision", "feature_env", "runner_up")     # kept as JSON
ARCHIVE_TIME_KEYS = {"deferred_fallback": "deferred_at", "refused_delivery": "refused_at", "stale_pick": "staled_at"}
_ARCHIVE_NAME = re.compile(r"^(deferred_fallback|refused_delivery|stale_pick)_\d{8}T\d{6}[+-]\d{4}\.json$")


def _json_or_none(value) -> str | None:
    return None if value is None else canonical_json(value)


def parse_pick_file(rel_path: str, data: bytes, *, kind: str = "pick_file") -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        out.quarantined.append(quarantine(rel_path, "file", str(exc)))
        return out
    if not isinstance(doc, dict) or not isinstance(doc.get("pick"), dict):
        out.quarantined.append(quarantine(rel_path, "file", "no_pick_object", raw=doc))
        return out
    values, file_absent, file_bad = take(doc, FILE_FIELDS)
    file_absent += [f for f in NESTED_FIELDS if f not in doc]
    slot_results, policy = doc.get("slot_results"), doc.get("policy_decision")
    day_result = values.pop("result")
    base = {"source_kind": kind, "source_path": rel_path, "content_sha256": content, **values,
            "day_result_raw": day_result, "has_double_down": doc.get("double_down") is not None,
            "slot_results_json": _json_or_none(slot_results), "policy_decision_json": _json_or_none(policy),
            "feature_env_json": _json_or_none(doc.get("feature_env")),
            "pick_policy_objective": typed(policy.get("objective"), "str")[0] if isinstance(policy, dict) else None,
            "archive_prefix": None, "archive_reason": None, "archived_at": None,
            "record_raw_json": canonical_json(doc)}
    for slot, key in SLOTS:
        pick, locator = doc.get(key), f"slot={slot}"
        if pick is None:
            continue
        if not isinstance(pick, dict):
            out.quarantined.append(quarantine(rel_path, locator, "slot_not_object", raw=pick))
            continue
        if not is_int(pick.get("batter_id")):
            out.quarantined.append(quarantine(rel_path, locator, "slot_missing_batter_id", raw=pick))
            continue
        if pick.get("game_pk") is not None and not is_int(pick["game_pk"]):
            # I13: a present game identity that cannot be typed is not an unrecorded (null) game.
            out.quarantined.append(quarantine(rel_path, locator, "slot_bad_game_pk", raw=pick))
            continue
        slot_values, absent, bad = take(pick, PICK_FIELDS, prefix=f"{key}.")
        slot_result, slot_bad = typed(slot_results.get(key), "str") if isinstance(slot_results, dict) else (None, False)
        out.rows.append({**base, **slot_values, "slot": slot, "locator": locator,
                         "obs_id": obs_id(rel_path, locator, content), "slot_result_raw": slot_result,
                         "absent_fields": joined(file_absent + absent),
                         "type_mismatch_fields": joined(file_bad + bad + ([f"slot_results.{key}"] if slot_bad else []))})
    return out


def pick_file_state(data: bytes, parsed: Parsed) -> dict:
    """What a surviving pick file claims, independent of how much of it parsed (Codex plan r2 #4, r3 #5): the
    slots its raw structure holds, whether every one of them parsed into a usable row, and its typed file-level
    fields (delivery signals included), None when the file is not a readable object. Only a complete file can
    agree with a decision (Interpretation I14)."""
    try:
        doc = load_json_bytes(data)
    except ValueError:
        doc = None
    slots = frozenset(slot for slot, key in SLOTS if isinstance(doc, dict) and doc.get(key) is not None)
    return {"slots": slots, "complete": not parsed.quarantined and {r["slot"] for r in parsed.rows} == set(slots),
            "file_fields": take(doc, FILE_FIELDS)[0] if isinstance(doc, dict) else None}


def parse_archive(rel_path: str, data: bytes) -> Parsed:
    match = _ARCHIVE_NAME.match(rel_path.rsplit("/", 1)[-1])
    if not match:
        return Parsed([], [quarantine(rel_path, "file", "unrecognized_archive_name")])
    parsed = parse_pick_file(rel_path, data, kind="archive")
    prefix = match.group(1)
    if parsed.rows:
        doc = load_json_bytes(data)
        info = doc.get(prefix) if isinstance(doc.get(prefix), dict) else {}
        for row in parsed.rows:
            row.update(archive_prefix=prefix, archive_reason=typed(info.get("reason"), "str")[0],
                       archived_at=typed(info.get(ARCHIVE_TIME_KEYS[prefix]), "str")[0])
    return parsed
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_pick_files.py -q`
Expected: `11 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/pick_files.py tests/scripts/season_ledger
git commit -m "feat(ledger): lossless pick-file, archive and repair-version parsers (task 2)"
```

---

### Task 3: O3 decisions, O4 scheduler state, O5 lineup evolution

**Files:**
- Create: `scripts/audit/season_ledger/sources/day_records.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `cand`, `decision_json`, `state_json`, `evolution_jsonl`)
- Test: `tests/scripts/season_ledger/test_day_records.py`

**Interfaces:**
- **Consumes:** Task 1.
- **Vendored helpers:** `ACCEPTED_SCHEMAS` and `decision_objective(rec)`, copied from `bts.daily_decision`. A test compares them with the production module across every schema × objective case.
- **Produces:**
  - `parse_decision(rel_path, data) -> Parsed`. One row, locator `file`:
    - typed `DECISION_FIELDS`
    - `objective_raw`, `objective`, `action_source_raw`, `action_source ∈ {mdp, heuristic, unknown}`
    - `{primary,double_down,second_candidate}_{batter_id,batter_name,team,game_pk,p_game_hit}`
    - `absent_fields`, `type_mismatch_fields`, `record_raw_json`
    - Container types are checked before any membership test. A record failing the vendored acceptance is quarantined (`invalid_decision_record`), and so is a single/double decision whose chosen candidate lacks an int64 `batter_id` (`decision_candidate_missing_batter_id`) or has a present `game_pk` that is not one (`decision_candidate_bad_game_pk`). Each keeps the raw record.
  - `parse_scheduler_state(rel_path, data) -> Parsed`. One row, locator `file`:
    - typed `STATE_FIELDS`
    - `final_skip_candidate_present`, `skip_candidate_batter_id`, `skip_candidate_game_pk`
    - `final_skip_candidate_json`, `delivery_refusals_json`, `fallback_refreshes_json`
    - `absent_fields`, `type_mismatch_fields`, `record_raw_json`
  - `parse_lineup_evolution(rel_path, data) -> Parsed`. One row per (line, slot), locator `line=N/slot=S`:
    - typed line and slot fields
    - `source_kind="lineup_evolution"`
    - `record_raw_json` = the line
- **Quarantine (lineup evolution):** invalid lines, non-object lines, lines without slots, non-object slots, slots without an int64 `batter_id` and slots whose present `game_pk` is not one are quarantined at their own locators, each with its raw value.

- [ ] **Step 1: Append builders** to `tests/scripts/season_ledger/builders.py`:

```python
def cand(batter_id: int, game_pk: int | None, *, team: str = "TB", name: str | None = None, p: float = 0.77) -> dict:
    return {"batter_id": batter_id, "batter_name": name or f"B{batter_id}", "team": team, "game_pk": game_pk,
            "p_game_hit": p}


def decision_json(date: str, *, action: str, primary: dict | None, double_down: dict | None = None,
                  schema: str = "bts_daily_decision_v3", **fields) -> bytes:
    """A decision record as bts.daily_decision.write_decision writes it (v1/v2 carry no objective)."""
    rec = {"schema_version": schema, "date": date, "action": action, "source": "mdp", "primary": primary,
           "double_down": double_down, "second_candidate": None, "streak": 0, "saver_available": None,
           "state_source": "contest", "state_status": "fresh", "allow_double": True, "contest_source_date": None,
           "delivery_status": "not_applicable" if action == "skip" else "delivered", "scoreable": action != "skip",
           "best_streak": None, "best_status": None, "effective_best": None, "tail_policy_sha256": None,
           "degraded_reason": None, "finalized_at": f"{date}T22:00:00.000000Z"}
    if schema == "bts_daily_decision_v3":
        rec["objective"] = "reach57"
    rec.update(fields)
    return dumps(rec)


def state_json(date: str, **fields) -> bytes:
    """A scheduler_state.json as scheduler.save_state writes asdict(SchedulerState)."""
    rec = {"date": date, "schedule_fetched_at": f"{date}T14:00:00-04:00", "games": [], "confirmed_game_pks": [],
           "runs_completed": [], "pick_locked": False, "pick_locked_at": None, "result_status": None,
           "next_wakeup": None, "final_skip_candidate": None, "committed_pick_written": False,
           "delivery_refusals": None, "fallback_refreshes": None}
    rec.update(fields)
    return dumps(rec)


def evolution_jsonl(date: str, entries: list[tuple[dict, dict | None]]) -> bytes:
    """entries: [(primary_slot, double_down_slot_or_None)] as append_lineup_evolution writes them."""
    lines = [json.dumps({"captured_at": f"{date}T1{i}:00:00+00:00", "date": date, "run_time": f"{date}T1{i}:00:00+00:00",
                         "primary": p, "double_down": d}) for i, (p, d) in enumerate(entries)]
    return ("\n".join(lines) + "\n").encode()
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_day_records.py`:

```python
import json

from scripts.audit.season_ledger.sources.day_records import (ACCEPTED_SCHEMAS, decision_objective, parse_decision,
                                                             parse_lineup_evolution, parse_scheduler_state)
from tests.scripts.season_ledger.builders import cand, decision_json, dumps, evolution_jsonl, state_json

EVO = "picks/lineup_evolution_2026-05-01.jsonl"


def test_v1_decision_reads_as_reach57_and_keeps_raw_objective():
    row = parse_decision("picks/2026-06-23/decision.json",
                         decision_json("2026-06-23", action="single", primary=cand(101, 5001),
                                       schema="bts_daily_decision_v1")).rows[0]
    assert (row["objective"], row["objective_raw"]) == ("reach57", None)
    assert (row["primary_batter_id"], row["primary_game_pk"], row["double_down_batter_id"]) == (101, 5001, None)


def test_invalid_v3_objective_is_unknown():
    row = parse_decision("picks/2026-09-05/decision.json",
                         decision_json("2026-09-05", action="double", primary=cand(101, 5001),
                                       double_down=cand(202, 5002), objective="bogus")).rows[0]
    assert row["objective"] == "unknown" and row["objective_raw"] == "bogus"


def test_vendored_decision_rules_match_production():
    from bts import daily_decision as prod
    assert ACCEPTED_SCHEMAS == prod.ACCEPTED_SCHEMAS
    for schema in (*prod.ACCEPTED_SCHEMAS, "bts_daily_decision_v9"):
        for objective in ("reach57", "emax_season_best", "bogus", None, "<absent>"):
            rec = {"schema_version": schema} if objective == "<absent>" else {"schema_version": schema, "objective": objective}
            assert decision_objective(rec) == prod.decision_objective(rec), (schema, objective)


def test_action_source_keeps_raw_and_normalizes_unknown():
    for raw in ("forced", "unknown"):
        row = parse_decision("picks/2026-08-10/decision.json",
                             decision_json("2026-08-10", action="single", primary=cand(101, 5001), source=raw)).rows[0]
        assert (row["action_source_raw"], row["action_source"]) == (raw, "unknown")


def test_invalid_decision_records_are_quarantined_not_crashed_on():
    for doc in ({"schema_version": "bts_daily_decision_v3", "scoreable": True},
                {"schema_version": "bts_daily_decision_v3", "action": {"x": 1}, "scoreable": True, "date": "d"}):
        parsed = parse_decision("picks/2026-08-10/decision.json", dumps(doc))
        assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "invalid_decision_record"
    # Codex plan r2 #3: a decision that names a selection needs a usable batter identity.
    bad = decision_json("2026-08-10", action="single", primary=dict(cand(101, 5001), batter_id="101"))
    parsed = parse_decision("picks/2026-08-10/decision.json", bad)
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "decision_candidate_missing_batter_id"
    assert json.loads(parsed.quarantined[0]["record_raw_json"])["primary"]["batter_id"] == "101"
    # Codex plan r3 #2: a present chosen game identity that cannot be typed quarantines too; a source null does not.
    bad_game = decision_json("2026-08-10", action="single", primary=cand(101, "bad"))
    assert parse_decision("picks/2026-08-10/decision.json", bad_game).quarantined[0]["reason"] == (
        "decision_candidate_bad_game_pk")
    unrecorded = decision_json("2026-08-10", action="single", primary=cand(101, None))
    assert parse_decision("picks/2026-08-10/decision.json", unrecorded).quarantined == []


def test_decision_keeps_its_raw_record_and_absent_candidate_fields():
    primary = {"batter_id": 101, "batter_name": "A", "team": "TB", "game_pk": 5001}     # no p_game_hit
    data = decision_json("2026-08-10", action="single", primary=primary)
    row = parse_decision("picks/2026-08-10/decision.json", data).rows[0]
    assert "primary.p_game_hit" in row["absent_fields"].split(",") and row["primary_p_game_hit"] is None
    assert json.loads(row["record_raw_json"]) == json.loads(data)


def test_scheduler_state_keeps_nested_records_losslessly():
    data = state_json("2026-09-19", final_skip_candidate={"primary": cand(303, 7001), "double": None, "streak": 4},
                      delivery_refusals=[{"at": "2026-09-19T18:00:00-04:00",
                                          "archive": "refused_delivery_20260919T180000-0400.json"}],
                      fallback_refreshes=[{"started": "x", "duration_sec": 3.5}])
    row = parse_scheduler_state("picks/2026-09-19/scheduler_state.json", data).rows[0]
    assert row["final_skip_candidate_present"] is True
    assert (row["skip_candidate_batter_id"], row["skip_candidate_game_pk"]) == (303, 7001)
    assert json.loads(row["final_skip_candidate_json"])["streak"] == 4
    assert json.loads(row["delivery_refusals_json"])[0]["archive"] == "refused_delivery_20260919T180000-0400.json"
    assert json.loads(row["fallback_refreshes_json"]) == [{"started": "x", "duration_sec": 3.5}]


def test_lineup_evolution_rows_per_slot_and_every_bad_record_quarantined():
    good = evolution_jsonl("2026-05-01", [({"batter_id": 101, "game_pk": 5001, "team": "TB"}, None),
                                          ({"batter_id": 111, "game_pk": 5011, "team": "TB"},
                                           {"batter_id": 202, "game_pk": 5002, "team": "NYY"})])
    extra = (b"{not json\n" + dumps({"date": "2026-05-01", "primary": "oops", "double_down": None}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": None}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": {"batter_id": None, "game_pk": 5001}}) + b"\n"
             + dumps({"date": "2026-05-01", "primary": {"batter_id": 101, "game_pk": "bad"}}) + b"\n")
    parsed = parse_lineup_evolution(EVO, good + extra)
    assert [(r["locator"], r["batter_id"], r["source_kind"]) for r in parsed.rows] == [
        ("line=1/slot=primary", 101, "lineup_evolution"), ("line=2/slot=primary", 111, "lineup_evolution"),
        ("line=2/slot=double_down", 202, "lineup_evolution")]
    assert [(q["locator"], q["reason"]) for q in parsed.quarantined] == [
        ("line=3", "invalid_json_line"), ("line=4/slot=primary", "slot_not_object"), ("line=5", "line_without_slots"),
        ("line=6/slot=primary", "slot_missing_batter_id"), ("line=7/slot=primary", "slot_bad_game_pk")]
    assert json.loads(parsed.quarantined[0]["record_raw_json"]) == "{not json"      # unparsed text kept as a string
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_day_records.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.sources.day_records'`.

- [ ] **Step 4: Write the implementation** — `scripts/audit/season_ledger/sources/day_records.py`:

```python
"""O3 decision files, O4 scheduler state, O5 lineup-evolution logs (spec §4)."""
from __future__ import annotations

import json

from ..ids import Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex, take, typed

# Vendored from bts.daily_decision so the box run executes only this package's code (Task 13);
# test_vendored_decision_rules_match_production pins them to the production module.
ACCEPTED_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2", "bts_daily_decision_v3")
_LEGACY_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2")
OBJECTIVES = ("reach57", "emax_season_best")

CANDIDATE_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "game_pk": "int", "p_game_hit": "float"}
DECISION_FIELDS = {"schema_version": "str", "date": "str", "action": "str", "source": "str", "streak": "int",
                   "saver_available": "bool", "state_source": "str", "state_status": "str", "allow_double": "bool",
                   "contest_source_date": "str", "delivery_status": "str", "scoreable": "bool", "objective": "str",
                   "best_streak": "int", "best_status": "str", "effective_best": "int", "tail_policy_sha256": "str",
                   "degraded_reason": "str", "finalized_at": "str"}
ACTION_SOURCES = ("mdp", "heuristic")
STATE_FIELDS = {"date": "str", "schedule_fetched_at": "str", "pick_locked": "bool", "pick_locked_at": "str",
                "committed_pick_written": "bool", "result_status": "str", "skip_notified_at": "str"}
STATE_NESTED = ("final_skip_candidate", "delivery_refusals", "fallback_refreshes")
EVOLUTION_LINE_FIELDS = {"captured_at": "str", "date": "str", "run_time": "str"}
EVOLUTION_FIELDS = {"batter_id": "int", "batter_name": "str", "team": "str", "p_game_hit": "float",
                    "projected_lineup": "bool", "game_pk": "int"}


def decision_objective(rec: dict) -> str:
    """As bts.daily_decision.decision_objective: pre-v3 records are reach57; a v3 record without a valid
    objective is unknown."""
    obj = rec.get("objective")
    if rec.get("schema_version") in _LEGACY_SCHEMAS:
        return obj if obj in OBJECTIVES else "reach57"
    return obj if obj in OBJECTIVES else "unknown"


def parse_decision(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    # Same acceptance as bts.daily_decision.load_decision, with container types checked first so an object
    # in any of these keys is quarantined instead of crashing a membership test.
    if (not isinstance(doc, dict) or not isinstance(doc.get("schema_version"), str)
            or doc["schema_version"] not in ACCEPTED_SCHEMAS or not isinstance(doc.get("action"), str)
            or doc["action"] not in {"skip", "single", "double"}
            or not isinstance(doc.get("scoreable"), bool) or "date" not in doc):
        return Parsed([], [quarantine(rel_path, "file", "invalid_decision_record", raw=doc)])
    named = ("primary", "double_down") if doc["action"] == "double" else ("primary",) if doc["action"] == "single" else ()
    for n in named:   # I13: a decision that names a selection must name a usable identity (a null game = unrecorded)
        chosen = doc.get(n)
        if not isinstance(chosen, dict) or not is_int(chosen.get("batter_id")):
            return Parsed([], [quarantine(rel_path, "file", "decision_candidate_missing_batter_id", raw=doc)])
        if chosen.get("game_pk") is not None and not is_int(chosen["game_pk"]):
            return Parsed([], [quarantine(rel_path, "file", "decision_candidate_bad_game_pk", raw=doc)])
    values, absent, bad = take(doc, DECISION_FIELDS)
    objective_raw, source_raw = values.pop("objective"), values.pop("source")
    row = {"obs_id": obs_id(rel_path, "file", content), "locator": "file", "source_kind": "decision",
           "source_path": rel_path, "content_sha256": content, **values, "objective_raw": objective_raw,
           "objective": decision_objective(doc), "action_source_raw": source_raw,
           "action_source": source_raw if source_raw in ACTION_SOURCES else "unknown",
           "record_raw_json": canonical_json(doc)}
    for name in ("primary", "double_down", "second_candidate"):
        cand = doc.get(name)
        if name not in doc:
            absent.append(name)
        if cand is not None and not isinstance(cand, dict):
            bad.append(name)
        cand_values, cand_absent, cand_bad = take(cand if isinstance(cand, dict) else {}, CANDIDATE_FIELDS,
                                                  prefix=f"{name}.")
        if isinstance(cand, dict):
            absent += cand_absent
        bad += cand_bad
        row.update({f"{name}_{k}": v for k, v in cand_values.items()})
    row["absent_fields"], row["type_mismatch_fields"] = joined(absent), joined(bad)
    return Parsed([row], [])


def parse_scheduler_state(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or "date" not in doc:
        return Parsed([], [quarantine(rel_path, "file", "invalid_scheduler_state", raw=doc)])
    values, absent, bad = take(doc, STATE_FIELDS)
    absent += [f for f in STATE_NESTED if f not in doc]
    skip = doc.get("final_skip_candidate")
    primary = skip.get("primary") if isinstance(skip, dict) else None
    row = {"obs_id": obs_id(rel_path, "file", content), "locator": "file", "source_kind": "scheduler_state",
           "source_path": rel_path, "content_sha256": content, **values,
           "final_skip_candidate_present": isinstance(skip, dict),
           "skip_candidate_batter_id": typed(primary.get("batter_id"), "int")[0] if isinstance(primary, dict) else None,
           "skip_candidate_game_pk": typed(primary.get("game_pk"), "int")[0] if isinstance(primary, dict) else None,
           **{f"{name}_json": None if doc.get(name) is None else canonical_json(doc[name]) for name in STATE_NESTED},
           "absent_fields": joined(absent), "type_mismatch_fields": joined(bad),
           "record_raw_json": canonical_json(doc)}
    return Parsed([row], [])


def parse_lineup_evolution(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        line_loc = f"line={line_no}"
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, line_loc, "invalid_json_line", raw=line))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, line_loc, "line_not_object", raw=doc))
            continue
        slots = [(s, doc[s]) for s in ("primary", "double_down") if doc.get(s) is not None]
        if not slots:
            out.quarantined.append(quarantine(rel_path, line_loc, "line_without_slots", raw=doc))
            continue
        line_values, line_absent, line_bad = take(doc, EVOLUTION_LINE_FIELDS)
        for slot, s in slots:
            locator = f"{line_loc}/slot={slot}"
            if not isinstance(s, dict) or not is_int(s.get("batter_id")):
                reason = "slot_not_object" if not isinstance(s, dict) else "slot_missing_batter_id"
                out.quarantined.append(quarantine(rel_path, locator, reason, raw=s))
                continue
            if s.get("game_pk") is not None and not is_int(s["game_pk"]):
                out.quarantined.append(quarantine(rel_path, locator, "slot_bad_game_pk", raw=s))
                continue
            values, absent, bad = take(s, EVOLUTION_FIELDS, prefix=f"{slot}.")
            out.rows.append({"obs_id": obs_id(rel_path, locator, content), "locator": locator,
                             "source_kind": "lineup_evolution", "source_path": rel_path, "content_sha256": content,
                             "line_no": line_no, **line_values, "slot": slot, **values,
                             "absent_fields": joined(line_absent + absent),
                             "type_mismatch_fields": joined(line_bad + bad), "record_raw_json": canonical_json(doc)})
    return out
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_day_records.py -q`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/day_records.py tests/scripts/season_ledger
git commit -m "feat(ledger): decision (vendored helpers), scheduler-state and lineup-evolution parsers (task 3)"
```

---

### Task 4: O6 contest ledger, slot history, streak chain, saver attempts

**Files:**
- Create: `scripts/audit/season_ledger/sources/contest_ledger.py`, `scripts/audit/season_ledger/contest.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `contest_line`, `rnd`, `slot`)
- Test: `tests/scripts/season_ledger/test_contest.py`

**Interfaces:**
- **Consumes:** Task 1.
- **Produces:**
  - `parse_contest_ledger(rel_path, data) -> Parsed`. Rows have `row_level ∈ {line, round, slot}` and locators `line=N`, `line=N/round=i` (every round, slotted or not) and `line=N/round=i/slot=j`:
    - All rows: `obs_id, source_path, content_sha256, line_no, recorded_at` (fixed UTC), `source_date, active_streak, best_streak`, plus `absent_fields`, `type_mismatch_fields` and `record_raw_json` for that level (a line without `predictions`, a round without `roundPredictions`, a slot).
    - Round and slot rows add `round_id, round_result, round_streak, round_streak_increase`.
    - Slot rows add `slot_number, unit_id, player_id, slot_result, slot_result_state, hits, hits_state, at_bats, at_bats_state`, with `*_state ∈ {value, null, absent, type_mismatch}`.
    - A line with a round lacking an integer `roundId` or a `result` key, or a slot lacking an integer `unitId`/`playerId` or a `result` key, is quarantined whole at `line=N` with its raw line (I1).
  - `parse_saver_transitions(rel_path, data) -> Parsed`. Rows: `attempted_at, attempt_source, attempt_outcome, record_raw_json`.
  - `contest.slot_history(contest_rows) -> list[dict]`
  - `contest.line_round_streaks(contest_rows) -> {line_no: {round_id: streak}}`
  - `contest.entered_rounds(contest_rows) -> list[int]`
  - `contest.streak_before(line_streaks, round_id, entered) -> int | None`

- [ ] **Step 1: Append builders** to `tests/scripts/season_ledger/builders.py`:

```python
def contest_line(recorded_at: str, rounds: list[dict], **fields) -> str:
    """One contest_ledger.jsonl line as the CLI appends it."""
    doc = {"recorded_at": recorded_at, "active_streak": 0, "best_streak": 18, "source_date": "2026-08-20",
           "predictions": rounds}
    doc.update(fields)
    return json.dumps(doc)


def rnd(round_id: int, result, streak, increase, slots: list[dict]) -> dict:
    return {"roundId": round_id, "result": result, "streak": streak, "streakIncrease": increase,
            "roundPredictions": slots}


def slot(unit_id: int, player_id, result, *, number: int = 1, hits=1, at_bats=4, drop: tuple = ()) -> dict:
    s = {"number": number, "unitId": unit_id, "playerId": player_id, "result": result, "hits": hits, "atBats": at_bats}
    for key in drop:
        s.pop(key)
    return s
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_contest.py`:

```python
import json

from scripts.audit.season_ledger.contest import entered_rounds, line_round_streaks, slot_history, streak_before
from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from tests.scripts.season_ledger.builders import contest_line, rnd, slot

PATH = "picks/account_state/contest_ledger.jsonl"


def _parse(*lines):
    return parse_contest_ledger(PATH, ("\n".join(lines) + "\n").encode())


def test_null_absent_and_wrong_type_stats_are_kept_distinct():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [
        slot(1928, 2513, "hit", hits=None), slot(1927, 1300, "hit", number=2, drop=("atBats",)),
        slot(1926, 1400, "hit", number=3, hits="1"), slot(1925, 1500, {"grade": "hit"}, number=4),
        slot(1924, 1600, "hit", number=5, hits=2 ** 80)])]))
    slots = {r["player_id"]: r for r in parsed.rows if r["row_level"] == "slot"}
    assert (slots[1500]["slot_result"], slots[1500]["slot_result_state"]) == (None, "type_mismatch")
    assert (slots[1600]["hits"], slots[1600]["hits_state"]) == (None, "type_mismatch")     # beyond int64
    assert (slots[2513]["hits"], slots[2513]["hits_state"]) == (None, "null")
    assert (slots[1300]["at_bats"], slots[1300]["at_bats_state"], slots[1300]["absent_fields"]) == (None, "absent", "atBats")
    assert (slots[1300]["hits"], slots[1300]["hits_state"]) == (1, "value")
    assert (slots[1400]["hits"], slots[1400]["hits_state"], slots[1400]["type_mismatch_fields"]) == (
        None, "type_mismatch", "hits")


def test_line_with_a_playerless_slot_is_quarantined_whole():
    # Review Focus 3 / Interpretation I1: identity must be complete.
    playerless = slot(1928, None, None, hits=None, at_bats=None)
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit"), playerless])]))
    assert parsed.rows == []
    (q,) = parsed.quarantined
    assert (q["locator"], q["reason"]) == ("line=1", "slot_missing_identity_or_result")
    assert json.loads(q["record_raw_json"])["predictions"][0]["roundId"] == 971      # the refused line is kept


def test_in_progress_round_with_null_result_qualifies():
    parsed = _parse(contest_line("2026-08-22T20:00:00Z", [rnd(972, None, None, None,
                                                              [slot(1930, 99, None, hits=None, at_bats=None)])]))
    (s,) = [r for r in parsed.rows if r["row_level"] == "slot"]
    assert parsed.quarantined == [] and (s["slot_result"], s["round_result"], s["round_streak"]) == (None, None, None)


def test_every_line_round_and_slot_is_an_occurrence_with_its_own_raw_record():
    # Codex plan r2 #2: a slotted round's own facts (here a wrong-typed streak) survive in its round row.
    parsed = _parse(contest_line("2026-05-14T14:30:00Z", [rnd(873, "void", 0, 0, []),
                                                          rnd(874, "hit", "RAW_STREAK", 1, [slot(1900, 11, "hit")])]))
    assert [(r["row_level"], r["locator"]) for r in parsed.rows] == [
        ("line", "line=1"), ("round", "line=1/round=0"), ("round", "line=1/round=1"), ("slot", "line=1/round=1/slot=0")]
    line, slotless, slotted, s = parsed.rows
    assert "predictions" not in json.loads(line["record_raw_json"])
    assert json.loads(slotless["record_raw_json"]) == {"roundId": 873, "result": "void", "streak": 0, "streakIncrease": 0}
    assert json.loads(slotted["record_raw_json"])["streak"] == "RAW_STREAK" and slotted["round_streak"] is None
    assert slotted["type_mismatch_fields"] == "streak" and s["round_streak"] is None


def test_round_growth_and_later_drop_are_tracked():
    parsed = _parse(
        contest_line("2026-08-20T20:00:00Z", [rnd(971, "hit", 7, 1, [slot(1928, 2513, "hit")])]),
        contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit"),
                                                                     slot(1927, 1300, "hit", number=2)])]),
        contest_line("2026-08-22T14:30:00Z", [rnd(972, "not_hit", 0, -8, [slot(1930, 99, "not_hit", hits=0)])]))
    hist = {(h["round_id"], h["player_id"]): h for h in slot_history(parsed.rows)}
    a, b = hist[(971, 2513)], hist[(971, 1300)]
    assert (a["first_seen"], a["last_seen"], a["n_observations"]) == (
        "2026-08-20T20:00:00.000000Z", "2026-08-21T14:30:00.000000Z", 2)
    assert a["changed"] is True and b["changed"] is False
    assert b["first_seen"] == "2026-08-21T14:30:00.000000Z"
    assert a["dropped_later"] is True and b["dropped_later"] is True     # round 971 absent from the 8/22 line
    assert hist[(972, 99)]["dropped_later"] is False
    assert (a["slot_result"], a["round_streak"]) == ("hit", 8)           # the older positive is kept


def test_a_malformed_later_line_does_not_mark_older_slots_dropped():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])]),
                    '{"recorded_at": "2026-08-22T14:30:00Z", "predictions": [{"result": "hit"}]}')
    assert parsed.quarantined[0]["locator"] == "line=2"
    (h,) = slot_history(parsed.rows)
    assert h["dropped_later"] is False and h["last_seen"] == "2026-08-21T14:30:00.000000Z"


def test_streak_before_uses_the_previous_entered_round_in_the_same_line():
    # Interpretation I2: no line ever reports 970, so it was a skip day and 971's previous entered round is 969.
    parsed = _parse(contest_line("2026-08-22T14:30:00Z", [
        rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
        rnd(971, "hit", 7, 2, [slot(1928, 2513, "hit")]),
        rnd(972, None, None, None, [slot(1930, 99, None, hits=None, at_bats=None)])]))
    streaks, entered = line_round_streaks(parsed.rows)[1], entered_rounds(parsed.rows)
    assert streak_before(streaks, 971, entered) == 5
    assert streak_before(streaks, 969, entered) is None
    assert streak_before(streaks, 972, entered) == 7


def test_streak_before_is_unknown_when_the_previous_entered_round_is_missing_from_the_line():
    # Codex plan r1 #6: line 1 reports 969 and 970; line 2 dropped 970, so 971's predecessor is not in line 2.
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
                                                          rnd(970, "not_hit", 0, -5, [slot(1901, 12, "not_hit", hits=0)])]),
                    contest_line("2026-08-22T14:30:00Z", [rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
                                                          rnd(971, "hit", 1, 1, [slot(1928, 2513, "hit")])]))
    assert streak_before(line_round_streaks(parsed.rows)[2], 971, entered_rounds(parsed.rows)) is None


def test_saver_transitions_are_reported_as_attempts_with_their_raw_record():
    data = (json.dumps({"ts": "2026-08-10T12:00:00+00:00", "source": "auto", "outcome": "rejected",
                        "new_state": "used"}) + "\n").encode()
    (row,) = parse_saver_transitions("picks/account_state/saver_transitions.jsonl", data).rows
    assert (row["attempted_at"], row["attempt_source"], row["attempt_outcome"]) == (
        "2026-08-10T12:00:00.000000Z", "auto", "rejected")
    assert json.loads(row["record_raw_json"])["new_state"] == "used"
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_contest.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.contest'`.

- [ ] **Step 4: Write the implementation**

`scripts/audit/season_ledger/sources/contest_ledger.py`:
```python
"""O6 contest ledger (`account_state/contest_ledger.jsonl`), lossless, and the saver-transition attempts log
(spec §4, §6). Each line, round and slot keeps its own raw JSON."""
from __future__ import annotations

import json

from ..ids import Parsed, canonical_json, is_int, joined, obs_id, quarantine, sha256_hex, take, typed, utc_iso

LINE_FIELDS = {"source_date": "str", "active_streak": "int", "best_streak": "int"}
ROUND_FIELDS = {"result": "str", "streak": "int", "streakIncrease": "int"}
SLOT_FIELDS = {"number": "int"}


def _state(record: dict, key: str, kind: str) -> tuple[object, str]:
    """A typed value plus how the source held it: value / null / absent / type_mismatch."""
    if key not in record:
        return None, "absent"
    value, bad = typed(record[key], kind)
    if bad:
        return None, "type_mismatch"
    return (value, "value") if value is not None else (None, "null")


def _disqualify(doc) -> str | None:
    """Interpretation I1: identity integers required; `result` keys present (null = in progress)."""
    if not isinstance(doc, dict) or utc_iso(doc.get("recorded_at")) is None or not isinstance(doc.get("predictions"), list):
        return "line_missing_recorded_at_or_predictions"
    for rnd in doc["predictions"]:
        if not isinstance(rnd, dict) or not is_int(rnd.get("roundId")) or "result" not in rnd:
            return "round_missing_roundId_or_result"
        slots = rnd.get("roundPredictions")
        if slots is not None and not isinstance(slots, list):
            return "round_predictions_not_list"
        for s in slots or []:
            if (not isinstance(s, dict) or not is_int(s.get("unitId")) or not is_int(s.get("playerId"))
                    or "result" not in s):
                return "slot_missing_identity_or_result"
    return None


def _lines(data: bytes):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]


def parse_contest_ledger(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        line_loc = f"line={line_no}"
        try:
            doc = json.loads(line)
            reason, raw = _disqualify(doc), doc
        except json.JSONDecodeError:
            reason, raw = "invalid_json_line", line
        if reason:
            out.quarantined.append(quarantine(rel_path, line_loc, reason, raw=raw))
            continue
        line_values, line_absent, line_bad = take(doc, LINE_FIELDS)
        base = {"source_kind": "contest_ledger", "source_path": rel_path, "content_sha256": content,
                "line_no": line_no, "recorded_at": utc_iso(doc["recorded_at"]), **line_values}
        out.rows.append(dict(base, row_level="line", locator=line_loc, obs_id=obs_id(rel_path, line_loc, content),
                             absent_fields=joined(line_absent), type_mismatch_fields=joined(line_bad),
                             record_raw_json=canonical_json({k: v for k, v in doc.items() if k != "predictions"})))
        for i, rnd in enumerate(doc["predictions"]):
            round_values, round_absent, round_bad = take(rnd, ROUND_FIELDS)
            rbase = dict(base, round_id=rnd["roundId"], round_result=round_values["result"],
                         round_streak=round_values["streak"], round_streak_increase=round_values["streakIncrease"])
            round_loc = f"{line_loc}/round={i}"
            # Every round is its own occurrence with its own raw record (Codex plan r2 #2).
            out.rows.append(dict(rbase, row_level="round", locator=round_loc,
                                 obs_id=obs_id(rel_path, round_loc, content), absent_fields=joined(round_absent),
                                 type_mismatch_fields=joined(round_bad),
                                 record_raw_json=canonical_json({k: v for k, v in rnd.items()
                                                                 if k != "roundPredictions"})))
            for j, s in enumerate(rnd.get("roundPredictions") or []):
                loc = f"{round_loc}/slot={j}"
                slot_values, slot_absent, slot_bad = take(s, SLOT_FIELDS)
                result, result_state = _state(s, "result", "str")
                hits, hits_state = _state(s, "hits", "int")
                at_bats, at_bats_state = _state(s, "atBats", "int")
                states = (("result", result_state), ("hits", hits_state), ("atBats", at_bats_state))
                out.rows.append(dict(rbase, row_level="slot", locator=loc, obs_id=obs_id(rel_path, loc, content),
                                     slot_number=slot_values["number"], unit_id=s["unitId"], player_id=s["playerId"],
                                     slot_result=result, slot_result_state=result_state, hits=hits,
                                     hits_state=hits_state, at_bats=at_bats, at_bats_state=at_bats_state,
                                     absent_fields=joined([f"round.{f}" for f in round_absent] + slot_absent
                                                          + [k for k, st in states if st == "absent"]),
                                     type_mismatch_fields=joined([f"round.{f}" for f in round_bad] + slot_bad
                                                                 + [k for k, st in states if st == "type_mismatch"]),
                                     record_raw_json=canonical_json(s)))
    return out


def parse_saver_transitions(rel_path: str, data: bytes) -> Parsed:
    """Rows are attempts (including rejected ones), never consumption times (spec §6)."""
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        loc = f"line={line_no}"
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, loc, "invalid_json_line", raw=line))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, loc, "line_not_object", raw=doc))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, loc, content), "locator": loc, "source_kind": "saver_transitions",
                         "source_path": rel_path, "content_sha256": content, "line_no": line_no,
                         "attempted_at": utc_iso(doc.get("ts")),
                         "attempt_source": None if doc.get("source") is None else str(doc.get("source")),
                         "attempt_outcome": None if doc.get("outcome") is None else str(doc.get("outcome")),
                         "record_raw_json": canonical_json(doc)})
    return out
```

`scripts/audit/season_ledger/contest.py`:
```python
"""Contest slot history, the streak chain, lookups and game matching (spec §6)."""
from __future__ import annotations

from collections import Counter

SLOT_VALUE_KEYS = ("slot_result", "slot_result_state", "hits", "hits_state", "at_bats", "at_bats_state",
                   "slot_number", "round_result", "round_streak", "round_streak_increase")


def _order(row: dict) -> tuple:
    return (row["recorded_at"], row["line_no"])     # fixed-precision UTC: string order is time order


def slot_history(contest_rows: list[dict]) -> list[dict]:
    """Per slot identity (round_id, unit_id, player_id): first/last seen, the last qualified values, whether
    values changed, and `dropped_later` when a later qualified line no longer shows it. A newer omission
    never erases the older positive observation; every earlier value stays in the occurrence table."""
    lines = sorted({_order(r) for r in contest_rows if r["row_level"] == "line"})
    groups: dict[tuple, list[dict]] = {}
    for r in sorted((r for r in contest_rows if r["row_level"] == "slot"), key=_order):
        groups.setdefault((r["round_id"], r["unit_id"], r["player_id"]), []).append(r)
    out = []
    for (round_id, unit_id, player_id), obs in sorted(groups.items()):
        last = obs[-1]
        out.append({"round_id": round_id, "unit_id": unit_id, "player_id": player_id,
                    "first_seen": obs[0]["recorded_at"], "last_seen": last["recorded_at"],
                    "last_line_no": last["line_no"], "n_observations": len(obs),
                    "changed": len({tuple(o[k] for k in SLOT_VALUE_KEYS) for o in obs}) > 1,
                    "dropped_later": any(line > _order(last) for line in lines),
                    "last_obs_id": last["obs_id"], **{k: last[k] for k in SLOT_VALUE_KEYS}})
    return out


def line_round_streaks(contest_rows: list[dict]) -> dict[int, dict[int, int | None]]:
    """line_no → {round_id: reported post-round streak} for every round present in that line."""
    out: dict[int, dict[int, int | None]] = {}
    for r in contest_rows:
        if r["row_level"] in ("round", "slot"):
            out.setdefault(r["line_no"], {})[r["round_id"]] = r["round_streak"]
    return out


def entered_rounds(contest_rows: list[dict]) -> list[int]:
    """Every roundId any qualified line reports — the account's entered rounds — sorted."""
    return sorted({r["round_id"] for r in contest_rows if r["row_level"] in ("round", "slot")})


def streak_before(line_streaks: dict[int, int | None], round_id: int, entered: list[int]) -> int | None:
    """Interpretation I2: the previous entered round is the greatest roundId below `round_id` that any
    qualified line reports; its streak counts only when that round is present in this same line."""
    earlier = [r for r in entered if r < round_id]
    return line_streaks.get(earlier[-1]) if earlier else None
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_contest.py -q`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/contest_ledger.py scripts/audit/season_ledger/contest.py tests/scripts/season_ledger
git commit -m "feat(ledger): lossless contest ledger, slot history, streak chain, saver attempts (task 4)"
```

---

### Task 5: O7 static captures, validated MLB schedules, lookups

**Files:**
- Create: `scripts/audit/season_ledger/sources/static.py`
- Modify: `scripts/audit/season_ledger/contest.py` (append lookups)
- Test: `tests/scripts/season_ledger/test_static.py`

**Interfaces:**
- **Produces (parsers).** Item locators are positional (`item=<i>`, `date=<i>/game=<j>`). Every row keeps `record_raw_json`, the raw values of the fields it normalizes. A quarantined item keeps the same raw values, and a capture without its list keeps its whole document:
  - `parse_rounds(rel_path, data)` → rows `{round_id, round_date, status}`
  - `parse_players(...)` → rows `{player_id, feed_id, squad_id, name}`
  - `parse_units(...)` → rows `{unit_id, feed_id, round_id, status, captured_at}`
  - `parse_schedule(rel_path, data)` → rows `{query_date, game_pk, away_abbr, home_abbr, coded_state, detailed_state, official_date, game_number}`. Date entries without games, and games lacking a gamePk or either abbreviation, are quarantined.
- **Produces (lookups in `contest.py`):**
  - `rounds_lookup(rows)`
  - `players_lookup(rows)`
  - `units_lookup(rows)`, each value `{"feed_ids", "round_ids"}`
  - `unit_status_history(rows) -> {unit_id: [(captured_at, status)]}`, oldest first
  - `team_games(rows) -> {(query_date, abbr): {game_pk}}`

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_static.py`:

```python
import json

from scripts.audit.season_ledger.contest import (players_lookup, rounds_lookup, team_games, unit_status_history,
                                                 units_lookup)
from scripts.audit.season_ledger.sources.static import parse_players, parse_rounds, parse_schedule, parse_units
from tests.scripts.season_ledger.builders import dumps, gz


def _game(pk, away, home, state="Final", number=1):
    return {"gamePk": pk, "gameNumber": number, "officialDate": "2026-05-10",
            "status": {"codedGameState": state[0], "detailedState": state},
            "teams": {"away": {"team": {"abbreviation": away}}, "home": {"team": {"abbreviation": home}}}}


def test_gzip_and_plain_captures_both_parse_and_corrupt_gzip_is_quarantined():
    # Review Focus 4
    body = dumps({"rounds": [{"id": 971, "date": "2026-08-20T08:00:00-04:00", "status": "complete"}]})
    assert parse_rounds("static/rounds/20260704T030011Z.json", body).rows[0]["round_date"] == "2026-08-20"
    assert parse_rounds("static/rounds/20260705T030011Z.json.gz", gz(body)).rows[0]["round_id"] == 971
    bad = parse_rounds("static/rounds/20260706T030011Z.json.gz", b"\x1f\x8bgarbage")
    assert bad.rows == [] and bad.quarantined[0]["reason"].startswith("bad_gzip")


def test_normalized_fields_keep_their_raw_values():
    # Codex plan r2 #2: a wrong-typed feedId is nulled in the typed column but survives in the raw record.
    (row,) = parse_units("static/units/20260801T150000Z.json", dumps({"units": [
        {"id": 2449, "feedId": "RAW_GAME", "roundId": 1009, "status": "scheduled", "lineups": [1, 2]}]})).rows
    assert (row["feed_id"], row["type_mismatch_fields"]) == (None, "feedId")
    assert json.loads(row["record_raw_json"]) == {"id": 2449, "feedId": "RAW_GAME", "roundId": 1009, "status": "scheduled"}


def test_a_capture_without_its_list_is_quarantined_with_its_raw_document():
    # A record that parsed keeps its parsed value; only undecodable bytes stay solely in the sealed bundle.
    (q,) = parse_units("static/units/20260801T150000Z.json", dumps({"error": "RAW_ERROR"})).quarantined
    assert (q["locator"], q["reason"], json.loads(q["record_raw_json"])) == ("file", "missing_units_list",
                                                                             {"error": "RAW_ERROR"})


def test_units_carry_status_and_capture_time_and_empty_lists_have_no_rows():
    rows = parse_units("static/units/20260801T150000Z.json.gz", gz(dumps({"units": [
        {"id": 2449, "feedId": 822679, "roundId": 1009, "status": "postponed"}]}))).rows
    assert (rows[0]["captured_at"], rows[0]["status"], rows[0]["locator"]) == (
        "2026-08-01T15:00:00.000000Z", "postponed", "item=0")
    empty = parse_units("static/units/20260927T230002Z.json.gz", gz(dumps({"units": []})))
    assert empty.rows == [] and empty.quarantined == []
    assert unit_status_history(rows) == {2449: [("2026-08-01T15:00:00.000000Z", "postponed")]}
    grab = parse_units("static/grab_20260927/003_units.json.gz", gz(dumps({"units": [
        {"id": 1, "feedId": 2, "roundId": 3, "status": "scheduled"}]}))).rows
    assert grab[0]["captured_at"] is None


def test_lookups_keep_conflicting_captures():
    units = parse_units("static/units/20260801T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 822679, "roundId": 1009}]})).rows \
        + parse_units("static/units/20260802T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 999999, "roundId": 1009}]})).rows
    assert units_lookup(units)[2449] == {"feed_ids": {822679, 999999}, "round_ids": {1009}}
    players = parse_players("static/players/p.json", dumps({"players": [{"id": 1300, "feedId": 680757, "squadId": 5,
                                                                          "name": "Steven Kwan"}]})).rows
    assert players_lookup(players) == {1300: {680757}}
    rounds = parse_rounds("static/rounds/r.json", dumps({"rounds": [{"id": 1, "date": "2026-03-25T08:00:00-04:00"}]})).rows
    assert rounds_lookup(rounds) == {1: {"2026-03-25"}}


def test_schedule_lists_every_game_including_postponed():
    body = dumps({"dates": [{"date": "2026-05-10", "games": [_game(824765, "TB", "BOS"),
                                                             _game(824999, "TB", "BOS", "Postponed", 2)]}]})
    rows = parse_schedule("schedules/2026-05-10.json", body).rows
    assert team_games(rows)[("2026-05-10", "TB")] == {824765, 824999}
    assert {r["detailed_state"] for r in rows} == {"Final", "Postponed"}


def test_incomplete_schedule_entries_are_quarantined_not_dropped():
    # Codex plan r1 #7: a game without team metadata must never leave the other game looking unique.
    body = dumps({"dates": [{"date": "2026-05-10", "games": [
        _game(824765, "TB", "BOS"),
        {"gamePk": 824999, "teams": {"away": {"team": {}}, "home": {"team": {"abbreviation": "BOS"}}}}]},
        {"date": "2026-05-11"}]})
    parsed = parse_schedule("schedules/2026-05-10.json", body)
    assert [r["game_pk"] for r in parsed.rows] == [824765]
    assert {(q["locator"], q["reason"]) for q in parsed.quarantined} == {
        ("date=0/game=1", "game_missing_team_abbreviation"), ("date=1", "date_without_games_list")}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_static.py -q`
Expected: collection ERROR — `cannot import name 'players_lookup'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/sources/static.py`:
```python
"""O7 BTS static captures (rounds, players, units) and acquired MLB schedule responses (spec §4, §6). Item
locators are positional, so a capture that lists an id twice still yields distinct occurrences. Each row keeps
the RAW values of every field it normalizes (`record_raw_json`), so a wrong-typed value survives in the
occurrence table; the rest of an item (e.g. unit lineups) stays in the sealed bundle, addressable by
(source_path, locator)."""
from __future__ import annotations

import re

from ..ids import (Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex,
                   stamp_to_utc, take, typed)

_SCHEDULE_PATH = re.compile(r"^schedules/(\d{4}-\d{2}-\d{2})\.json$")


def _items(rel_path: str, data: bytes, key: str):
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return None, Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
        return None, Parsed([], [quarantine(rel_path, "file", f"missing_{key}_list", raw=doc)])
    return doc[key], None


def _raw(item, keys: tuple[str, ...]):
    return {k: item[k] for k in keys if k in item} if isinstance(item, dict) else item


def _row(kind: str, rel_path: str, locator: str, content: str, raw, **fields) -> dict:
    return {"obs_id": obs_id(rel_path, locator, content), "locator": locator, "source_kind": kind,
            "source_path": rel_path, "content_sha256": content, **fields, "record_raw_json": canonical_json(raw)}


ROUND_KEYS, PLAYER_KEYS, UNIT_KEYS = ("id", "date", "status"), ("id", "feedId", "squadId", "name"), (
    "id", "feedId", "roundId", "status")
GAME_KEYS = ("gamePk", "gameNumber", "officialDate", "status", "teams")


def parse_rounds(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "rounds")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, r in enumerate(items):
        loc = f"item={i}"
        if not isinstance(r, dict) or not is_int(r.get("id")) or not isinstance(r.get("date"), str):
            out.quarantined.append(quarantine(rel_path, loc, "round_missing_id_or_date", raw=_raw(r, ROUND_KEYS)))
            continue
        status, mismatch = typed(r.get("status"), "str")
        out.rows.append(_row("rounds", rel_path, loc, content, _raw(r, ROUND_KEYS), round_id=r["id"],
                             round_date=r["date"][:10], status=status,
                             type_mismatch_fields="status" if mismatch else None))
    return out


def parse_players(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "players")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, p in enumerate(items):
        loc = f"item={i}"
        if not isinstance(p, dict) or not is_int(p.get("id")):
            out.quarantined.append(quarantine(rel_path, loc, "player_missing_id", raw=_raw(p, PLAYER_KEYS)))
            continue
        values, absent, mismatch = take(p, {"feedId": "int", "squadId": "int", "name": "str"})
        out.rows.append(_row("players", rel_path, loc, content, _raw(p, PLAYER_KEYS), player_id=p["id"],
                             feed_id=values["feedId"],
                             squad_id=values["squadId"], name=values["name"], absent_fields=joined(absent),
                             type_mismatch_fields=joined(mismatch)))
    return out


def parse_units(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "units")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    captured_at = stamp_to_utc(rel_path.rsplit("/", 1)[-1])
    for i, u in enumerate(items):
        loc = f"item={i}"
        if not isinstance(u, dict) or not is_int(u.get("id")):
            out.quarantined.append(quarantine(rel_path, loc, "unit_missing_id", raw=_raw(u, UNIT_KEYS)))
            continue
        values, absent, mismatch = take(u, {"feedId": "int", "roundId": "int", "status": "str"})
        out.rows.append(_row("units", rel_path, loc, content, _raw(u, UNIT_KEYS), unit_id=u["id"],
                             feed_id=values["feedId"],
                             round_id=values["roundId"], status=values["status"], captured_at=captured_at,
                             absent_fields=joined(absent), type_mismatch_fields=joined(mismatch)))
    return out


def _abbreviation(teams, side: str) -> str | None:
    entry = teams.get(side) if isinstance(teams, dict) else None
    team = entry.get("team") if isinstance(entry, dict) else None
    abbr = team.get("abbreviation") if isinstance(team, dict) else None
    return abbr if isinstance(abbr, str) and abbr else None


def parse_schedule(rel_path: str, data: bytes) -> Parsed:
    """Every listed game, whatever its status. A date entry without games, or a game without a gamePk or
    both team abbreviations, is quarantined — the compiler then treats that date's schedule as incomplete
    and allows no inference on it (Codex plan r1 #7)."""
    m = _SCHEDULE_PATH.match(rel_path)
    if not m:
        return Parsed([], [quarantine(rel_path, "file", "unexpected_schedule_path")])
    items, bad = _items(rel_path, data, "dates")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, day in enumerate(items):
        games = day.get("games") if isinstance(day, dict) else None
        if not isinstance(games, list) or not games:
            out.quarantined.append(quarantine(rel_path, f"date={i}", "date_without_games_list",
                                              raw=_raw(day, ("date", "games"))))
            continue
        for j, g in enumerate(games):
            loc = f"date={i}/game={j}"
            if not isinstance(g, dict) or not is_int(g.get("gamePk")):
                out.quarantined.append(quarantine(rel_path, loc, "game_missing_gamePk", raw=_raw(g, GAME_KEYS)))
                continue
            away, home = _abbreviation(g.get("teams"), "away"), _abbreviation(g.get("teams"), "home")
            if away is None or home is None:
                out.quarantined.append(quarantine(rel_path, loc, "game_missing_team_abbreviation",
                                                  raw=_raw(g, GAME_KEYS)))
                continue
            status = g.get("status") if isinstance(g.get("status"), dict) else {}
            out.rows.append(_row("schedule", rel_path, loc, content, _raw(g, GAME_KEYS), query_date=m.group(1),
                                 game_pk=g["gamePk"],
                                 away_abbr=away, home_abbr=home,
                                 coded_state=typed(status.get("codedGameState"), "str")[0],
                                 detailed_state=typed(status.get("detailedState"), "str")[0],
                                 official_date=typed(g.get("officialDate"), "str")[0],
                                 game_number=typed(g.get("gameNumber"), "int")[0]))
    return out
```

Append to `scripts/audit/season_ledger/contest.py`:
```python
def rounds_lookup(round_rows: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for r in round_rows:
        out.setdefault(r["round_id"], set()).add(r["round_date"])
    return out


def players_lookup(player_rows: list[dict]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for p in player_rows:
        if p["feed_id"] is not None:
            out.setdefault(p["player_id"], set()).add(p["feed_id"])
    return out


def units_lookup(unit_rows: list[dict]) -> dict[int, dict]:
    """unit_id → every feedId / roundId any capture recorded; conflicts are kept, never collapsed."""
    out: dict[int, dict] = {}
    for u in unit_rows:
        entry = out.setdefault(u["unit_id"], {"feed_ids": set(), "round_ids": set()})
        if u["feed_id"] is not None:
            entry["feed_ids"].add(u["feed_id"])
        if u["round_id"] is not None:
            entry["round_ids"].add(u["round_id"])
    return out


def unit_status_history(unit_rows: list[dict]) -> dict[int, list[tuple]]:
    """unit_id → [(capture time, status)] across captures, oldest first (Interpretation I6)."""
    out: dict[int, list[tuple]] = {}
    for u in unit_rows:
        out.setdefault(u["unit_id"], []).append((u["captured_at"], u["status"]))
    return {k: sorted(v, key=lambda x: (x[0] or "", str(x[1]))) for k, v in out.items()}


def team_games(schedule_rows: list[dict]) -> dict[tuple[str, str], set[int]]:
    """(query date, team abbreviation) → every listed gamePk, whatever its status (postponed, cancelled and
    suspended entries included), so 'exactly one game' is never produced by filtering."""
    out: dict[tuple[str, str], set[int]] = {}
    for g in schedule_rows:
        for abbr in (g["away_abbr"], g["home_abbr"]):
            out.setdefault((g["query_date"], abbr), set()).add(g["game_pk"])
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_static.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger tests/scripts/season_ledger
git commit -m "feat(ledger): static capture and validated schedule parsers, conflict-keeping lookups (task 5)"
```

---

### Task 6: Contest slot → game → selection matching

**Files:**
- Modify: `scripts/audit/season_ledger/contest.py` (append `match_slot`, `resolve_duplicate_links`)
- Test: `tests/scripts/season_ledger/test_matching.py`

**Interfaces:**
- **Consumes:**
  - Task 4 slot-history rows
  - Task 5 lookups
  - `schedule_status: {date: "complete"|"incomplete"}` (missing key = missing schedule)
  - `local_selections` rows with `date, batter_id, game_pk, team_at_pick, selection_id`
- **Produces:**
  - `match_slot(slot, *, rounds, players, units, team_games, schedule_status, local_selections) -> dict`. Keys: `round_id, unit_id, player_id, date, batter_id, game_pk, selection_id, match ∈ {evidenced, inferred, ambiguous, unmapped}, match_reason`.
  - `resolve_duplicate_links(matches)` (I8).

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_matching.py`:

```python
from scripts.audit.season_ledger.contest import match_slot, resolve_duplicate_links

ROUNDS = {971: {"2026-08-20"}}
PLAYERS = {2513: {802415}, 1300: {680757}}
SEL = {"date": "2026-08-20", "batter_id": 802415, "game_pk": 822934, "team_at_pick": "TB",
       "selection_id": "2026-08-20|primary|802415|822934"}
TB_ONE = {("2026-08-20", "TB"): {822934}}
COMPLETE = {"2026-08-20": "complete"}


def _slot(unit_id=1928, player_id=2513, round_id=971):
    return {"round_id": round_id, "unit_id": unit_id, "player_id": player_id}


def _match(*, units=None, games=None, sels=(SEL,), slot=None, status=None):
    return match_slot(slot or _slot(), rounds=ROUNDS, players=PLAYERS, units=units or {},
                      team_games=TB_ONE if games is None else games,
                      schedule_status=COMPLETE if status is None else status, local_selections=list(sels))


def test_unit_capture_gives_evidenced_match_to_the_local_selection():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {971}}}, games={})
    assert (m["match"], m["game_pk"], m["selection_id"], m["match_reason"]) == (
        "evidenced", 822934, SEL["selection_id"], "unit_capture")


def test_conflicting_unit_captures_are_ambiguous_and_never_fall_through_to_inference():
    m = _match(units={1928: {"feed_ids": {822934, 900001}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "conflicting_unit_evidence")


def test_unit_capture_seen_in_another_round_is_ambiguous():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {970, 971}}})
    assert (m["match"], m["match_reason"]) == ("ambiguous", "unit_round_contradiction")


def test_unit_capture_for_another_game_does_not_link():
    m = _match(units={1928: {"feed_ids": {900002}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("evidenced", None, "unit_capture_other_game")


def test_inference_uses_the_pick_time_team_not_the_current_one():
    # A traded player: a current-team lookup would say SEA; the pick file recorded TB.
    m = _match(games={("2026-08-20", "TB"): {822934}, ("2026-08-20", "SEA"): {900001}})
    assert (m["match"], m["game_pk"], m["match_reason"]) == ("inferred", 822934, "pick_time_team_single_scheduled_game")


def test_doubleheader_or_postponed_plus_played_game_is_ambiguous():
    m = _match(games={("2026-08-20", "TB"): {822934, 822935}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "team_schedule_not_unique")


def test_selection_without_a_recorded_game_is_ambiguous():
    # Review Focus 2: the 3/29–3/30 pick files carry game_pk = null.
    m = _match(sels=(dict(SEL, game_pk=None, selection_id="2026-08-20|primary|802415|None"),))
    assert (m["match"], m["match_reason"]) == ("ambiguous", "selection_game_pk_unrecorded")


def test_missing_or_incomplete_schedule_and_absent_team_are_ambiguous_with_distinct_reasons():
    assert _match(status={})["match_reason"] == "team_schedule_missing"
    assert _match(status={"2026-08-20": "incomplete"})["match_reason"] == "team_schedule_incomplete"
    assert _match(games={("2026-08-20", "NYY"): {5002}})["match_reason"] == "team_not_on_schedule"


def test_contest_only_slot_without_unit_capture_is_unmapped():
    m = _match(sels=())
    assert (m["match"], m["match_reason"]) == ("unmapped", "no_unit_capture_contest_only")


def test_unknown_round_is_unmapped_and_mapped_and_unmapped_slots_share_a_round():
    assert _match(slot=_slot(round_id=5))["match_reason"] == "round_date_unknown"
    known, unknown = _match(), _match(slot=_slot(unit_id=1927, player_id=42))
    assert (known["match"], unknown["match"], unknown["match_reason"]) == ("inferred", "unmapped", "player_unknown")


def test_two_contest_slots_linking_one_selection_are_both_demoted():
    a = {"round_id": 971, "unit_id": 1928, "player_id": 2513, "selection_id": "s", "match": "inferred",
         "match_reason": "x"}
    out = resolve_duplicate_links([a, dict(a, unit_id=1929), dict(a, unit_id=1930, selection_id="t")])
    assert [(m["unit_id"], m["match"], m["selection_id"]) for m in out] == [
        (1928, "ambiguous", None), (1929, "ambiguous", None), (1930, "inferred", "t")]
    assert out[0]["match_reason"] == "multiple_contest_slots_for_selection"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_matching.py -q`
Expected: collection ERROR — `cannot import name 'match_slot'`.

- [ ] **Step 3: Write the implementation** (append to `scripts/audit/season_ledger/contest.py`):

```python
def _result(slot: dict, *, date=None, batter_id=None, game_pk=None, selection_id=None, match: str, reason: str) -> dict:
    return {"round_id": slot["round_id"], "unit_id": slot["unit_id"], "player_id": slot["player_id"],
            "date": date, "batter_id": batter_id, "game_pk": game_pk, "selection_id": selection_id,
            "match": match, "match_reason": reason}


def match_slot(slot: dict, *, rounds: dict, players: dict, units: dict, team_games: dict,
               schedule_status: dict[str, str], local_selections: list[dict]) -> dict:
    """Spec §6. Never matches on the slot `number`. Only a unit with no capture at all may use the
    pick-time-team inference, and only on a date whose schedule is complete; unit evidence that is not one
    round-consistent feedId is ambiguous and transfers nothing. Only `evidenced` / `inferred` links carry a
    selection_id."""
    dates = rounds.get(slot["round_id"], set())
    if len(dates) != 1:
        return _result(slot, match="unmapped", reason="round_date_unknown" if not dates else "round_date_conflict")
    date = next(iter(dates))
    feeds = players.get(slot["player_id"], set())
    if len(feeds) != 1:
        return _result(slot, date=date, match="unmapped", reason="player_unknown" if not feeds else "player_conflict")
    batter = next(iter(feeds))
    candidates = [s for s in local_selections if s["date"] == date and s["batter_id"] == batter]
    unit = units.get(slot["unit_id"])
    if unit is not None:
        if len(unit["feed_ids"]) != 1:
            reason = "unit_capture_without_feed_id" if not unit["feed_ids"] else "conflicting_unit_evidence"
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)
        if unit["round_ids"] and unit["round_ids"] != {slot["round_id"]}:
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="unit_round_contradiction")
        game = next(iter(unit["feed_ids"]))
        same = [s for s in candidates if s["game_pk"] == game]
        if len(same) == 1:
            reason = "unit_capture"
        elif same:
            reason = "unit_capture_multiple_local"
        else:
            reason = "unit_capture_other_game" if candidates else "unit_capture_no_local_selection"
        return _result(slot, date=date, batter_id=batter, game_pk=game,
                       selection_id=same[0]["selection_id"] if len(same) == 1 else None,
                       match="evidenced", reason=reason)
    if not candidates:
        return _result(slot, date=date, batter_id=batter, match="unmapped", reason="no_unit_capture_contest_only")
    if len(candidates) > 1:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="multiple_local_selections")
    sel = candidates[0]
    if sel["game_pk"] is None:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="selection_game_pk_unrecorded")
    status = schedule_status.get(date)
    if status != "complete":
        reason = "team_schedule_missing" if status is None else "team_schedule_incomplete"
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)
    games = team_games.get((date, sel["team_at_pick"]), set())
    if games == {sel["game_pk"]}:
        return _result(slot, date=date, batter_id=batter, game_pk=sel["game_pk"], selection_id=sel["selection_id"],
                       match="inferred", reason="pick_time_team_single_scheduled_game")
    if not games:
        reason = "team_not_on_schedule"
    else:
        reason = "team_schedule_not_unique" if len(games) > 1 else "team_schedule_other_game"
    return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)


def resolve_duplicate_links(matches: list[dict]) -> list[dict]:
    """Interpretation I8: two contest slot identities linking one selection (an entry changed within a
    round) are both demoted to ambiguous; neither transfers a grade."""
    counts = Counter(m["selection_id"] for m in matches if m["selection_id"])
    return [dict(m, match="ambiguous", match_reason="multiple_contest_slots_for_selection", selection_id=None)
            if m["selection_id"] and counts[m["selection_id"]] > 1 else m for m in matches]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_matching.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/contest.py tests/scripts/season_ledger/test_matching.py
git commit -m "feat(ledger): contest slot to game matching (evidenced / inferred / ambiguous / unmapped) (task 6)"
```

---

### Task 7: Outcome normalization and comparison

**Files:**
- Create: `scripts/audit/season_ledger/outcomes.py`
- Test: `tests/scripts/season_ledger/test_outcomes.py`

**Interfaces:**
- **Produces:**
  - `LOCAL_NORMALIZATION` and `CONTEST_NORMALIZATION`
  - `normalize_local(raw)` and `normalize_contest(raw)`
  - `slot_disagreement(local_norm, contest_norm) -> bool | None`
  - `derived_single_result(pick_row) -> (value, source)`

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_outcomes.py`:

```python
from scripts.audit.season_ledger.outcomes import (derived_single_result, normalize_contest, normalize_local,
                                                  slot_disagreement)


def test_local_miss_and_contest_not_hit_agree():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("not_hit")) is False


def test_void_maps_to_hold_on_both_sides():
    assert normalize_local("void") == normalize_contest("void") == "HOLD"


def test_c03_pattern_disagrees():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("hit")) is True


def test_unknown_or_missing_labels_are_never_compared():
    assert normalize_local("suspended") == normalize_local("unresolved") == "UNKNOWN"
    assert slot_disagreement("UNKNOWN", "HIT") is None and slot_disagreement(None, "HIT") is None


def test_round_labels_are_not_slot_labels():
    assert normalize_contest("used_mulligan") == "UNKNOWN"


def test_single_pick_day_result_derives_only_for_single_pick_records():
    single = {"obs_id": "o1", "has_double_down": False, "slot_result_raw": None, "day_result_raw": "hit"}
    assert derived_single_result(single) == ("hit", "day_result_of_single_pick:o1")
    assert derived_single_result(dict(single, has_double_down=True, day_result_raw="miss")) == (None, None)
    assert derived_single_result(dict(single, slot_result_raw="miss")) == (None, None)
    assert derived_single_result(None) == (None, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_outcomes.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.outcomes'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/outcomes.py`:

```python
"""Closed outcome vocabulary and slot-level comparison (spec §7). HOLD = the slot neither extended
nor broke the streak (a Pass or a voided slot); round labels (`void`, `used_mulligan`, …) are not
part of this table."""
from __future__ import annotations

LOCAL_NORMALIZATION = {"hit": "HIT", "miss": "NO_HIT", "void": "HOLD"}
CONTEST_NORMALIZATION = {"hit": "HIT", "not_hit": "NO_HIT", "void": "HOLD"}
COMPARABLE = frozenset({"HIT", "NO_HIT", "HOLD"})


def normalize_local(raw: str | None) -> str | None:
    return None if raw is None else LOCAL_NORMALIZATION.get(raw, "UNKNOWN")


def normalize_contest(raw: str | None) -> str | None:
    return None if raw is None else CONTEST_NORMALIZATION.get(raw, "UNKNOWN")


def slot_disagreement(local_norm: str | None, contest_norm: str | None) -> bool | None:
    if local_norm not in COMPARABLE or contest_norm not in COMPARABLE:
        return None
    return local_norm != contest_norm


def derived_single_result(pick_row: dict | None) -> tuple[str | None, str | None]:
    """A per-slot value from the day result only for a record that is itself a single pick and has
    no per-slot result, bound to that record (spec §7)."""
    if pick_row is None or pick_row["has_double_down"] or pick_row["slot_result_raw"] is not None:
        return None, None
    if pick_row["day_result_raw"] in LOCAL_NORMALIZATION:
        return pick_row["day_result_raw"], f"day_result_of_single_pick:{pick_row['obs_id']}"
    return None, None
```

- [ ] **Step 4: Run to verify pass** — same command. Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/outcomes.py tests/scripts/season_ledger/test_outcomes.py
git commit -m "feat(ledger): closed outcome vocabulary, slot comparison, single-pick derivation (task 7)"
```

---

### Task 8: Day status, row kinds, commit / history status, delivery predicate

**Files:**
- Create: `scripts/audit/season_ledger/rows.py`
- Test: `tests/scripts/season_ledger/test_rows.py`

**Interfaces:**
- **Consumes:** parsed rows from Tasks 2–3 for one date:
  - the decision row or None
  - the production pick-file slot rows, and the file's `pick_file_state` (None when no pick file survives)
  - the scheduler-state row or None
  - `observations`: archive, manual- and repair-version rows and lineup-evolution rows
  - `unusable`: the day-evidence kinds (decision, scheduler state, archives, lineup evolution) with a quarantined record for the date (I15)
- **Produces:**
  - `selection_id(date, slot, batter_id, game_pk) -> "<date>|<slot>|<batter_id>|<game_pk>"`
  - `decision_names(decision, slot)`
  - `pick_delivery(pick)`
  - `delivery(*, decision_status, pick) -> (confirmed, basis, conflict)` (I4)
  - `commit_status(*, selection, slot, decision, file_pick, pick_agrees) -> (status, basis)` (I3/I14)
  - `history_status(thin, retained)` (I5)
  - `day_rows(date, *, decision, pick_rows, pick_file, state, observations, unusable=frozenset()) -> list[dict]`. Rows carry `pick_view_action`, `pick_file_complete`, `scheduler_commit_flag` and `pick_policy_objective`, plus the pick, decision and timeline columns. A day with no decision whose surviving pick file has no usable slot is `unfinalized_day` / `pick_file_unparseable`. The I15 reasons are `decision_unusable`, `skip_decision_with_unreadable_pick_file`, `skip_decision_with_unusable_state` and `unusable_evidence_only`. Task 10 fills the contest, outcome and eligibility columns.

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_rows.py`:

```python
from scripts.audit.season_ledger.rows import day_rows
from scripts.audit.season_ledger.sources.day_records import (parse_decision, parse_lineup_evolution,
                                                             parse_scheduler_state)
from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file, pick_file_state
from tests.scripts.season_ledger.builders import cand, decision_json, evolution_jsonl, pick_json, state_json

D = "2026-08-20"
A = {"batter_id": 101, "game_pk": 5001}                    # the builders' default primary
B = {"batter_id": 202, "game_pk": 5002, "team": "NYY"}     # the builders' default double-down


def dec(**kw):
    return parse_decision(f"picks/{D}/decision.json", decision_json(D, **kw)).rows[0]


def picks(data):
    """(parsed rows, pick_file_state) for a surviving pick file, as the compiler passes them."""
    parsed = parse_pick_file(f"picks/{D}.json", data)
    return parsed.rows, pick_file_state(data, parsed)


def state(**kw):
    return parse_scheduler_state(f"picks/{D}/scheduler_state.json", state_json(D, **kw)).rows[0]


def evo(entries):
    return parse_lineup_evolution(f"picks/lineup_evolution_{D}.jsonl", evolution_jsonl(D, entries)).rows


def archived(primary):
    return parse_archive(f"picks/{D}/deferred_fallback_20260820T110000-0400.json",
                         pick_json(D, primary=primary, deferred_fallback={"reason": "r",
                                                                          "deferred_at": f"{D}T11:00:00-04:00"})).rows


def rows(decision=None, pick=None, st=None, observations=(), unusable=()):
    return day_rows(D, decision=decision, pick_rows=list(pick[0]) if pick else [], pick_file=pick[1] if pick else None,
                    state=st, observations=list(observations), unusable=frozenset(unusable))


def test_decision_double_gives_two_committed_selections():
    out = rows(decision=dec(action="double", primary=cand(101, 5001), double_down=cand(202, 5002, team="NYY")),
               pick=picks(pick_json(D, dd={}, notification_sent=True, notification_id="dm-1")))
    assert [(r["slot"], r["finalization"], r["commit_status"], r["delivery_confirmed"]) for r in out] == [
        ("primary", "decision", "committed_evidenced", True), ("double_down", "decision", "committed_evidenced", True)]
    assert out[0]["commit_basis"] == "decision:delivered;delivery:dm_notification"
    assert out[0]["lineup_position"] == 1 and out[0]["predicted_at"] == f"{D}T15:00:00.000000Z"


def test_skip_decision_with_declined_candidate_is_one_skip_day_row():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)))
    assert [(r["row_kind"], r["selection_id"], r["declined_batter_id"]) for r in out] == [("skip_day", None, 303)]


def test_skip_decision_with_commit_evidence_is_not_a_skip():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)),
               pick=picks(pick_json(D, notification_sent=True, notification_id="dm-9")))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_decision_with_commit_evidence")]


def test_skip_decision_beside_an_unparseable_but_delivered_pick_file_is_not_a_skip():
    # Codex plan r3 #5: a file-level delivery signal survives the loss of the file's only slot.
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)),
               pick=picks(pick_json(D, primary={"batter_id": "bad"}, notification_sent=True, notification_id="dm-9")))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_decision_with_commit_evidence")]


def test_a_skip_is_not_established_beside_unreadable_evidence():
    skip = dec(action="skip", primary=cand(303, 7001))
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, pick=picks(b'{"pick": {'))] == [
        ("unfinalized_day", "skip_decision_with_unreadable_pick_file")]
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, unusable={"scheduler_state"})] == [
        ("unfinalized_day", "skip_decision_with_unusable_state")]
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, pick=picks(pick_json(D)))] == [
        ("skip_day", "decision_skip")]          # a readable, undelivered preview does not block a skip


def test_unusable_evidence_is_never_absent_evidence():
    # Codex plan r3 #5: a quarantined decision, or quarantined history alone, is not an unobserved day.
    assert [(r["row_kind"], r["reason"]) for r in rows(pick=picks(pick_json(D)), unusable={"decision"})] == [
        ("unfinalized_day", "decision_unusable")]
    assert [(r["row_kind"], r["reason"]) for r in rows(unusable={"lineup_evolution"})] == [
        ("unfinalized_day", "unusable_evidence_only")]
    assert [(r["row_kind"], r["reason"]) for r in rows()] == [("unobserved_day", "no_evidence")]


def test_scheduler_skip_candidate_alone_is_unfinalized_intent():
    out = rows(st=state(final_skip_candidate={"primary": cand(303, 7001), "double": None}))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_intent_only")]


def test_commit_flag_names_no_selection_and_never_commits_one():
    # Codex plan r1 #3: a date-level flag is not evidence that the surviving preview was the commit.
    st = state(final_skip_candidate={"primary": cand(303, 7001), "double": None}, committed_pick_written=True)
    assert [(r["row_kind"], r["reason"]) for r in rows(st=st)] == [("unfinalized_day", "commit_flag_without_record")]
    (sel,) = rows(st=st, pick=picks(pick_json(D)))
    assert (sel["row_kind"], sel["commit_status"], sel["scheduler_commit_flag"]) == ("selection", "unconfirmed", True)


def test_archive_only_lineup_only_and_no_evidence_days():
    assert rows(observations=archived(A))[0]["reason"] == "archived_candidates_only"
    assert rows(observations=evo([(A, None)]))[0]["reason"] == "lineup_evolution_only"
    assert [(r["row_kind"], r["reason"]) for r in rows()] == [("unobserved_day", "no_evidence")]


def test_decision_only_selection_has_no_pick_file_fields():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)))
    assert (r["finalization"], r["commit_status"], r["lineup_position"], r["pick_obs_id"]) == (
        "decision", "committed_evidenced", None, None)
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (True, "decision_delivered")


def test_decision_and_pick_file_naming_different_selections_is_unresolved_without_attachment():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099}, result="hit",
                                          slot_results={"pick": "hit"})))
    assert (r["finalization"], r["pick_view_batter_id"], r["pick_view_game_pk"]) == ("unresolved", 999, 5099)
    assert r["pick_obs_id"] is None and r["delivered_at"] is None and r["lineup_position"] is None
    assert r["p_stated"] == 0.77 and r["commit_status"] == "committed_evidenced"   # the decision's own values


def test_single_decision_against_a_double_pick_file_is_unresolved_for_the_whole_set():
    # Codex plan r1 #4: the same primary must not let the double's delivery or results attach to the single.
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, dd={}, notification_sent=True, notification_id="dm-4", result="hit",
                                          slot_results={"pick": "hit", "double_down": "hit"})))
    assert (r["finalization"], r["pick_view_action"], r["pick_view_batter_id"]) == ("unresolved", "double", 101)
    assert (r["pick_obs_id"], r["delivered_at"], r["delivery_basis"]) == (None, None, "decision_delivered")
    assert (r["commit_status"], r["commit_basis"]) == ("conflicted", "decision:delivered;other:delivery:dm_notification")


def test_same_primary_with_a_different_double_down_is_unresolved():
    out = rows(decision=dec(action="double", primary=cand(101, 5001), double_down=cand(202, 5002, team="NYY")),
               pick=picks(pick_json(D, dd={"batter_id": 303, "game_pk": 5003})))
    assert [(r["slot"], r["finalization"], r["pick_view_batter_id"]) for r in out] == [
        ("primary", "unresolved", 101), ("double_down", "unresolved", 303)]


def test_delivered_other_selection_makes_the_decision_selection_conflicted():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099},
                                          notification_sent=True, notification_id="dm-2")))
    assert (r["commit_status"], r["commit_basis"]) == ("conflicted", "decision:delivered;other:delivery:dm_notification")


def test_missing_decision_with_a_delivered_pick_is_committed_via_delivery():
    (r,) = rows(pick=picks(pick_json(D, notification_sent=True, notification_id="dm-7")))
    assert (r["finalization"], r["commit_status"], r["commit_basis"]) == (
        "pick_file_only", "committed_evidenced", "delivery:dm_notification")


def test_undelivered_preview_is_unconfirmed_and_a_lock_alone_proves_nothing():
    (r,) = rows(pick=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["delivery_confirmed"], r["locked_at"]) == ("unconfirmed", None, None)


def test_private_commit_is_committed_without_a_delivery_claim():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["commit_basis"]) == ("committed_evidenced", "decision:private_locked")
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (False, "decision_private_locked")
    assert r["locked_at"] == f"{D}T21:00:00.000000Z"


def test_locked_unconfirmed_is_committed_with_unknown_delivery():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="locked_unconfirmed"),
                pick=picks(pick_json(D, delivery_attempted=True)))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == (
        "committed_evidenced", None, "decision_locked_unconfirmed")


def test_legacy_public_post_is_a_delivery_signal():
    (r,) = rows(pick=picks(pick_json(D, bluesky_posted=True, bluesky_uri="at://post/1")))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == ("committed_evidenced", True, "public_post")
    (r2,) = rows(pick=picks(pick_json(D, bluesky_posted=True)))
    assert (r2["commit_status"], r2["delivery_basis"]) == ("unconfirmed", "bluesky_posted_without_uri")


def test_positive_pick_signal_outranks_private_status_and_is_flagged():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick=picks(pick_json(D, notification_sent=True, notification_id="dm-3")))
    assert (r["delivery_confirmed"], r["delivery_basis"], r["delivery_evidence_conflict"]) == (True, "dm_notification", True)


def test_known_incomplete_needs_a_version_whose_content_is_gone():
    d, p = dec(action="single", primary=cand(101, 5001)), picks(pick_json(D))
    assert rows(decision=d, pick=p)[0]["history_status"] == "unknown"
    lost = evo([({"batter_id": 111, "game_pk": 5011}, None), (A, None)])
    assert rows(decision=d, pick=p, observations=lost)[0]["history_status"] == "known_incomplete"
    # Codex plan r1 #5: the same change with the earlier version retained in an archive proves nothing is gone.
    kept = lost + archived({"batter_id": 111, "game_pk": 5011})
    assert rows(decision=d, pick=p, observations=kept)[0]["history_status"] == "unknown"
    # A failed append for an earlier selection followed by an overwrite leaves only consistent evidence.
    assert rows(decision=d, pick=p, observations=evo([(A, None)]))[0]["history_status"] == "unknown"


def test_known_incomplete_is_withheld_while_a_record_that_might_hold_the_version_is_unusable():
    d = dec(action="single", primary=cand(101, 5001))
    lost = evo([({"batter_id": 111, "game_pk": 5011}, None), (A, None)])
    assert rows(decision=d, pick=picks(pick_json(D)), observations=lost)[0]["history_status"] == "known_incomplete"
    assert rows(decision=d, pick=picks(pick_json(D)), observations=lost,
                unusable={"archive"})[0]["history_status"] == "unknown"
    partial = picks(pick_json(D, dd={"batter_id": "bad"}))
    assert rows(decision=d, pick=partial, observations=lost)[0]["history_status"] == "unknown"


def test_discarded_double_down_preview_marks_the_single_known_incomplete():
    out = rows(decision=dec(action="single", primary=cand(101, 5001)), pick=picks(pick_json(D)),
               observations=evo([(A, B), (A, None)]))
    assert [(r["slot"], r["history_status"]) for r in out] == [("primary", "known_incomplete")]


def test_a_decision_whose_saved_pick_file_is_gone_is_known_incomplete():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)), observations=evo([(A, None)]))
    assert r["history_status"] == "known_incomplete"


def test_delivered_at_is_the_first_pick_file_signal_and_is_normalized():
    (r,) = rows(pick=picks(pick_json(D, delivered_at=f"{D}T13:36:00-04:00", notification_sent=True,
                                          notification_id="dm")))
    assert (r["delivery_basis"], r["delivered_at"]) == ("delivered_at", f"{D}T17:36:00.000000Z")


def test_a_quarantined_double_down_leg_prevents_agreement():
    # Codex plan r2 #4: an unusable DD leg must not make a double pick file look like the decision's single.
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, dd={"batter_id": "not-an-id"}, notification_sent=True, notification_id="dm-5")))
    assert (r["finalization"], r["pick_view_action"], r["pick_file_complete"], r["pick_obs_id"]) == (
        "unresolved", "double", False, None)
    assert r["commit_status"] == "conflicted"


def test_an_unparseable_pick_file_is_a_conflicting_view_not_an_absence():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)), pick=picks(b'{"pick": {'))
    assert (r["finalization"], r["pick_file_complete"], r["pick_view_batter_id"]) == ("unresolved", False, None)
    assert [(x["row_kind"], x["reason"]) for x in rows(pick=picks(b'{"pick": {'))] == [
        ("unfinalized_day", "pick_file_unparseable")]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_rows.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.rows'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/rows.py`:

```python
"""Day status, row kinds, commit / history status and the delivery predicate (spec §5, §9)."""
from __future__ import annotations

from .ids import utc_iso

SELECTION_ACTIONS = frozenset({"single", "double"})
ARCHIVE_KINDS = frozenset({"archive", "manual_archive", "repair_archive"})


def selection_id(date: str, slot: str, batter_id, game_pk) -> str:
    return f"{date}|{slot}|{batter_id}|{game_pk}"


def decision_names(decision: dict | None, slot: str) -> tuple | None:
    """The (batter_id, game_pk) a single/double decision names for `slot`, else None."""
    if decision is None or decision["action"] not in SELECTION_ACTIONS:
        return None
    if slot == "double_down" and decision["action"] != "double":
        return None
    return decision[f"{slot}_batter_id"], decision[f"{slot}_game_pk"]


def pick_delivery(pick: dict | None) -> tuple[bool | None, str]:
    """The pick-file branches of the §9 era predicate, in order. An attempt alone is not delivery."""
    if pick is None:
        return None, "no_pick_file"
    if pick["delivered_at"]:
        return True, "delivered_at"
    if pick["notification_sent"] and pick["notification_id"]:
        return True, "dm_notification"
    if pick["bluesky_posted"] and pick["bluesky_uri"]:
        return True, "public_post"
    if pick["bluesky_posted"]:
        return None, "bluesky_posted_without_uri"
    if pick["delivery_attempted"]:
        return None, "attempt_only"
    return None, "no_delivery_evidence"


def delivery(*, decision_status: str | None, pick: dict | None) -> tuple[bool | None, str, bool]:
    """§9 for one selection (Interpretation I4). `decision_status` is the delivery_status of a decision naming
    this selection (else None); `pick` the attached pick-file slot (else None). A decision `delivered`, then
    a positive pick-side signal, confirm delivery; only without either does `private_locked` read false and
    `locked_unconfirmed` null. The flag marks private/lock status beside a positive pick-side signal."""
    pick_ok, pick_basis = pick_delivery(pick)
    conflict = decision_status in ("private_locked", "locked_unconfirmed") and pick_ok is True
    if decision_status == "delivered":
        return True, "decision_delivered", conflict
    if pick_ok:
        return True, pick_basis, conflict
    if decision_status == "private_locked":
        return False, "decision_private_locked", conflict
    if decision_status == "locked_unconfirmed":
        return None, "decision_locked_unconfirmed", conflict
    return None, pick_basis, conflict


def commit_status(*, selection: tuple, slot: str, decision: dict | None, file_pick: dict | None,
                  pick_agrees: bool) -> tuple[str, str]:
    """§5 and Interpretation I3, independent of contest entry: commit evidence naming this selection versus
    evidence naming something else. A pick file's confirmed delivery counts for the selection only when the
    file's whole selection set agrees (Interpretation I14). The scheduler's committed_pick_written flag
    names no selection and is never commit evidence; neither is a contest match, a generic pick_locked flag
    or a delivery attempt."""
    this: list[str] = []
    other: list[str] = []
    named = decision_names(decision, slot)
    if named is not None:
        (this if named == selection else other).append(f"decision:{decision['delivery_status']}")
    if file_pick is not None:
        ok, basis = pick_delivery(file_pick)
        if ok:
            (this if pick_agrees else other).append(f"delivery:{basis}")
    if other:
        return "conflicted", ";".join(this + [f"other:{o}" for o in other])
    if this:
        return "committed_evidenced", ";".join(this)
    return "unconfirmed", "no_commit_evidence"


def history_status(thin: list[dict], retained: list[dict]) -> str:
    """§5 and Interpretation I5: known_incomplete when a thin observation (a lineup-evolution entry) names a
    (slot, batter, game) that no retained full record for the date names — that version's content is gone.
    A retained archive or version is an observation, not missing content. Never `complete`."""
    kept = {(r["slot"], r["batter_id"], r["game_pk"]) for r in retained}
    return "known_incomplete" if any((o["slot"], o["batter_id"], o["game_pk"]) not in kept for o in thin) else "unknown"


def _decision_cols(decision: dict | None) -> dict:
    if decision is None:
        return {}
    return {"action": decision["action"], "action_source_raw": decision["action_source_raw"],
            "action_source": decision["action_source"], "objective": decision["objective"],
            "degraded_reason": decision["degraded_reason"], "decision_streak": decision["streak"],
            "decision_state_source": decision["state_source"], "decision_state_status": decision["state_status"],
            "decision_obs_id": decision["obs_id"]}


def _day_row(date: str, kind: str, reason: str, *, decision=None, state=None) -> dict:
    row = {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None, "selection_id": None,
           "reason": reason, "state_obs_id": state["obs_id"] if state else None,
           "scheduler_commit_flag": state["committed_pick_written"] if state else None, **_decision_cols(decision)}
    if decision is not None and decision["action"] == "skip":
        row.update(declined_batter_id=decision["primary_batter_id"], declined_game_pk=decision["primary_game_pk"])
    return row


def _selection_row(date, slot, selection, *, name, team, p, decision, pick, pick_view, pick_action, file_pick,
                   pick_agrees, pick_complete, state, finalization, history) -> dict:
    commit, basis = commit_status(selection=selection, slot=slot, decision=decision, file_pick=file_pick,
                                  pick_agrees=pick_agrees)
    named = decision_names(decision, slot) == selection
    confirmed, dbasis, conflict = delivery(
        decision_status=decision["delivery_status"] if (decision is not None and named) else None, pick=pick)
    locked = (state is not None and state["pick_locked"] and commit == "committed_evidenced"
              and finalization != "unresolved")
    sid = selection_id(date, slot, *selection)
    return {"row_id": sid, "row_kind": "selection", "date": date, "slot": slot, "selection_id": sid, "reason": None,
            "batter_id": selection[0], "batter_name": name, "team_at_pick": team, "game_pk": selection[1],
            "p_stated": p, "game_time": utc_iso(pick["game_time"]) if pick else None,
            "lineup_position": pick["lineup_position"] if pick else None,
            "projected_lineup": pick["projected_lineup"] if pick else None,
            "pitcher_id": pick["pitcher_id"] if pick else None, "finalization": finalization,
            "pick_view_batter_id": pick_view["batter_id"] if pick_view else None,
            "pick_view_game_pk": pick_view["game_pk"] if pick_view else None,
            "pick_view_action": pick_action if finalization == "unresolved" else None,
            "pick_file_complete": pick_complete,
            "commit_status": commit, "commit_basis": basis, "history_status": history,
            "scheduler_commit_flag": state["committed_pick_written"] if state else None,
            "pick_policy_objective": pick["pick_policy_objective"] if pick else None,
            "predicted_at": utc_iso(pick["run_time"]) if pick else None,
            "locked_at": utc_iso(state["pick_locked_at"]) if locked else None,
            "delivery_attempted": pick["delivery_attempted"] if pick else None, "delivery_attempted_at": None,
            "delivery_confirmed": confirmed, "delivery_basis": dbasis, "delivery_evidence_conflict": conflict,
            "delivered_at": utc_iso(pick["delivered_at"]) if pick else None,
            "game_eligibility": "unknown", "game_eligibility_at": None, "game_eligibility_basis": None,
            "pick_obs_id": pick["obs_id"] if pick else None,
            "pick_view_obs_id": pick_view["obs_id"] if pick_view else None,
            "state_obs_id": state["obs_id"] if state else None, **_decision_cols(decision)}


def day_rows(date: str, *, decision: dict | None, pick_rows: list[dict], pick_file: dict | None,
             state: dict | None, observations: list[dict], unusable: frozenset = frozenset()) -> list[dict]:
    """Spec §5 table. Declined skip candidates, scheduler intent, archives and lineup-evolution entries never
    become selections. `pick_file` is None when no pick file survives, else its `pick_file_state`. A decision
    and a surviving pick file agree only when the file parsed completely and names the identical selection set
    (Interpretation I14); only then are the file's facts attached. `unusable` names the other day-evidence kinds
    present for the date but quarantined: unknown content never reads as absent evidence, never establishes a
    skip, and withholds `known_incomplete` (Codex plan r3 #5)."""
    thin = [o for o in observations if o["source_kind"] == "lineup_evolution"]
    retained = [*pick_rows, *(o for o in observations if o["source_kind"] in ARCHIVE_KINDS)]
    unreadable_versions = bool(unusable & ARCHIVE_KINDS) or (pick_file is not None and not pick_file["complete"])
    history = "unknown" if unreadable_versions else history_status(thin, retained)
    by_slot = {r["slot"]: r for r in pick_rows}
    file_pick = pick_rows[0] if pick_rows else None
    complete = None if pick_file is None else pick_file["complete"]
    pick_action = None if not pick_file or not pick_file["slots"] else (
        "double" if "double_down" in pick_file["slots"] else "single")
    if "decision" in unusable:
        return [_day_row(date, "unfinalized_day", "decision_unusable", state=state)]
    if decision is not None and decision["action"] in SELECTION_ACTIONS:
        slots = ("primary", "double_down") if decision["action"] == "double" else ("primary",)
        chosen = [(slot, decision_names(decision, slot)) for slot in slots]
        agree = pick_file is None or (pick_file["complete"] and {(r["slot"], r["batter_id"], r["game_pk"])
                                                                  for r in pick_rows} == {(s, *sel) for s, sel in chosen})
        return [_selection_row(date, slot, sel, name=decision[f"{slot}_batter_name"], team=decision[f"{slot}_team"],
                               p=decision[f"{slot}_p_game_hit"], decision=decision,
                               pick=by_slot.get(slot) if agree else None,
                               pick_view=None if agree else by_slot.get(slot), pick_action=pick_action,
                               file_pick=file_pick, pick_agrees=agree, pick_complete=complete, state=state,
                               finalization="decision" if agree else "unresolved", history=history)
                for slot, sel in chosen]
    if decision is not None:   # action == "skip"
        if decision["scoreable"] is not False:
            return [_day_row(date, "unfinalized_day", "skip_decision_unexpected_shape", decision=decision, state=state)]
        file_fields = pick_file["file_fields"] if pick_file is not None else None
        if ((state is not None and state["committed_pick_written"]) or any(pick_delivery(p)[0] for p in pick_rows)
                or (file_fields is not None and pick_delivery(file_fields)[0])):
            return [_day_row(date, "unfinalized_day", "skip_decision_with_commit_evidence", decision=decision, state=state)]
        if pick_file is not None and file_fields is None:
            return [_day_row(date, "unfinalized_day", "skip_decision_with_unreadable_pick_file", decision=decision,
                             state=state)]
        if "scheduler_state" in unusable:
            return [_day_row(date, "unfinalized_day", "skip_decision_with_unusable_state", decision=decision, state=state)]
        return [_day_row(date, "skip_day", "decision_skip", decision=decision, state=state)]
    if pick_rows:
        return [_selection_row(date, p["slot"], (p["batter_id"], p["game_pk"]), name=p["batter_name"], team=p["team"],
                               p=p["p_game_hit"], decision=None, pick=p, pick_view=None, pick_action=pick_action,
                               file_pick=p, pick_agrees=True, pick_complete=complete, state=state,
                               finalization="pick_file_only", history=history) for p in pick_rows]
    if pick_file is not None:
        return [_day_row(date, "unfinalized_day", "pick_file_unparseable", state=state)]
    if state is not None and state["committed_pick_written"]:
        reason = "commit_flag_without_record"
    elif state is not None and state["final_skip_candidate_present"]:
        reason = "skip_intent_only"
    elif any(o["source_kind"] in ARCHIVE_KINDS for o in observations):
        reason = "archived_candidates_only"
    elif observations:
        reason = "lineup_evolution_only"
    elif state is not None:
        reason = "scheduler_ran_no_record"
    elif unusable:
        reason = "unusable_evidence_only"
    else:
        return [_day_row(date, "unobserved_day", "no_evidence")]
    return [_day_row(date, "unfinalized_day", reason, state=state)]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_rows.py -q`
Expected: `27 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/rows.py tests/scripts/season_ledger/test_rows.py
git commit -m "feat(ledger): day status, row kinds, set-level commit and history rules, delivery predicate (task 8)"
```

---

### Task 9: Routing, accounting, census, invariants, recipe rules

**Files:**
- Create: `scripts/audit/season_ledger/reconcile.py`
- Test: `tests/scripts/season_ledger/test_reconcile.py`

**Interfaces:**
- **Consumes:** Tasks 1–5 parsers.
- **Produces (routing and accounting):**
  - `route(rel_path)` and `exclusion_reason(rel_path)`
  - `PARSERS`, `KIND_DISPOSITION`, `DISPOSITIONS`
  - `account(files, routed, parsed) -> list[dict]`, with rows `{source_path, locator, obs_id, kind, state, reason, disposition, content_sha256, fields_json, record_raw_json}`; `disposition` is filled later by `assign_dispositions(accounting, dispositions)`
  - `required_locators(kind, data) -> set[str]`: every record the raw bytes hold, derived without the parsers. It is `{"file"}` for a file that is itself one record or whose container cannot be read, and the empty set only for a readable empty container.
  - `census_problems(files, routed, accounting) -> list[(problem, source_path, locator)]`, over every bundle path
  - `InvariantError`
  - `check_sources(files, routed, accounting)`: the source-phase checks, run before any ledger row exists; they include each occurrence's identity recomputed from (path, locator, content hash)
  - `check_invariants(files, accounting, matches, ledger_rows, membership, summary, season_dates)`: the output-phase checks, including recipe membership recomputed against `membership_slots(files)`
- **Produces (recipes):**
  - `RULES`, `RECIPE_KINDS`, `FILE_SETS`, `GRADED`
  - `recipe_closure() -> dict` (everything the evaluator reads) and `rules_fingerprint() -> str` (the rule table plus that closure)
  - `slot_value(doc, slot, grading)`, `counts(value, grading)`, `slot_inclusion(doc, primary_grading, leg_grading)`
  - `evaluate_rules(files, rules, mtimes, *, frozen_at) -> (summary, membership)`; each membership row names its record with `recipe_slot_key` (`<path>#<slot>`)
  - `membership_slots(files) -> set[(source_path, slot)]`: every slot of every universe record
  - `recipe_labels(summary)`

**The candidate rules and their evaluator are fixed here, before any real count (spec §8). They are pinned by `rules_fingerprint() == 5e9d74f2…`, by a pinned list of the closure's names, and by a grading truth table over every label shape. The fingerprint hashes the rule table and everything the evaluator reads (`recipe_closure`): every repo-local function reachable from `evaluate_rules` and `recipe_labels`, regexes with their flags, the time zone and tables; any value of an unknown kind fails closed.**
- **Membership universe:** every file under `picks/` whose top level has a `pick` object, any suffix, AppleDouble excluded. Every rule gets a row per universe slot, with `exclusion_reason ∈ {None, not_in_file_set, outside_window, value_not_counted}`. Task 10 links each row to the occurrence that accounts for its record: the slot, or the excluded or quarantined file that holds it.
- **Windows:** the date in the file name, falling back to the JSON `date`.
- **File sets:**
  - **F1** `picks/YYYY-MM-DD.json`.
  - **F2** the literal glob the 9/11 prose names (`data/picks/2026-*.json` → top-level `picks/2026-*.json`; adds `.shadow.json`).
  - **F3** every `picks/**/*.json` (adds scheduler archives, streak-repair `DATE.json` versions, `backup_shadow_*`).
- **Gradings:**
  - **G1:** the day `result ∈ {hit, miss}` counts a primary, and counts a leg when the file has a double-down.
  - **G2:** `slot_results.pick`/`.double_down ∈ {hit, miss}`. A file with no `slot_results` and no double-down uses its day result for the primary.
  - **G3:** as G1, but any non-null label counts.
  - **G4:** every present slot counts, ungraded included (added in rev 2 at Codex r1 #10's suggestion, before any count).
  - Recipes read the raw record, as the historical scripts did. G1 and G2 count only the string labels `hit`/`miss`, so a non-string value is never counted and never crashes; G3 counts any non-null value and G4 every present slot, as those naive rules would.
  - A slot is present when its value is not null (pinned by Codex code r1 #6, before any real count). In every production file that value is an object; a malformed non-null value is quarantined as an occurrence but still counts as a present slot for the recipes, exactly as the fingerprinted code reads it. "The file has a double-down" means its `double_down` is not null.
- **9/11 scorecard** (published 141 / 82; prose "prod pick files `data/picks/2026-*.json` … primary slot, hit/miss only … DD legs"):
  - S1–S8 = {F1, F2} × primary {G1, G2} × legs {G1, G2}, in that order.
  - Window 2026-03-29 → 2026-09-10; recipe date 2026-09-11.
  - Label `hypothesis` whatever the fit (a prose rerun).
- **9/14 naive tally** (published 191 / 157; no recorded recipe):
  - T1–T24 = {F1, F2, F3} × {G1, G2, G3, G4} (the same for both slots) × window end {2026-09-13, 2026-09-14}, in that order.
  - No lower bound; recipe date 2026-09-14.
  - Label `hypothesis` if any rule fits both totals, else `unrecoverable`.
- **Fit:** reported per rule as `both`, `primaries_only`, `legs_only` or `none`. Each fitting rule reads `matches published totals; historical membership unverified`.

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_reconcile.py`:

```python
import json

import pytest

from scripts.audit.season_ledger.ids import Parsed
from scripts.audit.season_ledger.reconcile import (PARSERS, RULES, InvariantError, account, assign_dispositions,
                                                   census_problems, check_invariants, check_sources, evaluate_rules,
                                                   exclusion_reason, recipe_closure, recipe_labels, route,
                                                   rules_fingerprint, slot_inclusion)
from tests.scripts.season_ledger.builders import cand, contest_line, decision_json, dumps, gz, pick_json, rnd, slot

# The recipe candidates frozen by this plan (Task 9 text). Recompute only by editing the plan before the real run.
RULES_SHA256 = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"
# Everything the recipe evaluation reads (reconcile.recipe_closure): a new dependency must show up here.
RECIPE_CLOSURE = [
    "callable:datetime.datetime",
    "module:gzip",
    "module:json",
    "module:re",
    "module:zlib",
    "scripts.audit.season_ledger.ids.load_json_bytes",
    "scripts.audit.season_ledger.reconcile.FILE_SETS",
    "scripts.audit.season_ledger.reconcile.GRADED",
    "scripts.audit.season_ledger.reconcile.RECIPE_KINDS",
    "scripts.audit.season_ledger.reconcile._APPLEDOUBLE",
    "scripts.audit.season_ledger.reconcile._ET",
    "scripts.audit.season_ledger.reconcile._FILE_DATE",
    "scripts.audit.season_ledger.reconcile._doc",
    "scripts.audit.season_ledger.reconcile._mtime_after",
    "scripts.audit.season_ledger.reconcile._natural",
    "scripts.audit.season_ledger.reconcile._universe",
    "scripts.audit.season_ledger.reconcile.counts",
    "scripts.audit.season_ledger.reconcile.evaluate_rules",
    "scripts.audit.season_ledger.reconcile.recipe_labels",
    "scripts.audit.season_ledger.reconcile.slot_inclusion",
    "scripts.audit.season_ledger.reconcile.slot_value",
]


@pytest.mark.parametrize("path,kind", [
    ("picks/2026-05-01.json", "pick_file"), ("picks/2026-05-01/decision.json", "decision"),
    ("picks/2026-05-01/scheduler_state.json", "scheduler_state"),
    ("picks/2026-08-30/deferred_fallback_20260830T120000-0400.json", "archive"),
    ("picks/archive/2026-04-11.json.postponed", "manual_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json", "repair_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z_missed_2/2026-05-24.json.before", "repair_archive"),
    ("picks/lineup_evolution_2026-05-01.jsonl", "lineup_evolution"),
    ("picks/account_state/contest_ledger.jsonl", "contest_ledger"),
    ("picks/account_state/saver_transitions.jsonl", "saver_transitions"),
    ("static/rounds/20260704T030011Z.json", "rounds"), ("static/units/20260801T150000Z.json.gz", "units"),
    ("static/grab_20260927/002_players.json.gz", "players"), ("schedules/2026-05-01.json", "schedule")])
def test_routes(path, kind):
    assert route(path) == kind


@pytest.mark.parametrize("path,reason", [
    ("picks/._2026-05-01.json", "appledouble_resource_fork"), ("static/rounds/._x.json", "appledouble_resource_fork"),
    ("picks/2026-05-01.shadow.json", "shadow_model_out_of_scope"),
    ("picks/backup_shadow_2026-05-09/2026-05-08.shadow.json", "shadow_model_out_of_scope"),
    ("picks/2026-05-01.policy_shadow.json", "skip_policy_shadow_out_of_scope"),
    ("picks/slates/2026-05-01.json", "slate_binding_out_of_scope"),
    ("picks/streak.json", "state_snapshot_not_a_record"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/streak.before.json", "state_snapshot_not_a_record"),
    ("picks/account_state/contest_streak.manual.json.archived_post_auto_20260601T000000Z", "state_snapshot_not_a_record"),
    ("picks/archive_replay_restore_20260601T000000Z_post_contest_state_deploy/README.txt", "documentation"),
    ("picks/.nrestarts_checkpoint", "runtime_marker"), ("logs/cron.log", "corroboration_only"),
    ("static/grab_20260927/004_squads.json.gz", "not_used_phase1"), ("picks/mystery.bin", "unrecognized_path")])
def test_exclusions(path, reason):
    assert route(path) is None and exclusion_reason(path) == reason


def _parse(files, routed):
    return {rel: PARSERS[kind](rel, files[rel]) for rel, kind in routed.items()}


def test_every_file_is_accounted_with_its_locator_fields_and_disposition():
    files = {"picks/2026-05-01.json": pick_json("2026-05-01"), "picks/._2026-05-01.json": b"\x00\x05",
             "picks/2026-05-02.json": b"{", "schedules/2026-05-02.json": None}
    routed = {"picks/2026-05-01.json": "pick_file", "picks/2026-05-02.json": "pick_file"}
    parsed = _parse(files, routed)
    obs = parsed["picks/2026-05-01.json"].rows[0]["obs_id"]
    acc = account(files, routed, parsed)
    assign_dispositions(acc, {obs: "canonical_selection"})
    assert {(a["source_path"], a["locator"], a["state"], a["reason"] or a["disposition"]) for a in acc} == {
        ("picks/2026-05-01.json", "slot=primary", "emitted", "canonical_selection"),
        ("picks/._2026-05-01.json", "file", "excluded", "appledouble_resource_fork"),
        ("picks/2026-05-02.json", "file", "quarantined", "invalid_json:JSONDecodeError"),
        ("schedules/2026-05-02.json", "file", "declared_missing", "declared_missing")}
    (emitted,) = [a for a in acc if a["state"] == "emitted"]
    assert emitted["obs_id"] == obs and json.loads(emitted["fields_json"])["batter_id"] == 101
    assert json.loads(emitted["record_raw_json"])["pick"]["batter_id"] == 101
    check_sources(files, routed, acc)


def test_census_requires_every_record_exactly_once():
    # Codex plan r1 #1 and r2 #1: omitted, phantom, empty-parser, dropped line record, foreign path, double cover.
    pick_rel, contest_rel = "picks/2026-05-02.json", "picks/account_state/contest_ledger.jsonl"
    line = contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")]), rnd(972, "void", 0, 0, [])])
    files = {pick_rel: pick_json("2026-05-02", dd={}), contest_rel: (line + "\n").encode()}
    routed = {pick_rel: "pick_file", contest_rel: "contest_ledger"}
    good = account(files, routed, _parse(files, routed))
    assert census_problems(files, routed, good) == []

    def problems(acc):
        return [(p, loc) for p, _rel, loc in census_problems(files, routed, acc)]

    assert problems([a for a in good if a["locator"] != "slot=double_down"]) == [("omitted", "slot=double_down")]
    assert problems(good + [dict(good[0], locator="slot=triple")]) == [("phantom_locator", "slot=triple")]
    empty = account(files, routed, {pick_rel: Parsed(), contest_rel: _parse(files, routed)[contest_rel]})
    assert problems(empty) == [("exclusion_over_records", "file"), ("omitted", "slot=double_down"),
                               ("omitted", "slot=primary")]
    assert problems([a for a in good if a["locator"] != "line=1"]) == [("omitted", "line=1")]
    assert census_problems(files, routed, good + [dict(good[0], source_path="picks/phantom.json")]) == [
        ("source_not_in_bundle", "picks/phantom.json", "file")]
    overlap = good + [dict(good[-1], locator="file", state="excluded", reason="no_records")]
    assert ("exclusion_over_records", "file") in problems(overlap)


def test_a_quarantined_ancestor_covers_its_records_and_double_cover_fails():
    rel = "picks/account_state/contest_ledger.jsonl"
    good = contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])])
    bad = contest_line("2026-08-22T14:30:00Z", [rnd(973, "hit", 1, 1, [slot(1930, None, "hit")])])
    files, routed = {rel: (good + "\n" + bad + "\n").encode()}, {rel: "contest_ledger"}
    acc = account(files, routed, _parse(files, routed))
    assert census_problems(files, routed, acc) == []          # line 2's round and slot sit under its quarantine
    leaked = acc + [dict(acc[0], locator="line=2/round=0/slot=0", state="emitted")]
    assert [p for p, _r, loc in census_problems(files, routed, leaked)] == ["covered_twice"]


def _acc(*pairs, kind="pick_file", disposition="not_selected"):
    return [{"source_path": p, "locator": loc, "obs_id": f"o:{p}:{loc}", "state": "emitted", "kind": kind,
             "disposition": disposition} for p, loc in pairs]


def _day(date, kind="unobserved_day", **refs):
    return {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None, **refs}


def test_source_checks_catch_unaccounted_and_duplicated_files():
    files = {"a": b"1", "b": b"2"}
    acc = account(files, {}, {})
    check_sources(files, {}, acc)
    with pytest.raises(InvariantError, match="not accounted"):
        check_sources(files, {}, acc[:1])
    with pytest.raises(InvariantError, match="duplicate occurrence"):
        check_sources(files, {}, acc + acc[:1])
    rel = "picks/2026-05-02.json"
    pf, routed = {rel: pick_json("2026-05-02", dd={})}, {rel: "pick_file"}
    full = account(pf, routed, _parse(pf, routed))
    with pytest.raises(InvariantError, match="census: \\[\\('omitted'"):
        check_sources(pf, routed, [a for a in full if a["locator"] != "slot=double_down"])


def test_source_checks_reject_the_codex_r3_census_probes():
    # Codex plan r3 #1: an empty single-record parse, a record relabelled missing, a row beneath an excluded path and
    # a forged occurrence id must each fail before any ledger row exists.
    dec_rel, contest_rel = "picks/2026-08-10/decision.json", "picks/account_state/contest_ledger.jsonl"
    shadow_rel, manual_rel = "picks/2026-08-10.shadow.json", "picks/archive/2026-08-10.json.postponed"
    files = {dec_rel: decision_json("2026-08-10", action="single", primary=cand(101, 5001)),
             contest_rel: (contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])])
                           + "\n").encode(),
             shadow_rel: pick_json("2026-08-10"), manual_rel: pick_json("2026-08-10", dd={})}
    routed = {rel: route(rel) for rel in files if route(rel)}
    parsed = _parse(files, routed)
    good = account(files, routed, parsed)
    check_sources(files, routed, good)
    with pytest.raises(InvariantError, match="exclusion_over_records"):
        check_sources(files, routed, account(files, routed, {**parsed, dec_rel: Parsed()}))
    relabelled = [dict(a, state="declared_missing", reason="declared_missing")
                  if (a["source_path"], a["locator"]) == (contest_rel, "line=1") else a for a in good]
    with pytest.raises(InvariantError, match="declared_missing_over_records"):
        check_sources(files, routed, relabelled)
    shadow = next(a for a in good if a["source_path"] == shadow_rel)
    with pytest.raises(InvariantError, match="file_accounting_shape"):
        check_sources(files, routed, good + [dict(shadow, locator="slot=phantom")])
    primary_id = next(a["obs_id"] for a in good if (a["source_path"], a["locator"]) == (manual_rel, "slot=primary"))
    forged = [dict(a, obs_id=primary_id) if (a["source_path"], a["locator"]) == (manual_rel, "slot=double_down")
              else a for a in good]
    with pytest.raises(InvariantError, match="identity does not match"):
        check_sources(files, routed, forged)


def test_only_a_readable_empty_container_is_a_file_without_records():
    empty, unreadable = "static/units/20260927T230002Z.json.gz", "static/units/20260926T230002Z.json"
    files = {empty: gz(dumps({"units": []})), unreadable: dumps({"error": "x"})}
    routed = {empty: "units", unreadable: "units"}
    parsed = _parse(files, routed)
    acc = account(files, routed, parsed)
    assert {(a["source_path"], a["state"], a["reason"]) for a in acc} == {
        (empty, "excluded", "no_records"), (unreadable, "quarantined", "missing_units_list")}
    check_sources(files, routed, acc)
    with pytest.raises(InvariantError, match="exclusion_over_records"):
        check_sources(files, routed, account(files, routed, {**parsed, unreadable: Parsed()}))


def _link(membership, accounting, ledger_rows):
    """The join the compiler performs (Task 10), written out plainly for these tests."""
    by_loc = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical = {r["pick_obs_id"]: r for r in ledger_rows if r.get("pick_obs_id")}
    out = []
    for m in membership:
        src = by_loc.get((m["source_path"], f"slot={m['slot']}")) or by_loc[(m["source_path"], "file")]
        link = canonical.get(src["obs_id"])
        out.append(dict(m, source_obs_id=src["obs_id"], source_state=src["state"], source_reason=src["reason"],
                        occurrence_disposition=src["disposition"], selection_id=link["selection_id"] if link else None,
                        canonical_bts_outcome=link.get("bts_outcome") if link else None))
    return out


def test_membership_checks_prove_the_join_the_cardinality_and_the_totals():
    # Codex plan r3 #3: a real occurrence id is not enough — it must be the right one, every rule must list every
    # universe slot exactly once, and the included rows must add up to the reported totals.
    prod, shadow = "picks/2026-08-20.json", "picks/2026-08-20.shadow.json"
    files = {prod: pick_json("2026-08-20", result="hit"), shadow: pick_json("2026-08-20", result="hit")}
    routed = {prod: "pick_file"}
    acc = account(files, routed, _parse(files, routed))
    assign_dispositions(acc, {a["obs_id"]: "not_selected" for a in acc if a["kind"] == "pick_file"})
    summary, membership = evaluate_rules(files, RULES, {}, frozen_at="t")
    linked = _link(membership, acc, [])
    check_invariants(files, acc, [], [], linked, summary, [])
    shadow_file = next(a for a in acc if a["source_path"] == shadow)
    wrong = [dict(m, source_obs_id=shadow_file["obs_id"], source_state="excluded", source_reason=shadow_file["reason"],
                  occurrence_disposition=None) if m["source_path"] == prod else m for m in linked]
    with pytest.raises(InvariantError, match="not linked to the occurrence"):
        check_invariants(files, acc, [], [], wrong, summary, [])
    with pytest.raises(InvariantError, match="exactly once"):
        check_invariants(files, acc, [], [], linked[1:], summary, [])
    with pytest.raises(InvariantError, match="exactly once"):
        check_invariants(files, acc, [], [], linked + linked[:1], summary, [])
    inflated = [dict(s, primaries=s["primaries"] + 1) if s["rule_id"] == "S1" else s for s in summary]
    with pytest.raises(InvariantError, match="add up"):
        check_invariants(files, acc, [], [], linked, inflated, [])


def test_output_checks_catch_bad_dispositions_references_joins_links_and_days():
    days, ok = ["2026-05-01"], [_day("2026-05-01")]
    acc = _acc(("a", "file")) + _acc(("d", "file"), kind="decision", disposition="canonical_decision")
    check_invariants({}, acc, [], [_day("2026-05-01", decision_obs_id="o:d:file")], [], [], days)
    with pytest.raises(InvariantError, match="without a disposition"):
        check_invariants({}, [dict(acc[0], disposition=None)], [], ok, [], [], days)
    with pytest.raises(InvariantError, match="unknown disposition"):
        check_invariants({}, [dict(acc[0], disposition="whatever")], [], ok, [], [], days)
    with pytest.raises(InvariantError, match="is not an emitted decision"):
        check_invariants({}, acc, [], [_day("2026-05-01", decision_obs_id="o:a:file")], [], [], days)   # wrong kind
    with pytest.raises(InvariantError, match="referenced by no ledger row"):
        check_invariants({}, acc, [], ok, [], [], days)
    canonical = _acc(("p", "slot=primary"), disposition="canonical_selection")
    with pytest.raises(InvariantError, match="exactly one ledger row"):
        check_invariants({}, canonical, [], ok, [], [], days)
    two = [{"selection_id": "s1", "round_id": 1, "unit_id": 1, "player_id": 1, "match": "inferred"},
           {"selection_id": "s1", "round_id": 1, "unit_id": 2, "player_id": 1, "match": "inferred"}]
    with pytest.raises(InvariantError, match="more than one contest slot"):
        check_invariants({}, _acc(("a", "file")), two, ok, [], [], days)
    with pytest.raises(InvariantError, match="no ledger row"):
        check_invariants({}, _acc(("a", "file")), [], [], [], [], days)
    sel = {"row_id": "2026-05-01|primary|1|2", "row_kind": "selection", "date": "2026-05-01", "slot": "primary"}
    with pytest.raises(InvariantError, match="mixes"):
        check_invariants({}, _acc(("a", "file")), [], [_day("2026-05-01"), sel], [], [], days)


def _rule(files, grading, published):
    return {"recipe": "tally_0914", "files": files, "primary": grading, "legs": grading,
            "window": ("2026-03-29", "2026-09-13"), "published": published, "recipe_date": "2026-09-14"}


def test_two_rules_with_equal_totals_and_different_members_are_both_reported():
    # Codex plan r1 #10: the alternatives must include different records, not only an ungraded extra file.
    files = {"picks/2026-05-01.json": pick_json("2026-05-01", result="hit"),
             "picks/2026-05-02.json": pick_json("2026-05-02", dd={}, result="miss", slot_results={"double_down": "hit"}),
             "picks/2026-05-03.json": pick_json("2026-05-03", slot_results={"pick": "hit"})}
    summary, membership = evaluate_rules(files, {"A": _rule("F1", "G1", (2, 1)), "B": _rule("F1", "G2", (2, 1))}, {},
                                         frozen_at="2026-09-28T16:00:00.000000Z")
    assert {s["rule_id"]: s["fit"] for s in summary} == {"A": "both", "B": "both"}
    members = {r: {m["recipe_slot_key"] for m in membership if m["rule_id"] == r and m["included"]} for r in "AB"}
    assert members["A"] != members["B"] and len(members["A"]) == len(members["B"]) == 3


def test_recipe_labels_follow_the_spec():
    summary = [{"recipe": "scorecard_0911", "fit": "none"}, {"recipe": "tally_0914", "fit": "primaries_only"},
               {"recipe": "tally_0914", "fit": "none"}]
    assert recipe_labels(summary) == {"scorecard_0911": "hypothesis", "tally_0914": "unrecoverable"}
    assert recipe_labels(summary + [{"recipe": "tally_0914", "fit": "both"}])["tally_0914"] == "hypothesis"


def test_candidate_rules_and_their_evaluator_are_fingerprinted():
    # Codex plan r2 #6 and r3 #4: the digest binds the rule table and everything the evaluator reads — the source of
    # every repo-local function reachable from the recipe roots and every configuration value, regex flags included.
    assert rules_fingerprint() == RULES_SHA256
    assert sorted(recipe_closure()) == RECIPE_CLOSURE
    assert sorted(RULES, key=lambda k: (k[0], int(k[1:]))) == [f"S{i}" for i in range(1, 9)] + [
        f"T{i}" for i in range(1, 25)]


def test_the_fingerprint_binds_regex_flags_and_fails_closed_on_unknown_configuration(monkeypatch):
    import re

    from scripts.audit.season_ledger import reconcile
    monkeypatch.setitem(reconcile.FILE_SETS, "F1", re.compile(reconcile.FILE_SETS["F1"].pattern, re.IGNORECASE))
    assert rules_fingerprint() != RULES_SHA256
    monkeypatch.setattr(reconcile, "GRADED", object())
    with pytest.raises(TypeError, match="cannot bind"):
        rules_fingerprint()


def test_grading_truth_table_covers_every_label_shape():
    labels = ("hit", "miss", "void", "suspended", "unresolved", None, {"odd": 1})
    single = {label if isinstance(label, (str, type(None))) else "object": {g: slot_inclusion(
        {"pick": {}, "double_down": None, "result": label}, g, g)["primary"][1] for g in ("G1", "G2", "G3", "G4")}
        for label in labels}
    assert single == {
        "hit": {"G1": True, "G2": True, "G3": True, "G4": True}, "miss": {"G1": True, "G2": True, "G3": True, "G4": True},
        "void": {"G1": False, "G2": False, "G3": True, "G4": True},
        "suspended": {"G1": False, "G2": False, "G3": True, "G4": True},
        "unresolved": {"G1": False, "G2": False, "G3": True, "G4": True},
        None: {"G1": False, "G2": False, "G3": False, "G4": True},
        "object": {"G1": False, "G2": False, "G3": True, "G4": True}}
    dd_slots = {"pick": {}, "double_down": {}, "result": "miss", "slot_results": {"pick": "void", "double_down": "hit"}}
    dd_bare = {"pick": {}, "double_down": {}, "result": "miss"}
    assert {g: {s: c for s, (_, c) in slot_inclusion(dd_slots, g, g).items()} for g in ("G1", "G2", "G3", "G4")} == {
        "G1": {"primary": True, "double_down": True}, "G2": {"primary": False, "double_down": True},
        "G3": {"primary": True, "double_down": True}, "G4": {"primary": True, "double_down": True}}
    assert {g: {s: c for s, (_, c) in slot_inclusion(dd_bare, g, g).items()} for g in ("G1", "G2", "G3", "G4")} == {
        "G1": {"primary": True, "double_down": True}, "G2": {"primary": False, "double_down": False},
        "G3": {"primary": True, "double_down": True}, "G4": {"primary": True, "double_down": True}}


def test_membership_universe_records_exclusions_and_the_evidence_interval():
    files = {f"picks/2026-05-0{i}.json": pick_json(f"2026-05-0{i}", result="hit") for i in (1, 2, 3)}
    files.update({"picks/2026-09-12.json": pick_json("2026-09-12", result="hit"),
                  "picks/2026-05-04.shadow.json": pick_json("2026-05-04", result="hit"),
                  "picks/2026-05-05.json": pick_json("2026-05-05", result="void"),
                  "picks/2026-05-06.json": pick_json("2026-05-06", result={"odd": 1})})   # never crashes a recipe
    mtimes = {"picks/2026-05-01.json": "2026-05-02T03:00:00.000000Z",    # 5/01 ET: before the 9/11 recipe
              "picks/2026-05-02.json": "2026-09-11T16:00:00.000000Z",    # same ET day: order unknown
              "picks/2026-05-03.json": "2026-09-12T16:00:00.000000Z"}    # after
    rule = {"R": {"recipe": "scorecard_0911", "files": "F1", "primary": "G1", "legs": "G1",
                  "window": ("2026-03-29", "2026-09-10"), "published": (3, 0), "recipe_date": "2026-09-11"}}
    summary, membership = evaluate_rules(files, rule, mtimes, frozen_at="2026-09-28T16:00:00.000000Z")
    by_path = {m["source_path"]: m for m in membership}
    assert {p: m["exclusion_reason"] for p, m in by_path.items()} == {
        "picks/2026-05-01.json": None, "picks/2026-05-02.json": None, "picks/2026-05-03.json": None,
        "picks/2026-09-12.json": "outside_window", "picks/2026-05-04.shadow.json": "not_in_file_set",
        "picks/2026-05-05.json": "value_not_counted", "picks/2026-05-06.json": "value_not_counted"}
    assert [by_path[f"picks/2026-05-0{i}.json"]["mtime_after_recipe_date"] for i in (1, 2, 3)] == [False, None, True]
    assert {(m["historical_membership"], m["frozen_at_utc"]) for m in membership} == {
        ("unknown", "2026-09-28T16:00:00.000000Z")}
    assert summary[0]["fit"] == "both"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_reconcile.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.reconcile'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/reconcile.py`:

```python
"""Path routing, occurrence accounting with an independent structural census (the anti-join), build
invariants, and the pre-declared recipe rules (spec §8)."""
from __future__ import annotations

import inspect
import itertools
import json
import re
import types
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from .ids import Parsed, canonical_json, load_json_bytes, obs_id, sha256_hex
from .sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from .sources.day_records import parse_decision, parse_lineup_evolution, parse_scheduler_state
from .sources.pick_files import parse_archive, parse_pick_file
from .sources.static import parse_players, parse_rounds, parse_schedule, parse_units

_D = r"\d{4}-\d{2}-\d{2}"
_APPLEDOUBLE = re.compile(r"(^|/)\._[^/]*$")
ROUTES = [(re.compile(p), kind) for p, kind in [
    (rf"^picks/{_D}\.json$", "pick_file"),
    (rf"^picks/{_D}/decision\.json$", "decision"),
    (rf"^picks/{_D}/scheduler_state\.json$", "scheduler_state"),
    (rf"^picks/{_D}/(deferred_fallback|refused_delivery|stale_pick)_[^/]+\.json$", "archive"),
    (rf"^picks/archive/{_D}\.json\.postponed$", "manual_archive"),
    (rf"^picks/archive_actual_streak_repair_[^/]+/{_D}\.json(\.before)?$", "repair_archive"),
    (rf"^picks/lineup_evolution_{_D}\.jsonl$", "lineup_evolution"),
    (r"^picks/account_state/contest_ledger\.jsonl$", "contest_ledger"),
    (r"^picks/account_state/saver_transitions\.jsonl$", "saver_transitions"),
    (r"^static/rounds/[^/]+$", "rounds"),
    (r"^static/units/[^/]+$", "units"),
    (r"^static/players/[^/]+$", "players"),
    (r"^static/grab_20260927/[^/]*rounds[^/]*$", "rounds"),
    (r"^static/grab_20260927/[^/]*units[^/]*$", "units"),
    (r"^static/grab_20260927/[^/]*players[^/]*$", "players"),
    (rf"^schedules/{_D}\.json$", "schedule"),
]]
EXCLUSIONS = [(re.compile(p), reason) for p, reason in [
    (r"(^|/)\._[^/]*$", "appledouble_resource_fork"),
    (rf"^picks/{_D}\.shadow\.json$", "shadow_model_out_of_scope"),
    (r"^picks/backup_shadow_[^/]+/", "shadow_model_out_of_scope"),
    (rf"^picks/{_D}\.policy_shadow\.json$", "skip_policy_shadow_out_of_scope"),
    (r"^picks/slates/", "slate_binding_out_of_scope"),
    (r"(^|/)streak(\.before)?\.json$", "state_snapshot_not_a_record"),
    (r"^picks/account_state/(contest_streak|saver_state)[^/]*$", "state_snapshot_not_a_record"),
    (r"(^|/)README\.txt$", "documentation"),
    (r"^picks/\.nrestarts_checkpoint$", "runtime_marker"),
    (r"^logs/", "corroboration_only"),
    (r"^static/grab_20260927/[^/]*squads[^/]*$", "not_used_phase1"),
]]
PARSERS = {
    "pick_file": parse_pick_file, "archive": parse_archive,
    "manual_archive": lambda rel, data: parse_pick_file(rel, data, kind="manual_archive"),
    "repair_archive": lambda rel, data: parse_pick_file(rel, data, kind="repair_archive"),
    "decision": parse_decision, "scheduler_state": parse_scheduler_state,
    "lineup_evolution": parse_lineup_evolution, "contest_ledger": parse_contest_ledger,
    "saver_transitions": parse_saver_transitions, "rounds": parse_rounds, "players": parse_players,
    "units": parse_units, "schedule": parse_schedule,
}
KIND_DISPOSITION = {"archive": "history_evidence", "manual_archive": "history_evidence",
                    "repair_archive": "history_evidence", "lineup_evolution": "history_evidence",
                    "scheduler_state": "day_evidence", "contest_ledger": "contest_evidence",
                    "saver_transitions": "reported_attempt", "rounds": "lookup", "players": "lookup",
                    "units": "lookup", "schedule": "lookup"}   # pick_file / decision come from the ledger
DISPOSITIONS = frozenset(KIND_DISPOSITION.values()) | {"canonical_selection", "unresolved_pick_file_view",
                                                       "not_selected", "outside_season_window", "canonical_decision"}
_OWN_COLUMNS = frozenset({"obs_id", "locator", "source_path", "content_sha256", "record_raw_json"})
_PICK_LIKE = frozenset({"pick_file", "archive", "manual_archive", "repair_archive"})
_ITEM_KEYS = {"rounds": "rounds", "players": "players", "units": "units"}


def route(rel_path: str) -> str | None:
    if _APPLEDOUBLE.search(rel_path):
        return None
    for pattern, kind in ROUTES:
        if pattern.match(rel_path):
            return kind
    return None


def exclusion_reason(rel_path: str) -> str:
    for pattern, reason in EXCLUSIONS:
        if pattern.search(rel_path):
            return reason
    return "unrecognized_path"


def account(files: dict[str, bytes | None], routed: dict[str, str], parsed: dict[str, Parsed]) -> list[dict]:
    """Spec §8 table 1: every occurrence ends emitted (with its parsed fields and raw record), excluded(reason),
    quarantined(reason, with its raw record when it had one) or declared_missing. Dispositions are assigned
    after the ledger exists (`assign_dispositions`)."""
    out = []
    for rel, data in files.items():
        kind = routed.get(rel)
        sha = None if data is None else sha256_hex(data)
        blank = {"source_path": rel, "kind": kind, "content_sha256": sha, "disposition": None,
                 "fields_json": None, "record_raw_json": None}
        if data is None:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", "missing"),
                        "state": "declared_missing", "reason": "declared_missing"})
            continue
        if kind is None:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", sha), "state": "excluded",
                        "reason": exclusion_reason(rel)})
            continue
        result = parsed[rel]
        out += [{**blank, "locator": q["locator"], "obs_id": obs_id(rel, q["locator"], sha), "state": "quarantined",
                 "reason": q["reason"], "record_raw_json": q.get("record_raw_json")} for q in result.quarantined]
        out += [{**blank, "locator": r["locator"], "obs_id": r["obs_id"], "state": "emitted", "reason": None,
                 "fields_json": canonical_json({k: v for k, v in r.items() if k not in _OWN_COLUMNS}),
                 "record_raw_json": r.get("record_raw_json")} for r in result.rows]
        if not result.rows and not result.quarantined:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", sha), "state": "excluded",
                        "reason": "no_records"})
    return out


def assign_dispositions(accounting: list[dict], dispositions: dict[str, str]) -> None:
    for a in accounting:
        if a["state"] == "emitted":
            a["disposition"] = dispositions.get(a["obs_id"]) or KIND_DISPOSITION.get(a["kind"])


def _doc(data: bytes):
    try:
        return load_json_bytes(data)
    except ValueError:
        return None


def _jsonl(data: bytes) -> list[tuple[int, object]] | None:
    """(line number, parsed value or None) for every non-blank line; None when the bytes are not UTF-8."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                out.append((n, json.loads(line)))
            except json.JSONDecodeError:
                out.append((n, None))
    return out


def required_locators(kind: str, data: bytes) -> set[str]:
    """An independent structural census of the records one file holds, walked from its containers without type
    checks (spec §8; Codex plan r3 #1). A file that is itself one record (a decision, a scheduler state, a pick
    file without slots), or whose container cannot be read, holds exactly one record: `file`. Only a readable
    container with no entries holds none — the one case `no_records` may describe."""
    if kind in _PICK_LIKE:
        doc = _doc(data)
        slots = {f"slot={slot}" for slot, key in (("primary", "pick"), ("double_down", "double_down"))
                 if isinstance(doc, dict) and doc.get(key) is not None}
        return slots or {"file"}
    required: set[str] = set()
    if kind in ("lineup_evolution", "saver_transitions", "contest_ledger"):
        lines = _jsonl(data)
        if lines is None:
            return {"file"}
        for n, doc in lines:
            if kind == "lineup_evolution":
                slots = {f"line={n}/slot={s}" for s in ("primary", "double_down")
                         if isinstance(doc, dict) and doc.get(s) is not None}
                required |= slots or {f"line={n}"}
                continue
            required.add(f"line={n}")
            if kind == "contest_ledger" and isinstance(doc, dict) and isinstance(doc.get("predictions"), list):
                for i, rnd in enumerate(doc["predictions"]):
                    required.add(f"line={n}/round={i}")
                    slots = rnd.get("roundPredictions") if isinstance(rnd, dict) else None
                    if isinstance(slots, list):
                        required |= {f"line={n}/round={i}/slot={j}" for j in range(len(slots))}
        return required
    if kind in _ITEM_KEYS:
        doc = _doc(data)
        items = doc.get(_ITEM_KEYS[kind]) if isinstance(doc, dict) else None
        return {f"item={i}" for i in range(len(items))} if isinstance(items, list) else {"file"}
    if kind == "schedule":
        doc = _doc(data)
        dates = doc.get("dates") if isinstance(doc, dict) else None
        if not isinstance(dates, list):
            return {"file"}
        for i, day in enumerate(dates):
            games = day.get("games") if isinstance(day, dict) else None
            required |= ({f"date={i}/game={j}" for j in range(len(games))} if isinstance(games, list) and games
                         else {f"date={i}"})
        return required
    return {"file"}


def _ancestors(locator: str) -> list[str]:
    if locator == "file":
        return []
    parts = locator.split("/")
    return ["file"] + ["/".join(parts[:k]) for k in range(1, len(parts))]


_OVER_RECORDS = {"excluded": "exclusion_over_records", "declared_missing": "declared_missing_over_records"}


def census_problems(files: dict, routed: dict[str, str], accounting: list[dict]) -> list[tuple[str, str, str]]:
    """(problem, source_path, locator) for every breach of exactly-once accounting (Codex plan r2 #1, r3 #1), over
    EVERY bundle path. A declared-missing input, an excluded path and a readable empty container each have exactly
    one `file` row in that state, with its reason. Every other file accounts for each census record exactly once —
    emitted or quarantined at its locator, or under one quarantined ancestor — with no row that is not a record or
    an ancestor, no row in any other state, and no emitted row that is not a record."""
    by_file: dict[str, list[dict]] = {}
    for a in accounting:
        by_file.setdefault(a["source_path"], []).append(a)
    problems = [("source_not_in_bundle", rel, "file") for rel in sorted(set(by_file) - set(files))]
    for rel in sorted(files):
        rows, data, kind = by_file.get(rel, []), files[rel], routed.get(rel)
        required = set() if data is None or kind is None else required_locators(kind, data)
        if not required:
            want = (("declared_missing", "declared_missing") if data is None else
                    ("excluded", exclusion_reason(rel)) if kind is None else ("excluded", "no_records"))
            if [(a["locator"], a["state"], a["reason"]) for a in rows] != [("file", *want)]:
                problems.append(("file_accounting_shape", rel, "file"))
            continue
        allowed = required | {anc for loc in required for anc in _ancestors(loc)}
        accounted = {a["locator"]: a for a in rows}
        for loc, a in sorted(accounted.items()):
            if loc not in allowed:
                problems.append(("phantom_locator", rel, loc))
            elif a["state"] not in ("emitted", "quarantined"):
                problems.append((_OVER_RECORDS.get(a["state"], "unknown_state"), rel, loc))
            elif a["state"] == "emitted" and loc not in required:
                problems.append(("emitted_off_record", rel, loc))
        for loc in sorted(required):
            covers = (loc in accounted) + sum(accounted.get(anc, {}).get("state") == "quarantined"
                                              for anc in _ancestors(loc))
            if covers != 1:
                problems.append(("omitted" if covers == 0 else "covered_twice", rel, loc))
    return problems


class InvariantError(Exception):
    pass


def check_sources(files: dict, routed: dict[str, str], accounting: list[dict]) -> None:
    """Source-phase build failures, run before any canonical row exists: an unaccounted file, a duplicated
    occurrence, any census problem, or an occurrence whose identity does not match its source — its obs_id
    recomputed from (path, locator, content hash), its content hash or its kind (Codex plan r3 #1). Recomputed ids
    are also unique, because (path, locator) pairs are."""
    missing = sorted(set(files) - {a["source_path"] for a in accounting})
    if missing:
        raise InvariantError(f"files not accounted: {missing[:5]}")
    dupes = [k for k, n in Counter((a["source_path"], a["locator"]) for a in accounting).items() if n > 1]
    if dupes:
        raise InvariantError(f"duplicate occurrence rows: {dupes[:5]}")
    problems = census_problems(files, routed, accounting)
    if problems:
        raise InvariantError(f"census: {problems[:5]}")
    for a in accounting:
        data = files[a["source_path"]]
        sha = None if data is None else sha256_hex(data)
        want = (obs_id(a["source_path"], a["locator"], "missing" if sha is None else sha), sha, routed.get(a["source_path"]))
        if (a["obs_id"], a["content_sha256"], a["kind"]) != want:
            raise InvariantError(f"occurrence identity does not match its source: {a['source_path']} {a['locator']}")


_REF_KINDS = {"pick_obs_id": "pick_file", "pick_view_obs_id": "pick_file", "decision_obs_id": "decision",
              "state_obs_id": "scheduler_state", "contest_obs_id": "contest_ledger"}


def check_invariants(files: dict, accounting: list[dict], matches: list[dict], ledger_rows: list[dict],
                     membership: list[dict], summary: list[dict], season_dates: list[str]) -> None:
    """Output-phase build failures (spec §8): an undisposed or unknown disposition; a ledger reference to an
    occurrence that was not emitted or is of the wrong kind; a canonical disposition no ledger row references;
    recipe membership that does not list every rule × universe slot exactly once, does not add up to the reported
    totals, or links a row to anything but the occurrence that accounts for its record (Codex plan r3 #3); a
    contest slot identity twice; a selection linked to two contest slots; duplicate row ids; a malformed season
    day."""
    undisposed = [(a["source_path"], a["locator"]) for a in accounting
                  if a["state"] == "emitted" and not a["disposition"]]
    if undisposed:
        raise InvariantError(f"emitted occurrences without a disposition: {undisposed[:5]}")
    unknown = sorted({a["disposition"] for a in accounting if a["disposition"]} - DISPOSITIONS)
    if unknown:
        raise InvariantError(f"unknown disposition values: {unknown}")
    emitted = {a["obs_id"]: a for a in accounting if a["state"] == "emitted"}
    referenced = Counter()
    for r in ledger_rows:
        for column, kind in _REF_KINDS.items():
            value = r.get(column)
            if value is None:
                continue
            if value not in emitted or emitted[value]["kind"] != kind:
                raise InvariantError(f"ledger row {r['row_id']} {column} is not an emitted {kind} occurrence")
            referenced[(column, value)] += 1
    for a in emitted.values():
        if a["disposition"] == "canonical_selection" and referenced[("pick_obs_id", a["obs_id"])] != 1:
            raise InvariantError(f"canonical selection {a['obs_id']} is not referenced by exactly one ledger row")
        if a["disposition"] == "canonical_decision" and not referenced[("decision_obs_id", a["obs_id"])]:
            raise InvariantError(f"canonical decision {a['obs_id']} is referenced by no ledger row")
    keys = Counter((m["rule_id"], m["source_path"], m["slot"]) for m in membership)
    if set(keys) != {(s["rule_id"], rel, slot) for s in summary for rel, slot in membership_slots(files)} \
            or any(n > 1 for n in keys.values()):
        raise InvariantError("recipe membership does not list every rule and universe slot exactly once")
    for s in summary:
        included = [m["slot"] for m in membership if m["rule_id"] == s["rule_id"] and m["included"]]
        if (included.count("primary"), included.count("double_down")) != (s["primaries"], s["legs"]):
            raise InvariantError(f"recipe {s['rule_id']} membership does not add up to its totals")
    by_locator = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical = {r["pick_obs_id"]: r for r in ledger_rows if r.get("pick_obs_id")}
    for m in membership:
        source = by_locator.get((m["source_path"], f"slot={m['slot']}")) or by_locator.get((m["source_path"], "file"))
        link = canonical.get(source["obs_id"]) if source else None
        got = (m["source_obs_id"], m["source_state"], m["source_reason"], m["occurrence_disposition"],
               m["selection_id"], m["canonical_bts_outcome"])
        if source is None or got != (source["obs_id"], source["state"], source["reason"], source["disposition"],
                                     link["selection_id"] if link else None, link.get("bts_outcome") if link else None):
            raise InvariantError(f"recipe row {m['recipe_slot_key']} is not linked to the occurrence that accounts for it")
    if any(n > 1 for n in Counter((m["round_id"], m["unit_id"], m["player_id"]) for m in matches).values()):
        raise InvariantError("a contest slot identity appears twice")
    if any(n > 1 for n in Counter(m["selection_id"] for m in matches if m["selection_id"]).values()):
        raise InvariantError("a selection is linked to more than one contest slot")
    if any(n > 1 for n in Counter(r["row_id"] for r in ledger_rows).values()):
        raise InvariantError("duplicate ledger row ids")
    by_date: dict[str, list[dict]] = {}
    for r in ledger_rows:
        if r["row_kind"] != "contest_only":
            by_date.setdefault(r["date"], []).append(r)
    for d in season_dates:
        rows = by_date.get(d, [])
        kinds = {r["row_kind"] for r in rows}
        if not rows:
            raise InvariantError(f"no ledger row for {d}")
        if "selection" in kinds and len(kinds) > 1:
            raise InvariantError(f"{d} mixes a day row with selections")
        if "selection" not in kinds and len(rows) != 1:
            raise InvariantError(f"{d} has {len(rows)} day rows")
        slots = [r["slot"] for r in rows if r["row_kind"] == "selection"]
        if len(slots) > 2 or len(slots) != len(set(slots)):
            raise InvariantError(f"{d} has malformed selection slots {slots}")
```

Append to `scripts/audit/season_ledger/reconcile.py` (the recipe rules):
```python
# ---- recipe rules: fixed before any real count (plan Task 9 text; pinned by rules_fingerprint) -----------
GRADED = frozenset({"hit", "miss"})
FILE_SETS = {"F1": re.compile(rf"^picks/{_D}\.json$"),
             "F2": re.compile(r"^picks/2026-[^/]*\.json$"),
             "F3": re.compile(r"^picks/.*\.json$")}
RECIPE_KINDS = {"scorecard_0911": "prose_rerun", "tally_0914": "candidate_search"}
_SCORECARD = {"recipe": "scorecard_0911", "published": (141, 82), "recipe_date": "2026-09-11",
              "window": ("2026-03-29", "2026-09-10")}
_TALLY = {"recipe": "tally_0914", "published": (191, 157), "recipe_date": "2026-09-14"}
RULES: dict[str, dict] = {}
for _i, (_f, _p, _l) in enumerate(itertools.product(("F1", "F2"), ("G1", "G2"), ("G1", "G2")), start=1):
    RULES[f"S{_i}"] = {**_SCORECARD, "files": _f, "primary": _p, "legs": _l}
for _i, (_f, _g, _end) in enumerate(itertools.product(("F1", "F2", "F3"), ("G1", "G2", "G3", "G4"),
                                                      ("2026-09-13", "2026-09-14")), start=1):
    RULES[f"T{_i}"] = {**_TALLY, "files": _f, "primary": _g, "legs": _g, "window": ("0000-00-00", _end)}
_ET = ZoneInfo("America/New_York")
_FILE_DATE = re.compile(_D)


def slot_value(doc: dict, slot: str, grading: str):
    """The value a grading reads for one slot of a pick-object record (the recipe's own reading)."""
    if slot == "double_down" and doc.get("double_down") is None:
        return None
    if grading == "G2":
        sr = doc.get("slot_results")
        if isinstance(sr, dict):
            return sr.get("pick" if slot == "primary" else "double_down")
        return doc.get("result") if (slot == "primary" and doc.get("double_down") is None) else None
    return doc.get("result")     # G1, G3 and G4 read the day result


def counts(value, grading: str) -> bool:
    """Whether a grading counts a recipe value. G1/G2 count only the string labels in GRADED; G3 counts any
    non-null value and G4 every slot, as those naive rules would — a malformed value never crashes a recipe."""
    if grading == "G4":
        return True
    if grading == "G3":
        return value is not None
    return isinstance(value, str) and value in GRADED


def slot_inclusion(doc: dict, primary_grading: str, leg_grading: str) -> dict[str, tuple]:
    out = {}
    for slot, grading in (("primary", primary_grading), ("double_down", leg_grading)):
        if slot == "double_down" and doc.get("double_down") is None:
            continue
        value = slot_value(doc, slot, grading)
        out[slot] = (value, counts(value, grading))
    return out


def _mtime_after(mtime_utc: str | None, recipe_date: str) -> bool | None:
    """Interpretation I9: the filesystem mtime (suggestive, never proof) against the recipe's ET date — True
    after, False before, None on the same ET day (order unknown) or without an mtime."""
    if mtime_utc is None:
        return None
    day = datetime.fromisoformat(mtime_utc).astimezone(_ET).date().isoformat()
    return None if day == recipe_date else day > recipe_date


def _natural(rule_id: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", rule_id)]


def _universe(files: dict[str, bytes | None]) -> dict[str, tuple[dict, str]]:
    """Every pick-object record under picks/ (any suffix, AppleDouble aside) with its window date."""
    universe: dict[str, tuple[dict, str]] = {}
    for rel, data in files.items():
        if data is None or not rel.startswith("picks/") or _APPLEDOUBLE.search(rel):
            continue
        doc = _doc(data)
        if isinstance(doc, dict) and isinstance(doc.get("pick"), dict):
            m = _FILE_DATE.search(rel.rsplit("/", 1)[-1])
            universe[rel] = (doc, m.group(0) if m else str(doc.get("date") or ""))
    return universe


def membership_slots(files: dict[str, bytes | None]) -> set[tuple[str, str]]:
    """(source_path, slot) for every slot of every universe record — what each rule must list exactly once."""
    return {(rel, slot) for rel, (doc, _day) in _universe(files).items()
            for slot in ("primary", "double_down") if slot == "primary" or doc.get("double_down") is not None}


def evaluate_rules(files: dict[str, bytes | None], rules: dict, mtimes: dict[str, str | None], *,
                   frozen_at: str) -> tuple[list[dict], list[dict]]:
    """Spec §8 table 3. Every rule gets a row per universe slot, included or excluded with a reason, so no candidate
    record disappears. `recipe_slot_key` names the recipe's record; the compiler links each row to the source
    occurrence that accounts for it (Codex plan r2 #5)."""
    universe = _universe(files)
    summary, membership = [], []
    for rule_id in sorted(rules, key=_natural):
        rule = rules[rule_id]
        n = {"primary": 0, "double_down": 0}
        lo, hi = rule["window"]
        for rel in sorted(universe):
            doc, day = universe[rel]
            in_set = FILE_SETS[rule["files"]].match(rel) is not None
            in_window = lo <= day <= hi
            for slot, (value, counted) in slot_inclusion(doc, rule["primary"], rule["legs"]).items():
                reason = (None if (in_set and in_window and counted) else "not_in_file_set" if not in_set
                          else "outside_window" if not in_window else "value_not_counted")
                n[slot] += reason is None
                membership.append({"rule_id": rule_id, "recipe": rule["recipe"], "source_path": rel, "slot": slot,
                                   "recipe_slot_key": f"{rel}#{slot}", "file_date": day,
                                   "recipe_value": None if value is None else str(value), "included": reason is None,
                                   "exclusion_reason": reason, "historical_membership": "unknown",
                                   "source_mtime_utc": mtimes.get(rel),
                                   "mtime_after_recipe_date": _mtime_after(mtimes.get(rel), rule["recipe_date"]),
                                   "frozen_at_utc": frozen_at})
        pub_p, pub_l = rule["published"]
        fit = {(True, True): "both", (True, False): "primaries_only", (False, True): "legs_only",
               (False, False): "none"}[(n["primary"] == pub_p, n["double_down"] == pub_l)]
        summary.append({"rule_id": rule_id, "recipe": rule["recipe"], "files": rule["files"],
                        "primary_grading": rule["primary"], "leg_grading": rule["legs"],
                        "window": list(rule["window"]), "primaries": n["primary"], "legs": n["double_down"],
                        "published_primaries": pub_p, "published_legs": pub_l, "fit": fit,
                        "label": ("matches published totals; historical membership unverified" if fit == "both"
                                  else f"does not reproduce both totals ({fit})")})
    return summary, membership


def recipe_labels(summary: list[dict]) -> dict[str, str]:
    """Spec §8 and Interpretation I9. The 9/11 scorecard rerun is a hypothesis whether or not it reproduces
    its totals; the 9/14 tally is a hypothesis if some pre-declared rule reproduces both totals, else
    unrecoverable. A single matching total pins nothing. `exact` and `partial` need independent per-record
    evidence, which Phase 1 does not have, so they are never emitted."""
    out = {}
    for recipe in sorted({s["recipe"] for s in summary}):
        if RECIPE_KINDS.get(recipe) == "prose_rerun":
            out[recipe] = "hypothesis"
        else:
            out[recipe] = "hypothesis" if any(s["fit"] == "both" for s in summary if s["recipe"] == recipe) \
                else "unrecoverable"
    return out


_RECIPE_ROOTS = ("evaluate_rules", "recipe_labels")
_PACKAGE = __name__.rsplit(".", 1)[0]


def _bound(value):
    """A JSON description of one configuration value the recipe code reads; any other kind fails closed."""
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": int(value.flags)}
    if isinstance(value, ZoneInfo):
        return {"zone": value.key}
    if isinstance(value, dict):
        return {"dict": {canonical_json(_bound(k)): _bound(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"seq": [_bound(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"set": sorted(canonical_json(_bound(v)) for v in value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"rules_fingerprint cannot bind {type(value).__name__}")


def _global_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def recipe_closure() -> dict[str, object]:
    """Everything the recipe evaluation reads (Codex plan r2 #6, r3 #4), by qualified name: the source of every
    repo-local function reachable from `evaluate_rules` and `recipe_labels` (imported ones included), and every
    configuration value those functions read — compiled regexes with their flags, the time zone, tables. Modules
    and outside callables are bound by name; their behaviour is pinned by the versions `build.json` records.
    Anything else fails closed."""
    bound: dict[str, object] = {}
    todo = [globals()[name] for name in _RECIPE_ROOTS]
    while todo:
        fn = todo.pop()
        key = f"{fn.__module__}.{fn.__qualname__}"
        if key in bound:
            continue
        bound[key] = inspect.getsource(fn)
        for name in sorted(_global_names(fn.__code__)):
            if name not in fn.__globals__ or (name.startswith("__") and name.endswith("__")):
                continue          # an attribute, a local, a builtin, or interpreter bookkeeping (e.g. __file__)
            value = fn.__globals__[name]
            module = getattr(value, "__module__", None) or ""
            if isinstance(value, types.FunctionType) and module.startswith(_PACKAGE):
                todo.append(value)
            elif isinstance(value, types.ModuleType):
                bound[f"module:{value.__name__}"] = value.__name__
            elif callable(value) and not module.startswith(_PACKAGE):
                bound[f"callable:{module}.{getattr(value, '__qualname__', name)}"] = name
            else:
                bound[f"{fn.__module__}.{name}"] = _bound(value)
    return bound


def rules_fingerprint() -> str:
    """One digest over the rule table and everything the recipe evaluation reads (`recipe_closure`): any edit that
    can move a recipe total or label — a predicate, the JSON decoder, a regex or its flags, a table, even a
    comment in a reachable function — changes the digest predeclared in the exposure register."""
    return sha256_hex(json.dumps({"rules": _bound(RULES), "closure": recipe_closure()}, sort_keys=True).encode())
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_reconcile.py -q`
Expected: `42 passed`. If `test_candidate_rules_and_their_evaluator_are_fingerprinted` fails, the rule table, a function reachable from the recipe roots, or a value those functions read differs from this plan. The fingerprint hashes source text, so whitespace and comments count. Fix the code to match this plan; never change `RULES_SHA256`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/reconcile.py tests/scripts/season_ledger/test_reconcile.py
git commit -m "feat(ledger): routing, accounting, structural census, invariants, frozen recipe rules (task 9)"
```

---

### Task 10: Offline compile pipeline and outputs

**Files:**
- Create: `scripts/audit/season_ledger/compile.py`
- Test: `tests/scripts/season_ledger/test_compile.py`

**Interfaces:**
- **Consumes:** everything above.
- **Produces:**
  - `SEASON_DATES` (2026-03-25 → 2026-09-27)
  - `compile_bundle(bundle_root, out_dir, *, uv_lock_sha256, code_sha=None) -> dict`
  - `_eligibility(row, match, unit_status, refusals)` (I6)
  - `_outcome_status(slot_result, slot_result_state)` (I11)
- **Unusable evidence (I15):** a routed decision, scheduler-state, archive or lineup-evolution file with any quarantined record adds its kind to that date's `unusable` set for `day_rows`. The pick-file occurrences of a `decision_unusable` day are `unresolved_pick_file_view`.
- **Output directory:** `out_dir` must not exist. Every table is built and the output checks pass before the compiler reserves the directory with `mkdir`, which fails if it exists (even empty), so neither a repeated nor a concurrent build can share it.
- **Outputs:**
  - `season_2026_ledger.parquet`
  - `season_2026_ledger_occurrences.parquet`
  - `season_2026_ledger_contest_slots.parquet`
  - `season_2026_ledger_reconciliation.parquet`
  - `season_2026_ledger_build.json`, which records builder, code sha, Python, pyarrow, environment lock, bundle manifest sha and acquisition time, rules fingerprint, and every count
  - `season_2026_ledger_summary.md`

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_compile.py`:

```python
import json
import random
import shutil

import pyarrow.parquet as pq
import pytest

from scripts.audit.season_ledger.compile import compile_bundle
from tests.scripts.season_ledger.builders import (cand, contest_line, decision_json, dumps, gz, pick_json, rnd,
                                                  seal_bundle, slot, state_json)

ROUNDS = {971: "2026-08-20", 972: "2026-08-21", 973: "2026-08-22", 974: "2026-08-23", 975: "2026-08-24",
          976: "2026-08-25", 977: "2026-08-26", 978: "2026-08-27"}
PLAYERS = {2513: 802415, 1300: 202, 1001: 101, 1777: 777, 1404: 404, 1505: 505, 1606: 606, 1707: 707, 1808: 808,
           1708: 708}


def _game(pk, away, home):
    return {"gamePk": pk, "status": {"codedGameState": "F", "detailedState": "Final"},
            "teams": {"away": {"team": {"abbreviation": away}}, "home": {"team": {"abbreviation": home}}}}


def _schedule(day, *games):
    return dumps({"dates": [{"date": day, "games": list(games)}]})


def _units(*units):
    return gz(dumps({"units": [{"id": u, "feedId": f, "roundId": r, "status": s} for u, f, r, s in units]}))


def _state(day):
    return state_json(day, pick_locked=True, pick_locked_at=f"{day}T17:00:00-04:00", committed_pick_written=True)


def _season_files() -> dict[str, bytes]:
    """8/20 C-03 double; 8/21 entered-but-undelivered preview; 8/22 contest-only; 8/23 Pass on a game postponed
    after lock; 8/24 game postponed before lock; 8/25 saver-absorbed miss; 8/26 partial-void double; 8/27 a
    postponement superseded by a later 'scheduled' capture before lock; 8/28 a single decision against a double
    pick file. Round facts are coherent."""
    ledger = contest_line("2026-08-28T14:30:00Z", [
        rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit", number=1), slot(1928, 2513, "hit", number=2)]),
        rnd(972, "hit", 9, 1, [slot(1929, 1001, "hit")]),
        rnd(973, "hit", 10, 1, [slot(1930, 1777, "hit")]),
        rnd(974, "void", 10, 0, [slot(2404, 1404, "void", hits=0, at_bats=0)]),
        rnd(975, "void", 10, 0, [slot(2505, 1505, "void", hits=0, at_bats=0)]),
        rnd(976, "used_mulligan", 10, 0, [slot(1931, 1606, "not_hit", hits=0)]),
        rnd(977, "hit", 11, 1, [slot(1932, 1707, "void", hits=0, at_bats=0), slot(1933, 1808, "hit", number=2)]),
        rnd(978, "hit", 12, 1, [slot(2708, 1708, "hit")])])
    return {
        "picks/2026-08-20.json": pick_json("2026-08-20", primary={"batter_id": 802415, "game_pk": 822934, "team": "TB"},
                                           dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                                           notification_sent=True, notification_id="dm-1"),
        "picks/2026-08-20/decision.json": decision_json("2026-08-20", action="double", primary=cand(802415, 822934),
                                                        double_down=cand(202, 5002, team="NYY")),
        "picks/2026-08-20/scheduler_state.json": _state("2026-08-20"),
        "picks/2026-08-21.json": pick_json("2026-08-21"),
        "picks/2026-08-23/decision.json": decision_json("2026-08-23", action="single", primary=cand(404, 6004, team="SEA")),
        "picks/2026-08-23/scheduler_state.json": _state("2026-08-23"),
        "picks/2026-08-24/decision.json": decision_json("2026-08-24", action="single", primary=cand(505, 6005, team="SEA")),
        "picks/2026-08-24/scheduler_state.json": _state("2026-08-24"),
        "picks/2026-08-25/decision.json": decision_json("2026-08-25", action="single", primary=cand(606, 6006, team="SEA")),
        "picks/2026-08-26/decision.json": decision_json("2026-08-26", action="double", primary=cand(707, 6007),
                                                        double_down=cand(808, 6008, team="NYY")),
        "picks/2026-08-27/decision.json": decision_json("2026-08-27", action="single", primary=cand(708, 6708, team="SEA")),
        "picks/2026-08-27/scheduler_state.json": _state("2026-08-27"),
        "picks/2026-08-28/decision.json": decision_json("2026-08-28", action="single", primary=cand(909, 6909)),
        "picks/2026-08-28.json": pick_json("2026-08-28", primary={"batter_id": 909, "game_pk": 6909},
                                           dd={"batter_id": 910, "game_pk": 6910}),
        "picks/._2026-08-20.json": b"\x00\x05\x16\x07",
        "picks/2026-08-21.shadow.json": pick_json("2026-08-21", result="hit"),      # excluded, yet in the recipe universe
        "picks/account_state/contest_ledger.jsonl": (ledger + "\n").encode(),
        "static/rounds/20260828T120000Z.json.gz": gz(dumps({"rounds": [
            {"id": r, "date": f"{d}T08:00:00-04:00"} for r, d in ROUNDS.items()]})),
        "static/players/20260828T120000Z.json.gz": gz(dumps({"players": [
            {"id": p, "feedId": f} for p, f in PLAYERS.items()]})),
        "static/units/20260823T150000Z.json.gz": _units((2404, 6004, 974, "scheduled")),
        "static/units/20260824T020000Z.json.gz": _units((2404, 6004, 974, "postponed")),   # after the 8/23 lock
        "static/units/20260824T150000Z.json.gz": _units((2505, 6005, 975, "postponed")),   # before the 8/24 lock
        "static/units/20260827T100000Z.json.gz": _units((2708, 6708, 978, "postponed")),
        "static/units/20260827T200000Z.json.gz": _units((2708, 6708, 978, "scheduled")),   # superseded before lock
        "schedules/2026-08-20.json": _schedule("2026-08-20", _game(822934, "TOR", "TB"), _game(5002, "NYY", "BAL")),
        "schedules/2026-08-21.json": _schedule("2026-08-21", _game(5001, "BOS", "TB")),
        "schedules/2026-08-22.json": _schedule("2026-08-22", _game(5777, "LAD", "SD")),
        "schedules/2026-08-25.json": _schedule("2026-08-25", _game(6006, "SEA", "HOU")),
        "schedules/2026-08-26.json": _schedule("2026-08-26", _game(6007, "TB", "CLE"), _game(6008, "NYY", "KC")),
    }


FILES = _season_files()     # one byte set, reused by every test (Codex plan r1 #12)


def _compile(tmp_path, name, files=None):
    bundle = tmp_path / f"bundle_{name}"
    seal_bundle(bundle, FILES if files is None else files, missing=["schedules/2026-08-23.json"])
    out = tmp_path / f"out_{name}"
    compile_bundle(bundle, out, uv_lock_sha256="test-lock")
    return bundle, out


def _ledger(out):
    return {r["row_id"]: r for r in pq.read_table(out / "season_2026_ledger.parquet").to_pylist()}


def test_c03_disagreement_slot_order_and_streaks(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    p, dd = led["2026-08-20|primary|802415|822934"], led["2026-08-20|double_down|202|5002"]
    assert (p["match"], p["entry_status"], p["bts_outcome"], p["local_slot_result_raw"]) == (
        "inferred", "confirmed", "hit", "miss")
    assert p["local_vs_contest_disagreement"] is True and dd["local_vs_contest_disagreement"] is False
    assert (p["slot_number"], dd["slot_number"]) == (2, 1)            # the contest's order differs from ours
    assert (p["streak_before"], p["streak_after"], p["saver_available_before"]) == (None, 8, None)
    assert p["commit_status"] == "committed_evidenced" and p["locked_at"] == "2026-08-20T21:00:00.000000Z"
    assert (p["scheduled_games"], p["round_id"]) == (2, 971)


def test_entered_preview_contest_only_and_unobserved_days(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    preview = led["2026-08-21|primary|101|5001"]
    assert (preview["commit_status"], preview["entry_status"], preview["match"]) == ("unconfirmed", "confirmed", "inferred")
    only = led["contest|973|1930|1777"]
    assert (only["row_kind"], only["match"], only["bts_outcome_status"], only["slot"]) == (
        "contest_only", "unmapped", "unmapped", None)
    assert led["day|2026-08-22|unobserved_day"]["scheduled_games"] == 1
    assert led["day|2026-03-25|unobserved_day"]["reason"] == "no_evidence"


def test_a_pass_is_an_outcome_and_eligibility_reads_the_latest_status_before_lock(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    after, before = led["2026-08-23|primary|404|6004"], led["2026-08-24|primary|505|6005"]
    superseded = led["2026-08-27|primary|708|6708"]
    assert (after["match"], after["bts_outcome"], after["contest_norm"]) == ("evidenced", "void", "HOLD")
    assert after["game_eligibility"] == "unknown"                     # postponed only after lock
    assert (before["game_eligibility"], before["game_eligibility_at"]) == (
        "postponed_evidenced", "2026-08-24T15:00:00.000000Z")
    assert superseded["game_eligibility"] == "unknown"                # Codex plan r1 #8: rescheduled before lock
    assert after["streak_before"] == 10 and after["scheduled_games"] is None   # 8/23 schedule declared missing


def test_round_labels_stay_on_the_round_and_legs_keep_their_own_grades(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    saver = led["2026-08-25|primary|606|6006"]
    assert (saver["bts_outcome"], saver["contest_round_result"], saver["contest_norm"]) == ("not_hit", "used_mulligan", "NO_HIT")
    assert saver["saver_available_before"] is None
    p, dd = led["2026-08-26|primary|707|6007"], led["2026-08-26|double_down|808|6008"]
    assert (p["bts_outcome"], p["contest_norm"], dd["bts_outcome"]) == ("void", "HOLD", "hit")
    assert p["contest_round_result"] == dd["contest_round_result"] == "hit"


def test_every_occurrence_is_accounted_resolvable_and_joined_to_the_reconciliation(tmp_path):
    out = _compile(tmp_path, "a")[1]
    occ = pq.read_table(out / "season_2026_ledger_occurrences.parquet").to_pylist()
    states = {(o["source_path"], o["state"]) for o in occ if o["locator"] == "file"}
    assert ("picks/._2026-08-20.json", "excluded") in states
    assert ("schedules/2026-08-23.json", "declared_missing") in states
    emitted = {o["obs_id"]: o for o in occ if o["state"] == "emitted"}
    assert all(o["disposition"] and o["fields_json"] for o in emitted.values())
    led = _ledger(out)
    assert all(r[c] in emitted for r in led.values() for c in ("pick_obs_id", "decision_obs_id", "contest_obs_id")
               if r[c] is not None)
    build = json.loads((out / "season_2026_ledger_build.json").read_text())
    assert build["row_kinds"]["contest_only"] == 1 and len(build["rules_fingerprint"]) == 64
    recon = pq.read_table(out / "season_2026_ledger_reconciliation.parquet").to_pylist()
    (row,) = [m for m in recon if m["rule_id"] == "S1" and m["source_path"] == "picks/2026-08-20.json"
              and m["slot"] == "primary"]
    assert (row["recipe_value"], row["canonical_bts_outcome"], row["occurrence_disposition"], row["selection_id"]) == (
        "miss", "hit", "canonical_selection", "2026-08-20|primary|802415|822934")
    assert row["source_obs_id"] in emitted
    occ_ids = {o["obs_id"] for o in occ}
    shadow = [m for m in recon if m["source_path"] == "picks/2026-08-21.shadow.json"]
    assert shadow and all(m["source_obs_id"] in occ_ids and (m["source_state"], m["source_reason"], m["selection_id"])
                          == ("excluded", "shadow_model_out_of_scope", None) for m in shadow)   # Codex plan r2 #5


def test_outputs_are_byte_identical_across_runs_roots_and_discovery_order(tmp_path):
    bundle_a, out_a = _compile(tmp_path, "a")
    items = list(FILES.items())
    random.Random(7).shuffle(items)
    out_b = _compile(tmp_path, "b", dict(items))[1]
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(bundle_a, moved)
    out_c = tmp_path / "out_c"
    compile_bundle(moved, out_c, uv_lock_sha256="test-lock")
    names = sorted(p.name for p in out_a.iterdir())
    assert len(names) == 6
    for f in names:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes() == (out_c / f).read_bytes(), f
    manifest = moved / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()                       # a hand-reordered manifest: only its own hash may change
    manifest.write_text(json.dumps(doc))
    out_d = tmp_path / "out_d"
    compile_bundle(moved, out_d, uv_lock_sha256="test-lock")
    for f in (n for n in names if n.endswith(".parquet")):
        assert (out_a / f).read_bytes() == (out_d / f).read_bytes(), f


def test_compile_refuses_any_existing_output_directory(tmp_path):
    bundle, out = _compile(tmp_path, "a")
    with pytest.raises(FileExistsError, match="already exists"):
        compile_bundle(bundle, out, uv_lock_sha256="test-lock")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        compile_bundle(bundle, tmp_path / "empty", uv_lock_sha256="test-lock")


def test_raw_values_round_trip_through_the_occurrence_table(tmp_path):
    # Codex plan r2 #2: wrong-typed values are nulled in typed columns but recoverable from the compiled output.
    files = {"static/units/20260801T150000Z.json": dumps({"units": [{"id": 2449, "feedId": "RAW_GAME", "roundId": 1009,
                                                                    "status": "scheduled"}]}),
             "picks/account_state/contest_ledger.jsonl": (contest_line("2026-08-02T14:30:00Z", [
                 rnd(990, "hit", "RAW_STREAK", 1, [slot(3001, 4001, "hit")])]) + "\n").encode()}
    seal_bundle(tmp_path / "b", files)
    compile_bundle(tmp_path / "b", tmp_path / "o", uv_lock_sha256="test-lock")
    occ = {(o["source_path"], o["locator"]): o for o in
           pq.read_table(tmp_path / "o" / "season_2026_ledger_occurrences.parquet").to_pylist()}
    unit = occ[("static/units/20260801T150000Z.json", "item=0")]
    assert json.loads(unit["record_raw_json"])["feedId"] == "RAW_GAME" and json.loads(unit["fields_json"])["feed_id"] is None
    round_occ = occ[("picks/account_state/contest_ledger.jsonl", "line=1/round=0")]
    assert json.loads(round_occ["record_raw_json"])["streak"] == "RAW_STREAK"


def test_outcome_status_distinguishes_a_malformed_grade_from_a_source_null():
    from scripts.audit.season_ledger.compile import _outcome_status
    assert (_outcome_status("hit", "value"), _outcome_status(None, "null"), _outcome_status(None, "type_mismatch")) == (
        "graded", "matched_ungraded", "unknown")


def test_a_conflicting_pick_file_is_kept_whole_as_an_unresolved_view(tmp_path):
    # Codex plan r1 #4: both legs of the double pick file stay visible; neither is silently "not selected".
    out = _compile(tmp_path, "a")[1]
    occ = pq.read_table(out / "season_2026_ledger_occurrences.parquet").to_pylist()
    assert {o["locator"]: o["disposition"] for o in occ if o["source_path"] == "picks/2026-08-28.json"} == {
        "slot=primary": "unresolved_pick_file_view", "slot=double_down": "unresolved_pick_file_view"}
    (r,) = [r for r in _ledger(out).values() if r["date"] == "2026-08-28"]
    assert (r["finalization"], r["pick_view_action"], r["pick_obs_id"]) == ("unresolved", "double", None)


def test_unusable_evidence_never_finalizes_a_day_or_reads_as_absent(tmp_path):
    # Codex plan r3 #2 and #5: malformed game identities never agree, and quarantined evidence is never absence.
    files = {"picks/2026-08-20.json": pick_json("2026-08-20", primary={"game_pk": "bad_file_game"},
                                                notification_sent=True, notification_id="dm-1"),
             "picks/2026-08-20/decision.json": decision_json("2026-08-20", action="single",
                                                             primary=cand(101, "different_bad_game")),
             "picks/2026-08-21.json": pick_json("2026-08-21"),
             "picks/2026-08-21/decision.json": decision_json("2026-08-21", action="single",
                                                             primary=dict(cand(101, 5001), batter_id="101")),
             "picks/lineup_evolution_2026-08-22.jsonl": b"{not json\n"}
    seal_bundle(tmp_path / "b", files)
    compile_bundle(tmp_path / "b", tmp_path / "o", uv_lock_sha256="test-lock")
    days = {r["date"]: (r["row_kind"], r["reason"]) for r in _ledger(tmp_path / "o").values()
            if r["date"] in ("2026-08-20", "2026-08-21", "2026-08-22")}
    assert days == {"2026-08-20": ("unfinalized_day", "decision_unusable"),
                    "2026-08-21": ("unfinalized_day", "decision_unusable"),
                    "2026-08-22": ("unfinalized_day", "unusable_evidence_only")}
    occ = {(o["source_path"], o["locator"]): (o["state"], o["reason"] or o["disposition"]) for o in
           pq.read_table(tmp_path / "o" / "season_2026_ledger_occurrences.parquet").to_pylist()}
    assert occ[("picks/2026-08-20.json", "slot=primary")] == ("quarantined", "slot_bad_game_pk")
    assert occ[("picks/2026-08-21.json", "slot=primary")] == ("emitted", "unresolved_pick_file_view")


def test_a_refusal_counts_only_with_a_reason_and_a_time_before_a_known_lock():
    from scripts.audit.season_ledger.compile import _eligibility
    row = {"locked_at": "2026-08-28T22:00:00.000000Z", "date": "2026-08-28", "slot": "primary", "batter_id": 1,
           "game_pk": 2}
    key = ("2026-08-28", "primary", 1, 2)
    assert _eligibility(row, None, {}, {key: [("2026-08-28T21:00:00.000000Z", "past_submission_cutoff")]}) == (
        "refused_evidenced", "2026-08-28T21:00:00.000000Z", "past_submission_cutoff")
    assert _eligibility(row, None, {}, {key: [(None, "past_submission_cutoff")]})[0] == "unknown"
    assert _eligibility(row, None, {}, {key: [("2026-08-28T23:00:00.000000Z", "late")]})[0] == "unknown"
    assert _eligibility(row, None, {}, {key: [("2026-08-28T21:00:00.000000Z", None)]})[0] == "unknown"   # no reason
    unlocked = dict(row, locked_at=None)                                     # Codex plan r2 #8: lock unknown
    assert _eligibility(unlocked, None, {}, {key: [("2026-08-28T21:00:00.000000Z", "past_submission_cutoff")]})[0] == "unknown"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_compile.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.compile'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/compile.py`:

```python
"""Offline compilation from a sealed bundle (spec §3–§10). Reads only the bundle; no network. Every table is
built (and type-checked) before any file is written, and each build writes a fresh output directory."""
from __future__ import annotations

import json
import platform
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from . import BUILDER_VERSION
from .bundle import open_bundle
from .contest import (entered_rounds, line_round_streaks, match_slot, players_lookup, resolve_duplicate_links,
                      rounds_lookup, slot_history, streak_before, team_games, unit_status_history, units_lookup)
from .ids import canonical_json, sha256_hex, utc_iso
from .io import build_table, write_table
from .outcomes import derived_single_result, normalize_contest, normalize_local, slot_disagreement
from .reconcile import (PARSERS, RULES, account, assign_dispositions, check_invariants, check_sources,
                        evaluate_rules, recipe_labels, route, rules_fingerprint)
from .rows import day_rows
from .sources.pick_files import pick_file_state

SEASON_DATES = [(date(2026, 3, 25) + timedelta(days=i)).isoformat() for i in range(187)]   # 3/25 → 9/27
_SEASON = frozenset(SEASON_DATES)
POSTPONED_UNIT_STATUSES = frozenset({"postponed"})
DAY_KINDS = frozenset({"pick_file", "archive", "manual_archive", "repair_archive", "decision", "scheduler_state",
                       "lineup_evolution"})
OBSERVATION_KINDS = ("archive", "manual_archive", "repair_archive", "lineup_evolution")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
S, I, B, F = pa.string(), pa.int64(), pa.bool_(), pa.float64()

LEDGER_SCHEMA = pa.schema([
    ("row_id", S), ("row_kind", S), ("date", S), ("round_id", I), ("slot", S), ("selection_id", S), ("reason", S),
    ("scheduled_games", I),
    ("batter_id", I), ("batter_name", S), ("team_at_pick", S), ("game_pk", I), ("game_time", S),
    ("lineup_position", I), ("projected_lineup", B), ("pitcher_id", I), ("p_stated", F),
    ("finalization", S), ("pick_view_batter_id", I), ("pick_view_game_pk", I), ("pick_view_action", S),
    ("pick_file_complete", B),
    ("commit_status", S), ("commit_basis", S), ("history_status", S), ("scheduler_commit_flag", B),
    ("action", S), ("action_source_raw", S), ("action_source", S), ("objective", S), ("pick_policy_objective", S),
    ("degraded_reason", S), ("decision_streak", I), ("decision_state_source", S), ("decision_state_status", S),
    ("declined_batter_id", I), ("declined_game_pk", I),
    ("predicted_at", S), ("locked_at", S), ("delivery_attempted", B), ("delivery_attempted_at", S),
    ("delivery_confirmed", B), ("delivery_basis", S), ("delivery_evidence_conflict", B), ("delivered_at", S),
    ("game_eligibility", S), ("game_eligibility_at", S), ("game_eligibility_basis", S),
    ("entry_status", S), ("match", S), ("match_reason", S), ("unit_id", I), ("player_id", I), ("slot_number", I),
    ("entry_observed_at", S), ("contest_slot_grade_raw", S), ("bts_outcome", S), ("bts_outcome_status", S),
    ("contest_round_result", S), ("streak_before", I), ("streak_after", I), ("saver_available_before", B),
    ("local_slot_result_raw", S), ("local_day_result_raw", S), ("local_slot_result_derived", S),
    ("derivation_source", S), ("local_norm", S), ("contest_norm", S), ("local_vs_contest_disagreement", B),
    ("decision_obs_id", S), ("pick_obs_id", S), ("pick_view_obs_id", S), ("state_obs_id", S), ("contest_obs_id", S),
])
CONTEST_SCHEMA = pa.schema([
    ("round_id", I), ("unit_id", I), ("player_id", I), ("date", S), ("batter_id", I), ("game_pk", I),
    ("selection_id", S), ("match", S), ("match_reason", S), ("first_seen", S), ("last_seen", S),
    ("n_observations", I), ("changed", B), ("dropped_later", B), ("slot_number", I), ("slot_result", S),
    ("slot_result_state", S),
    ("hits", I), ("hits_state", S), ("at_bats", I), ("at_bats_state", S), ("round_result", S),
    ("round_streak", I), ("round_streak_increase", I), ("last_obs_id", S)])
OCCURRENCE_SCHEMA = pa.schema([("source_path", S), ("locator", S), ("obs_id", S), ("kind", S), ("state", S),
                               ("reason", S), ("disposition", S), ("content_sha256", S), ("fields_json", S),
                               ("record_raw_json", S)])
RECONCILIATION_SCHEMA = pa.schema([
    ("rule_id", S), ("recipe", S), ("source_path", S), ("slot", S), ("recipe_slot_key", S), ("file_date", S),
    ("recipe_value", S), ("included", B), ("exclusion_reason", S), ("historical_membership", S),
    ("source_mtime_utc", S), ("mtime_after_recipe_date", B), ("frozen_at_utc", S), ("source_obs_id", S),
    ("source_state", S), ("source_reason", S), ("occurrence_disposition", S), ("selection_id", S),
    ("canonical_bts_outcome", S)])


def _group(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _counts(rows, key: str) -> dict:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def _eligibility(row: dict, match: dict | None, unit_status: dict, refusals: dict) -> tuple:
    """§9 and Interpretation I6: only evidence timed before a KNOWN lock counts, and the latest such evidence
    decides. A refusal binds when its archive names this date, slot, batter and game, carries a reason, and was
    written before the lock; without a known lock nothing can be shown to precede it."""
    locked = row["locked_at"]
    if locked is None:
        return "unknown", None, None
    if match is not None and match["match"] == "evidenced":
        before = [(t, s) for t, s in unit_status.get(match["unit_id"], []) if t is not None and t < locked]
        if before and before[-1][1] in POSTPONED_UNIT_STATUSES:
            return "postponed_evidenced", before[-1][0], "unit_capture_status_postponed"
    timed = sorted((t, reason) for t, reason in refusals.get((row["date"], row["slot"], row["batter_id"],
                                                                row["game_pk"]), [])
                   if t is not None and reason and t < locked)
    if timed:
        return "refused_evidenced", timed[-1][0], timed[-1][1]
    return "unknown", None, None


def _outcome_status(slot_result, slot_result_state: str) -> str:
    """I11 with I13: a string grade is `graded`; a source null is `matched_ungraded`; a present value of the wrong
    type is `unknown` — never presented as a source null."""
    if slot_result_state == "type_mismatch":
        return "unknown"
    return "graded" if slot_result is not None else "matched_ungraded"


def compile_bundle(bundle_root, out_dir, *, uv_lock_sha256: str | None, code_sha: str | None = None) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out} (each build writes a new directory)")
    manifest, files = open_bundle(bundle_root)
    mtimes = {e["rel_path"]: e.get("source_mtime_utc") for e in manifest["entries"]}
    routed = {rel: kind for rel, data in files.items() if data is not None and (kind := route(rel))}
    parsed = {rel: PARSERS[kind](rel, files[rel]) for rel, kind in routed.items()}

    def rows_of(*kinds: str) -> list[dict]:
        return [r for rel, k in routed.items() if k in kinds for r in parsed[rel].rows]

    for rel, kind in routed.items():
        if kind in DAY_KINDS:
            m = _DATE.search(rel)
            for r in parsed[rel].rows:
                r["file_date"] = m.group(0) if m else None
    accounting = account(files, routed, parsed)
    check_sources(files, routed, accounting)          # §8 anti-join, before any canonical row exists

    # §5 day rows
    decisions = {r["file_date"]: r for r in rows_of("decision")}
    states = {r["file_date"]: r for r in rows_of("scheduler_state")}
    picks_by_date = _group(rows_of("pick_file"), "file_date")
    pick_state_by_date = {_DATE.search(rel).group(0): pick_file_state(files[rel], parsed[rel])
                          for rel, k in routed.items() if k == "pick_file"}
    obs_by_date = _group(rows_of(*OBSERVATION_KINDS), "file_date")
    unusable_by_date: dict[str, set[str]] = {}       # day evidence present but quarantined (Codex plan r3 #5)
    for rel, kind in routed.items():
        if kind in DAY_KINDS and kind != "pick_file" and parsed[rel].quarantined and (m := _DATE.search(rel)):
            unusable_by_date.setdefault(m.group(0), set()).add(kind)
    ledger: list[dict] = []
    for d in SEASON_DATES:
        ledger += day_rows(d, decision=decisions.get(d), pick_rows=picks_by_date.get(d, []),
                           pick_file=pick_state_by_date.get(d), state=states.get(d),
                           observations=obs_by_date.get(d, []), unusable=frozenset(unusable_by_date.get(d, ())))
    selections = [r for r in ledger if r["row_kind"] == "selection"]

    # §6 contest evidence and matching
    contest_rows = rows_of("contest_ledger")
    streaks_by_line, entered = line_round_streaks(contest_rows), entered_rounds(contest_rows)
    schedule_rows = rows_of("schedule")
    schedule_status = {rel[len("schedules/"):-len(".json")]: "incomplete" if parsed[rel].quarantined else "complete"
                       for rel, k in routed.items() if k == "schedule"}
    games_by_date: dict[str, set] = {}
    for g in schedule_rows:
        games_by_date.setdefault(g["query_date"], set()).add(g["game_pk"])
    rounds = rounds_lookup(rows_of("rounds"))
    lookups = {"rounds": rounds, "players": players_lookup(rows_of("players")),
               "units": units_lookup(rows_of("units")), "team_games": team_games(schedule_rows),
               "schedule_status": schedule_status}
    matches = resolve_duplicate_links([{**h, **match_slot(h, local_selections=selections, **lookups)}
                                       for h in slot_history(contest_rows)])
    round_of_date: dict[str, list[int]] = {}
    for rid, dates in rounds.items():
        if len(dates) == 1:
            round_of_date.setdefault(next(iter(dates)), []).append(rid)

    def day_context(d: str | None) -> dict:
        ids = round_of_date.get(d, [])
        complete = schedule_status.get(d) == "complete"
        return {"round_id": ids[0] if len(ids) == 1 else None,
                "scheduled_games": len(games_by_date.get(d, ())) if complete else None}

    for row in ledger:
        row.update(day_context(row["date"]))

    # §7 outcomes onto selections
    unit_status = unit_status_history(rows_of("units"))
    refusals: dict[tuple, list] = {}
    for r in rows_of("archive"):
        if r["archive_prefix"] == "refused_delivery":
            refusals.setdefault((r["file_date"], r["slot"], r["batter_id"], r["game_pk"]), []).append(
                (utc_iso(r["archived_at"]), r["archive_reason"]))
    picks_by_obs = {r["obs_id"]: r for r in rows_of("pick_file")}
    contest_by_obs = {r["obs_id"]: r for r in contest_rows}
    linked = {m["selection_id"]: m for m in matches if m["selection_id"]}

    def grade_raw(m: dict) -> str | None:
        if m["slot_result_state"] != "type_mismatch":
            return m["slot_result"]
        return canonical_json(json.loads(contest_by_obs[m["last_obs_id"]]["record_raw_json"])["result"])
    unlinked_keys = {(m["date"], m["batter_id"]) for m in matches
                     if not m["selection_id"] and m["match"] in ("ambiguous", "evidenced")}
    for row in selections:
        pick = picks_by_obs.get(row["pick_obs_id"])
        derived, source = derived_single_result(pick)
        row.update(local_slot_result_raw=pick["slot_result_raw"] if pick else None,
                   local_day_result_raw=pick["day_result_raw"] if pick else None,
                   local_slot_result_derived=derived, derivation_source=source, saver_available_before=None)
        row["local_norm"] = normalize_local(row["local_slot_result_raw"] or derived)
        m = linked.get(row["selection_id"])
        eligibility, at, basis = _eligibility(row, m, unit_status, refusals)
        row.update(game_eligibility=eligibility, game_eligibility_at=at, game_eligibility_basis=basis)
        if m is None:
            row["entry_status"] = "unknown"
            row["bts_outcome_status"] = ("match_ambiguous" if (row["date"], row["batter_id"]) in unlinked_keys
                                         else "unknown")
            continue
        row.update(entry_status="confirmed", match=m["match"], match_reason=m["match_reason"], round_id=m["round_id"],
                   unit_id=m["unit_id"], player_id=m["player_id"], slot_number=m["slot_number"],
                   entry_observed_at=m["first_seen"], contest_slot_grade_raw=grade_raw(m),
                   bts_outcome=m["slot_result"], bts_outcome_status=_outcome_status(m["slot_result"],
                                                                                    m["slot_result_state"]),
                   contest_round_result=m["round_result"], streak_after=m["round_streak"],
                   streak_before=streak_before(streaks_by_line.get(m["last_line_no"], {}), m["round_id"], entered),
                   contest_norm=normalize_contest(m["slot_result"]), contest_obs_id=m["last_obs_id"])
        row["local_vs_contest_disagreement"] = slot_disagreement(row["local_norm"], row["contest_norm"])

    # §5 contest-only rows
    for m in matches:
        if m["selection_id"]:
            continue
        status = {"unmapped": "unmapped", "ambiguous": "match_ambiguous"}.get(
            m["match"], _outcome_status(m["slot_result"], m["slot_result_state"]))
        ledger.append({"row_id": f"contest|{m['round_id']}|{m['unit_id']}|{m['player_id']}",
                       "row_kind": "contest_only", "date": m["date"], "slot": None, "selection_id": None,
                       **day_context(m["date"]), "round_id": m["round_id"],
                       "batter_id": m["batter_id"], "game_pk": m["game_pk"], "entry_status": "confirmed",
                       "match": m["match"], "match_reason": m["match_reason"], "unit_id": m["unit_id"],
                       "player_id": m["player_id"], "slot_number": m["slot_number"],
                       "entry_observed_at": m["first_seen"], "contest_slot_grade_raw": grade_raw(m),
                       "bts_outcome": m["slot_result"] if status == "graded" else None, "bts_outcome_status": status,
                       "contest_round_result": m["round_result"], "streak_after": m["round_streak"],
                       "contest_norm": normalize_contest(m["slot_result"]), "contest_obs_id": m["last_obs_id"]})

    # §8 dispositions (canonical links), recipes, then the output-phase checks
    dispositions: dict[str, str] = {}
    for r in rows_of("pick_file"):
        dispositions[r["obs_id"]] = "not_selected" if r["file_date"] in _SEASON else "outside_season_window"
    for r in rows_of("decision"):
        dispositions[r["obs_id"]] = "outside_season_window"
    unresolved_dates = {r["date"] for r in ledger if r.get("finalization") == "unresolved"
                        or r.get("reason") == "decision_unusable"}
    for r in rows_of("pick_file"):
        if r["file_date"] in unresolved_dates:
            dispositions[r["obs_id"]] = "unresolved_pick_file_view"
    for r in ledger:
        if r.get("pick_obs_id"):
            dispositions[r["pick_obs_id"]] = "canonical_selection"
        if r.get("decision_obs_id"):
            dispositions[r["decision_obs_id"]] = "canonical_decision"
    assign_dispositions(accounting, dispositions)
    summary, membership = evaluate_rules(files, RULES, mtimes, frozen_at=manifest["acquired_at_utc"])
    by_locator = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical_of = {r["pick_obs_id"]: r for r in ledger if r.get("pick_obs_id")}
    for m in membership:   # link every recipe row to the occurrence that accounts for its record (Codex plan r2 #5)
        source = by_locator.get((m["source_path"], f"slot={m['slot']}")) or by_locator.get((m["source_path"], "file"))
        link = canonical_of.get(source["obs_id"]) if source else None
        m.update(source_obs_id=source["obs_id"] if source else None, source_state=source["state"] if source else None,
                 source_reason=source["reason"] if source else None,
                 occurrence_disposition=source["disposition"] if source else None,
                 selection_id=link["selection_id"] if link else None,
                 canonical_bts_outcome=link.get("bts_outcome") if link else None)
    check_invariants(files, accounting, matches, ledger, membership, summary, SEASON_DATES)

    tables = {
        "season_2026_ledger.parquet": build_table(ledger, LEDGER_SCHEMA, sort_keys=["row_id"], name="ledger"),
        "season_2026_ledger_occurrences.parquet": build_table(accounting, OCCURRENCE_SCHEMA,
                                                              sort_keys=["source_path", "locator"], name="occurrences"),
        "season_2026_ledger_contest_slots.parquet": build_table(
            [{k: m.get(k) for k in CONTEST_SCHEMA.names} for m in matches], CONTEST_SCHEMA,
            sort_keys=["round_id", "unit_id", "player_id"], name="contest_slots"),
        "season_2026_ledger_reconciliation.parquet": build_table(membership, RECONCILIATION_SCHEMA,
                                                                 sort_keys=["rule_id", "source_path", "slot"],
                                                                 name="reconciliation"),
    }
    sels = [r for r in ledger if r["row_kind"] == "selection"]
    emitted = [a for a in accounting if a["state"] == "emitted"]
    saver = rows_of("saver_transitions")
    build = {"builder_version": BUILDER_VERSION, "code_sha": code_sha, "python_version": platform.python_version(),
             "pyarrow_version": pa.__version__, "environment_lock_sha256": uv_lock_sha256,
             "bundle_manifest_sha256": sha256_hex((Path(bundle_root) / "manifest.json").read_bytes()),
             "bundle_acquired_at_utc": manifest["acquired_at_utc"], "rules_fingerprint": rules_fingerprint(),
             "season_dates": [SEASON_DATES[0], SEASON_DATES[-1]],
             "row_kinds": _counts(ledger, "row_kind"), "finalization": _counts(sels, "finalization"),
             "commit_status": _counts(sels, "commit_status"), "history_status": _counts(sels, "history_status"),
             "entry_status": _counts(sels, "entry_status"),
             "bts_outcome_status": _counts([r for r in ledger if r["row_kind"] in ("selection", "contest_only")],
                                           "bts_outcome_status"),
             "match": _counts(matches, "match"), "match_reason": _counts(matches, "match_reason"),
             "game_eligibility": _counts(sels, "game_eligibility"),
             "local_vs_contest_disagreement": _counts(sels, "local_vs_contest_disagreement"),
             "schedule_status": _counts([{"s": schedule_status.get(d, "missing")} for d in SEASON_DATES], "s"),
             "occurrence_states": _counts(accounting, "state"),
             "exclusion_reasons": _counts([a for a in accounting if a["state"] == "excluded"], "reason"),
             "quarantine_reasons": _counts([a for a in accounting if a["state"] == "quarantined"], "reason"),
             "dispositions": _counts(emitted, "disposition"),
             "type_mismatch_rows_by_kind": _counts([r for rel, k in routed.items() for r in parsed[rel].rows
                                                    if r.get("type_mismatch_fields")], "source_kind"),
             "saver_transition_attempts": {"n": len(saver), "by_outcome": _counts(saver, "attempt_outcome")},
             "recipes": summary, "recipe_labels": recipe_labels(summary)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()                       # reserves the directory: a concurrent or repeated build fails here
    for name, table in tables.items():
        write_table(table, out / name)
    (out / "season_2026_ledger_build.json").write_text(json.dumps(build, indent=1, sort_keys=True) + "\n")
    (out / "season_2026_ledger_summary.md").write_text(_summary_md(build))
    return build


def _summary_md(build: dict) -> str:
    lines = ["# Season 2026 ledger — Phase 1 build summary", "",
             f"Builder `{build['builder_version']}` · code `{build['code_sha']}` · Python {build['python_version']} · "
             f"pyarrow {build['pyarrow_version']} · environment lock `{build['environment_lock_sha256']}` · bundle "
             f"manifest `{build['bundle_manifest_sha256']}` (acquired {build['bundle_acquired_at_utc']}) · recipe "
             f"rules `{build['rules_fingerprint']}`", ""]
    for key in ("row_kinds", "finalization", "commit_status", "history_status", "entry_status", "bts_outcome_status",
                "match", "match_reason", "game_eligibility", "local_vs_contest_disagreement", "schedule_status",
                "occurrence_states", "exclusion_reasons", "quarantine_reasons", "dispositions",
                "type_mismatch_rows_by_kind"):
        lines += [f"## {key}", *(f"- {k}: {v}" for k, v in build[key].items()), ""]
    s = build["saver_transition_attempts"]
    lines += ["## saver_transitions.jsonl (attempts, not consumption times)", f"- rows: {s['n']}",
              *(f"- {k}: {v}" for k, v in s["by_outcome"].items()), "",
              "## Recipe candidates (historical membership unverified)", "",
              "| rule | recipe | files | primary | legs | window | primaries | legs | published | fit |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule_id']} | {r['recipe']} | {r['files']} | {r['primary_grading']} | {r['leg_grading']} | "
              f"{r['window'][0]}→{r['window'][1]} | {r['primaries']} | {r['legs']} | "
              f"{r['published_primaries']}/{r['published_legs']} | {r['fit']} |" for r in build["recipes"]]
    lines += ["", *(f"- **{k}: {v}**" for k, v in build["recipe_labels"].items()), ""]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_compile.py -q`
Expected: `12 passed`. If the determinism test fails, find the unordered value (a set iteration or dict order reaching an output) and sort it; do not weaken the test.

- [ ] **Step 5: Run the whole ledger suite**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger -q`
Expected: `146 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/compile.py tests/scripts/season_ledger/test_compile.py
git commit -m "feat(ledger): offline compile pipeline, build-then-write deterministic outputs (task 10)"
```

---

### Task 11: Acquisition and CLI

**Files:**
- Create: `scripts/audit/season_ledger/acquire.py`, `scripts/audit/build_season_ledger.py`
- Test: `tests/scripts/season_ledger/test_acquire.py`

**Interfaces:**
- **Produces:** `acquire(*, snapshot_root, out_root, dates, fetch, now_utc) -> Path`, returning the manifest path. The CLI is `build_season_ledger.py acquire --snapshot --out` and `… compile --bundle --out [--code-sha] [--uv-lock]`.
- **Acquisition sources**, all read from the snapshot root:
  - the whole `data/picks/` tree
  - every `static_snapshots/{rounds,units}` capture
  - the first and last `static_snapshots/players` capture (I10)
  - the whole `final_grab_20260927/raw/static/`
  - `cron.log` and `journal_bts-scheduler_retained.txt`
  - one schedule response per date
- **Missing inputs:** an expected input that is absent becomes a declared `missing` entry. Dotfile markers are not inputs, and a non-empty target is refused.

- [ ] **Step 1: Write the failing test** — `tests/scripts/season_ledger/test_acquire.py`:

```python
import pytest

from scripts.audit.season_ledger.acquire import acquire
from scripts.audit.season_ledger.bundle import open_bundle
from tests.scripts.season_ledger.builders import gz


def test_acquire_copies_sources_declares_missing_inputs_and_seals(tmp_path):
    snap = tmp_path / "final-20260928"
    picks = snap / "data" / "picks"
    (picks / "2026-05-01").mkdir(parents=True)
    (picks / "2026-05-01.json").write_bytes(b'{"pick": {}}')
    (picks / "2026-05-01" / "decision.json").write_bytes(b"{}")
    static = snap / "data" / "leaderboard" / "static_snapshots"
    for feed in ("rounds", "units", "players"):
        (static / feed).mkdir(parents=True)
        (static / feed / ".last_sha256").write_text("x\n")
    (static / "rounds" / "20260704T030011Z.json.gz").write_bytes(gz(b'{"rounds": []}'))
    for stamp in ("20260704T030011Z", "20260801T030011Z", "20260928T123001Z"):
        (static / "players" / f"{stamp}.json.gz").write_bytes(gz(b'{"players": []}'))
    grab = snap / "data" / "leaderboard" / "final_grab_20260927" / "raw" / "static"
    grab.mkdir(parents=True)
    (grab / "002_players.json.gz").write_bytes(gz(b'{"players": []}'))
    (snap / "cron.log").write_text("log\n")

    def fetch(day):
        if day == "2026-05-02":
            raise OSError("timeout")
        return b'{"dates": []}'

    out = tmp_path / "bundle"
    acquire(snapshot_root=snap, out_root=out, dates=["2026-05-01", "2026-05-02"], fetch=fetch,
            now_utc=lambda: "2026-09-28T16:00:00.000000Z")
    manifest, files = open_bundle(out)
    assert files["picks/2026-05-01.json"] == b'{"pick": {}}'
    assert sorted(k for k in files if k.startswith("static/players/")) == [
        "static/players/20260704T030011Z.json.gz", "static/players/20260928T123001Z.json.gz"]
    assert files["static/units/NO_CAPTURES"] is None and "static/rounds/.last_sha256" not in files
    assert files["static/grab_20260927/002_players.json.gz"].startswith(b"\x1f\x8b")
    assert files["logs/cron.log"] == b"log\n" and files["logs/journal_bts-scheduler_retained.txt"] is None
    assert files["schedules/2026-05-01.json"] == b'{"dates": []}' and files["schedules/2026-05-02.json"] is None
    entries = {e["rel_path"]: e for e in manifest["entries"]}
    assert entries["schedules/2026-05-02.json"]["note"] == "fetch_failed:OSError"
    assert entries["picks/2026-05-01.json"]["source_path"] == "data/picks/2026-05-01.json"
    assert entries["picks/2026-05-01.json"]["source_mtime_utc"].endswith("Z")
    with pytest.raises(FileExistsError):
        acquire(snapshot_root=snap, out_root=out, dates=[], fetch=fetch, now_utc=lambda: "x")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_acquire.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.acquire'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/acquire.py`:
```python
"""Acquisition into a sealed evidence bundle (spec §3). Reads the frozen W0.7 snapshot; the only
network call is the MLB schedule fetch, whose failures become declared `missing` entries."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import BUILDER_VERSION
from .bundle import BundleEntry, write_manifest
from .ids import UTC_FORMAT, sha256_hex

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&gameType=R&hydrate=team"
STATIC = Path("data/leaderboard/static_snapshots")
GRAB_STATIC = Path("data/leaderboard/final_grab_20260927/raw/static")
LOGS = ("cron.log", "journal_bts-scheduler_retained.txt")


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(UTC_FORMAT)


def _captures(directory: Path) -> list[Path]:
    """Files in a capture directory; dotfiles (.last_sha256 markers) are not inputs."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def acquire(*, snapshot_root, out_root, dates: list[str], fetch: Callable[[str], bytes],
            now_utc: Callable[[], str]) -> Path:
    snap, out_root = Path(snapshot_root), Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"bundle directory not empty: {out_root} (a new acquisition is a new version)")
    picks = snap / "data" / "picks"
    if not picks.is_dir():
        raise FileNotFoundError(f"no picks tree under {snap}")
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[BundleEntry] = []

    def copy(src: Path, rel: str) -> None:
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        data = dest.read_bytes()
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=src.relative_to(snap).as_posix(), source_mtime_utc=_mtime_utc(src)))

    def missing(rel: str, note: str) -> None:
        entries.append(BundleEntry(rel_path=rel, status="missing", note=note))

    for src in sorted(p for p in picks.rglob("*") if p.is_file()):
        copy(src, "picks/" + src.relative_to(picks).as_posix())
    for feed in ("rounds", "units"):
        found = _captures(snap / STATIC / feed)
        for src in found:
            copy(src, f"static/{feed}/{src.name}")
        if not found:
            missing(f"static/{feed}/NO_CAPTURES", "no captures found")
    players = _captures(snap / STATIC / "players")
    for src in sorted({players[0], players[-1]}) if players else []:
        copy(src, f"static/players/{src.name}")
    if not players:
        missing("static/players/NO_CAPTURES", "no captures found")
    grab = _captures(snap / GRAB_STATIC)
    for src in grab:
        copy(src, f"static/grab_20260927/{src.name}")
    if not grab:
        missing("static/grab_20260927/NO_FILES", "grab static directory empty or absent")
    for name in LOGS:
        if (snap / name).is_file():
            copy(snap / name, f"logs/{name}")
        else:
            missing(f"logs/{name}", "not in snapshot")
    for day in dates:
        rel = f"schedules/{day}.json"
        try:
            data = fetch(day)
        except Exception as exc:   # recorded, never fatal: the bundle declares the gap
            missing(rel, f"fetch_failed:{exc.__class__.__name__}")
            continue
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=SCHEDULE_URL.format(date=day), source_mtime_utc=now_utc()))
    return write_manifest(out_root, entries, acquired_at_utc=now_utc(), builder_version=BUILDER_VERSION,
                          source_root=str(snap))
```

`scripts/audit/build_season_ledger.py`:
```python
"""Season 2026 ledger, Phase 1 — `acquire` (box: reads the frozen snapshot, fetches MLB schedules)
and `compile` (anywhere, offline, from the sealed bundle).

  .venv/bin/python scripts/audit/build_season_ledger.py acquire \
      --snapshot data/hetzner_results/season_2026_snapshot/final-20260928 \
      --out data/hetzner_results/season_2026_ledger_evidence/v1
  .venv/bin/python scripts/audit/build_season_ledger.py compile \
      --bundle data/hetzner_results/season_2026_ledger_evidence/v1 \
      --out data/validation/season_2026_ledger/<sha>-<run-id> --code-sha <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audit.season_ledger.acquire import SCHEDULE_URL, acquire  # noqa: E402
from scripts.audit.season_ledger.compile import SEASON_DATES, compile_bundle  # noqa: E402
from scripts.audit.season_ledger.ids import UTC_FORMAT  # noqa: E402

USER_AGENT = "bts-season-ledger/1 (one-pass audit acquisition)"


def _fetch(day: str) -> bytes:
    req = urllib.request.Request(SCHEDULE_URL.format(date=day), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    time.sleep(0.5)   # courteous pacing to a public API
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Season 2026 ledger, Phase 1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compile")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--code-sha", default=None)
    c.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    args = ap.parse_args(argv)
    if args.cmd == "acquire":
        path = acquire(snapshot_root=args.snapshot, out_root=args.out, dates=SEASON_DATES, fetch=_fetch,
                       now_utc=lambda: datetime.now(timezone.utc).strftime(UTC_FORMAT))
        print(f"sealed {path}")
        return 0
    lock_sha = hashlib.sha256(args.uv_lock.read_bytes()).hexdigest() if args.uv_lock.is_file() else None
    build = compile_bundle(args.bundle, args.out, uv_lock_sha256=lock_sha, code_sha=args.code_sha)
    print(f"row kinds {build['row_kinds']} | matches {build['match']} | recipes {build['recipe_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass, then the suites**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger -q`
Expected: `147 passed`.

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q`
Expected: the previous fast-suite count + 147, all passing. Report any failure by name.

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit/build_season_ledger.py compile --help`
Expected: usage text listing `--bundle`, `--out`, `--code-sha`, `--uv-lock`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/acquire.py scripts/audit/build_season_ledger.py tests/scripts/season_ledger/test_acquire.py
git commit -m "feat(ledger): acquisition into a sealed bundle and the build CLI (task 11)"
```

---

### Task 12: Fresh-session Codex code review (no data access)

**Files:** `.codex-review/season-ledger-code/prompt-rN.md` (gitignored); each round is archived to `docs/audit/<date>-season-ledger-codex-code-rN.md`.

- [ ] **Step 1: Start a new Codex session** in its own herdr tab (not `bts-codex`). The session that reviewed the design and this plan is anchored; this follows memory `fresh-reviewer-after-approve`.
- [ ] **Step 2: Write a self-contained brief:**
  - **Stance:** assume defects exist.
  - **Scope:** the Task 1–11 commit range against spec v4, this plan's Interpretations I1–I14 and Review Focus.
  - **Part 1 is blind:** it may not open the plan-review or design-review archives until its own findings are on disk. Part 2 reconciles against them.
  - **Hard rule, no data access:** nothing under `data/`, no snapshots, no ssh, no network.
  - **Asks:**
    - mutant-proven false greens
    - determinism
    - census/anti-join soundness
    - the §5 table and set rules
    - matching
    - the frozen recipe rules
    - real-snapshot hazards
  - **Deliverable:** a file with BLOCKER/SHOULD/NIT findings and SIGN/BLOCK, then exactly `DONE`.
- [ ] **Step 3: Run it** through the herdr round-trip (consulting-codex §Transport): record the recipient identity, Monitor the wait, and validate the file before acceptance.
- [ ] **Step 4: Triage every finding** as real, false flag or over-engineered, with evidence. Fix real ones test-first, then re-run `tests/scripts/season_ledger` and the fast suite. Commit each fix with its finding number.
- [ ] **Step 5: Re-review in the same fresh session** until SIGN, capped at 3 rounds. Surface any unresolved disagreement to Eric, archive every round under `docs/audit/`, and commit.

---

### Task 13: Real run on the box, records

Every box job goes through one runner, `run.sh <run-id> <mode>`, with modes `acquire`, `compile2` and `backup`. The runner:
- prints `RUN=<id> START <mode>` first and `RUN=<id> EXIT=<status>` last, and exits with that status (the EXIT trap is installed before anything else can fail, including a missing run id)
- never touches production code, config or services
- is waited on by a read-back of the finished run from its log, never by a live tail

- [ ] **Step 1: Predeclare the read (exposure register X-19) and commit before any real data is compiled.** Append this row to `docs/audit/2026-09-22-exposure-register.md`, fill in `<date>`, then commit with the command below.

> **X-19 — W1.1 season ledger, Phase 1 build (predeclared <date>).**
>
> **What is read:** the frozen W0.7 snapshot (production picks, decisions, scheduler state, lineup evolution, contest ledger, BTS static captures) through `scripts/audit/season_ledger` at the reviewed commit, compiled offline from the sealed bundle `season_2026_ledger_evidence/v1`.
>
> **What the build computes and prints:** occurrence, row-kind, commit, history, entry, match and outcome-status counts; the count of local-vs-contest slot disagreements (the C-03 check); and the per-rule recipe totals for the frozen rules and evaluator `5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f`.
>
> **Not computed:** no rates, hit percentages or model comparisons.
>
> **Scope of this row:** any later analysis of the ledger's outcomes is a new read with its own row.

```bash
cd /Users/eric/projects/bts && git add docs/audit/2026-09-22-exposure-register.md && \
git commit -m "docs(wrap): X-19 predeclared before the season-ledger build"
```

- [ ] **Step 2: Ship the reviewed code to a unique directory (no deploy) and install the runner.** Both commands run from the Mac, and the heredoc is written verbatim.

```bash
cd /Users/eric/projects/bts && SHA=$(git rev-parse HEAD) && ssh bts-hetzner "test ! -e /tmp/ledger_code_$SHA" && \
git archive "$SHA" scripts/__init__.py scripts/audit/__init__.py scripts/audit/season_ledger scripts/audit/build_season_ledger.py \
  | ssh bts-hetzner "mkdir /tmp/ledger_code_$SHA && tar -x -C /tmp/ledger_code_$SHA && echo $SHA > /tmp/ledger_code_$SHA/CODE_SHA"
```

```bash
cd /Users/eric/projects/bts && SHA=$(git rev-parse HEAD) && ssh bts-hetzner "cat > /tmp/ledger_code_$SHA/run.sh" <<'EOF'
#!/bin/bash
# usage: run.sh <run-id> acquire|compile2|backup — runs from the production checkout (read-only for the ledger).
# Prints "RUN=<id> START <mode>" first and "RUN=<id> EXIT=<status>" last, and exits with that status.
set -uo pipefail
run="${1:-}"; mode="${2:-}"
echo "RUN=${run:-unnamed} START $mode"
trap 'status=$?; echo "RUN=${run:-unnamed} EXIT=$status"; exit $status' EXIT
[ -n "$run" ] || { echo "usage: run.sh <run-id> acquire|compile2|backup"; exit 2; }
here=$(cd "$(dirname "$0")" && pwd) || exit 3
sha=$(cat "$here/CODE_SHA") || exit 3
py=/home/bts/projects/bts/.venv/bin/python
cli="$here/scripts/audit/build_season_ledger.py"
bundle=data/hetzner_results/season_2026_ledger_evidence/v1
cd /home/bts/projects/bts || exit 3
case "$mode" in
  acquire)
    "$py" "$cli" acquire --snapshot data/hetzner_results/season_2026_snapshot/final-20260928 --out "$bundle"
    ;;
  compile2)
    a="data/validation/season_2026_ledger/$sha-$run"; b="/tmp/ledger_check_$sha-$run"
    "$py" "$cli" compile --bundle "$bundle" --out "$a" --code-sha "$sha" || exit $?
    "$py" "$cli" compile --bundle "$bundle" --out "$b" --code-sha "$sha" || exit $?
    "$py" - "$a" "$b" "$run" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
a, b, run = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
na, nb = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
if na != nb or len(na) != 6:
    sys.exit(f"file sets differ: {na} vs {nb}")
diff = [n for n in na if (a / n).read_bytes() != (b / n).read_bytes()]
if diff:
    sys.exit(f"bytes differ: {diff}")
fp = json.loads((a / "season_2026_ledger_build.json").read_text())["rules_fingerprint"]
if fp != "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f":
    sys.exit(f"rules fingerprint {fp} is not the predeclared one")
receipt = {"run": run, "accepted_at_utc": datetime.now(timezone.utc).isoformat(), "files": na,
           "compared_with": str(b), "rules_fingerprint": fp}
tmp = a / "ACCEPTED.json.tmp"                    # written and synced first, then renamed: never a partial receipt
with open(tmp, "w") as fh:
    fh.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, a / "ACCEPTED.json")
print(f"accepted {a}: {len(na)} identical files; rules fingerprint as predeclared")
PY
    ;;
  backup)
    start=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
    set +u; set -a                               # as production's cron does: no nounset, and the status is checked
    . ./.env || { echo "cannot source .env"; exit 3; }
    set +a; set -u
    UV_CACHE_DIR=/tmp/uv-cache /home/bts/.local/bin/uv run bts backup run --set archive || exit $?
    "$py" - "$start" "$bundle" <<'PY'
import hashlib, json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from bts.data.backup import restic_bin, restic_env
start, bundle = datetime.fromisoformat(sys.argv[1]), Path(sys.argv[2])
env = restic_env(dict(os.environ)); rb = restic_bin(env)
def restic(*args, text=True):
    return subprocess.run([rb, *args], env=env, capture_output=True, text=text, check=True).stdout
def when(t):          # restic prints nanoseconds; keep microseconds
    return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", t))
snaps = [s for s in json.loads(restic("snapshots", "--tag", "archive", "--json")) if when(s["time"]) >= start]
if len(snaps) != 1:
    sys.exit(f"expected exactly one archive snapshot since {start}, found {len(snaps)}")
sid, base = snaps[0]["id"], "/data/hetzner_results/season_2026_ledger_evidence/v1"
manifest = json.loads((bundle / "manifest.json").read_text())
if hashlib.sha256(restic("dump", sid, base + "/manifest.json", text=False)).hexdigest() != \
        hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest():
    sys.exit(f"snapshot {sid}: backed-up manifest differs from the sealed one")
nodes = [json.loads(line) for line in restic("ls", sid, base, "--recursive", "--json").splitlines() if line.strip()]
listed = {n["path"] for n in nodes if n.get("type") == "file"}      # the snapshot header line has no "type"
declared = {f"{base}/{e['rel_path']}" for e in manifest["entries"] if e["status"] == "present"}
missing = sorted(declared - listed)
if missing:
    sys.exit(f"snapshot {sid}: {len(missing)} bundle members missing, e.g. {missing[:3]}")
print(f"snapshot {sid}: manifest identical; all {len(declared)} bundle members present")
PY
    ;;
  *)
    echo "usage: run.sh <run-id> acquire|compile2|backup"; exit 2
    ;;
esac
EOF
ssh bts-hetzner "bash -n /tmp/ledger_code_$SHA/run.sh && echo syntax-ok"
```

- [ ] **Step 3: Acquire** as a transient unit. Start it, wait for the run's `EXIT` line, and read the whole run back:

```bash
cd /Users/eric/projects/bts && SHA=$(git rev-parse HEAD) && RUN=$(date -u +%Y%m%dT%H%M%SZ)-acquire-$(uuidgen | cut -c1-8 | tr 'A-Z' 'a-z') && echo "$RUN" && \
ssh bts-hetzner "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemd-run --user --unit=bts-ledger-$RUN --collect \
  -p StandardOutput=append:/home/bts/logs/ledger.log -p StandardError=append:/home/bts/logs/ledger.log \
  /bin/bash /tmp/ledger_code_$SHA/run.sh $RUN acquire"
```

```bash
RUN=<the id printed above>; box() { ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 bts-hetzner "$@"; }; \
deadline=$((SECONDS + 7200)); until box "grep -q 'RUN=$RUN EXIT=' ~/logs/ledger.log" || [ $SECONDS -ge $deadline ]; do sleep 20; done; \
box "awk '/RUN=$RUN START/,/RUN=$RUN EXIT=/' ~/logs/ledger.log; grep -q 'RUN=$RUN EXIT=' ~/logs/ledger.log || echo 'NO EXIT LINE within 2 hours'"
```

Run the wait with Bash `run_in_background`. Require `RUN=<id> EXIT=0`, preceded by `sealed …/manifest.json`. Every ssh call is non-interactive and times out on a stalled connection, and the wait gives up after two hours. `NO EXIT LINE` means the run has not finished, never started, or was killed before its trap ran. Check `systemctl --user status bts-ledger-<id>` on the box: if the unit is still running, run the wait again; otherwise read `journalctl --user -u bts-ledger-<id>` and treat the run as a stop. Then check the manifest; this exits non-zero on any missing entry outside `schedules/`:

```bash
ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python -c "
import json, sys, collections
m = json.load(open(\"data/hetzner_results/season_2026_ledger_evidence/v1/manifest.json\"))
print(len(m[\"entries\"]), dict(collections.Counter(e[\"rel_path\"].split(\"/\")[0] for e in m[\"entries\"])))
miss = [(e[\"rel_path\"], e[\"note\"]) for e in m[\"entries\"] if e[\"status\"] == \"missing\"]
print(\"missing:\", miss)
sys.exit(1 if any(not p.startswith(\"schedules/\") for p, _ in miss) else 0)"'
```

Expected: about 3,600 entries (picks tree, 173 rounds, 2,335 units, 2 players, 4 grab files, 2 logs, 187 schedules), with missing entries at most among schedules.

- [ ] **Step 4: Compile twice, compare and accept.** Same pattern, with `-compile2-` in place of `-acquire-` in the run id and mode `compile2`. Require `RUN=<id> EXIT=0`, preceded by `accepted data/validation/season_2026_ledger/<sha>-<run>: 6 identical files; rules fingerprint as predeclared`. Only a directory whose `ACCEPTED.json` parses and names this run and its six files is a published build; the receipt is renamed into place only after it is fully written.
- **An `InvariantError` or any other `EXIT` is a stop.** Diagnose with the systematic-debugging skill, fix test-first on the Mac, commit, then re-run Steps 2 and 4 under the new SHA. Skip Step 3: the sealed v1 bundle is reused and re-verified by `open_bundle`, and each attempt writes its own `<sha>-<run>` directories.
- **Keep failed attempt directories**; the memo names them. A new acquisition is only ever an intentional `v2`.

- [ ] **Step 5: Back up and verify the exact snapshot.** Same pattern, with `-backup-` in place of `-acquire-` in the run id and mode `backup`. Require `RUN=<id> EXIT=0`, preceded by `snapshot <id>: manifest identical; all <n> bundle members present`, and record the snapshot id.

- [ ] **Step 6: Record.** Copy `season_2026_ledger_build.json`, `season_2026_ledger_summary.md` and `ACCEPTED.json` from the accepted directory to the Mac. Then:
- Write `docs/audit/<date>-season-ledger.md`, covering:
  - code sha, bundle manifest sha, environment lock, Python/pyarrow, rules fingerprint and restic snapshot
  - input coverage and `missing` entries
  - row kinds, finalization, commit, history, entry and outcome-status counts
  - match counts and reasons (incl. `player_unknown`, `selection_game_pk_unrecorded`, `team_schedule_*`)
  - occurrence states, exclusion and quarantine reasons, and dispositions
  - type-mismatch rows by kind
  - the disagreement count
  - the recipe table and labels
  - failed attempts, if any
  - known limits: Phase 1 scope, saver unknown, eligibility only from July unit captures, the §12 local-grader finding
- Complete X-19 with the run's facts.
- Update the W1.1 row in `docs/audit/2026-season-wrap-index.md`.
- Commit.

- [ ] **Step 7: Optional result review.** If anything in Step 6 looks surprising, run a Codex round on the produced outputs; data access is allowed for this one.

---

## Self-Review (done while writing)
- **Spec coverage:**
  - §3 → Tasks 1, 11, 13.
  - §4 (typed, lossless, persisted) → Tasks 1–5, 9, 10.
  - §5 → Task 8.
  - §6 → Tasks 4–6, 10.
  - §7 → Tasks 7, 10.
  - §8 (census anti-join, output checks, universe membership linked to occurrences, spec labels) → Tasks 9–10.
  - §9 (`LEDGER_SCHEMA`) → Tasks 8, 10.
  - §10 → Task 10.
  - §11 → Tasks 2–10.
  - §12 → not built.
  - Determinism → Tasks 1, 10, 13.
- **Placeholders:** none; every code step carries its code.
- **Code identity:** every code block in this plan is generated from the tested scratch tree. The replay of this document's fences passed 147 tests.
- **Mutants:** 83 planted defects were killed at their intended tests (the ledger is in the commit message). They include:
  - a probe for each Codex r2 and r3 finding
  - a comment-only edit to the shared JSON decoder, and a time-zone change, each caught only by the fingerprint
  - Codex's surviving mutants: G1 counting `suspended` (now fails the fingerprint test and the truth table) and F1 compiled with IGNORECASE (now fails the fingerprint test) Rev 2's no-op mutant (a `pick_locked` probe planted after `commit_status` lost its state argument) and its crash-only kill were replaced by real probes.
- **Type consistency:**
  - `selection_id` is built only by `rows.selection_id`.
  - Parsed-row keys used downstream match their producers.
  - Every ledger key is in `LEDGER_SCHEMA` (`build_table` refuses unknown columns).
- **Review Focus:** each line has its test in the named task.
