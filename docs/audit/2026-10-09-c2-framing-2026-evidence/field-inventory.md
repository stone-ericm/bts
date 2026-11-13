# 2026 framing test: the declared-field inventory (round 3, written before any round-3 code)

Written by the lead (bts-lead3) on 2026-10-09, before writing any round-3 test or fix, against the code at `39793e1`
(the branch as found: `fccb6b9` plus two documentation commits). It applies the lesson of the three BLOCKs (c1, c2, f1):
every acceptance check so far compared a run's records with each other, so coherent forgeries passed. This inventory
lists **every field a run or the process declares** and, for each, **the trusted source it is checked against**. A
field with no trusted check is a finding, fixed test-first in this round (`round3-tdd-ledger.md`), or is marked
REPORTING with the reason no decision reads it.

**Trusted sources (the only things an acceptance check may compare a declaration with):**
- **PINS** — the admission record's `input_pins`, bound by the inputs row's digest (recorded after X-37), every file
  read through `admission.read_pinned` (the parsed bytes are the hashed bytes).
- **CONST** — constants of the admitted code (`f26`, `screen`): the admission gate binds the executable closure to the
  reviewed commit, so these are the frozen design's constants as reviewed.
- **PROD** — production constants at the admitted commit: `bts.model.predict.LGB_PARAMS`, `BLEND_CONFIGS`,
  `bts.features.compute.FEATURE_COLS` / `STATCAST_COLS`, the feature settings.
- **GIT** — git at the exposure commit and HEAD: `admission_check`, `accepted_identity`, `head_admitted`,
  `source_inventory`, the after-X-37 chronology of the inputs and preparation rows.
- **REG** — the register at HEAD: Eric's allowance row (source token `Eric`), the inputs row, the preparation row.
- **C1** — the launcher's own records under C1's own rules: `PENDING_<unit>.json` (written by the launcher before
  `systemd-run`), `TERMINAL_<unit>.json` (written by the guard), judged by `launch.terminal_problems`;
  `RECONCILED_<unit>.json` as the exact deterministic record `launch.reconcile` writes; `compute_ledger.tsv` through
  `ledger.read_tsv` / `ledger.total_hours`.
- **RETAINED→RECOMPUTED** — a pure function of retained bytes under the admitted code: scorecards from profiles, diffs
  from scorecards, summaries from diffs, rank-1 vectors from profiles.
- **PINNED→DERIVED** — a derivation from the pinned inputs, never from a run: the calendar, the scoreable batter-games
  and their original-portion labels, each day's side-games with PA counts, each side-game's team and catcher by arm,
  the as-of values (recomputed in the aggregate), the resumed-flag totals.

**Status:** CHECKED (source) · **FINDING** (fixed this round; its RED/GREEN is in `round3-tdd-ledger.md`) · REPORTING
(declared only; the reason it is not relied on is stated) · BOUND (a shape or range check that refuses, not an
independent verification of the value: the field is relied on only to that extent).

**Revision 2 (round 4, after review f1's round 2, `…-code-codex-f1r2.md`):** its corrections are applied in place and
marked "(r4)": the row count is stated as measured by script; `wall_s` is BOUND, not REPORTING; the preparation and
expectation records and the off-launcher rows are listed field by field; TERMINAL's failure-variant fields are listed;
the register the gates read is the checked-out text, not `git show HEAD:`; and the three round-4 findings (B1 the
preparation chain, B2 the accounting, B3 other invocations) have their rows. Their RED/GREEN is in
`round4-tdd-ledger.md`.

## A. `manifest.json` (one per run)

| Field | Written from | Trusted check | Status |
|---|---|---|---|
| `schema` | constant | `== "c2_framing_2026_run_v1"` (CONST) | CHECKED |
| `head` | `git rev-parse HEAD` at run | 40 hex; `head_admitted`: a commit of this repository, the exposure commit its ancestor, no closure file differs from the reviewed commit (GIT) | CHECKED |
| `claim_sha256` | sha256 of `CLAIM.json` as written | `== sha256(retained CLAIM.json)` | CHECKED |
| `seed` | the argument | `== seed` of the namespace `seed_<seed>` the run directory sits in | CHECKED |
| `identity` (5 keys) | the admission gate | exactly `IDENTITY_KEYS`; `== accepted_identity` at the exposure commit + the admission record's sha256 (GIT) | CHECKED |
| `input_pins` | the admission record | exactly `INPUT_NAMES`, 64 hex each; `== admitted pins` (PINS) | CHECKED |
| `inputs_digest` | `pins_digest` | `== pins_digest(manifest pins)` | CHECKED |
| `test_season` | constant | `== 2026` (CONST) | CHECKED |
| `arms` | constant | `== ARMS` (CONST) | CHECKED |
| `basis` | `screen.BASIS` | `==` (CONST) | CHECKED |
| `retrain_every` | `screen.RETRAIN_EVERY` | `==` (CONST) | CHECKED |
| `lgb_params` | `dict(LGB_PARAMS)` at run | **only `deterministic` and `force_row_wise` were checked.** Now: `== {**LGB_PARAMS at the admitted commit, deterministic: True, force_row_wise: True}`, no key more or less (PROD) | **FINDING F2a** (f1 F2) |
| `blend_configs` | **was not recorded** | Now recorded per arm as `[name, cols]` lists and `== screen.blend_configs("baseline" / "A")` at the admitted commit (PROD, CONST) | **FINDING F2b** (f1 F2) |
| `feature_settings` | `screen.check_settings()` (refuses at run if the live module values differ) | `== screen.SETTINGS` (CONST) | CHECKED |
| `scoring` | `screen.SCORING` | `==` (CONST) | CHECKED |
| `env.BTS_LGBM_DETERMINISTIC` | the environment | `== "1"` | CHECKED |
| `env.BTS_LGBM_RANDOM_STATE` | set by `run` | `== str(seed)` | CHECKED |
| `env.TZ` | the environment | none | REPORTING — no registered computation reads it (`ny_clock` names its zone; dates are ISO strings) |
| `self_check.identical` | the run's own feature frame | required `True` at `validate_run` (a declaration); the aggregate recomputes the features from the pinned inputs and requires the recomputed self-check identical (PINNED→DERIVED) | CHECKED at the aggregate; a declaration at `validate_run` (disclosed) |
| `self_check.rows`, `self_check.max_abs_diff` | the run | none | REPORTING |
| `calendar` | `expected_calendar(pa_2026)` at run | `== Trusted.calendar` from the pinned `pa_2026` (PINNED→DERIVED) | CHECKED |
| `allowance` (`cap`, `budget`, `first_unit_stop`) | `allowance(register)` at run | **was not checked at acceptance.** Now `== allowance(register at HEAD)` with `cap == ledger.CAP_H` (REG, CONST) | **FINDING F3a** (f1 F3) |
| `launcher_unit` | the kernel cgroup + `PENDING` at run | unit name pattern and seed index (CONST); clean `TERMINAL` and `RECONCILED` (C1). **Not read: `PENDING`; not checked: `TERMINAL.budget_seconds`; `RECONCILED` compared field by field, not as the exact record.** Now see §J | **FINDING F3b** (f1 F3) |
| `resumed_portion_rows` ({season: rows}) | `screen.resumed_counts` on the **feature frame** | **none.** Now computed from the pinned PA rows (`raw_df`), the 2026 entry checked at `validate_run` against the pinned `pa_2026`, the whole dict checked at the aggregate against the loaded pinned inputs (PINNED→DERIVED) | **FINDING F5a** (the lead's) |
| `resumed_flag_2026` ({flagged, unflagged}) | `resumed_totals(raw_df)` | **none.** Now `== Trusted.resumed_flag_2026` from the pinned `pa_2026` (PINNED→DERIVED) | **FINDING F5b** (the lead's) |
| `features_cpu_s` | `cpu_seconds() - t0` | **none.** Now finite, `>= 0`, `<= results.total_cpu_s` (same clock origin) | **FINDING F5c** (the lead's) |

## B. `results.json`

| Field | Written from | Trusted check | Status |
|---|---|---|---|
| `seed`, `head` | the run | `== seed`; `== manifest.head` | CHECKED |
| `units` | see §C; `== units.json` | | CHECKED |
| `total_cpu_s` | `cpu_seconds() - t0` | finite, `>= 0`, `<= TERMINAL.cpu_seconds` (C1), `>= sum(units.cpu_s)` | CHECKED |
| `p_at_1` ({arm: {"2026": p}}) | the scorecards | `==` the scorecard recomputed from the retained profiles; `== sum(rank-1 hits) / len(calendar)` (RETAINED→RECOMPUTED) | CHECKED |
| `rank1` ({arm: [0/1 per calendar date]}) | the profiles | `== rank1(retained profiles, calendar)` (RETAINED→RECOMPUTED) | CHECKED |
| `arms` ({arm: summary}) | the diffs | `== arm_summary(retained diff)`, the diff `== diff_scorecards(retained scorecards)` (RETAINED→RECOMPUTED) | CHECKED |

## C. `units` (in `results.json`, and `units.json`, which must equal it)

| Field | Written from | Trusted check | Status |
|---|---|---|---|
| `arm`, `season` | the loop | `== UNIT_ORDER` (CONST) | CHECKED |
| `cpu_s` | `cpu_seconds()` around the walk-forward | finite, `>= 0`, `sum <= total_cpu_s`; **the first unit (baseline) `<= allowance.first_unit_stop * 3600`** (REG) — the stop the run enforces on itself was not re-checked at acceptance | **FINDING F3a** (f1 F3) |
| `wall_s` | `time.monotonic()` | finite and `>= 0`; a negative or non-finite value refuses the run | BOUND (r4: it was marked REPORTING, but the bound refuses, so it is relied on to that extent) |
| `labels.changed`, `labels.void_dropped` | `screen.relabel` | none possible: the walk-forward's pre-relabel labels are not retained. The retained labels themselves are checked row by row (§F) | REPORTING — no decision reads them |
| `catcher.counts` (side_games / pa_rows × 3 reasons) | `ArmTransform` | `==` the counts re-derived from the retained evidence, which is itself re-derived from the pinned table, `pa_2026` and expectation (§G) | CHECKED |
| `catcher.identified_ids` | `ArmTransform` | `== sorted ids` of the re-derived evidence | CHECKED |

## D. `CLAIM.json`, `STOPPED.json`

| Field | Trusted check | Status |
|---|---|---|
| `CLAIM.run`, `CLAIM.code` | `== run directory name`; `== manifest.head` | CHECKED |
| `CLAIM.pid`, `CLAIM.claimed_utc` | none | REPORTING |
| `STOPPED.json` present | the run is refused | CHECKED |
| `STOPPED.*` fields | none; a stopped run is never accepted | REPORTING |

## E. `profiles_<arm>_2026.parquet` (columns: `PROFILE_COLUMNS` + `ESTIMATED_PA_EXTRA_COLUMNS` + `season`)

| Column | Trusted check | Status |
|---|---|---|
| `date` | year 2026; `==` the batter-game's trusted official date (PINNED→DERIVED) | CHECKED |
| `rank` | integers `1..n` per day, `n <= 10`; exactly one rank 1 per calendar date and no other date. **Not checked: that ranks follow `p_game_hit`.** Now `p_game_hit` must be non-increasing with rank within every date; exact ties between adjacent ranks are counted and reported (an exact tie cannot be ordered from retained evidence; see the ledger's disclosed limit) | **FINDING F1** (f1 F1) |
| `batter_id`, `game_pk` | integer columns; `(batter, game)` a trusted scoreable batter-game of that date, once per day (PINNED→DERIVED) | CHECKED |
| `p_game_hit` | numeric, finite, in `[0, 1]`; **the value is the model's output and cannot be re-derived without retraining** | CHECKED (shape); the value is a disclosed limit |
| `actual_hit` | `==` the trusted original-portion label (PINNED→DERIVED) | CHECKED |
| `p_game_hit_basis` | `== "estimated_pa"` (CONST) | CHECKED |
| `season` | integer `== 2026` | CHECKED |
| `n_pas`, `est_pas`, `starter_pas`, `reliever_pas`, `source_n_pas`, `p_hit_vs_reliever`, `total_batter_games`, `starter_matchup_batter_games`, `dropped_no_starter_matchup` | none; the scorecard scores `date`, `rank`, `season`, `actual_hit`, `p_game_hit` only | REPORTING |
| (the set) which ten batter-games a day are retained | not re-derivable without the day's predictions | disclosed limit |

## F. `catcher_<arm>_2026.parquet` (`EVIDENCE_COLS`)

| Column | Trusted check | Status |
|---|---|---|
| columns | exactly `EVIDENCE_COLS` | CHECKED |
| (`date`, `game_pk`, `fielding_side`) | exactly the trusted side-games, once each (pinned `pa_2026`) | CHECKED |
| `n_pa` | `==` the trusted PA-row count of that side-game | CHECKED |
| `team_id` | `==` the pinned table's | CHECKED |
| `catcher_id` | `==` the pinned table's starter proxy (A-posted) / the projection from the pinned table (A-projected) | CHECKED |
| `value` | `==` the pinned expectation exactly (`validate_run`); `==` the as-of value recomputed from the pinned inputs (aggregate) | CHECKED |
| `reason` | `==` derived from the catcher and the expected value | CHECKED |

## G. `scorecard_<arm>.json`, `diff_<arm>.json`

| Field | Trusted check | Status |
|---|---|---|
| every scorecard key but `timestamp` | `==` `compute_full_scorecard(retained profiles, **SCORING)` (RETAINED→RECOMPUTED); `p_at_1_by_season` keys exactly `["2026"]` | CHECKED |
| `timestamp` | none | REPORTING |
| every diff key | `== diff_scorecards(retained baseline card, retained arm card)` | CHECKED |

## H. The launcher's records (C1)

| Record / field | Trusted check | Status |
|---|---|---|
| `PENDING_<unit>.json` | **was not read at acceptance.** Now: exists (root or `jobs/`, one version), `unit == manifest.launcher_unit`, `declared_cpu_hours == allowance.budget`, `limit_cpu_seconds == int(budget * 3600)` (the C1 gate reserves the full declared budget whenever it admits a launch), `max_hours == MAX_WALL_H` (the reviewed wrapper's argument) (C1, REG, CONST) | **FINDING F3b** |
| `PENDING.written_utc` | none | REPORTING |
| `TERMINAL_<unit>.json` | `unit`, `result == "exit"`, integer `rc == 0`, finite `cpu_seconds` were checked. **`budget_seconds` was not.** Now `launch.terminal_problems(TERMINAL, PENDING, unit) == []`: the budget equals PENDING's limit, a known result, a measured CPU total within the budget (C1's own rule) | **FINDING F3b** |
| `TERMINAL.act_at_seconds`, `poll_s`, `slack_s`, `ncpu`, `started_utc`, `ended_utc`, `leftover` | none | REPORTING |
| `TERMINAL.reason`, `signal`, `payload_empty` (the guard's failure variants: `guard-error`, `terminated`) | none; a receipt with such a result is not a clean exit and the run is refused; the variant is reported in `other_units` (r4) | REPORTING |
| other units of this seed (`c1-c2-f26-seed<k>-*`, root and `jobs/`) | **was claimed as reported and was not (f1 r2, B3).** Now `other_units`: every other invocation with its RECONCILED result (or the receipt's, or `unreconciled`), its OVERRUN marker, and whether Eric acknowledged it (row `C2-framing-2026-invocation-<unit>`, RULED grammar, source Eric). `seed_allowed` refuses any unacknowledged invocation for any seed up to the current one; the aggregate refuses too and reports all | **FINDING B3** (r4) |
| `RECONCILED_<unit>.json` | was compared field by field. Now `==` exactly the record `launch.reconcile` writes for a clean exit: `{"unit", "result": "exit", "cpu_seconds": TERMINAL.cpu_seconds, "rc": 0, "problems": []}` (C1) | **FINDING F3b** |
| `compute_ledger.tsv` (C1's total) | **was not read.** Now: before a launch the effective total = C1 ledger total + the test's off-launcher CPU record (including the wrapper's own charge, recorded first) must leave room for the full per-seed budget under the cap (design §7; the manager's rulings (a)–(d) of 2026-10-09). The reservation is decided once, in `launch`; the run inside the unit is guarded by its cgroup and C1's own gate and does not re-decide it (r4, f1 r2 B2). The aggregate reports the ledger total, the off-launcher total and the effective total | **FINDING F3c** (f1 Part 2), **B2** (r4) |

## I. The preparation, the expectation and the off-launcher CPU record

| Artifact / field | Trusted check | Status |
|---|---|---|
| `PREPARED.json` in `<OUT_ROOT>/inputs` (r4: field by field) | `schema` == the constant; `exposure_commit` == the admission's (GIT); `inventory_sha256` == X-37's cited sha256 (GIT); `data_dir`, `raw_dir`, `screen_inputs` == the inventory's resolved directories (GIT); `out_dir` == the fixed namespace; `pins` exactly the prepared inputs with the screen's historical pins (PINS), each generated file read through its pin; `cpu_s` BOUND; **the whole file's sha256 == the register row `C2-framing-2026-prepared` recorded after X-37 (REG + GIT chronology): the acquisition record that fixes which preparation is trusted (f1 r2 B1)**; `games`, `proxy_reasons`, `game_number_fallbacks`, `lookup_2026_missing_probable_sides` REPORTING. It is written into a directory `prepare` creates exclusively (the file write itself is a hard link, `exclusive_write`) | **FINDING F4**, **B1** (r4) |
| `expect`: its directories and output namespace | no caller-supplied directory at all (r4): it reads and writes only `<OUT_ROOT>/inputs`; `data_dir` must be the inventory's `pa_dir` (GIT) | **FINDING F4**, **B1** (r4) |
| `expect`: its pins | from the bound `PREPARED.json` only (no `--pins`); the historical pins must be the screen's; every input is read through `read_pinned` | **FINDING F4**, **B1** (r4) |
| `expect`: its output | created exclusively (`os.link` of a fsynced temporary onto the final name: atomic, fails if the name exists), after an early existence check; one expectation per preparation, since the namespace is fixed | **FINDING F4**, **B1** (r4) |
| `EXPECTED.json` (r4: field by field) | `schema`, `pins` (exactly the admission's), `prepared_sha256` (== `PREPARED.json`'s bytes == the prepared row) and `sha256` (== the expectation's pin) are consumed by `preparation_chain_problem` in `run` and `aggregate`: the admitted pins descend from the recorded preparation (B1); `file`, `rows`, `identified`, `pins_digest`, `cpu_s` REPORTING | **FINDING B1** (r4) |
| the expectation's content | its sha256 is pinned (PINS, bound by the inputs row); every value recomputed from the pinned inputs at the aggregate | CHECKED |
| the off-launcher record `<OUT_ROOT>/off_launcher_cpu.jsonl` (one namespace, whatever the CLI arguments; r4) | append-only; its first row is the lead's seed row `{step: "prior", cpu_s, source}` carrying the prior off-launcher total the C2 index records and the box ledger does not (ruling (b)); a missing record, a record not beginning with the seed row, a malformed line or an invalid CPU refuses every step before any work (ruling (c)). Rows: `step`, `cpu_s`, `recorded_utc`; prepare adds `out_dir`; expect adds `file`; launch adds `seed` and `refused`; the launcher-process row adds `seed`, `rc`; aggregate adds `seeds`; a step that fails after spending CPU appends `failed: true` from a `finally` (ruling (c)). The gate and the aggregate sum `cpu_s` only; `step`, `source` and the extras are REPORTING (r4) | **FINDING F3c**, **B2** (r4) |

## J. The register rows and the admission record

| Row / field | Trusted check | Status |
|---|---|---|
| X-37 description | the PREDECLARED grammar (date, scope, review path, sha256 prefix, reviewed commit); first published in the exposure commit, absent at its parent, unchanged at HEAD; each field `==` the admission record (GIT) | CHECKED |
| X-37 source-inventory citation | path `== INVENTORY_REL`, 64 hex `== sha256` of the file at the exposure commit; the file absent at the parent commit (GIT) | CHECKED |
| `C2-framing-2026-inputs` | the PINNED grammar; digest `== pins_digest(admission pins)`; absent at the exposure commit (REG, GIT) | CHECKED |
| `C2-framing-2026-prep-read` | the DECLARED grammar; absent at the exposure commit (REG, GIT); required by `prepare` and `expect` | CHECKED |
| `C2-framing-2026-allowance` | the RULED (Eric) grammar; source cell's first token `Eric`; finite positive numbers; stop below budget; `cap == ledger.CAP_H` — at `seed_allowed` and at acceptance (REG, CONST). The register the gates read is the checked-out text of the admitted checkout, not `git show HEAD:`; its commit is the process's to fix (r4, f1 r2) | CHECKED; **FINDING F3a** |
| `C2-framing-2026-prepared` (r4) | the PREPARED grammar with the sha256 of `PREPARED.json`; absent at the exposure commit (REG, GIT); required by `expect` and by the chain check in `run` and `aggregate` | **FINDING B1** (r4) |
| `C2-framing-2026-invocation-<unit>` (r4) | the RULED (Eric) ACKNOWLEDGE grammar naming exactly that unit; source token `Eric`; consulted by `seed_allowed`, `validate_run` and the aggregate for every other invocation of a seed | **FINDING B3** (r4) |
| `admission_2026.json`: `reviewed_commit`, `review_report`, `exposure_commit` | `admission_check`: a plain SIGN whose single `Reviewed-commit` is the reviewed commit, the reviewed commit an ancestor of the exposure commit, which is an ancestor of HEAD; closure unchanged; nothing loose (GIT) | CHECKED |
| `admission_2026.json`: `input_pins` | shape; the inputs row; historical pins `==` the screen's accepted pins (PINS) | CHECKED |
| the source-inventory file | exactly `{pa_dir, raw_root, screen_inputs, selection, extraction}`; absolute paths; `selection == SELECTION`; `extraction == EXTRACTION_FIELDS` (CONST); `prepare`'s three directories `==` its (and now `expect`'s `data_dir`) | CHECKED (+ F4) |

## K. The pinned inputs themselves

| Input | Trusted check | Status |
|---|---|---|
| `pa_2017`…`pa_2025` | pins `==` the screen's accepted pins | CHECKED |
| `pa_2026` (the frozen copy) | pinned at preparation; X-37 names the directory; the calendar, labels, side-games and resumed totals derive from it | CHECKED |
| the lookup, the starter-proxy table, the raw-source manifest | pinned at preparation; the table and manifest are reconciled with the pinned 2025/2026 game ids (`sources_problem`, `table_games_problem`); the table's full schema is validated (`table_frame`). **Their derivation from the raw feeds is the admitted preparation's work and is not re-derived at acceptance** (the raw feeds are not run inputs) | CHECKED; the derivation is a disclosed limit |
| the expectation | see §I | CHECKED |

## Counts

Counted by script over the status cells above (one row = one field or field group as listed; 90 rows, revision 2): **CHECKED 55 · FINDING 23 · REPORTING 10 · BOUND 1**; 3 of these rows also carry a disclosed limit. The FINDING rows map to the ten round-3 findings below and the three round-4 findings B1–B3.

**The finding list for this round (each gets a RED, a fix and a GREEN in `round3-tdd-ledger.md`):**
- **F1** rank order follows `p_game_hit` within each date (ties reported).
- **F2a** `lgb_params` equals the reviewed recipe plus the two deterministic flags, exactly.
- **F2b** the blend configurations are recorded per arm and equal the reviewed `screen.blend_configs`.
- **F3a** the manifest's allowance equals Eric's row at HEAD with the launcher's cap; the first unit's CPU is within the first-unit stop.
- **F3b** `PENDING` read and bound (unit, Eric's budget, the full-budget limit, the wrapper's wall time); `TERMINAL` judged by `launch.terminal_problems`; `RECONCILED` is the exact reconciled record.
- **F3c** the §7 effective total (C1 ledger + the test's append-only off-launcher CPU record) must leave room for the full budget before any launch; the aggregate reports all three totals.
- **F4** `prepare` writes `PREPARED.json`; `expect` has no `--pins`, requires the inventory's PA directory and the prepared record, re-checks the historical pins, creates its output and `EXPECTED.json` exclusively.
- **F5a** `resumed_portion_rows` computed from the pinned PA rows and checked (2026 at validation; all seasons at the aggregate).
- **F5b** `resumed_flag_2026` checked against the pinned `pa_2026`.
- **F5c** `features_cpu_s` (and `wall_s`) finite, nonnegative and bounded by the run total.

**Disclosed limits (no trusted source exists in the retained evidence):** the `p_game_hit` values and the choice of the
ten retained batter-games (both need the day's predictions); exact probability ties between adjacent ranks (reported,
not ordered); the lookup's and table's derivation from the raw feeds (the admitted preparation's work).
