# 2026 framing test: test-first ledger for round 3 (the fresh review f1's findings and the field inventory)

Written by the lead (bts-lead3) on 2026-10-09. Round 3 starts from `field-inventory.md` (sha256 in the commit message):
every field a run or the process declares, with the trusted source that checks it. Its ten findings (f1's four, F1–F4,
and the lead's F5a–F5c) each got a failing test first, then the fix, then the green run. The rules of the earlier
ledger hold: one line per finding; the RED's command and failing assertion; raw outputs in `tdd-round3/`; a RED that
fails on a missing attribute or key is weaker than a failing assertion and is labelled.

**The RED run** (`tdd-round3/R3-red.txt`, sha256 `84c5bf5313d8ee1c712f505cc772088a37c8d04f08ca29c6c8219ecb2dece156`),
on the code at `39793e1` plus the new tests only: 39 failed, 1 passed, 121 deselected in 229 s.

    UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run --offline pytest -q -p no:cacheprovider \
      tests/scripts/c2_framing/test_f26_run.py -k 'test_f1_ or test_f2_ or test_f3_ or test_f4_ or test_f5_'

**The GREEN run** (`tdd-round3/R3-green.txt`, sha256 `d721d55f6f85e56e7ea2a0464328cf6e59088aec15b5d0a234e147f3382b0f83`),
the same command after the fixes: **42 passed, 121 deselected in 229 s** (the 40 RED-run tests plus the two guards
added with M95 and M97). A first GREEN attempt (`tdd-round3/R3-green-attempt1.txt`, sha256
`4a441b7efa4946c4a6bba7a40641fc141400133db4a951e480fadabbb1e57d4e`) had 40 passed and 2 failed, both on the test side,
not the code: the `_expect_world` helper passed `pa_dir` twice (a `TypeError` in one parametrized setup), and one
assertion looked for the off-launcher record under the world's `out` root while `expect` records it beside the inputs
directory (its parent, which is `OUT_ROOT` in production). Both corrected; the code was not changed between the two
attempts.

The one that passed is `test_f3_the_real_reconcilers_records_are_accepted_at_the_root_and_in_jobs`: a no-false-refusal
guard that runs C1's real `launch.reconcile` over a PENDING record and a receipt with every field the guard writes; it
needs no new code and its failing direction is a mutant (M87).

| Finding | Failing tests (`test_f26_run.py`) | RED (before the fix) | GREEN (after) | Mutants |
|---|---|---|---|---|
| F1. Ranks do not have to follow the retained probabilities (f1 F1) | `test_f1_a_coherent_rank_swap_that_changes_p_at_1_is_refused` (ranks 1 and 3 swapped on a date whose labels differ; every derived record recohered; the forgery moved the rank-1 vector), `test_f1_exact_ties_between_adjacent_ranks_are_reported_not_refused` | `DID NOT RAISE RunInvalid` (the forgery was accepted); `KeyError: 'ties'` (weaker: the report did not exist) | passed (see the GREEN run) | M75–M77 |
| F2. Only the two deterministic flags were checked; the blend configurations were not recorded (f1 F2) | `test_f2_a_common_wrong_model_recipe_is_refused_by_validation_and_by_the_aggregate` (3 cases: one tree with a 0.9 learning rate, an extra key, a missing key, in all ten manifests), `test_f2_the_manifest_records_the_reviewed_blend_configurations`, `test_f2_a_coherent_wrong_blend_configuration_is_refused_everywhere` (3 cases) | 3 × `DID NOT RAISE RunInvalid`; `KeyError: 'blend_configs'` (weaker: the field did not exist) ×4 | passed (see the GREEN run) | M78–M81 |
| F3a. The manifest's allowance and the first-unit stop were not checked at acceptance (f1 F3) | `test_f3_the_manifests_allowance_must_equal_erics_row_at_validation` (another budget in the manifest; the row changed after the run; no row), `test_f3_a_baseline_unit_over_the_first_unit_stop_is_refused_at_acceptance_and_at_the_next_seed` (4.2 CPU-h against the 4.1 stop with agreeing receipts) | `DID NOT RAISE RunInvalid` ×2 | passed (see the GREEN run) | M82, M83 |
| F3b. PENDING was never read; TERMINAL's budget was not checked; RECONCILED was compared loosely (f1 F3) | `test_f3_acceptance_applies_c1s_own_rules_to_the_launcher_records` (7 cases: PENDING missing, another declared budget, a limit short of the full budget, another wall time, another unit; a TERMINAL budget of 50 s with a clean RECONCILED, f1's reproduction; an extra RECONCILED field), `test_f3_the_real_reconcilers_records_are_accepted_at_the_root_and_in_jobs` (guard) | 7 × `DID NOT RAISE RunInvalid`; the guard passed | passed (see the GREEN run) | M84–M87 |
| F3c. The §7 effective total (C1 ledger + off-launcher CPU) was not applied before a launch or run; off-launcher CPU was printed, not recorded (f1 Part 2; the manager's erratum) | `test_f3_the_effective_total_must_leave_room_for_the_full_budget_before_a_launch_or_a_run` (152.9 h in C1's ledger: a 12 h budget fits; 0.2 h more off-launcher: launch and run refuse), `test_f3_a_malformed_off_launcher_record_fails_closed`, `test_f3_the_aggregate_reports_the_ledger_the_off_launcher_record_and_the_effective_total` | `AttributeError: no attribute 'off_launcher_rows'` / `'OFF_LAUNCHER_NAME'` / `'record_off_launcher'` (weaker: the record did not exist) | passed (see the GREEN run) | M88–M92 |
| F4. `expect` took caller-supplied directories and pins and overwrote its output; `prepare` left no durable record (f1 F4) | `test_f4_expect_takes_its_pins_from_the_prepared_record_and_creates_its_output_once`, `test_f4_expect_refuses_an_unbound_preparation` (7 cases: no record, another exposure commit, another inventory sha, another PA directory in the record, a historical pin not the screen's, a missing pin, another schema), `test_f4_expect_refuses_a_prepared_pin_that_is_not_the_files_bytes`, `test_f4_expect_refuses_a_pa_directory_that_is_not_the_inventorys`, `test_f4_expect_refuses_a_caller_pa_directory_that_is_not_the_inventorys` (guard, added with M95), `test_f4_exclusive_write_never_replaces` (guard, added with M97), `test_f4_the_expect_command_has_no_pins_option`, `test_f4_prepare_leaves_a_durable_record_of_its_pins_and_cpu` | `AttributeError: no attribute 'cited_inventory_sha256'` ×10 (weaker: the binding did not exist); `FileNotFoundError` for `--pins` (the option still existed) and for `PREPARED.json` | passed (see the GREEN run) | M93–M98 |
| F5a–c. `resumed_portion_rows`, `resumed_flag_2026`, `features_cpu_s` and `wall_s` were declared and never checked (the lead's) | `test_f5_declared_totals_and_timings_are_checked_against_the_pinned_rows_and_the_run_total` (4 cases), `test_f5_a_negative_wall_time_is_refused`, `test_f5_the_aggregate_checks_every_seasons_resumed_rows_against_the_pinned_inputs` | 6 × `DID NOT RAISE RunInvalid` | passed (see the GREEN run) | M99–M102 |

**The three f26 test files after the fixes** (`tdd-round3/f26-files.txt`, sha256
`2e26e064efe373c3bbcf6a8e15dbd0a154966299751e1e7fef0ebbd06f5f5d7c`): 277 passed, 1 failed. The failure was
`test_a_later_seed_waits_for_the_earlier_seeds_clean_receipt`, whose `match="TERMINAL"` no longer fits: with no
launcher record at all, `launcher_problems` now reports the missing PENDING record first. The refusal is unchanged
(a later seed still waits for the earlier seed's clean launcher records); the match became `"PENDING|TERMINAL"` with a
comment, and the test was rerun alone (`tdd-round3/f26-files-rerun.txt`). One older test changed with the fixes, as
in round 2's ledger: `test_the_expect_step_writes_the_evidence_every_run_must_carry` now uses the bound signature
(`expect(data_dir, inputs_dir)` with a `PREPARED.json`) and still checks the preparation-row refusal. The fixture
`_receipts` now writes `max_hours` and `written_utc` into PENDING, as the launcher does.

**How the fixes work** (all in `scripts/audit/c2_framing/f26.py`; no change under `src/`):
- **F1:** `order_problem` requires `p_game_hit` non-increasing with rank within each date; exact ties between adjacent
  ranks are counted (`ties`: all adjacent pairs, and the pairs at ranks 1 and 2) and reported by `validate_run` and in
  the aggregate's `per_seed.rank_ties`. **Disclosed limit:** a tie cannot be ordered from retained evidence, so a swap
  of two exactly tied rows with different labels is not refused; the count makes it visible for the results note.
- **F2:** `reviewed_lgb_params` = production's `LGB_PARAMS` at the admitted commit plus the two flags, compared exactly;
  `recorded_blend_configs` = `screen.blend_configs` per arm as JSON lists, recorded in the manifest and compared exactly.
- **F3a:** `validate_run` reads the register at HEAD, requires Eric's allowance row with the launcher's cap, requires the
  manifest's allowance to equal it, and requires the first unit's CPU within the stop. A later change to the row makes
  earlier runs invalid, deliberately: a changed allowance needs a decision, not silent acceptance.
- **F3b:** `launcher_problems` reads PENDING (unit, declared budget == Eric's, `limit_cpu_seconds == int(budget*3600)`
  since C1 admits a launch only when the full budget fits, `max_hours == 16`), applies `launch.terminal_problems`
  to TERMINAL against that PENDING record, requires a clean exit, and requires RECONCILED to equal the exact record
  `launch.reconcile` writes. Checked against C1's real reconciler in the guard test (root and `jobs/`).
- **F3c:** `record_off_launcher` appends to `<OUT_ROOT>/off_launcher_cpu.jsonl` (fsynced) from `prepare`, `expect`,
  `launch` (refused or not) and `aggregate`; `effective_total_problem` (in `seed_allowed`, so in `run` and `launch`)
  refuses when C1's `compute_ledger.tsv` total + the record + the full budget exceeds the cap; a malformed record fails
  closed; the aggregate reports `ledger_total_cpu_h`, `off_launcher_cpu_h` and `effective_total_cpu_h`. The manager's
  hand record remains the backstop for a step that crashes before its append.
- **F4:** `prepare` writes `PREPARED.json` (exposure commit, cited inventory sha256, resolved directories, pins, counts,
  CPU). `expect(data_dir, inputs_dir)` has no `--pins`: the PA directory must be the inventory's; `PREPARED.json` must
  carry the admission's exposure commit, the cited inventory sha256 and the inventory's PA directory; its pins must be
  exactly the prepared inputs with the screen's historical pins; the output and `EXPECTED.json` are created by
  `exclusive_write` (a fsynced temporary hard-linked onto the final name: atomic, fails if it exists). An early
  existence check refuses before the recomputation; the link is the race-proof layer (both tested).
- **F5:** `Trusted.resumed_flag_2026` from the pinned `pa_2026`; the run records `resumed_portion_rows` from the pinned
  rows (`raw_df`, not the feature frame); `validate_run` checks the 2026 entries, `aggregate` the whole dict against
  `recompute_asof`'s second return; `features_cpu_s` and `wall_s` are finite, nonnegative and bounded.

**The mutation checker** (`mutation_check.py`): round 3 adds M75–M102 (one per new guard) and repairs five anchors my
edits had made stale or ambiguous (M42 and M74 now name their function's own `source_inventory` call; M47 carries the
new `budget` argument; M48 targets the new clean-exit test; M50 follows `launch`'s new indentation; M68 the renamed
`cited` variable). All 100 anchors match exactly once and every substituted source compiles (checked with plain
python, `ast.parse`, before any run). **Mutation runs are paused by the manager until the disk margin is back; no
round-3 ledger exists yet.** Known MASKED pair, disclosed in advance: `expect`'s early existence check and
`exclusive_write`'s link are two layers with different messages; M97 targets the link through its own unit test.
