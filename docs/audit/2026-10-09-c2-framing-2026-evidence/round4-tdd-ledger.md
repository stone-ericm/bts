# 2026 framing test: test-first ledger for round 4 (review f1's round 2: B1, B2, B3 and its corrections)

Written by the lead (bts-lead3) on 2026-10-09. Review f1's second and last round (`docs/audit/2026-10-09-c2-framing-2026-code-codex-f1r2.md`,
sha256 `2fa88ce0f31c32d73b29cbb8c53cd124e50c17bbd9859b3d0ed6c860b9e85aa6`, BLOCK on `d73693d`) closed its F1, F2
and F3 and found F4 partial; it raised B1 (the expect step trusted a caller-selected prepared record), B2 (the §7
accounting: a launch-boundary disagreement, a namespace that followed the caller, a missing record indistinguishable
from empty, failed steps and the launcher process uncharged) and B3 (other invocations of a seed neither reported nor
judged, contrary to the inventory). The manager's rulings (a)–(d) of 2026-10-09 fix the accounting's record, its seed
row, its fail-closed rule and the cap's enforcement; this round encodes them.

**How this round was written, disclosed.** Both the tests and the fixes were written as file edits during a hold on
pytest (a full suite was running in the main checkout under the serialization rule). So the RED could not be run
before the fixes were written. The RED record below is the round-4 selection run with `f26.py` swapped to its bytes at
`d73693d` (the pre-fix code; sha256 recorded in the output) and the new tests in place, then the fixed `f26.py`
restored byte for byte before the GREEN. A RED obtained this way shows that the tests fail against the pre-fix code; it
does not show that the fixes were written after the tests, which they were not in time, only in content. The
mutants M103–M114 (one per new guard) are the stronger evidence, when mutation runs resume.

**The RED run** (`tdd-round4/R4-red.txt`, sha256 `941651bf05ba137ae18b6111a0f8774722c838758aec3e1a829b8a5b4c8832ea`): `f26.py` swapped to its `d73693d` bytes (sha256 `a0b011f0d38fdb7d9f48eaffb2ecc9691b6e1afe8d84dd843b5efbd5c12ba9de`), the round-4 selection `-k 'test_b1_ or test_b2_ or test_b3_'`: **5 failed, 20 errors, 163 deselected in 3.6 s**. Every one is an `AttributeError` at fixture or test setup (`PREPARED_ROW` ×19, `acknowledged_invocation` ×5, `seed_record` ×1): the world fixture itself needs the round-4 API, so no round-4 test reached a behavioral assertion against the pre-fix code. This is the weakest kind of RED the round-3 ledger describes; it shows the tests cannot pass without the new code, not that the old code accepted each forgery. The fixed `f26.py` (sha256 `010d698b565f1e8d38c713e22d52defa35e1ce0f19b529a6d88ccc7b62a96a47`) was restored and its sha verified before the GREEN. The behavioral evidence for each guard is its mutant (M103–M114, and the re-anchored M88–M92), when mutation runs resume, and the fresh reviewer's own probes.

**The GREEN run** (`tdd-round4/R4-green.txt`, sha256 `f2c138d0fbe1ae0e6bbb60699c4fd9e502cabdb64fad9009477c33fb519dbd7b`), the same selection on the fixed `f26.py`: **26 passed, 163 deselected in 223 s**. A first GREEN attempt (`R4-green-attempt1.txt`, sha256 `221fae2bcbc891c55a03283a5f69d650cf3162acbb389ba2f720d7e9070315bb`) had 24 passed and one setup error: the seeded-record test combined the `world` and `prep` fixtures, and `prep`'s stubbed admission gate (no `input_pins`) broke the `ten` fixture's runs; the test was split in two, no code changed. **The three f26 test files** (`tdd-round4/f26-files.txt`, sha256 `c2d4e865402833bda50e7402337fe0a35923d504dbd46c3db023f01e4ed20af1`): **304 passed, 0 failed in 748 s**. Their first attempt (`f26-files-attempt1.txt`, sha256 `16eb8bcee93fe437a43b8380d59e63481e6c69d9f0fedfe6d452be61a08f8635`): 301 passed, 1 failed, 1 error — the same setup error, and `test_the_preparation_reports_its_own_cpu` still calling `prepare` with the removed positional output directory (moved to the `_prepare` helper; a test-side change). The code did not change between the attempts.

| Finding | Failing tests (`test_f26_run.py`) | RED (before the fix) | GREEN (after) | Mutants |
|---|---|---|---|---|
| B1. `expect` trusted a caller-selected PREPARED record and a caller-selected directory; nothing bound the admitted pins to the recorded preparation | `test_b1_prepare_and_expect_use_only_the_fixed_inputs_namespace`, `test_b1_expect_requires_the_prepared_row_binding_the_records_bytes` (no row; a row binding other bytes; the wrong grammar), `test_b1_a_replaced_table_with_its_own_pin_in_a_rewritten_record_is_refused` (f1's reproduction), `test_b1_expect_refuses_a_record_whose_directories_or_cpu_are_not_the_inventorys` (4), `test_b1_the_run_and_the_aggregate_require_the_preparation_chain` (6) | `AttributeError` at fixture setup (the world needs `PREPARED_ROW`): the weakest RED, disclosed above | passed (13 tests) | M103–M108 |
| B2. The §7 accounting | `test_b2_prepare_refuses_without_the_seeded_record`, `test_b2_the_record_must_be_seeded_with_the_prior_total_before_any_step`, `test_b2_a_failed_step_is_still_charged`, `test_b2_the_launch_gate_reserves_with_its_own_charge_first_and_the_run_does_not_re_decide` (f1's boundary probe), `test_b2_the_launcher_process_is_charged_from_its_own_rusage`; the round-3 F3c tests re-stated: `test_f3_the_effective_total_must_leave_room_for_the_full_budget_before_a_launch`, `test_f3_a_malformed_off_launcher_record_fails_closed` (3 texts), `test_f3_the_aggregate_reports_the_ledger_the_off_launcher_record_and_the_effective_total` | `AttributeError` at fixture setup (`seed_record`, `PREPARED_ROW`): the weakest RED | passed (5 new + 3 re-stated) | M88–M92 (re-anchored), M109–M111 |
| B3. Other invocations of a seed | `test_b3_another_invocation_for_a_seed_is_reported_and_blocks_progress_until_eric_acknowledges_it` (a terminated second seed-1 unit reconciled by the real reconciler; refused; a non-Eric row refused; Eric's row accepted), `test_b3_an_unacknowledged_invocation_of_the_current_seed_refuses_its_launch_and_run` (a clean earlier attempt: a rerun), `test_b3_the_aggregate_reports_every_invocation_and_refuses_an_unacknowledged_one`, `test_b3_the_acknowledgement_row_grammar` (5) | `AttributeError` (`acknowledged_invocation`, `PREPARED_ROW`) at setup: the weakest RED | passed (8 tests) | M112–M114 |

**How the fixes work** (all in `scripts/audit/c2_framing/f26.py`; nothing under `src/`):
- **B1.** `prepare` writes only `OUT_ROOT/inputs` (no `--out`; a fixed namespace created exclusively) and returns
  `PREPARED.json`'s record; the lead then records that file's sha256 in a new register row,
  `C2-framing-2026-prepared` ("**PREPARED <date>: the catcher framing 2026 test's preparation record `<sha256>`**",
  after X-37 and the preparation row; `prepared_row_problem`). `expect` has no directory argument: it reads and writes
  only `OUT_ROOT/inputs`, requires that row to bind the bytes of the `PREPARED.json` it finds there, requires the
  record's exposure commit, cited inventory sha256, all three inventory directories, output namespace and a finite CPU,
  takes its pins from it, and creates the expectation and `EXPECTED.json` exclusively. `run` and `aggregate` then
  require the chain: `EXPECTED.json` carries exactly the admitted pins and the prepared record's sha256, and that record
  is the row's (`preparation_chain_problem`). Which record is trusted and when it becomes immutable: the register row,
  when the process commits it; `expect` and the chain read the checked-out register, as every gate does.
- **B2.** One accounting namespace: every step records at `OUT_ROOT` (prepare and expect no longer follow a caller
  directory). The record must be seeded by the lead (`f26 seed-record --cpu-s 80.01 --source "C2 index rows of
  2026-10-08 and 2026-10-09"`; the first row has step `prior` and a source; ruling (b)); a missing record, a record
  without the seed row, a malformed line or an invalid CPU refuses every step before any work (`require_record`;
  ruling (c)); every step runs under `charged`, which appends its row from a `finally`, `failed: true` when it raised
  (ruling (c)). The reservation is decided once, in `launch`: the wrapper records its own CPU so far, then
  `effective_total_problem` (C1's ledger + the record + the full budget under the cap) decides; the C1 launcher's own
  process is charged afterwards from this process's child rusage (`launcher-process` row with its rc); a refused launch
  charges the CPU it spent after its first row. `run` inside the unit no longer re-decides the reservation (its guards
  are the cgroup and C1's own gate), which removes the boundary disagreement f1 measured. One cap: Eric's row's cap
  equals `CAP_H` (ruling (d)).
- **B3.** `other_units` scans C1's records (root and `jobs/`) for every other `c1-c2-f26-seed<k>-*` unit of a seed and
  reports its reconciled result (or the receipt's, or `unreconciled`), its OVERRUN marker and whether a register row
  `C2-framing-2026-invocation-<unit>` ruled by Eric acknowledges it ("**RULED <date> (Eric): ACKNOWLEDGE invocation
  `<unit>` of the catcher framing 2026 test**"). `validate_run` returns them; `seed_allowed` refuses any unacknowledged
  invocation for any seed up to the current one (the current unit excluded in `run`); the aggregate refuses too and
  reports every invocation per seed. When such a case arises it goes to Eric in plain words, and his row is verbatim.

**Corrections from review f1's round 2, applied:** the inventory's count is now measured by script and the method
stated (its revision 2 counts 90 rows); `wall_s` is BOUND, not REPORTING; PREPARED, EXPECTED and the off-launcher rows
are listed field by field with their classification; TERMINAL's failure-variant fields are listed; the register the
gates read is described as the checked-out text; the inventory's false claim that other invocations were reported is
replaced by B3's row. In the round-3 ledger: the F4 RED's wrong-PA-directory case was a setup `TypeError` (the helper
passed `pa_dir` twice), not a behavioral RED, and is now described so; six mutant definitions were repaired, not five;
M87's selector targets the extra-field RECONCILED case, which is what kills it (the real-reconciler guard is a
no-false-refusal guard and would still accept clean records under M87).

**Test changes this round, disclosed:** the world fixture now carries the preparation chain (`PREPARED.json`,
`EXPECTED.json`, the prepared row) and a seeded record; its inputs directory moved under the run root; the stubbed
admission returns an exposure commit; three older tests that change a pin on purpose re-bind the chain
(`_rechain`) so the check they target is the one that refuses; `expect` tests use the new signature; the round-3 F3c
tests are re-stated for the launch-only gate (the run no longer refuses on the effective total; a regression test
asserts it). One new B2 boundary test's first ledger figure was corrected from 152.99 to 152.9 before any run, since
152.99 + 0.0225 + 12 exceeds 165 (the lead's arithmetic, not a test run).
