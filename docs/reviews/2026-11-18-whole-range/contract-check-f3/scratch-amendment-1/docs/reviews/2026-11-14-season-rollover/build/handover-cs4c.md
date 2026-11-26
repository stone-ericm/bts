# CS4C handover

Base `1c6cbc77eea237a5cffd47147b4219f0f15ec747` on `sb-cs4c`. Commit `9fc2ee6d5eec51c7ab1a7789b46e2849917b7e8c` records the automatic reading used for qualifying scheduler deliveries. Under Amendment 1, a public pick without a warning recipient still stamps the reading and reports staleness to stderr. The amended kickoff remains a worktree edit.

## Verification runs

All logs are in `.codex-review/cs4c/`. The transport tests ran offline with captured pick and health transports.

| Run | Result |
| --- | --- |
| Initial baseline, `baseline-fast.log` | Interrupted at 13% under the original stop instruction. |
| Accepted sandboxed baseline before any test file, `baseline-restart.log` | 8,481 passed, 41 failed, 16 errors, 23 skipped, 6 deselected and 1 xfailed in 28m01s. The 57 IDs are the 53 in `~/bts-runs/gate-failed.txt` plus `tests/test_sd_notify.py::test_sends_watchdog_message_when_socket_set`, `::test_sends_ready_message_when_socket_set`, `::test_non_encodable_message_does_not_raise` and `tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring`. Those four could not bind local Unix sockets in the sandbox. The authorizing follow-up accepted this baseline and confirmed that the unsandboxed run of the same base commit had exactly the 53 known IDs. |
| RED on `1c6cbc77`, `red.log` | 89 failed, 310 passed and 4 skipped. Of the failures, 87 exercised new requirements. The other two used a delivery recovery fixture without a notification ID. |
| Corrected GREEN controls on `1c6cbc77`, `red-guard-control.log` | 16 passed. The fixture now provides the ID required to recognize an already delivered pick. |
| Initial feature run, `green-local.log` | 395 passed, 4 failed and 4 skipped. The four failures used an escaped backslash in the test's WARN-header check. The corrected test asserts the complete transport envelope. |
| Combined unsandboxed run, `green-combined.log` | 412 passed and 4 skipped in 8.07s in one pytest process. Included both new test modules, both migrated modules, `test_lightgbm_optional.py` and `test_web_saver.py`. |
| Pinned 2027-07-15 and 2028-07-15 wall clocks, `clock-2027.log` and `clock-2028.log` | 399 passed and 4 skipped each. Both runs covered every added or edited test file with Python's `datetime` and `date` wall clocks pinned. |
| Archived base and branch delivery comparison, `matrix.log` | 630 cases. See the comparison below. |
| Final unsandboxed fast suite, alone on `9fc2ee6d`, `final-fast.log` | 8,618 passed, 37 failed, 16 errors, 23 skipped, 6 deselected and 1 xfailed in 25m04s. Wrapper exit 1. The 53 failure and error IDs exactly match `~/bts-runs/gate-failed.txt`. All four sandbox-only baseline failures passed. |

## Contract rows

| Row | Implementation | Verification |
| --- | --- | --- |
| 1 | `src/bts/contest_state.py:82` adds the trailing field. `:632` copies the selected observation's literal value. Model-only mode uses the default. | `tests/test_automatic_reading_cs4c.py:31` loads dated, automatic, legacy manual and model-only states from real fixtures. RED on the base for the new field. |
| 2 | `src/bts/contest_state.py:86` defines the 14-hour constant. `:96` bounds the retained timestamp, `:102` parses and normalizes to UTC, `:114` computes the record, and `:139` validates decoded records. | `tests/test_automatic_reading_cs4c.py:53`, `:82`, `:96`, `:109` and `:142` cover the pure interface, clock and I/O spies, both strict thresholds with one-microsecond deltas, DST, malformed input, UTC overflow, schema, types, and canonical field order. New interfaces and valid-record acceptance were RED. Normalizing malformed decoded values to absence was a GREEN control on the base. |
| 3 | `src/bts/picks.py:251` adds the last dataclass field. Saves at `:398` and `:502` use the current dataclass. `:553` and `:611` apply the same validator for shadow, normal and byte loaders. `src/bts/scheduler.py:2275` appends archive metadata after the pick fields. | `tests/test_automatic_reading_cs4c.py:142` and `:160` cover all loaders, canonical bytes, provenance, shadow re-save, literal receipt and ledger hashes, and historical exports. `tests/test_stale_delivery_cs4c.py:83`, `:236`, `:373` and `:489` exercise attempt, delivery, post, preview, reuse, grading, reconcile and archive re-saves. Valid-field retention, canonical bytes and archive order were RED. Legacy decoding and malformed-as-absent cases were GREEN controls. |
| 4 | `src/bts/scheduler.py:1174` stamps the delivery snapshot using the final cutoff sample. The DM and public calls are at `:1194` and `:1225` directly before the attempt marker. | `tests/test_stale_delivery_cs4c.py:83` sees the stamp at transport entry and after confirmation with both fresh and saved picks, including a saved pick whose reading crossed 14 hours. `:189` checks dated clearing and `:329` injects a failed calculation. These were RED. |
| 5 | `src/bts/scheduler.py:63` guards the diagnostic. `:70` formats, prints and dispatches the warning using the emission ET date. The warning calls at `:1217` and `:1247` follow the original confirmation work and its exception handler. | `tests/test_stale_delivery_cs4c.py:83`, `:161`, `:287`, `:319`, `:329`, `:392`, `:425`, `:446` and `:478` check order, exact stderr and WARN-attention envelope, the three reasons, public delivery without a recipient, midnight rollover and a send across midnight, raised dispatcher and swallowed transport failures, the real `run_day`, per-date dedup, retry without resurface, and critical-status isolation. New stamp and warning assertions were RED. Critical-status isolation without a recipient was a GREEN control. |
| 6 | `src/bts/scheduler.py:1035` retains early guards and the existing recovery/failure paths. The new work is limited to the two final-cutoff success paths under required configuration. | `tests/test_stale_delivery_cs4c.py:139`, `:161`, `:209`, `:236`, `:373` and `:518` cover refusals, missing DM recipient, retry, the three restart windows, same-pick re-entry, legacy commands, and optional/private/marker guards. Existing cutoff, missing DM recipient and early guard behavior was GREEN on the base. Record persistence over retry, restart and legacy re-save was RED. |
| 7 | `scripts/live_forward_capture_once.py:191` and `scripts/live_forward_provenance_inventory.py:72` project away null readings. | `tests/test_automatic_reading_cs4c.py:204` and the migrated legacy fixture check absent/null equivalence, drift when a non-null record is added, removed or changed, and literal hash changes through both actual snapshot consumers. Absent/null equivalence was RED. |
| 8 | The production diff is limited to the timestamp copy and new helper code in contest state, pick model/read mappings, the delivery additions and the two projections. | `git diff --unified=0 1c6cbc77 HEAD -- src scripts` bounds the changed hunks. There are zero changes to `select_pick`, `run_single_check`, fallback functions, `saved_pick_refusal`, `chosen_on` or its validator, `season_contract`, the loader's `status` and `message`, contest health/attention/dispatch, fetch, or the `decision.json` writer. The matrix compares the existing decision output to the base. |

The `run_day` case chose Steven Kwan with a covered automatic file aged 20 hours, sent the pick DM, locked it, and sent one WARN-attention DM. The file reported `stale: true` and `reason: "old"`. The cached case sends with `selection=None` and retains its earlier `chosen_on` while recording the age of the delivery reading.

## Existing test migrations

- `tests/test_pick_chosen_on_cs5a.py::test_round_trip_and_exact_prior_encoding`, four cases. Kept the exact pre-5a prefix comparison and asserted the exact bytes and order of the two trailing fields.
- `tests/test_pick_chosen_on_cs5a.py::test_legacy_read_rewrite_and_projection`. Seeded a pre-5a fixture without either field, checked both null additions and both projections, and kept the existing non-null `chosen_on` drift assertions.
- `tests/test_saved_pick_paths_cs5a.py::test_marker_archive_and_grading_saves_preserve_chosen_on`, two cases. Checked the last three archive keys in order and kept the existing `chosen_on` value checks.

## Recorded delivery comparison

`compare_delivery.py` archived `1c6cbc77:src` and drove the real `_deliver_and_lock_pick` from both trees. The axes were seven reading cases, three delivery modes, required or optional configuration, three marker states, and five selection cases. The recorded results are `matrix-base.json` and `matrix-branch.json`. The comparison checked return values, captured pick sends, existing stderr, critical alerts and status, attempt markers, persisted lock state, `decision.json`, lineup evolution and pick fields. It compared the new field and warning outputs separately to the row expectations.

| Cells | Count | Observed difference |
| --- | ---: | --- |
| Existing controls | 590 | Matched across both trees. Includes private and optional deliveries and unreachable sends. Row 3 accounts for the new nullable wire field. |
| Qualifying dated sends | 8 | Same pick send. Row 4 writes a null reading, serialized by row 3. |
| Qualifying fresh automatic sends | 8 | Same pick send. Row 4 adds the fresh reading to both attempt and delivered files. |
| Qualifying stale automatic sends | 24 | Same pick send. Row 4 adds the stale reading to both attempt and delivered files. Row 5 adds exactly one stderr line, one exact WARN-attention DM, and the dedicated status file per independent cell. |

## Known limits

- The warning follows confirmed delivery. A correction afterwards leaves that day's pick as delivered. The morning schedule warning is CS6b.
- With serialized scheduler calls and a readable, writable warning status file, successful warning DMs deduplicate by the warning ET date and the key `WARN_ATTENTION:contest_state:stale_reading`. The warning has its own status file. A failure can be retried on a later eligible delivery without a resurface banner. A crash or lost status may lose the warning or allow a duplicate. The feature has no replay or exactly-once guarantee. A warning DM requires a recipient.
- The rule measures the age of the last accepted fetch. Missed refreshes can warn while the value is correct, and a fresh value can be wrong. Settlement coverage remains the concern of the existing loader status and health checks.
- `recorded_at` is the fetch command's start time. The threshold is fixed at 14 hours. The legacy commands do not warn. A qualifying delivery under a dated observation records `null`.
