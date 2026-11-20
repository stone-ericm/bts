# Whole-range C2 handover

Commit `dfe774df` on `fix/wr-c2`, parent `386aa846`. One local commit. The remote-tracking ref moved during the work and the commit stayed on the requested parent. No push, network service, box, or contest account was used.

## Changes and base evidence

The retained `contract-check-c2/check_base.py` driver ran on the base with the archived Codex and Carbon slice directories on `PYTHONPATH`. Its integrated execution called the receipt and route probes, the corrected replay and Carbon probe 2, and the interrupt verification. Carbon captured zero DMs, a scoreable `locked_unconfirmed` decision and model streak 1. The Codex runs captured zero recovery DMs for both acceptance windows and the dated correction, with respectively zero, one and zero recommendation sends. The durable confirmation control captured one recommendation, zero recovery DMs and next-morning `entered`. The correction changed the current choice from an attempted double at streak 1 to a single at 46.

| Row | Committed behavior and location | Base and regression evidence |
| --- | --- | --- |
| U1 | `scheduler.py:1160` stamps the selected ET slate into the dedicated receipt and dispatcher call. `health/alert.py:250` checks the receipt entry and descriptor, rejects aliases and special files, and replaces a receipt atomically via the verified directory descriptor. | `test_unknown_receipt_isolated_from_cutoff_and_eod` reproduced cross-suppression in both orders. `test_unknown_receipt_uses_slate_across_midnight` was RED on the first missing restart DM. The base receipt probe also showed duplicate direct alerts after unrelated failure, another processed day and rollover, plus shared-target modification through an alias. |
| U2 | `scheduler.py:1246` calls the helper after the existing direct route's state save and commit attempt. | `test_direct_unconfirmed_uses_own_receipt_after_commit` checks the on-disk lock and decision inside the captured send, the dedicated acknowledgement, unchanged shared bytes, and `False`. The new receipt and body were absent on base. |
| U3 | `scheduler.py:3096` derives eligibility from the loaded pick and dispatches immediately after the classification decision helper returns. | All three `test_restart_pages_unknown_delivery_without_resending` cases were RED on the missing C2 DM. `test_classified_restart_preserves_existing_commit_and_terminal_result` checks a fresh write, both prior decision labels, terminal result and a writer returning no record. Base was silent with each eligible classification. |
| U4 | `health/alert.py:300` requires a nonblank string message ID for an acknowledged receipt and uses the same body on retry. `health/alert.py:363` exposes the keyword-only opt-in. | `test_only_unknown_alert_requires_message_id` covers four unusable replies, a raised send, and an acknowledged reply. The base persisted `sent` for all four unusable replies. `test_no_recipient_retries_when_configured` covers the skipped checkpoint and later configuration. `test_legacy_health_dispatch_matches_base` checks bytes, logs, return values and bodies for omitted and explicit false. |
| U5 | `health/alert.py:238` renders one best-effort bounded stderr diagnostic and `scheduler.py:1160` contains ordinary preparation and dispatch failures. `KeyboardInterrupt` and `SystemExit` propagate after the state and decision-write attempt. | `test_unknown_alert_failure_is_contained_after_commit` checks both routes, open/write failure, formatting and dispatcher failure, multiline and unprintable errors, and stderr failing during reporting. Base dispatch and formatting exceptions escaped. `test_restart_retries_unknown_alert_after_interrupt` checks both sides of the receipt checkpoint and both interrupt classes. Base propagated the pre-receipt interrupt but stayed silent on classification restart. |
| U6 | The lock-route predicates and receipt separate C2 from other health alarms. | Green preservation controls in `test_unknown_alert_leaves_pick_lifecycle_and_backstops_unchanged`, the real replay, cutoff isolation cases, existing decision integration tests, and the caller suite. |
| U7 | `scheduler.py:1165` uses the exact dated operator copy and ASCII apostrophe through the standard CRITICAL envelope. | `test_unknown_alert_operator_copy` compares both routes and retries for entry and cutoff states. The corrected real replay and rollover tests also compare the complete body. Direct base copy was different and classification had no body. |

## Acceptance cases

| Case | Test coverage |
| --- | --- |
| Crash before acceptance, crash after acceptance, and dated correction before restart | `test_restart_pages_unknown_delivery_without_resending` uses the archived real scheduler replay and captures recommendation sends independently of health DMs. The first restart runs through the real CLI. A second explicitly dated launch keeps the decision bytes and emits no second C2 DM. |
| Existing scoreable authority, terminal result and failed decision write | `test_classified_restart_preserves_existing_commit_and_terminal_result` checks state at send time, exact prior decision bytes, result polling and the independent missed-pick alert when there is no record. |
| Midnight and New Year | `test_unknown_receipt_uses_slate_across_midnight` runs the scheduler before and after each ET boundary for the same slate, then runs the next slate and checks separate receipts and shared bytes. |
| Direct route and cutoff in either order | `test_direct_unconfirmed_uses_own_receipt_after_commit` and `test_unknown_receipt_isolated_from_cutoff_and_eod`. The latter has separate pick stores with one health root and interleaves a transport failure and a real health-runner dispatch for another processed day. |
| Ordinary transport failure, unconfirmed reply, missing recipient and retry | `test_only_unknown_alert_requires_message_id` and `test_no_recipient_retries_when_configured`. |
| Interrupt before and after the receipt | `test_restart_retries_unknown_alert_after_interrupt` uses real scheduler restarts. An interrupt before receipt persistence allows a repeated alert. An acknowledged receipt suppresses subsequent alerts. Both windows retain the pick attempt and never resend the recommendation. |
| Unusable receipt | `test_unknown_receipt_ignores_unusable_path` covers truncated JSON, array, null, a malformed acknowledgement, directory, FIFO under a five-second alarm, unreadable file, receipt symlink or hardlink to the shared file, dedicated-directory symlink, and blocked parent. The shared file stays byte-identical. |
| Confirmed public/DM delivery, private with no attempt, stale preview, pre-attempt, dry run, off-day and schedule error | `test_unknown_alert_leaves_pick_lifecycle_and_backstops_unchanged`. The retained base rehearsal independently checks the after-confirmation crash window. |
| Operator states | `test_unknown_alert_operator_copy` covers no send, accepted and entered, post-cutoff, post-first-pitch, overnight and a carried attempt in private mode, with identical transport copy in both routes and on retry. The corrected replay exercises the new account observation and the New Year rollover test exercises the actual slate change. |
| Watchdog, 08:30, EOD, cutoff and void backstops | The real replay checks the B2 watchdog text and 08:30 outcome. `test_crash_then_corrected_real_scheduler_replay` independently pins the three B2 nags. Lifecycle and cutoff isolation tests check `pick_delivery`, void and cutoff. |

The standing hostile-input cases are covered by the named interrupt, unusable receipt, rollover, acknowledgement, and real restart tests above. The receipt path uses descriptor verification after open and an atomic replacement in the verified directory. Invalid or unavailable storage leaves an alert eligible. A pre-upgrade shared-only acknowledgement can result in an extra C2 DM on the first qualifying encounter.

## Existing test amendment

Only `tests/test_alert_status_callers_cs6b2.py:30` changed. Before C2, `classified_unconfirmed` asserted no dispatch and `unconfirmed` expected the shared status file. Both now require exactly one `late_delivery` CRITICAL, the dedicated slate receipt and unchanged shared status, in both the missing and corrupt shared-file fixtures. All other cases and assertions were retained. The three existing message-fragment pins passed without edits.

## Runs

- Base offline probes via `check_base.py` passed their evidence assertions with the archived slice directories supplied on `PYTHONPATH`.
- Initial RED wrapper run collected 67 cases. It reported 59 failures and 8 passes, including fixture defects in the second-restart clock and preservation controls. The corrected RED run collected 76 cases. It reported 65 failures and 11 passes. All three real restart cases reached the missing-DM assertion. The explicit-false legacy cases also failed on base because the keyword did not exist. The omitted-keyword baseline assertions executed before those failures.
- Initial focused run reported 294 passes and four fixture failures. Three simulated lineups never reached a scheduled check at the pinned clock. The fourth fixture counted the independent missed-pick alert and expected polling with no scoreable decision. The corrected focused run passed 359 cases.
- Final focused run passed 361 cases in 11.18 seconds, in one process with `test_lightgbm_optional.py`, `test_web_saver.py`, the shared caller suite, late-delivery guard, both stale-delivery suites, B2 edges, scheduler decision integration and morning state.
- The one unsandboxed fast suite ran on `dfe774df` through `yield_run.sh`. It reported 10,191 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed in 26m 01s. The sorted `FAILED` and `ERROR` test IDs matched all 53 entries in `~/bts-runs/gate-failed.txt` exactly. All 78 C2 cases passed. The committed tree was clean after the run. Logs are under `.codex-review/wr-c2/`.

## Ledger

`CLAUDE.md:92` records C2 and the deferred EOD second chance. A failed or unacknowledged C2 send retries only at a subsequent qualifying lock-route encounter on the same slate. The watchdog, normal EOD checks and 08:30 remain eligible independently. The lock is saved and the decision write is attempted before dispatch, including on interrupt. A lost acknowledgement or receipt write can repeat the alert. Concurrent invocations also have a window before an acknowledgement is persisted. The recommendation is never resent by the recovery routes. An undated start after ET midnight selects the new slate.

## Round 2

`e897c766` is the one new commit on top of `dfe774df`. Amendment 1 and the round-1 evidence were read from main at `d502148a`. The worktree stayed on the C2 branch and the commit was kept local.

| Diff base | CLAUDE.md | alert.py | scheduler.py | existing caller test | C2 recovery test | Carbon adoption | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dfe774df` | 2 | 3 | 0 | 0 | 123 | 377 | 4 files, 464 insertions and 41 deletions |
| `386aa846` | 3 | 154 | 48 | 9 | 567 | 377 | 6 files, 1,136 insertions and 22 deletions |

1. **B1 · Acknowledgement schema.** `src/bts/health/alert.py:324` requires the exact C2 schema string in the opt-in dedup predicate. The Codex `probe_receipts.py` damaged-schema case reproduced zero sends through the two scheduler encounters on round 1. On Round 2 it sent once, replaced the receipt and deduplicated the next encounter. `tests/test_wr_c2_recovery.py:250` exercises both routes using a list, an unknown string or a missing schema, and incorrect types for status, slate, sources and message ID. It asserts a valid replacement and subsequent deduplication.
2. **Carbon adoption.** `tests/test_wr_c2_carbon_v2.py` is byte-identical to the adopted file. Its 21 tests passed on Round 2 before the isolated mutant checks. The four source variants use Carbon's exact replacement snippets in separate scratch copies with verified import paths.
3. **N3 · Send snapshots.** `tests/test_wr_c2_recovery.py:93`, `:116` and `:492` capture state, decision bytes, or operator context at transport time, then check the snapshots and `sent` receipt outside the mocked transport. The classification test also snapshots the writer call count if it returns no record. Carbon's classification test at `tests/test_wr_c2_carbon_v2.py:77` covers the Codex probe's send-time disk snapshot, so no duplicate probe test was adopted.
4. **N1 · Watchdog disagreement.** `CLAUDE.md:92` records the known case. Codex's `probe_operator.py::delivered_label_watchdog` reproduced it on Round 2 using captured transports. A recommendation send without a message ID left a scoreable `delivered` decision and an unconfirmed pick. C2 instructed the operator to avoid entering or changing a pick based on the alert, while the watchdog still said “Fix it now!” for that slate. The watchdog is outside this change.

### Round 2 verification

| Scratch mutant | Detecting Carbon test | Full adopted-file result |
| --- | --- | --- |
| `m5_before_write_classification_decision` | `test_classification_alert_runs_strictly_after_decision_commit`, line 77 | 4 failed, 17 passed |
| `m6b_banner_drop_require_id_guard` | `test_prior_failed_with_critical_count_never_adds_resurface_banner`, line 180 | 4 failed, 17 passed |
| `m7b_follow_dir_symlinks` | `test_symlinked_receipt_directory_neither_reads_nor_writes_target`, line 142 | 4 failed, 17 passed |
| `m8_drop_replace_entry_recheck` | `test_mid_send_alias_or_special_at_receipt_is_left_in_place`, line 216 | 3 failed, 18 passed |

The adopted-file head run and each mutant run used separate serial `yield_run.sh` invocations. The wrapper was checked with `pgrep -f yield_run.sh` before each launch. The combined focused run on the commit passed all 405 tests in 10.18 seconds, including the optional-LightGBM, web-saver and existing caller suites.

The single unsandboxed fast suite on `e897c766` reported 10,235 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed in 25m 56s. The wrapper exited with code 1. The 53 failing and error test IDs matched `~/bts-runs/gate-failed.txt` exactly. All 122 C2 cases passed and the committed tree was clean afterward. Logs are in `.codex-review/wr-c2-r2/`.

## C2b

Commit `6663045e3c2afa742988a921485127a2d9035de8` on `fix/wr-c2`, parent `e897c766`.

| File | Against `e897c766` | Against `386aa846` |
| --- | ---: | ---: |
| `CLAUDE.md` | +1 / −0 | +3 / −1 |
| `src/bts/health/alert.py` | +25 / −5 | +170 / −4 |
| `src/bts/scheduler.py` | 0 | +32 / −16 |
| `tests/test_alert_status_callers_cs6b2.py` | 0 | +8 / −1 |
| `tests/test_wr_c2_carbon_v2.py` | +4 / −0 | +381 / −0 |
| `tests/test_wr_c2_r2_carbon.py` | +233 / −0 | +233 / −0 |
| `tests/test_wr_c2_receipt_identity.py` | +165 / −0 | +165 / −0 |
| `tests/test_wr_c2_recovery.py` | +3 / −1 | +569 / −0 |
| Total | 6 files, +431 / −6 | 8 files, +1,561 / −22 |

### Receipt identity

The opt-in reader rejects repeated JSON member names at any depth via the object-pairs hook in `src/bts/health/alert.py:260` and `:278`. The dedup predicate at `:334` requires the exact six-member key set, the current schema, sent status, the same slate and source list, a usable message ID, and the lowercase SHA-256 of the exact UTF-8 DM body. The digest is computed at `:317` and persisted only for an acknowledged send at `:360`. Failed and skipped receipts retain their original payloads. `test_legacy_health_dispatch_matches_base` covers both the implicit and explicit-false legacy route.

A five-member receipt from `e897c766` or `dfe774df` permits one repeated alert and then deduplicates. These receipts existed only in test directories. A byte-copy of a genuine acknowledgement for the same slate still deduplicates. The digest identifies the slate and text but cannot establish where a file came from. A long receipt path can fill the 200-character diagnostic and omit the file name and failure reason. That diagnostic limit is ledgered in `CLAUDE.md:93`.

### RED on e897c766

All cases below exercised the direct attempt branch and the real classification restart except the dedicated nested-reader check. The route tests counted health DMs over consecutive encounters and asserted that the recommendation transport had no calls.

| Fixture or control | Test | Base result |
| --- | --- | --- |
| Extra error, extra unknown member, duplicate status, duplicate schema, nested duplicate | `test_rejects_invalid_receipt` | RED for each fixture on both routes |
| Old five-field receipt, simultaneously the missing-hash case | `test_rejects_invalid_receipt` | RED on both routes |
| Null hash, numeric hash, uppercase hash, padded hash, hash from another slate | `test_rejects_invalid_receipt` | RED for each fixture on both routes |
| Missing and wrong-value or wrong-type schema, status, sent day, sent sources and message ID, with the correct hash | `test_rejects_invalid_receipt`, `wrong_field_cases()` | GREEN for all 52 fixtures on each route |
| Real opt-in acknowledgement for cutoff text on the same day and source | `test_foreign_dispatcher_receipts_do_not_suppress_c2[foreign-body]` | RED on both routes |
| Real default cutoff receipt copied in, and the same receipt with an ID inserted | `test_foreign_dispatcher_receipts_do_not_suppress_c2` | GREEN for the untouched cutoff receipt and RED with an inserted ID, on both routes |
| Valid receipt for another slate copied in | `test_foreign_dispatcher_receipts_do_not_suppress_c2[other-slate]` | GREEN on both routes |
| Genuine acknowledgement as written, with keys reordered, or with changed JSON whitespace | `test_genuine_receipt_deduplicates` | GREEN in all three cases on both routes |
| On-disk digest equals SHA-256 of full transport body | `test_receipt_hash_matches_exact_transport_body` | RED on both routes |
| Duplicate key inside an object nested in an array | `test_reader_rejects_duplicate_members_inside_nested_objects` | RED, the base parser kept the last value |
| Original Carbon round-2 adoption | `carbon_original.py` in the base scratch copy | All 18 cases GREEN |

### Existing assertion amendments

- `tests/test_wr_c2_recovery.py:310`, `test_unknown_receipt_ignores_unusable_path`. The exact rewritten receipt assertion expected five fields. It now expects the sixth field, calculated from the full C2 body.
- `tests/test_wr_c2_carbon_v2.py:120`, `test_classification_alert_runs_strictly_after_decision_commit`. Its exact acknowledged receipt assertion adds the full-body digest to the previous five fields.
- `tests/test_wr_c2_carbon_v2.py:276`, `test_readonly_receipt_directory_dedups_sent_and_reports_write_failure`. Its five-field genuine prior acknowledgement fixture now includes the digest of the full C2 body.
- `tests/test_wr_c2_r2_carbon.py:104`, `test_foreign_or_noncanonical_receipt_never_suppresses_c2`. The exact replacement assertion now includes the full-body digest alongside the five original fields.
- `tests/test_wr_c2_r2_carbon.py:111`, `test_opt_in_multi_source_receipt_rejects_unsorted_sources_and_dedups_swapped_inputs`. The old swapped-input call expected zero further sends. Under Amendment 2a it expects exactly one more send, a rewritten acknowledgement hashed from the newly ordered body, and deduplication on a further call in that order. The unsorted source-list fixture has the correct body digest so the list alone makes it ineligible. The first replacement still stores a sorted source list and a repeat in its original order deduplicates.

### Hostile inputs

| Case | Verification |
| --- | --- |
| Interrupt during receipt persistence | `test_restart_retries_unknown_alert_after_interrupt` covers `KeyboardInterrupt` and `SystemExit` before and after the durable acknowledgement. Both cases of the offline `test_interrupt_at_receipt_replace` also passed with the interrupt injected at the atomic replacement. The temporary file was cleaned up and the locked slate retried the alert once on restart without resending the recommendation. |
| Malformed or non-regular receipt | `test_unknown_receipt_ignores_unusable_path` covers malformed input, directory, FIFO with a five-second alarm, unreadable files and filesystem aliases. `test_reader_rejects_duplicate_members_inside_nested_objects` isolates recursive duplicate detection. |
| Midnight ET and year boundary | `test_unknown_receipt_uses_slate_across_midnight` checks both time boundaries with slate-specific receipts. |
| Failed or unconfirmed send | `test_only_unknown_alert_requires_message_id` covers transport failure, unusable IDs and acknowledgement on retry. |
| Restart mid-step | `test_restart_pages_unknown_delivery_without_resending` covers both crash windows and a dated correction between the marker and restart. |

### Runs

- The wrapped RED run against scratch `e897c766` sources reported 29 failed and 132 passed in 3.62 seconds. Its failure IDs are in `.codex-review/wr-c2b/pytest-red.log`.
- The one-process wrapped focused run passed all 469 cases in 10.23 seconds. It included every `test_wr_c2_*.py` module and the seven requested compatibility modules. The log is `.codex-review/wr-c2b/pytest-focused.log`.
- The one unsandboxed fast suite ran on committed `6663045e` through the wrapper. It reported 10,396 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed in 25m 32s. All 283 C2 cases passed. The wrapper exited with code 1. The comparison against `~/bts-runs/gate-failed.txt` matched all 53 failing and error IDs exactly, with no missing or unexpected IDs. The tracked tree was clean afterward. The fast-suite log is `.codex-review/wr-c2b/pytest-gate.log`.

### Scratch mutants

Each variant used a separate source copy and a serial wrapped pytest run. Each named route test failed on both direct and classification. All selected tests failed and all unselected cases were deselected. The tracked source stayed intact.

| Mutant | Test that detected it | Result |
| --- | --- | --- |
| Drop exact key-set check | `test_rejects_invalid_receipt[extra-error]` | 2 failed, 141 deselected |
| Accept a superset of keys | `test_rejects_invalid_receipt[extra-unknown]` | 2 failed, 141 deselected |
| Drop duplicate-member rejection | `test_rejects_invalid_receipt[duplicate-status]` and `test_reader_rejects_duplicate_members_inside_nested_objects` | 3 failed, 140 deselected |
| Drop body-hash comparison | `test_foreign_dispatcher_receipts_do_not_suppress_c2[foreign-body]` | 2 failed, 141 deselected |
| Hash a constant | `test_receipt_hash_matches_exact_transport_body` | 2 failed, 141 deselected |
| Hash only the alert message | `test_receipt_hash_matches_exact_transport_body` | 2 failed, 141 deselected |
| Write the hash but compare only the day | `test_foreign_dispatcher_receipts_do_not_suppress_c2[foreign-body]` | 2 failed, 141 deselected |
