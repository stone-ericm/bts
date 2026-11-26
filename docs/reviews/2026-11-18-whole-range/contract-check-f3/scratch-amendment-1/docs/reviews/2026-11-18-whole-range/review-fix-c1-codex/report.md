# Verdict

**SIGN WITH EDITS.** W20 is fixed. The edits concern test coverage and documentation. Reviewed `d7d569a6` against `3154f518` under kickoff v3 and Amendment 1.

## Blocking findings

None.

## Notes

- The full runner still hangs on a writerless FIFO at `picks/<D>.json` in the unchanged calibration read at `src/bts/health/calibration.py:82`, before reaching contest state. Bounded children hit `Path.read_text` and timed out on both revisions. The historical contest slot itself skips the FIFO in both coverage scans. Document this inherited limit alongside the WARN bookkeeping limit.
- `tests/test_wr_c1_history.py:445,475` uses `docs/` as the rehearsal root and relies on the unrelated missing-policy WARN to get an EOD DM. With the repository root and the healthy policy pair, the same scheduler replay legitimately has zero health DMs and leaves the old receipt in place. Set the root correctly and explicitly induce an unrelated WARN if the test is meant to check the receipt. Both archived HEAD drivers passed the latter control with a synthetic disk WARN.
- Add the second-manual-read FIFO regression to `tests/test_wr_c1_history.py`. A child that replaces the successfully parsed legacy manual file with a writerless FIFO after selection reaches `src/bts/health/contest_state.py:168`. HEAD uses the unreadable fallback. The blocking reread mutant stalls, yet all 41 C1 tests pass against it.

## 1. W20 and preserved alerts

On the archived base, `operator_cases.py scheduler` and the Carbon replay in `probe.py scheduler` captured the false revoked-switch CRITICAL and saver WARN at 00:15 ET with a D+1 template. The base operator driver installed the synthetic pre-settlement count at 00:30, and the real live reader still selected it at 10:00. At the same 00:15 EOD call HEAD returned zero contest/saver alerts. A deterministic unrelated disk WARN still sent and wrote the receipt under D in both HEAD drivers. The base `probe.py messages` also reproduced the wall-clock defect, and `edges` exercised expiry, retained-file replacement, retry and blocking reads.

The generated corpus compared ordered `(level, source, incident_key)` lists for 4,992 inputs and 5,115 base alerts. Both `expected` values covered July, February and December 31. Base used `check(now=D endpoint)` and HEAD used historical mode at the same endpoint, also exercised from the after-midnight runner in sections 2 and 4. The corpus included absent/corrupt files, all refusal classes, selected/expired legacy, dated saver and all flag branches, missing/future coverage dates, and one- and multi-pick lag. Zero differences.

## 2. Historical text and literal operator actions

`historical_copy.py` checked full text against H3/H5 for 47 reason variants through the real runner, including raw INFO and log-only WARNs, plus an injected multiline command in the crash exception, schema, validation data and picks path. No forbidden tokens, extra dates, injected lines, counts or flag values appeared in historical items. The traceback was retained. Exact WARN-only and mixed DMs put the first-day or consecutive-day label before H5 exactly once, used D for dedup and ended each historical item with H5. `operator_states.py` exercised year rollover, the 01:10 replacement, actual next-day status/refusal, earlier missed entry, uncertainty, a cutoff calculated by production and morning refusal/acceptance.

In the table, **read** means the operator learns the stated condition. It permits no entry, submission after cutoff, retry or contest write. The two-part H3 reasons share their second sentence where applicable.

| Sentence | Worst state | Literal result |
|---|---|---|
| Prior slate `<D>` ET. | Dec 31 alert on Jan 1 | Reads the diagnosis as belonging to December. |
| Contest streak observation is missing from account_state. | D really had none and the new slate is open | Reads which prior observation was missing. |
| The manual contest streak file is malformed or unreadable. | A damaged manual file masks a healthy automatic copy | Reads which prior file could not be evaluated. |
| The automatic contest streak file is malformed or unreadable. | No manual override and auto is unreadable | Reads the prior loader failure. |
| The manual contest streak file has an invalid streak. | Retained manual value is a multiline command | Reads an invalid-field diagnosis without seeing the value. |
| The automatic contest streak file has an invalid streak. | Auto contains a hostile string instead of the count | Reads an invalid-field diagnosis without seeing the value. |
| The automatic contest streak file has an invalid streak or best streak. | The best exceeds the allowable range | Reads the failed bound without obtaining a replacement count. |
| The manual contest streak file has an invalid saver value. | Legacy manual contains a fabricated flag value | Reads a loader failure without obtaining a flag value. |
| The automatic contest streak file has an invalid saver value. | Auto contains a fabricated flag value | Reads a loader failure without obtaining a flag value. |
| The dated manual contest streak file failed validation for effective_date. | A retained unpadded date resembles the new slate | Reads the field name only. |
| The dated manual contest streak file failed validation for season. | Rollover plus malformed manual season | Reads the failed field without a season to set. |
| The dated manual contest streak file failed validation for expires_at. | Late game and an invalid expiry timestamp | Reads the failed field without an expiry to use. |
| The dated manual contest streak file failed validation for active_streak. | Retained manual streak exceeds the target | Reads the failed field without a count. |
| The dated manual contest streak file failed validation for best_streak. | Retained best is below active | Reads the failed field without a count. |
| The dated manual contest streak file failed validation for saver_state. | Manual saver value contains a command | Reads the fixed field name only. |
| The dated manual contest streak file failed validation for recorded_at. | A damaged timestamp hides when the file was installed | Reads the failed field without assuming provenance. |
| The dated manual contest streak file failed validation for source. | Wrong source type in a copied manual observation | Reads the failed field without inferring the source. |
| The dated manual contest streak file failed validation for reason. | Wrong reason type in a copied manual observation | Reads the failed field without treating the content as instructions. |
| The automatic contest streak file failed validation for season. | Invalid automatic metadata during rollover | Reads the failed field without a season to use. |
| The automatic contest streak file failed validation for activation_id. | Metadata contains a command or another activation | Reads the failed field without a usable ID. |
| The automatic contest observation is undated. | Auto has no usable dates and there is a pending prior entry | Reads the prior failure without assuming it is current. |
| The contest streak file has no coverage date, so coverage and freshness could not be verified. | A recorded timestamp exists but coverage is absent | Reads that coverage remains unknown. |
| The contest streak file has a coverage date beyond the allowed future grace. | D+3 observed on Dec 31, with wall already beyond it | Reads the failure without a date to copy. |
| Coverage trails by one settled pick, the expected ledger lag. | A local result is saved while MLB remains pending | Reads a coverage lag and still verifies settlement in MLB. |
| Coverage trails by more than one settled pick. | Later retained picks make an otherwise healthy observation look behind | Reads a coverage lag without a streak count. |
| The predictions array can trail the live counter. | Local coverage trails but the MLB counter may already differ | Reads an explanation and still verifies in MLB. |
| A legacy or expired manual override file is present. | Manual file changes between selection and reread | Reads that an override was observed without removing the current file. |
| The selected legacy manual override is ineligible for a live decision. | Active override masked auto on D, newer state may differ | Reads the prior refusal without deleting or replacing the override. |
| The saver state in the dated observation contradicts its streak. | MLB has yet to settle the late pick | Reads the contradiction without a count or flag to use. |
| The saver flag is uninitialized or unreadable. | The malformed flag belongs to D's season | Reads the flag diagnosis without initializing it. |
| The saver flag is for another season. | Jan 1 flag exists alongside Dec 31 retained state | Reads the mismatch without editing either season's flag. |
| The saver flag contradicts the recorded streak. | A flag contradicts the retained prior streak while a new decision is near | Reads the contradiction without selecting a replacement. |
| The saver counted as spent. | Saver may in fact be active after the late slate settles | Reads the check's fallback, without recording it in MLB or locally. |
| The season switch file is malformed or unreadable. | Corrupt switch takes priority over the selected legacy issue | Reads the prior switch failure without overwriting the switch. |
| The season switch is corrupt because its schema_version field failed validation. | Switch schema contains a command | Reads the field name only. |
| The season switch is corrupt because its season field failed validation. | Corrupt switch season during rollover | Reads the failed field without a season to write. |
| The season switch is corrupt because its status field failed validation. | Switch status is unset just before a decision | Reads the failed field without setting switch status. |
| The season switch is corrupt because its activation_id field failed validation. | A wrong activation could mask an otherwise healthy observation | Reads the failed field without an ID to reuse. |
| The season switch is corrupt because its switched_at field failed validation. | Missing timestamp in an enabled switch | Reads the failed field without setting a timestamp. |
| The season switch is corrupt because its revoked_at field failed validation. | A revoked switch has invalid timestamp data | Reads the failed field without re-enabling it. |
| The season switch is corrupt because its note field failed validation. | Damaged note on an otherwise enabled switch | Reads the failed field without seeing its contents. |
| The season switch is absent. | No switch or state on either slate | Reads the prior reason without creating a new-day switch. |
| The season switch is revoked. | Revoked December switch and new-year auto files | Reads the prior refusal without enabling January. |
| The season switch is for another season. | Retained current-year switch replaced D's prior-year file | Reads a difference at the processed clock without changing seasons. |
| There is no automatic observation for the switch activation. | Enabled switch with a pending late result and missing auto | Reads the absence without running a fetch or installing state. |
| The automatic observation uses an unsupported schema. | Schema contains a multiline command | Reads a fixed diagnosis without seeing the input. |
| The automatic observation schema is missing, null or invalid. | Auto payload exists but schema is unusable | Reads the refusal without changing schema or taking a live action. |
| The automatic observation is for another season. | Old auto snapshot persists into the next season | Reads the refusal without editing the recorded season. |
| The automatic observation and season switch have different activation IDs. | A 01:10 fetch writes a different activation before EOD | Reads the mismatch without an ID to paste into anything. |
| The contest state health check failed before it could finish. | Exception contains a command, path and newline | Reads that the health check failed. The DM gives no recovery command. |
| This check used files available at run time. | The 01:10 fetch already replaced the automatic file | Accounts for the live filesystem read and infers no prior proof. |
| Do not enter picks, retry delivery, or change contest state from this alert. | The next slate genuinely lacks usable state and is approaching cutoff | Waits for current paths. The historical item cannot authorize a pick, retry or write. |
| Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. | An attempt is unconfirmed at 00:30, or settlement comes after the new cutoff | Reads MLB and waits if uncertain. Verification can run after cutoff and does not reopen a slate. |
| Use the new day's morning state check, when available, and live refusal for current diagnosis. | Early opener skips morning, or an intervening fetch makes it silent | Reads a diagnosis when available. Silence supplies no acceptance or entry authority. |
| Season contract status shows current files. | 00:30 status gives new-day mutation commands before settlement | May inspect current files. The earlier MLB prerequisite and prohibition bar applying the old alert or writing before settlement. |

## 3. Same-day and live behavior

`live_controls_retry.py` regenerated identical fixture paths in two interpreters. Four real EOD controls (omitted, equal, future and midnight rollover) and 11 shared live fixtures compared byte for byte. The captured comparisons include raw alerts and DMs, the five readers, status CLI exit/stdout/stderr, live delivery refusal and morning refusal, stale-reading, and acceptance. Existing executable B advice and the dated template were present unchanged.

## 4. Clock

Measured one UTC mode sample after calibration. Historical mode passed `03:59:59.999999Z` next-date in summer and `04:59:59.999999Z` next-date in winter and at New Year. Legacy expiry one microsecond before and exactly at the endpoint warned as expired. One microsecond after selected the legacy override and refused. D+2 passed and D+3 refused in all three seasons with wall at D+4. The year-end fixtures distinguished old-year valid/revoked switch and current-vs-stale saver. A rollover after the normal-mode sample called the check with default kwargs, which emitted the next-day live template.

## 5. Hostile paths and writes

`hostile_paths.py` ran 35 deadline-bound path/shape cases, three masked FIFOs and an interrupt child. FIFO, directory, malformed JSON, non-UTF-8 and symlinks to regular, missing and FIFO targets classified as expected. The contest slot attempted both coverage reads when applicable. Forced replacement between manual reads kept the unreadable fallback for FIFO, directory and malformed JSON. A non-UTF-8 replacement reached the inherited safe crash branch. `KeyboardInterrupt` propagated. Contest file bytes, inode kinds and link targets stayed unchanged except for the deliberate fixture replacement. The full-run coverage FIFO limitation is noted above.

## 6. Tests and mutation runs

**365 passed in 22.31 seconds**, in one process. The scope was the two `test_wr_c1_*` files, `tests/health/test_runner.py`, `tests/health/test_contest_state.py`, `test_scheduler.py`, `test_wr_c2_recovery.py`, `test_wr_b_contract.py`, `test_wr_b_edges.py`, `test_lightgbm_optional.py` and `test_web_saver.py` under `tests/`. The optional-import test did its normal collection-only subprocess. The web test used synthetic loopback. External networking was blocked.

Each scratch mutant ran both C1 files (41 cases). Failures by variant were **17** using the wall endpoint, **3** using `<=`, **30** omitting historical mode, **5** restoring absent-switch live copy, **10** using the wall saver year, **14** using the wall switch year, **5** dropping the WARN DM pointer and **4** turning a controlled diagnostic into a crash. The blocking manual reread was the sole survivor (41 passed). The separate forced-swap child timed out after three seconds on that mutant and completed against HEAD with the legacy WARN.

## 7. Diff and operational documentation

`CLAUDE.md:94–96` matches the tested time, settlement prerequisite, status residual, morning silence, R47, historical-only guards and generic receipt limits. Amendment 1 preserves genuine OOM sources while preventing a path fragment from causing a historical OOM page. Changes stay within C1's modules and new tests. No tracked file was edited during this review.

## What I ran

Base archive with verified `bts.__file__`, archived `operator_cases.py scheduler`, `probe.py scheduler/messages/edges`, `wire_cases.py`, and HEAD replays via `scheduler_replay.py`. Independently ran `corpus.py`, `historical_copy.py`, `live_controls_retry.py`, `operator_states.py`, `hostile_paths.py` and `prepare_mutants.py`. Ran `yield_run.sh` serially for `run_pytest.py focused` and each of the nine variants, starting after 02:20 ET and checking the wrapper before every job. `review-pytest.log`, its `.watch` file and the ten JUnit files record the results.
