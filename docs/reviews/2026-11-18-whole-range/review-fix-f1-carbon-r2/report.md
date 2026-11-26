# Verdict

**SIGN** for `1d30de86` and `15b27474` on base `1e69d0f0`.

## Blocking findings

None.

## Edits

None.

## Notes

- **Test coverage gap for non-null, non-integer `game_pk` values (`tests/test_wr_f1_wrong_game.py`, `tests/test_wr_f1_review.py`).** Both F1 test files exercise `game_pk = None`, never `bool`, `str`, or `float`. Three scratch changes on `src/bts/cli.py` — weakening `type(slot["game_pk"]) is int` to `isinstance(slot["game_pk"], int)` (allowing `True`/`False`), weakening it to `slot["game_pk"] is not None`, and weakening `type(game_pk) is int` in `LEGS` rendering to `game_pk is not None` — pass all 83 F1 tests. Our probe `probe_2c_rows_and_texts.py` checks `True`, `False`, `"101"`, and `101.0` in both `N12` and `R2` leg formatting and catches all three (`log_5_mutations.json`).
- **Environment prerequisite (`tests/test_audit_driver.py`).** Fails with 33 failures and 16 setup errors because GNU `rsync` 3.x is absent on this machine, matching base `1e69d0f0`.
- **Unreachable inner time fallback (`src/bts/cli.py:2200`).** As R2's limit note states, an unreadable saved `game_time` raises `ValueError` in `_earliest_pick_game_et` before any fetch on both base and `15b27474`, so the `LEGS` first-pitch `except` fallback cannot be reached through the CLI today.
- **Unchanged `entry_receipt.qualify` with null `batter_id` (`src/bts/entry_receipt.py:151`).** If a hand-edited pick file had `batter_id: null` while an entered row's `playerId` was absent from `crosswalk`, `crosswalk.get(r["playerId"]) == s["batter_id"]` evaluates `None == None`. Saved `Pick` records always carry integer `batter_id`s, and `qualify` is unchanged by contract.

## 1. Scope and base comparison grid

`git diff --name-status 1e69d0f0 15b27474 -- src docs/ops tests scripts` confirmed only `src/bts/cli.py`, `src/bts/entry_receipt.py`, `docs/ops/pick-entry-receipt-v1.md`, `tests/test_wr_f1_wrong_game.py`, and `tests/test_wr_f1_review.py` changed; zero existing test or script files were touched. Archived `1e69d0f0` `src` into `carbon-review/wr-f1-r2/base_archive/` (`probe_1b_parity_runner.py` + `probe_1_parity_grid.py`, confirming `bts.__file__` on both sides) and ran 1,724 paired CLI checks across verifier results, single and double `qualify` slot states, all 16 escalation subsets, prior marker statuses/dates/unreadable JSON, DM success/failure/no-recipient, delivery states, fetch/identity failures, and window/tier/cutoff times (`log_1_parity_grid.json`). All 495 cases outside A1(a), A1(b), and prior wrong-game markers (including all 160 clean-day sweep cases) were byte-for-byte identical after ID/time normalization; the remaining 1,229 cases matched A1–A4, R1, and R2.

## 2. Contract rows N1–N13 and M1–M9 (plus M10–M13)

Ran `probe_2c_rows_and_texts.py` through the real Click `check-pick-entered` and `day-outcome` commands with real `.json.gz` captures on disk (`log_2_rows_and_texts.json`). Every row N1–N13 and M1–M13 matched its expected exit code, stdout, stderr, DM text, marker fields, receipt fields, EOD health output, and morning outcome, including window boundaries (`18:24` outside vs `18:25` inside, `19:35` cutoff outside) and both leg orders on doubles (M3, M13).

## 3. False alarms and missed cases

Across the 1,724-case grid (`probe_1b_parity_runner.py`, including 49 ordered double slot-state pairs, pending vs settled profile vs duplicate-both response shapes, shared-batter selections, exact duplicate selections, and `game_pk` values `None`, `"101"`, `101.0`, `100.0`, `True`, `False`, `0`, `-5`), 0 false alarms withheld confirmation outside A1(a)/A1(b) and 0 missed cases wrote `confirmed` when a distinct-batter slot with exact `int` `game_pk` qualified as `wrong_game`.

## 4. Escalation tiers (A2)

Ran `probe_3b_tiers_and_r1_hunt.py` across a full multi-step lifecycle (`wrong_game` fail then retry -> repeat silent -> removal fail then retry before `T-30` sending `initial` immediately -> `t30` fail then retry -> shared `t30` silent after removal -> `t15` once -> tiers exhausted -> full game proof sending all-clear -> later capture change skipped at `already confirmed`), no-recipient runs, cutoff runs, and failed confirmation DMs (`log_3_tiers_and_r1_hunt.json`). Each tier fired once when delivered, retried after failure, preserved `initial` across `wrong_game`, and sent the all-clear only when A1(b) and A2 require.

## 5. Operator texts under R64 (A3, Amendment 3, R2)

`probe_2c_rows_and_texts.py` checked character-for-character equality against the contract templates across single and double wrong-game legs, case (b) legs without `wrong_game` (omitting `, the entry is recorded in game ...`), missing/empty/non-string batter names (`None`, `""`, `0`, `False`) in both `LEGS` and `locked_unconfirmed`, non-integer `game_pk` (`recommended game ID unavailable`), `locked_unconfirmed` precedence at `wrong_game`/`t30`/`t15`, unresolved stdout roles (`pick`, `double-down`, `pick + double-down`), cutoff stdout, and Amendment 3 `match_game_unchecked` stdout/stderr/DM. All matched verbatim; no R64 issue exists.

## 6. Standing rows (A7, A5, Amendments 2 and 3)

`probe_4b_standing_and_receipts.py` checked 25 malformed/non-regular capture shapes directly against `entry_receipt.load_units` and through the CLI (`log_4_standing_and_receipts.json`): all 15 raising shapes (non-gzip, truncated gzip, invalid UTF-8 `0xff`, UTF-16 BOM, broken JSON, non-object top levels, non-iterable `units`, directory, `chmod 000`) gave terminal `confirmed`/`match_game_unchecked`, two stderr lines, and no sealed receipt; all 9 silent shapes gave base `confirmed`/`match` with `unit_unverified` and empty stderr; mixed valid+malformed items still caught `wrong_game`. Checked named pipes (blocking before marker on `match` and only at `publish` after `alerted` marker on `no_pick`), `KeyboardInterrupt` and `SystemExit(17)` (single vs double read failure, plus interrupt before marker write), connection reset after DM send, 5 inherited malformed marker shapes (matching base), and the Dec 31 ET / Jan 1 UTC capture-filename year gap (A6.6).

## 7. Receipt and morning report (`bts day-outcome`, A6.4, A6.5)

In `probe_1b_parity_runner.py` and `probe_4b_standing_and_receipts.py`, wrong-game receipts differed from base only in `marker_status` and `exit_code`, keeping `verifier` (`ok: true, reason: "match", game_qualified: false`) unchanged. Checked separate capture reads between `qualify_run` and `publish()` (reproducing both watchdog `WARN` + morning `entered` and watchdog `confirmed` + morning `entry_unverifiable`), `day-outcome` across `entered`, `entered_game_unverified`, `entered_late`, `not_entered`, `entry_unverifiable`, `deliberate_skip`, and `no_games`, and real rollback execution on archived `1e69d0f0`.

## 8. Tests and scratch changes

Ran all 63 matched test files in bounded foreground batches (`nice -n 19`, `TZ=America/New_York`): `tests/test_wr_f1_wrong_game.py` (70 passed), `tests/test_wr_f1_review.py` (13 passed), 60 existing files (3,456 passed, 4 skipped, 1 xfailed; `test_field_products_builders.py` is a helper module with 0 tests), and `tests/test_audit_driver.py` (33 failed, 16 errors, unchanged missing GNU `rsync` 3.x). In `probe_5b_mutations.py`, all 6 required scratch changes from the brief (plus 2 `game_unresolved` omission changes) failed the F1 tests (`M13` now catches confirming when any leg is confirmed); 3 subtler `game_pk` type-check changes (`bool`/`str`/`float` vs `None`) passed the F1 tests and were caught by `probe_2c_rows_and_texts.py` (recorded in Notes).

## 9. Other diff review

Reviewed every line of `1d30de86` and `15b27474` in `src/bts/cli.py`, `src/bts/entry_receipt.py`, and `docs/ops/pick-entry-receipt-v1.md`. `selection_slots` preserves `EntryReceipt._selection` output identically, `mark` attaches `game_unresolved` in one place only on non-`confirmed` markers, and the added ops sentence matches runtime behavior. Nothing else in the diff is wrong.

## 10. Round 1's finding and its class (R1)

Archived `1d30de86` into `carbon-review/wr-f1-r2/r1_archive/` and ran `probe_3b_tiers_and_r1_hunt.py`: all 6 M11/M12 bridge cases (`crosswalk`, `removal`, `mismatch` with and without prior `initial`) failed on `1d30de86` and passed on `15b27474`. Then ran 116 multi-step sequences combining 8 opening wrong-game states (failed/succeeded/no-recipient DMs, cutoff on first check, and legacy/Round-1 markers without `game_unresolved`) with 1- and 2-step bridges (`present_unverified`, `no_pick`, `mismatch`, fetch/identity failure, `outside_window`, cutoff pass, `t30`/`t15` transitions) and 8 unproved match endings (conflicting capture, null `feedId`, `wrong_round`, `ambiguous`, string `playerId`, shared-batter switch, and both partial double proofs), plus midnight ET rollover: 0 sequences confirmed or sent an all-clear before full game proof of every leg.

## 11. Unchanged outside wrong-game days and reader compatibility

Across all clean-day cases in `probe_1b_parity_runner.py` (including stale prior-date markers carrying `game_unresolved: true` and unreadable markers), every observable matched base `1e69d0f0` identically. Checked every repository reader of `pick_entry_check.json` (`src/bts/cli.py`, `src/bts/health/pick_entry.py`, and rolled-back `1e69d0f0` in `probe_2c_rows_and_texts.py`): none mis-reads `"game_unresolved": true`.

## 12. Round 1's five edits (R2)

`probe_2c_rows_and_texts.py` verified all three R2 text changes verbatim (`locked_unconfirmed` fallback names `batter 1010 + DD Batter 1020`, `recommended game ID unavailable` for non-integer `game_pk`, and `during this check` vs `during fetch` on cutoff) and empirically verified all three corrected limit sentences (`A6.2` EOD health levels across all 4 marker statuses, `A6.4` rollback confirmation and all-clear DM on `1e69d0f0`, and `A6.5` clean morning outcomes alongside unresolved watchdog warnings).

ran: base (`1e69d0f0`), round-1 (`1d30de86`) and candidate (`15b27474`) archives, 1,724-cell parity/qualifier grid, N1–N13 and M1–M13 rows, R64 text and R2 limit probes, 116-sequence R1 class hunt, 25 capture-shape + FIFO/interrupt/NYE/marker/receipt/morning probes, all 63 matched test files, and 11 scratch changes; did not run: full fast suite, network/live transports, or `test_audit_driver.py` assertions gated on absent GNU `rsync` 3.x.
