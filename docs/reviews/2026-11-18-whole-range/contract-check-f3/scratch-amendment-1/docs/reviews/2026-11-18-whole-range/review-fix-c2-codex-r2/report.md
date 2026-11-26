# C2 round 2 review of `e897c766` against `386aa846`

## Verdict

**BLOCK**

## Blocking findings

- **B1 · Conflicting or extra receipt fields suppress the only immediate C2 page (U1/U4).** At 18:20 ET on 2027-03-25, save a pick with `delivery_attempted=true` and no confirmation. Configure a recipient and seed `health_state/delivery_outcome_unknown/2027-03-25.json` with the correct schema, `status="sent"`, `sent_day="2027-03-25"`, `sent_sources=["CRITICAL:late_delivery"]`, `message_id="seed-id"`, and `error="send_dm returned no usable message id"`. The direct path and the real `run_day` restart each locked `locked_unconfirmed`, sent zero DMs on two encounters and preserved the file byte for byte. An unknown extra member had the same result. So did raw JSON with `"status":"failed"` before a second `"status":"sent"`, or a bad schema before a second, correct schema. The predicate ignores extra fields and `json.load` silently takes the last value for a duplicate member. Require the exact sent-receipt keys and reject duplicate members in the opt-in reader. The scratch correction sent once and deduped on the next encounter on both routes. [Reader](../../src/bts/health/alert.py#L269) · [Predicate](../../src/bts/health/alert.py#L324) · [HEAD probe](receipt-identity.log) · [Scratch proof](receipt-correction.log)
- **B2 · Acknowledging another alert can silence C2 (U1/U4).** Use the same slate and pick state. Capture the real cutoff-refusal body, send it with the real opt-in dispatcher and an acknowledged ID, and place its emitted five-field receipt at the slate's own path. The body was `LATE DELIVERY REFUSED`, and the dispatcher wrote a receipt with the same schema, slate and source key C2 expects. Both scheduler routes locked and sent zero C2 DMs, with no warning or receipt rewrite. An untouched default-dispatcher cutoff receipt was correctly rejected. A copy of it with its cutoff message ID added also suppressed C2. Bind the opt-in receipt to the actual alert identity. Storing and checking the fixed U7 body's SHA-256 was enough in scratch. It sent once against a valid receipt for a different body and still deduped genuine C2 acknowledgements. The foreign opt-in call was a probe setup, as the scheduler's cutoff caller currently uses the default path. [Predicate](../../src/bts/health/alert.py#L324) · [Writer](../../src/bts/health/alert.py#L344) · [HEAD probe](receipt-identity.log) · [Scratch proof](receipt-correction.log)

## Notes

- **N1 · A long receipt path can consume the entire 200-character diagnostic.** With a 303- or 311-character path and an injected `OSError("disk full")`, both routes sent the alert and reported one bounded line. The line ended mid-directory and omitted the date-stamped filename and the error. The round-one Codex containment probe also tripped on its full-path assertion after relocation. It passed all 18 cases with its temporary root shortened. Keep the filename and failure text when abbreviating long paths. [Formatter](../../src/bts/health/alert.py#L242) · [Probe](diagnostics.log)
- **N2 · The archived earlier Carbon drafts are superseded.** The five report-designated versions ran green. All 12 earlier drafts returned the same exit status and final exception on `dfe774df` and `e897c766`. [Draft comparison](replay-draft-controls.log)

## 1. Receipt validation and deduplication

`probe_receipt_identity.py` ran 126 first-plus-repeat cases across direct and classification routes. The original B1 is fixed. Missing, mistyped or wrong-valued required fields, malformed JSON, and an unaltered receipt copied from March 26 all sent one March 25 alert and then deduped. Changing or adding elements in `sent_sources` did not suppress. The two-source dispatcher control rejected reversed order, rewrote sorted order and deduped next time. A real C2 acknowledgement suppressed both repeats. Six other fixtures suppressed the alert on each route as described above. No pick was resent. [Probe log](receipt-identity.log)

`probe_receipt_correction.py` verified strict receipt validation and body binding on scratch source. A prior five-field record is ambiguous under body binding and would be re-acknowledged once. The candidate preserved the default dispatcher's base parity. [Scratch proof](receipt-correction.log) · [Legacy comparison](receipt-correction-legacy.log)

## 2. Round-one replay and legacy parity

- Codex `probe_restart` and Carbon `probe_1c` reran real restarts before acceptance, after acceptance and after correction. Base had zero recovery DMs. HEAD had one per slate over two restarts, with recommendation-send counts 0/1/0. The Codex comparison matched persisted pick and decision hashes, return values and polling to base. Classification controls covered existing scoreable decisions, terminal picks and the best-effort decision writer. [Codex](restart-head-direct.log) · [Carbon](replay-carbon-probe_1c_restart_and_base_parity-head.log)
- Codex `probe_isolation` and Carbon `probe_2c` passed both cutoff orders, unrelated failure and EOD writes, midnight and New Year, and explicit versus undated launches. The legacy file and the per-slate receipt stayed independent. [Codex](replay-codex-probe_isolation-head.log) · [Carbon](replay-carbon-probe_2c_isolation_and_dedup-head.log)
- Codex `probe_receipts`, Carbon `probe_3b` and Carbon `probe_4b` passed invalid-ID and transport recovery, no-recipient behavior, special files, aliases, and interrupts before and after checkpoint. Both parity probes matched the default and explicit-false legacy cases byte for byte with `386aa846` on bodies and status files, and matched returned booleans, stderr and log content. Exception comparisons normalized relocated traceback coordinates. `probe_diagnostics.py` also confirmed the independent EOD backstop. [Receipts](replay-codex-probe_receipts-head.log) · [Legacy](replay-codex-probe_legacy_dispatch-head.log) · [Carbon acknowledgement](replay-carbon-probe_3b_ack_and_legacy_parity-head.log) · [Carbon storage](replay-carbon-probe_4b_damaged_receipts_and_interrupts-head.log)

## 3. U7 text and documentation

Both operator probes matched the full U7 transport body on both routes and retries, including the ASCII apostrophe. The pinned fixtures covered an unsent pick, a corrected double, a matching entry, cutoff, first pitch, midnight, New Year and private mode. The `status` invocation was read-only. The C2 ledger matches normal and backstop behavior, subject to B1 and B2. A real recommendation send returning `None` produced a scoreable `delivered` decision without a pick ID. The C2 alert told the operator to avoid entering or changing a pick based on the alert, and the entry watchdog said “Fix it now!”, as ledgered. [Codex](replay-codex-probe_operator-head.log) · [Carbon](replay-carbon-probe_5g_operator_text_and_watchdog-head.log) · [Ledger](../../CLAUDE.md#L92)

## 4. Round-two tests and mutations

The amended callbacks capture observations and the tests assert them outside `send_dm`. The direct, classification and operator tests require a `sent` receipt. The adopted Carbon test file is byte-identical to the supplied version.

All **308 tests passed in one process** on the approved rerun. The first run had 307 passes and one sandbox failure when the web fixture bound `127.0.0.1`. The rerun passed that fixture and the optional-LightGBM nested collection check. [Sandboxed run](head-combined.log) · [Approved rerun](head-combined-loopback.log)

Each scratch mutant ran both C2 test files together. **No survivors.**

| Mutation | Failures / passes | Detecting checks |
|---|---:|---|
| Drop schema check | 6 / 116 | Recovery rejects wrong type, wrong version and missing schema on both routes. [Log](mut-drop-schema-check.log) |
| Page before classification decision write | 9 / 113 | The moved recovery assertions catch all five ordering cases. Carbon catches four independently. [Log](mut-before-classification-decision.log) |
| Follow receipt-directory symlink | 4 / 118 | Carbon catches read and write through the directory link on both routes. [Log](mut-follow-receipt-directory.log) |
| Drop post-send entry recheck | 3 / 119 | Carbon catches replacement of the mid-send symlink, hard link and FIFO. [Log](mut-drop-post-send-entry-recheck.log) |
| Drop resurface guard | 4 / 118 | Carbon catches the legacy banner for a failed own receipt or shared alias on both routes. [Log](mut-drop-resurface-guard.log) |
| Accept `None` | 18 / 104 | Recovery catches the false `sent` receipt and lost retry. Carbon catches the lost classification retry. [Log](mut-accept-none.log) |

## 5. Remaining diff

`git diff --check 386aa846 HEAD` was clean. The existing caller test changes the two authorized unconfirmed cases and checks the shared file remains absent or `[]`. The EOD invocation, other production modules and the legacy dispatcher path retain their existing behavior. No further production findings beyond those listed above.

## What I ran

- Reviewed the whole-range diff, amendment, U1–U7 replacement text, both round-one reports and Carbon's adopted tests. Used `git archive` for the base and round-one source. Kept tracked files intact.
- Ran the archived probes via `replay_round1.py` and direct restart invocations, plus `replay_draft_controls.py`, `probe_receipt_identity.py`, `probe_diagnostics.py` and `probe_receipt_correction.py`. Archived replay copies adjust checkout-relative paths. Direct probes used `.venv/bin/python`, pinned ET clocks, synthetic state and captured transports.
- Ran the two C2 files with `test_lightgbm_optional.py`, `test_web_saver.py`, `test_alert_status_callers_cs6b2.py`, `test_late_delivery_guard.py`, `test_stale_delivery_cs4c.py`, `test_stale_delivery_cs4c_carbon.py` and `test_morning_state_cs6b1.py`. `prepare_mutants.py` made six isolated source copies. Each mutant ran both C2 files with its own source on `PYTHONPATH`.
- All eight pytest invocations went through `~/bts-runs/yield_run.sh`, serially after 21:45 ET, with `pgrep -f yield_run.sh` before each. The `.watch` logs record the commands and exits. The suite used a loopback server in the web fixture. The direct probes used offline fixtures.
