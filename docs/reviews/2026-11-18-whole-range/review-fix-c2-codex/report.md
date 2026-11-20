# C2 adversarial review of `dfe774df` against `386aa846`

## Verdict

**BLOCK**

## Blocking findings

- **B1 · A malformed own receipt can restore the silent day (U1/U4).** On 2027-03-25 at 18:20 ET, save a pick with `delivery_attempted=true`, no confirmation, and a configured recipient. Seed `health_state/delivery_outcome_unknown/2027-03-25.json` with `{"schema_version":["damaged"],"status":"sent","message_id":"prior-ack","sent_day":"2027-03-25","sent_sources":["CRITICAL:late_delivery"]}`.
  Both the direct encounter and the real `run_day` restart suppress the DM. The bad file stays byte-identical, the pick locks, and no diagnostic is printed. The receipt is malformed under the format the C2 writer emits. The dedup predicate checks the other fields and accepts any schema. [src/bts/health/alert.py:324](../../src/bts/health/alert.py#L324)
  Require `prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION` in the opt-in dedup predicate and add the damaged-schema fixture. `probe_receipts.py` observed 0 DMs on HEAD. With that one condition in a scratch copy it observed 1 DM, a valid replacement receipt, and dedup on the next encounter.

## Notes

- **N1 · Existing watchdog can contradict C2.** A recommendation transport returning `None` really writes a scoreable `delivered` decision without a pick ID. C2 correctly preserves that decision and pages. With no matching MLB entry and 33 minutes to submit, the watchdog's untouched generic branch names the unconfirmed pick and says “Fix it now!” The `locked_unconfirmed` branch agrees with C2. [src/bts/cli.py:2126](../../src/bts/cli.py#L2126)
- **N2 · Coverage gaps.** The existing “matches base” parametrization substitutes synthetic messages into the dispatcher. It does not run those named callers and omits the missed-pick caller. The live caller comparison below covers them. The receipt parametrization misses a damaged schema and checks neither the receipt-directory symlink's target nor its diagnostics.
- **N3 · Ordering assertion is swallowed by the sender.** Moving the classification alert before decision finalization passed all 78 new tests. The test asserts inside `send_dm`, where an assertion failure is contained as a transport error. Capture the snapshot in the callback and assert outside it, as in `probe_restart.py`, which failed on the early-call mutation. [tests/test_wr_c2_recovery.py:136](../../tests/test_wr_c2_recovery.py#L136)

## 1. Real restarts and state preservation

`probe_restart.py` drove the real Click scheduler and `run_day` on both revisions. Each crash ran through two recovery launches. Before acceptance, after acceptance before checkpoint, and correction after the attempt marker produced 0/1/0 recommendation DMs respectively on both revisions and no resends. Each went from zero recovery DMs on base to one on HEAD.

Pick and decision hashes, lock, `committed_pick_written`, one normal result poll and two normal `None` returns matched base. The second launch retained the decision bytes. Next-morning state stayed `not_entered` in these no-entry fixtures.

Fresh classification, pre-existing scoreable `locked_unconfirmed`, `delivered` and `private_locked`, a terminal hit, and a decision writer returning `None` all paged on HEAD. At send time the lock and the decision-write attempt had happened. Existing decision bytes stayed exact. Polling was skipped for the terminal result and for the missing decision, whose committed flag stayed false and missed-pick alert still ran. State and returns otherwise matched base.

## 2. Receipt identity and isolation

`probe_isolation.py` ran both cutoff/recovery orders under one health root with separate pick stores. Both routes sent once. An unrelated failed health DM and an actual previous-slate EOD dispatch changed only the shared status. Every recovery encounter preserved the shared bytes, and the intervening legacy sends preserved the own receipt.

At March 25→26 and December 31→January 1, explicit `run_day(D)` sent once for D before midnight and deduped D at 00:01 ET. The D+1 launch wrote its own receipt and sent independently. A shared-only prior send did not suppress the first own alert.

## 3. Acknowledgement and default dispatcher parity

`probe_receipts.py` ran the direct encounter followed by two real restarts. `None`, empty/whitespace strings, an integer and a transport exception wrote `failed` without sent fields. The next encounter attempted again, then deduped after acknowledgement. An ID with surrounding spaces was stored exactly. No recipient wrote `skipped_no_recipient` with no transport, then sent once configured. Attempted booleans and full bodies matched the contract, including against a shared failed status.

`probe_legacy_dispatch.py` captured the actual EOD, 4c stale warning, 6b-1 refused and stale morning checks, cutoff, void, both missed-pick copy branches and contest-state refusal. It exercised no recipient, duplicate, exception, absent ID and resurface retry through that last caller. Base, omitted keyword and explicit false matched for all 14 cases on bodies, returned values, status bytes, stderr and log records. Exception comparison kept levels, messages, types and source statements and ignored checkout paths and relocated traceback line numbers.

## 4. Damaged storage, diagnostics and interrupts

`probe_receipts.py` exercised empty object, truncated JSON, arrays, null, wrong-shaped sent keys, blank ID and wrong date. Each sent and replaced the readable file. Direct and classification encounters with a directory, timeout-bounded FIFO, unreadable file, symlink, hard link, unwritable or blocked parent, or parent symlink both attempted and returned normally. Special entries retained their inode and type, the shared file stayed exact, and each attempt emitted one bounded diagnostic. B1 is the malformed-schema exception.

`probe_failures.py` injected formatting, dispatch, mkdir, read, write, transport, simultaneous write/transport failure, closed alert stderr and unprintable errors on both routes. Lock and decision were already recorded, ordinary errors were contained, and the failure line was bounded and escaped or suppressed when stderr was unavailable. EOD still independently sent `pick_delivery`. With the C2 attempt unacknowledged it made no C2 retry until the next qualifying restart.

`probe_receipts.py` interrupted immediately before and after the atomic replace, after captured health transport acceptance, on both routes. `KeyboardInterrupt` and `SystemExit` propagated. Before replace there was no receipt and the next start repeated the alert. After replace the checkpoint suppressed both subsequent starts. Neither window resent the recommendation or changed decision bytes.

## 5. Operator text and watchdog

`probe_operator.py` captured these exact bytes on both routes, including the first missing-ID attempt and the retry. Version 3 specifies the ASCII apostrophe.

```text
BTS health CRITICAL alert(s):
- [late_delivery] DELIVERY OUTCOME UNKNOWN for 2027-03-25 ET. A send attempt for that slate was never confirmed, and the scheduler locked the slate without resending. Check the DM or public feed and the MLB entry for 2027-03-25 ET. Do not enter or change a pick based on this alert. If 2027-03-25 is still today in ET and its submission window is open, run bts season-contract status with --picks-dir set to the scheduler's picks directory before making any entry decision for that slate.
```

| Exercised state | Literal action and consequence |
|---|---|
| Never left, 18:20 ET | Inspect the dated DM/feed and MLB entry and find neither. Read status while the window is open. The alert supplies no batter and forbids using it to enter. |
| Correction at 15:25:30, recovery 15:26 | Inspect the old attempt and read live status showing streak 46. The real fresh choice is a single at 46, with the same primary 27250 as the old double at 1. The alert gives no permission to reuse the old double. |
| Remote accepted and entry matched | Inspect the matching dated evidence, leave the entry alone. The watchdog reported a match and sent no nag. |
| 18:56 after cutoff, 19:04 after pitch | Inspect history only. The status instruction's open-window condition fails. The watchdog was outside its window. |
| 00:01 after midnight or New Year | Inspect the old dated slate. The status instruction's same-day condition fails, and the alert gives no instruction for the new slate. |
| Private/research | A fresh private lock emitted nothing. A carried interrupted outward attempt still paged after the switch, with the same dated inspection and no entry instruction. |

The sole command in the alert is read-only `bts season-contract status`. Actual invocations left the fixture trees byte-identical. A refused status can display the existing conditional fetch/enable instructions. Those commands are absent from the C2 DM. The real B watchdog gave verification and the read-only status command for `locked_unconfirmed`. See N1 for the preserved `delivered` case.

## 6. Edited existing test and CLAUDE.md

The existing parametrization changes only the two authorized unconfirmed cases. It now requires a classification DM, asserts the shared file remains missing or exactly `[]`, and reads the slate receipt. Assertions for the other callers are preserved. The `CLAUDE.md` summary matches observed normal receipt and retry behavior, backstop independence, and undated slate selection. B1 is a damaged-receipt exception to that summary.

## 7. Tests, base failures and mutations

The requested eight-file run passed **264** tests in one process, including all **78** C2 cases. The optional-LightGBM test's nested collection check also passed. [head-combined.log](head-combined.log)

The C2 file on base had **67 failures, 11 passes**. The amended caller file added four expected failures and 18 passes. [base-red.log](base-red.log)

- The three real restarts, five classification fixtures and two rollover fixtures lacked the required recovery DM. The direct checkpoint, six reply cases and no-recipient case lacked the own receipt.
- The shared sent fixture suppressed all eleven damaged-path cases. The twelve copy cases got old text or silence. Both cutoff orders exposed shared-key suppression. Four interrupt cases failed on missing retry or missing own receipt.
- Eleven containment cases lacked C2 diagnostics or let a setup error escape. The nine legacy comparisons passed the omitted-keyword assertions, then failed on base's absent keyword. Ten lifecycle controls and the classification-with-stderr-closed control passed.

Each scratch mutant ran the full 78-case C2 file.

| Mutation | Result and detecting cases |
|---|---|
| Wall date as dedup key | 2 failures, both midnight boundaries |
| Wall date as receipt filename | 2 failures, both midnight boundaries |
| Shared status file | 31 failures, including both isolation orders |
| Accept `None` as acknowledged | 14 failures, including the reply assertion and retry copy |
| Drop classification alert | 23 failures, including the real restart cases |
| Call before decision write | **Survived**, 78 passes. The independent snapshot probe rejects it. |
| Restore failed-send banner | 17 failures in reply and operator-copy retries |
| Follow both file and directory symlinks | 1 failure, the file symlink to shared status |
| Follow the receipt-directory symlink alone | **Survived**, 78 passes. The independent damaged-path probe rejects it. |

## 8. Other diff findings

No further production blocker established. The normal once-per-acknowledged-slate behavior and sequential recovery stay within the accepted limits.

## What I ran

- `git diff 386aa846 HEAD`, `git diff --check 386aa846 HEAD`, and the base and HEAD `src` archives. No tracked-file changes.
- Small offline probes using `.venv/bin/python`, `TZ=America/New_York`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH` for the revision. Retained `probe_*.py`, comparisons and logs in this directory.
- `prepare_mutations.py` produced separate source copies for each mutation and the one-line B1 fix.
- `run_test_matrix.py` started at 20:50 ET and checked `pgrep -f yield_run.sh` before each serial wrapper invocation. The web test uses its local loopback fixture. No external transport was used.
