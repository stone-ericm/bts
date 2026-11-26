# CS4C adversarial review

## Verdict

**BLOCK** for `9fc2ee6d` against `1c6cbc77`. Stale-reading code can prevent a pick send or interrupt the return after it. Both outward branches can also warn without a confirmation identifier.

## Blocking findings

1. **The new calculation and warning can interrupt delivery.** With a covered 20-hour-old reading, raise `KeyboardInterrupt` or `SystemExit` from the calculation after the final cutoff check. The base sends once, locks, records the decision, calls capture, and returns `True`. The branch raises before the attempt save, with zero sends, lock, decision, or capture. Inject the same errors in the warning dispatcher and the branch raises after saving the same markers, lock and decision as the base. A warning-only stderr interrupt also skips dispatch. Both modes reproduced this behavior with `GeneratorExit` and a custom `BaseException`. The dispatcher probe also covered a `BaseExceptionGroup` containing an interrupt and a diagnostic whose `__str__` raises. Use `BaseException` for the protected calculation, warning, warning stderr, and diagnostic operations at `src/bts/scheduler.py:1178`, `:108`, `:92`, and `:66`, respectively. Keep those handlers scoped to the new work. [Fault replay](delivery-comparison.log)
2. **An unconfirmed send emits a warning that says the pick was delivered.** With a stale reading, a captured success response body of `{"id":null}` for DM or `{"uri":null}` for public reaches the scheduler as `None` through the real adapters. An empty string behaves the same. Both branches retain the base's historical `True`, lock and decision, and then print and send the new warning. The persisted DM fails `pick_was_delivered`. Public passes that predicate on its legacy posted flag alone. At `src/bts/scheduler.py:1216` and `:1246`, require a nonempty string identifier from the corresponding transport before invoking the warning. Preserve the base's attempt, markers, lock, decision, and return. [Adapter replay](probe-notifications.log) [Base comparison](delivery-comparison.log)

## Notes

- A synthetic builder return with a string age writes successfully, produces the bounded failure diagnostic, and reloads with no record. A return containing a raw Python object reaches the attempt save, raises `TypeError`, and preempts the send. The real builder produces serializable primitives and the loaders cannot construct that object. [Fault replay](delivery-comparison.log)
- The combined run needed a separate invocation outside the sandbox because the sandbox refused the web saver's loopback bind. The offline guard allowed literal loopback and blocked outward connections and external DNS. Both clock runs and the archived-base run were rerun with it. [Verification](verification-approved.log)
- The reader inventory was available in the tracked CS5a handover under `docs/reviews/2026-11-14-season-rollover/build/`. I reviewed its listed typed, projected, raw, and hash readers. The dashboard and representative wire, export, ledger, receipt, and experiment paths ran offline. The other direct readers were inspected in source. [Reader inventory](../../docs/reviews/2026-11-14-season-rollover/build/handover-cs5a.md#pick-file-readers)

## Is delivery unchanged

All 630 matrix cells match the base on pick body and count, return, attempted and delivered markers, lock and capture invocation, scheduler state, decision bytes, existing stderr, existing alerts and health status bytes, and saved pick bytes with the trailing field removed. The matrix crossed seven readings, DM/public/private, required/optional, three marker states and five selection cases. Among 40 qualifying attempts, eight each stamped dated, fresh, old, future, and unreadable readings. The last three groups emitted 24 warnings. [Matrix](delivery-comparison.log)

The 64 fault cells include ordinary sends that raise, null and empty identifiers, initial void and late guards, final cutoff, missing recipient, a transport crash, malformed records, closed stderr, and warning failures. Ordinary calculation or dispatcher exceptions, an emission clock `RuntimeError`, and read-only or directory-shaped status paths preserve the send, durable markers, lock, capture, decision, and return. Closing stderr after the existing lock line still permits dispatch. Closing it on entry reproduces the base's existing `ValueError` path. The blocking paths above explain the changed send or return. [Fault replay](delivery-comparison.log)

## The stamp

In DM and public, the qualifying path samples the final cutoff clock, computes the record from the existing delivery-state object, and stamps immediately before the attempt marker and save. In each mode I replaced the automatic file immediately after its one real read and advanced the clock across 14 hours between guard samples. The stamp used the original record and the final sample, producing `old` at rounded `14.0`. Dated delivery clears a seeded reading to null. Spies confirmed no compute on cutoff, early guard, private, optional, state-read failure, 5a refusal, and DM-without-recipient paths. [Snapshot replay](probe-daily.log) [Delivery tests](combined.log)

At send entry the attempted file already carries the sample. On an ordinary failed send the marker clears and the sample survives to the next qualifying attempt. Before-attempt crash preserves prior bytes. In-transport crash preserves the sample and yields the existing locked-unconfirmed recovery. A crash after the confirmed save preserves the stamp and yields delivered recovery. Neither recovery sends a warning or re-sends the pick. The saved-pick fixture with `selection=None` replaces an earlier fresh record with the stale delivery sample. [Delivery tests](combined.log)

## The function, the validator and the file

- The independent oracle matched 5,127 inputs, including the four real kinds, unrecognized kind, both strict thresholds at ±1 µs, UTC offsets and `Z`, DST, bounded malformed strings, non-string inputs, UTC overflow in the reading, and valid year-1/year-9999 caller instants. Spies saw no I/O or wall-clock calls and all produced records validated. Outside the caller precondition, naive `now` uses local time and a UTC-unrepresentable `now` can raise `OverflowError`. [Wire probe](probe-wire.log)
- All three loaders agreed with the oracle for 9,677 decoded shapes, including missing/extra keys, type substitutions, inconsistent combinations, nonfinite numbers, range and rounding limits, and both display-boundary ambiguities. All accepted records had canonical key order. Canonical pre-5a and 5a inputs re-saved with exactly their respective null tails. Shadow and normal saves place `automatic_reading` last. Archive metadata follows it. [Wire probe](probe-wire.log) [Archive tests](combined.log)
- Real provenance, attempt, delivered, posted, grading, reconcile, archive, shadow, preview, run, and orchestrate saves retained valid values. Both live-forward readers equated absent and null. Adding, removing, or changing a non-null value produced drift, and all tested rewrites had distinct literal hashes. Historical and normalized slot projections matched. Receipt, ledger, experiment and raw readers continued to account for the literal file. [File tests](combined.log) [Wire probe](probe-wire.log)

## The warning

The exact transport assertion uses `BTS health WARN attention:\n- [contest_state] ` followed by the message below. The matching stderr line starts with two spaces and `STALE CONTEST READING — `. [Copy tests](combined.log)

> The pick was delivered using an automatic contest reading flagged stale (<detail>). Check MLB. If the streak or saver differs, use bts set-contest-streak --date 2027-07-15 --streak N --best-streak B --saver S with the scheduler's picks directory and the values shown in MLB. Use the current ET date when you run it. Check the latest fetch status.

- Old used `last accepted fetch 2027-07-14T22:35:00-04:00, about 20.00 hours ago`. Future used `its recorded_at 2027-07-15T23:00:00Z is in the future`. Unreadable used `it has no usable recorded_at`.
- A DM pick and a separate public pick on July 15 yielded one warning DM. A third pick on July 16 yielded a second. The saved key was `WARN_ATTENTION:contest_state:stale_reading` in `picks_dir.parent / "health_state" / "contest_stale_reading_dm_delivery_status.json"`. A send spanning midnight used July 16 in the text and dedup, while age came from the pre-send sample. A failed warning retried on the next eligible delivery without a resurface banner. [Notification replay](probe-notifications.log) [Warning tests](combined.log)
- The existing health status bytes survived every success, failure, and no-recipient warning case with same-day critical success and failure seeded separately. The next critical stayed deduplicated after a prior success and resurfaced after a prior failure, consistent with the unchanged dispatcher. Public with no recipient stamps and prints only, with no dispatch or warning status file. DM without one follows the existing refusal. Legacy commands, recovery, and same-pick re-entry emitted no warning. The null/empty identifier path above is a false-positive warning. [Warning tests](combined.log)

## Through run_day

The real scheduler picked Steven Kwan, sent one pick DM, locked, and wrote the delivered decision with a covered 20-hour reading, a fresh 10-hour reading, and a dated observation. The first warned once, the fresh case stamped a non-stale record, and the dated case stamped null. In a separate replay a 13h55m reading at the 18:25 choice aged to 14h05m at the 18:35 fallback, where a no-pick refresh reused the saved pick with `selection=None` and sent one warning. Each restart preserved the stamp and sent neither the pick nor a warning. [Scheduler replay](probe-daily.log)

## Tests and false greens

- Archived base source and scripts yielded **87 failed, 312 passed, 4 skipped**. Of the passes, 259 were pre-existing assertions in the migrated files. The 53 new GREEN controls were 32 missing/invalid value cases in `test_all_pick_loaders`, 16 existing guard cases, both final-cutoff no-compute cases, the DM refusal in the recipient test, and the two no-recipient critical-isolation cases. On the base those cases either have no qualifying feature work or already treat the field as absent. [Archived-base results](red-base.log)
- The four exact-encoding cases, the pre-5a fixture rewrite and both archive-order cases failed on base for the added key or projection. The diff preserved the existing byte-prefix, value, and drift assertions and strengthened the trailing-key and legacy-fixture assertions. No loosened assertion surfaced. The other 80 failures were in the two new test files. [Archived-base results](red-base.log)
- The suite caught 17 of 19 scratch mutants. It caught omitting each stamp, moving each stamp before its cutoff, warning before the send or after a failed send, sharing health status, dropping the incident key, both 14-hour mutations, moving the future threshold, accepting an integer age, stamping other kinds, escaping an ordinary warning exception, stamping private or optional calls, and warning from delivered recovery. [Mutation results](mutations.log)
- **False greens.** Omitting `now_et_date` passed all 133 new tests because the dispatcher and scheduler share the fixture clock. The independent test skewed only the dispatcher and caught an extra same-day warning. Accepting a bool age passed the validator cases because their `True` is paired with `reason="old"` and fails the 14-hour bound anyway. A usable reading with `age_hours=True`, `stale=False`, and `reason=None` exposed it through the loaders. [Supplemental mutation evidence](evidence-probe.log)

## Rows 1 to 8

1. **Holds.** The trailing timestamp copies literally for dated, automatic and legacy manual. Model defaults to null. Real loader tests and 66 base comparisons covered switch, manual, uncovered, revoked and corrupt inputs.
2. **Holds.** Independent producer and decoded-validator oracles agree over boundary, malformed, and generated inputs.
3. **Holds.** Three loaders, exact historical tail bytes, save key order and real preservation paths passed.
4. **Fails.** Final-cutoff snapshot and stamp placement pass. Calculation interrupts prevent delivery in both modes.
5. **Fails.** Null and empty confirmation identifiers warn. Warning-only interrupts can escape after durable confirmation.
6. **Fails in the calculation-interrupt case.** No-compute guards, retries, recovery, legacy commands and both Amendment 1 recipient paths otherwise match.
7. **Holds.** Both live-forward projections and literal hash probes agree on null and non-null cases.
8. **Holds.** Source and AST diff preserve all named functions, modules and existing decision logic. The loader diff is the literal timestamp copy, confirmed by the 66-cell replay.

## What I ran

- Archived `src` and the projection scripts from `1c6cbc77`. Ran `delivery_probe.py replay` and `edge_probe.py` in wire, daily, and notifications modes. Ran `evidence_probe.py` for the source audit and surviving mutants. Captured the schedule, predictions, sleeps, pick and alert transports. [Delivery log](delivery-comparison.log) [Source and mutation evidence](evidence-probe.log)
- Wrapper runs used `review_harness.py verification` and `review_harness.py mutations` with Python 3.12, ET, and pytest cache disabled. Pinned July 15 runs in 2027 and 2028 each yielded **399 passed, 4 skipped**. The four touched files plus LightGBM and web saver checks together yielded **412 passed, 4 skipped**. [Verification](verification-approved.log) [Mutations](mutations.log)
- `git diff --unified=0 1c6cbc77 HEAD -- src scripts`, `git diff --check`, and tracked status passed the scope check. Review probes, logs and this report are in the scratch directory.
