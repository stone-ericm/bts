# CS6b-1 adversarial review, round 2

## Verdict

**SIGN** for `7efc3cea` against `1bf518a2`. All three round-one blockers are closed.

## Blocking findings

None.

## Notes

- The dated acceptance test still populates `strategy._mdp_cache` from `None` and leaves the dict installed after teardown. This reproduces the round-one note. The requested mixed suite passed in that process. [Cache observation](combined-detail.log) · [Test](../../tests/test_morning_state_cs6b1.py#L112) · [Cache](../../src/bts/strategy.py#L69)
- The lead’s malformed-record test discards every exception after the run, so it establishes that the advisory sent but leaves the first-check handoff unverified. The independent file probes verified that handoff. Existing lineup readers can still raise on malformed records, as amendment 1 records. [Test](../../tests/test_morning_state_cs6b1_review.py#L83) · [File probes](production-review.log)
- The unchanged dispatcher still raises on a status file containing `[]` or `null`. Both fixtures reached the first prediction on time and left the bad status bytes in place, so the morning DM stays suppressed until the file is corrected or removed. The fix did not worsen the round-one status or cache notes. [Replay](remaining.log) · [Status reader](../../src/bts/health/alert.py#L149)

## B1

- Real SIGINT propagated in all 17 replays, including the decision, pick and state reads, dispatcher, actual stderr writes for refusal, stale, dated acceptance, automatic acceptance, due and empty plans, the failure diagnostic, and status writes. The stalled transport was interrupted by a second thread using `os.kill`. The base and branch both exited with `KeyboardInterrupt` before any check. Non-advisory files matched byte for byte apart from the heartbeat phase. The branch was still `running` in startup while the base was `sleeping` for the first target. [Replay](production-review.log) · [Base snapshot](faults-base.json) · [Branch snapshot](faults-head.json)
- The dispatcher writes a `sent` status after the transport returns. A cancelled or failed transport left no sent status and restarted with a send. An interrupt before or during persistence of a successful send allowed a repeat. A fully written sent status suppressed the repeat after the transport had already sent. No replay suppressed a send on restart without a preceding send. [Status order](../../src/bts/health/alert.py#L234) · [Restart replay](production-review.log)
- Eleven ordinary exception kinds at 19 sites gave 209 checks of isolation, including an ordinary `ExceptionGroup`, a broken `__str__`, a real stderr write failure, the sampled clock, `min`, the required gate and the age calculation. Every fault was hit. First-prediction timing and pre-check bytes matched the base after accounting for advisory status. The empty-plan variant had zero checks in both versions. [Replay](production-review.log) · [Guards](../../src/bts/scheduler.py#L115) · [Call site](../../src/bts/scheduler.py#L2940)

## B2

- All 34 malformed-file inputs across the two slots sent the exact morning CRITICAL, performed one state read and reached the first-check handoff without a morning failure diagnostic. The files covered list and object actions, string scoreability, 20,000 and 100,000 nested arrays, invalid UTF-8, a directory, a symlink loop, empty files, 50 MB of ASCII and binary zeros, and each JSON scalar type. [File probes](production-review.log) · [Read recoveries](../../src/bts/scheduler.py#L144)
- Readable scoreable and delivered records each suppressed the state read. A captured read failure of either valid file let the advisory proceed. A CRITICAL used its own status, with no pick or posting effect. With stale state, a genuine decision’s captured `MemoryError` triggered the WARN and persisted the shared 4c key. This suppressed an independently invoked same-day 4c DM while preserving its journal line. That dedup is the contract’s existing WARN behavior. [Commit controls](production-review.log) · [4c replay](remaining.log)

## B3

- The real missing-switch refusal and a 1,040-character multiline non-ASCII refusal each appeared once in stderr, as the expected single 800-character escaped excerpt. Captured root logs contained the transport failure and no refusal or chained exception context. Round one reproduced the chain in both cases. Stock failure log messages, failed status and the exact restart banner matched round one. [Replay](parity.log) · [Round-one capture](messages-round1.json) · [Branch capture](messages-head.json) · [Refusal handler](../../src/bts/scheduler.py#L159)

## Nothing else moved

- The production diff touches only the three morning exception handlers, two record-read recoveries and the refusal block. The existing loop and delivery functions are byte-identical in source.
- Ten paired real `run_day` replays covered dated, fresh automatic, stale automatic, refused with and without a cached preview, MDP skip, no games, a thin slate at 9m59s lead and two same-day restarts. The pre-existing file bytes, checks, heartbeat, polling handoff and existing sends matched, including the skip DM and real EOD runs in the refusal cases. The contract accounts for the morning journal and CRITICAL status and for moving the 4c WARN to startup. [Day comparison](parity.log) · [Remaining cases](remaining.log)
- The dated, fresh, stale and refused output also matched round one. Both dispatches still pass the sampled `now.date()`. CRITICAL uses `CRITICAL:contest_state:morning` in `contest_morning_check_dm_delivery_status.json`. WARN uses `WARN_ATTENTION:contest_state:stale_reading` in `contest_stale_reading_dm_delivery_status.json`. Acceptance writes the same line and sends no DM. [Round-one parity](parity.log) · [Dispatch tests](../../tests/test_morning_state_cs6b1_review.py#L130)

## Tests

- Archived `0e0c49a3` yielded **31 failed, 64 passed**. The failures were eight author interrupt cases, fourteen lead cases and nine Carbon cases. The explicit dispatch-date and accepted-state read-count checks already passed there. The branch’s three review files passed all 95 cases under each simulated Python wall clock in July 2027 and July 2028. [Red and wall-clock runs](pytest-matrix.log)
- The author’s single edit changes eight collected cases, including the two required-gate cases beyond the six described in the brief. The six read, dispatcher and transport cases and the two gate cases all change from containment to propagation. Comparing Carbon against the original adopt file shows the same amendment-only change to stderr and exotic interrupt expectations, with the ordinary unprintable case retained. No unrelated assertion was removed. [Author cases](../../tests/test_morning_state_cs6b1.py#L359) · [Carbon cases](../../tests/test_morning_state_cs6b1_carbon.py#L142)
- The tests killed all nine scratch mutants. Failure and advisory stderr widening and outer widening produced 2, 6 and 23 failures. Narrowing decision and pick recovery produced 3 and 2. Moving dispatch into the handler produced 1. Removing the CRITICAL and WARN date arguments produced 2 and 3. Reading accepted state twice produced 4. [Mutation runs](pytest-matrix.log)
- A separate scratch mutant sent the wrong interrupt or an unexpected interrupt out of the morning call. Pytest completed all 95 items with 11 failures and 84 passes. The requested eight-file mixed process passed **285 tests with four skips**. [Interrupt failure handling](pytest-interrupt.log) · [Mixed run](pytest-combined.log)

## What I ran

- Inspected the source and test diffs, ran `git diff --check`, and used `git archive` for the base, round one and scratch mutations. Tracked files were untouched.
- Ran the direct production probes with captured schedule, predictions, posting and DM transports and pinned persisted clocks. The comparison harness needed a fixed Matplotlib cache path and a corrected assertion for the pre-existing skip DM. The successful replays are in [fault and file probes](production-review.log), [message and day parity](parity.log) and [remaining parity and status edges](remaining.log).
- Ran all pytest families serially through `~/bts-runs/yield_run.sh`. The Python guard permitted numeric loopback for the web fixture, supplied its localhost name locally and denied DNS and outward connections. No external service, box or contest account was used. The full fast suite was left to the gate.
