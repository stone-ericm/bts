# CS4c adversarial review, round 2

## Verdict

**SIGN** for `687f9d55`. Both round-1 blockers are closed under amendment 2.

## Blocking findings

None.

## Notes

- **The identifier type check has a test gap.** Replacing either guard with `required and identifier` passes all 63 cases in the two added files. A captured 2xx body with `id: 19` or `uri: 19` reaches the corresponding guard as a truthy integer. The shipped code suppresses the warning. Add a truthy non-string case to each mode to catch this regression. [Mutants](controls.log) · [HTTP replay](delivery-verified.log)
- **The Carbon interrupt test can terminate pytest on a regression.** Its injected `KeyboardInterrupt` stopped the archived run before the final fallback control. A review-only pytest hook converted that one escaping interrupt into a failed assertion, letting all cases run. The current tests complete together and restore their patched bindings. [Original run](mutations.log) · [Completed archive run](controls.log)
- **Round-1 notes.** A null or empty DM identifier still reaches the existing locked-unconfirmed recovery on re-entry, with a critical alert and zero pick re-sends. Public takes already-delivered recovery. The revised tests exercise both. The stamp's aware-clock precondition and the synthetic unserializable-record case have the same code paths as round 1. All five warning-date and validator mutants cited by Carbon are now caught. [Second review](round1-carbon.md) · [First review](round1-codex.md) · [Controls](controls.log)

## Finding 1

- **Closed in both modes.** Ran 84 cells from a covered 20-hour-old automatic reading. Each of `KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, a custom `BaseException`, a `BaseExceptionGroup` containing `KeyboardInterrupt`, and an exception whose `__str__` raises went through each of seven sites. Those sites were the calculation, dispatcher, transport inside the real dispatcher, warning stderr, failure diagnostic print, warning clock, and status-file write. Every cell matched the unfaulted `1c6cbc77` send count, `True`, markers, lock, scheduler state, decision bytes, and capture call. A failed calculation stamped null. Failed warning stderr still allowed the DM and status write. [Replay](delivery-verified.log) · [Fault snapshots](delivery-branch.json)
- **Handler scope.** In `src/bts/scheduler.py`, lines 66–69 cover diagnostic rendering and printing, 74–111 cover the stale-warning body and its dispatch, 92–95 isolate the warning stderr line, and 1178–1182 guard the calculation and assignment. The two send blocks and `_record_commit` retain their `Exception` handlers. In 16 paired cases, `KeyboardInterrupt` escaped the pick transport, both pick saves, `save_state`, `_record_commit`'s callee, capture trigger, cutoff calculation, and final cutoff clock, with the same side effects as the base. [Replay](delivery-verified.log)
- **Real SIGINT.** `signal.raise_signal(SIGINT)` in the warning DM transport returned `True` in both modes with the pick committed and the warning DM status absent. The same signal at the archived base's `return True`, just after its lock line, raised `KeyboardInterrupt` with identical durable state. The operator's single interrupt during warning transport is consumed and the caller can continue through the locked-pick path. Amendment 2 explicitly requires this return, and the next scheduling step sees the same lock and committed decision. [Replay](delivery-verified.log)

## Finding 2

- **Closed in both modes.** Captured transports returning `None`, `""`, `0`, `False`, an ordinary string, and a string of spaces. Also ran the real adapters with captured HTTP for null, empty, missing, ordinary string, integer, Boolean, and spaces. Every case preserved base behavior and stamped the file before the send. Null and empty kept the legacy `True` and lock. A missing JSON key raised `KeyError` in the adapter and used the legacy `False` path with the attempt marker cleared. The stderr line, warning DM, and dedicated status file appeared exactly for non-empty strings, including spaces. [Replay](delivery-verified.log)
- `bts.dm.send_dm` (`src/bts/dm.py:88`) and `bts.posting.post_to_bluesky` (`src/bts/posting.py:163`) index the decoded JSON directly. Their annotations allow `str`, and their implementations can pass through malformed JSON values. Every well-formed confirmation identifier passes the guard. A malformed 2xx may still represent a remote send, but it supplies no confirmation under amendment 2. [HTTP replay](delivery-verified.log)

## Nothing else moved

- `git diff 9fc2ee6d HEAD -- src` changes only `scheduler.py`. The changes are four wider handlers, the diagnostic's matching annotation, two warning eligibility conditions, and comments.
- The real delivery replay matched `1c6cbc77` in all **432** cells across the six readings, three modes, required/optional, three markers, and four selection paths. It saw 68 outward pick sends and 20 qualifying attempts, with 16 non-null automatic records and 12 stale warnings. The saved pick bytes matched after removing the added key. The attempt and delivered markers, returns, send payloads, lock, scheduler and decision bytes, capture calls, existing stderr and critical alerts matched. For eight refusal cases, I first verified the recorded health DM hash against its original body, then normalized the scratch path in both the body and hash. Public without a DM recipient stamped and printed once. DM without a recipient refused. [Replay](delivery-verified.log)

## Tests

- On archived `9fc2ee6d`, all **43** fix-specific cases failed for the intended behavior and **20** controls passed. All **63** passed at HEAD as part of the combined run. The archived source path was verified inside each pytest process. [Archived controls](controls.log) · [HEAD](combined-verified.log)
- All eight requested mutants were caught. Reverting handlers at lines 68, 94, 110, and 1180 produced 8 failures each, with a further Carbon interrupt at line 110. Dropping each full identifier check produced 5 failures per mode. Replacing it with `is not None` produced 3 per mode. Retaining only the type check was also caught in both modes. [Requested mutants](mutations.log) · [Additional controls](controls.log)
- A direct diff against Carbon's original adopted test file has exactly two hunks. The first changes the expected stale warning count from one to zero. The second changes the expected escaping interrupt to a `True` return. Both keep the surrounding persistence and recovery assertions. The lead test patches `scheduler.print` with `monkeypatch`, and Carbon scopes `sys.stderr` to a context. Neither file installs a process hook. The combined runner verified restoration of builtin `print`, stderr, and tracing.
- Each combined run covered all six requested change-set files plus `test_lightgbm_optional.py` and `test_web_saver.py` in one pytest process. The real clock, **2027-07-15**, and **2028-07-15** each yielded **475 passed, 4 skipped**. Both altered runs verified `datetime.now()` and `time.time()` before collection. [Combined runs](combined-verified.log)

## What I ran

- Archived `1c6cbc77 src` for the delivery replay and `9fc2ee6d src` for archived pytest. Ran `delivery_probe.py` and `delivery_verify.py` with captured pick DM, public post, warning DM, and schedule. The pytest fixtures also captured schedule and prediction calls for `run_day`. [Replay scripts](delivery_probe.py) · [Verifier](delivery_verify.py)
- Ran pytest through `~/bts-runs/yield_run.sh`, sequentially. The sandbox denied the web saver's loopback bind. The first approved run reached the HTTP server's reverse lookup, which the offline guard blocked. The final approved run used a test-only numeric loopback name shim and exercised the real loopback HTTP test. The review guard blocked DNS and non-loopback sockets in pytest and its Python children. [Combined verification](combined-verified.log) · [Mutation verification](mutations.log) · [Controls](controls.log)
- Checked the source diff, Carbon's original assertions, and the tracked worktree. The review artifacts are confined to `.codex-review/cs4c-r2/`.
