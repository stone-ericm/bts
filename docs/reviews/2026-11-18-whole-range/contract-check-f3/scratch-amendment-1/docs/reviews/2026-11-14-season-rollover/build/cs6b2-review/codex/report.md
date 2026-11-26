# Verdict

**BLOCK** `38d707bd` against `59efa825`. Replacing the status path with a FIFO between the type check and open can stop the scheduler on both the read and write paths.

## Blocking findings

1. **Read can block before either gate.** At `src/bts/health/alert.py:174–176`, `stat()` checks one object and `read_text()` opens whatever is at the path afterward. I started with a regular `{}` file, let the real `stat()` return its mode, and atomically replaced it with a FIFO with no writer. The real `read_text()` hit a one-second alarm in all three modes: CRITICAL, empty list, and no recipient. No DM went out and the call did not return. An unwrapped scheduler caller therefore never reaches its next step. Keep the precheck for stable special paths and use a nonblocking open followed by `fstat()` on the opened descriptor before reading. Treat a changed type as an unreadable snapshot. [Race results](dispatcher.log#L38)
2. **Persistence can block after the DM.** At `src/bts/health/alert.py:101–112`, the same replacement after the write-side `stat()` reaches the blocking `write_text()` at line 87. The CRITICAL probe captured one successful send and then timed out with no `sent` record or persistence warning. The no-recipient probe also timed out. The empty-body control returned `False`. Open for writing with `O_NONBLOCK` and validate the opened descriptor before truncation. Handle a nonregular descriptor or failed open with the existing persistence warning. Keep the existing payload, creation mode, and regular and dangling symlink behavior. [Race results](dispatcher.log#L41)

The replacement is expressly in the brief's matrix and requires neither resource exhaustion nor a stalled filesystem. The filesystem and `os.replace()` both completed normally. The wait was inside the FIFO open. The handover's assumption that the path remains stable adds a condition absent from v2.

## Notes

- The authored FIFO probes cover stable FIFOs. Add a bounded replacement case at each open boundary to cover the two stalls.
- The authored parity cases check outcome, DM args, and bytes for 36 cases. The wider matrix below also checks log messages, day rollover, omitted path and date, and empty-body and recipient ordering.
- Some intentional base-green controls lack a GREEN label. The counts below identify them. The first independent caller harness produced ten fixture failures because scratch paths changed DM hashes and the slate writer used wall time. The rerun used one pathname per pair and pinned the slate clock. [Initial run](caller-branch.log) · [Corrected run](caller-branch-verified.log)

## Conforming files

Archived `59efa825 src`. Generated prior bytes using the real base dispatcher for sent with one key, several keys, WARN-only, banner-only, failed with and without CRITICALs, and skipped. Added legacy without dedup fields and missing. Compared **2,016** pairs across seven alert mixes, old/new keys, same/next ET day, explicit/implicit ET date, present/omitted path, recipient present/absent, and transport success/failure.

All pairs matched in banner, suppression, captured DM recipient/body, boolean, log level and message, exception type for transport failures, and exact persisted bytes. The clock was pinned at 23:59 ET, after UTC midnight. Empty body exits before dedup or persistence. Coverage suppresses before the recipient gate. A qualifying banner can supply the entire body and follows the same recipient, send, and write sequence. For a conforming snapshot, `failed` can supply the banner and `sent` can supply coverage. A stable file cannot do both. [Dispatcher matrix](dispatcher.log#L1)

## Malformed files and special paths

- The three modes matched missing-file send decisions for all 28 malformed or ignored-field cases. They included both nesting depths, invalid syntax and UTF-8, empty and 50 MB files, wrong field types, invalid counts, duplicate keys, and a list of 100,000 strings. Writable cases recorded `sent` following an attempt with successful transport, or `skipped_no_recipient` when appropriate. [Content results](dispatcher.log#L3)
- Row 2 honored Python's last duplicate key and gave the exact text for positive integer failures, including the accepted 4,300-digit count. A 5,000-digit token fell back to missing under the pinned digit cap. Zero, negative, null, string, float, and `true` counts produced no banner. The positive-count banner alone still sent. [Count results](dispatcher.log#L25)
- Row 3 suppressed a covering mixed list and sent for a noncovering list. It wrote the sorted union, including a prior escaped surrogate string and deduplicated keys. Stale and non-`sent` snapshots contributed zero keys. [Dedup results](dispatcher.log#L31)
- All 11 stable path cases finished in each mode. These included a FIFO with and without an open writer, a bound Unix socket, `/dev/null` through a symlink, a directory, a looping symlink, a non-directory parent, and an actual `PermissionError` on read after mode `000`. Attempted writes to those unwritable or special paths emitted one persistence warning. A regular symlink, a dangling symlink with a creatable target, and a missing parent gave the same bytes and logs as the base. The dangling target was created and the link preserved. The replacement cases are the two blockers above. [Path results](dispatcher.log#L38)

## The callers

**26 cases on each source** with `[]` and 20,000-deep JSON, each paired with the missing-file control. On the branch all 24 dispatching cases matched the control in health and pick DMs, status bytes, return or idle continuation, schedule and prediction calls, in-memory state, and persisted pick, scheduler, and decision records. The two classified-unconfirmed controls reached recovery with zero dispatches and kept the initial status. [Branch replay](caller-branch-verified.log)

| Real path | Archived base with malformed status |
|---|---|
| `run_single_check` refusal and `_deliver_and_lock_pick` state-read refusal | Raised `AttributeError` for `[]` and `RecursionError` for nesting. Zero health DMs. |
| Saved-pick refusal | Contained the error, returned false, and sent no health DM. |
| `run_day` in-loop and final fallback refusals | Independently observed calls at `scheduler.py:3193` and `:3346`. The final fixture skips the in-loop handler. Each exception stopped the day. |
| Missed-pick, late, void, and direct unconfirmed | Raised in both content variants before sending the health DM. |
| EOD runner with real CRITICALs, startup morning check, and confirmed delivery's stale warning | Contained the error and sent no DM for the affected lane. Morning and stale used their own files. The stale fixture still sent the pick and persisted the delivered decision. |

The recorded base exceptions and contained failures left the malformed bytes in place. [Base replay](caller-base-verified.log)

## Tests and false greens

Against archived base source the two authored files and migrated test gave **141 failed, 219 passed**. All 219 passes were controls or already-contained behavior. Named GREEN cases account for 119. The exact node IDs are in the log. [Base suite](base-authored.log)

| Base passes | Why |
|---|---|
| 96 status shapes | Named syntax, UTF-8, injected read denial, directory, regular symlink and missing cases, plus dangling and looping symlinks. |
| 18 failed counts | Positive integer banner, zero and negative controls. |
| 54 prior-source cases | All other-day and other-status cases, plus same-day null and empty sources. |
| 3 persistence cases | Directory failure was already logged and did not prevent repeat attempts. |
| 36 parity cases | Named conforming GREEN controls. |
| 12 caller cases | Eleven named missing-status controls and the malformed classified-unconfirmed case that never dispatches. |

All **13 scratch mutations** failed their targeted tests. The shape tests caught removal of the dict guard, read/write regular-file guards, recursion handling, and the send. Count tests caught bool and zero acceptance. Dedup tests caught truthiness, non-string retention, other status, other day, and lost union. The open counter caught the second read. Both FIFO mutants failed via `pytest.fail` in about two seconds and the wrapper continued. [Mutation summary](pytest-review.log#L9)

Only `test_refusal_survives_non_object_health_status` was changed among existing tests. It now expects one health DM, `sent` and no alert-failed line. It retains refusal, original pick bytes, unlocked state, no decision, and no pick DM.

Both orders with the morning and saved-pick suites finished with **515 passed, 4 skipped**. Per-test snapshots confirmed restoration of `_mdp_cache` identity, CWD, recursion and digit limits, and alarm handler/timer state. The combined process with the optional-LightGBM, loopback web, morning, saved-pick and existing health tests gave **551 passed, 4 skipped**. Native July 15 wall-clock runs for 2027 and 2028 each gave **454 passed, 4 skipped**, including the migrated test. [Suite summary](pytest-review.log#L3)

## Rows 1 to 5

1. **Does not hold** for a replaced path. The stable-input and one-open probes passed. The bounded FIFO swap blocked the read.
2. **Holds.** Positive exact-integer, malformed-count, and banner-only probes.
3. **Holds.** Mixed covering and noncovering lists, same-day check, stale and non-`sent` cases, and exact written union.
4. **Does not hold** for persistence through a replaced path. The 2,016 conforming comparisons passed and stable special paths warn. FIFO replacement stalled the write.
5. **Holds.** `git diff 59efa825 HEAD -- src` names only `health/alert.py`. Archived function bytes match for every preexisting definition except the changed dispatcher and its banner helper, including `dispatch_dm_for_critical` and the writer.

## What I ran

- `dispatcher_probe.py` with archived base source, pinned clocks and bounded actual file operations. Unix socket binding required the unsandboxed rerun. [Log](dispatcher.log)
- `run_pytest_review.py` and `run_caller_verified.py` serially under `~/bts-runs/yield_run.sh`, using `.venv/bin/python`, `TZ=America/New_York`, no bytecode and no pytest cache. Kept the native clock shim beside its source. Captured DM, posting, schedule and prediction transports. The Python socket guard rejects DNS and outward connections, and allows numeric loopback and the HTTP server's local lookup. [Suites](pytest-review.log) · [Caller rerun](caller-verified.log)
- `git diff 59efa825 HEAD --check` and a tracked working-tree diff. Review files are under `.codex-review/cs6b2-review/`. The review exercised the targeted suites only.
