## Verdict

**BLOCK.** Reviewed only the pinned executable checkout at `81f937e7044cbc6c72b4c483a715595aade34c41`, including `66e1b68` and `b959376` and the downloader's preceding receipt changes. The five requirements are not all satisfied on failure and concurrent paths.

| Requirement | Disposition |
|---|---|
| CPU cap across child processes | The cgroup reader counts contained children, but the hard bound is defeated by delayed polling, marker I/O, monitor failure and incomplete containment/termination (B1, B4, B7). |
| An overrun pauses all C1 | Not reliably: admission can miss a just-finished overrun; alternate roots, lost terminal evidence and invalid releases bypass the pause (B2–B5). |
| Two launches cannot race | The lock excludes launchers using the same directory. Different `--data-root` values select different locks and permit two starts (B3). |
| Downloader cannot report success with unfinished requests | A static unresolved intent fails verification; a request started during unlocked verification can still produce exit 0 (B6). |
| Cached schedule is not read twice | Met in the inspected cached-schedule path: binding, parsing and logged hash use the same `sched_bytes` buffer. The replacement-reader regression test passes. |

The permitted command `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_c1_cycle.py tests/scripts/test_c1_r3_acquire.py -p no:cacheprovider` passed **79 tests**. Additional checks ran read-only Python against synthetic inputs, with external I/O and process operations intercepted. Their results establish the Python decisions described below, not Linux scheduling, cgroup permissions or a measured production overrun. The prompt's 36-second/24.009-second box witness is supplied evidence of one successful guard trip; I did not independently inspect that unit.

No `data/` read, network/SSH/gh operation, tracked-file edit, commit or push was performed. Process deviation: a memory-registry search ran alongside the initial prompt read, before its checkout-only restriction was known. That external material was excluded from this review's findings and verdict; no subsequent outside-checkout source was consulted.

## Findings

### B1 — High: the claimed hard CPU bound assumes bounded scheduling and persistence latency; a marker failure skips termination

**Locations:** `scripts/audit/c1/guard.py:9`, `:67–84`, `:116–123`; `scripts/audit/c1/launch.py:233`; cycle index lines 18–19.

`budget - ncpu * poll` reserves only two nominal seconds. `sleep(2)` does not impose a maximum interval between CPU samples. The job continues running while the guard is descheduled or stopped, and while `on_overrun` performs directory creation, two fsyncs, rename and stderr output. These delays are not included in the bound. The guard and payload also share nice 10; the code supplies no independent deadline for the monitor or marker write. `LimitCPU` still limits each process separately.

**Verified synthetic counterexamples, using actual `watch`:**

- Budget 100 CPU-seconds, eight CPUs, first usage 70, and a nominal two-second sleep returning after five seconds of eight-child CPU growth: the marker and kill occur at **110 CPU-seconds**.
- Usage reaches the action threshold 84; a five-second durable-marker delay while eight CPUs remain busy makes the kill occur at **124 CPU-seconds**.
- `on_overrun` raising `OSError("synthetic full pause filesystem")` at usage 85 propagates without calling `kill` at all. A blocked write similarly leaves the payload running. Subsequent systemd cleanup is not the immediate aggregate-budget enforcement claimed by this function.
- Budget 10 with eight CPUs produces an overrun at **0.0 CPU-seconds**, because the action threshold is negative. Separately, `plan_launch(cpu_hours=1e-5, ...)` reports `ok=True` and emits `limit_cpu_seconds=0`; the guard then refuses that nominally accepted job. Finite positive hours do not establish an executable positive seconds budget.

These are modeled delays, not measured box latency. They nevertheless refute the unconditional bound: the allowed execution contains no constraint excluding those delays. Merely observing an early kill once, or subtracting another guessed polling interval, does not establish that bound.

**Smallest necessary correction:** validate the effective seconds budget before submission; remove the dependency of payload termination on successful marker I/O; and establish an enforcement bound that includes monitor scheduling and termination latency. Where that bound cannot be established, refuse admission rather than claiming a hard cap. A durable pre-launch pending record can preserve the cycle pause even if the guard cannot write its overrun marker. Stop the payload before potentially unbounded persistence work using a supervisor/termination arrangement that leaves the supervisor able to record the result. Keep the declared and remaining-cycle budgets as the actual limits.

### B2 — High: an existing job can finish between the sweep and idle check, admitting the next job with stale accounting and no pause

**Locations:** `scripts/audit/c1/launch.py:300–328`, especially the journal snapshot at 301, stop scan at 305 and active-unit query at 315.

The launch lock serializes launchers, not the running unit's completion. `_locked` snapshots the journal, writes the ledger and scans overrun markers **before** asking whether a unit is active. If the prior unit finishes between those operations and the active query, it disappears from the active set, but its CPU and failure were absent from the snapshot used for admission. A guard marker appearing after the earlier marker scan has the same ordering problem. The next unit can run before the timeout/overrun is acknowledged, and before the 50/100-hour gate includes the prior job.

**Verified synthetic interleaving through actual `_locked`, `sweep`, parser, stop checks and planner:** the persisted ledger contains 49 CPU-hours; the initial journal snapshot is empty while the existing job is running. At the active-unit query, that job stops after two CPU-hours and its timeout and CPU records enter the synthetic journal. The query returns no active units. `_locked` returns **0**, submits one new start, writes no overrun marker and retains **49** ledger hours, although the modeled actual total is **51**. No alternate root, second launcher or journal vacuum is needed.

**Smallest fix:** check for an existing active/activating/deactivating job before the final sweep and admission, refusing that launch if one exists. Then reconcile every previously admitted invocation's terminal accounting and limit result before starting another job. A disappearing unit with no reconciled completion must be pending, not treated as absent/accounted. The same-root lock alone does not provide this closure.

### B3 — High: `--data-root` changes the cycle's lock and stop namespace while still launching real C1 units

**Locations:** `scripts/audit/c1/launch.py:56–60`, `:180–189`, `:278`, `:290–294`, `:303–324`; cycle index lines 25–31.

Two different directories select two different `.launch.lock` files, ledgers, acknowledgments, CPU-overrun scans and 403/429/unresolved-receipt scans. Both launchers still submit `c1-*` units to the same user manager. An alternate-root launch does not inspect an existing production guard marker or the production acquisition stop/receipts. Historical CPU already retained only in the production ledger is also lost from a fresh root once the corresponding journal records expire.

The active-unit query does not repair different-lock concurrency: both launchers can observe no unit before either submits its start. **Verified synthetic main-path interleaving:** with exclusion keyed by the directory actually selected by `main`, A pauses immediately before submitting its start, B uses another root and submits its start, then A submits its already-approved start. Both `main` calls return **0**, and the intercepted start list contains two different `c1-*` units. This uses modeled external state, not two real Linux launches. The supplied scratch-root witness itself confirms that a scratch guard marker was intentionally invisible to the real C1 root even though its CPU was included in the real ledger.

**Smallest fix:** make live launch/status accounting and all C1 stop gates use one canonical cycle root and one lock regardless of artifact destination. Reject a different `--data-root` for live submissions, or remove that live override. Keep scratch testing isolated through injected test dependencies rather than an option that launches real C1 work in an independent pause namespace. Apply this to `status` as well as `run`.

### B4 — High: guard failure and unswept journal loss can erase a cycle-level stop and CPU consumption

**Locations:** `scripts/audit/c1/guard.py:98–123`; `scripts/audit/c1/launch.py:45`, `:118–134`, `:250–272`, `:300–305`; `scripts/audit/c1/ledger.py:66–88`.

No durable admitted-but-unfinished record is created before the job starts. `read_usage`, `sleep`, marker writing and kill operations can fail without a guard completion. SIGTERM forwards to one process group and exits without checking/persisting final aggregate usage; SIGKILL cannot run any handler. A resulting `signal` or guard `exit-code` failure is not classified as a cycle pause. That is unsafe after the cap monitor disappears: an ordinary payload error and a failed enforcement mechanism are different conditions.

**Verified synthetic decision:** a C1 journal result `signal` causes zero marker writes; the planner admits a different candidate (`ok=True`). The existing timeout test at `tests/scripts/test_c1_cycle.py:278–292` explicitly expects a `signal` result to produce no marker. A guard killed first or a per-process CPU-limit signal therefore has no general fallback pause unless a guard marker happened to survive already.

Timeout/OOM markers are durable only **after a later launch/status sweep sees the journal entry**. If the unit is collected and those entries are vacuumed or lost before that sweep, `_journal()` can succeed with no relevant records and the next launch cannot distinguish the lost overrun from no job. Its CPU can disappear too. **Verified synthetic decision using actual sweep/parser/gates:** an existing 49-hour ledger plus a modeled unswept two-hour timeout whose journal is now empty admits another job, retains 49 hours and creates no pause. This is an evidence-loss scenario, not a claim that the real journal was vacuumed.

There is also a persistence gap in `ledger.write_tsv`: it fsyncs the temporary file but does not fsync the containing directory after `os.replace`. Its new “durable” description is therefore not established for a crash during replacement. This is a code-inspection finding; no power-loss experiment was performed. Missing ledger state is currently interpreted as an empty cycle.

**Smallest fix:** persist an invocation-bound launch/pending record before submission and require a durable terminal receipt containing aggregate CPU and the enforcement result. Materialize limit pauses at termination, not only on the next journal sweep. If completion/accounting is unavailable, keep all C1 paused until reconciled; retain conservative budget accounting rather than dropping the job. Fsync the ledger directory after replacement. An independently witnessed ordinary **payload** exit-code failure can continue to block only the same name; guard failure, unclassified signals and unknown termination cannot inherit that exception. Preserve the timeout/OOM global-stop rule.

### B5 — High: a RULED row mentioning the unit is not necessarily an authorization to resume it

**Locations:** `scripts/audit/c1/launch.py:145–175`; `tests/scripts/test_c1_cycle.py:274–306`.

The release validator checks `"**RULED" in row` and `unit in row`, while trusting the release file's self-declared `approved_by="Eric"`. It does not require the ruling to authorize release, verify the row's approving source, or match the unit as an exact identity. It accepts a refusal, a historical unrelated decision that names the unit, or a unit-name prefix contained in another unit's identity.

**Verified synthetic refusal case:** a release with the correct unit, reason, `approved_by="Eric"` and `register_row="deny"` passes against `| deny | <exact unit> | **RULED 2026-10-06: DO NOT RESUME** | Eric |`: `overrun_stops` returns **[]**.

**Verified with actual checkout register text and a synthetic marker/release:** `C1-4b-generator-commit` also clears a synthetic overrun for `c1-r4b-gencheck-20261005T163415Z`. That row records generator reproduction, not permission to resume after an overrun. This is a release-validator counterexample concerning the common infrastructure; it neither reopens nor reviews the deferred generator work.

**Smallest fix:** bind the release to an explicit recorded Eric decision whose action is to resume this exact overrun/unit. Use an unambiguous structured action and exact unit identity in the ruling, not arbitrary Markdown substring matches. Bind it to the invocation/overrun identity so a same-second name reuse or clock rollback cannot replay a prior release. Require the approving source in that recorded decision; the JSON field alone is insufficient. Add negative tests for explicit denial, unrelated ruling naming the exact unit, wrong approving source, prefix collision and stale release. The current valid fixture must express the actual release action.

### B6 — High: `--verify` can return success while a concurrent request remains unfinished

**Locations:** `scripts/audit/c1_r3/acquire.py:333–382`, `:400–406`; writer lock at `:119–130`, acquisition locking at `:273` and `:416`; tests at `tests/scripts/test_c1_r3_acquire.py:363–381`.

Verification takes no writer lock. It reads receipts once at line 342, then scans/hashes artifacts and decides using that old receipt list. A downloader can append a new durable intent and enter a request after that snapshot but before verification reports success. The new `unresolved` field cannot see it. The launcher makes conforming C1 units sequential, but `--verify` itself has no such admission check, and its public entry point permits this concurrent read.

**Verified read-only synthetic `main --verify` counterexample:** `read_receipts` returns a closed empty snapshot. During the subsequent feed-directory scan, the in-memory receipt log receives intent `inflight` for game 101, with no completion. Actual verification main returns **0** while `unresolved_intents(current_log)` is **["inflight"]**. No request or filesystem write was made by the probe.

**Smallest fix:** hold the same writer lock throughout receipt acquisition, artifact reconciliation and success reporting, refusing verification when an acquisition owns it. Ensure the callable `verify` path is protected too, with a separate internal helper if needed to avoid nested locking. Test an actual held acquisition lock and the append-after-snapshot interleaving. Keep the newly correct static unresolved-intent and receipt-named missing-schedule checks.

### B7 — High: process-group fallback does not establish whole-cgroup termination, and escape is not prevented

**Locations:** `scripts/audit/c1/guard.py:101–114`; `scripts/audit/c1/launch.py:235–242`.

The guard preflights CPU readability but not its ability to terminate the cgroup. If `cgroup.kill` exists but its write fails, the exception escapes before the process-group fallback. If it is absent, `killpg(proc.pid)` targets only that process group: a child that creates a new session remains in the unit's cgroup but is outside that group. It can keep consuming CPU until some later unit-wide cleanup. The configured argv provides no explicit immediate fallback deadline that closes the aggregate-budget claim during that interval.

**Verified synthetic `guard.main` decisions, with no actual process or signal:** an absent killer produces a marker and only `killpg(101, SIGKILL)`; an existing but unwritable killer produces a marker and raises `PermissionError`, with **no killpg call**. Neither proves all descendants stopped. The supplied successful Linux `cgroup.kill` witness does not test these failure cases.

**Reasoning-only containment case:** the reader covers only processes remaining in the selected unit's cgroup. If the payload can ask the user manager to start work in another unit, or move work to another permitted cgroup, that work is outside the observed total and kill scope; a non-`c1-*` unit is also outside the ledger and admission census. The launcher accepts arbitrary commands and its argv does not establish that escape is denied. I did not attempt an escape or verify the box's D-Bus/cgroup permissions. The universal child-process claim therefore needs a demonstrated containment precondition rather than an assumption about cooperative children.

**Smallest fix:** establish whole-cgroup termination capability before starting the payload; refuse a missing/unusable killer unless an independently tested equivalent covers every descendant and the termination bound. Do not substitute a single process group for that capability. Establish and test the allowed payload's confinement, including new sessions and external-unit attempts; refuse execution paths whose CPU leaves the accounted unit. Keep an unreconciled-stop pause if termination cannot be confirmed.

### Evidence and test strength; remaining guarantee checks

The 79 green tests are useful but narrower than their names and the index's claims:

- `test_the_guard_kills_the_whole_job_before_its_cumulative_cpu_reaches_the_budget` supplies a fake reader whose increments already assume the bounded poll interval and a `kill` callback that just appends an event. It tests comparison/order under that model; it does not create children, kill anything or establish the real interval/persistence bound. The exit-code and exit-overrun tests exercise actual branches, but likewise have no live cgroup.
- The cgroup-reader test parses a synthetic path and temporary `cpu.stat`; it establishes format handling, not real cgroup membership or permission. The marker test checks contents and absence of temporary files, not fsync behavior under failure. The unique-temporary-file test establishes unique rename sources, not post-crash ledger durability.
- The lock test holds the same file's flock and observes refusal, then one mocked submission. It is a meaningful exclusion check for that inode, not two independent Linux launches, different roots, or a prior job completing during admission.
- The timeout and release tests establish their supplied journal/row cases. They omit lost terminal evidence and explicit denial; the signal exclusion expectation entrenches B4. The unresolved-intent test establishes a preexisting unfinished request's refusal, not writer exclusion during verification.
- The schedule replacement-reader test is nonvacuous for the addressed defect: it would offer game 202 on a second read, but records one schedule read and game 101 passed to acquisition. Acquisition is intercepted, so it proves buffer reuse rather than any network/feed-inventory claim. The missing receipted schedule test also proves the intended static refusal.

For conforming sequential launches with intact evidence, the argv retains nice 10, `MemoryMax=12G`, `OOMScoreAdjust=1000`, the declared `RuntimeMaxSec` and per-process `LimitCPU`. The ordinary job-exit path propagates its exit code. The 403/429/no-further-request protocol, unresolved-receipt scan, and cached-schedule receipt refusal remain present; their cycle-wide admission strength is limited by B2/B3/B4, not by removal of those checks. The status path uses the lock and reports stop lists, but performs the same root-selected, journal-dependent sweep; its `gate` field is the ledger gate, not a complete launch-readiness decision.

The season guard is unchanged by these commits; the suite establishes its ordered synthetic sleep/non-sleep cases and March boundary. An additional synthetic clock-rollback check exposed an existing limitation in `launch.py:99`: it selects the maximum realtime timestamp rather than the last journal event. An old 24-hour idle line at 14:00 followed in journal order by an active lineup-check line after a rollback to 13:00 yields `season_guard(..., max_hours=1) == True`. This is a preexisting season-guard defect, **not a regression attributed to the two fixes**. The pinned code therefore does not justify an unconditional clock-independent season guarantee. Preserve fail-closed behavior and use event order/current-boot clock evidence when addressing it; no change to season policy is requested here.

Expected-feed inventory and promotion-link identity/hash validation remain outside this ruling's downloader repair scope. This report does not make them new blockers or reopen other deferred rank-4b findings. No activation or D7 authority follows from this review.

## Required changes

1. **B1/B7 — enforce the actual aggregate limit and terminate reliably.** Preflight the effective positive seconds budget, monitoring and whole-cgroup kill capabilities, and confinement before submission. Reject budgets below the enforceable margin rather than spawning and declaring a zero-usage overrun. Close delayed-monitor, blocked/failed-marker and failed-kill paths; kill/freeze the payload independently of marker persistence and preserve a pending pause if the supervisor fails. Establish the CPU-growth/termination bound for the hard-cap claim; an ideal two-second reader test is insufficient. Preserve wall, memory, priority and per-process backstops. Require Linux failure-direction evidence for detached-session descendants, monitor suspension/death and missing/unwritable cgroup termination; the supplied scratch witness covers only the successful path.
2. **B2 — close completion versus admission ordering.** Check for an existing job before the final accounting/stop sweep and refuse that launch when one exists. Admit only after every prior invocation has reconciled terminal CPU and stop evidence. Add the 49-hour plus newly completed two-hour timeout interleaving, and a guard marker appearing after the old scan; both must refuse the next start.
3. **B3 — use one cycle namespace.** Remove/reject alternate roots for live C1 submissions, or make the canonical lock, ledger, checkpoint acknowledgment and all stop scans independent of output-root selection. Apply the same accounting semantics to status. Test simultaneous different-root/different-name calls and alternate-root calls with a production CPU-overrun marker, rate-limit stop and unresolved receipt. None may bypass the common gate.
4. **B4 — retain launch/terminal evidence durably.** Create the invocation-bound pending record before launch, reserve/account its budget conservatively, and clear it only through a durable reconciled terminal receipt. Record timeout/OOM/enforcement failures independently of a later journal sweep. Missing completion or CPU evidence pauses C1 even after `--collect` and journal loss. Distinguish ordinary payload exit failure from supervisor/enforcement failure; keep the former's existing same-name retry policy. Fsync the ledger's containing directory after rename. Test lost journal records, guard signal/error, failed marker persistence and missing ledger/terminal evidence.
5. **B5 — authorize this exact release action.** Require an explicit recorded Eric ruling to resume the exact invocation/overrun, with an exact identity and approving source; reject denial, mere mention, prefix match and stale approval. Replace the substring predicate and expand the release negatives accordingly. No existing reproduction/scope ruling supplies an overrun release.
6. **B6 — serialize verification with acquisition.** Acquire the existing writer lock before reading any receipts or artifacts and retain it through success reporting; refuse a busy writer. Protect direct callers as well as the CLI. Add held-lock and concurrent-intent tests. Retain the single verified schedule buffer, static unresolved check, missing-schedule check and existing receipt/403/429 protocol.
7. **Documentation and review evidence.** Correct the guard/index's unconditional polling-bound and durability claims until the corresponding enforcement and persistence evidence exists. Describe what the real scratch witness and each synthetic test establish. Preserve the existing caps, frozen candidate scope and separate D7 gate. The preexisting realtime-order season limitation must remain explicit unless it is corrected with journal-order/current-boot evidence; it must not be represented as newly caused or fixed by this change.

These are the complete material issues identified in this first infrastructure round. **BLOCK is not repair, execution, live-acquisition, commit, push, deployment or activation authority.**
