## Verdict

**BLOCK — final round.** Reviewed detached checkout `85adb839a17579f2505ea3bdd83d17173dc67994`, changes `788c3a7` and `85adb83`, against the r1 report and the r2 prompt. Reconciliation is not crash-safe; submission failure can discard evidence of an executed job; failed-job retry protection still depends on the journal; reused unit identities can suppress accounting and release a new overrun with an old approval. These are failures within the stated model, independent of the disclosed scheduling and external-unit limits.

The permitted suite collected **91 tests: 89 passed, 2 failed**. Diagnostic reruns of the same two files, adding local-variable/verbosity output, reproduced both failures. Both receipts say `guard-error`, with reason `the job could not start: Exception occurred in preexec_fn.` The underlying child exception was not exposed. I do not infer that the Linux guard fails from these Mac failures.

All additional probes were read-only Python over synthetic inputs: filesystem changes, unit starts, signals and external systemd state were intercepted in memory. Their results establish the actual Python decisions, not measured Linux timing or a real systemd failure. T1–T5 and the launcher smoke are **supplied box observations**, not independently inspected here. No data read, outside-checkout search, network/SSH/gh operation, tracked-file edit, commit or push was performed in this round.

| Eric's requirement | Current disposition |
|---|---|
| CPU cap across children | Conditionally credible for ordinary descendants confined to the payload and timely monitoring. The old persistence-before-kill defect is removed. Normal-exit cleanup does not cover nested payload cgroups, and accounting/reuse defects remain (N1, N4, N5). |
| An overrun pauses all C1 | Not met on every path: N1, N2 and N4 can lose or bypass the required evidence/pause. |
| Two launches cannot race | Resolved for conforming launches on this account using the canonical root and shared flock. Real concurrent-launch behavior was not witnessed on the box; the local exclusion test passes. Unit-name reuse is a separate identity defect (N4). |
| Downloader cannot report success with unfinished requests | The verification snapshot is now protected. The CLI releases the lock before publishing/deciding success, leaving the exact r1 reporting boundary incomplete (N7). |
| Cached schedule is not read twice | Met in the inspected cached-schedule path; the original replacement-reader test still passes. |

The three-second scheduling assumption and prohibition on creating work in another unit are useful, explicit limitations. I do not re-block merely because those assumptions are unenforced. They qualify a monitored cap, not a kernel-enforced absolute guarantee. Their current wording does not cover work moving within the unit or descendants occupying nested payload cgroups (N5). `prepare` checks CPU readability and `exists`/`os.access` for the killer; it does not actually perform a preflight kill write. Describe that distinction accurately. The known realtime-order season limitation remains disclosed and unchanged; it is not a new regression or a new required season-policy change.

## Disposition of r1 changes 1–7

1. **Aggregate enforcement/termination (B1/B7): partially resolved.** `guard.py:85–107` supplies a positive minimum and delegated guard/payload split; `launch.py:288–307` validates the effective CPU seconds and supplies `Delegate=yes`. `guard.py:126–145,180–212` kills before receipt/logging work and classifies excess CPU/OOM. The old marker-write latency no longer leaves the payload deliberately running. Supplied T1 covers detached-session children, T2 guard death, T3 stop and T5 failed preparation. The local ordering/minimum/preparation tests pass. Remaining limitations are N5, the failed executable guard tests (N8), and the fact that normal-priority scheduling is a stated assumption. No new stronger scheduling guarantee is required just to repeat the disclosed limit.
2. **Completion/admission ordering (B2): partially resolved.** `_locked` checks activity before any sweep (`launch.py:372–379`) and again during planning (`:395–399`). The previous running-job-finishes-after-snapshot interleaving is closed for conforming launches. The 49-hour plus ended-job receipt test passes (`test_c1_cycle.py:364–374`). However, the receipt-to-ledger transaction can itself lose the completion at a crash (N1), so admission still lacks reliable closure.
3. **One cycle namespace (B3): resolved.** `main` has no data-root argument, uses `DATA_ROOT` for paths, and enters the common lock (`launch.py:347–365`); stop scans use the same root. Both alternate-root parser negatives and the held-flock refusal pass (`test_c1_cycle.py:431–443`). This is account-local canonical-root/exclusion evidence, not an independent Linux two-launch witness.
4. **Durable lifecycle/accounting and failure distinction (B4): partially resolved.** PENDING precedes submission (`launch.py:403–407`); missing TERMINAL reserves the budget and writes a pause; non-exit receipts do likewise (`:166–200`). `ledger.py:103–118` now fsyncs the rename directory. Missing-receipt and clean/non-exit fixtures pass; supplied T2 and smoke establish corresponding guard/normal-launch observations. N1–N4 defeat crash recovery, submission uncertainty, historical failure disposition and invocation binding. A declared-budget reservation is retained accounting for an unknown run, not a proved upper bound on CPU after monitor failure.
5. **Exact overrun release (B5): partially resolved.** Unit, full marker hash and an explicit structured RESUME ruling now replace the arbitrary-mention predicate (`launch.py:208–235`). Denial, wrong row/unit/hash and the existing wrong-source fixtures pass (`test_c1_cycle.py:405–428`). The source check still accepts another name beginning with Eric (N6), and marker binding does not protect against reuse of the unit identity and its existing marker (N4).
6. **Verification exclusion (B6): partially resolved at the requested reporting boundary.** Public `verify` takes the writer lock; `_verify_unlocked` runs inside it (`acquire.py:333–390`). Held-writer and acquisition-at-receipt-read tests pass (`test_c1_r3_acquire.py:406–434`), closing the old mid-scan mutation. The CLI prints and decides success after that lock is released (N7). Static unresolved/missing-schedule refusal and the single cached-schedule buffer are preserved.
7. **Documentation/evidence (r1 change 7): partially resolved.** The rewritten index distinguishes the superseded witness, current direct guard tests, real launcher smoke and unobserved Linux cases, and states scheduling/escape/season limits. That is substantially closer to the evidence. The claims of idempotent reconciliation and universal receipt/descendant handling need correction or implementation repairs below. The current local suite is not green (N8). The index's “covered only by unit tests” must not imply crash/collection/concurrency closure that those fixtures did not test. Deferred 4b, expected-feed inventory and promotion-link validation remain outside this review.

## New findings

### N1 — High: reconciliation archives its only recovery inputs before committing their accounting

**Location:** `scripts/audit/c1/launch.py:195–200,378–379`.

`reconcile` adds CPU/reservation rows only in memory, then moves PENDING and TERMINAL to `jobs/`. The caller writes the revised ledger afterward. If that write fails, or the launcher dies after either move, the next sweep scans only top-level PENDING files. It does not recover the archived receipts. A clean exit has no pause marker to prevent the next start. Neither archive directory is fsynced after the moves, creating additional power-loss/reappearance states; moving two files is not an atomic commit.

**Verified crash/retry probe:** a 49-hour ledger plus PENDING budget 14,400 seconds and a valid clean-exit TERMINAL for 7,200 CPU-seconds. The second ledger write was interrupted after the actual reconciler moved both receipts. Persisted accounting remained **49 hours**, top-level PENDING count became **0**, and both receipts were in `jobs/`. On retry, actual `_locked` returned **0**, submitted another start, and retained **49 hours**, despite the modeled total of **51**. The external journal was empty, the failure direction the redesign explicitly intends to support. This is a process-crash model, not a filesystem power-loss experiment.

**Smallest fix:** durably commit the accounting and required pause before consuming/moving recovery inputs. Make partial archive states recoverable and replay archived evidence when necessary; fsync both affected directories and the creation of `jobs/`. Test interruption before/after each move and ledger commit, including restart via `status`. Merely adding a directory fsync to the existing order does not fix lost accounting.

### N2 — High: `LoadState=not-found` after a failed submission does not prove the unit never ran

**Location:** `scripts/audit/c1/launch.py:408–412`; false-green fixture at `tests/scripts/test_c1_cycle.py:446–453`.

The failure handler deletes PENDING based solely on a later `systemctl show` result. The submitted unit uses `--collect`. A created unit may run, finish and be collected before that query; a client/submission error does not establish non-execution. Deleting PENDING makes an existing TERMINAL an orphan that reconciliation never processes. The deletion is also not directory-fsynced.

**Verified synthetic failure/retry:** intercepted submission wrote an overrun TERMINAL and raised `CalledProcessError`; the subsequent show returned `not-found`. Actual failure handling removed PENDING while leaving TERMINAL. A retry with an empty journal returned **0**, created no pause and retained zero CPU accounting for that run. The ordering is modeled; I did not witness that systemd/client timing on Linux. It disproves the handler's inference even when a terminal receipt is already present.

**Smallest fix:** retain PENDING on every uncertain submission failure, including `not-found`, and let normal missing/terminal receipt reconciliation resolve it. Only independently established non-submission can remove it. Reverse the existing fixture's expectation that `not-found` is sufficient proof.

### N3 — High: a receipted ordinary failure can be relaunched without acknowledgment when its journal is absent

**Location:** `scripts/audit/c1/launch.py:180–198,316–326,398–399`.

Reconciliation treats `result="exit"` as non-pausing regardless of rc, which is appropriate for a witnessed ordinary payload error. But `failed_units` is populated only from the current journal. The durable nonzero rc is archived and never enters the same-name failure gate. Thus the purported journal-loss repair preserves CPU but loses the earlier retry protection.

**Verified:** PENDING and a same-job TERMINAL `result="exit", rc=5, cpu_seconds=1`, empty journal, no `--ack-failure`. Actual `_locked` returned **0**, submitted a same-name start and created no pause. This is not a request to pause the entire cycle for ordinary errors.

**Smallest fix:** persist and consume failed-job disposition from current and archived terminal receipts, unioning it with journal failures before planning. A nonzero/negative payload rc must require the existing same-name acknowledgment even after collection, journal loss and repeated status calls. Other names may retain their current ordinary-failure behavior.

### N4 — High: unit names are reused as invocation identities; terminal data is not bound or validated

**Location:** `scripts/audit/c1/launch.py:176–188,298,403–405`; `scripts/audit/c1/ledger.py:54–84`.

The unit is still `c1-<name>-<UTC second>`. Two quick completed calls can reuse that name within one second; clock rollback can reuse an older name. PENDING/TERMINAL/archive paths and guard/reservation rows use only that name. `_journal_units` also suppresses a new guard row if **any** prior journal invocation has that unit name. `reconcile` leaves an existing overrun marker unchanged, so its old RESUME can release a different later overrun with the same name.

**Verified identity probes:**

- Two actual `plan_launch` calls with the same name/second produce the same unit. A prior journal row of 3,600 seconds makes `add_guard_row` ignore a new run's 7,200 seconds completely.
- With a prior released marker and prior 10-second journal row, a new same-named PENDING/TERMINAL overrun of **3,540 seconds** is reconciled. Accounting stays **10 seconds**, the marker remains byte-identical, and the old structured RESUME clears the stop: `overrun_stops == []`.

Terminal consumption also does not check `t.unit`, its budget against PENDING, the rc schema, or that an exit has a valid CPU measurement. Actual synthetic `_locked` admitted clean exits with a **different unit**, **boolean CPU** (counted as 1 second), and **null CPU** (budget reserved but no pause). These malformed-receipt tests establish fail-open reconciliation behavior; they are not assertions that the current guard normally emits those malformed values. Missing or contradictory measurement must not become a verified exit merely because the result string says exit.

**Smallest fix:** give each submission a fresh non-reusable invocation identity, carry it through PENDING, guard arguments/TERMINAL, archive, ledger and pause/release binding, and prevent overwriting evidence. Match journal supersession to that invocation. Validate exact receipt identity/budget, result/rc and finite nonnegative non-boolean CPU before trusting it. Invalid/incomplete exits must remain unreconciled and pause; a measured excess cannot be hidden by an exit label. Legacy ambiguous identities require an explicit unavailable/reconciliation disposition, not a guessed merge.

### N5 — High: normal-exit cleanup checks only direct payload members, missing nested descendants

**Location:** `scripts/audit/c1/guard.py:180–185,204–212`; index lines 24–26.

`payload/cgroup.procs` lists direct members, not members of child cgroups. After the job leader exits, a descendant in `payload/workers/` can remain while the direct list is empty. The guard then skips `cgroup.kill`, samples CPU and writes `exit` before that descendant stops. Even the kill path waits for the leader, not for confirmed emptiness of the recursive payload. Later systemd cleanup is not proof that the terminal CPU was sampled after all payload work stopped.

**Verified synthetic `guard.main`:** empty direct `payload/cgroup.procs`, descendant pid 999 in `payload/workers/cgroup.procs`, normal leader exit, readable unit CPU and writable killer. Actual main returned **0**, made **no kill write**, and recorded `exit` with `payload_left=[]`; the modeled descendant remained. No real cgroup or process was created. Supplied T4 proves detached-session cleanup in the direct payload leaf, not this nested case.

This stays inside the unit and therefore satisfies the currently stated “do not ask the manager for another unit” precondition. Moving work into a sibling leaf inside the delegated unit is another possible escape from the payload kill boundary; it also is not covered by that wording. That sibling case is reasoning-only, not a measured permission/escape test.

**Smallest fix:** detect recursive payload occupancy, kill every surviving descendant and establish emptiness before recording final CPU/exit; failed confirmation must remain an enforcement failure, not success. State the full confinement assumption: work must remain in the payload subtree, not elsewhere in the delegated unit. Do not claim the external-unit prohibition alone supplies that boundary. Add nested-descendant/failed-empty-confirmation tests; the scheduling bound must cover enforcement until payload work has stopped.

### N6 — Medium: the source predicate accepts “Erica” as Eric

**Location:** `scripts/audit/c1/launch.py:231–232`.

`row[-1].startswith("Eric")` admits another name such as `Erica` or `Ericson`. **Verified:** a correctly structured and hash-bound RESUME row with source `Erica 2026-10-06` returned `overrun_stops == []`. The existing wrong-source test uses only `Codex`, so it misses this prefix collision. The ruling text/unit/hash checks otherwise close the previous arbitrary-mention cases.

**Smallest fix:** require Eric as a complete source token, allowing the existing dated/relay source suffix, and add prefix-name negatives.

### N7 — Medium: the CLI still unlocks before reporting verification success

**Location:** `scripts/audit/c1_r3/acquire.py:337–338,408–418`.

The data scan is now a consistent locked snapshot, a real improvement. However, `verify` exits its lock before CLI printing and the success decision. The r1 required edit explicitly retained the lock through success reporting.

**Verified:** the synthetic writer-lock context releases after a clean receipt/artifact scan; a new acquisition appends intent `new-inflight` at that release. CLI main then prints `unresolved=[]` and returns **0**, while the current log has that unfinished intent. This is a new request after the scan, not the old mid-scan mutation. A historical snapshot claim would be accurate; the current unqualified “success is never reported while a request is unfinished” claim is stronger.

**Smallest fix:** for the CLI, perform the scan, success decision and flushed output in one writer-lock interval. Direct `verify` callers receive a completed snapshot; no claim of perpetual liveness after a return is required. Preserve the existing public locked wrapper.

### N8 — Medium: the permitted executable guard checks fail, and their fake cgroup is not a kill witness

**Location:** `tests/scripts/test_c1_cycle.py:244–263`.

Both the normal-exit and overrun `guard.main` tests fail with rc **87**, before the job executes, on the required Mac checkout. Full local-variable diagnostics expose the generic preexec error quoted in the verdict; they do not identify which child operation failed. The fake cgroup is a directory of regular files: writing `cgroup.kill` there cannot kill its real subprocess, and its `cpu.stat` is a planted constant. Those tests cannot independently establish real cgroup membership, aggregate CPU or termination, even when repaired to pass.

**Smallest fix:** identify the preexec failure and make the synthetic tests deterministic/portable without hiding a production error. Keep separate assertions for entering the payload and applying nice 10; do not replace the failed checks with unconditional success. Retain the supplied Linux observations as observations and add only the missing behavior tests needed for these changes. No broader suite or additional box operation is authorized by this review.

## Required changes

1. **N1 — crash-safe reconciliation: needs a re-check.** Commit ledger rows and required pauses durably before removing recovery inputs; recover every partial archive state and inspect retained archive evidence when needed. Fsync source/destination directories. Required probes: interruption before/after each receipt move and accounting commit, a failed ledger write, lost journal, and repeated run/status reconciliation. A simple fsync-only patch is insufficient. Existing archived receipts must not become invisible accounting evidence after a missing ledger.
2. **N2 — submission uncertainty: can be applied verbatim.** Replace the entire `except subprocess.CalledProcessError:` handler in `_locked` with:

   ```python
   except subprocess.CalledProcessError:
       raise SystemExit(
           f"systemd-run failed for {p['unit']}; launch state is unresolved, PENDING retained"
       ) from None
   ```

   Remove the LoadState-based deletion. Update the existing `not-found` test to require retained PENDING; add a preexisting-terminal/collected-unit case and verify normal reconciliation accounts/pauses it. These are required regression checks for the verbatim edit, not a claim that a new independent review round is automatically authorized.
3. **N3 — durable ordinary-failure gate: needs a re-check.** Derive same-name failed units from reconciled/current/archive TERMINAL `exit` receipts with nonzero or signaled rc, as well as journal failures. Preserve `--ack-failure` and different-name ordinary-error behavior. Test missing journal, repeated status, retry without acknowledgment, and acknowledgment of exactly that failed invocation. Do not turn every ordinary payload error into a global stop.
4. **N4 — invocation identity and receipt validation: needs a re-check.** Allocate a fresh submission identity, prevent evidence overwrite, propagate/bind it consistently through the lifecycle and ledger, and reject mismatched or unsupported receipt fields. Supersede fallback accounting only with the matching journal invocation. A new overrun must get a new marker/release identity even if a human job name or wall-clock second repeats. Test same-second repeats, clock rollback, prior released overrun reuse, prior journal-row suppression, wrong unit/budget, boolean/null/negative/nonfinite CPU, bad rc/result and contradictory excess-CPU exit. This is infrastructure identity work, not deferred 4b promotion-link validation.
5. **N5 — recursive termination boundary: needs a re-check.** Use recursive payload occupancy and confirmed post-kill emptiness before accepting the terminal measurement; failure to establish it must pause through the existing pending/terminal protocol. Cover nested cgroups and failure/delay in termination confirmation. Add the accurate payload-subtree confinement precondition to both guard documentation and the index; it must include moves within the unit. Scheduling/kill availability remains a disclosed conditional bound, not an enforced absolute guarantee.
6. **N6 — approving source token: can be applied verbatim.** In `overrun_stops`, replace:

   ```python
   and row[-1].startswith("Eric")):
   ```

   with:

   ```python
   and re.match(r"^Eric(?:$|\s)", row[-1])):
   ```

   Keep the other structured ruling/hash/unit checks. Add `Erica` and `Ericson` to the existing wrong-source cases; retain acceptance of `Eric` and `Eric 2026-10-06`/the established dated relay source. Run the permitted regression checks.
7. **N7 — verification reporting interval: can be applied verbatim.** Replace the complete `if args.verify:` block in `acquire.main` with:

   ```python
   if args.verify:
       try:
           with writer_lock(out):
               v = _verify_unlocked(out, args.feeds.expanduser().resolve())
               ok = not (
                   v["mismatched"] or v["missing"] or v["unreceipted"]
                   or v["responses"]["missing"]
                   or any(b is None for b in v["schedules"].values())
                   or v["unresolved"] or v["schedules_missing"]
               )
               print(json.dumps(
                   {k: (val[:20] if isinstance(val, list) else val) for k, val in v.items()}
                   | {f"n_{k}": len(val) for k, val in v.items() if isinstance(val, list)},
                   indent=1,
               ), flush=True)
               return 0 if ok else 1
       except Busy:
           print("refusing: an acquisition holds the writer lock; verify after it ends", file=sys.stderr)
           return 3
   ```

   Leave the public `verify` wrapper locked. Add a CLI output/decision-boundary test alongside the existing held-lock and mid-scan tests. This does not change inventory, promotion, outcome or request policy.
8. **N8 and documentation: test repair needs a re-check; the following evidence text can be applied verbatim.** Diagnose the actual preexec exception, repair the two failing synthetic checks, and run the permitted suite. Preserve their real assertions and the distinction between fake files and real cgroups. Replace the index's final sentence “Those are covered only by unit tests with systemd replaced.” with:

   > Local tests exercise selected decision branches with systemd or cgroup state replaced. They do not establish Linux crash recovery, collected-unit failure handling, concurrent-launch behavior or real OOM/RuntimeMaxSec enforcement. At review pin `85adb83`, the permitted Mac suite reports 89 passed and two guard preexec failures. T1–T5 and the launcher smoke are separately supplied box observations.

   Describe reconciliation as incomplete until N1–N4 are closed. Describe `prepare` as checking CPU readability and killer existence/write access, rather than claiming an actual preflight write. Keep the explicit scheduling, confinement and preexisting season-clock qualifications; do not claim terminal CPU includes the guard's subsequent receipt/logging overhead. No policy, cap, candidate scope or D7 change is requested.

The verbatim edits are not enough to turn this verdict into SIGN: N1, N3, N4, N5 and the executable-test repair require re-checking their resulting behavior. This is the **final authorized infrastructure round**, not a request or authorization for a third one. Under the pace rule the unresolved implementation remains unsigned unless Eric makes a separate decision. No repair, box execution, acquisition, commit, push, deployment or activation was performed or authorized by this verdict.
