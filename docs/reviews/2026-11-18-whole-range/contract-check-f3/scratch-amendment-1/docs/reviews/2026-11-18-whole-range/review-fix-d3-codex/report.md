# D3 review, round 1

## Verdict

**BLOCK** `d3d05abd` against `24019a9a`, using kickoff v2 and both amendments. The failure path can leave a restarted scheduler running even when both SSH scripts reach the box. Two verification paths can overstate their evidence, and the activation checklist can put the scheduler into live mode before the checked deploy.

## Blocking findings

1. **Both handlers can skip every containment action after one missed clock sample each.** After a successful reset and restart, HEAD verification returned 17. The first `clock` in deploy containment returned 124 once, then worked. Matching recovery took the lock and read `begun`, and its first containment clock also returned 124 once. Neither issued stop or POST or replaced the `running` heartbeat. The record stayed `begun`. `deploy.yml:444–450,478–510,1094–1100,1130–1160` sets the deadline to zero and every later `run` refuses to launch. Give an authorized handler a bounded fallback when the first time sample fails, with a reserved budget for stop, marker and POST. A third invocation with a working clock and the same record contained the box. [Probe](containment_probe.py) · [Four transcripts](containment-clock.jsonl)
2. **A phase-2 signal can print a false preflight line and skip local containment.** TERM during the scheduler query after reset and restart, with the phase budget exhausted before the trap's record read, printed `DEPLOY NOT STARTED` and claimed the checkout and services were unchanged. The candidate HEAD and `begun` remained, and no stop was issued. A one-off read status 126 reproduced the same outcome without an exhausted budget. `deploy.yml:528–541` should use a fresh, bounded adjudication read and retain the successful begin decision in memory. After a confirmed begin, an indeterminate commit must take containment. Matching recovery contained both cases when invoked. [Evidence](shell-evidence/late-signal.json) · [Read error](shell-evidence/record-read-126.json) · [Recovery](shell-evidence/late-signal-recovery.json)
3. **A blank `ControlGroup=` can certify a populated scheduler subtree.** The stop shim returned inactive/PID 0, local link disabled, no matching jobs and `pgrep` 1. The service subtree's `cgroup.events` remained `populated 1`. With its path in the manager reply the handler printed `unverified`. With a blank property it skipped the check and printed `Scheduler stopped: verified.` `deploy.yml:247–262,429–436,485` and its recovery copy accept an empty property as proof of absence. Carry the pre-stop group path into verification, and use `unverified` when absence cannot be established. This demonstrates the verifier's behavior with contradictory observations under shims. [Blank property](shell-evidence/empty-control-group.json) · [Named group](shell-evidence/known-control-group.json)
4. **A page to an expanded URL can verify the literal endpoint.** For `.env` containing `https://checks.invalid/{other}`, the handler passed that URL plus `/fail` to curl. Its default glob parser targeted `/other/fail`. With HTTP 200 there, the handler logged `verified`. `{one,two}` modeled two requests in one invocation. `deploy.yml:231–246,498–505,1148–1155` needs `--globoff` in both POST calls, or validation rejecting raw glob syntax in URL paths. Native curl expanded these exact inputs with all transport protocols disabled. [Single request](shell-evidence/post-single-glob.json) · [Two requests](shell-evidence/post-multiple-glob.json) · [Offline curl parser](curl-disallowed-protocol.log)
5. **Commit publication loses the failure status.** With the stamp already installed, either record staging or rename returned 17 or 124 and did not publish `committed`. All four cases logged `Check commit-record failed (1).` `deploy.yml:363–377,716–717` replaces the failing status with its read-back classification. Preserve the numeric status, including `timeout`, when exact read-back cannot confirm publication. Keep accepting a matching `committed` record after a failed rename return. The four cases contained and left the newly renamed stamp whole. [17 and 124 transcripts](shell-publication.log)
6. **A5 puts activation edits before B8a's safety checks.** A5 directs DM/enter edits and cron install before directing the reader to B8a. Start after a research deployment with the boot link enabled, both flags set and today's eligible dated reading on disk. The A5 edit alone changes the real scheduler startup and morning gate from private to accepted DM. A boot then runs live before the operator reaches stop, boot-disable, state checks or checked deploy. Put B8a's prerequisites first and place A5's config and cron commands explicitly within that sequence. [Checklist](../../docs/ops/2027-season-start.md) · [Startup trace, A5 rows](canary-pinned.log)

## Notes

- The shell suite's timeout shim returns an injected code or immediately dispatches its child. Its `flock -w` stand-in probes the kernel lock nonblocking. These exercise branch handling and contention. A7 still needs the Debian status pairs and remote signal, wait and release evidence.
- The stamp transcript test reconstructs the candidate from a diff computed from that same candidate, then checks a fixed subsequence. The saved transcripts are evidence. The reconstruction does not independently qualify the complete order.
- The unchanged dashboard-only quick advice in `INCIDENT.md:109–110` can be read as permission for an immediate deploy, despite the incident checks at lines 99–102. The pre-existing filesystem-restore path at lines 111–125 also ends in manual enable/start. Both are outside the edited instructions.
- B8a's command examples use the CLI default `data/picks`, while its new instruction names the scheduler's configured directory. A custom directory needs `--picks-dir` on status, revoke, enable, fetch and dated observations. The top-level directory instruction applies to those examples.

## 1. Phase 2, signals and closed output

- Blocking findings 1–2 show two distinct gaps. The deferred-TERM probe let reset finish before the single stop. An enable shim installed the boot link and then returned 124. Containment disabled the link, stopped the unit, wrote the failure marker and preserved candidate HEAD. There was no reset to the old SHA. [Publication and deferred signal](shell-publication.log)
- The actual 30-second shim sleep completed in 30.010 seconds. A shim sleeping 44 seconds hit the 33-second TERM deadline and logged `Check wait failed (timeout).` before containment. The timing adapter delegated only the shimmed sleep to the local timeout executable. [Healthy wait](wait-real.log) · [Timeout](wait-timeout.log)
- Closing the stdout pipe at the reset barrier produced an empty log. The healthy control still committed. A reset that mutated HEAD and returned 17 still contained. [Closed output on success](closed-success.log) · [Closed output on failure](closed-failure.log)
- The focused run covered HUP, INT, TERM and PIPE across the phase command matrix, begin/commit boundaries, the second signal during containment, ordinary failures and injected timeouts. The new clock and commit-status counterexamples exercise inputs beyond its green matrix. [Focused tests](focused.log)

## 2. Verification lines

- Scheduler and POST have the false-verified cases in findings 3–4. Known populated cgroup, redirect and nonzero transport are negative controls. The same common helpers are present in both literals.
- Replacing the marker payload before its read-back gave `Failure heartbeat written: unverified.` Recovery replaced a surviving writer's `running` heartbeat with a fresh failure marker for the recorded step and retained the matching prior HTTP result without a second POST. L6 covers writes after verification. [Marker comparison](shell-evidence/marker-changed-before-verification.json) · [Recovery](attempt-refresh.log)

## 3. Attempt record, mutex and cancellation

- Recovery-first fenced a late SSH invocation. A new attempt could deploy. Older or different IDs and matching `committed` changed no service. Clearing temporary `/run` gave the no-record branch. A held kernel lock refused both scripts, and the lock's inode and existing bytes survived. [Ordering](attempt-order.log)
- Both scripts refused directory, FIFO, dangling and existing symlink, and oversized record/mark cases. Extra record controls covered embedded NUL and impossible state/result combinations. Wrong-mode mark refused. A target-mismatch, duplicate invocation and publication during owner death are in the focused suite. [Records](attempt-records.log) · [Marks](attempt-marks.log)

## 4. Success path and stamp

- Ran the exact base and candidate YAML for absent and present legacy units. The full argv and output are saved with a unified diff for each case. Base fetched, reset, synced and restarted before checking HEAD and printing an early `Deployed`. Candidate adds preflight, `begun`, soft disable and its check, then reset, fixed-venv sync and conditional per-unit legacy disable/stop/unlink, reload and `pkill`, restart, full HEAD, bounded wait, four ordered checks, stage/fsync, enable/check, atomic stamp and directory fsync, then checked `committed` and PASS. The old stamp survives failures before rename. F4 Amendment 1 places the stamp after `begun` and before `committed`, and permits the complete new stamp on a later failure. [Absent delta](shell-evidence/absent-transcript.diff) · [Legacy delta](shell-evidence/legacy-transcript.diff)

## 5. Deploy canary

- Ran 78 paired cases against the real CLI startup gate and real morning read, with the scheduler day body replaced by a call to the morning read on the same files and pinned instant. Both live modes exercised every switch refusal class, stale/fresh automatic, effective dated override, either flag, neither flag, ET midnight and year rollover. Private and the invalid table/delivery cases followed the shared gate. Six distinct config errors returned class-only output. File and directory snapshots matched, and prediction, MLB and transport entry points were barred. [Paired run](canary-pinned.log)
- A shell shim emitting an embedded NUL inside the F9 prefix still got PASS because Bash command substitution removed the NUL before validation. The raw path is at `deploy.yml:680–693`. Validate the prefix before returning text into Bash. [Transcript](shell-evidence/contest-nul-inside-prefix.json)

## 6. Heartbeat checker

- The base and candidate produced identical output bytes, exit, outgoing ping method/URL/timeout and churn-history bytes across 32 ordinary cases under the pinned clock. Freshness thresholds, serialized CRLF, missing/invalid JSON, symlink-to-regular and crash-loop override were included. Four `deploy_failed` cases emitted the bounded reason first, used `/fail` and skipped churn. FIFO, directory, over-limit, bad UTF-8, non-object and naive time cases completed with the fixed malformed reason. Python 3.11.18 and 3.12.16 agreed. An optional Unix socket bind was denied in the sandbox. [3.11](heartbeat-311-nosocket.log) · [3.12](heartbeat-venv.log)

## 7. Operator sentences, in source order by group

| Source and sentence | Worst state and literal operator action |
|---|---|
| Workflow `Pre-deploy SHA` and `Checking requested target` | Candidate reset can fail partially. Record the two SHAs for triage and wait for the result. Neither line authorizes a start. |
| `Waiting 30s` and clean exit awaiting relaunch | The scheduler can send during the wait (L5). Wait for the result. The relaunch sentence depends on the checked tuple. |
| F9 private skip and live accepted | Private skips the state read. Accepted is a single ET read. Check actual delivery separately. The binary-output case in §5 is a shell acceptance gap. |
| F9 refused with problem and error with class | Refusal directs today's dated correction within the safe window. Error calls for diagnosis. |
| `Check … failed` and `DEPLOY NOT STARTED` | Read the first reason and both step logs. After a phase deadline the preflight claim can be false (finding 2). Before publication a prior failed run may still own the reached box. |
| `DEPLOY FAILED`, no automatic rollback, recovery unverified and procedure | A partial tree stays in place. Read §4 and verify the reached state before touching picks or services. |
| Stop verified/unverified | A residual cgroup defeats the stop evidence in finding 3. The operator checks both scopes. |
| Heartbeat verified/unverified | The marker report reflects a read-back. Re-check if a writer survived (L6). |
| HTTP verified/unverified | A wrong endpoint defeats the response evidence in finding 4. Check provider history and actual receipt separately. |
| Recovery unreadable, lock unavailable, invalid inputs or mark unavailable | State and ownership are uncertain. Follow §4 and check the scopes by SSH before retrying. |
| No record on this boot and different attempt owns record | Check boot time and the appropriate run when `/run` is gone. Inspect the newer owner when the record identifies another attempt. |
| Committed and containment re-checked | Read the deploy log for a committed attempt. A re-check reports fresh stop/marker evidence and retained or new HTTP evidence. |
| `Deployed` and `CANARY PASSED` | Publication gates PASS in the normal path. Verify the pick and MLB entry separately. |
| POST body and stale reason | Both direct §4 even with odd heartbeat fields. Preserve delivery evidence and avoid any resend. |
| `INCIDENT.md` §1 deploy/SSH and check/reset/start/wait sequence | A failed checked deployment can reach its new tree and leave a scheduler alive. Inspect Actions. |
| §1 containment, recovery eligibility and verification | After a lost action, read both steps and follow §4. Eligibility tells the operator where to look. |
| §1 PASS, independent delivery/entry, no manual restart, branch and SHA | A passing check can coexist with an uncertain send. Inspect the separate records and choose the tested `deploy` SHA for a checked deployment. |
| §2 approved ship/revert, dispatch branch and default gate | An urgent fix still needs approval. Choosing `main` fails the branch policy. Read both logs if a step failed. |
| §2 dispatch uses current SHA | A moved ref changes the target at dispatch. §5 supplies the same-SHA rule. |
| §4 read logs and preflight claim, recovery separately, commit after log loss | Finding 2 can mislead about the phase. A matching commit can survive lost logs. Check recovery and box evidence independently. |
| §4 absent/unverified/cancelled/conflicting escalation and prior/candidate still running | At the worst loss boundary, contact Eric before retry and have him check both scopes and the boot link. |
| §4 preserve files and check ET, original delivery and MLB entry read-only | A successful send with a lost acknowledgement still requires preserving its pick and receipt evidence. Read the original delivery and MLB before any retry. |
| §4 saver/entry controls, preview, resend, uncertainty, dashboard and window | Leave saver and entry alone. Resolve uncertainty with Eric while the scheduler is stopped, including for a dashboard fault. The unchanged dashboard shortcut is in Notes. |
| §5 tested normal ship, emergency ship/merge and revert | A partial checkout goes to §4. An approved fix-forward or current revert uses the gate and preserves branch history. |
| §5 older SHA and 2027 support | An October rollback risks overwriting live state. Rehearse before any approved older revision and use a 2027-capable candidate. |
| §5 refusal, configured directory and live decision | A dated read can coexist with a refused automatic route. Read today's live decision from the scheduler's files. |
| §5 approved remaining window, same SHA, changed ET day, direct start and missed window | Verify delivery and entry before a checked retry. Recheck if the day changed. If the window closed, record the missed day and check stop. |
| §5 grading/backfill and moved branch | Preserve evidence before chronological backfill with Eric. A moved `deploy` ref can ship a new SHA. |
| §6 GitHub notice, immediate POST, Cron and missed pings | Failed SSH may yield no page or stop. Check independent provider history, integrations and two check receipts. |
| §6 cron/timer and DM inventory | Scoring, reconcile and fetch can write to the reached tree. Backup and timers continue. Listed DMs are outside pick delivery. |
| `CLAUDE.md` deployment, checked shipping, dispatch scope and soft hold | A bad deploy stays on the reached tree. Use §4 and the complete approved workflow. |
| `CLAUDE.md` L1, L3, L4, L6 and L12 | Inspect old/partial-tree writes, then check continuity of cron/page, marker and actual receipt. |
| `CLAUDE.md` L2, L5, L7–L11 | A lost session or reboot needs independent stop and boot evidence. Verify any canary send separately, read the raw logs, bound scope evidence and prove the runtime in A7. |
| A5 restore DM and intent, install cron, then B8a | The edits can put a live config behind a still-enabled link (finding 6). The P4 history and repeated B8a link come later in the row. |
| A7 research first, both unpaused checks, failure/cancellation/timeout/lost SSH, observations, Actions outcome/predicate/reachability | An interrupted action can lack a predicate result. Record it as unavailable, observe separately, and keep the rehearsal in research/private. |
| A7 both scopes and exact status, shared runtime/path, healthy liveness, response/history/receipt, venv aside/isolation/restore, missed pings, GitHub notice, SSH escalation and final PASS | A sibling job may recreate the venv. Isolate that check, restore before deployment, and measure actual cron ticks separately. Unverified stop goes to §4. No secrets should enter evidence. |
| B8a B8c first, stop and disable before edits, and keep the hold | Starting from an enabled research link, these prerequisites prevent a local boot start while preparing state, delivery and cron. |
| B8a revoke and verify today's reading, approved SHA, same sequence on cutover | A stale automatic file left by the restore cannot replace today's verified dated reading. Use the checked deploy in the approved window. |
| B8a flags and startup/canary cases | Missing flags can refuse before the first heartbeat. A failed restart is checked before the canary. A live-read refusal is checked if the canary runs. |
| B8a revoke/status, dated observation and timing, delayed correction, no forced restart, entry-intent and cron | A late observation can miss the day. Enter verified MLB values only to the configured directory, preserve current delivery, and run cron install under the stopped/disabled condition. |
| B8a automatic enable/fetch/status, refusal/interrupt, saver init only if uninitialized, recheck chronology, live acceptance and approved deploy, expected/stale alarms, no dry-run | A failed fetch can coexist with accepted dated state. Recheck current day and active year, use the dated route if needed, and avoid writing a saver value initialized concurrently. A scheduler dry run can write or idle. |
| B8b midnight warning, evidence preservation/lead handoff, replay and decision warnings, D3 and 2027 guard | A midnight alert or missing acknowledgement is insufficient to change or set aside files. Apply ET cutoff and verified read-only checks, then use §4 and avoid pre-2027 code. |
| B8c rescue/control boot, verified backup and stop, timer catch-up, old jobs, research plus flags, approved deploy, PASS/HEAD then cron and A7/B8a | At 10:29 or with queued timer activity, rescue and set aside the old cron before execution. Preserve its backup and wait for checked PASS/HEAD before installing current cron. Stop and hand off if old jobs ran or boot/deploy evidence is uncertain. |

## 8. Tests and mutations

- All 379 tests passed in one pytest invocation with `-m "slow or not slow"`. This included 254 workflow cases, canary, both heartbeat files, optional LightGBM and web saver. Web saver used its local loopback fixture. The wrapper started at 10:50:58 ET and exited 0. [Pytest](focused.log) · [Wrapper](focused.log.watch)
- The unmodified scratch controls passed. All eight requested mutations were detected. The nominal stamp-and-PASS test alone missed the stamp reorder, which the slow signal row caught. [Mutation run](mutations.log) · [Runner](mutation_probe.py)
- Both audited literals used shim-only PATH and an empty environment with synthetic IDs, checkout, legacy and run directories, cgroup root, Python, uv, venv entry point and null sink redirected. The `/bin/bash` frame and clean-env launcher intentionally use the local executables. The optional actual wait adapter uses local timeout only on the sleep shim.

| Scratch mutation | Test that detected it |
|---|---|
| Skip containment after reset | `test_failing_phase_command_contains_first_failure` with reset results 17 and 124 |
| Print verified stop unconditionally | `test_stop_verification_requires_all_evidence[pending-job]` |
| Ignore recovery attempt IDs | `test_older_recovery_does_not_mutate_newer_commit` |
| Publish `committed` before stamp | `test_owner_signal_at_each_phase_boundary[stamp-TERM]` |
| Use plain `sleep 30` | `test_healthy_wait_deadline` |
| Accept redirect as a page | `test_post_requires_exit_zero_and_2xx[post-0-301]` |
| Skip the live read for public delivery | `test_live_modes_use_either_contest_flag` for public, with each of the three flag combinations |
| Check timestamp before `deploy_failed` | `test_deploy_failure_precedes_timestamp` for bad and missing timestamps |

## What I ran

- Local archive and diff, exact YAML literal execution, shell and attempt probes, pinned canary/heartbeat comparisons, 3.11 syntax checks for the two inline helpers, and offline curl URL parsing with every protocol disabled. Probe code and logs are in this review directory. No tracked file changed. [Probe adapters](probe_adapters.py) · [3.11 inline compile](inline-python311.log)

## What I did not run

- No real-host SSH, GitHub Actions dispatch, remote network request, Debian systemd, installed cron, provider receipt or actual reboot. A7 owns those observations. No full fast suite.
