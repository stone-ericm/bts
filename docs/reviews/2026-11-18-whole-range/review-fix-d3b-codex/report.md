# D3 and D3b deploy review

## Verdict

**BLOCK** for `24019a9a..90a9b6f0` plus test-only patch `4733a298`.

## Blocking findings

1. **A 2xx for a rewritten URL is recorded as a verified page.** Put `https://checks.invalid/configured/../different-check` in the scheduler heartbeat assignment and fail reset, or recover a matching `begun`. The parser admits the dot segment and curl rewrites the request path to `/different-check/fail`. Both scripts print and store verified after a shimmed 200. The offline libcurl URL API confirmed the actual and literal paths differ. Add `--path-as-is` to the two failure POSTs or reject dot segments in the endpoint parser. `.github/workflows/deploy.yml:244–259,607–611,1005–1020,1368–1372`. [Probe](counterexamples.py) · [Trace](counter-dot-path.log)
2. **Two incomplete legacy state checks can certify a stop.** In deploy reset failure and matching recovery, an inactive, PID-zero, job-free `not-found` response with no fragment can still supply a known group. With `/legacy-group` populated, the absence shortcut requests no group and prints and stores verified. Separately, a previously loaded legacy unit becomes `not-found` with its fragment still present. Empty observed pre/post groups and successful stop/disable evidence make the fallback branch certify it without proving `loaded` or checked absence. Request and retain the group in both legacy absence samples, check each non-empty path, and allow the loaded proof only for `LoadState=loaded`. These are shimmed manager observations. `.github/workflows/deploy.yml:451–466,503–523,586–591,1212–1227,1264–1284,1347–1352`. [Probe](counterexamples.py) · [Trace](counter-absence.log)
3. **Repeat recovery can spend four seconds in a two-second report reserve.** Seed a matching `contained` record with an unverified page, then freshly verify stop, marker and HTTP. In the elapsed-time model, near-cap successes and a timed-out contained rename and cleanup leave 277 s elapsed. Block the output writer. The shell launches one 2 s timeout for the result lines and a new 2 s timeout for the re-checked line, ending at 281 s against the 280 s cap. Combine all final lines in one bounded print. `.github/workflows/deploy.yml:543–553,615–621,1304–1314,1376–1382`. [Probe](timing_model.py) · [Trace](counter-report-time.log)

## Notes

- Move INCIDENT §4's restore hold before the displayed restic and artifact-sync commands. On a running rebuilt box, executing in displayed order can replace pick, receipt, saver and streak files before the stop check. Make B8c's hold through the transition to normal boot explicit. [Order check](operator_check.log)
- INCIDENT §6 should say recovery pages again if the record lacks a verified 2xx. A prior real 2xx can be lost with a failed record rewrite. CLAUDE L10 can likewise use the workflow's “record says committed” wording. These are precision edits.
- The real manager's cgroup pruning, runtime equivalence, independent starts and erased `/run` state remain within the stated limits. S7/S9 deliberately contain on a published begin or an unreadable record even before the first service command. [Replay](replay-carbon.log)
- Applied `4733a298` to a scratch copy. The old B8a assertion fails and the amended assertion passes. The amended assertion still rejects missing readiness and the missing initialization guard, but also matches a typo appended to `saver-state`. The direct K20 probe checks both actual commands. [Patch check](test-only-patch.log)

## 1. Must still catch

| Case | Input run | Result |
|---|---|---|
| K1 | First/all containment clock samples fail, deploy and recovery | **PASS** stop, marker, POST, contained record |
| K2 | 790/795/805 s modeled; wait, query, enable, stamp, restart; handler read would exit 126 | **PASS** containment, no handler read or not-started line |
| K3 | Populated prior group then blank, or always blank, both scopes | **PASS** unverified |
| K4 | `{other}`, `{one,two}`, `[1-1]`, `[1-2]`; literal URL and dashboard controls | **PASS** no glob POST, unverified; executed curl calls use `--globoff` |
| K5 | Stage and rename 17/124/137, read-back 124/137; rename with error after exact commit | **PASS** numerical status or timeout and containment; exact read-back permits PASS |
| K6 | A5 and both B8a cutovers | **PASS** edits follow the stop and boot-link check |
| K7 | Raw NUL in a valid zero-exit canary prefix | **PASS** query failure, accepted line suppressed |
| K8 | Same attempt with foreign run, HTTP 301, real redirected lock and mark with injected failures, held inode | **PASS** no foreign stop, no redirect option, checked helpers, waiter proceeds after release; all five matching edits detected |
| K9 | Four signals before begin stage/rename with committed, contained, begun and absent prior | **PASS** not-started, prior bytes and service state preserved |
| K10 | Four signals after begin rename; fresh read 126 before rename, each prior | **PASS** one containment |
| K11 | Legacy loaded and group populated, then not-found, deploy and recovery | **PASS** recorded group checked and unverified |
| K12 | Empty orphan, no known path, absent throughout, pending job, failed query, dangling link | **PASS** specified controls; nearby positive-proof failures in finding 2 |
| K13 | Clock unavailable, slow stops/queries, old proof-window control | **PASS** reserve launches pgrep with TERM 10 s in both scripts; old cutoff misses it; finding 3 is an extra repeat/blocked-output case |
| K14 | Early and near-cap residual, marker and endpoint runs in both scripts | **PASS** marker by 240, endpoint by 254, POST by 270, curl gets 15 s TERM with modeled late valid URL |
| K15 | Actual commit rename followed by read 126 | **PASS** logs 126 and contains |
| K16 | Distinct pre/post groups, both population directions and empty control | **PASS** both scopes reject a populated group; two empty paths are read |
| K17 | Foreign begin, exact and four mismatched committing reads, same-ID beginning forgeries | **PASS** only exact committing read exits successfully |
| K18 | Retained page, stale stored stop/marker, independent fresh failure and failed rewrite | **PASS** no second POST and re-checked requires fresh stop and marker, independent of rewrite |
| K19 | Extracted shared proof and literal result assignments/lines | **PASS** byte-identical proof and sole promotions/emission inside it |
| K20 | Full restore unit instruction and both actual saver invocations with custom picks path | **PASS** activation through §5; both commands use configured path and season |
| Appended | Commit renamed, read-back 126, contained rename fails | **PASS** surviving record reported as a record, with §4 hand verification |

`boundary_probe.py` replays publication, orphan, read-back and mutex cases. `probe_r2_audit_new_logic.py` keeps the prior owners' bytes, contains after a published begin and preserves the pruning caveat. Both `budget_probe.py` inputs passed with the D3b outcomes. [Replay adapter](replay_prior.py) · [Slow proof](replay-budget-slow.log)

## 2. Claims and decisions

Both literals share `prove` and containment. Only deploy arms a handler; its decision delegates to `signal_rule`. Checked begin, exact commit, fresh marker, carried HTTP, positive and negative repeat evidence were traced. Findings 1–2 establish two verified results with missing or wrong-scope observations. [Commit/record](required-commit.log) · [Repeat](required-recheck.log)

## 3. Failure, signal and recovery order

Phase-command 17/124, all four trapped signals, earlier committed/contained/begun records and publication boundaries ran under shims. Before traps were armed, owner-only INT during a successful fetch could continue to checked commit, within L2/L10. Recovery-first fenced late SSH, matching old recovery worked in either order after an unpublished begin, and delayed recovery left a newer owner intact. Closed output preserved success and failure status. Killing only the owner left the mutator holding the real advisory lock until exit; later recovery contained. Clearing simulated `/run` removed the fence as L9 describes. [Phase probes](phase_probes.py) · [Recovery and transport](orders.py)

## 4. Time

The model launched residual at 204 s, with TERM 10 s. It reached a capped marker by 239, accepted the late URL at 247 and got to curl with TERM 15 s. With the old proof cutoff the model skipped residual. The timed historical proof replay launched it and finished in 199.772 s, with verified marker and HTTP and an unverified stop. The timed marker timeout still posted and recorded an unverified marker. The two final logging calls can exceed the cap in repeat recovery, as shown in finding 3. [Timeline](timing-model.log) · [Proof replay](replay-budget-slow.log) · [Marker replay](replay-budget-slow-marker.log)

## 5. Whole-branch controls

Archived base and round-2 compared with D3b, for active, auto-restart and installed legacy cases. Candidate checked HEAD, ran four checks, enabled the link, renamed the stamp and checked commit before PASS. Pre-rename faults preserved old stamp bytes; later failures retained a complete new stamp. CLI matched the real startup guard and morning read over 24 startup and 72 state pairs plus the ET boundary. Python 3.11.18 and 3.12 compared 55 ordinary heartbeat CLI cases with base and exercised malformed/marker paging. Stock 3.11 also ran the inline workflow Python. [Transcript](success-success.log) · [Stamp](success-stamps.log) · [Parity](canary-parity.log) · [Heartbeat 3.11](heartbeat-311.log) · [Inline](inline_runtime.log)

## 6. Operator sentences

The [full sentence audit](operator-audit.md) covers the changed workflow lines, POST body, INCIDENT, CLAUDE and checklist. Key outcomes follow.

| Sentence | Worst literal outcome |
|---|---|
| Workflow not-started / no service change | A previous invocation or a matching recovery can still have stopped or started a scheduler. |
| Workflow previous SHA / requested target | Neither identifies the reached partial tree. |
| Workflow relaunch / PASS | Checks describe run-local observations; later starts and pick delivery are separate. |
| Workflow stopped / heartbeat / HTTP verified | Legacy proof can certify a populated group; heartbeat can be overwritten afterward; curl can get 2xx from the wrong path. |
| POST failure, target and stop status | Can send the wrong stop status to the wrong route. §4 directs read-only triage. |
| Workflow containment re-checked / committed record / no record present | A latest record write can fail, or reboot can erase it. The separate final print also breaks the time reserve. |
| INCIDENT §§1–2 approved ship, revert and re-run | A re-dispatch of a moved branch can restart the scheduler on another SHA. The checked retry rule is in §5. |
| INCIDENT §4 read logs, preserve evidence, verify MLB and delivery | Read-only triage and no resend guard against changed or duplicate picks. |
| INCIDENT §4 dashboard and filesystem restore | Dashboard redeploy restarts both units. Restore hold occurs after data replacement in written order; move it before restore. |
| INCIDENT §5 approve same SHA and ET window, keep stopped if too late | Can still lose a day, but avoids unsupported or manual activation. |
| INCIDENT §6 page and cron | Prior 2xx can be repeated after failed record publication; active cron may write against the reached tree. |
| CLAUDE deploy and emergency lines | Approval and checked SHA matter; soft hold only affects the local boot link. |
| CLAUDE limits and stop-proof statement | Later starts, receipt, runtime differences and cron writes are outside a verified snapshot. Use exact record wording for L10. |
| A5 postpone delivery/intent/cron | Keeps live configuration behind B8a's stopped/disabled check. |
| A7 rehearse, verify stop and receipts, restore venv before checked deploy | Failed rehearsal still needs same §4 checks, and sibling cron can recreate the venv. |
| B8a stop/disable, prepare today's readings and deploy approved SHA | Right path and date protect live inputs; two actual saver commands operated only in the custom directory. |
| B8a automatic cutover and fetch failure instruction | Keeps approved dated fallback and explicit cutover checks. |
| B8b preserve evidence, recover under §4, prohibit old code | Avoids erasing pick files or running October semantics on 2027 state. |
| B8c maintenance, back up crontab, use research, confirm PASS/HEAD before cron | A rescue-to-normal boot can run an old enabled unit unless the hold persists. Keep the old table set aside and consult Eric if old jobs ran. |

## 7. Mutations

| Scratch change | Existing selected tests |
|---|---|
| Skip recorded legacy group after not-found; drop residual window | Both detected |
| Unreadable beginning becomes not-started; exact committed read while beginning becomes success | Both detected |
| Accept curl 7 with 200; print result outside prove | Both detected |
| Ignore dangling link; ignore pre-stop user group query status | Both detected |
| Drop run ID match; follow redirect; break lock-file creation; break cancellation mark; drop recovery wait | All five detected |
| Drop recheck evidence gate, omit exact committing record match, accept nonzero MainPID | All three detected |
| Increase each final report grace from 1 to 5 s | **Survived** the six selected deadline, proof and recheck assertions; independent timing probe detects cumulative final time |

Sixteen of seventeen isolated edits were detected by the selected existing tests. [Commands and failures](mutations.log)

## What I ran

Independent shell shims, prior-review replay inputs, selected Git archives, real local advisory locking, offline libcurl URL parsing, local CLI calls and both Python versions. The wrapped six-file focused suite collected 490 tests, including slow cases. 489 passed. The existing `test_do_post_saver_transition_socket_wiring` failed at a loopback bind with sandbox `PermissionError`. Wrapped mutations and the timed historical replay ran separately. [Focused log](focused.log) · [Ready commands](READY-FOR-PYTEST) · [Slot completion](PYTEST-DONE)

## What I did not run

Full fast suite, real systemd, real host, network, SSH action/provider delivery, or physical reboot. Elapsed model and shimmed service behavior are labeled in the evidence.
