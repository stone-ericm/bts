# D3 review, round 2

## Verdict

**BLOCK** `d57f4013` against `d3d05abd`, using the D3 contract through amendments 3 and 3b. The original blocking inputs now behave as required. The revised publication trap can stop a healthy scheduler before it owns the attempt. Stop verification can overlook a known populated cgroup or skip a required residual-process check despite time remaining.

## Blocking findings

1. **TERM before publication can contain a box this invocation never changed.** Start with run `101.1` committed, HEAD at `aaaa…`, an enabled and running scheduler, and a `running` heartbeat. Run `123.1` handles TERM while staging `begun` or before the rename. The bounded read succeeds and sees the same older committed record, proving the new begin did not publish under the lock. `deploy.yml:576–595` treats only a missing record as a confirmed unpublished begin. It disables and stops the old scheduler, overwrites the heartbeat and prior record, and sends a POST despite never resetting HEAD. The same staging signal on `d3d05abd` leaves the scheduler alone. The new code also leaves it alone if there was no previous record, and contains if the matching begin was actually renamed. When `publication=beginning`, treat a valid different record as an unpublished begin and preserve the fail-closed path for unreadable evidence. [Probe](boundary_probe.py) · [Five traces](publication.jsonl)
2. **A legacy unit that becomes `not-found` bypasses its recorded cgroup.** Matching `begun` recovery sees a loaded legacy scheduler with no fragment and records `/legacy-orphan`. Legacy disable and stop return zero. The later lookup reports named `not-found`, inactive, PID 0, with no jobs, while the recorded `cgroup.events` still says `populated 1`. `deploy.yml:524–530,1216–1222` takes the absence branch without calling the cgroup helper and prints `Scheduler stopped: verified.` Keep the recorded path in the proof even on later checked unit absence. With the same file and the legacy unit still reported loaded, recovery checks the group and prints `unverified`. This is a verifier counterexample from shimmed manager observations. Actual Debian unit collection remains an A7 observation. [Probe](boundary_probe.py) · [Traces](legacy.jsonl)
3. **The fallback proof cutoff can skip the mandatory residual-process check with time left.** Matching recovery with the clock unavailable stopped both scopes. A slow legacy jobs query started at 203.3 s. The 220-second proof cutoff shortened that query and left too little for the mandatory five-second grace on the next call. No residual `pgrep` was issued. The marker started at 215.4 s, the POST and record update succeeded, and the run ended in 216.4 s against the 280-second handler cap. `deploy.yml:291–309,517–540,1209–1232` lets an earlier proof query consume the time for an independent required check. Reserve a launch window for residual `pgrep` while preserving the marker and POST budgets. [Timed probe](budget_probe.py) · [Timeline](budget-proof.jsonl) · [Wrapper](budget-proof.jsonl.watch)

## Notes

- A commit read-back that exits 126 after the rename is reported as `2` by `publish`. The original 17/124 publication faults now retain status, and read-back 124 and 137 log `timeout`. The remaining 126 case still contains and leaves a complete stamp. [Read-back traces](readback.jsonl)
- The new `INCIDENT.md` restore sentence points to a checked deploy. The preceding legacy instruction still says to install unit templates, then enable and start. The two sentences should make the checked deploy the scheduler activation step. The installer itself copies and reloads. It neither enables nor starts units.
- B8a's new custom-directory sentence omits `saver-state`. The literal commands initialize `data/picks` when the live scheduler uses a different directory. The real CLI reported `used` in the default directory and `uninitialized` in the configured directory. Add `saver-state` to the list that takes `--picks-dir`. [CLI trace](operator.jsonl)
- Bash `SECONDS` now controls all containment runs, including successful external clock samples. The sampled origin cancels out of the elapsed calculation. The tests’ recovery lock uses real advisory exclusion and a one-second simulated wait. Production still requests 1,500 seconds.

## 1. Round-one blockers on their original inputs

| Original fault | Round-two outcome |
|---|---|
| Codex R1, first containment clock fails on deploy and matching recovery | Both issued stop, wrote `deploy_failed`, sent one POST and published `contained`. The same worked with every later external sample unavailable. [Clock replay](replay-codex-clock.jsonl) |
| Codex R2, expired (805 s) scheduler query and read returning 126 after restart | Both ran containment with candidate HEAD and disabled boot link. The trap used the confirmed begin without requiring the old read. [Signal replay](replay-codex-signals.jsonl) |
| Carbon B1, HUP at 795 s in the wait, TERM after enable/stamp at 795 s, and read failure after restart | All three contained, disabled the boot link and paged once, with no `DEPLOY NOT STARTED`. [Carbon replay](replay-carbon.jsonl) |
| Codex R3, unknown path with a populated subtree, and the named-path control | Both reported `unverified`. The new legacy absence branch in finding 2 is an additional case. [Proof replay](replay-codex-proof.jsonl) |
| Codex R4, `{other}` and `{one,two}`, and Carbon B3, `[1-1]/uuid` | All three refused before POST. Valid POST and dashboard argv carry `--globoff`. [Brace and NUL replay](replay-codex-proof.jsonl) · [Bracket replay](replay-carbon-glob.jsonl) |
| Codex R5 staging/rename statuses 17 and 124, and Carbon B2 staging, rename, shared-budget expiration | Each result printed 17 or `timeout` as appropriate, contained, and preserved the complete renamed stamp. [Codex replay](replay-codex-commit.jsonl) · [Carbon replay](replay-carbon.jsonl) |
| Codex R6, enabled research link plus accepted dated observation | The old A5 edit made the real startup/morning gate accept DM. A5 now performs no edit while the link is enabled. B8a places the edits after stop and disable. [Operator replay](operator.jsonl) |
| Codex R7, NUL inside the accepted prefix | The shims passed a zero-exit malformed line to the raw capture. `contest-state` failed with `query` and there was no PASS. [Proof replay](replay-codex-proof.jsonl) |

## 2. Round-two delta and additional boundaries

- The runtime delta is in the two SSH literals. It introduces a Bash budget, the four publication states, a fresh 15-second uncertain-publication read, pre-stop cgroup discovery, raw contest capture, and curl literal mode. CLI, heartbeat monitor, cron line and action predicate are byte-identical to round one. The shared containment helpers remain byte-identical across both SSH literals.
- Repeated the 78 paired ET startup and morning-read cases and six bounded-config error cases. Repeated pinned base parity for 32 ordinary heartbeat states, four `deploy_failed` cases and special-file cases under Python 3.11.18 and 3.12.16. Both inline helpers compile on 3.11. [Canary](canary-parity.jsonl) · [Heartbeat 3.11](heartbeat-311.log) · [Heartbeat 3.12](heartbeat-venv.log) · [Inline compile](inline-311.log)
- Publication controls covered signal before staging with absent and prior records, signal during a failed rename, signal after a real begin rename, and the archived late-mutation inputs. Finding 1 is the new unsafe stop. A late signal after a verified commit can still exit successfully. A prior unrelated commit cannot turn the new invocation into success.
- Recovery acquired the exact temporary lock after the holder released it and created an empty mark. Holding the lock past the shim deadline left the record and lock contents intact and created no mark or service mutation. Both logged `flock -w 1500 9`. [Lock traces](locks.jsonl)
- The new pre-stop cgroup discovery shifts user/legacy stop cutoffs from 80/170 seconds to 90/180 seconds. The proof window shrinks from 50 to 40 seconds. The elapsed-time probe resolved both stops, then skipped `pgrep` under that proof cutoff. Finding 3 records the measured timeline.
- The fallback did terminate a blocked marker command. With every containment clock sample unavailable, a marker shim set to run for 30 seconds was terminated. Recovery ended after 17.1 seconds, saved `contained`, reported the marker `unverified`, and still issued one POST. [Timed control](budget-marker.jsonl)
- Forcing the containment clock to fail in the legacy-absence replay also produced the false verified stop with time available for its cgroup check. The cgroup bypass is common to both budget modes. [Fallback control](legacy-fallback.jsonl)

## 3. New or changed operator sentences

| Sentence or output | Worst state and literal action |
|---|---|
| Workflow `DEPLOY NOT STARTED` now depends on publication memory or the unpublished-read branch | A lost late session leaves a changed box, but the original late inputs now show actual containment. With an intact prior record and no new publication, the revised workflow can report `DEPLOY FAILED` and stop the old unit. Read both logs and verify live state before retrying. |
| `Check commit-record failed (<status>)` preserves publication status | The new stamp may already be present when the record fails, so act on stop evidence and the attempt record. A complete stamp alone cannot authorize a start. |
| `Scheduler stopped: <result>` and `DEPLOY RECOVERY — containment re-checked.` use the pre-stop group | A blank group after a known populated snapshot normally reports `unverified`. A legacy name that vanishes can still falsely verify, which can also satisfy the re-check summary when marker and page evidence are positive. Check the known cgroup on the host before retrying. |
| `Failure heartbeat written` and `Failure ping received HTTP 2xx` after clock fallback and literal URL validation | When an old clock sample fails, read the independent stop, marker and response evidence. The glob URLs no longer produce positive HTTP evidence. A 2xx still needs provider history and receipt checks. |
| F9 line accepted, skipped, refused or error after raw-byte handling | A corrupted candidate line can now fail the check without being printed. A normal accepted line is a state read. Resolve uncertain delivery through §4 before acting. |
| `INCIDENT.md:110` applies §4 and §5 to a dashboard-only restart | A dashboard outage with a potentially sent pick requires the same delivery, ET and approval checks before restarting through the workflow. |
| `INCIDENT.md:127` follows §4/§5 for activation after restore | A copied October unit and crontab can still be present. The new sentence leads to checked activation, but line 126 can still be followed literally as a direct start before it. See Notes. |
| `CLAUDE.md:149` follows B8a in an approved sleep window | A research deploy can keep an enabled boot link. Do the B8a hold and state checks before the checked live deploy. |
| A5 follows B8a before changing delivery, intent or cron | With a still-enabled link, A5 directs the operator to B8a before changing any of those inputs. A reboot in that interval stays in research. |
| B8a defaults to `data/picks` and names commands for the custom path | The listed reads and edits will use the live scheduler's directory when the operator supplies it. The omitted `saver-state` command leaves a separate wrong-directory write path. |
| B8a sets both flags, then restores DM and sets `enter` | With the scheduler stopped and its local boot link off, a reboot during edits cannot start it through that link. The live scheduler still needs today's verified contest state. |
| B8a exports `.env`, runs cron `show`, then runs `install` | The entry checker and 07:40 reconcile are set up while the scheduler hold is in effect. Reconcile can write its usual outputs. This is followed by the approved checked deploy. |

## 4. Tests and scratch mutations

- All 421 focused tests passed in one process with `-m "slow or not slow"`, including 296 workflow cases. The wrapper started at 14:35:46 ET and exited 0. The harness used the exact YAML, shim-only `PATH` and redirected production paths. Web saver used its local loopback fixture. [Pytest](focused.log) · [Wrapper](focused.log.watch)
- All unmodified scratch controls passed. All 18 mutations were detected, with zero survivors. The four Carbon gaps were replayed, including separate lock-file and mark faults. [Mutation results](mutations.log) · [Wrapper](mutations.log.watch) · [Runner](test_driver.py)

| Mutation | Detecting test |
|---|---|
| Skip containment after reset | `test_failing_phase_command_contains_first_failure` on reset 17 and 124 |
| Print verified stop unconditionally | `test_stop_verification_requires_all_evidence[pending-job]` |
| Ignore recovery attempt IDs | `test_older_recovery_does_not_mutate_newer_commit` |
| Publish committed before stamp | `test_owner_signal_at_each_phase_boundary[stamp-TERM]` |
| Plain `sleep 30` | `test_healthy_wait_deadline` |
| Accept redirect as a page | `test_post_requires_exit_zero_and_2xx[post-0-301]` |
| Skip public contest read | `test_live_modes_use_either_contest_flag` for public with flags |
| Judge timestamp first | `test_deploy_failure_precedes_timestamp` with absent and bad timestamps |
| Print not started after a post-begin read error | `test_round1_late_signal_uses_confirmed_begin[record-read-126]` |
| Drop the fallback clock | `test_round1_clock_failure_does_not_skip_containment` in both scripts |
| Allow blank cgroup to verify absence | `test_round1_unknown_control_group_cannot_verify_stop` and the user and legacy populated-subtree cases |
| Drop `--globoff` | `test_round1_curl_disables_globs` |
| Collapse commit-record failure status | `test_round1_commit_status_survives_failed_publication` and `test_round2_all_commit_timeouts_keep_timeout_status` |
| Ignore a different run ID at the same attempt | `test_round2_different_run_with_same_attempt_cannot_recover` |
| Add `-L` to POST | `test_round2_failure_post_never_follows_redirects` |
| Break the lock-file helper or the mark helper | `test_round2_recovery_creates_real_lockfile_and_mark` for each |
| Use nonblocking `flock` in recovery | `test_round2_recovery_waits_for_a_held_lock` |

## What I ran

- Local commit diffs and archive reads, both reviewers' archived shim inputs, boundary and lock probes, and the pinned real-CLI operator replay. The archived scripts came from local commits `f4e42626` and `079bd6c4`. Separate wrapper runs executed focused pytest, scratch mutations and the elapsed-time recovery probe. Traces omit repeated inline helper source. No tracked files changed.

## What I did not run

- No real SSH, production tool commands, workflow dispatch, provider request or receipt, actual boot, real Debian manager, installed cron or author gate. The lock wait used a compressed shim deadline. The old and new B8a reboot paths used a simulated local boot link and the real CLI startup gate. The direct heartbeat run omits the Unix socket bind denied in round one.
