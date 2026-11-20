# Verdict — BLOCK

Reviewed `b128854d..46ce572d` against v5 and its two amendments. The checkout's tracked files are untouched.

## Blocking findings

1. **A cron writer escapes the hold.** Preflight accepts all eight expected rows plus `uv run 'bts' 'reconcile'` or an indirect shell wrapper. The scan recognizes only contiguous unquoted tokens. A real `/bin/sh` replay dispatched the accepted quoted command and wrote the fixture while `pending`. A commented-out heartbeat row also passed. Validate an exact approved inventory, including enabled monitor, and refuse unapproved command rows until reviewed. `scripts/install-deploy-guard-hetzner.sh:648–661`, [preflight.log](preflight.log).
2. **A dead owner can admit scheduler and writers.** After the prior MainPID dies while `activating`, its unit can be `deactivating` as `ExecStopPost` is starting. A competing deploy takes the exclusive lock. Given that state, the actual gate let the scheduler and four jobs pass and the old hook failed to get the lock. If the competing deploy then refuses `guard-stale`, neither path stops the old activation. Bind the matching live systemd invocation to the actual lock holder, and let the hook finish containment after bounded contention. The manager states were shimmed. `scripts/install-deploy-guard-hetzner.sh:248–266,510–517,765–782`, [preflight.log](preflight.log).
3. **Loaded unit drift can defeat admission and certify the wrong state tree.** The parser accepts `path=/usr/bin/true` alongside expected guard argv and never checks `SuccessExitStatus=1`, which systemd 252 treats as a successful condition result. It also never asks for `WorkingDirectory`. With the same relative `picks_dir`, the real canary accepted in the project and refused in the alternate loaded workdir. Pin effective executable, flags, workdir and runtime environment, and require guard exit 1 to skip. `scripts/install-deploy-guard-hetzner.sh:625–647,665–669,1116–1127`, [preflight.log](preflight.log), [canary.log](canary.log), [local v252 review](../../docs/reviews/2026-11-18-whole-range/contract-check-deploy/contract-check-r3.md).
4. **The first guarded deploy leaves the old saver endpoint writable.** A page opened before arm can POST to the already loaded base dashboard until the restart after sync. The base handler returned 303 and changed saver and audit bytes with a hold present. The documented unit installer reloads units and leaves processes running. Require a verified stop of the old dashboard before the first arm, including in A7 and `CLAUDE.md`. `scripts/install-deploy-guard-hetzner.sh:798–809`, `docs/ops/2027-season-start.md:16`, `CLAUDE.md:116`, [dashboard.log](dashboard.log).
5. **The canary certifies a box that cannot pick or deliver.** DM, public and research each passed with `cascade_budget_min="forty"`, then real scheduler startup threw before READY. With an accepted dated observation, missing `tiers` raised at real selection and missing `bluesky.dm_recipient` refused real delivery inside the cutoff. Those last two can survive startup verification until a lineup check. Share read-only runtime configuration validation with the scheduler, including enabled tier schemas, required paths, typed settings and DM recipient. `src/bts/cli.py:2644–2661`, `src/bts/scheduler.py:1315–1319,2892`, `src/bts/orchestrator.py:272`, [canary.log](canary.log).
6. **A dated deploy can release after the certified observation expires.** With the final real canary at 23:59:59 ET and the modeled activation completing at 00:00:01, the actual owner unlinked and printed success. The same real canary then refused today's state. Validate the certification's ET date and input generation at activation and immediately before unlink, and retain the hold if changed and revalidation fails. `scripts/install-deploy-guard-hetzner.sh:813–825`, [canary.log](canary.log).
7. **The recovery instructions can write the wrong tree or start the old workflow.** For `picks_dir=data/live-account`, I followed the unqualified dated write and status commands. `data/picks` gained an accepted observation and `data/live-account` stayed refused. Specify target plus locked sync, use `uv run --no-sync` and an explicit verified `--picks-dir`, and require direct readback before retry. The revert paragraph also omits retaining the guard protocol. An exact base payload replay while held reset the checkout twice following its rollback branch. Put a guarded-workflow/installer compatibility requirement before all ship, dispatch and revert commands. `INCIDENT.md:97–100,141–184`, [operator.log](operator.log).

## Notes

- Immediate page evidence is marked `attempted` before the POST. A kill there left `page=attempted` with zero sends. Hook and next preflight skipped the send. Preserve a truthful claim/receipt distinction and use a retry or explicit uncertainty on a lost claim. The periodic/missed-ping channel remains available. The next preflight also reconciles the abandoned hold but can leave the old run result missing. `scripts/install-deploy-guard-hetzner.sh:488–499,775–782`.
- A pre-arm refusal can leave the prior process running; the runbook's blanket held-failure claims overstate containment. `Activation failed after certified release` describes the wrong durable state, and the candidate output with ET classification is discarded by `require`. A TERM inside the selector can be classified as a step timeout; its local child was killed. `scripts/install-deploy-guard-hetzner.sh:360–403,665–669,843–865`.
- A verified start can publish its sleeping heartbeat before unlink and then sleep until the next check without updating that file. V11 correctly pages on that heartbeat after release, which makes a normal sleep-window deploy page until wake. Emit a genuine scheduler heartbeat during long waits. `src/bts/scheduler.py:1455–1458,3005–3013`, [monitor-39.log](monitor-39.log).
- Accepted limits include snapshot fence R, V4 reminders, unheld analytics/queued work, and the frozen parser's `canary=absent` classification. The harness cannot establish Debian systemd 252 condition evaluation, cgroups, real `RuntimeMaxSec`/`ExecStopPost`, power-loss durability, uv 0.11.6, or notification receipt. A7 must do those checks, including real Python 3.11, separately.

## 1. Owner, scheduler and writers

- Exact Bash 3.2 workflow payload and phase digest, then embedded owner code with safe launch/child-command boundaries. The matrix used failure, TERM, KILL, lost SSH and modeled runtime TERM at 21 boundaries each, with hook and next-preflight paths. All scheduler callers share the same tested predicate. `pending` and `failed` closed it and all four writers. A live `activating` state opened them after both checks, and abandoned activation closed them before reconciliation. Lost SSH after acceptance preserved the owner and durable result. KILL after the verified unlink left clear with verified evidence and sometimes no result. The page-claim exception and competing owner defect are above. [Hook traces](owner-kill-hook.log), [next-preflight traces](owner-kill-preflight.log), [SSH traces](owner-lost-ssh.log), [timeout traces](owner-runtime.log), [command child](owner-command.log).

## 2. Canary

- Snapshotted every fixture file's bytes, mode and mtime. Good DM/public use the actual required-state loader with one ET sample. Both accepted dated and same-generation automatic observations. Missing state, wrong automatic generation and mismatched modes refused, and consistent private/research skipped the read. The active observation expired at midnight and at December 31 / January 1 in the CLI. The owner can still release after rollover. Required scheduler/model and delivery imports were observed. [canary.log](canary.log).

## 3. Guard and preflight

- Fingerprint mismatch refused before Git config/fetch, reset or service actions. Loaded command, condition count, ignore-errors and reload controls ran against command shims. Active/waiting limits, a 29:59 vs 30:00 sleeping wake, absent/stale heartbeat with exact recovery SHA, and both owner/writer contention passed controls. Canonical cron counts were 1/2/4/1. Indirect, fully quoted and disabled monitor variants exposed the failures above. [preflight.log](preflight.log).

## 4. Monitor

- Python 3.9.6, the oldest available, matched base stdout, stderr, exit, ping method/suffix/body and churn bytes in 372 pinned no-hold comparisons with a valid guard directory. **No-hold regression** with the directory or `state.lock` missing: the same fresh sleeping input changed from exit 0 and GET success to exit 1 and POST `/fail`. Preserve base behavior on confirmed absence of both records. Young matching live owners suppressed, other holds paged every run, and old/equal heartbeats after both release kinds failed. Ran 108 held cases plus FIFO and dangling symlink. Python 3.11 absent locally. [monitor-39.log](monitor-39.log).

## 5. Cron installer

- Replayed all eight gated rows in both modes, plus 9 changed unheld research rows and 10 enter rows, through real `/bin/sh` with cron `%` conversion. Working directory, exported environment, ET date arguments and logging matched. The grader's mocked `flock` used actual fcntl exclusion. `pull`/`build` failure stopped the chain and preview/R2 failures remained independent. Show headers matched, six refusal cases preserved the crontab, and the existing entry-intent assertions were unchanged. [cron.log](cron.log).

## 6. Dashboard

- 24 full GET response hashes matched `b128854d`, including both audit-progress outcomes and held/clear controls. A preloaded tab, unavailable guard and arm-first cases returned exact 409 bytes and touched no fixture state. A POST holding shared admission prevented arm until the saver/audit commit. The base process accepted under a hold as above. [dashboard.log](dashboard.log).

## 7. Hand release

- No hold, pending, activating, mismatch, dirty tracked tree, uv exit 1/2/missing, invalid reasons, busy owner, unreadable hold and canary failure all refused with byte-identical guard files. Real file/directory fsync preceded unlink and exact two-line success followed it. Audit fsync error retained the hold and the attempted intent append. Real pinned uv 0.11.6 remains A7. [preflight.log](preflight.log).

## 8. Operator text

The expanded audit records exact output and changed sentences individually. [Operator audit](operator-audit.md)

| Text or source | Literal action and worst-state outcome |
|---|---|
| Workflow `DEPLOY NOT STARTED — guard failed.` | Guard unavailable. A half-built prior tree stays in place. Inspect before retry. |
| Handoff `… target failed.` / `… phase payload invalid.` / `Deploy run <id>.` | Record the ID if available. No implication for a previously delivered pick. |
| Handoff `Deploy outcome unknown (run <id>).` / `Inspect the on-box result and hold read-only.` / `Contact Eric before any retry.` | Lost SSH mid-send. Inspect or escalate. No rerun while outcome is unknown. |
| Owner `DEPLOY NOT STARTED — abandoned deployment reconciled.` / `… hold-busy failed.` / `… <step> failed.` | Inspect the relevant run. A half-built prior run or concurrent owner can remain. |
| Owner `… guard-stale failed.` / `Re-run scripts/install-deploy-guard-hetzner.sh as root from <target7> before deploying it.` | First provision the verified target copy. A stale guard only refused this request. |
| Owner `This run did not change the checkout or services.` / `Previous SHA: <sha>.` | Pre-arm refusal leaves any earlier delivery and partial environment as found. |
| Owner `DEPLOY FAILED at <step> — requested target <sha>, previous SHA <sha>.` / `No automatic rollback was attempted.` / `Recovery has not been verified.` | During activation a DM may already be out. Use receipt and MLB, then fix forward. |
| Owner `Deployment hold: verified./unverified.` / `Scheduler stopped: verified./unverified.` / `Failure ping received HTTP 2xx: verified./unverified.` | Check unresolved containment. Verified 2xx is HTTP acceptance, an unverified stop can coexist with a mid-send. |
| Owner `Activation failed after certified release.` / `Verify the new incident hold and any delivery/entry during activation before retrying.` / `Recovery procedure: INCIDENT.md §4.` | Inspect original failed hold and independently reconcile delivered/entered status. “After certified release” is inaccurate before the unlink. |
| Owner `Scheduler activation passed.` / `Calibration timestamp update failed (stamp write failed).` / `Record the warning and follow §4 for calibration diagnostics.` | Read the cutoff. A pick mid-send stays undisturbed. |
| Owner `Deployed <sha>` / `CANARY PASSED — <sha> is live and healthy` | Observational. On a dated midnight, success can outlive the validated date (block above). |
| Guard `bts-deploy-hold 1` / `no deploy hold` / `unreadable` / `deploy hold lookup failed` / `<state> at <step> (deploy …)` / equivalent JSON | Read-only. No authority to retry an unknown handoff. `check-clear` is silent. |
| Guard `[<UTC>] bts-deploy-hold: <job> skipped: <reason>` / usage and `expected an absolute command` | Fix input or inspect lock/hold; no manual grader start on a live slate. |
| Hand release `hand release refused: <reason>` / `… environment could not be verified as synced for <sha>; use a deploy` / `… guard or transaction unavailable` | The half-built checkout stays held. Follow §5 before a verified deploy. |
| Hand release `hand release outcome unverified: inspect hold and release audit` | Inspect if the command lost its completion audit. Cron may already be eligible. |
| Hand release `Deploy hold released by hand at … Reason: ….` / `Cron and boot activation are now permitted.` / `This command did not request a scheduler start.` | On a tick or reboot, jobs can already be active. Recheck prior send and MLB before Eric's start. |
| Guard installer `<item> installed/already current` / `FAILED: run the installer as root` / `FAILED: invalid DESTDIR` / `FAILED: deploy guard installation or verification` | In a partial install, verify loaded condition and matching fingerprint. Status lines alone grant no start. |
| Guard installer `FAILED: installer source must be a regular file` / `FAILED: installer source size` / `FAILED: installer source changed during installation` / `FAILED: scheduler user manager needs reload or could not be queried` | Stop provisioning and use the verified release artifact and effective unit. No pick action. |
| Cron installer `Install the deploy guard with scripts/install-deploy-guard-hetzner.sh` / `ERROR: <project>/.env not found.` / missing ping environment refusal | Provision before retry. Old crontab remains as found. |
| Cron installer `ERROR: refusing to <action>: no valid …` / `… unexpected entry-intent output: …` / `ERROR: cannot filter the current crontab` / `ERROR: cannot read the current crontab: …` | A research/DM mismatch or unreadable tab should remain unresolved until verified. No entry or resend. |
| Cron installer `Installed BTS Hetzner cron jobs (entry_intent = …). Verify with: crontab -l` / `Current BTS-HETZNER cron entries:` / `(none)` / `Entry intent: …` / `Would install:` / rendered rows / `Removed BTS Hetzner cron jobs.` / usage | Inspect actual schedules and guarded commands. The schedule listing includes a ping endpoint; keep it out of copied evidence. |
| Monitor `deploy in progress (run …, armed …s ago); no ping` / `Deploy hold: <state> at <step> (requested target …). See INCIDENT.md §4.` / `Deploy hold present and unreadable.` / `Deploy hold lookup failed.` | Loss of SSH calls for run inspection. Monitor output cannot settle a delivery. The competing-owner false suppression follows the gate defect. |
| Monitor `Deploy hold release record unreadable.` / `Deploy hold released at … (<kind>); no scheduler heartbeat since. See INCIDENT.md §4.` / `Failure ping received HTTP 2xx: …` | A released scheduler may sleep before its next heartbeat. Inspect the specified generation. |
| Monitor legacy `[<UTC>] stale=<bool>  reason=<reason>` / `  (no --ping-url provided; alerts disabled)` / `churn state write failed: …` / `churn check errored (continuing with liveness only): …` / `ping failed: …` | Diagnose monitoring. Prior send or current entry requires external evidence. |
| Canary `deploy canary <ET>: runtime unavailable` / `configuration refused` / `contest state refused` / `contest state unreadable` | On a half-sync or just after midnight, verify runtime and actual configured path before any edit. |
| Canary `deploy canary <ET>: research; contest state skipped` / `live contest state accepted (<kind>, streak <n>)` | No entry action. Acceptance is sampled; midnight or unusable configuration can still make deploy success wrong. |
| Dashboard `deploy hold present: saver changes are refused` / 408, 413, 404 and ordinary saver responses / `BTS Dashboard running at <url>` | Inspect only. Before upgrade, the original handler still writes on a valid old-tab POST (block above). |
| `INCIDENT.md:27–32` arm/reset; stop/drain/sync/candidate/dashboard; enter-mode sampled date; starts then clears; failure keeps checkout/closes jobs; verified fix forward | Prior pick already delivered, pre-arm refusal can leave it and the process in place. A partial reset also leaves an uncertain tree. |
| `INCIDENT.md:42,47` ship after §5; dispatch is a full deploy of selected SHA | Check the exact SHA and recovery window. An old workflow selected by the revert path is a block. |
| `INCIDENT.md:90–96` read result and status; exit 1 expected; contact Eric on doubt; preserve hold/files; verify ET, Bluesky and MLB; dashboard inspection; no entry/resend from a preview/alert; prior activation delivery possible | Safe on an unknown remote result with a pick mid-send or already entered. Status does not replace outcome checks. |
| `INCIDENT.md:97–100` read status; check prior settlement and enter today's observation; rerun in window; keep held if uncertain | Without explicit configured path and synced runtime, the real CLI can file and report success on the wrong tree. This can lose the day. |
| `INCIDENT.md:101–103` repair via targeted deploy; check stop and ping evidence; 2xx acceptance; missed-ping fallback | Preserves partially built runtime and receipts, subject to §5. |
| `INCIDENT.md:104–109` Eric checks revision/state/delivery/entry/window; release command checks runtime; keep hold on uncertainty; cron/boot can start immediately; recheck before start | Safe literal sequence when prerequisites hold. At midnight recheck the date and state too. |
| `INCIDENT.md:110–114` record/calibration cutoff; read-only gap review; chronological backfill after Eric checks | No redeploy for a timestamp. Backfill should explicitly exclude concurrent grader/reconcile before touching live scoring. |
| `INCIDENT.md:150,176–184` window before ship; verify revision/state/intent/delivery/entry; revert compatibility; October SHA cannot serve; wait on writers; recheck window; full dispatch; exact recovery SHA | Preserve an already delivered slate. Require the guard-compatible workflow as well as state/job compatibility before every command. |
| `INCIDENT.md:191–193` held/released/missed heartbeat, independent box check, yesterday disposition | Read diagnostics only. Yesterday's message cannot authorize today's entry. |
| `CLAUDE.md:115` hold, stop, drain, stage, verify; failure containment and page; fix forward | An early refusal and interrupted cleanup require checking run evidence before assuming service stopped. |
| `CLAUDE.md:116–117` install units and reload; `--no-sync`; stale unit refusal; install/reinstall guard; fingerprint refusal; verify A7 | First deploy can leave an already loaded unguarded dashboard. Provisioning needs an explicit verified stop. |
| `CLAUDE.md:119,121` grader to ~06:00; busy writer refuses; wait and verify; triggering SHA; reset/sync/restart/verify; exact recovery per §5 | Verify prior entry and wait. No instructions to interrupt an existing job or resubmit. |
| Season A5 new R and B8a prerequisite; verify guard and ET window | Research restore stays fenced under R. On a live dated day, configuration and observation need rechecking just before start. |
| Season A7 rehearse after R; install/reload units; `--no-sync`; stale-unit refusal; install/reinstall guard; fingerprint refusal; verify root, systemd, uv and 3.11; exercise failures/missed pings; record receipts | Exercise fixtures only. On a provisioned box running the old dashboard, reload leaves the unsafe handler in place (block above). |
| Season B4 verify 08:30 line; rehearse with isolated absent-decision fixture | The in-season delivered pick and its original files stay in place. |
| Season B8a suspension until R; current-day live refusal; dated-state stderr/CLI behavior | just after midnight, preserve late-game state and await verified settlement and date. |
| Season B8b W12/W13 require verified D; failed deploy holds and attempts containment/page; October SHA cannot serve 2027 | Inspect the phase before assuming a hold exists. No rollback or retry based on the sentence. |
| Season B8c suspension/fence; install/reload target units; `--no-sync`; stale refusal; install guard/reinstall changed bytes; fingerprint refusal; verify cron; avoid October commands; running boxes refuse unchecked guard/cron | R governs restore. On a running box, matching loaded properties alone leave the old dashboard and indirect cron hazards above. |

## 9. Combined tests and mutation checks

- One focused pytest process ran the seven changed files plus heartbeat staleness, heartbeat churn, optional LightGBM collection and scheduler. **341 passed, 1 skipped** for missing local Python 3.11. The seven-file scratch control had **241 passed, 1 skipped**. [Focused log](focused-pytest.log), [scratch control](mutation-control.log).

| Scratch mutation | Outcome in all seven changed test files |
|---|---|
| Scheduler `check-clear` accepts `pending` | **Survived**. Existing CLI checks cover clear and malformed records. The independent probe exercised a valid pending record. |
| Remove writer exclusion during arm | Caught by the writer-before-arm contention test. |
| Release before verified activation | Caught by eight owner/evidence and boundary cases. |
| Skip `guard-stale` comparison | Caught by local mismatch and remote-only target cases. |
| Ping success with hold present | Caught by three monitor tests. |
| Skip the contest read for DM | Caught by refusal and both ET rollover cases. |
| Hook ignores incomplete failed record | Caught by the reconciliation evidence test. |
| Saver permits POST with unavailable guard | Caught by exact response check, 303 instead of 409. |

The mutation diff, status and failing test IDs are preserved in the wrapper log. [Mutation results](focused-and-mutations-wrapper.log)

## 10. Other diff observations

- Target-unit templates set `--no-sync` for scheduler, dashboard and lineup collector. Preflight validates scheduler/dashboard argv with the weaknesses above and does not verify lineup collector drift. Dashboard probe also refuses a nonzero Tailscale lookup before the documented loopback fallback, unlike the base, even when the listener responds locally. `scripts/install-deploy-guard-hetzner.sh:679–686`. CI's 3.11 step pins the interpreter by version and fails if coverage skips. The current runtime test covers clear and unreadable only, and its regular malformed-record branch attempts the installed guard path; it does not establish the live-owner overlay on a real 3.11 setup.

## What I ran

- Base archive and `bts.__file__` verification, 210 workflow owner cases and foreground shims, canary, preflight and hand release, Python 3.9 monitor comparison, cron shell replay, pinned dashboard differential and operator recovery witnesses. The shared wrapper ran `test_probe.py all` after 16:45 ET, with loopback-only test sockets and scratch mutations. Probe scripts and logs are beside this report. `git diff --check b128854d HEAD` and tracked status were clean. [Wrapper timing](focused-and-mutations-wrapper.log.watch)
