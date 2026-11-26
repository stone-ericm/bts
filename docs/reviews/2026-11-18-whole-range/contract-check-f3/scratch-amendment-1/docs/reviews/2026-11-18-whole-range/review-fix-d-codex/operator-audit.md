# Operator output and runbook wording

Inputs used in the review were an already delivered in-season pick, a send interrupted during activation, a private research box, a partial sync, a dated observation expiring at midnight ET, and a detached owner whose SSH follower lost the result. Output templates below use `<…>` for the values substituted by the code. The rendered cron rows, full state/result/evidence combinations, and before/after text are recorded in the linked review logs. [Report](report.md), [cron replay](cron.log), [owner matrix](owner-kill-hook.log), [runbook diff](operator-doc-diff.log), [literal recovery replay](operator.log)

| Output line or sentence | Literal consequence in the worst relevant state |
|---|---|
| `DEPLOY NOT STARTED — guard failed.` | Inspect installed guard and earlier run. A partial environment and old process can remain. |
| `DEPLOY NOT STARTED — target failed.` | No accepted owner. Verify target and prior run before retrying. |
| `DEPLOY NOT STARTED — phase payload invalid.` | Preserve original delivery and entry evidence. No owner reached mutation. |
| `Deploy run <id>.` | Record the ID when the follower disconnects. This is only creation of run metadata. |
| `Deploy outcome unknown (run <id>).` | A detached owner might be mid-send. No inference about admission or entry follows. |
| `Inspect the on-box result and hold read-only.` | Safe when the follower lost the result. Observe both the exact run and any current hold. |
| `Contact Eric before any retry.` | Prevents overlapping recovery or a false resend before reconciliation. |
| `DEPLOY NOT STARTED — abandoned deployment reconciled.` | Read old hold and stop/page evidence. The new run refused. |
| `DEPLOY NOT STARTED — guard-stale failed.` | Compare target and installed source. A prior incompatible failed activation may still need containment. |
| `Re-run scripts/install-deploy-guard-hetzner.sh as root from <target7> before deploying it.` | Provision the approved immutable revision. Installation alone does not stop the old dashboard. |
| `DEPLOY NOT STARTED — <step> failed.` | If busy, wait. If window refused, recheck timing and prior delivery/entry. |
| `This run did not change the checkout or services.` | Previous partial state may remain. Git metadata may have changed. |
| `Previous SHA: <value>.` | Diagnostic reference only. Reverting to that SHA is unauthorized by this line. |
| `DEPLOY NOT STARTED — hold-busy failed.` | Preserve the other owner's operation and inspect that ID before any repeat. |
| `DEPLOY FAILED at <step> — requested target <target7>, previous SHA <old7>.` | A partially reset checkout or prior send remains possible. |
| `No automatic rollback was attempted.` | Preserve existing state and fix forward through §5. |
| `Recovery has not been verified.` | A failure notice alone cannot settle a mid-send or authorize entry. |
| `Deployment hold: <verified\|unverified>.` | Inspect the durable record when unverified. No state change by the operator. |
| `Scheduler stopped: <verified\|unverified>.` | A verified stop cannot recall an accepted DM. Unverified may still be mid-send. |
| `Failure ping received HTTP 2xx: <verified\|unverified>.` | Verify human receipt separately and read the fallback path. Page evidence can precede the attempt. |
| `Activation failed after certified release.` | The hold was still present. The phrasing misstates the failure phase. |
| `Verify the new incident hold and any delivery/entry during activation before retrying.` | Verify original hold and both independent records before taking any new action. |
| `Recovery procedure: INCIDENT.md §4.` | Read first. The unqualified contest-state branch can write the wrong tree. |
| `Scheduler activation passed.` | The timestamp warning alone grants no restart during an active send. |
| `Calibration timestamp update failed (stamp write failed).` | The prior cutoff may remain. Preserve it for read-only inspection. |
| `Record the warning and follow §4 for calibration diagnostics.` | Check the existing file and record the warning. The send remains undisturbed. |
| `Deployed <sha>` | Read as deployment evidence only. The certified ET observation can already have expired. |
| `CANARY PASSED — <sha> is live and healthy` | Does not prove another pick can be computed or delivered. The rollover and malformed config problems make this misleading. |
| `bts-deploy-hold 1` | Protocol indicator. A half-installed tool still requires guard/preflight verification. |
| `no deploy hold` or `{"state":"clear"}` | Inspection only. Another action could arm after the check. |
| `unreadable` or `{"state":"unreadable"}` | Report and preserve the file. Do not treat absence of parsed data as no hold. |
| `deploy hold lookup failed` or `{"state":"lookup-failed"}` | A missing guard directory or unreadable lookup means unknown, not permission to retry. |
| `<state> at <step> (deploy <id>, target <target7>, previous <old7>, armed <ISO>)` or sorted full record JSON | Distinguish previous and intended SHA. A live-looking activation can refer to a dead owner in the competing-lock case. |
| `[<UTC>] bts-deploy-hold: <job> skipped: <held\|unreadable\|lock unavailable\|lock busy\|command unavailable>` | Keep the skip, inspect the cause and preserve chronological scoring. The wording grants no manual writer. |
| Guard argument usage/errors, including `expected an absolute command` | Correct the supplied form and recheck the window. Checking the scheduler gate has no stdout. |
| `hand release refused: invalid reason` | Supply a valid reason only after all original verification steps. |
| `hand release refused: failed hold required` | Preserve pending or activating records. No manual release. |
| `hand release refused: target mismatch; use a deploy` | Use a verified fix-forward target through §5. A partial checkout stays fenced. |
| `hand release refused: checkout differs; use a deploy` | Diagnose the tracked dirty checkout. Keep earlier receipts in place. |
| `hand release refused: the environment could not be verified as synced for <target7>; use a deploy` | A uv usage failure, missing uv or unsynced venv preserves the hold. |
| `hand release refused: <guard/preflight/canary/window step>` | Inspect the specific unmet prerequisite. Research and live checks remain distinct. |
| `hand release refused: guard or transaction unavailable` | Busy owner, broken lock, unreadable hold or audit error. Keep fences and evidence in place. |
| `hand release outcome unverified: inspect hold and release audit` | The unlink may already have happened. Verify result before retry or start. |
| `Deploy hold released by hand at <ISO> (deploy <id>, failed at <step>, target <target7>).` | An audit fact. Inspect already scheduled work and any existing delivery on the slate. |
| `Reason: <reason>.` | A recorded reason. It does not certify a pick or authorize a replay. |
| `Cron and boot activation are now permitted.` | Jobs can act after unlink. Prerequisite and date verification must already be done. |
| `This command did not request a scheduler start.` | Another caller can start it. Use a verified-window `start` only after rechecking. |
| Guard installer `<writers.lock\|transaction.lock\|state.lock\|bts-deploy-hold\|50-deploy-hold.conf> installed` or `… already current` | Verify effective loaded condition. A running base dashboard is still the base handler. |
| `FAILED: run the installer as root` | Refer provisioning to Eric using the approved source. No start. |
| `FAILED: invalid DESTDIR` | Correct the rehearsal root. Never switch to a live installation to complete a probe. |
| `FAILED: deploy guard installation or verification` | Stop the rollout and inspect which artifacts are incomplete. |
| `FAILED: installer source must be a regular file` | Use the reviewed regular installation artifact. |
| `FAILED: installer source size` | Verify the actual installation artifact and refuse a truncated copy. |
| `FAILED: installer source changed during installation` | Withhold deploy. Reverify source and loaded guard. |
| `FAILED: scheduler user manager needs reload or could not be queried` | The guard installation is not enough to admit production; fix provisioning and verify. |
| Cron required-var diagnostic `set HEALTHCHECKS_PING_URL (e.g. in .env, then: set -a && . ./.env && set +a) — refusing to bake a hardcoded ping URL into cron` | Supply the already approved monitoring configuration. Keep the endpoint out of receipts. |
| `ERROR: refusing to <install/show>: no valid [scheduler].entry_intent in <config> (see entry-intent above)` | Resolve the intended delivery mode before installing cron. A research machine should not receive entry cron. |
| `ERROR: refusing to <install/show>: unexpected entry-intent output: <value>` | Inspect CLI, configuration and environment. The run grants no change of intent. |
| `ERROR: cannot filter the current crontab` | Keep existing tab intact and inspect. |
| `ERROR: cannot read the current crontab: <stderr>` | A failed read is not an empty tab. Inspect permissions and preserve schedule. |
| `Install the deploy guard with scripts/install-deploy-guard-hetzner.sh` | Provision with Eric. A refused install leaves the old cron entries untouched. |
| `ERROR: <project>/.env not found.` | Restore the intended config by verified procedure. No permission to overwrite state. |
| `Installed BTS Hetzner cron jobs (entry_intent = <intent>).` | Inspect the loaded tab. A stale or separately managed row can still exist. |
| `Verify with: crontab -l` | Read loaded entries. Confirm masks and exact schedule without echoing credentials in evidence. |
| `Current BTS-HETZNER cron entries:` / the old rows or `(none)` | Read-only display. No implication for unmarked rows. |
| Empty line / `Entry intent: <intent> (check-pick-entered <installed\|left out>)` | The declared condition belongs to this render, while an old tab can still exist. |
| `Would install:` / exact schedule rows | Preview only. On a late-game or held slate, these are not direct commands to execute. [Rows checked](cron.log) |
| `Removed BTS Hetzner cron jobs.` | Confirmation of a removal. Preserve unrelated cron and verify monitoring before any planned downtime. |
| `Usage: <script> [install\|show\|remove]` | Read syntax. Never use `remove` to clear a deploy hold. |
| `deploy in progress (run <id>, armed <age>s ago); no ping` | A valid active run normally suppresses paging. A competing lock can falsely stand in for the dead owner. |
| `Deploy hold: <state> at <step> (requested target <target7>). See INCIDENT.md §4.` | Inspect the current run and evidence. A prior live delivery is still possible. |
| `Deploy hold present and unreadable. See INCIDENT.md §4.` | Preserve and inspect the unknown incident. No command to clear it. |
| `Deploy hold lookup failed. See INCIDENT.md §4.` | Diagnose a lost lock/lookup. This pages even before provisioning with no hold records. |
| `Deploy hold release record unreadable. See INCIDENT.md §4.` | Inspect durable release. Do not infer either success or permission to retry from corruption. |
| `Deploy hold released at <ISO> (<kind>); no scheduler heartbeat since. See INCIDENT.md §4.` | Could be sleeping after startup. Check generation and verify any delivery before intervention. |
| Alert body `Deploy failed at <step>, requested target <target7>, scheduler stop <verified\|unverified>. See INCIDENT.md §4.` | The failed activation can already have sent. Check independent state before retrying. |
| Legacy monitor `  (no --ping-url provided; alerts disabled)` | Diagnose liveness only. No claim of notification. |
| Legacy monitor `[<UTC>] stale=<bool>  reason=<reason>` | This is a liveness assessment, not proof of send or entry. |
| Legacy monitor `churn state write failed: <error>` | Investigate aux state. Original pick files remain outside this instruction. |
| Legacy monitor `churn check errored (continuing with liveness only): <error>` | Check churn separately. A success ping does not establish a successful pick. |
| Legacy monitor `ping failed: <error>` | Verify periodic/missed-ping fallback. Avoid sharing endpoint-bearing output. |
| `deploy canary <ET>: runtime unavailable` | Fix dependencies through a verified deploy. No live scheduler probe. |
| `deploy canary <ET>: configuration refused` | Verify configured delivery and effective paths. No entry action. |
| `deploy canary <ET>: research; contest state skipped` | Retain research mode. No authorization to switch to live or enter a pick. |
| `deploy canary <ET>: contest state refused` | After midnight confirm prior settlement and use the explicit live path. |
| `deploy canary <ET>: contest state unreadable` | Preserve and diagnose files. Do not replace evidence based on an unreadable response. |
| `deploy canary <ET>: live contest state accepted (<kind>, streak <n>)` | Check current ET date before relying on the result. No authorization to enter or resend. |
| Dashboard `deploy hold present: saver changes are refused` | Observe only. The first deploy's loaded base handler can still bypass this response. |
| Dashboard 408 timeout / 413 oversized / 404 unsupported / 303 redirect and existing saver result messages | Resolve the request in current UI only after verifying the hold and the account. Refusals grant no alternate state-write path. |
| Dashboard `BTS Dashboard running at http://<host>:3003` | Bind report only. It proves neither target readiness nor safe entry. |

| Changed runbook sentence | Literal consequence in the worst relevant state |
|---|---|
| INCIDENT §1 “The on-box owner arms a durable hold before resetting to the triggering SHA.” | Verify the phase when the result is unknown. An old unguarded workflow lacks this protection. |
| “It stops the scheduler, drains writers, syncs locked dependencies and checks candidate readiness and the dashboard.” | A failed stop or reload requires actual evidence. Old dashboard code can still write before restart. |
| “In enter mode the canary requires usable contest state for the sampled ET date.” | Check the effective path and date; a different working directory can disagree. |
| “The owner starts production and durably clears the hold after verification.” | Activation is an outward-live interval. Treat a lost receipt as unknown. |
| “Failure keeps the candidate checkout in place and closes the guarded jobs.” | Overstates pre-arm and incomplete containment. Inspect the actual result/hold first. |
| “Recovery is by verified fix forward.” | Preserve prior delivery and entry and verify the window before any retry. |
| §2 “Ship it by pushing to `deploy` after the checks in §5.” | Safe only once the actual payload retains the guard; §5 does not expressly require this for a revert. |
| “This is a full deploy of the selected `deploy` SHA, subject to §5.” | A rerun can change code and state admission; confirm the triggering SHA and window. |
| §4 “Deploy failed or its result is unknown.” | Treat as an incident if SSH disappeared mid-send. |
| “Read the Actions result, then use the run ID to inspect the on-box result and `bts-deploy-hold status` read-only.” | Gives operator a safe way to identify a detached transaction. |
| “Status exit 1 is expected for a hold.” | A held status can be unreadable. Check evidence rather than assuming safe closure. |
| “If the result or containment is uncertain, contact Eric before any retry.” | Avoids repeat deployment or entry on unknown results. |
| “Keep the hold and preserve pick, receipt, saver and streak files.” | Protects a delivered, private or mid-send record. |
| “Verify today's ET slate, the original Bluesky delivery and MLB entry read-only.” | Verify all three even with missing local acknowledgement. |
| “Use the dashboard for inspection.” | Do not submit saver on a page opened before arm. |
| “A missing receipt, preview or alert authorizes no entry or resend.” | No duplicate or incorrect entry after an interrupted activation. |
| “A delivery may already have been admitted during activation.” | Check Bluesky and MLB directly, even when the hold is failed. |
| “Read `bts season-contract status`.” | Uses `data/picks` without explicit path and can report the wrong state. |
| “On a dated day, verify MLB has settled the prior slate before entering today's observation.” | Good settlement check. Filing through unqualified commands writes the wrong tree when the config differs. |
| “Rerun only inside a verified window.” | Recheck date, prior entry and acceptance in the actual picks tree before spending the window. |
| “If delivery or entry is uncertain, or the window is too short, keep the hold and report to Eric.” | Can lose the slate by design instead of duplicating or damaging state. |
| “Repair dependencies through a deploy of the intended revision.” | Keeps partial env fenced. Compatible guard and operator timing still required. |
| “Check stop and paging evidence.” | Distinguish unverified stop and HTTP acceptance from receipt. |
| “A 2xx response records HTTP acceptance.” | No human receipt claim if communication fails. |
| “If checkout monitoring is unavailable, consult the rehearsed missed-ping path.” | Verify alternate signal after a partial reset. |
| “Confirm the intended revision, current ET state, prior delivery and MLB entry and the recovery window before running `bts-deploy-hold release --by-hand --reason "<reason>"`.” | Correct ordering for a mid-send recovery. Actual command may refuse missing 0.11.6 support. |
| “The command checks the clean checkout, loaded guard, installed cron, canary and offline locked uv check.” | Existing checks are weaker than prose for indirect cron and malformed config. |
| “Keep the hold if any prerequisite is uncertain.” | Preserves late-game evidence and avoids unknown activity. |
| “Cron and boot can run as soon as the hold clears.” | Finish verification before release when a tick or reboot is near. |
| “Recheck the window before using `systemctl --user start bts-scheduler` as `bts`.” | `start` avoids interrupting an already running activated scheduler. Recheck the sampled ET date as well. |
| “Record the warning and inspect `data/.last_deploy_iso` read-only.” | Leaves a mid-send and calibration evidence intact. |
| “The previous cutoff can remain in use.” | No reason to deploy again only to update the stamp. |
| “Review grading, reconciliation and observation gaps read-only.” | Preserves results and delivery evidence after a held day. |
| “Backfill in verified chronological order after Eric checks the dates, the applied results and the current streak.” | Coordinate exclusion of the active grader and reconcile. Temporal order alone cannot exclude concurrent writes. |
| §5 command comment “ship inside the verified window” | A queued Actions run can still reach a later window. Require actual host preflight. |
| “Eric verifies the intended revision, season state, configured delivery, prior send and MLB entry and a safe sleep window before shipping or retrying.” | Safe if the window covers the queue and operation and verification reads the proper account. |
| “A revert must serve the current season with today's state and installed jobs.” | Omits preserving the guarded workflow. The base payload replayed both resets during a hold. |
| “Commit `619abda7` cannot serve 2027.” | Removes that specific rollback choice. It does not rule out other pre-guard deploy workflows. |
| “For writer contention, wait for completion and verify a new window.” | No interruption of the in-progress long grader. |
| “For a refused window, recheck the current slate and prior delivery and entry before retrying.” | Avoids repeating a send or running across a game deadline. |
| “A rerun through `workflow_dispatch` performs a full deploy.” | Treat it as a deploy of the exact verified guarded SHA, not a safe dashboard restart. |
| “When the heartbeat is absent, stale or unreadable, Eric supplies `recover_sha`, the exact 40-hex target, after verifying prior delivery and entry and a safe window.” | Required for an unknown heartbeat. Humans still verify timing and outcome. |
| “That input authorizes only the specified SHA.” | No transitive authorization of a later branch revision. |
| §6 “A hold, a missing fresh heartbeat after release or missed pings can take this check down.” | Diagnostic only. Distinguish sleeping startup from incident by evidence. |
| “Box liveness remains independent.” | An up host can still be fenced or fail live readiness. |
| “The 08:30 day-outcome DM reports yesterday by its own evidence.” | Yesterday's delivery cannot authorize today's pick. |
| CLAUDE Deployment “The on-box owner arms `/var/lib/bts-deploy/hold`, stops the scheduler, drains writers, stages the tested SHA, checks readiness and verifies production activation.” | Treat it as the intended sequence; actual `stop` and certification still need evidence. |
| “Failure keeps the hold, stops the guarded jobs and attempts containment and paging.” | Overstates pre-arm failure. Check actual output, hold and scheduler. |
| “Recovery is by verified fix forward.” | Do not remove files or restore the October state. |
| “Before the first guarded deploy, run `bash scripts/install-systemd-hetzner.sh` as `bts` from the target revision and verify the user manager reloaded.” | Reload does not stop the already running dashboard. Require explicit verified old-process stop. |
| “The scheduler, dashboard and lineup collector unit templates now use `uv run --no-sync`.” | Read installed unit inventory. The old loaded process and actual venv can differ. |
| “Preflight refuses stale loaded units at `guard`.” | Detects some stale properties but not workdir, condition executable path or condition success override. |
| “Install the guard as root with `scripts/install-deploy-guard-hetzner.sh` from the target revision.” | Use reviewed source and verify the loaded unit before first arm. |
| “Re-run it whenever the target's installer bytes change.” | Install before deployment, under Eric's verified provisioning direction. |
| “A missing or mismatched fingerprint refuses at `guard-stale`.” | Exact target blob refusal. It has no containment effect on a prior unrelated process. |
| “Verify the loaded scheduler condition and guarded cron through checklist A7.” | Include indirectly managed jobs and the actual condition exit behavior. |
| “The 01:00 grading cron can run until about 06:00 ET.” | Do not assume scheduling time is completion time. |
| “A busy writer refuses deployment before arm.” | Keep the writer running and preserve its results. |
| “Wait for the writer and verify a new sleep window.” | Do not interrupt the grader or reuse the old window. |
| “Every push to `deploy` selects its tested triggering SHA.” | Verify the workflow itself and exact dispatch selection before permission. |
| “The owner resets to that SHA as `bts`, syncs locked dependencies, restarts the dashboard and verifies scheduler activation.” | Dashboard is not protected on first arm if it still serves the old handler. |
| “For an unavailable heartbeat Eric uses `recover_sha` through the recovery checks in INCIDENT §5.” | No bypass of the required current slate checks. |
| Season A5 “Restored-box execution follows R and B8a.” | Stay offline under R. The provisioned live procedure should not start the October checkout. |
| “Verify the root guard before installing guarded cron, and verify the current ET slate and window before starting the scheduler.” | Checks in correct order, provided R and actual configured state are verified. |
| Season A7 “Rehearse the deploy hold after R provisioning.” | Keep fixture rehearsals behind the R fence. |
| “Before the first guarded deploy, run `bash scripts/install-systemd-hetzner.sh` as `bts` from the target revision and verify its user-manager reload.” | Old dashboard stays writable. Add its verified stop before the first guarded arm. |
| “The scheduler, dashboard and lineup collector now use `uv run --no-sync`.” | Check effective installed state, not just files in the repo. |
| “Stale loaded units cause a `guard` refusal.” | A stale workdir can still pass today. Verify the full effective unit in rehearsal. |
| “Install the guard as root from that target and reinstall whenever the target's installer bytes change.” | Use the reviewed installation copy and do it before issuing deploy. |
| “A missing or mismatched fingerprint causes a `guard-stale` refusal.” | Refusal alone cannot prove containment of an abandoned activation. |
| “Verify root ownership, effective loaded condition and systemd 252 start/restart/dependency skip behavior, completion hook and cleanup, uv 0.11.6 offline verification and the system 3.11 monitor with a test hold.” | Use only isolated rehearsal fixtures. Record what actually started/stopped. |
| “Exercise explicit failures and missed pings with distinct test checks.” | Does not disturb live paging. |
| “Verify Eric receives GitHub, scheduler heartbeat and box-liveness notifications and record grace, repeat settings and receipt times.” | Checks receipt rather than interpreting 2xx as delivery. |
| Season B4 “After cron install, verify the one `day-outcome` line.” | Inventory only. No implication for yesterday's settled delivery. |
| “Rehearse failure with an isolated temporary picks tree that has no decision file and a designated test notification route.” | Does not move a live `decision.json` or generate a live notification. |
| Season B8a “Restored-box execution suspended pending R.” | Keep restored research box from running old checkout. |
| “Complete R before first normal boot or live activation of snapshot `446156433`.” | No activation before the offline fence. |
| “A live-mode deploy refuses when today's contest observation is unusable.” | Wrong workdir or an ET rollover makes this claim unreliable. |
| “With DM delivery and neither setting, `bts schedule` exits 2 before its first heartbeat and says why on stderr only: the deploy canary refuses the live configuration.” | Fix the effective live gate while fenced. A log-only refusal confers no permission to enter. |
| Season B8b “W12/W13 operate on this box only after D is shipped and verified with the guard and approved cron.” | Conditional status. No risk from acting literally if the verified guard/provisioning covers the old dashboard and indirect jobs. |
| “A failed deploy holds the scheduler and guarded jobs, attempts containment and paging, and fixes forward.” | Overstates pre-arm refusal and blocked hook. Inspect result and containment independently. |
| “`619abda7` cannot serve 2027.” | Preserve current state and avoid that rollback. |
| Season B8c “Restored-box execution suspended pending R.” | Do not run October code. |
| “Snapshot `446156433` requires R's offline first-boot fence and approved provisioning before normal boot or any deploy.” | Correct scope for first boot, before cron and linger. |
| “Before the first guarded deploy, run `bash scripts/install-systemd-hetzner.sh` as `bts` from the target revision and verify its user-manager reload.” | Same old loaded dashboard gap as A7 and CLAUDE. |
| “The scheduler, dashboard and lineup collector now use `uv run --no-sync`.” | Verify effective flags and that dependencies are already synced. |
| “Stale loaded units cause a `guard` refusal.” | Current parser does not check several material effective properties. |
| “Install the guard as root from that target and reinstall whenever the target's installer bytes change.” | Trusted target identity matters before running the provisioning script. |
| “A missing or mismatched fingerprint causes a `guard-stale` refusal.” | Refusal does not stop a prior failed activation. |
| “Verify current guarded cron before shipping.” | Inventory indirect and quoted commands explicitly. |
| “Do not use October checkout commands to initialize 2027 state.” | Preserves season tags and current contest evidence. |
| “On an already running box the new deploy refuses an unverified guard or ungated writer entries.” | False for accepted quoted, indirect and wrong effective guard cases. |
