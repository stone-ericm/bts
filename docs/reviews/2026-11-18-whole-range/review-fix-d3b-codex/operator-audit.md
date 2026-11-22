# Operator text review

This audit covers statements added or changed against `24019a9a`. Timing, scheduler and HTTP observations in the review came from redirected shim boxes. Commands below identify instructions under review.

| Location and sentence | Worst literal outcome or evidence |
|---|---|
| Workflow `Pre-deploy SHA` | This is the locked preflight read. A later partial reset can change the checkout. |
| Workflow `DEPLOY NOT STARTED — … failed.` | Identifies this invocation. A prior owner may have changed the box. |
| Workflow `This deploy invocation did not change the checkout or services.` | Refers to the worktree and services. Git config, fetched metadata and lock files can change in preflight. |
| Workflow `Previous SHA` | Prints the preflight observation or `unknown` if the read did not occur. |
| Workflow `Recovery has not been verified.` | Requires independent recovery evidence. A matching `begun` can survive an inconclusive publication. |
| Workflow `Checking requested target` | Post-restart HEAD was read and matched. |
| Workflow `Waiting 30s` | A progress line before the bounded sleep. Termination may prevent the full wait. |
| Workflow `completed cleanly and is waiting for relaunch` | Describes the checked systemd tuple. The scheduler can fail after that sample. |
| CLI and workflow canary accepted line | Some valid contest state existed for the sampled ET date. Pick delivery and entry can still fail. |
| CLI and workflow canary non-live line | No live contest-state loader call occurred. The printed mode follows the scheduler resolver. |
| CLI and workflow canary refusal and error lines | A refusal can include a command template. Operator must substitute observed MLB values for today's date and use the correct picks directory. |
| Workflow `Check … failed` | The probe logs the command failure, its normalized status, or malformed evidence. |
| Workflow `DEPLOY FAILED … requested target … previous SHA` | Neither SHA names the reached tree. Reset can leave a partial checkout. |
| Workflow `No automatic rollback was attempted.` | No reset to the old SHA occurs in any reviewed failure branch. |
| Workflow `Recovery has not been verified.` in initial failure | Printed before the stop, marker and POST attempts. |
| Workflow `Scheduler stopped` | Scope is two fixed names, their observed groups, the local link and the residual match. The two incomplete legacy branches can print a false positive. |
| Workflow `Failure heartbeat written` | Atomic replacement and read-back confirm a snapshot. A later writer can replace it. |
| Workflow `Failure ping received HTTP 2xx` | Can reuse the same attempt's recorded 2xx. Dot paths can obtain that 2xx from a different path. Provider delivery remains separate. |
| Workflow `Recovery procedure` | §4 preserves evidence and directs hand verification. |
| Workflow `containment re-checked` | Fresh stop and marker plus a fresh or recorded POST status. A failed record rewrite can leave old fields on disk. |
| Workflow recovery invalid inputs, lock unavailable, unreadable record | No service mutation. An earlier scheduler can still run. |
| Workflow `cancellation mark could not be verified` | Late SSH can still claim the attempt. Operator needs §4. |
| Workflow `No service change was attempted` | Describes this recovery invocation after cancellation-mark failure. |
| Workflow `no phase-2 record is present` | A reboot can remove the record of an actual deploy. |
| Workflow `Check the deploy log and the host's boot time` | Read-only investigation of the missing record. |
| Workflow `a different attempt owns the box record` | The matching new mark fences only the obsolete request. |
| Workflow `this attempt's record says committed` | Can accompany an actual contained box after a failed rewrite. The wording describes the surviving record. |
| Workflow `Check the deploy log and verify current state` | Prevents reading an old commit as current service health. |
| Workflow `Deployed` and `CANARY PASSED` | Checked run-local status before a later deploy or external change. The commit can survive a lost log. |
| POST `Deploy failed at … requested target … scheduler stop …` | Describes current containment's stop determination. The legacy proof can make its strongest claim falsely. |
| POST `See INCIDENT.md §4` | Recovery procedure points to read-only triage before retry. |
| Heartbeat `Deploy failed at … requested target …` | Stored marker can outlive a failed attempt. Fields are allowlisted and time is irrelevant for this state. |
| INCIDENT §1 preflight, soft hold, reset, sync, restart and wait | Warn that old or candidate scheduler can run through the soft hold. See L5/L8/L9. |
| INCIDENT §1 canary checks four inputs | Can confirm a useful state snapshot while a subsequent delivery still fails. |
| INCIDENT §1 attempts containment after authorization | Prior owner may still be running after refusal. |
| INCIDENT §1 recovery eligibility | Eligibility alone leaves reachability and execution unresolved. |
| INCIDENT §1 verify Actions and follow §4 | Read-only and routes uncertain results to the operator. |
| INCIDENT §1 PASS records the checks | Snapshot evidence only, including several successive observations. |
| INCIDENT §1 verify Bluesky and MLB separately | Preserves evidence and prevents acting on a preview. |
| INCIDENT §2 ship approved fix and revert via §5 | Both trigger scheduler restart. Ship the approved SHA in the reviewed window. |
| INCIDENT §2 re-run by selecting `deploy` | Selecting current ref can ship a different SHA. Read with §4 and the explicit same-SHA rule in §5. |
| INCIDENT §2 dispatch uses the ref's current SHA | Read-before-dispatch is a snapshot. Retrying the original run fixes the original SHA where eligible. |
| INCIDENT §2 read both logs and the Actions result | Recovery can be missing even if the deploy command touched services. |
| INCIDENT §2 failure can precede canary | Look for preflight, reset, sync, restart, or lost session evidence. |
| INCIDENT §4 read Action and deploy/recovery logs | No state-changing instructions. |
| INCIDENT §4 not-started describes its invocation | Prior runs can have changed the box. |
| INCIDENT §4 read recovery separately, verify results | A retained page result describes earlier HTTP evidence. |
| INCIDENT §4 box may have committed | Read-back/transport errors are independent of current scheduler state. |
| INCIDENT §4 scope of verified stop | Snapshot and fixed scopes, subject to the legacy false positives. |
| INCIDENT §4 independently started services, argv and UIDs | Remain possible and outside the stated residual check. |
| INCIDENT §4 commit can survive failed containment rewrite | Reproduced with commit read 126 and contained rename 17. |
| INCIDENT §4 disagreement or missing logs go to Eric | Check actual state before a retry can start the scheduler. |
| INCIDENT §4 absent/unverified result or cancellation goes to Eric | Keeps the system in read-only triage. |
| INCIDENT §4 Eric checks both scopes and boot link | Matches the manual check required after an unavailable result. |
| INCIDENT §4 old or candidate scheduler can still run | True under L2/L8/L9 and in the incomplete stop-proof cases. |
| INCIDENT §4 preserve pick, receipt, saver and streak | No evidence lost from these instructions. |
| INCIDENT §4 verify ET slate, Bluesky and MLB read-only | Prevents write side effects from a false assumption. |
| INCIDENT §4 leave saver/MLB controls untouched, no preview entry or resend | Avoids changing an existing pick or creating a second delivery. |
| INCIDENT §4 uncertainty keeps stopped and routes to Eric | Requires a known stop first. |
| INCIDENT §4 apply checks and approved window to dashboard fault | Redeploy restarts the scheduler too. |
| INCIDENT §4 dashboard is for inspection | No saver controls or preview entry during recovery. |
| INCIDENT §4 added dashboard restart sentence | Routes the unconditional restart summary through the incident and checked-deploy rules. |
| INCIDENT §4 restore keep scheduler stopped and link disabled | The instruction follows the restic restore and data sync commands. On an already-running rebuilt box, a reader who executes in order can replace active operational files before checking the hold. Move the sentence before the first restore action. |
| INCIDENT §4 installer copies and reloads | Installer code only copies/reloads. Its unchanged closing hint suggests a manual restart, which the next sentence overrides. |
| INCIDENT §4 activate only through §5 | No instruction to call that manual restart. |
| INCIDENT §5 normal ship command and gate sentence | Starts scheduler after the test gate if approved. Prior recorded candidate state can change during the canary. |
| INCIDENT §5 route failure/cancellation to §4 | Read evidence before another restart. |
| INCIDENT §5 leaves reached checkout | Can be partial after reset failure. |
| INCIDENT §5 emergency true-merge eligibility | Ruleset rejects the described non-fast-forward push. Review evidence is static repo text. |
| INCIDENT §5 older commit needs approval and rehearsal | Rehearse with current files, dependencies and cron before a reset that could use old semantics. |
| INCIDENT §5 forbid pre-2027 code on a 2027 box | Explicit guard against unsupported scheduling and rewriting old-format files. |
| INCIDENT §5 prefer a revert commit | Still goes through gate, scheduler restart and canary. |
| INCIDENT §5 fix forward on code failure | Tested fix can still activate in a live window, so apply §4. |
| INCIDENT §5 state refusal uses current ET and MLB data under B8a | Dated correction can affect a later lineup check, so use correct path and actual values. |
| INCIDENT §5 require `live decision now: accepted` | Confirms the current input to the loader, distinct from a sent pick. |
| INCIDENT §5 approved same-SHA retry | Requires remaining lineup/submission window. |
| INCIDENT §5 use original rerun or checked dispatch | Rerun preserves SHA. A ref can move between reading and dispatch, acknowledged below. |
| INCIDENT §5 ET date changed, verify again | Prevents reusing an expired observation. |
| INCIDENT §5 no manual retry restart | Preserves checked workflow activation. |
| INCIDENT §5 no useful window, stop and record missed day | Avoids a restart that could duplicate or miss a pick. |
| INCIDENT §5 backfill only with Eric in order | Preserves evidence and avoids out-of-order streak changes. |
| INCIDENT §5 moved branch can ship another SHA | Makes the dispatch race explicit. |
| INCIDENT §6 GitHub notice depends on result and settings | No reliable page for an unreachable or cancelled action. |
| INCIDENT §6 preflight/cancelled/SSH failure can leave old scheduler running | Consistent with L2. |
| INCIDENT §6 resend if previous response unverified | Better said as “no matching recorded verified response.” Prior 2xx can be lost with the attempted record rewrite. |
| INCIDENT §6 2xx is response only | True for the destination curl used, which can differ from the literal path. |
| INCIDENT §6 check provider and Eric's receipt | Separate evidence checks. |
| INCIDENT §6 heartbeat cron cadence and stock interpreter | Depends on installed D3 line, reached tree and usable environment. |
| INCIDENT §6 liveness miss, heartbeat miss, schedule/grace/integration | Correct separation of provider checks. |
| INCIDENT §6 cron and timers continue | They can overwrite information in the reached tree during failed deploy settlement. |
| INCIDENT §6 scoring, reconcile, four fetch times and possible writes | Leaves known state-mutating jobs running. |
| INCIDENT §6 data build/preview, park drag, shadow, collectors, analysis | Dashboard and recommendations can be stale or differ from live state. |
| INCIDENT §6 ops/weekly backups and prune | Backup can capture partial state. Restore rules require the checked snapshot. |
| INCIDENT §6 entry, outcome and fetch can DM | A cron from October can send health/entry-related DMs even if research is set. |
| INCIDENT §6 these jobs send no pick | The scheduler itself can still send a pick during the canary or via L8. |
| CLAUDE remote and phone incident note | Separates cloud evidence from SSH access. Cookie capture requires authorized box access. |
| CLAUDE canary/containment workflow sentences | Same limits and snapshot evidence as INCIDENT §§1 and 4. |
| CLAUDE emergency dropdown, SHA, environment and key | Branch check and tested SHA matter to the proposed operation. Environment settings and rotation were only read from local material. |
| CLAUDE recover through §4 and whole deploy, avoid manual restart | Avoids unsupported activation. |
| CLAUDE every push, tested SHA, soft hold and no extra restart | A push starts the configured workflow. The local boot link does not stop a running process. |
| CLAUDE L1–L3 | October tasks can corrupt legacy-formatted state, missed/out-of-order evidence is possible, and ownership can be lost. These are accepted limits. |
| CLAUDE L4–L6 | Paging may fail and a candidate can send before the canary. A late heartbeat can overwrite the marker or ping after failure. |
| CLAUDE L7–L9 | Frozen parser does not recognize new lines. Independent start paths remain. Reboot can erase ownership and allow late SSH. |
| CLAUDE L10 | “Recovery reports a matching committed attempt” can sound like a current-state claim. Prefer the workflow's “record says committed.” |
| CLAUDE L11–L14 | Runtime divergence, HTTP-vs-receipt, legacy snapshot, and D-Bus NUL limits bear their scoped meanings. |
| CLAUDE stop-proof sentences | Structural claim holds. Scope and pruning caveat stand, subject to the incomplete legacy verdict. |
| CLAUDE 2026 wrap activation | Routes via B8a and the approved window. |
| A5 defer all edits into B8a | Prevents live delivery config from preceding the stopped/disabled check. |
| A7 rehearsal scope and two provider checks | Runs only under research/private or a suitable rehearsal host. A real installation still requires verification. |
| A7 canary failure, cancellation, short timeout and transport loss | Can strand a research scheduler and can consume the soft-hold window. Final §4 and approved restore are necessary. |
| A7 owner/PID/link/order and Actions outcome/predicate/recovery reachability | Preserves observations separately and explicitly records unavailable evidence. |
| A7 check named scopes, exact disabled output/status, legacy absence and runtime/heartbeat path | Reduces false confirmation. The actual Debian behavior remains to rehearse. |
| A7 check liveness, response, provider history and receipt | The immediate response alone does not establish delivery. |
| A7 temporarily move venv, isolate races, verify real cron, restore venv | Existing `uv run` tasks can recreate it. Restoring before the next deploy prevents a false follow-up failure. |
| A7 exercise missed pings, periods, grace, receipt and GitHub notice | Keeps independent channel failures visible. |
| A7 uncertain containment goes to Eric and SSH, finish on approved SHA with PASS/stamp/link | Prevents treating rehearsal timeout as containment. The final deploy still restarts the rehearsal scheduler. |
| A7 keep secrets out of evidence | Required when recording URLs and logs. |
| B8a complete B8c first | Installs current checkout and cron in research before editing delivery. |
| B8a confirm stopped/link disabled and retain that state | Guards initial config and cron changes against an old boot start. |
| B8a default vs configured picks directory | Status, revoke, enable, fetch, dated writes and saver reads/writes stay in the intended path. |
| B8a October revoke, today's accepted dated state and same SHA | Can archive old manual/saver evidence before activation. The dated state must be checked before deployment. |
| B8a repeat the stop/link/deploy sequence at automatic cutover | Covers the separate future state change. |
| B8a flags, DM and enter config | Activates live intent after the soft hold checks. |
| B8a no flags refuses before first heartbeat, restart failure vs canary refusal | Scheduler can be restarted during canary, so refusal and containment differ. |
| B8a October revoke and status | Captures resulting switch state before activation. Use configured path per lead sentence. |
| B8a entry-intent check, source .env and show/install cron | Installs the entry and reconcile schedules. A cron can write against the shared files before scheduler restart. |
| B8a ensure stop/disable, let fetch finish, enable and fetch/read accepted state | Live data may change between read and fetch. Stop and check active season before proceeding. |
| B8a failed fetch keep dated observations and checked same-SHA deploy | Avoids enabling automation with an unaccepted reading and preserves approved activation path. |
| B8a saver show and guarded init with explicit path and ET year | Avoids writing the default path or older season. Read MLB state first. |
| B8a confirm live decision then deploy approved same SHA | Uses the intended input and approved activation window. |
| B8b(3) reached checkout and containment/paging attempt | A failed or cancelled operation may leave partial state or no page. |
| B8b(3) verify and recover under §4, prohibit old code | Protects evidence and does not authorize another restart alone. |
| B8c enter rescue or controlled maintenance before October execution | Requirement starts before the scheduler or old cron may touch evidence. |
| B8c save/verify crontab then set aside and keep services stopped | Before switching from rescue to normal boot, make explicit how user boot starts remain suppressed. A stop in rescue alone does not persist. |
| B8c persistent timer catch-up and Eric review if old jobs ran | Timers can update files even with old cron removed. Review evidence before activation. |
| B8c research/private and both flags before approved deploy | Activates only the approved code in research through the workflow. |
| B8c verify PASS and HEAD before cron install from current tree | Avoids running October cron against a 2027 checkout or vice versa. |
| B8c keep old table set aside, use §4 and do A7 then B8a | Preserves crontab backup and defers live changes to the approved sequence. |
