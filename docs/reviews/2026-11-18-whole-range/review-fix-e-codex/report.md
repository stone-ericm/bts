# Verdict

**BLOCK** on `db52bd02` against `63669c42`.

## Blocking findings

- **B1 — The morning instruction can install pre-settlement state.** At 00:20 ET on July 15 with the July 14 entry live, the stale automatic status was 13 and the account could still show 14. The exact morning command accepted 14. Once the prior entry settled as a miss, automatic 0 lost to that dated 14. The real lineup check at 82% and 81% changed from double to single. Replace the conditional command sentence in [scheduler.py:207](../../src/bts/scheduler.py#L207) with “If prior entries are settled or absent and MLB differs from status (B8a), use bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR on {D} ET only.” Apply the same settlement condition to the morning branch of the stale paragraph in [B8a](../../docs/ops/2027-season-start.md#L31). `operators-settlement.log` records the status, dated write and both selections.
- **B2 — Bad saved data renders impossible probabilities.** A decoded `best_p=1e308, bar=1e307` produced “inf%, at or above the streak-10 bar (inf%)” in the shared body, DM and dashboard. The parser accepted both as finite and percentage formatting overflowed. Reject values outside [0, 1] in [scheduler.py:1750](../../src/bts/scheduler.py#L1750) and let the existing unknown-probability and no-comparison wording apply. `rendering.log` contains two persisted examples.
- **B3 — A5 lists a shortcut past the deploy and revocation gates.** On a restored October checkout, following “steps 2 and 5–7 in order” reaches DM/enter without step 3's deploy or step 4's revocation. The archived October `entry-intent` printed `enter`, and its installer enabled the entry checker with the 08:30 `day-outcome` job absent. B8d refers first-time activation back to the full B8c sequence. Replace the first instruction in [A5](../../docs/ops/2027-season-start.md#L14) with “On the restored box follow B8c steps 1–7 in order, with B8d governing step 5.” See `ops-october.log` and `operators-cli.log`.
- **B4 — A reviewed checkout can deploy a different commit.** From a branch checked out at the reviewed commit, the literal `git push origin main:deploy` pushed the stale local `main`. The workflow tests and canaries the pushed commit, so verifying the resulting tested SHA against box HEAD accepts the wrong revision. Replace the push/dispatch sentence in [B8c step 3](../../docs/ops/2027-season-start.md#L33) with “Confirm `git rev-parse main` equals the approved SHA, then run `git push origin main:deploy`. If `deploy` already points to the approved SHA, dispatch that ref.” See the local Git push in `ops-refs.log` and the shell trace in `ops-deploy.log`.
- **B5 — The scheduler hold can disappear across reboot.** With scheduler enablement under `default.target.wants`, step 3 restores the scheduler file and its enablement link resolves again. If step 4 pauses in December after `stop`, a reboot with a lingering user manager starts the scheduler before revocation. Add to the pause sentence in [B8c step 4](../../docs/ops/2027-season-start.md#L33) “If pausing for 2027, mask the stopped scheduler through any reboot. Restore its reviewed unit and reload the user manager when resuming, and leave it stopped until step 7.” The target links in `ops-refs.log` reproduce the enablement state with the shipped unit.
- **B6 — The later-day status command can read another account-state directory.** In a second checkout, literal `bts season-contract status` printed accepted automatic 14 while the scheduler's path printed 7. An operator whose MLB screen also shows 14 skips the needed correction in the live directory. Replace the status sentence in [B8b(1)](../../docs/ops/2027-season-start.md#L32) with “After verifying settlement or no prior entry in MLB, run bts season-contract status --picks-dir DIR using the scheduler’s absolute picks path, then file any needed dated observation before the first lineup check.” Apply the explicit path to B8a's standalone saver read. See `handoffs.log`.
- **B7 — A service restart via dispatch can redeploy the failed SHA.** After a rollback, box HEAD is the old SHA and `deploy` can still point to the failed target. Literal dispatch under [INCIDENT §2](../../INCIDENT.md#L50) “with no code change” or [§5](../../INCIDENT.md#L186) “Just restart services” installs the target again. In both sections use “For a restart only, have the on-box operator restart the required units during the §2 quiet window. Use dispatch when you intend to apply the current `deploy` SHA.” The shell ran with October HEAD and a different target and restarted at the target in `ops-deploy.log`.
- **B8 — The cookie location contradicts the runnable recovery step.** The [INCIDENT §3 table](../../INCIDENT.md#L79) calls capture “on-box.” The new instructions and real auth-error CLI output say Mac. An operator SSHing to the VPS misses the required Mac login and capture. Replace the reason cell with “Interactive on Eric's Mac. Copy the resulting cookie file to the box.” `operators-cli.log` shows the exact auth, rate-limit and transient outputs.

## Notes

- The source comparison ignores the expected message-derived body hash. The warning/refusal status paths, keys, sent days and substantive data matched. The original loader raises on a directory before the renderer gets a summary. The dashboard loader maps that path to an empty state.
- The B8e fixture removed leaderboard timer enablement and modeled both masks. If boot already happened, stopping the timer leaves its enablement link. Apply the same masks before another boot. The actual snapshot and unit manager were outside the offline fixture.
- The tests named “legacy summary keeps old wording” assert the new wording. Their old names can confuse readers. The existing assertions and fixtures were unchanged outside the four allowed files and the requested walk test addition. The only non-text assertion change was the authorized `Steven Kwan` to `B1` candidate.

## 1. Exact text and operator outcomes

`probe_operators.py` matched the composed contract character for character on five real skip decisions, both warning envelopes and both refusals. The delivery journal matched its DM body, with its original prefix. The morning journal kept the stale detail only. The longest tested accepted timestamp gave 881 and 842 characters including the envelopes.

| Sentence or row | Worst ordinary result when followed |
| --- | --- |
| Skip DM header and dashboard heading | A late or final check still speaks about that check. It offers no follow-up pick. |
| K1 candidate. K1 reason. | Terminal tail stop at 0/18 with eight days left. Candidate is the selected row. The reason explains the skip. No pick entry follows. |
| K2 comparison | 99.9% removed by game status. Actual 75% was below 79.6%. At nearly equal percentages the phrase “using unrounded values” explains the result. |
| K3 comparison | 99% exceeded 82.5% on Aug 30. Exact equality also gives K3. Neither invites an entry. |
| K4 candidate. K4 no comparison. | All bins skip at streak 56 on July 15. A missing probability also makes comparison unavailable. See B2 for impossible stored percentages. |
| K5 no candidate. K5 top prediction. | With both games ineligible, the 99.9% prediction is explicitly unverified. The text supplies no action. |
| Shared streak suffix | Tells which streak the completed check used. None, negative, Boolean and wrong-type streaks omit it. |
| Delivery S1 and morning M1 stale reading. M2 eligibility. | Captured old/future/unreadable warnings. A confirmed stale automatic delivery remains delivered. The morning reading can still be used while it remains current. |
| S2/M3 check MLB. S3/M4 compare and write. DIR sentence. | Absolute DIR works from elsewhere. Post-send July 15 correction preserved the pick, entry check, grade and 08:30 result while clearing historical coverage alerts. Morning can write too early (B1). Omitted DIR goes to the wrong checkout (B6). |
| S3/M4 sent-pick and saver sentences. | A dated write leaves both intact. Between selection and guard a changed observation can refuse the pending delivery. During the transport it does not stop the send. |
| S4 same-slate/date sentence. S4 B8b handoff. | After midnight the old literal date exits 2. A delivery warning emitted at 00:05 prints the new date. The slate check prevents copying that date into a prior-slate correction. Wait for settlement before any later-day update. |
| S6/M6 read saver. Mark active/not_earned. Uninitialized reference. | Both writable states became used with and without a dated observation. Used was already recorded. Uninitialized exited 1 on `--use` and needs current-year init. A dated saver still wins for that day. Incorrectly marking used at 14 and 79% changed single to skip. Jan 1 operations default to the new year. A stale explicit old-year init can overwrite the current file. |
| S5/M5 fetch-status sentence. | Read-only. B8a identifies the sibling health-state path. |
| L1 late header and P1 void header. Both withheld sentences. | Three minutes past cutoff or a postponed game. Pick removed, archived and no transport for it. |
| Both later recommendation sentences. Both entry-window sentences. | Later eligible game can yield a new pick, and absent a later check nothing arrives. The sentence restricts entry to a later official message in its open window. |
| Both manual-entry sentences. Both possible-mismatch sentences. | In-window verification can report a conflicting manual entry. An absent or failed check can leave it unresolved. Neither sentence asks for a manual entry. |
| A4 authentication evidence. Offline rehearsal. | Pre-enable ledger and post-enable observation are different files. Fixture entry check needs no live account. A real receipt needs an authorized window. |
| A5 first activation. Later switch. Cron and environment. | See B3. B8d handles later activation. Installer includes entry check only for enter and includes 07:40 for either mode. Without exported ping URL it exits 1. |
| A7 know SHA/config. Overnight stop. Set pair. Install and restart. | Quiet window protects lineup, send, grade and entry jobs from interruption. Mixed pair refused with crontab untouched. Private/research keeps four reads and 08:30 and removes entry cron and fetch failure DMs. |
| A7 already sent. 08:30 result. Switch-back. | A prior scoreable delivery stays delivered. Real 08:30 returned `delivered_under_research`, exit 1. Follow B8d for reactivation. |
| B4 verify cron. Run on copy. Choose played slate. Remove copy decision. Run day-outcome. Verify and clean up. | During an entry window the copy leaves live decision and receipt binding intact. Played copy gave `no_decision`, exit 1 and the captured DM path. The off-day control returned `no_games`, exit 0. |
| B8 replacement. B8a flags and order. | Accepted dated/automatic state is read before selection. DM/enter without flags is refused. October private/research stays quiet for pick delivery during deploy. |
| B8a daily settlement, placeholders and first check. Missing reading. Pending saved pick. Committed pick. | Dated CLI rejects a different ET date. Settled account, absolute DIR and accepted values matter. New decision refuses absent required state. A change before the send guard can refuse a pending pick. A committed or attempted pick stays in place. |
| B8a startup margin. Silence. No restart. | No startup check inside ten minutes or on an empty slate. Silence is inconclusive. Restarting during work risks a second cascade or unconfirmed send, so the sentence calls for leaving it alone. |
| B8a ledger before enable. Compare. Cutover window. Four CLI steps and acceptance. Failure handling. | Enable alone does not install a reading. No overlapping fetch during cutover. Stop relying on automatic until status accepts it and values match MLB. A dated observation remains available on failure. |
| B8a saver reads/init/race. B8a stale paragraph. | Current-year check and explicit DIR protect the right saver file. A concurrent init requires reread. Date-keyed warn offers no assurance it arrives. Apply B1/B6 where the operating instructions diverge. |
| B8b(1) alert time and no command. No correction from alert. Settlement/status and quiet morning. | Prior slate can still be live after midnight. Wait for settled/no entry, then check current state. Bare status can read the wrong directory (B6). No DM is compatible with no check or healthy state. |
| B8b(3) workflow hold, HEAD, failure, verification, October exclusion. | Rollback attempts the saved SHA. Sync/restart can leave target live. Even “Rolled back cleanly” appeared with a failed reset. Verify each state before retrying. |
| B8c restore facts, day choice and game-day hold. | A game-day scheduler with no reading refused and paged. With a valid reading it privately locked a pick. The no-game control never ran selection. Keep the hold on a game day. |
| B8c step 1 rescue, cron, masks, other timer hold, boot verification, old writes. | The literal paths can hold the known user jobs and empty cron. An already launched leaderboard service needs stopping independently of the timer. Inspect the state from an unfenced boot. |
| B8c step 2 settings, private/research and verification. | October can start after a failed first deploy, with public pick delivery still held. Both settings prepare the incoming checkout. |
| B8c step 3 restore/reload, push, passed canary, both units, no-game startup, failure steps. | Pass starts both units. Sync/restart failure may leave target installed, rollback reset can fail, and a passed canary can be followed by a failed job. Keep cron held, inspect actual HEAD and stop both on failure. B4/B7 distinguish reviewed, deployed and live SHAs. |
| B8c step 4 stop, revoke, status, preserve, wait. | Real revoke exited 2 in December and succeeded on Jan 1. It reports the scoped state. Stopping alone does not preserve the hold over reboot (B5). |
| B8c steps 5–7 switch, intent, export, show, install, verify, start, final warning. | Valid intent exits 0 and matching cron installs from empty. The order avoids restarting the scheduler before cron validation. Failed October deployment keeps cron empty. |
| B8d private lock. First and later switch. Verify evidence, stop, hand off. | Existing private lock survived DM/enter. The entry CLI named it in a mismatch DM at 18:45 and sent nothing at 19:05, the cutoff. New-day and no-activity checks prevent this carryover when followed. A switch discovered too late calls for preserving evidence and ignoring entry advice for the lock. |
| B8e account, persistent catch-up, installers, July history, hold, inspect, authority. | Catch-up can start the login before an operator connects. Both masks and removing the timer link prevent launch. The standard deploy leaves leaderboard alone. The dedicated installer calls `enable --now` and is reserved for Eric's approved reactivation. |
| INCIDENT §1 resets, restarts, canary criteria, failure limitations, verify, October bound. | Shims accepted HTTP 204 and clean scheduler auto-restart, and exercised early and post-canary failure. Canary cannot certify live picks. First restore follows B8c. |
| INCIDENT §2 quiet window, first restore, both restarts, read logs and verify. | Deploy while a lineup, send or grade is active restarts the scheduler, and deploying while the entry checker runs permits overlapping actions. Verify quiet and on-box state. Dispatch can change live code (B7). |
| INCIDENT §3 failure diagnosis, rate limit, transient, expired login, ambiguous shape, example. | Real 429 asks for backoff and no capture. Transient says refresh is not indicated. Auth error names Mac. Escalating ambiguous errors preserves evidence. The table sends capture to the wrong place (B8). |
| INCIDENT §4 deploy triage, dashboard restart and snapshot restore. | Read the attempted target, verify actual state before retrying. On-box dashboard-only restart leaves scheduler running. Standard unit installation under holds reloads units; B8c controls the restored boot. |
| INCIDENT §5 normal ship, compatible rollback, forward fix, guard retention, dispatch. | Quiet-window ship stays clear of scheduled work. A formerly served SHA can still be incompatible with current season state. For dispatch versus a local restart see B7. |

## 2. Words only

`probe_differential.py` imported both actual packages from the archived and working sources and printed each `bts.__file__`. All 380 frame/decision/candidate combinations preserved the None gate and other keys. Ten lineup return comparisons preserved all fields apart from the permitted display fields, including terminal tail, heuristic, status failure, clamped double and injected `sel=None`. All 23,884 bar comparisons matched across both regimes including bounds and trust states.

Alert and skip effects matched across build and dispatch faults, failed sends, same-day dedup/restart and the year boundary. The unconfirmed skip notice remains marked in memory and can duplicate if interrupted before persistence. Missed-pick gating matched on absent, empty, legacy and no-decision summaries. Refusal records, archives, levels, sources and status paths matched. The late and postponed guards still leave the day unlocked.

## 3. Decoded state

`probe_rendering.py` round-tripped base→HEAD and HEAD→base, exercised legacy and malformed shapes, escaped HTML and flattened line breaks. All tested decoded objects reached the renderers without an exception. Only K2 generated its template comparison. Invalid JSON was quarantined and a directory-path case stayed in the loader. Finite percentage overflow is B2.

## 4. One order

`probe_ops.py` used the unchanged workflow body, current cron installer, real entry-intent CLI and command shims. The canary passed, failed and stopped early as described in item 1. A5 skips required B8c steps when read literally. INCIDENT's restart-only dispatch conflicts with its checkout-reset description and its cookie table conflicts with its Mac recovery procedure. B3–B8 give the tested counterexamples and corrections.

## 5. Season walk

The fresh walk ran 4,599 records with zero problems and matched the committed output. Compared with the saved base, every action, bar, day count and selected pick agreed. Among the 841 former failures, 229 rendered K3 and 612 rendered K4. The first planted check selected an out-of-season record, which the checker excludes. Repeating on a real in-season K3 returned the expected “skip message” failure. See `walk.log` and `walk-checker.log`.

## 6. Tests and scratch changes

The requested files passed in one process, 291 tests in 43.92 seconds. The direct dashboard socket test needed loopback permission, and the runner blocked outbound connections. The suite, walk and scratch changes ran directly under the later instruction. Each of the nine scratch changes ran separately against the same 291 tests.

| Scratch change | Tests that failed |
| --- | ---: |
| Name the top prediction instead of the decided candidate | 4 |
| Say below when the probability equals the bar | 1 |
| Drop the unverified eligibility wording | 12 |
| Swap K1 and K5 when both apply | **0** |
| Add the obsolete lineup promise to the K3 journal line only | **0** |
| Leave the batter name unescaped in the banner | 8 |
| Drop the mark-used sentence from the delivery warning | 25 |
| Restore the late refusal's last manual-entry sentence | 2 |
| Raise on a string probability | 1 |

The two misses are test coverage notes. The suite lacks a fixture with both a reason and a no-decision basis, and it does not pin the complete K3 journal line. The review probes exercise both cases. See `combined.log`, `mutations.log` and the individual `mutation-*.log` files.

## 7. Other reviewed changes

The runtime diff stays in the allowed scheduler and banner code. The walk production change passes the real primary candidate. The comment above the scheduler's summary call still says a skip means “below the pick bar,” which is false for K3. No tracked files were edited.

ran: offline exact-text, CLI, shell and state probes, archived-base comparisons, 4,599-record season walk with planted text, 291 focused tests in one process, nine serial scratch changes; did not run: external network, production services or cron, any wrapper job, or the full fast suite
