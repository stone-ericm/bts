# Verdict

**DEPLOYABLE AFTER THESE FIXES** for 619abda7 → bc143131. The guarded dated path works, including the singleton opener. The remaining blockers affect delivery idempotency, cutover, and instructions an operator receives while a slate is actionable.

## Blocking findings

1. **A lost DM response can send a pick twice.** With a valid dated 0, the real DM client made its non-retried send at 16:10. The fixture accepted the request and timed out reading its response. The scheduler cleared the durable attempt marker and sent the same pick at 17:10. The 401-before-send control had one acceptance. The retry seam is inherited and weakens C2's no-resend assurance. Make an ambiguous create response take the unconfirmed-delivery path and preserve the marker. Clear it only for a known pre-send failure. The public branch shares the same exception policy. [scheduler.py:1330](../../src/bts/scheduler.py#L1330) · [DM loss](interactions-dm-response_lost.json) · [401 control](interactions-dm-not_authenticated.json)

2. **Public catch-up can defeat the scheduler's lock.** After a confirmed stale-reading DM at 18:25, real bts run reposted the same pick publicly at 18:50. A separate bts orchestrate invocation also publicly reposted a confirmed scheduler DM. With a crash that left the 16:10 DM unconfirmed, bts run caught up publicly at 19:09, after its 19:05 cutoff. The scheduler scored the public marker and issued a public result reply. Required state/provenance reads passed. The catch-up branches test only the public flag and omit the cutoff. Early-return for any durable delivery or unconfirmed attempt, and apply the date/cutoff checks to any remaining outward catch-up. [cli.py:1205](../../src/bts/cli.py#L1205) · [orchestrator.py:386](../../src/bts/orchestrator.py#L386) · [confirmed](interactions-cli-stale.json) · [unconfirmed](interactions-cli-unknown-public.json) · [orchestrate](interactions-cli-orchestrate-public.json)

3. **A private lock can become an instruction to enter last year's pick.** The restored private/research setup chose on old automatic 17. Switching to DM/enter with today's verified dated 0 before pitch retained that scoreable private lock and produced no new recommendation. At 18:45 the entry watchdog told the operator “Fix it now!” with the old Batter 1. Keep private locks scoreable for research, exclude them from actionable entry prompts, and report a research-to-live lock as a missing live recommendation. Complete activation before any lock for the affected slate. [cli.py:1957](../../src/bts/cli.py#L1957) · [scheduler.py:1830](../../src/bts/scheduler.py#L1830) · [fixture](deploy-private-research-00.json)

4. **A stale-reading DM can install the wrong next-day observation.** The post-send text says to use the current ET date. In the 22:10 fixture the operator opened it at 00:05 while the prior game still ran, and entered the displayed 6. Settlement made it 7. The next fetch installed 7, but the dated 6 won selection at 18:25. Scope the instruction to the DM's slate, and require verified prior-slate settlement or absence of entry before any new-day correction, as historical health already does. Update B8a and the stale warning together. [scheduler.py:74](../../src/bts/scheduler.py#L74) · [fixture](operator-stale-midnight.json)

5. **The late-refusal DM can cause a conflicting entry.** At 19:05:30, after Batter 1's cutoff, the DM advised manual entry. The fixture entered another eligible batter. At 19:25 the scheduler DMed Batter 2 for a later game, and at 19:45 the watchdog instructed the operator to fix the resulting mismatch. The postponed-refusal template gives the same mixed instruction. Remove the invitation while automatic selection remains active and make any manual handoff explicit. [scheduler.py:2458](../../src/bts/scheduler.py#L2458) · [postponed copy](../../src/bts/scheduler.py#L2512) · [fixture](operator-manual-after-refusal.json)

6. **The activation and failure-direction instructions conflict with the work they launch.** B8a puts flags before deploy/start. B8c puts deploy before flags and start after cron install, while the workflow starts both services. Running the real shell body with shims reproduced the documented workflow risks. Early sync/restart failures bypassed the canary, and a failed canary restarted BASE. Publish one ordered first-boot procedure with verified protection from old writers and unsafe rollback before live activation. B4's live decision-file move also broke a correct entry's receipt. The 18:45 receipt lacked a hash, the 19:00 watchdog said already confirmed, and next morning paged ENTRY UNVERIFIABLE after file restore. Run the failure-direction drill on an isolated copy. A7 should restore agreeing intent and cron together. [checklist](../../docs/ops/2027-season-start.md#L14) · [workflow](../../.github/workflows/deploy.yml#L99) · [B4 drill](operator-decision-drill.json) · [A7 replay](operator-a7.json)

7. **The entry watchdog can send an all-clear for a known wrong doubleheader game.** After the 18:30 missing-entry DM, the fixture entered the recommended batter in the second game. At 18:45 the watchdog sent “BTS pick entry confirmed.” Its sealed receipt already said wrong_game from the local units capture, but the 19:00 verifier skipped as already confirmed. The next morning's audit only cited earlier absence. The correct-game control was entered, and the unbound-unit control was entered_game_unverified. Check currently available game qualification before writing a confirmed marker or sending all-clear. At minimum, withhold confirmation and expose a known mismatch while the operator can act. This is separate from the accepted limit when the unit has no known game binding at the time of verification. [cli.py:2056](../../src/bts/cli.py#L2056) · [entry_receipt.py:163](../../src/bts/entry_receipt.py#L163) · [wrong game](interactions-entry-wrong_game.json) · [correct game](interactions-entry-right_game.json)

## Findings that need an owner decision

- **Automatic reuse can make the wrong double look fresh.** On 09/20, a 16:10 projection used streak 45 and active saver from a 00:10 reading. A 16:11 fetch installed 0 with best 45 and the operator marked the saver used. With later prediction failures, the old double went out at 18:35. The DM said 0, chosen_on said 45/available, and automatic_reading said fresh at 2.40 h. The fresh-resolution control skipped because the reachable maximum was 14. R33 accepts automatic-to-automatic reuse. Its combination with 4c conceals the age and state of selection. Decide when a changed reading must invalidate a pending pick. If reuse remains, expose both snapshots and warn on material changes. [picks.py:323](../../src/bts/picks.py#L323) · [scheduler.py:1291](../../src/bts/scheduler.py#L1291) · [cached](interactions-automatic-correction-cached.json) · [fresh](interactions-automatic-correction-fresh.json)
- **An activation proves local coverage while the upstream counters can be from last season.** After two accurate dated days, the first post-enable response carried 17/best 18 and settled history dated only 2026-09-14. Fetch labeled it v2/2027/new activation, status accepted, and the scheduler sent on 17. The age stamp was fresh. The actual legacy file alone was refused. Fresh-but-wrong input is already a known risk. Require independent verification of the newest post-enable counters, and decide whether prior-year-only evidence should quarantine a nonzero opening reading. Current-season ledger coverage can legitimately lag. [cli.py:2410](../../src/bts/cli.py#L2410) · [season_contract.py:111](../../src/bts/season_contract.py#L111) · [cutover](opening-old_profile-steady.json)
- **Local grading corruption can persist behind a clean entry report.** A valid JSON list in streak.json allowed live selection on a dated reading and a successful entry receipt. Scoring failed before health, two scheduler starts failed, 01:00 grading and 02:00 reconcile failed, and 08:30 was clean. The documented unreadable model-file refusal is preserved. Here its purported replay recovery also needed the ungraded day resolved. Cron stderr and the reconcile receipt are the signals. Route grading/replay failures or include a dedicated model diagnostic. [picks.py:643](../../src/bts/picks.py#L643) · [scheduler.py:2806](../../src/bts/scheduler.py#L2806) · [fixture](interactions-refusal-model_list.json)
- **Enter cron misses early slates.** The 06:10 singleton delivered at 05:10. The first installed watchdog slot, 10:00, was 230 minutes after pitch, even though the fixture held the right entry. At 08:30 next day the checker paged ENTRY UNVERIFIABLE. The 10:30 and 13:30 fetches also share a minute with the authenticated watchdog. Earlier windows or an explicit early-slate procedure require a decision on account traffic. [cron](../../scripts/cron-setup-hetzner.sh#L58) · [early fixture](cron-early-entry.json)
- **A JSON list at the old automatic filename disables the fetch repair path.** Each fetch raised AttributeError before the throttled alert. Schedule and 08:30 still flagged the missing live outcome. The fetch job itself leaves a traceback. Treat the non-object prior as an explicit refusal with normal throttled diagnostics and evidence-preservation guidance. [cli.py:2386](../../src/bts/cli.py#L2386) · [fixture](interactions-refusal-auto_list.json)

## Notes

- BASE wrote the 2026 synthetic pick, model streak, contest observation, saver, scheduler/health state and research cron. Every HEAD run used independent fixture state. The review used bundled policy artifacts, synthetic probabilities and captured transports. EOD digests include fixture-only missing blend and local disk warnings.
- The sandbox printed “nice: setpriority: Operation not permitted” on the short probes. The final probe runs completed and their exit codes and fixture assertions were checked. The user ran the queued mutation commands through the wrapper.
- Accepted notes include 5b, W3/W4, R37, R48, C1's retained-evidence limits and W12/W13/W17 in the unchanged deploy shell. The decision reader ignored JSON truncated after an opening brace. The accepted R37 crash loop reproduced with a JSON list as the action, which raised in real schedule and day-outcome with no exception DM.

## 1. Opening Night and first three days

The 19:10 singleton had checks at 16:10, 17:10 and 18:25, with a 19:05 cutoff. Fifteen variants ran steadily, with a 13:00 interruption and with an interruption at send where a send existed. Cutover variants also attempted Day 3 interruptions. Pick and health DMs used offline.test. The watchdog fixtures here verified batter identity. Section 5 covers receipt game binding.

| Case | ET output | Durable outcome |
|---|---|---|
| Correct | 03/24 single on 0 at 16:10, then 03/25 and 03/26 doubles on 1 and 3 at 18:25 | Delivered and scoreable from dated input. Hits advance model 1 → 3 → 5 with one applied marker per slate. Clean entered_game_unverified |
| Missing day 1 | 10:00 morning and 16:10 live refusal, EOD diagnostics, then 08:30 NO DECISION | No pick/decision on 03/24. Dated 0 and normal send on 03/25 |
| Filed between morning/lineup | Morning warning, noon dated 0, 16:10 pick | The next cycle read the corrected observation |
| Filed too late | 18:26 write accepted, no pick, 08:30 backstop | Observation present, no commit. Accepted 5b limit |
| Wrong 30 corrected | Noon repair goes out on 0. After a projected choice, the corrected 0 goes out at 18:25. After confirmed delivery, correction does not repick | Pending selection replaced where a fresh check remained. Confirmed selection unchanged |
| Day 3 automatic | Pre-enable fetch ledger only. Post-enable matched fetch at 10:00. Recommendation on 3 at 18:25 | Covered v2, current activation, saver not_earned, fresh automatic stamp |
| Day 3 October file | Enable alone leaves old file, 10:00 and lineup refusal, next 08:30 NO DECISION | Legacy automatic remains blocked. Distinct old-profile re-stamp risk above |
| After-midnight result | 22:10 game, 19:10 recommendation, 00:20 finish | Grade stays 03/24 and historical health uses old dated observation |
| Midday or mid-send restart | Remaining lineup checks run. Restart after confirmed delivery skips prediction and resend. An unacknowledged send produces one dedicated warning | Delivery-attempt flag survives and locks unconfirmed. Watchdog advises verification. 08:30 classification depends on receipts |

The events and pick/decision/streak/health snapshots are in opening-*.json. [rehearsal](opening.py)

## 2. Midseason, late season and finale

| Input | BASE | HEAD |
|---|---|---|
| 07/15, 12/best 15 and saver active, high probabilities | Zero days, skip | 74 days, single at 18:25, model 13 |
| 07/15, low probabilities | Genuine skip | Genuine skip, precise threshold instead of fallback estimate |
| 07/15, game 101 postponed after lock check | Calendar skip | Refused/archived game 101, warning, later policy evaluation |
| 07/15, doubleheader | Detects Game 2, calendar skip | Detects Game 2, single with available lineups |
| 09/20, 45/best 45 and saver active | Calendar skip | Seven days, double. Walk-only yields void and keeps 45. Hitless official AB resets model 45 |
| 09/26, streak 56 | Zero days, skips even at 99% | One day, single and model reaches 57 |

Identical committed doubles through real check-results advanced 12→14 on two hits in both versions. BASE called walks misses. HEAD gave 12 for two voids and 13 for hit+void. Both sent an unchanged double with one postponed game and scored the remaining hit. BASE sent a sole postponed pick and scored it void. HEAD archived and refused it. The calendar, guard, grading and wire differences match the documented changes. [comparison](midseason.py)

With Game 2 at 16:10 and lineups first ready after the 15:25 last check, both schedulers detected confirmation at 15:40 during their doubleheader re-check. Neither ran a new prediction. HEAD's next 08:30 audit classified NO DECISION. This is the documented no-pick limit after the last prediction check. [late Game 2](midseason-head-late-dh-lineup.json)

## 3. Restore, startup and rollback

- On synthetic restore day 03/03 the original configuration and required DM configuration both idled on the empty schedule with no pick or alert. On the opener the original private/research configuration sees old 17 and locks privately. Independent EOD health can still DM “pick delivery failed” for a research lock. Either required flag suffices for the live gate. DM with both off refused repeatedly with exit 2, no heartbeat and no DM. Dry-run warned and wrote a heartbeat. Required with no valid switch or dated observation started, paged, and slept toward the next wake. [off-day start](deploy-preseason-offday.json)
- The installed unit uses Restart=always and a 30-second delay. An exit-2 start repeats stderr and can age out the heartbeat without a per-restart DM. Actual Linux unit scheduling/limits were outside the fixture.
- The embedded shell with shims passed active and successful-exit cases, rolled back on failed canary and exited before rollback on failed sync or first restart. It can pass a running scheduler that refuses the day's pick. This uses TARGET_SHA reset, while older prose describes pull or a remote ref. See B8 ordering finding.
- BASE read HEAD's pending pick, decision, scheduler state, newer account reading through legacy loader, saver, generic health status and sealed receipt. It deduplicated the generic alert, did not resend the committed pick, graded it, removed automatic_reading on rewrite and saved untagged model state without applied dates. The BASE fetch rewrote v2 to v1 despite the switch file. HEAD then refused automatic until a covered fetch. New switch and stale warning files stayed inert under BASE. [deploy probes](deploy.py)
- The separate leaderboard unit template changes its auth failure recipient from stoneericm.bsky.social to stonehengee.bsky.social. Invoking both templates' real CLI text with captured AuthError confirmed the different recipients. I found no corresponding CLAUDE.md note. The deploy shell leaves installed leaderboard units alone. [unit template](../../deploy/systemd/bts-leaderboard.service#L15) · [BASE](deploy-leaderboard-base.json) · [HEAD](deploy-leaderboard-head.json)

## 4. Cron

- Ran the script's show body with uv/crontab shims for research/private and enter/DM and for the mismatched enter/private mode. Both agreeing configurations rendered. The mismatch refused. Click parsed and ran --help for each bts invocation, including the four nested 03:00 commands. Checked heartbeat and curl help. The BASE research render supplied the restored cron fixture.
- Under enter, check-pick-entered can DM missing, mismatched or unconfirmed recommendations and an all-clear in the entry window. Fetch failure DMs are enabled at 01:10, 02:10, 10:30 and 13:30. Fetches lack the recipient under research. Day-outcome at 08:30 can DM under either intent. The scheduler and external heartbeat have independent signals.
- 01:00 grading can overlap scheduler scoring and the 02:00 reconcile while games are pending. Scoring uses a shared lock and dates applied once. 07:40 remains before the correction cutoff. 03:00 starts after first reconcile and can overlap a late scorer. Its data sync includes pruning by default. The synthetic object store preserved the old objects when the guard refused, and pruned them after the only manifest was manually removed and an empty sync succeeded. The entry check has the documented W16 last-minute gap and the early-start exposure above. Scheduled authenticated jobs also coincide at 10:30 and 13:30 in enter. [cron probes](cron.py) · [artifact drill](operator-r2-manifest.json)

## 5. Interactions

- A preview selected on the old 17 before the morning observation and wrote chosen_on.required=false. Both that preview and a projection chosen before a dated correction were refused at fallback when fresh predictions failed. Matching cached dated state delivered at 18:35. Warmup reuse exercised both in-loop and final fallback refusals with a deduplicated contest DM, 19:00 missed-pick DM and next morning audit.
- At the next 02:00 reconcile, the refused preview remained unresolved and blocked replay. The real CLI printed “Season replay unavailable for 2027-03-24 (no final result on the pick)” and preserved the file and model streak. This matches the accepted replay limit and B8b's preserve-and-escalate action.
- A dated delivery has null automatic_reading. With stale automatic state, the scheduler stamped age and shared the morning/delivery warning receipt. Fresh or catch-up bts run lacks that stamp of its own. The same catch-up seam publicly replayed a covered automatic pick for the prior slate at 00:05. [prior-slate catch-up](interactions-cli-prior-slate.json)
- After confirmed delivery a same-slate restart skipped morning selection-state checking despite the dated file having been removed. Same-day EOD could still report unusable state. After midnight health used the prior slate's last ET microsecond. Replacing the old observation before EOD produced the documented prior-slate warning with diagnostic-only copy and a settlement prerequisite. An undated launch at 00:05 chose the new slate and left the old grade pending. [historical warning](interactions-historical-health.json)
- The unconfirmed send marker produced locked_unconfirmed, a dedicated warning and no automatic resend on restart. The entry checker gave caution copy and status advice. Day-outcome classified successful account evidence of absence as NOT ENTERED and no decision as NO DECISION. A confirmed batter can classify clean even if later scoring fails.
- The real watchdog produced sealed receipts from a synthetic local units capture. Day-outcome classified an exact match entered and an unbound game entered_game_unverified. The known wrong-game fixture is in the blocking findings.
- 01:00 after the opener graded on the pick's date, reset a readable 2026 model file and saved 2027/streak 1. The re-score after a simulated crash between model update and pick result kept 1 and finalized the pick. At 02:00 a hit→void rescore made it 0, and 07:40 kept it. A synthetic Dec 31 pick graded on Jan 1 under 2027 (42→43). 02:00 corrected the old pick and saved the empty 2028 replay at 0, as did 07:40.
- A spring-only feed and a weather sidecar gave a cold build exit 0 with no 2027 parquet or refresh marker. Real prediction loaded 12 prior-season rows and reached the first fit seam, where the probe stopped. The bullpen feature remained 0.5 both before and after changing the later same-day PA pitcher to a reliever. The probable cache dropped two bad entries and recovered one from raw, preserved evidence and reported health. An unreadable whole-cache case rebuilt usable raw entries, kept the old bytes and warned. B7 rejected a stale store and a header-only seed, then succeeded with a populated synthetic 03/24 store at 07:45 next day. [interactions](interactions.py) · [cold probe](cold.py)

## 6. Clock and season reads

| Consumer | Review and observed boundary |
|---|---|
| Strategy and tail health | Slate date determines remaining days: 187 on 03/24, 74 on 07/15, 7 on 09/20, 1 on 09/26, 0 on 09/27 and 12/31. Unconfigured 2028 uses 09/28 fallback and the health source warns when run. BASE's fixed 2026 boundary has zero days throughout 2027 |
| CLI dispatch and scheduler plans | Schedule chooses launch ET date before loading config and carries an explicit slate through waits. Day-outcome defaults to current ET day and cron passes yesterday. Preview defaults to tomorrow ET. Thin-slate plan filters elapsed early ET checks |
| Dated override and saver | Dated observation is in effect through the last microsecond ET. Yesterday's command exits 2 at midnight. Saver flag and dashboard default to current ET year. The CLI's explicit-year mutation remains writable across New Year |
| Switch, coverage, fetch and ledger | UTC timestamps encode instants. ET year and activation gate automatic. At 20:30 ET on Dec 31 the live gate still uses the old ET year. Fetch keys freshness to the start instant. Date coverage independently informs status and best trust |
| Scheduler warning, recovery and EOD | Age compares elapsed time in UTC. Unknown-delivery receipts key to slate. Historical EOD uses the previous slate's last ET microsecond. The documented same-day mode sample can cross midnight |
| Scoring and replay | Grade uses pick date's year, replay samples current ET year under lock, and display paths follow the date they report. Documented applied-date spelling limit preserved |
| Data, probable cache and cron | Runtime build year is the box's calendar year. Refresh uses target date and the inherited 03/20 floor, independent of official opener. Evidence uses UTC timestamps and ET status partitions. Health checks the processed and prior dates |
| Dashboard and research | Dashboard retains host local date defaults under box ET assumption. Archived model/strategy constants and 2026 research literals were scoped by reading only. No studies run |

The retained 2027 automatic switch went from accepted at 23:59:59.999999 Dec 31 to refused at Jan 1 midnight. The synthetic 2028 opener without reactivation paged and sent no live recommendation. 09/27 and 12/31 off-day checks classified no_games. [clock fixture](interactions-clock-rollover.json) · [range search](clock-range.log)

## 7. Fail-closed and fail-loud

| Family | Persistent case and signal |
|---|---|
| Config gate and intent | Startup exit 2 and stderr before READY/heartbeat, no per-restart DM. Cron intent refusal preserves table. External stale-heartbeat signal remains |
| Live state, activation and saved selection | Invalid/missing state or switch, legacy automatic, and dated mismatch stop selection or delivery. Morning/lineup refusal, dispatch status and day-outcome are backstops. Good dated observation can recover an otherwise refused automatic read |
| Account setters, saver and switch writes | Bad input prints usage or a refusal. Write errors report failure or partial append/rename on stderr. No separate DM for an interactive command |
| Fetch | Auth, identity and shape/switch errors normally use stderr and throttled configured DM. A non-object prior bypasses this. Research omits fetch DM intentionally |
| Schedule, pick, and decision shape | Strict malformed schedule raises after READY and leaves journal/restart signal, then 08:30 schedule_unverifiable DM. The R37 malformed action retained its known silent application-alert gap |
| Late/void/provenance delivery and missing recipient | Late/void archive and page. Provenance refusal leaves the candidate and signals contest state. An empty recipient only logs the pick failure and the next cron audit can page using its own recipient |
| Model score and replay | Corruption can fail scoring and reconciliation while day-outcome sees a clean entry. A replay-blocked past pick gives stderr and receipt without a separate DM |
| Build/pull/features and research validation | Empty preseason exits 0. Malformed feed/directory fails the 03:00 chain and logs. Missing future blend can page. Probable cache reports stderr and EOD. Tool-only validation exceptions surface to their caller |
| Health receipt and transport | Malformed alert receipt cannot suppress the send, status-write failures log. Missing verified message ID leaves unconfirmed status and can retry on next encounter. Durable underlying transport ambiguity needs the delivery fix above |

The real loader refused a damaged switch, other activation, wrong season, selected legacy override and malformed dated record. A newly verified dated write restored live readiness in each case. The corrupt-switch status still exited 1. A directory at the fetch-status filename caused fetch exit 2 after writing automatic 13, which status confirmed was accepted. The explicit failure output and acceptance check both matter during cutover. [fetch write-order fixture](interactions-refusal-fetch_status_directory.json) · [refusal search](refusal-range.log)

## 8. Mutation sample

The user ran all 30 queued commands through the wrapper, one at a time. The unmodified scratch baseline passed all 72 parametrized cases across 29 test functions. Each matching single-source mutant then failed its selected test. **29 detected, 0 survivors, 0 inconclusive.**

| Tested behavior | Mutants detected |
|---|---:|
| Calendar and unknown-year warning | 4 |
| Singleton, moved-up game, clustering, regular slate and restart planning | 7 |
| Day outcome, game binding, cutoff, receipt SHA, void and policy-skip classification | 7 |
| Date idempotency and saver replay | 2 |
| Automatic reading kind, age boundary and elapsed time across DST | 3 |
| Live startup and intent gates | 2 |
| Dated correction before delivery | 1 |
| Morning lead time, durable commit suppression and warning day | 3 |

The mutations exercise the selected branches. The integration defects above came from the synthetic scheduler and transport probes. [Source edits and test map](mutations.py) · [baseline](mutation-baseline.log) · [completion marker](slot-runs-done.txt)

## 9. Operator text

The operator table records instructions, worst states, consequences and verdicts. INCIDENT's generic re-capture instruction conflicts with the CLI's throttle/transient handling. B8b(1) still describes the pre-C1 wall clock. Its diagnostic action remains safe. [Operator table](operator-text.md) · [source search](operator-source-hits.log)

## 10. Other findings

- **An old saver recovery command can overwrite the active year's flag.** At 23:55 Dec 31 the real CLI printed “run bts saver-state --init used --season 2027.” At 00:10, after initializing 2028 as not_earned, that literal command exited 0, wrote the only flag with 2027/used and made current status uninitialized. B8a asks for an ET-year recheck. Add an equivalent guard to the CLI by limiting mutations to the current ET year or giving historical state its own file. The dashboard displays and changes the flag while an in-effect dated observation keeps its own saver, as documented. [cli.py:1622](../../src/bts/cli.py#L1622) · [rollover](operator-saver-year.json) · [dashboard](operator-saver-dashboard.json)
- **Day-outcome applies current intent to a historical slate.** The same fully verified 03/24 delivery read entered under enter and DELIVERED UNDER RESEARCH after switching the 03/25 config. That retrospective alarm is misleading during an intent transition. Persist slate intent for historical checks or include the current-intent assumption in the result. [day_outcome.py:145](../../src/bts/day_outcome.py#L145) · [fixture](interactions-entry-right_game.json)
- **Skip explanation can show an impossible comparison.** A later status update removed a 99.9% batter. The policy's skip used a 75% eligible runner-up, while the alert called the 99.9% batter below the 79.6% threshold. Build the summary from eligible decision inputs. [scheduler.py:1688](../../src/bts/scheduler.py#L1688) · [fixture](midseason-head-postpone.json)

## What I could not exercise

- Actual Hetzner boot, systemd/watchdog and cron scheduling, GitHub actions, R2, Tailscale, Bluesky or MLB. The embedded deploy shell and real CLI used local shims. Cookie refresh and actual MLB game/unit identity were outside the fixture. Synthetic exact, wrong and absent game binding were exercised.
- Process death at every write boundary, real simultaneous cron writers, filesystem failure matrices, and schedule/settlement behavior on the real dates. Simulator restarts reopen disk in the same process. No service was killed.
- Production model inference, training, backtests, official rule verification, actual 2028 calendar or real 2026 outcomes. The real cold path stopped before any fit. Operator instructions requiring external actions were inspected and marked unclear where not established.

ran: BASE snapshot writers, HEAD and BASE scheduler/Click, pinned delivery/score/rollover/cold and operator probes, cron/deploy shims, user-run mutation baseline and 29 mutants; did not run: real services/accounts, whole fast suite, training/backtests
