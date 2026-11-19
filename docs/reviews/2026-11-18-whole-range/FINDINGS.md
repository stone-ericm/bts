# Whole-range review of `review-candidate-2026-11-18` (`d1919004`): findings and what each gets

Three fresh Codex reviewers (highest effort, memories off, earlier review records removed from their copies), one slice each; reports and probes in `slice-1-codex/`, `slice-2-codex/`, `slice-3-codex/`. All three: **DEPLOYABLE AFTER THESE FIXES**. Carbon's read of slice 1 is added when it reports. Lead's ruling R44 below.

| # | Finding (slice) | In the range? | Decision |
|---|---|---|---|
| W1 | **The policy counts one day too few all 2027 season and zero on the last day** (2): `SEASON_END_DATES[2027]` is the last game day (09-26) where 2026's value was the day AFTER it (09-28 for a 09-27 finale); `_days_remaining` subtracts exclusively. Streak 56 on the last day skips; 09-20 at 43 skips. The lead's own bug of 2026-11-13 | Yes | **Fix set A** |
| W2 | **A pick chosen this cycle is sent after the dated observation changed between selection and the delivery read** (1 and 3, independently): scheduler, `bts run`, `bts orchestrate`; message shows the new streak, action is from the old one | Yes (5a exempts the same-cycle choice) | **Fix set A** |
| W3 | **`run_single_check` computes `lock_decision` and does not return it**, so the in-loop fallback starts a second cascade instead of reusing an overrun check's pick and can miss the cutoff (3) | Older, reachable | **Fix set A** |
| W4 | **A still-eligible saved pick is stranded when a check returns nothing and the next check is after the fallback deadline** (3): no fallback, no pick, no missed-pick DM, no decision | Older, reachable | **Fix set A** |
| W5 | **A malformed schedule body is read as "no games"** by the scheduler and by `bts day-outcome` (3): silent off-day | Partly (day-outcome is new) | **Fix set A** |
| W6 | The saver warning tells the operator to use `bts set-contest-streak` without the dated form; that installs a legacy override the live path refuses (1) | Yes | **Fix set B** |
| W7 | After a crash mid-send and a correction, the entry watchdog names the unconfirmed, obsolete pick and says "Fix it now!" (1) | Partly | **Fix set B**: on an unconfirmed delivery the message says to verify the DM and MLB first |
| W8 | `bts day-outcome` prints "A + B" for a single (the decision record keeps the runner-up in `double_down`) (2) | Yes | **Fix set B** (render only; the record's shape is not changed) |
| W9 | `bts season-contract status` says "live decision now: accepted" from a dated observation while the automatic reading behind it is unusable (1, owner question) | Yes | **Fix set B**: status shows automatic readiness separately (lead's recommendation; Eric can overrule). Checklist B8a reworded |
| W10 | `contest_streak_fetch_status.json` holding `[]` makes a successful fetch raise after writing the reading (3, note) | Older | **Fix set B** (non-object reads as missing, as the health status files do) |
| W11 | A lost DM response leads to the same pick being DM'd twice (1) | Older | **Not fixed.** A duplicate identical DM is the safer side of the alternative (not resending a pick that may not have arrived) |
| W12 | **The automatic rollback target (`619abda7`) cannot serve 2027** and damages state the new code then refuses (2) | Yes, by construction | **For Eric** (deploy workflow and checklist): no automatic rollback on a live 2027 box; stop outward sending and page |
| W13 | **`deploy.yml`: a failed sync or first restart exits before the rollback branch** (2) | Older | **For Eric** with W12 |
| W14 | A refused preview left on disk as an unresolved pick makes the 02:00 season replay unavailable indefinitely (3, owner question) | Partly | **For Eric**; lead's recommendation: the replay excludes a pick that is provably uncommitted, and `reconcile` prints a warning when the replay is unavailable |
| W15 | A valid-JSON probable-pitcher cache with a non-object entry raises in every feature run (3 and both cache reviewers) | Older | **For Eric**; lead's recommendation: drop invalid entries on read with the same warning |
| W16 | The entry checker can miss a pick sent in the last minutes before first pitch (2, owner question) | Older | **For Eric** |
| W17 | The deploy canary passes on a box that starts, pages and idles with no usable contest state (2, owner question) | Yes | **For Eric** |
| W18 | New Year: grading that finishes after the 02:00 reconcile overwrites the new-season streak file (3) | Yes | Not fixed: no contest game is played on December 31 |
| W19 | Checklist B8a tells the operator to initialize the saver flag after the first automatic fetch, which has already done it (the command then exits 1) (1) | Docs | Reword B8a with fix set B |

## Ruling R44
Fix sets A and B are built as two tier A change sets, each from its own contract (checked first, with the standing hostile-input rows), each reviewed by fresh Codex and Carbon reviewers; the slices' probes are adopted as regression tests (slice 1's three-day rehearsal first). W12 to W17 go to Eric in plain language before anything in the deploy workflow or those behaviours changes. The candidate tag moves only when A and B have landed; a confirmation pass then re-runs the three slices' harnesses on the new head. Cost if wrong: W11 and W18 are left on purpose; if a duplicate DM or a December 31 game ever matters, they are documented here.

## Ruling R45 (2026-11-18, after the contract checks)
Both contract checks returned BUILD WITH THESE EDITS (`contract-check-a/`, `contract-check-b/`); their replacement texts are the contracts (version 2). **W3 and W4 leave fix set A and become their own change sets**, one piece each, from their replacement texts: the check showed that returning the lock decision alone lets a replaced pick file be sent under another pick's commit, and that the stranded-pick fix has to reschedule the missed-pick alert too. Fix set A is W1, W2 and W5. The contract check also established that the day-count fix changes one shipped-policy action (a double becomes a skip on 09-25 at streak 55 with the saver unavailable and a top probability of .60): accepted, since it is the policy as solved at the right day count. Cost if wrong: W3 and W4 stay open until their change sets land (both need a prediction pass to overrun or fail on a day with a saved pick).

## Carbon's read of slice 1 (2026-11-19 00:06; `slice-1-carbon/`): DEPLOYABLE AFTER THESE FIXES
Unreproduced by the lead so far (policy: reproduce every Gemini finding before acting); each goes through a contract check that reproduces it first.

| # | Finding | Relation to the table above | Decision |
|---|---|---|---|
| W20 | **The end-of-day health run after midnight judges the contest state at the wall clock, not at the processed day** (`health/runner.py` calls `contest_state.check` without `now`). On a dated-observation day whose last game ends after 00:00 ET, yesterday's observation has expired, the check falls back to whatever automatic file is on disk and sends a false `CRITICAL` with a template for the NEW day, before MLB has settled the night's pick; following it locks in a pre-settlement streak for the next day | New. Bites only while the season runs on dated observations (the first days), on any late game | **Fix set C** |
| W21 | The saver warning's bare command | Same as W6 | In fix set B |
| W22 | **After a restart between the attempt marker and the send, no `DELIVERY OUTCOME UNKNOWN` alert is sent:** the restart locks `locked_unconfirmed` through the classification path, which never reaches the code that pages; the unsent pick is then scored | Extends W7 (Codex saw the entry watchdog nag; Carbon shows a day with no DM at all when the entry check cannot run) | **Fix set C** |
| W23 | A non-scoreable pick file (a refused 03:00 preview, a skip day) makes `_replay_season_streak` return nothing for the rest of the season | Same as W14, found independently, with a two-line fix | **Fix set C** (moved from "for Eric": two reviewers, and the replay is the documented way to rebuild `streak.json`) |
| W24 | The check-time refusal and the 5a saved-pick refusal share one alert key, so the second is deduplicated on the same day | Known (shared `contest_state` source) | Note for fix set C's contract check: say whether a distinct key is safe |
| — | The missing `lock_decision` | Same as W3 | Its own change set (R45) |
| — | `bts orchestrate` raises an uncaught `ContestStateError` if its later reads straddle midnight | Note | Not fixed (the box runs `bts schedule`) |

**Ruling R46:** fix set C is W20, W22 and W23, built after A and B land, from a checked contract that first reproduces each. W3 and W4 follow as their own change sets. The slices' harnesses, Carbon's included, are re-run on the head that has A, B and C.

## Ruling R47 (2026-11-19 02:15, after the fix set C contract check)
The check (`contract-check-c/`) reproduced every finding, Carbon's included, and returned BUILD WITH THESE EDITS. **Fix set C is split into three change sets, one piece each:** C3 (the season replay: W14/W23) is built now from the check's replacement text; C1 (the after-midnight health read, W20) and C2 (the unknown-outcome alert after a restart, W22) follow after fix set B lands, each from a trimmed contract (the check's full texts add retained-evidence provenance to C1 and a second-chance send to C2: the lead will take the smallest version of each that removes the false page and the silent day, and ledger the rest). **W24 is closed, not built:** a distinct key for the saved-pick refusal double-paged for one correction episode in the check's probe. Cost if wrong: until C1 lands, a late game during the dated-observation days gives a false CRITICAL with an unsafe template (checklist B8a will say to ignore a contest-state alert that arrives between midnight and the morning wake on those days); until C2 lands, a crash mid-send is silent until the entry watchdog or 08:30.

## Ruling R48 (2026-11-19 04:35): fix set C3 is STOPPED after three BLOCKs; W14/W23 stays open
`STOP-C3.md` has the record and the obligations. Two skip rules each let the replay run around a pick that had been entered and save a wrong model streak where the base refused to replay: the files on disk cannot prove a pick was not entered. `fix/wr-c3` is not merged. What may be built instead is visibility only (`bts reconcile` names the blocking date; nothing is skipped; the operator archives a never-entered pick file by hand). Cost: a refused preview left on disk still disables the nightly rebuild of the MODEL streak until someone removes it, and until the visibility change lands `bts reconcile` does not say so. The contest streak, which decides picks in a live configuration, does not depend on this replay.

## State of the whole-range findings at 2026-11-19 04:35
| Finding | State |
|---|---|
| W1 day count, W2 same-cycle pick check, W5 schedule parser | **Merged** (fix set A, `082b8506`) |
| W6 saver warning, W7 unconfirmed-delivery wording, W8 day-outcome names, W9 status line, W10 fetch status file, W19 checklist | **Merged** (fix set B, `a1bf1612`); full gate on the merged tree green (9,444 passed) |
| W14 / W23 season replay | **Stopped** (R48); visibility-only change to be built fresh |
| W20 after-midnight health read; W22 unknown-outcome alert after a restart | Contract check done (`contract-check-c/`), not built: each its own change set, trimmed scope (R47) |
| W3 lost lock decision; W4 stranded saved pick | Contract check done (`contract-check-a/`), not built: each its own change set (R45) |
| W11 duplicate DM after a lost response | Left on purpose; note that W14's analysis shows the same window (the cleared attempt marker) is what makes an entered pick invisible afterwards |
| W12, W13 rollback; W15 cache entry; W16 late entry check; W17 canary | For Eric |
| W18 New Year grading order; W24 shared refusal key | Not built |
| Re-run of the four slice harnesses on the final head | Not done |
