# W3 review, round 2

## Verdict

**BLOCK** on `c0917aa7`. Both reviewers' original blocking inputs now pass. The revised planner can still discard an enterable pick, and it can deliver a pick it learned was postponed.

## Blocking findings

1. **The fallback state save can expire the window after the final replan.** Game 101 has a confirmed primary, a 19:05 ET cutoff, a 12-minute cascade budget and a 10-minute operator reserve. At 18:42:59, the overdue game 102 window is feasible until 18:43. The added reads finish on time. The `save_state` at `src/bts/scheduler.py:3541` takes two seconds. At 18:43:01, `:3548` checks the cutoff, then `:3554` archives on the old plan from `:3532`. The next prediction is empty. W3 ends with no pick or alert. In a paired base run whose two-second refresh ends at the same instant, game 101 is sent. With a best-only correction at that boundary, the base sends `contest_state` and `missed_pick`, while W3 sends neither. Run the live feasibility check after the state write, immediately before deferral. The scratch repair restored delivery and both alerts in the respective cases. [Head](planner-head.log) · [Base](planner-base.log) · [Repair](planner-repair.log).
2. **The replan can discard a timely all-void result and send the postponed batter.** On the same gap window, the defer-only status read returns `Postponed` for game 101 and advances to 18:43:01, still before the 19:05 cutoff. `src/bts/scheduler.py:3532` changes the plan to deliver, bypassing the known-void check at `:3542`. The transport guard's second status read gets `OSError`, and `:2646` supplies an empty map. W3 sends and commits game 101 without a postponed alert. `59b72c04` used the first result to emit `postponed_pick` without naming the batter. Preserve a timely all-void refusal from the polled defer when feasibility changes. The scratch repair refused the pick with no transport. [Head](planner-head.log) · [Prior W3](planner-old.log) · [Repair](planner-repair.log).

## Notes

- An authoritative commit can replace the batter during the poll. The new shadow hook at `src/bts/scheduler.py:3510` then passes the old candidate's name. The real inline shadow runner printed `AGREES` for Synthetic Batter 101 while the preserved production commit named Other Committed Batter. Pass the preserved primary's name or omit the comparison if its name is absent. This affects the journal. The correct durable pick still controls scoring. [Recovery](recovery.log).
- The authored W3 fixture still runs without its policy files, as Carbon noted. The independent binding, policy and Carbon replays load the shipped policy. The archived status-delay probe still counts a skip notice as a pick DM. Its separate pick-only replay recovers the expected 19:17 no-commit result. [Archived](archive-r2.log) · [Corrected](archive-corrected.log).

## 1. Round-one blockers on their original inputs

| Finding | Replay on `c0917aa7` |
| --- | --- |
| Codex R1 | Matching scoreable unconfirmed record preserved byte-for-byte with one alert. A new attempt with `action: []` recovered without the prior parser error. `r1-durable.log` |
| Codex R2 | The original two seconds during the status lookup moved the plan to deliver at 18:43:01. The dated-correction variant emitted both alerts. `r1-gap.log` |
| Codex R3 | For DM and public, an initial `EIO` during known-unsent marker clear was retried. Both runs archived the late refusal and alerted. `r1-clear.log` |
| Carbon R4 | Carbon's original deliver at 19:05:30 and defer whose state save crossed cutoff both emitted one unnamed `postponed_pick`. Replayed both variants from Carbon's save-hook and shipped-policy probes. `carbon-block.log` |

## 2. Required round-one probes

| Row | Observed result |
| --- | --- |
| M1 | Real 18:25 check overran to 19:00 and sent once. One cascade, zero reuse duration, matching commit. `binding.log` |
| M2 | Candidate replacement, audit append replacement and whitespace triggered legacy refresh. Warmup cache and mismatched probabilities used saved rows and metadata. Matching single retained its runner-up. `binding.log`, `policy-meta.log`, `capture.log` |
| M3 | Actual fallback sleep refreshed. Equality at the deadline reused. The later scheduled check ran after reuse deferral. `binding.log`, `coalescing.log` |
| M4 | Standing MDP skip survived cached Warmup. A genuine later single cleared the skip and delivered. `policy-early.log` |
| M5 | Playable candidate expired during reload, final read or attempt write and took the late refusal. Finding 1 covers an earlier feasibility deadline. `reload.log`, `cutoff.log` |
| M6 | Best-only and saver-only corrections produced `contest_state` and `missed_pick` with no attempt or commit. `outcomes.log` |
| M7 | Confirmed stale DM and public delivery each recorded the reading and warned once. A failure or missing ID did not count as confirmation for the warning. `outcomes.log` |
| M8 | Immediate and would-defer void cases refused. Partial-void double kept both legs. Finding 2 covers a timely void result lost on a changed plan. `carbon-block.log`, `lookup.log`, `policy-meta.log`, focused suite |
| M9 | Initial absent/dangling yielded no pick. Regular symlinks read through. Bad JSON, shape, time, unreadable path and directory raised. Initial FIFO hit a bounded alarm. The extra nonblocking read kept evidence and left a later check runnable. `pick-initial.log`, `pick-extra.log` |
| M10 | Delivered, attempted and authoritative committed cases recovered. Matching slots polled and mismatches did not. Initial changed bytes/marker took legacy refresh. `legacy.log`, `r1-durable.log`, focused suite |
| M11 | Locked, empty, skip and handled-error returns kept prior shapes. Candidate write failure propagated without a reusable return. `policy-early.log` |
| M12 | Both updated consumers, the Kwan slept fallback and measured-duration tests passed in the focused run. `focused.log` |
| Check 2 LW3-1 / LW3-2 | DM/public cutoff guard cleared an ordinary known-unsent marker. Before-read, in-read and marker-write rollover refused at midnight and New Year with a future cutoff. Automatic-to-automatic validation sent. `cutoff.log`, `calendar.log` |
| Check 2 decision and ordering | Decision absence reused. Wrong date, invalid or empty shape, invalid encoding, unreadable, FIFO, directory and dangling inputs disabled reuse. Scoreable bytes prevailed. New attempts recovered during both plan outcomes, including unreadable record and FIFO. `decision.log`, `r1-durable.log`, `recovery.log` |
| Check 2 shared budget and planner | Slow and failed HTTP bodies, side-partial response, threshold at three seconds, late joined daemon worker and later viable check all reran. Planner received all contenders, effective budget, reserve and only non-coalesced runs. `lookup.log`, `policy-meta.log`, `coalescing.log` |
| Check 2 M9 wording | Initial and extra nonblocking readers verified separately and externally bounded. `pick-initial.log`, `pick-extra.log` |
| Keyword ruling | Only the reused delivery call passes `reused_check_cascade=True`. All original callers keep their existing keyword shape. `legacy.log` |
| Post-loop ruling | Abandoned reuse left a real final cascade, contest-state refusal and missed alert. Five paired legacy traces, including result scoring, had identical journal, transport and compared file bytes. `abandoned.log`, `legacy.log`, `legacy-base.log` |
| Standing interrupts and restart | Both exceptions before/after marker, after durable confirmation, and in each poll. Fresh invocation used durable files and did not reuse the key. `interrupt.log` |
| Standing send outcomes | Known failure cleared and retried. Missing transport ID left the marker and produced no stale warning. `outcomes.log` |

## 3. What the amendments changed

- **Recovery and receipt.** Two real `run_day` calls hit the unknown-outcome helper twice for the same scoreable slate. An acknowledged receipt kept transport at one alert and the record byte-identical. A missing alert ID left a failed receipt, which the restart retried and acknowledged. FIFO, denied read and malformed decision with a newly observed attempt all recovered the marker. `recovery.log`, `receipt-retry.log`, `r1-durable.log`
- **Planner and void order.** Original controls pass. The new failure depends on which I/O takes the last two seconds. A slow status returning scheduled replans correctly in R2's original case. A slow state save leaves a stale defer. A timely postponed status consumed before the new plan is lost if the second status lookup fails. A normal delivery with trustworthy status and the original late/void races retained void precedence. `planner-head.log`, `r1-gap.log`, `carbon-block.log`
- **Retry and interrupts.** Two `OSError`s, either interrupt on the first clear, and either interrupt during the retry propagated before transport. Before restart there was a durable attempt but no refusal alert or commit. Restart raised the receipt-tracked unknown-outcome alert without resending. Exercised the cutoff in DM and midnight in public. `refusal-edges.log`
- **Shadow.** Observed initial preserved commit, injected matching attempted snapshot, post-poll preserved commit and post-poll attempt/delivery locks. Each called the inline trigger. The replacement plus simultaneous scoreable record exposed the false agreement noted above. `recovery.log`

## 4. Operator sentences

| Sentence | Worst literal reading |
| --- | --- |
| `FALLBACK: reusing this check's fresh cascade decision (no second refresh).` | The in-loop reuse passed its reads. A later independent final fallback can still start after refusal. The line is absent when those reads abandon reuse. `abandoned.log`, `r1-gap.log` |
| `FALLBACK DEFERRED — … window at …; archived …` | Finding 1 shows an already-infeasible window reported as the reason to archive. An operator could wait for a pick that never comes. `planner-head.log` |
| `FALLBACK — delivering … before the … cutoff.` | An attempt announcement can precede a refusal. In finding 2 the announcement leads to a delivery even with a prior known postponed result. `planner-head.log` |
| `Game status unreadable before delivery …; delivering unverified.` | In finding 2 only the last read was unreadable. The earlier read already established a postponement. The operator could treat the ensuing named pick as eligible. `planner-head.log` |
| `Pick locked. Stopping lineup checks.` | The scheduler stops on durable evidence, including an attempted or earlier pick. The sentence gives no entry instruction or delivery confirmation. `recovery.log` |
| `[SHADOW MODEL] … AGREES` | A replaced authoritative commit can yield false agreement with the old preview. An operator could count that as production agreement. `recovery.log` |
| `ET date changed before contest state could be validated` | Two date changes exhausted the retry with no attempt. The scheduler can run a later final fallback. The sentence gives no command. `calendar.log` |
| `ET date changed after the final contest-state read` | The guard refused before transport and cleared the marker in the normal case. A later fallback can still run. The sentence gives no command. `calendar.log` |
| `DELIVERY OUTCOME UNKNOWN for … ET.` | Means an unacknowledged attempt exists for that slate. It gives no entry authority. `recovery.log` |
| `A send attempt for that slate was never confirmed, and the scheduler locked the slate without resending.` | An operator may infer the first attempt reached transport. A marker also survives an interruption before transport. `refusal-edges.log` |
| `Check the DM or public feed and the MLB entry for … ET.` | Checking both can resolve ambiguity without changing the entry. `recovery.log` |
| `Do not enter or change a pick based on this alert.` | Following it avoids an entry based solely on the marker. `recovery.log` |
| `If … still today in ET … run bts season-contract status … before making any entry decision …` | The command is a state check during the live window. It gives no transport or entry instruction. `recovery.log` |
| `LATE DELIVERY REFUSED: … would have been sent … after the … cutoff.` | The named batter is past cutoff. Both transient retry cases kept the existing refusal message. `r1-clear.log` |
| `DELIVERY REFUSED for … game … is Postponed.` | The alert identifies the game and omits the batter. Carbon's two original states both produce it. `carbon-block.log` |

## 5. Focused tests and mutations

The eleven requested files ran together on `c0917aa7`. **342 passed, 4 skipped** in 23.68 s, including the optional-model collection and the web saver's synthetic loopback tests. [Focused](focused.log).

All **20** targeted scratch mutations were detected. No mutation survived. [Full results](mutations.log).

| Mutations | Detection |
| --- | --- |
| Round-one binding and timing | Removing byte match, allowing sleep reuse, removing post-marker cutoff or ET-day guards, passing an unchecked selection, coalescing after reuse, and capturing bytes by read-back all failed. |
| Round-one recovery and inputs | Ignoring scoreable records, omitting either final read, polling mismatched slots, or collapsing side confirmations to game checks all failed. |
| Round-one survivors | Applying a late result now fails the joined-worker check. Giving status a fresh budget now fails the combined elapsed-time test. |
| New planner, refusal and recovery | Removing the final replan, clear retry, or unconfirmed-record alert, or putting late before void, all failed. |
| New shadow and journal | Removing the initial scoreable-lock shadow call or moving the reuse line ahead of the rereads both failed. |

These tests cover the amendments' original timing and recovery cases. The two counterexamples above change the I/O timing after the replan or fail the follow-up status read.

## What I ran

- Independently replayed both reviewers' original blocking inputs, the required round-one table and new edge cases against `c0917aa7`. Paired the new counterexamples with `ceab9854` or `59b72c04` and verified narrowly changed scratch code. Replayed the pre-implementation contract examples on the base archive. All direct probes were bounded offline runs. `nice -n 19` warned about setting priority.
- The focused wrapper started at 10:29:38 ET after the author's gate cleared. The mutation wrapper started at 10:31:09 ET after the focused wrapper exited. Both exited zero. Python children blocked external socket connections. [Focused wrapper](focused.log.watch) · [Mutation wrapper](mutations.log.watch).

## What I did not run

- The author's separate full-suite gate or any real host/account operation.
