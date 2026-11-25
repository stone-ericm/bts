# Verdict

**BLOCK** — two stale-delivery instructions can install a wrong dated contest streak. Reviewed `b1abc2bf..1ed57c47` against the closed scope and the brief's R64 rule.

## Blocking findings

- **B1 — Later-date write with an intervening slate pending.** A July 15 warning read July 17 at 09:30 permits the operator to write 13/14/active after July 15 settles, even though July 16 is pending. July 16 then settles and the 10:30 fetch accepts streak 14, but the July 17 dated row wins and the real selection uses the wrong streak, 13. The captured warning led to the actual `status`, dated-write and fetch CLI commands. The control without the write selected on 14. Replace the later-date waiting sentence with “For any date later than the pick's slate, including D when the warning was sent after midnight ET, wait until MLB has settled all prior entries or confirmed their absence after their entry windows closed.” [B8a, line 31](../../docs/ops/2027-season-start.md#L31) · [Replay](later-slate.log)
- **B2 — Same-slate write with yesterday's entry pending.** At 11:40 ET July 15 the confirmed delivery warning prompts a comparison of stale automatic 12 with MLB's 13/14/active. Both dates are July 15, so the text permits a dated write while July 14 is still pending. It leaves July 15's live contest streak at 13 after July 14 settles and the 12:10 fetch accepts 14. The real send path and CLI reproduced the wrong state. Replace “Write only if the pick's slate and today's ET date both equal {D}.” with “Write for the {D} slate only on that ET date, with prior entries settled or absent.” The longest enveloped text would be 898 characters. In B8a replace its same-slate sentence with “On a warning after a delivery, write that command only on D ET for the D slate, after MLB has settled all prior entries or confirmed their absence after their entry windows closed. The sent pick and saver flag stay as they were.” [Scheduler, lines 91–96](../../src/bts/scheduler.py#L91) · [B8a, line 31](../../docs/ops/2027-season-start.md#L31) · [Replay](same_day_settlement.log)

## Edits

- **E1 — `src/bts/scheduler.py:1802`.** For a summary without a valid basis, replace “Streak used at this check: {s}.” with “Streak recorded with this check: {s}.” A same-day upgrade displayed the old suffix for a real base check that made no decision. The banner only describes the recorded summary, so this route causes none of R64's outcomes. [Replay](legacy-head.log)
- **E2 — `src/bts/scheduler.py:1766–1779`.** Reserve the literal “below the” for K2 by normalizing it to “under the” in interpolated display strings. For the stored K1 example replace “Reason: Observations below the expected count.” with “Reason: Observations under the expected count.” The phrase reached all shared renderings as data and did not change the check, notice gate or operator action. [Replay](bad-head.log)
- **E3 — `docs/ops/2027-season-start.md:31`.** Replace “After the switch: a stale automatic timestamp alone permits live delivery.” with “After the switch, live delivery remains eligible when the automatic timestamp is stale and the other delivery checks pass.” The original can read as a sufficient condition. At cutoff the real guard still refuses delivery, so the sentence alone causes none of R64's outcomes.

## Notes

- The equality refusal journal's “passed 0.0 min ago” is the accepted limit. The scheduler loader's `IsADirectoryError` on a directory in place of its state file reproduces on both revisions. The dashboard ignores it. The brief and Ea contract supply R64's rule. This revision's `FINDINGS.md` ends at R63.
- The first handoff probe stopped on a missing scratch-script import. The separate recheck completed C7, C15 and C18. The OS rejected the priority adjustment on direct runs, which completed within their timeouts. The approved process check found no `victoria3` process at the start.

## 1. Must still catch

| Row | Rebuilt result |
|---|---|
| C1 | Aug 30, streak 56, 99%: “at or above the streak-56 bar (82.5%).” |
| C2 | Jun 25, streak 56, every bin skips: “No probability comparison is available for this check.” |
| C3 | Final status removes 99.9%. The decision names Batter 1 at 75.0% against 79.6%. |
| C4 | All final, unreadable statuses, and absent selection each render “No candidate was chosen” and the unverified 99.9% prediction. |
| C5 | CLI reads active and marks used. The next automatic day sees the saver unavailable. |
| C6 | At 00:05 the prior slate is live. The original date exits 2, and B8a waits for that slate. See B1 for a different later date. |
| C7 | Late and postponed refusals withhold the archived pick. Both permit only the later official message in its window. A real replacement DM and entry mismatch followed. |
| C8 | Real 00:20 warning with the prior entry pending names July 16 and conditions the command on settlement or absence. [Replay](morning-midnight.log) |
| C9 | Before the first check, the CLI installs the correction. Doing it after selection blocks before the guard or leaves an in-flight send on the old snapshot. The morning guard excludes both. |
| C10 | 1e308, 1.5e308 and other out-of-range values give `?` for the invalid probability and no comparison. |
| C11 | After 18:35 skip, 18:55 empty check and restart, the actual page says “LAST RECORDED NO-PICK SUMMARY.” |
| C12 | Raising candidate class lookup leaves the unverified base result. A line-break-only reason disappears. |
| C13 | Persisted damaged fields render without raising and HTML is escaped. A literal phrase can get in via display data. See E2. |
| C14 | All 19 scratch changes detected. |
| C15 | No decision at 3, then a write of 12. Stored streak is 12. DM, journal and banner omit the suffix. |
| C16 | At 19:05:00 the DM says the 19:05 ET cutoff had been reached. Stored `late_min` is 0.0. |
| C17 | Both rows present. MLB 12/14/active matches automatic, but dated 3/14/used is in effect. B8a selects the dated comparison and CLI installs 12/14/active. |
| C18 | Send at 23:59:50, confirmation at 00:00:05. July 16 prints for a July 15 slate. B8a covers July 16 and the probe writes only after the entry settles. |
| C19 | Timestamp in future after successful fetch. Status has null `last_error` and B8a directs the operator to the host clock. |
| C20 | Dated delivery has null automatic reading. B8a describes that explicitly. |
| C21 | All three restored phrases detected. |

## 2. Sentences and operator actions

Exact strings checked through real renderers and the page, plus direct contract extraction for both warnings, warning journals, refusal envelopes and amended B8a text. Longest real envelopes are 895 morning and 881 delivery. `messages.log`, `operator.log`, `handoffs-recheck.log`, `status_selection.log`, `contract_text.log`.

| Sentence or heading | Worst ordinary state and result |
|---|---|
| Skip DM “No pick at this check” and journal “SKIP” | Empty selection after predictions. Both describe this one check. |
| Banner “LAST RECORDED NO-PICK SUMMARY” | Empty later check and restart. Earlier summary still correctly labeled. |
| K1/K4 candidate description | Stronger batter removed. The verified label names the actual decision candidate. Missing verification gives the top-model label. |
| K1 reason | Tail stop with a bar present. Reason wins and has one terminal ending. |
| K2 below using unrounded values | Values can round to the same percentage. The real comparison is strict. |
| K3 at or above | A 99% candidate can clear 82.5% and still skip. No pick instruction. |
| K4 no probability comparison | All bins skip or the bar is unavailable. No comparison implied. |
| K5 no candidate was chosen | No decision when games are final or statuses fail. No recommendation implied. |
| K5 top prediction, eligibility unverified | The 99.9% batter can already be final. The label makes the uncertainty explicit. |
| Skip streak suffix | Structured decisions use their stored streak. A legacy restart can falsely describe its display read as used. See E1. |
| Morning startup headline | Stale timestamp reported from the startup snapshot, including a future stamp. |
| Morning eligibility | A stale accepted snapshot can still be used. A dated correction supersedes it. |
| Morning conditional CLI write | Before lineup check with settled/absent prior entries it accepts today's row. It excludes grading pending, in-flight sends and yesterday's date. |
| Both warnings define DIR | Absolute path selects the scheduler even from a different cwd. A different directory reports its own state. |
| Morning and B8a say dated write leaves saver flag | Dated row changes today's decision input and leaves the independent flag. |
| Both warnings and B8a say read saver on D | Real reads distinguish active, not_earned, used and uninitialized. Jan 1 reads the new year, so the old DM's date condition prevents a write. |
| Both warnings and B8a say use for active/not_earned | Both transitions remove availability on the next automatic day. Already used needs no write. |
| Both warnings point uninitialized to B8a | B8a gives year recheck and exact init command. CLI initialized used in the intended directory. |
| Both warnings say check latest fetch status, B8a names the path | Real fetch writes it beside picks in `health_state`. A successful fetch with a future timestamp has no fetch failure. |
| Delivery stale headline | Confirmed send on stale automatic state, including one confirmed after midnight. |
| Delivery check MLB | Read only while a prior game is still live. |
| Delivery compare and command | The CLI accepts same-day input, with the pending-entry failure in B2. |
| Delivery sent pick and flag unchanged | Same-day correction keeps the pick and separate flag. Restart sends no duplicate. |
| Delivery write only when dates agree | Blocks re-dating at 00:05, but permits the same-day pending-entry write in B2. |
| Delivery later-slate pointer | B8a includes the midnight warning date. On a delayed read its settlement condition misses intervening slates. See B1. |
| Late headline | Equality is “had been reached.” Cutoff journal and state retain their prior timing. |
| Void headline | Refused game listed as Postponed when the real guard archives it. |
| Both “This pick was withheld” | No pick delivery or readable current pick file after refusal. |
| Both “A later check may produce another recommendation” | A later game can produce a new DM. Last check makes no promise. |
| Both “Use a later official pick message only while the entry window is open” | Real entry checker handles 19:30 before the replacement's cutoff and exits outside the window at 20:25. |
| Both “A manual entry can differ” | Old game's manual selection mismatched the late and void replacements in the fixture. |
| Both “The entry check may report a mismatch…” | Real CLI emitted mismatch in time and no second alert after the window. |
| B8a stale timestamp permits delivery | Stale alone allowed the send. Cutoff still blocked. See E3. |
| B8a confirmed automatic delivery record | Real confirmed send persists the automatic reading. |
| B8a dated delivery timestamp | Real dated send clears a seeded automatic record. |
| B8a startup/confirmed warning dedup | Morning success suppresses same-date delivery DM, while the delivery journal still writes. Midnight uses the new warning date. |
| B8a DIR/status/N/B/S/D definitions | Real CLI grants the dated write on D with numeric MLB fields and the explicit absolute picks directory. |
| B8a dated row comparison | C17 selects 3/14/used for comparison even with accepted automatic 12 and flag active. |
| B8a otherwise automatic comparison | Without a dated row the automatic streak/best and current-season flag are separately readable. |
| B8a morning write | Pending prior entry excludes correction. Settled before check permits it. |
| B8a Eric verifies after first check starts | Before the guard a late write can block, during the send it cannot amend the sent pick. The text routes to verification. |
| B8a same-slate warning | Correction leaves the sent pick and flag, but can install wrong contest state. See B2. |
| B8a later-slate wait | Correct for the adjacent settled slate in C18. Fails when the warning predates a later pending entry. See B1. |
| B8a later-slate compare and file | With settlement confirmed it files today's correction before lineup check. With B1's incomplete gate it accepts 13 instead of 14. |
| B8a uninitialized recheck/init/reread | Correct season initializes used. An already initialized flag stays intact without force. |
| B8a failure diagnostics | Condition requires a recorded failure. Successful future-stamp fixture takes no error action. |
| B8a future timestamp | Real warning names the future time and directs a clock check. |

## 3. Words only

`probe_parity.py` ran on the archived base and branch with both `bts.__file__` paths asserted. Across 600 summary inputs and eight full check returns, only the allowed candidate fields and additions differed. The notice/retry/restart and missed-pick gates agreed. Warning dispatch, date, dedup, status and exception boundaries agreed, as did the refusal journals, archives, sources and state. The only differences in those side effects were four hashes of changed bodies. `probe_all_bars.py` matched 85,148 reach-57/tail lookups across all reachable streak/day/saver combinations and each reachable tail best, including missing/untrusted values. [Diff](parity.diff) · [Base bars](bars-base.log) · [Branch bars](bars-head.log)

## 4. Bad data

Both wire directions read each other's files. Wrong types, nested JSON, huge numbers, NaN and invalid basis rendered safely. Banner payloads were escaped, missing or empty objects gave no banner, and damaged JSON was quarantined on both revisions. Only K2 uses the comparison template. E2 records the literal-data exception. [Matrix](bad-head.log)
## 5. Season walk

4,599 runs, zero problems. Per-record actions, bars and day counts equal `walk-bc143131.json.gz`. Former flags split into 229 K3 and 612 K4. Every field matched carried `walk-e` and a planted false comparison was detected. [Comparison](walk-compare.log)
## 6. Tests

All 314 requested tests passed in one process, using captured traffic and local socket pairs for real HTTP handler tests. All 22 required and two extra scratch changes were detected by assertions, with one source change per copy. No survivor. [Suite](suite.log) · [C14/C21 log](mutations-2.log)

## 7. Whole diff

Only the named B8a bullet changes in the ops checklist. `INCIDENT.md` and `strategy.py` have no diff. The walk harness change passes the real candidate to the summary builder. No tracked files changed during this review.

ran: C1–C21, sentence and CLI handoffs, base parity, bad-state wire probes, season walk, requested suite and 24 scratch changes; did not run: the whole fast suite, network/production calls or deferred change sets.
