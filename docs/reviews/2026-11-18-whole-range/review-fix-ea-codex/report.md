# Ea operator text review

## Verdict

**BLOCK** at `7faef255`. The rendered strings match the written contract in the replayed cases. Six findings require changes to what the operator reads.

## Blocking findings

- **B1. The skip summary reports a streak the check never used.** On July 15 at 18:35 ET all three games are final. The real lineup selection returns no decision at streak 3. A dated correction to 12 lands before the scheduler’s separate display read. The DM, journal and dashboard say `Streak used at this check: 12.` The same claim appeared on status failure and with no selection. At `src/bts/scheduler.py:1800`, omit this sentence for `basis == "no_decision"`. Keep the stored field. [Surfaces](messages.log) · [Interleaved CLI write](handoffs.log)
- **B2. The late DM calls the cutoff itself “after” the cutoff.** At exactly 19:05:00 ET for a 19:10 game, the real guard archives the pick and the DM says `0 min after the 19:05 ET cutoff`. At `src/bts/scheduler.py:2508`, replace the first sentence with `LATE DELIVERY REFUSED: the {cutoff} cutoff for {names} had been reached.` [Captured refusal](operator.log)
- **B3. “MLB differs from status” gives conflicting answers on a dated-observation day.** After a 10:00 morning warning, `status` showed automatic 12/14/active and a dated 3/14/used already in effect. MLB shows 12/14/active. Comparing the automatic rows leaves the wrong reading in force before the first check. With a 76.0% top prediction the real policy recommended a double from that reading and skipped with the corrected reading. In `docs/ops/2027-season-start.md:31`, follow the status command definition with `When a dated manual row is in effect, compare MLB with that row’s streak, best and saver. Otherwise compare with the accepted automatic streak and best and the current-season saver flag.` [CLI output and decisions](status-selection.log)
- **B4. A warning sent across midnight has no clear correction path on the new slate.** A July 15 slate pick sent at 23:59:50 was confirmed July 16 at 00:00:05. Both dates printed in the warning were July 16, while the pick’s slate was July 15. The same-day instruction prohibits the write, and “for a later date” can send the operator past July 16. The grading probe settled the live game at 03:10. At 10:00 the CLI would accept the July 16 correction from 12 to 13 before its first check. At `src/bts/scheduler.py:98`, use `For a later slate date, see B8a.` At `docs/ops/2027-season-start.md:31`, start the later-date rule with `For any date later than the pick’s slate, including D when the warning was sent after midnight ET, wait until MLB has settled that slate’s entry or confirms there was none.` Keep the existing compare-and-file condition for the new slate. [Delivery and grading](handoffs.log)
- **B5. B8a sends the operator looking for a fetch failure that never happened.** A successful synthetic fetch recorded 09:00 ET and `last_error: null`. After the clock was corrected to 08:45, the real startup warning flagged a future timestamp. The last sentence of the changed B8a bullet says to find out why the fetch stopped from its failure messages. At `docs/ops/2027-season-start.md:31`, use `Read the latest fetch status in health_state/contest_streak_fetch_status.json beside the picks directory. If it reports a failure, read the fetch’s error messages. If the warning says the timestamp is in the future, check the scheduler host’s clock.` [Status and warning](operator.log)
- **B6. B8a promises automatic evidence for a confirmed dated delivery.** The real DM and public delivery tests confirm a pick chosen on a dated observation and persist `automatic_reading: null`, even if the pick previously had a stale automatic record. An operator looking for an automatic record would treat normal dated evidence as missing. At `docs/ops/2027-season-start.md:31`, use `A confirmed delivery using an automatic contest reading normally saves its timestamp status as automatic_reading. A delivery using a dated observation has no automatic timestamp.` [Real delivery cases, `test_dated_attempt_clears_seeded_record`](../../tests/test_stale_delivery_cs4c.py#L196) · [Suite](suite-offline.log)

## Notes

- The unchanged refusal journal also says the cutoff “passed 0.0 min ago” at equality. The refusal DMs are in scope. The journal is outside this review’s changed text. [Cutoff run](operator.log)
- The new tests pin the K5 streak sentence and the rounded late-refusal headline. The status probe shows the consequence of each comparison by running a real lineup decision. [Exact-text tests](suite-offline.log) · [CLI probe](status-selection.log)
- The sandbox rejected the dashboard test’s TCP bind. The full rerun carried actual HTTP requests over local socket pairs with DNS and outbound connects disabled. `nice -n 19` emitted a `setpriority` warning and the bounded runs completed. [First run](suite.log) · [Offline rerun](suite-offline.log)

## 1. Must still catch

| Row | What the real paths produced |
|---|---|
| C1 | Aug 30, streak 56, 99.0%. `at or above the streak-56 bar (82.5%).` |
| C2 | Jun 25, streak 56, 99.0%, every bin skips. `No probability comparison is available for this check.` |
| C3 | 99.9% Batter 0 was final. The 75.0% Batter 1 was `Candidate used in the decision`, below 79.6%. |
| C4 | All three no-decision routes name the top prediction as unverified and omit the bar comparison. The streak suffix is still false in these runs. B1. |
| C5 | Correcting the saver separately made the next automatic day see `used`. Exercised `active`, `not_earned`, `used`, and `uninitialized`. |
| C6 | At 00:05, the old printed date exits 2. The B8a wait protects the live slate in the usual read-after-midnight case. The cross-midnight send moves D itself. B4. |
| C7 | Both DMs carry the five common sentences and no manual-entry invitation. Both later recommendations were enterable. The late heading is false at equality. B2. |
| C8 | The morning DM says prior entries must be settled or absent. At 00:20, the pending prior slate leaves that condition false. |
| C9 | The morning DM says before the first check. A deliberate late write before the guard withheld the pick. A write inside the send left the outgoing streak 12. |
| C10 | Persisted `1e308` and `1.5e308` produced `?` for the bad probability and the no-comparison sentence. |
| C11 | The 18:55 check had no predictions. The real page showed the 18:35 summary under `LAST RECORDED NO-PICK SUMMARY`, including after a restart. |
| C12 | Raising candidate class lookup preserved the summary and left it unverified. A line-break-only reason did not produce `Reason: .`. |
| C13 | Decoded bad fields rendered without exceptions and were escaped in the banner. K2 alone supplied `below the` from a template. See item 4 for stored text. |
| C14 | All 19 single scratch changes caused assertion failures. |

Evidence for the table. [Skip surfaces](messages.log) · [Warnings and refusals](operator.log) · [Handoffs](handoffs.log) · [Decoded state](bad-head.log) · [Mutations](mutations.log)

## 2. Sentence replay and operator actions

Compared the full DM, journal and page banner for each skip case, the complete warning bodies and delivered warning journal, and both refusal bodies. Measured the longest enveloped texts at 875 (delivery) and 895 (morning) Unicode characters. The rows below cover shared sentences on each surface where they appear. [Exact outputs](messages.log) · [Commands and warnings](operator.log) · [Handoffs and entry checks](handoffs.log)

| Surface and sentence | Worst ordinary state and result |
|---|---|
| Skip DM opening `No pick at this check.` | At 18:35 no pick was returned. A later pick remains possible. No action is requested. |
| Dashboard heading `LAST RECORDED NO-PICK SUMMARY` | After the failed 18:55 check and restart, the heading still identifies the older record. |
| K1 candidate sentence | The tail stop result names the candidate actually passed to the decision. A stronger prediction cannot supply this name. |
| K1 `Reason: …` | The tail stop reason supplies the reason and its terminal punctuation. No entry instruction. |
| K2 comparison | A removed 99.9% prediction yields the 75.0% candidate and a 79.6% bar. A legacy summary labels eligibility unverified. Rounding to an equal display is qualified by “using unrounded values.” |
| K3 comparison | A skipped 99.0% candidate is above 82.5%. The wording also covers equality. Reaching the bar gives no instruction to enter. |
| K4 candidate sentence | The all-skip table produces a named candidate. This describes the policy input, with no instruction to use it. |
| K4 `No probability comparison…` | A missing bar or bad number gives no comparison. The operator is given no threshold to act on. |
| K5 `No candidate was chosen.` | No eligible batter, a failed status read, and `sel=None` all left the pick empty. |
| K5 `Top model prediction (eligibility unverified)…` | Final games can supply the top model row. It is explicitly unverified. No entry instruction. |
| Skip streak suffix, K1–K5 | K1–K4 real decisions carried their own streak. In K5 the later state read can supply a streak unseen by the policy. B1. |
| Delivery opening `The pick was delivered…` | Emitted after the transport returned and the pick was persisted. A journal copy can recur without a second DM. |
| Delivery `Check MLB.` | The read can expose a still-live prior slate. Following it changes no state. |
| Delivery `If MLB differs from status…` | A stale warning can outlive the reading it described. The referenced CLI exposes two readings with opposing answers. B3. |
| Delivery and morning `DIR is the absolute scheduler picks path.` | A quoted absolute path with spaces worked from the wrong working directory. A different picks path read its own uninitialized flag. |
| Delivery `The sent pick and saver flag stay as they were.` | A same-day observation changed the live reading and preserved the saved pick and separate flag. Restart did not resend it. |
| Delivery `Write only if…` | The old printed date fails after midnight. When the send itself crosses midnight, the slate check still forbids that direct write. |
| Delivery `For a later date, see B8a.` | The ordinary next-day rule waits for settlement. When confirmation moves D forward, it leaves the rest of D unclear. B4. |
| Delivery and morning `On D ET away from midnight…` | The read-only saver command distinguishes all four flags. New Year reads the new season, so an old warning’s date condition prevents acting then. |
| Delivery and morning `For active or not_earned, run…` | Both CLI transitions produced `used`. An already-used flag needed no write. The flag governed the next automatic day. |
| Delivery and morning `For uninitialized, see B8a.` | B8a supplies the season check and explicit initialization CLI. This initialized the intended flag. |
| Delivery and morning `Check the latest fetch status.` | The path in B8a was correct and held a successful fetch with a future timestamp. Its follow-up diagnosis is B5. |
| Morning opening `Morning check for D…` | The pinned startup produced an accepted automatic reading marked stale. The journal reported the same detail. |
| Morning `This reading is eligible if it remains in use.` | A dated correction before the first check replaced the automatic reading. The eligibility statement is conditional. |
| Morning `Before the first check, if prior entries…` | Prior slate pending means wait. Once the lineup check starts, following the timing guard avoids the pre-delivery refusal and mid-send stale choice tested in C9. The status comparison itself needs B3. |
| Morning `This leaves the saver flag unchanged.` | The dated observation set saver `used` for its own day while the separate CLI flag still showed the original state. |
| Late refusal opening | At the cutoff timestamp, the headline describes a time strictly after it. B2. |
| Postponed refusal opening | The guard identified the correct slate, game 101 and `Postponed`, then archived the pick. |
| Both refusals `This pick was withheld.` | No pick transport ran for either refused candidate. |
| Both refusals `A later check may produce…` | One test supplied a later eligible game and got a recommendation. With no later check there is no promise of one. |
| Both refusals `Use a later official pick message only while…` | The replacement for a 20:30 game was actionable at 19:30. The real entry check reported outside the window at 20:25. |
| Both refusals `A manual entry can differ…` | A captured MLB pending row for another player differed from the delivered replacement. |
| Both refusals `The entry check may report a mismatch…` | Real CLI sent the mismatch at 19:30 and sent no follow-up at cutoff. |
| B8a `After the switch: a stale automatic timestamp alone…` | A successful stale-reading delivery confirms that timestamp age alone did not prevent it. |
| B8a `A confirmed delivery records automatic_reading.` | Automatic delivery persisted the record. Confirmed dated delivery persisted `null`. B6. |
| B8a `Startup or confirmed delivery can send…` | Both warning routes used the same ET day and key. The other route did not send a second DM after successful dispatch. |
| B8a terms for `DIR`, `status`, `N`, `B`, `S`, `D` | The CLI accepted an absolute picks directory, MLB counts and each saver value. D came from the warning clock. The status definition fails to select an authority. B3. |
| B8a `On a morning warning…` | The tested command updated the reading before the first check, after settlement or absent entries. Comparing the wrong status row can leave 3/14/used active. B3. |
| B8a `If the first lineup check has started…` | Verification is requested before using a pending or delivered pick. A write during send did not change the already-composed message. |
| B8a `On a warning after a delivery…` | A same-day dated write left the sent pick and the saver flag intact. The later-date path is needed when dates differ. |
| B8a `For a later date…` | The prior game stayed live at 00:38 and local grading finished at 03:10. The later observation was accepted before the next check. B4 covers the unclear case when the later date equals D. |
| B8a `The saver: a dated observation…` | Separate flag read proved the observation did not write it. |
| B8a `On D ET away from midnight…` | On D the flag was read before acting. On Jan 1 the prior season’s flag could still be inspected explicitly and was not selected by default. |
| B8a `For active or not_earned…` | `--use` set the existing flag to `used` for the correct path. |
| B8a `For uninitialized…` | The explicit season initializer worked. The CLI already displays the actual final flag, which is the value to reread if the check raced. |
| B8a `Then read the latest fetch status…` | The given path worked. A valid success record offered no failure messages or reason for a stopped fetch. B5. |

## 3. Words only

Archived `63669c42` source and verified both `bts.__file__` paths. Compared 600 frame/candidate combinations, all return fields in eight lineup cases, same-day notice retry and restart, missed-pick suppression, warning level/source/key/status/date/dedup and fault boundaries, and both refusal archives, records and sources. Both revisions propagated skip-notice interrupts, contained them in the delivery warning, and let morning interrupts propagate. Status-file differences were the SHA-256 values of the changed bodies. All 23,884 bar lookups matched (19,180 reach-57, 4,704 tail). `strategy.py` had zero diff. [Base](parity-base.log) · [Branch](parity-head.log) · [Status-file diff](parity.diff)

## 4. Bad data

Passed the persisted wrong types, nested values, NaN/infinities, huge numbers, out-of-range probabilities and bars, HTML, line breaks, blank reason and legacy shape through the scheduler and real page. Both state-write/read directions worked. Only K2 emitted a comparison with `below the`. A literal copy of those words in corrupted free text (`Observations\nbelow the expected count`) was conserved as data in K1 on both surfaces. [Branch reads](bad-head.log) · [Base reads](bad-base-read.log)

## 5. Season walk

The complete walk produced 4,599 runs and zero problems. The action, bars and day counts matched `walk-bc143131.json.gz` record by record. The carried E evidence matched in full. The 841 former flags became 229 K3 and 612 K4. One planted in-season false “below” was flagged. [Run](walk.log) · [Comparison and planted record](walk-compare.log)

## 6. Tests

The specified 11 test paths ran in one pytest process. All 314 passed in the offline rerun. All 19 C14 mutations failed assertions: top prediction, equality, K2 label, both case orders, three journal versions, all three escapes, both saver instructions, both refusals, string probability, old banner, old pointer, and missing first-check guard. Each mutant used the isolated source copy and was restored before the next run. [Suite](suite-offline.log) · [Mutations](mutations.log)

## 7. Other diff observations

The tracked checkout is clean. The only changed checklist text is the B8a warning bullet. `INCIDENT.md` and the standing findings in R63 were outside this review. `git diff --check b1abc2bf HEAD` passed.

ran: pinned-ET scheduler, warning, grading, entry and CLI probes, archived-base parity and state compatibility, full season walk, specified suite and 19 scratch mutations; did not run: full fast suite, shared wrapper, external transport, live MLB, TCP bind outside the sandbox, or deployment.
