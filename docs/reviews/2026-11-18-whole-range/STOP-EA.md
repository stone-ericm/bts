# STOP: change set Ea (the operator messages only, R63) after its second and last round (2026-11-24 22:03, `bts-wr-fresh10`)

**Not merged.** Branch `fix/wr-ea` (pushed): code `7faef255` (round 1) and `1ed57c47` (round 2), contract `kickoff-fix-ea.md` with both rounds recorded. `main`'s `src/`, checklist and `INCIDENT.md` are unchanged, so everything `STOP-E.md` lists as still holding on `main` still holds.

## What happened
- Round 1 (`bts-wr-fresh9`): Codex xhigh BLOCK `WREAREV1-21601b4a`, six findings.
- Round 2 (`bts-wr-fresh10`), under the verdict rule R64: B3 and B4 of round 1 reproduced first; all six findings built (three message changes by a Codex author, tests RED first; four sentences of the one B8a bullet). Codex xhigh: **BLOCK `WREAREV2-06022bce`**: two blocking findings and three edits (`review-fix-ea-codex-r2/report.md`; summary in `kickoff-fix-ea.md`, "Round 2 result").
- Both blocking findings are the same thing: the delivery warning and its B8a sentences let the operator write a dated observation while an earlier slate's entry is still pending; when that entry settles, the dated row (now one too low) outranks the correct automatic reading for the rest of the day, and that day's pick is chosen on the wrong streak. The morning warning already carries this condition (`if prior entries are settled or absent`, round 1 of E); the delivery warning and the later-slate rule do not.
- No gate has run on any Ea commit. No Carbon review in round 2.

## What held in round 2
All 21 must-still-catch rows; every rendered string equal to the contract; nothing but words changed against `63669c42`; bad data; the season walk; 314 focused tests; 24 of 24 scratch changes noticed.

## The trend
New findings per round of this text: 8, 7, 6, then 5 (2 blocking under R64, 3 edits). None repeated. Each round's findings are narrower than the last; round 2's two are real under R64 (a wrong contest-state write).

## Ways forward (for Eric; R63 says the messages go to him)
1. **Apply the reviewer's five texts and merge on a gate plus a check limited to those sentences.** The two blocking replacements add the same "prior entries settled or absent" condition the morning warning already has; the delivery message stays under 900 characters (898). Cost if wrong: a fifth round might have found a sixth sentence; the applied sentences are the reviewer's own.
2. **Leave `main` as it is until January** and take the messages with the documents (Eb) and the Debian rehearsal. Cost: the season's first weeks would run on messages with the faults `STOP-E.md` lists (the skip notice's "below", "enter a pick manually", "use the current ET date") if nothing lands before the restore; nothing is deployed before then anyway.
3. **Merge only the parts no round has faulted** (the skip notice and banner, the two refusal DMs) and leave the two stale-reading warnings and the B8a bullet for later. Cost: a third contract for a subset; the stale-reading warnings keep `main`'s text.

The session's recommendation: 1. It is not the session's or the lead's to take (R51: no override of a second-round BLOCK).

## 2026-11-25 00:32 (`bts-wr-fresh11`): option 1 was built; its check is a BLOCK. Not merged; for Eric
Eric's option 1 was carried out on `fix/wr-ea` (pushed, head `be65e087`; code `988708df` on top of `7faef255` and `1ed57c47`): the reviewers' replacement texts applied as written (the four B8a sentences by the previous session, the four message items by a Codex author, tests first), then ONE fresh Codex xhigh check limited to the changed sentences under R64. **Verdict: BLOCK `WREACHK3-f254bd86`** (`review-fix-ea-check/` on the branch; summary in `kickoff-fix-ea.md`, "Follow-up result"). The kickoff says a BLOCK here stops and goes to Eric. No gate has run on any Ea commit (queued, cancelled before it started).

**In plain words.** Round 2 blocked because the checklist let the operator write a correction while an earlier day's entry was still unsettled. The fix was "wait until ALL earlier entries are settled". The new check found that this wait can now last past the day's first lineup check (a suspended game), and that the sentence for "after the first check has started" only says to have Eric verify a pending or delivered pick. If that first check produced no pick, there is nothing to verify, nothing gets filed, and the next check picks on the stale reading (in the replay: a double sent on streak 0 where MLB's 15 calls for a skip). I re-ran the reviewer's replay and it reproduces. It needs three things at once: a stale automatic reading (the fetch has been failing for 14 hours or more), an earlier entry unsettled past the first check, and a first check that produced no pick.

**What held:** every other changed sentence; the code change is words only; both warnings under 900 characters (892 and 889); 252 focused tests; all five scratch changes noticed. The one blocking finding is in the checklist bullet (B8a), not in a message; the two edits are one message sentence with its B8a twin (clearer after a send that crosses midnight) and one line of code (`below themodel` rendered as `under themodel`). The reviewer wrote a replacement for each.

**The trend:** new findings per round 8, 7, 6, 5, 3; blocking under R64: 2, then 1. The blocking finding is again in a sentence written to fix the round before.

### Ways forward (Eric's to choose)
1. **Apply the reviewer's three texts, run the gate, merge; no further check** (recommended). The blocking replacement adds one sentence ("If the state differs while the slate is unlocked and a later lineup check or fallback refresh remains, have Eric run the same command between checks"). Cost if wrong: that sentence and the two edits ship read only by their author (the reviewer) and me; five rounds say a sixth reader would probably find another edge sentence, most likely in the B8a bullet. About one hour (author 10 minutes, gate 30 once the slot is free).
2. **The same, plus one more check limited to those three texts.** Cost: about 30 minutes more, and on this trend a real chance of another BLOCK and another trip to you.
3. **Merge the messages without the B8a bullet's later-date rules.** Not recommended: the messages say "see B8a" for a later date, so the bullet has to say something there.
4. **Leave `main` as it is until January** and take the messages with the documents (Eb) and the Debian rehearsal. Cost: `main` keeps the message faults `STOP-E.md` lists; nothing is deployed before then anyway.
