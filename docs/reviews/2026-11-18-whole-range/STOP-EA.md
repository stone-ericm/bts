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
