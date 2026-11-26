# Change set F1 (the entry watchdog and the wrong game of a doubleheader) STOPPED after round 2 (2026-11-25 20:28)

Session `bts-wr-fresh12`. Branch `fix/wr-f1` (pushed with this record, NOT merged): contract `kickoff-fix-f1.md` version 1 with Amendments 1 to 4, contract check `contract-check-f1/` (Codex xhigh, `WRCHKF1-d2c4d503`, BUILD WITH THESE EDITS), code commits `1d30de86` and `15b27474`. `main` is unchanged: `bts check-pick-entered` still confirms the right batter in the wrong game. Under R51 a second-round BLOCK is not overridden and the session builds no follow-up of its own (R57): the lead rules. The kickoff stops the session here; F5, F2, F3 and F4 were not started (a draft contract for F5 is at `~/bts-runs/draft-kickoff-fix-f5.md`).

## In plain words
The fix works when the watchdog's runs happen one after another, which is how cron runs them. The second review blocked on what happens when two runs OVERLAP: a run that started earlier and was stuck waiting for MLB can finish after a newer run and write its older picture over the newer run's "wrong game seen" note; a later run can then say "entry confirmed" without proof of the game. For that, one run has to still be going when the next starts 15 minutes later. Each of its requests has a 30-second limit, so this needs a badly stalled connection, not an ordinary evening; but nothing prevents it, and the review is right that the contract did not say so.

## The rounds
| Round | Codex xhigh (the gate) | Carbon High |
|---|---|---|
| 1 on `1d30de86` | **BLOCK** `WRF1REV1-d44caf34` (`review-fix-f1-codex/`): one finding (after a FAILED first wrong-game DM, the next different result overwrote the marker's reason and a later batter match with an unknown binding confirmed and sent the all-clear); five text edits, none blocking under R64 | **SIGN WITH EDITS** `WRF1CARB1-0bc79b95` (`review-fix-f1-carbon/`): no blocking finding; two contract-text edits (A6.4, A6.5); five scratch changes round 1's tests missed |
| 2 on `15b27474` | **BLOCK** `WRF1REV2-dffa862e` (`review-fix-f1-codex-r2/`): round 1's finding closed on its inputs and over 54 serial orders; ONE new finding, below; no edits | see "Carbon, round 2" at the end |

## Round 2's blocking finding (Codex; I did not re-run its probe: the stop came first)
**B1, overlapping runs.** 18:25 no entry, `initial` sent. Run A starts at 18:30, reads that marker and waits on a response. Run B at 18:45 sees the batter in the wrong game, its DM fails, it writes `alerted`/`wrong_game` with `game_unresolved: true`. A then resumes with its empty response and writes `alerted`/`no_pick` from its 18:30 picture, without the field. With a second capture that makes the binding unknown, the 19:00 run writes terminal `confirmed`, exits 0 and sends the all-clear. The marker is read once before the fetch (`cli.py:2004`) and every marker write uses that snapshot. The reviewer's fix: serialize the marker decision for a picks directory (after fetching, load the latest marker under a lock, decide A1 (b) and the tier from it, hold the lock through the marker write and the all-clear decision). Probe `review-fix-f1-codex-r2/interleaving.py` (it nests run B inside run A's captured fetch to control the order).

## What held in both rounds (two independent reviewers, three reviews so far)
- Outside a day with a wrong-game observation every observable is identical to the base: 5,510 paired runs in round 1 (4,690 identical, the rest exactly A1 and A2), 10,368 in round 2 (7,449 identical; 816 A1 (a), 480 A1 (b), 227 full-proof resolutions, 1,396 nonterminal updates carrying the new field); Carbon 4,969 scenarios per side.
- Rows N1 to N13 and M1 to M13 through the real command with real capture files; no false alarm and no miss in 36 doubles and the game-id type cases; the texts character for character, none with a countdown or an instruction to move an entry; the standing rows; the receipt (only `marker_status` and `exit_code` differ); 63 matched test files; in round 2 all six scratch changes are noticed by tests.
- The lead's ruling (Amendment 3: an unreadable capture gives "the batter is entered; the game could not be checked", terminal) behaved as written.

## Assessment (the session's reading)
1. **The class is "the marker is one file, read early and written late".** Round 1's finding was a later SERIAL run losing the observation; I changed the mechanism (the observation is its own field, written by the one marker writer) and round 2 confirmed that closed over every serial order it tried. Round 2's finding is the same loss through a CONCURRENT run. It exists on the base for the base's own tiers (the contract's A5.4 says only "overlapping runs can both send", which is true and incomplete: an overlapping run can also erase a newer marker).
2. **My own part:** A5.4 named overlapping runs and I wrote down only the harmless half. When I ruled the mechanism change for round 2 I asked the author for every ordered pair and triple of SERIAL results and did not ask what a stale snapshot does. The standing hostile-input rows have "a restart mid-step" and "an interrupt mid-step", and I answered both with "the command is one cron run" instead of asking whether two can be alive at once.
3. **How ordinary is it?** Each of the run's requests has a 30-second timeout and the login has two attempts; a run that outlives the next cron slot needs a stalled or trickling connection. The cron line has no singleton lock (the 01:00 grading line has `flock -n`).

## Ways forward (for the lead; R51, R57)
- **A singleton lock on the cron line, built with F5 (recommended).** F5 already changes exactly that cron line. `flock -n <lockfile>` there (the pattern of the `check-results` line) makes a second run exit at once while one is alive, so two watchdog runs cannot overlap on the box and the round-2 sequence cannot occur there. It is one more field on a line F5 rewrites, testable with the installer's existing test shims. F1's code then merges as it stands with the limit written truthfully ("a run started by hand while cron's run is alive can overwrite a newer marker"), on a fresh verdict limited to that claim, or on Eric's acceptance.
- **The reviewer's fix in code:** a lock file beside the marker, the marker re-read under it after the fetch. More code in the path that must never fail to alert (a lock that cannot be taken needs its own rule), and a third review round on new material.
- **Eric's acceptance of F1 as it stands,** with the overlap written as a known limit: the base has the same stale-snapshot overwrite today, and F1 is strictly better than the base on every serial order.
- **Leave F1 unmerged:** the base keeps confirming a known wrong game (BL-B7).

## Obligations if F1 continues
A gate result on the final commit (the gate on `15b27474` was running when this was written: see below); Carbon round 1's two contract-text edits (A6.4, A6.5) applied; Carbon's round-1 probes kept as the regression corpus for the scratch changes it found unnoticed (round 2's tests notice all six of Codex's); `CLAUDE.md` and the `FINDINGS.md` state table updated only on a merge.

## Gate
Round 1's gate (on `cd6fbf41`) was cancelled at 16% when round 1 blocked. The gate on the round-2 commit `15b27474` started 20:16 and was left running at the stop, because its result is needed if F1 lands as it stands; the result is appended below.

## Carbon, round 2
Still running when this record was written; its verdict is appended when it reports. It cannot change the stop: Codex's verdict is the gate.

## Gate on `15b27474` (appended 2026-11-25 20:50)
20:16 to 20:44 through the wrapper: 11,104 passed, failing set exactly the pinned 53 (`gate-failed.txt`). Hold 20:16 to 20:44.

## The lead's ruling R67 (2026-11-25, after this record)
F1b is F1's one follow-up: F1 as it stands plus a singleton `flock -n` on the `check-pick-entered` cron line, with a run limit shorter than the cron interval (the session's addition, agreed by the lead); the class named "two writers of the watchdog marker"; a must-still-catch table of both rounds' findings; fresh Codex xhigh and Carbon review, at most two rounds; if it stops, F1 goes to Eric. Contract: `kickoff-fix-f1b.md`.
