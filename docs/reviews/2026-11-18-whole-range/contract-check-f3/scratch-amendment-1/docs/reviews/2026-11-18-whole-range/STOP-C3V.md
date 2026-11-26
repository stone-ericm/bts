# STOP: whole-range change set C3v (reconcile names why the season replay did not run)

Stopped 2026-11-19 06:50 ET by the fresh session (`bts-wr-fresh`) after a second-round BLOCK, as its kickoff requires. **Branch `fix/wr-c3v` (head `6ff1e87e`, worktree `~/projects/bts-work/wr-fix-c3v`) is NOT merged.** Records: `kickoff-fix-c3v.md` (version 1, amendments 1 and 2), `review-brief-fix-c3v.md`, `review-brief-fix-c3v-r2.md`, `review-fix-c3v-codex/`, `review-fix-c3v-carbon/`, `review-fix-c3v-codex-r2/`, `review-fix-c3v-carbon-r2/`.

## Verdicts
| Round | Codex (the gate) | Carbon |
|---|---|---|
| 1, `cb3bc37c` | BLOCK, one finding (B1: a pick filename whose stem ends in a newline split the one-line stderr message) | SIGN, notes and an adopt file (four surviving mutants) |
| 2, `6ff1e87e` | BLOCK, one finding (B2: the checklist step) | SIGN, notes (the same checklist gap) and an adopt file (two further mutants) |

## What held (both reviewers, both rounds)
The CODE does what the contract says and nothing more. Over the 630-history corpus, the 18 entered-pick and writer sequences (round 1 Codex, round 2 Codex, Carbon round 2 of the stopped C3), and 366 to 916 hostile states per reviewer, the replay's return or exception and the bytes of `streak.json` after `bts reconcile` are identical to `main` (`88b69a16`); 0 divergences. The stderr line is one printable ASCII line for every stem the scan accepts; the receipt keeps the raw values; no existing test file is changed; every named mutant is killed. Gate on `f786f00a` and on `6ff1e87e`: failing set exactly `~/bts-runs/gate-failed.txt` (10,092 and 10,111 passed).

## Why it stopped (B2, Codex round 2; Carbon round 2 raised the same gap as a note)
The contract (row V5, written by this session) told the operator to MOVE a blocking pick file aside after confirming in MLB that nothing was entered, and amendment 2 added "unless `decision.json` says `"scoreable": true`". That is not enough: a private-mode pick (research intent) interrupted after `save_state` wrote `pick_locked: true` and before the decision record leaves NO MLB entry by design and a missing or `scoreable: false` decision. Following the step sets that committed day aside, and the next reconcile saves a streak without it (9 where 10 is right; `review-fix-c3v-codex-r2/operator.log`). Carbon also noted an unreadable decision and a pick file with delivery flags. This is the STOP-C3 lesson again, in the operator's hands: the files on disk cannot prove a pick was never committed.

## What a fresh session may do (obligations)
1. **Recommended (lead's decision):** the code of `fix/wr-c3v` can land with checklist row B8b item 2 replaced by a step that MOVES NOTHING: "`bts reconcile` prints `Season replay unavailable for <date> (<reason>); streak retained, not recalculated.` Leave every file in place and report the message to Claude; the model streak (not the contest streak, which decides live picks) stays as it was until the day is resolved." Any later instruction to set a file aside must first list every way a pick can be committed or entered without a durable record (private lock before the decision, lost DM acknowledgement, `bts run` and `orchestrate` interrupted after posting) and say how the operator detects each; until then, none. A docs-only change of that kind is tier D, but it changes what an operator does with the streak, so the lead should say whether it needs a review round.
2. Both reviewers' adopt files (`review-fix-c3v-carbon/adopt/` is already in the branch; `review-fix-c3v-carbon-r2/adopt/test_wr_c3v_r2_adopt.py` is not) and Codex's round-2 probes are kept for that landing.
3. Codex round 2 notes, not fixed: the printed escaped date (`2027-04-18\n`) cannot be pasted into a shell command to find an LF-named file (the receipt's raw value does identify it); an injected failure inside `ReconcileReceipt.replayed` prints the old wording (no real input reaches that).
4. Pre-existing, unchanged, from `STOP-C3.md` obligation 3: the date scan's silent drops on unreadable listings and directories, the FIFO hangs, the receipt's `outcome=completed` for a blocked replay with a corrupt `streak.json`.
