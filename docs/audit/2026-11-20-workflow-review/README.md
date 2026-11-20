# Daily multi-model workflow review, 2026-11-20

Two fresh reviewers (a Claude subagent and a Codex session at the highest effort, memories off) each read `evidence-brief.md` (the agent log, landings and outcomes for 2026-11-18 21:40 to 2026-11-20 13:30 ET, without the lead's diagnosis) and answered separately: `opinion-claude.md`, `opinion-codex.md`. The previous review is `../2026-11-18-workflow-review/`.

## Where they agree
- **The weakest point was the lead's own override of the two-round stop.** C3v was merged minutes after a second-round BLOCK with the blocked operator step rewritten by the lead and seen by no reviewer. (C2's exception at least got a fresh confirmation review.)
- **Delegation had no liveness check:** a fresh session sat on a permission prompt for 13 hours; the watcher added afterwards expires hourly and is re-armed by hand.
- **The single test slot is run by convention** (time windows, process-name checks, one hold file with several writers), and it failed in small ways several times.
- **"Gate green" and a Carbon SIGN were both present before every Codex BLOCK:** neither is evidence of correctness by itself. A report cut off by the content filter can lack its test runs: record what was actually run beside the verdict.
- **Operator instructions are part of the change** and need their own review: three times a session wrote one that was unsafe in some state.

## Where they differ
- **Delivery.** Codex: 11-22 to 11-24, assuming clean rounds and prompt owner decisions. Claude: most likely 11-27 to 11-30 (earliest 11-24, tail to 12-04), with about a 60% chance that one of the two scheduler timing fixes stops, and Thanksgiving inside the window.
- **The closing test.** Claude: the planned re-run of four harnesses mostly re-proves known findings; add a blind read of the final head, give Carbon slices 2 and 3 (its read of slice 1 found two things Codex's did not), and add a plain harness that walks every 2027 date through the decision path. Codex: tie every finding to a fix or an owner acceptance and the harness results to the final commit.
- **The lead.** Claude: replace the long-lived lead with a fresh session from a written state file, and take the override away from it. Codex: keep test dispatch, the stop decision and the owner queue with the lead, with one shared record.
- **The backtests.** Claude: move the remaining ten to the cloud. Codex: an intended cost of sharing the machine.

## What the lead adopts now (ruling R51)
1. **No more overrides by the lead.** After a second-round BLOCK the change set stops; what lands next is a materially revised scope with a FRESH verdict on the changed material, or Eric's recorded acceptance. Text shown to the operator (checklist rows, alert wording, printed commands) is reviewed as part of the change it belongs to.
2. **Repair of the one unreviewed text:** the checklist step the lead wrote into the C3v merge (row B8b item 2, "move nothing and delete nothing") and its `CLAUDE.md` sentence get a fresh Codex review now.
3. **A standing watcher** (a detached script under `~/bts-runs/`, not hand-armed, not under `/tmp`): it answers the known read prompt, and it messages the lead's pane when a delegated session has been blocked for more than five minutes or the backtest runner has died.
4. **The closing test is widened:** after the last merge, (a) the four slice harnesses are re-run on the final head; (b) one blind read of the final head by a reviewer who has seen nothing; (c) Carbon reads slices 2 and 3; (d) a season-walk harness runs every 2027 date through the real decision path with the shipped policies and checks the plain things (days left, a pick or a named skip, the last day playable).
5. **Verdict records say what ran:** each archived review gets one line "ran: …; did not run: …".
6. **Owner questions go in one list a day,** design choices before any contract is drafted (change set D burned two drafts before the owner chose a different design).
7. **A real lock in the test wrapper** replaces the conventions, once the backtest queue is finished (the wrapper must not be edited while an instance runs).

## For Eric
- **Whether to replace the lead session with a fresh one** working from a written state file (the Claude reviewer's recommendation; this session has run since 11-15 and is the one he talks to).
- The backtest location is his decision of 11-19 and stands unless he reopens it; the Claude reviewer estimates the cloud would pull delivery in by about a day for under 15 euros.
- The stopped season-replay fault (one stale never-committed pick file disables the nightly model-streak rebuild; now visible, not fixed) needs his explicit acceptance to be closed.
