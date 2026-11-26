# Daily multi-model workflow review, 2026-11-21

Two fresh reviewers (a Claude subagent and a Codex session at the highest effort, memories off) each read `evidence-brief.md` (the agent log, landings and outcomes for 2026-11-20 13:30 to 2026-11-21 16:10 ET, without the lead's diagnosis) and answered separately: `opinion-claude.md`, `opinion-codex.md` (`WFREV3-acffcdd0`). Ran: one read of the brief each; nothing else. The previous review is `../2026-11-20-workflow-review/`.

## Where they agree
- **The test slot starved the owner's deploy-condition backtest:** nothing ran 06:30 to 16:04 on 11-21 while holds spanned author and review work. Holds must be set only for a job that is ready to run.
- **The deploy-workflow track keeps reaching code with its critical behaviour unproven** (D, D2, D3 all stopped; D2's checks said it needs real systemd and Debian facts). If D3b stops, it goes to Eric.
- **Carbon's SIGN is not a gate.** It signed while Codex blocked, more than once.
- The watcher's blocked-dialog path is still unverified; a quiet window proves nothing.

## Where they differ
- **The follow-up rule.** Claude: as used, a lead-ruled follow-up with fresh reviewers resets the round counter, "an override by another name"; W15b only merged after its own BLOCK found the same class again. Codex: W15b is the rule working, provided the parent BLOCK stays attached until every finding has a disposition.
- **D3b.** Claude: pause it and give Eric the choice (a real Debian test machine; accept; defer to January). Codex: continue, with the proof for each open blocker in the kickoff; to Eric if it blocks again.
- **Delivery.** Codex: 11-23 to 11-25. Claude: likely 12-03 to 12-08 (optimistic 11-27, pessimistic about 12-18), with one in four fresh code pieces getting through two rounds in this window.

## What the lead adopts now (ruling R57)
1. **One follow-up per stopped piece, then Eric.** A follow-up (W15b, D3b) names the defect CLASS, changes the mechanism for that class, and carries a "must still catch" table of every earlier finding plus new cases of the class, given to the fresh reviewers as inputs they must probe besides their own work. If the follow-up stops, the piece goes to Eric as accept, defer or drop; the lead does not rule a second follow-up. Each follow-up is reported to Eric when it opens (R55 and R56 were).
2. **D3b continues under item 1** (it is narrow and under way; every finding reproduces under shims): the class is "recovery claims a state it did not prove"; `Scheduler stopped: verified.` (and any other "verified" line) may be printed only from one proof function, and the signal handler's containment decision is a single named rule with its table. Eric is asked NOW what happens if D3b stops, so that answer does not wait on him later.
3. **The withdrawn fallback:** a Carbon-only verdict never merges a change (the 11-20 17:50 fallback is withdrawn; it was never used).
4. **The slot:** a hold only when a gate or a reviewer's focused run starts within 10 minutes, released when it ends; reviewers' focused runs batched behind one hold. Already in fresh session 6's kickoff; the same line goes into every later kickoff.
5. **The closing test's blind read also lists every operator-facing text on the final head that no review covered** (the Claude reviewer's point that the C3v step may not have been the only one).
6. **Delivery estimate the lead uses: the later one, early December (12-03 to 12-08).** The box restore is about March 3, so there is slack; it is spent on fixes that change mechanisms, not on more rounds.
7. **The watcher is checked by the lead at every report** (`pgrep -f watch_sessions.sh`), and its blocked-dialog path is tested at the next real blocked prompt or with a staged one.

## For Eric
- If D3b stops: accept its remaining findings as known limits, or defer the deploy workflow to January's rehearsal (today's `deploy.yml` limits and checklist B8b item 3 and B8c stand until then)?
