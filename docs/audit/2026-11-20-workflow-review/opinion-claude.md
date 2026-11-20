# Workflow review, Claude subagent (fresh, evidence brief only), 2026-11-20 about 13:35 ET

**Headline:** the opener is four months away, so calendar time is cheap and a wrong "done" is expensive. The workflow is tuned the other way: overnight sprints, stop-rule overrides to keep things landing, and a closing test that mostly re-checks what is already known.

## 1. Flaws, ranked by expected cost
1. The closing test is weak. Seven merges landed after the tagged candidate, each with only per-change review: the same kind of review that passed the lead's 11-13 day-count bug. Re-running four harnesses on the final head mostly proves the known findings stay fixed.
2. The whole-range review looks under-sampled. Three Codex slices took 17 to 20 minutes each and each returned exactly four findings (a stopping habit, not exhaustion). The one Carbon read (slice 1, 80 minutes) found two new ones. Slices 2 and 3 got no second read.
3. Review effort goes to exotic inputs (a NaN constant, a special file, a newline in a file name, a schema version). The costliest bug found was ordinary arithmetic on the normal path. Nothing walks a whole season of dates through the decision path.
4. The most worn session holds the override. The lead overrode the two-round stop twice; on C3v it rewrote an operator instruction and merged with no reviewer seeing the final text.
5. Unwatched stalls: thirteen hours lost on one permission prompt; the fix is a watcher that expires hourly and is re-armed by hand.
6. The backtests sit on the critical path (6 of 16 in 15.5 hours; 5.5 hours waiting behind gates; one run killed). A cloud run under 15 euros would have removed this.
7. Owner latency is unmanaged: three questions waited 6 to 15 hours; change set D burned two contract drafts before the owner picked a different design.
8. The single test slot runs on the honour system; one hold file has several writers.

## 2. False greens
- "BUILD WITH THESE EDITS" from the contract check: five of five times; four then drew a round-1 BLOCK and C3's design was unsound. Only "NOT YET" has carried information.
- Carbon SIGN: six times on a commit Codex blocked; it blocked only where Codex also blocked.
- Author "gate green": green before every BLOCK; the pinned 53 failures can never show a regression on this host.
- C1's one-round SIGN at 02:30 from a reviewer the content filter had cut off once.
- C2's confirmation SIGN: narrowed to two receipt fixes although each earlier round had found something new.
- C3's round-1 Codex report was written without its pytest and mutation runs.
- Change set D, when it passes: a deploy workflow and canary reviewed with no box to run them on.
- C3v merged reads as fixed, but the underlying fault remains and needs the owner's explicit acceptance.

## 3. Wasted motion
- C3: about 2.5 hours for a design the files on disk could never support; a one-hour "can this be decided from disk at all?" probe would have killed it.
- Carbon in round 2 and confirmation rounds changed no outcome.
- Up to three 26-minute gates per change set on the one slot.
- Code blocked by prose (C3v's second BLOCK was a checklist step).
- The cache and void fixes bundled, with no contract check, half withdrawn.
- About 22 of the 40 hours moved no review work.

## 4. Two changes for tomorrow
1. Change the closing test: after the last merge, one blind read of the final head by a reviewer who has seen no earlier round; give Carbon slices 2 and 3 with the time it had on slice 1; add one plain harness that runs every 2027 date through the decision path and checks the obvious (days left, a pick or a named skip).
2. Remove the two queues: move the remaining backtests to the cloud (owner's call); send the owner one batched decision list at a fixed time each day, with design choices before any contract is drafted.
(Third: a real lock in the test wrapper, with per-owner hold files.)

## 5. The fresh session under the lead
Working when watched, not proven better (sample of two; session 2 had the easier work). Changes: pre-grant the reads in the kickoff; a heartbeat line per session and a standing alarm on 45 minutes of silence; take the override away from the old lead (a second-round BLOCK goes to the owner or to a fresh reviewer who sees the final text); replace the lead itself with a fresh session from a written state file; a contract check for every change set, including ones built from a stop record; operator instructions reviewed as their own item, default "leave in place and report".

## 6. Delivery estimate
Likely 11-27 to 11-30; earliest 11-24; tail to about 12-04. Critical path: the backtests (evening of 11-21 to midday 11-22 with contention); the owner reads the result; the two scheduler timing fixes in series (about 60% chance at least one stops); the harness re-run plus a blind read (expect it to find something); owner sign-off on the leftovers. Set by owner latency and the backtest queue, not agent capacity. Thanksgiving on 11-26 falls inside the window.
