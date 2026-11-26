# Opinion (Claude, independent reader of evidence-brief.md only), 2026-11-21

## 1. Flaws, ranked by expected cost
1. **Deploy-workflow code is being built and reviewed without the system it runs on.** D, D2, D3 used about 13 of the window's working hours and all three stopped. Every round-2 finding depends on the environment: text parsing of crontab and units, a signal timing window, whether recovery is "verified". D2's own contract check said "needs real systemd/Debian facts". No gate on a Mac can settle those, and two reviewers reading code are standing in for a test machine. D3b continues the same loop. Expect a fourth stop or a weak pass.
2. **The owner's deploy condition sits on the critical path and is starved.** The paired backtest was idle 06:30 to 16:04 today (about 9.5 h, so about 4 to 5 runs lost). The holds were set when an author started, not when a test job was actually queued (08:08 hold, gate at 09:22). The review work holding the slot was the stalled deploy track from item 1. The cost here is the review's closing date, not effort.
3. **Fixes patch the single case, not the class, and the follow-up rule hides that.** W15's round-2 finding was a path in alert text matching an out-of-memory regex. W15b's fresh round 1 found another text path into the same regex. D's round 2 was all parsing, and D3's round 2 found two more "verified versus unverified" spots. Each round confirms the last batch of fixes, then finds the next case of the same defect.
4. **Operator text merged without review is a live hazard.** The C3v step the old lead merged unreviewed drew 5 MUST FIX findings. Nothing in the brief says anyone has looked for other unreviewed operator text on main. D3's "verified" where "unverified" is right is the same kind of defect: wording that looks reassuring when it should not.
5. **Owner latency is not planned around.** One answer took about 4 h overnight (22:40 to 03:00). Owner-only gates arrive one at a time instead of in a batch with a recommendation each.

## 2. False greens
- **"Every round-2 BLOCK confirmed all round-1 fixes."** This sounds like progress, but it only confirms the specific cases fixed. Item 3 shows the defect class survived.
- **Carbon SIGN.** Carbon signed W15 round 2, W15b round 1 and D round 1's fixed state, while Codex blocked each time. The 17:47 ruling allowed a Carbon-only fallback on a Carbon SIGN, which is exactly the weaker signal. Withdraw that ruling.
- **Review verdicts with nothing run.** wr-d-codex2 had "pytest/mutation not run at the lead's request", and reviewers were told "no pytest before X". A BLOCK reached that way is fine. A SIGN reached that way would have been a read-only opinion counted as a gate.
- **The watcher.** Only its runner-down path was tested. Its blocked-dialog path has never fired, and "no session found stuck" says nothing about whether it works. Nothing in the brief says anyone watches the watcher: if the loop dies, it fails silently.
- **"Gate green" for deploy code.** The fast suite on macOS does not run systemd, cron or Debian. For D3b, green means the mocks agree with the author.
- **Clock times in the records.** Records were stamped ahead of the real clock three times. Earlier records written before the `date` rule may hold wrong times. Do not use their timestamps as evidence of order.

## 3. Wasted motion
- **D2's contract versions v1 to v3 and D3's v7 contract.** The checks kept answering "needs facts" and were answered with more prose instead of facts.
- **Slot holds set ahead of need.** These cost backtest hours for no gain.
- **Repeated contract checks.** Almost every contract went NOT YET, NOT YET, BUILD WITH EDITS. Authors write the contract before reading the code paths the checker then reads. Run one quick "list every caller and every writer" pass by Codex first.
- **Ramp-up for every piece.** Each piece gets a new session plus a kickoff plus two new reviewers. That is right for independence, but W15 to W15b to merge took about 11 h for a cache-entry health warning.

## 4. Two changes for tomorrow
A. **Stop building deploy-workflow code until it can run somewhere real.** Pause D3b after its contract. Ask Eric for one decision with options:
   - (i) a throwaway Debian 12 with systemd (local container or VM on the Studio, or the cheapest cloud VM for a day; spending is his call), where D3's round-2 cases are reproduced first and then fixed;
   - (ii) recorded acceptance of D3's remaining findings, none of which can send a pick, with the "verified" wording replaced by "unverified" as text-only;
   - (iii) defer the deploy workflow to the January rehearsal and accept today's `deploy.yml` limits until then.
   Spend the freed slot and sessions on W3 and W4 (scheduler timing), which are on the pick path.
B. **Give the backtest the slot by default.** Set the hold only when a gate or test is ready to start within 10 minutes, and run gates in planned windows between 2 h runs. Better still, ask Eric whether the backtest can run on another machine, so the deploy condition stops sharing a slot with review.

## 5. "Stop after round 2, then a small follow-up with fresh reviewers"
As used, this resets the round counter. It is an override by another name, softened because the reviewers change. The lead, who may not override a round-2 BLOCK, is the one ruling each follow-up (R55, R56). Fresh reviewers do not inherit the round-2 findings, so they re-earn them by luck. W15b did merge, but only after its own BLOCK found the same class again: a third and fourth round in practice. To keep the rule honest:
- **The follow-up must name the defect class and change the mechanism.** For example, the health classifier must stop matching free alert text, or recovery may print "verified" only from one proof function. A "must still catch" table, holding every earlier finding plus new cases of the class, must be written before an author starts.
- **Fresh reviewers get that table as must-probe inputs,** in addition to their own blind work.
- **One follow-up per piece.** If it stops, the decision goes to Eric as accept or drop, not to the lead.
- **Each follow-up is reported to Eric when it opens.** It should not be ledgered quietly as a ruling.

## 6. Delivery estimate: whole-range review closed, closing test green on the final head
- **Critical path:**
  - the single test slot: 5 backtest runs (about 10 h), plus gates for D3b, W3 and W4, plus the reviewers' focused runs, all done one after another;
  - then W3 and W4 (tier A; this window's base rate is that 1 of 4 fresh code pieces got through two rounds; expect a follow-up each);
  - then the closing test (four harnesses, a blind read, Carbon on two slices, a walk of every 2027 date);
  - the paired comparison can fail and hold everything for Eric.
- **Estimate:**
  - optimistic 2026-11-27, if D3b is accepted or deferred and W3 and W4 land in two rounds;
  - likely 2026-12-03 to 12-08;
  - pessimistic about 12-18, if the blind read reopens items (the last whole-range read produced about 20 findings) or the deploy track keeps cycling without a real machine.
- **The biggest lever is A above:** taking the deploy workflow off the path removes the largest and least convergent queue item.
- **Slack:** the box is not restored before about March 3, so the schedule has slack. Use it to make fixes structural, not to justify more follow-up rounds.
