# BTS workflow opinion — November 21, 2026

## 1. Flaws ranked by expected cost

1. Test-slot priority is open-ended. Backtests are a deploy condition, yet none ran from 06:30 to 16:04 on November 21. The log shows a 9 h 34 min drought across W15b and D3. It gives no breakdown of time spent actually running tests. Broad holds put a known ten hours of remaining backtests behind an uncertain review schedule.
2. The deploy work keeps reaching implementation with critical behavior still to prove. D spent about five hours before it stopped. D2 needed systemd and Debian facts and stopped at contract review. D3 spent nearly eight hours from author start to its second BLOCK. The last findings could stop a healthy scheduler or incorrectly claim that recovery was verified. Those are consequential deploy failures even though they cannot send a pick.
3. The review record makes it too easy to confuse a verdict, a provider interruption, and a closed finding. W15 round two had a recorded Codex BLOCK despite interruptions before its marker. The workflow needs to carry that verdict forward. The clock errors also make it harder to reconstruct events. Using `date` addresses that specific problem.

## 2. False greens

- A commit gate pass covers the suite on the tested commit. The three round-two BLOCKs all confirmed the round-one fixes and still found release blockers. A contract verdict of BUILD WITH EDITS authorizes implementation once its edits are applied. Review and closing are still ahead.
- Carbon's W15 SIGN left a live Codex BLOCK. A filter stop is an interrupted review turn. If there is no verdict in the record, the Codex decision is pending. The suggested Carbon-only fallback for D was never used.
- D3 could report “verified” in two cases that warranted “unverified.” W15 had the opposite kind of false signal. Text in a warning could produce a false CRITICAL.
- The watcher proved that a simulated runner failure could reach the lead. Its blocked-dialog path remains unverified. A quiet window provides no evidence about how that path behaves.
- Eleven finished backtest runs establish progress. Completion of all 16 and the paired comparison are required to judge the result.

## 3. Wasted motion

- Reworking D2's contract without obtaining the missing host facts consumed three checks. Its early stop was cheaper than building another speculative version. The second round on D also found more defects in parsing and in a duplicate validator.
- Repeated attempts through a provider filter cost review time. D's fresh plain-worded retry delivered a verdict. Use that path promptly when the original reviewer cannot finish a report.
- Reserving the wrapper before a gated job is ready to run can burn the same time a backtest needs. The log shows long holds spanning review and author work, but it cannot tell us how much of each hold was reclaimable. Fresh reviewers, the short C3v text review, and decisions reserved to Eric all served a purpose.

## 4. Two changes tomorrow

1. Schedule the wrapper by ready jobs. Give the remaining backtests named two-hour slots whenever the host is available. Reserve a hold for a specific, ready wrapped job or a small queued batch, identify who releases it and when, and release it at completion. Let a backtest already running finish. Plan around the game engine and order the next jobs explicitly when it stops.
2. Put the proof for each open blocker in the next kickoff. For D3b, cover the signal window and each false “verified” path with a focused check that fails on the stopped head and passes on the candidate. Record the stopped head, open findings, repair commits, test evidence, and fresh reviewers' verdicts on the full candidate. Mark interruptions separately. If D3b blocks again, take the evidence to Eric before another attempt.

## 5. Round-two stops and fresh follow-ups

W15b is one example of the rule working. W15 stopped, the follow-up materially changed the candidate, fresh reviewers saw the whole branch, and fresh Codex found another path into the same regex before signing with edits. D stopped, D2 stopped at contract review, and D3 stopped. D3b has yet to earn a verdict.

The follow-up becomes an override if the blocked behavior can ship on a renamed branch without a material repair and fresh gate verdict or Eric's recorded acceptance. Keep the parent BLOCK attached until every open finding has that disposition. For SIGN WITH EDITS, record that the prescribed edits are on the candidate that gets merged. Use the same record at whole-range closing.

## 6. Delivery estimate

**November 23–25, 2026 ET** is the working range. It assumes D3b and W3/W4 stay narrow, the host makes time for the backtests, and Eric can dispose of any findings that need his decision. With uninterrupted access from the evidence cutoff, the five runs alone would take until about 02:10 ET on November 22. Allow roughly a day for W3/W4 and another half to full day for closing as planning allowances. The brief supplies no measured durations for that closing pass.

D3b and the backtests compete for the wrapper. After the backtests, complete the paired comparison and take W3 and W4 through their gates and reviews. Once every finding has landed or has Eric's recorded acceptance, close on the final head with the four harness reruns, blind read, Carbon on two slices, and the walk of every 2027 date. Any fix from closing moves the final head and requires a green closing pass on that head.

The estimate treats the January Debian rehearsal and full on-box hold decision as later deployment work. If either remains an open finding for this review and Eric keeps it open, closure moves to January or later. Another stopped change or sustained loss of host time also pushes the estimate out.
