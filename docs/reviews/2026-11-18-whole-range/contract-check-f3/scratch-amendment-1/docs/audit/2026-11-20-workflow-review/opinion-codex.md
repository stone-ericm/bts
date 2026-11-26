# Whole-range workflow review

## 1. Flaws ranked by expected cost

1. The stop rule is weakest at the merge decision. C3v landed three minutes after a second-round BLOCK,
   with its unsafe checklist step replaced and no fresh sign-off. C2b also got an exception, but both
   reviewers signed the revised receipt fix. Treat those as different levels of assurance. Operator
   instructions can change production state, so review them as part of the change they accompany.
2. Delegation has a liveness gap. A routine permission prompt stopped work for 13 hours, and the
   hourly manual re-arm leaves the same failure possible. Three owner questions separately waited
   6–15 hours. Track who can unblock each wait and when it needs escalation.
3. The test slot is managed by convention. Time windows, process-name checks and a hold file failed
   to give everyone a reliable view of the slot. The watcher matched its own job checks, overlap was
   explicitly allowed, and about 5.5 hours of backtest wait accumulated. The restart cost a partial run.
4. The contract check saves work but misses decisions that belong before implementation. A dropped two
   unsafe rows at the check and still needed three amendments during the build. C3 reached round two
   before the team stopped a design that relied on disk evidence that an entered pick might never leave.

## 2. False greens

- The author gates for C2b and C1 matched the pinned 53 failures. That shows baseline parity for
  those commits. It leaves the four final-head harness results outstanding.
- A passing author gate says little about conditions the suite never exercised. C3’s gates passed
  and both reviewers still blocked the design. Carbon’s SIGN on C3v and C2 left the Codex blocks unresolved.
- “Deployable after these fixes” is a list of conditions. The sliced review still needs each finding
  tied to a fix or owner acceptance and the four harness results tied to the final commit.
- A completed review report can have incomplete verification. One filtered Codex session finished
  without its planned test and mutation runs. Record what was actually run alongside the verdict.

## 3. Wasted motion

The clearest losses were 13 hours at a routine prompt, idle authors holding agent slots through
review, and six minutes before the lead freed the slot from a stuck test. The backtest queue also
waited for scheduled gates, an intended consequence of sharing the machine. Several review rounds
rediscovered gaps in malformed state and operator instructions. The contract checks that cut unsafe
scope and Carbon’s two new findings in slice 1 earned their time.

## 4. Two changes tomorrow

- Have the lead keep one finding-to-merge record. Before BUILD, require concrete expected outcomes
  for missing or malformed state and for each requested operator action. Record exact reviewed commits,
  gate evidence, and open obligations. After a second BLOCK, carry those obligations into a materially
  revised scope and get a fresh verdict on the changed material before merging. Record any acceptance
  of a finding as Eric’s decision.
- Give the lead one shared dispatch queue with an exclusive test slot across all runners. Publish the
  active job and next reservation, let a running backtest finish, and reserve needed gates before
  starting the next long run. Add a persistent alarm for permission prompts, stalled jobs and missed
  check-ins, with an initial alert within 10 minutes and a named person responsible for responding.

## 5. Fresh Claude sessions

Keep fresh sessions for bounded design resets, with reviewers briefed independently. The second
session got W22 and W20 merged and correctly parked D after two failed contract checks. The first
stalled, and the lead made the exception that landed C3v. The third is too early to judge. Give each
delegate one bounded outcome and a written handoff, and free the author’s slot once the review commit
is fixed. Keep test dispatch, the stop decision and the owner queue with the lead. The lead’s Claude
balance fell 18 percentage points during the window. Use the shared record to reduce repeated handoffs.

## 6. Delivery estimate

Plan for November 22–24 ET. Ten backtests at roughly two hours each put their uninterrupted finish
around 09:30 ET November 21. Then come the two scheduler timing fixes, by owner decision. Advance D
and the cache fix in parallel where the shared slot permits. The last of those fixes, the scheduler
fixes and any remaining finding dispositions determine when all four harnesses can run on the final
head. Confirm the owner’s disposition of the policy-artifact finding and reconcile the stopped
replay’s obligations against the original findings. The range assumes ordinary review turnaround,
prompt owner decisions and a clean final harness pass. A new design stop or harness finding pushes
closure beyond it.
