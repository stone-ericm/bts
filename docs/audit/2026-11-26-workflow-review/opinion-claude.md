# Workflow opinion, 2026-11-26 (Claude, independent; evidence brief only)

## 1. Flaws, ranked by expected cost
1. **The thing that admits code is the owner, not the review.** Of about eleven reviewed pieces in the window, only H2 and F1b passed review. Ea, G, H, the R52 tool and its extension all ended in BLOCK and then merged or ran because Eric accepted them. Most of about 25 owner questions were answered "1" (the lead's recommendation), several after midnight. So the real gate is the lead's recommendation plus a tired owner's yes. The two-round review is mostly a long, expensive way to produce a STOP file for Eric to sign. That is the highest-cost flaw because it looks like heavy assurance and is not.
2. **The deploy condition was passed by changing the experiment after it failed.** R52 FAILED on 8 seeds (two seasons below -1.0 pt). The owner then chose more seeds, and 24 seeds PASSED (lower bound -0.05). Adding samples after a failure is optional stopping. The fail was also not small: two seasons broke the per-season floor. The PASS record also says the mean max streak is 0.74 lower with the fix, which is the metric the contest is about. The tool that decided it was BLOCKed six times and run on acceptance.
3. **The closing test is stale and the head keeps moving.** It read FINAL bc143131 on 11-24. Since then the season walk, Ea, G, H, H2 and F1/F1b merged, and H is a policy change (a new streak rule from 57). Scope was supposed to be frozen. Instead the closing test's 19 findings became five or more new change sets, two of them policy. The review is creating work faster than it closes it.
4. **Sessions die by rule, not because they are failing.** Six delegated sessions started in four days. Five died at their third BLOCK after fewer than two pieces each, and BLOCK is the normal outcome here. Every restart costs a kickoff brief, context loss and handoff errors. The lead also ruled mid-stream (02:17) that two Ea checks "do not count" to keep one session alive. When the rule is reinterpreted to get the result you want, the rule is measuring the wrong thing.
5. **Codex is nearly out.** The weekly Codex allowance is about 98% used, and Codex's verdict is the gate. Every remaining piece is on the critical path behind that refill. Nobody shows a plan for it.
6. **Supervision gaps.** The daily workflow review did not run for four days. A delegated session sat idle for 22 hours and nobody noticed until the owner asked. The lead made its own errors: it typed clock times instead of reading the clock (twice), started a job while the single test slot was held, and pushed a commit during another session's gate, which plausibly caused the ".git/objects changed" flakes.

## 2. False greens
- **Merges on "owner acceptance" of BLOCKed code** (Ea, G, H). The BLOCK findings still hold; only the label changed.
- **H's gate caught a regression** that neither of the three contract checks nor either review round had run. It was fixed in 6 lines. Nothing shows that fix was reviewed. The reviews are not running the basic suite on what they read.
- **Gates called green with failures explained away:** Ea merged with "the pinned 53 plus one test that passes alone", and the R52 gate had two failures put down as flaky. "Passes alone" can mean an order-dependent bug. A flake that appears right after the lead pushed during a gate was never shown to be environmental.
- **Carbon SIGNs are weak evidence.** Carbon signed H round 1 and then BLOCKed round 2 on the same class. It signed G round 2, D3b round 2 and F1 round 2 while Codex blocked them. A Carbon SIGN next to a Codex BLOCK tells you little.
- **"BUILD WITH THESE EDITS" contract checks:** nearly every one returned this, and the pieces still BLOCKed in review. The checks are not catching what review catches.
- **R52 PASS** (see 1.2), and a closing test called green on a head that no longer exists.

## 3. Wasted motion
- **Carbon keeps reviewing after Codex has decided** ("for the record"), up to an hour per piece.
- **Hold-and-cancel churn.** Gates are queued behind backtests and then cancelled before they start when a review BLOCKs, at least eight times.
- **A docs commit for every BLOCK, stop, clock-time correction and ruling.** Records get more effort than the fixes.
- **154 GB of review copies** piled up until every job stopped.
- **Round-2 reviews re-hunt the whole branch** instead of checking the round-1 fixes. G BLOCKed twice on pre-existing decision-record behaviour that the change did not introduce.
- **Overnight decision-making.** The deadline is three months out (box restore about 03-03), yet owner rulings land at 00:45 and 02:17. Nothing in the schedule needs that pace, and it lowers the quality of the one gate that actually works (Eric).

## 4. Changes for tomorrow (F2b, F3, F4, F5b, closing re-run)
1. **Bound round 2 and send Eric one batch in daytime.** In round 2 the reviewer answers only two questions: did the change fix each round-1 finding, and did the fix add a defect. Pre-existing behaviour and new classes found elsewhere go to the ledger and are not a BLOCK (the tier and R64 rules already point this way). Before work starts, put to Eric in one daytime message, per piece: what he would accept if it stops, and whether F3/F4 are needed before deploy at all or can become checklist rows. That turns overnight "1"s into decisions made up front.
2. **Freeze the head, then run the closing test once, last.** Nothing merges while the re-run is underway. Its outcome is limited to "fix only if it can cause a wrong pick, wrong state, a lost alert or a lost day; otherwise ledger or accept". Without that limit the re-run spawns another E-to-H cycle. Also stop pushing to main during any gate, and treat "passes alone" as a failure until it is shown to be order-independent.

Also tomorrow: check when Codex's allowance refills, and schedule F5b/F2b for that time instead of starting them on the last 2%. Count BLOCKs per piece, not per session, so a session stops when its own quality drops, not when its pieces are hard.

## 5. Converging, or a long route to the owner?
For most pieces it is a long route to the owner. The rules converge on a STOP file, not on code that signs. One piece did converge: Ea's findings fell 8, 7, 6, 5, 3, 1, and the cap cut it off and sent it to Eric anyway. The other pieces kept finding new classes, which is not convergence. So the cap stops the piece that was converging and lets the stuck ones pass on acceptance. My change:
- Allow a third round only when the finding count strictly falls and there are no new classes.
- Send a piece to Eric after round 1 when its findings are pre-existing or design questions; only he can resolve those, and more rounds will not.
- Make "owner acceptance" list each open finding with the cost if it is wrong, so it reads as risk accepted, not as green.

## 6. Delivery estimate (whole-range review closed, closing test green on the final head)
- **Critical path:** Codex refill, then F5b and F2b (one follow-up each, likely BLOCK, then Eric), then F3 and F4 (contract unchecked, two rounds each), then the head freeze, then the closing re-run (about 5 hours of work; the last one took 26 hours wall-clock because of the idle gap), then triage of what it finds.
- **Best case 11-29:** Codex refills soon and the re-run finds nothing that needs a fix.
- **Likely 12-02 to 12-06:** one further round of re-run findings, handled under the limit in 4.2.
- **Bad case 12-12 to 12-20:** the re-run produces new change sets again under the current rules, or a policy piece reopens.
- All of these are well before 03-03. Schedule is not the binding risk; the false greens in section 2 are.
