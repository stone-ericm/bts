# Independent workflow review

## 1. Flaws ranked by expected cost

1. **The release-wide check is still outstanding.** About 2,300 changed source lines await a fresh review against the last deployed version. A defect where individually signed changes meet could cause a wrong pick or a missed day. The same-day feature leak found by a short discovery run shows the value of checking behavior beyond the current patch.
2. **Contract and review decisions can get ahead of the version they cover.** The 4c author hit conflicting contract rows. The 6b-2 author started version 2 immediately after a NOT YET on draft 1, with no recorded recheck of the revised contract in the brief. Reviewers later caught a hardcoded season date the lead had approved. Close rejected or conflicting requirements before building and record which revision the approval covers.
3. **Bundling unrelated fixes turns local defects into hours of shared rework.** The F4–F8 batch occupied roughly 4½ hours through the fresh-session handoff, and only F6 eventually merged. F4 could have suppressed real failures, so treating all alert changes as routine instrumentation understates the in-season cost.
4. **The lead is an approval bottleneck, and approvals cannot enforce the test limit.** Five scratch commit requests in ten minutes consumed attention while sandboxed probes and a failed wrapper attempt showed how tests can still overlap. The all-night pace is a poor basis for forecasting. The allowance figures give little evidence that model quota was the binding constraint in this window.

## 2. False greens

- A passing suite establishes the cases it exercises. The clean 6b-2 run still left a missing review verdict. Carbon's signed review under R38 was the evidence that closed that gate. The empty F6 harness output established nothing because it used the wrong layout. The adopted test supplied the evidence for that check.
- One reviewer’s SIGN can hide an open finding. Carbon alone caught the 4b-1 display cases, Codex alone found blockers on 6a, 4c and 6b-2, and both flagged one issue on 6b-1. Treat SIGN WITH EDITS as pending until the edit is verified. Confirm any later behavior change on the candidate that will actually merge, as the team did on 6a.
- A new session provides fresh eyes, while unresolved obligations carry forward. F8 drew another BLOCK after the reset. The two-round cap and three-BLOCK stop rule also need a clear counting rule when two reviewers report in one round.

## 3. Wasted motion

- Keep one daily review appointment for tier B, with separate merge decisions for independent fixes. Give alarm changes that can hide real failures a short behavior contract up front. When a piece is blocked, carry it forward separately so a clean change can land.
- Skip routine scratch commits and reuse the recorded full-suite result for an identical candidate. Give reviewers the shared gate evidence and reserve the single test slot for changed candidates and targeted probes. Keep both adversarial readers on pick-path changes. The observed disagreements are worth more than the time saved by dropping either reader.

## 4. Two changes tomorrow

1. **Start the whole-range review on November 19.** Pin the last deployed base and a release candidate, and give them to a reviewer who has seen none of the earlier rounds. Settle which pending fixes enter the candidate and record every subsequent change for review on the final head. Get Eric's ruling on the late-season policy bins in parallel.
2. **Enforce the one-job rule at launch.** Put full suites and short probes behind one shared queue that takes the slot before spawning a test child and holds it until the child exits. Make direct test runs use the same queue, including sandboxed runs and retries. Preserve the current approval boundaries.

## 5. Delivery estimate

**Deployable code November 25–December 1 is realistic if the whole-range review starts November 19 and finds localized fixes.** Allow one to two working days for the first pass and roughly three to five more for triage, repairs, the serial gate and independent confirmation. The observed pace came from small slices and overnight work. If the review finds a cross-cutting issue or Eric’s policy decision requires implementation and another review, pencil in December 2–8.

The critical path runs from the pinned candidate through fresh whole-range review, fixes, the passing gate and confirmation on the final head, with any release-affecting owner decision resolved before sign-off. Schedule the multi-hour bullpen backtest around test capacity. Its result joins the critical path if Eric makes it a release condition. If March 24 is confirmed, restore and season-start checks begin around March 3, with deployment subject to those checks and Eric's approval. A lockout moves the restore date with the opener.
