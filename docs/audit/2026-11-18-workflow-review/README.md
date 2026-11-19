# Daily multi-model workflow review, 2026-11-18

Two fresh reviewers (a Claude subagent and a Codex session at the highest effort, memories off) each read `evidence-brief.md` (facts from the agent log and the landing history for 11-17 03:00 to 11-18 21:40 ET, without the lead's diagnosis) and answered separately: `opinion-claude.md`, `opinion-codex.md`.

## Where they agree
- **The release-wide check is the biggest open risk.** Nine change sets were each reviewed alone; nothing has run the season-start sequence end to end.
- **Round-1 reviewers rarely find the same defect** (Codex alone on 6a, 4c, 6b-2; Carbon alone on 4b-1; both on 6b-1), so a clean pair of SIGNs is weaker than it reads. Keep both readers on pick-path changes.
- **The F4 to F8 batch was the day's largest waste** (about 4.5 hours, one cron flag merged): five unrelated fixes in one change set, no contract check, and alarm changes treated as routine instrumentation.
- **The lead is an approval bottleneck**, and reading dialogs did not stop a hardcoded season date; the one-job rule is not enforced by anything.
- **Delivery:** a whole-range review signed on a frozen commit between 11-25 and 12-01, slipping to about 12-02 to 12-12 if the review finds something cross-cutting or the policy-bin finding or the bullpen measurement needs an owner decision and a build. Both note that deployability is finally shown only on the restored box (about 03-03 for a 03-24 opener).

## Where they differ
- Claude: rehearse a scripted opening week BEFORE the whole-range review, and split that review into slices by scenario. Codex: start the whole-range review tomorrow on a pinned candidate.
- Claude: freeze scope (no more discovery lanes) and give the lead a nightly stop; Codex does not raise either.

## What the lead adopts now (ruling R43)
1. **Scope freeze.** After the cache fix lands, `main` is tagged as the review candidate. No new discovery lane and no new fix enters it unless the whole-range review asks for it; new findings go to the ledger.
2. **The whole-range review is split into three slices by scenario,** each with a fresh reviewer who has seen no earlier round: (1) the season's first days from the restored box's state, which IS the rehearsal (its harness is adopted as a test afterwards); (2) a mid-season and a late-season day on the old and new code, the deploy, the rollback and cron; (3) interactions between change sets, clocks and years, fail-closed paths, and tests that cannot fail. Codex is the gate on each; Carbon reads slice 1 as well.
3. **Contracts gain standing hostile-input rows** (an interrupt mid-step, a malformed or non-regular file, midnight and the year boundary, a failed or unconfirmed send, a restart mid-step), since round-1 BLOCKs kept landing on those.
4. **A change that silences an alarm or loosens a verifier is tier A:** one piece per change set, a contract check, and a "must still catch" table written first.
5. **Independent fixes merge independently:** a clean piece of a batch lands when it is clean.
6. **The test wrapper takes a lock** so two jobs cannot overlap, whoever starts them (to be done when no job is running).

## For Eric
- The bullpen fix's effect on backtest numbers is unmeasured; both reviewers flag it. It is a multi-hour run: say whether it should run on the studio under the one-job rule (free, slow) or remotely (spending), and whether its result is a condition for the deploy.
- The late-season policy-bin finding (`../2026-11-18-proptest-lane2/README.md`, P1) needs a ruling.
- Work-policy additions 3 and 4 above are the lead's; overrule if you disagree.
