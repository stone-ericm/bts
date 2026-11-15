# Design pass, part 2: attack the proposed design for the season-boundary streak and saver rule

Tier A (pick-path behaviour). You are an adversarial reviewer of a DESIGN, not of code. Assume the designer is wrong. No code is to be written or changed in tracked files; you may write and run probes in a scratch folder inside this copy.

## Background
Read `docs/reviews/2026-11-14-season-rollover/design/brief-carbon.md` (the designer's brief: the system, the problem, the cost order of the failure directions, the reading list) and `docs/reviews/2026-11-14-season-rollover/FRESH-START.md` (why three earlier rules failed, the open findings, what is not in dispute). Then read the proposal: `design-carbon.md` in the repository root of this copy.

The lead (the session that decides) wrote three rules that each failed review, and has a leading option of its own inside the designer's brief (item 5). Attack that option as hard as the designer's.

## What to do
1. **Break the state space.** Find states the design does not list or lists as unreachable, and show each is reachable: a sequence of real events (fetch outcomes, operator actions, restores, restarts, clock changes, picks settling or not) that produces it on the real code as it is on `main`. Run the sequence where you can and say what you ran.
2. **Break the decision table.** For every row, look for a state in which the row's action is wrong in one of the three failure directions, in the owner's cost order: refuses every pick with no recovery; picks on a stale non-zero streak; treats the saver as spent or in reserve wrongly. Give the concrete state, what the design does, and what a correct system would do. Use the probes in `docs/reviews/2026-11-14-season-rollover/probes*/` as raw material, and check that each of the earlier counterexamples (Codex r2 B1 to B4, Carbon r2 B1 to B3, audit F1 to F4) is closed by this design and not just renamed.
3. **Break the evidence.** For each piece of evidence the design relies on: can it be absent, stale, written by a failed run, overwritten by a later weaker observation, or wrong after a restore from backup? Does anything still infer the contest's account state from the presence or absence of local files?
4. **Break the unknowns.** The design must behave acceptably whatever the contest's profile and rounds endpoints do at the season boundary, which nobody has observed. Construct the endpoint behaviours that hurt it most, including ones the design does not consider, and say what happens on Opening Night (2027-03-24, one game) and on each of the next ten days.
5. **Break the operator steps.** Any step a person must take: what if it is skipped, done late, done twice, done on the wrong day, or left in place too long? What does the system tell the operator, when, and through which channel that exists today?
6. **Compare the alternatives** the design offers, and name the one you would build, or a better one of your own with its table. State the cost of being wrong for each.
7. **False greens.** For the tests the design proposes: which could pass while the behaviour is wrong?

## Deliverable
Write `design-attack-codex.md` in the repository root of this copy:
- Verdict on the design as proposed: BUILD AS IS, BUILD WITH THESE CHANGES (list them, each tied to a counterexample), or DO NOT BUILD (and what instead).
- Findings, most severe first, each with: the state or sequence, what the design does, what should happen, whether you demonstrated it by running something, and the smallest change to the design that closes it.
- The states you tried and could not break.
- Anything that needs the owner (a decision about operator steps, a request to the real contest account) rather than a design change.

Do not contact the contest, the box or any network service. Keep the report under 300 lines. When the file is complete, print `SRD2-<first 8 hex characters of the sha256 of design-attack-codex.md>` and then `DONE`.
