# Design pass, part 5: revise the build specification after its check

Your `design-v2.md` was checked (`design-v2-check-codex.md` in the repository root of this copy; verdict NOT YET for change sets 4 to 6) and the lead has ruled on each finding (`docs/reviews/2026-11-14-season-rollover/design/RULING-R25.md`). Read both in full, then `docs/reviews/2026-11-14-season-rollover/build/kickoff-cs1.md`, which is the contract change set 1 is being built from right now: your CS1 must match it.

Write `design-v3.md` in the repository root: the whole specification again, self-contained, so an implementer reads one document. The rules of the earlier briefs apply (real code for every claim about current behaviour, say what you ran, no network, "unsure" where unsure, disagree openly with a counterexample if you think a ruling is wrong).

What must be different from version 2:
1. Every change set carries a compatibility line: what it still reads and still writes from before it, and which existing tests it changes and why. Walk each merge boundary and state that the system picks correctly with only the earlier sets merged.
2. The restore-state file, the restore command and the 36-hour rule are gone (R25 C4, C5). Say plainly what the code cannot detect.
3. The decision table is total: enumerate the inputs and show that every combination lands on exactly one row, including research and preview, small future timestamps, and a status file that disagrees with the accepted observation.
4. `retry-pick`: results, exit codes, the latest-start arithmetic from the scheduler's real budget and reserve, and the shared check transaction with the daemon (R25 C2, C3).
5. Alerts: the three rules of R25 C7, with the dispatcher change they need.
6. The switch to automatic: the manual-observation history, the attestation, the write order and crash behaviour, and the endpoint behaviours the contract does and does not support (R25 C8).
7. Every command in a recovery column exists on `main` (say where) or is specified in a named change set.
8. Each threshold keeps a one-line basis and the realistic day on which it gives a wrong answer, with what the system does then.

Keep it under 380 lines. When complete, print `SRD5-<first 8 hex characters of the sha256 of design-v3.md>` and `DONE`.
