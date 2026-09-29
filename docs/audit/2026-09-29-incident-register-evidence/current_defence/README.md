# Current defence — prepared inputs (W1.5 Phase 1, plan rev 4 Task 6)

**Not yet run.** These are inputs for Task 6. They run only after Codex phase-1 r4 signs the tooling.

| Path | What |
|---|---|
| `specs/I-*.json` | The nine plan-named certificate specs (8/11 ×2, 8/13 ×2, 8/30 ×4, 9/03). Each spec is also the contract sheet: contract, certificate level with its justification, allowed mocks, production entry, boundaries, the smallest semantic mutant, the branch anchor, and killing nodes with assertion anchors. `baseline` is the placeholder `BASELINE`, set to the pinned review baseline at run time. Statically checked against `321f401`: every edit and assertion anchor resolves uniquely, and the mutation paths pass the hard rule. |
| `worklist.md` | The 30 fixed Tier-A observed incidents from the consolidated drafts. For each fixed link it lists the fix commits, the deploy bound, and the test files those commits touch. 62 links in all; for 14 of them, no test is touched by the fix. |

A certificate needs two things:
- the runner's verdict `accepted`, in the acceptance object written by `current_defence`;
- my recorded reviewer decision `accept`.

Survivors get a design §9.5 class.
