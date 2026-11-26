# Current defence — prepared inputs (W1.5 Phase 1, plan rev 8 Task 6)

**Not yet run.** These are inputs for Task 6. They run only after Codex signs the tooling.

| Path | What |
|---|---|
| `specs/I-*.json` | The nine plan-named certificate specs (8/11 ×2, 8/13 ×2, 8/30 ×4, 9/03). Each spec is also the contract sheet: contract, certificate level with its justification, allowed mocks, production entry, boundaries, the smallest semantic mutant, the branch anchor, and killing nodes with assertion anchors. `baseline` is the placeholder `BASELINE`, set to the pinned review baseline at run time. Statically checked against `321f401`: every edit and assertion anchor resolves uniquely, and the mutation paths pass the hard rule. |
| `worklist.md` | The 30 fixed Tier-A observed incidents from the consolidated drafts. For each fixed link it lists the fix commits, the deploy bound, and the test files those commits touch. 62 links in all; for 14 of them, no test is touched by the fix. |

A certificate needs two things:
- the runner's verdict `accepted`, in the acceptance object written by `current_defence`;
- my recorded reviewer decision `accept`.

Survivors get a design §9.5 class.

## Phase 1 boundary (plan ruling 10, rev 8)
Phase 1 certifies positive witnesses only: a wrong or extra event, or a wrong returned value. A missing-event (`absence`) spec is refused by the runner before anything runs, with the reason `certify.ABSENCE_REFUSAL`. Its link reads `unavailable (absence not certifiable by the Phase 1 recorder)` and keeps its regression evidence (design §9.3 v3.3).

| Spec | Link | Kind | Phase 1 |
|---|---|---|---|
| I-0811-a | I-082 link 1 | return | eligible |
| I-0811-b | I-082 link 2 | event | eligible |
| I-0813-a | I-083 link 1 | return | eligible |
| **I-0813-b** | **I-083 link 2** (missing E3 alert) | **absence** | **unavailable**: refused; the run is executed only to record the refusal |
| I-0830-a | I-084 link 2 | event | eligible |
| **I-0830-b** | **I-084 link 1** (missing timely pick) | **absence** | **unavailable**: refused; the run is executed only to record the refusal |
| I-0830-c | I-084 link 3 | event | eligible |
| I-0830-d | I-084 link 4 | return | eligible |
| I-0903-a | I-085 link 1 | return | eligible |

Before an eligible spec runs, I review its admitted closure against the bounded model (`certify.COVERAGE`). That closure is its test files, their conftest and helpers, and the closure screen's hits (`../tooling/closure_screen.txt`). The symptom kind of every other worklist link is decided before it is specified. An absence-kind link reads unavailable without a run.

Deferred, not dropped (ruling 10): a missing-event link may later get an independent recorder at a mediated boundary (a collector process that owns the receipt channel, with a baseline receipt control). It may instead get a positive witness of the wrong decision, which certifies only that decision. For example, I-0830-b's `plan_fallback_action` returning `FallbackPlan("defer", …)` would certify the planner regression, never "no pick DM reached the operator".
