# Design pass, part 4: check the build specification before the larger change sets are built

You are an adversarial reviewer of a SPECIFICATION. Assume it is wrong. No tracked file is to be changed; probes go in a scratch folder inside this copy. No network, box or contest account.

Read, in this order: `docs/reviews/2026-11-14-season-rollover/design/RULING-R24.md` (the ruling and the owner's decision the specification must implement), `design-attack-codex.md` (the earlier attack whose findings F1 to F6 it must close), then `design-v2.md` (the specification), all in this copy. Check its claims about the code against `main` as it is here.

Report on:
1. **Fidelity.** Anything in the specification that departs from the ruling or the owner's decision, adds a rule the ruling did not make, or quietly reintroduces an inference the ruling removed (a season or freshness inferred from local files, confidence passed between observations).
2. **Closure.** For each of F1 to F6, and each row of the earlier coverage and operator-mistake tables: is it closed by something an implementer can build and a test can fail, or only asserted?
3. **The decision table.** Input states that match no row, or two rows with different outcomes, under the stated precedence. Rows whose recovery command does not exist on `main` and is not specified. Outcomes that contradict the owner's decision that an in-season stale reading keeps picking.
4. **Numbers.** Every threshold (for example 14 hours, 36 hours, 5 minutes, the fetch times, 10:00 ET): what it rests on, and a realistic day on which it gives the wrong answer (a late West Coast game, a double-header, an off-day, the All-Star break, a day with no fetch between midnight and first pitch).
5. **Sequencing.** The specification lands in six change sets. For each boundary between them: is the system coherent with only the earlier ones merged (existing tests, the files the current fetch writes, a schema that is rejected before its writer exists)? Which existing tests will each change set break?
6. **What an implementer must guess.** Every place where two competent implementers would build different behaviour.
7. **Operator traps.** Steps that cannot be undone, commands that succeed while leaving the system unable to pick, and anything that depends on a person reading a DM within a window that the specification does not state.

Deliverable: `design-v2-check-codex.md` in the repository root of this copy. Verdict first: BUILD CS4 TO CS6 FROM THIS, BUILD WITH THESE EDITS (each edit tied to a finding), or NOT YET. Then findings, most severe first, each with the state or sequence, what the specification says, what should happen, and whether you demonstrated it by running something. Under 220 lines. When complete, print `SRD4-<first 8 hex characters of the sha256 of design-v2-check-codex.md>` and `DONE`.
