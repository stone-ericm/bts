# Kickoff A3: one blocking finding from the first review round (S1)

The fresh Codex reviewer blocked the merged change on one point in your set. Everything else in S1 to S5 passed its probes and stays as it is.

## The finding
`load_decision_streak_state` now calls `_resolved_pick_dates(picks_dir)` on every decision. That helper does `body.get("result")` on each decoded root-level pick file. A historical file whose JSON is valid but not an object (`[]`, `null`, a number, a string) raises `AttributeError`; an object whose `result` is unhashable (`[]`, `{}`) raises `TypeError` at the set membership. Neither is a `ContestStateError`, so nothing catches it, and every decision fails until someone repairs a file that has nothing to do with today. At the base the decision loader did not need this scan: with no contest observation it returned streak 0, model-only; with a healthy first-fetch observation it returned streak 0, stale.

## Required
| Decoded root-level pick file | Today | Required |
|---|---|---|
| valid JSON that is not an object | `AttributeError` out of `_resolved_pick_dates`, and so out of every decision | skipped, exactly as a file with invalid JSON is skipped |
| an object whose `result` is not a string | `TypeError` (unhashable) or a silent non-match | skipped |
| an object with a string `result` | as today | unchanged |

`_has_unconfirmed_miss` in the same file has the same loop shape (`body.get("result") == "miss"`): give it the same object check so a non-object file is skipped there too. Do not change what either function returns for well-formed files.

## Tests (RED first)
- Loader: for each bad shape (`[]`, `null`, `7`, `"x"`, `{"result": []}`, `{"result": {}}`) in an old pick file dated last season, `load_decision_streak_state` returns what it returned at the base, with no contest observation and with a healthy first-fetch observation (`{"active_streak": 0, "recorded_at": "<this year>"}`), for `require_contest_state` true and false.
- The same shapes in a pick file dated THIS year: skipped, and the model streak stays 0 if that was the only current-year file.
- One CLI regression through `click.testing.CliRunner` if you can do it without training a model (for example a command that only loads the decision state); if every CLI path trains, say so and leave it to the loader tests.
- Mutation: remove the object check; report which tests fail.

## Deliverable
`.codex-review/sr/handover-a3.md`, under 30 lines; same focused command and counts; `git diff --stat` against your previous commit `66f3992`. Do not commit. Print `SRA3-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
