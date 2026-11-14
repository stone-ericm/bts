# Kickoff 5: adopt the final reviewer's probes as regression tests (tests only)

Tier D: test code only. No production file changes. The final review of the g2 fixes signed (`/Users/eric/projects/bts-g2-final/.codex-review/final/report.md`) and asked that an integration case with the real downstream path be kept, because the cold-start tests in `tests/test_refresh_memo.py` stub training and prediction. The reviewer's probes are in `/Users/eric/projects/bts-g2-final/.codex-review/final/scratch/test_adversarial.py` (read-only for you; copy what you need).

Work in this worktree on a NEW branch from `main`: `git fetch` is not needed, run `git checkout -b cold-start-tests main` (`main` already contains the merged g2 fixes).

## What to add: `tests/test_cold_start_integration.py`
Port these three probes. They must not read or write anything under `.codex-review`, must not depend on a branch name or shell out to git themselves, and must keep the network, the clock and the DM transport faked as the probes do.
1. `test_cold_pipeline_training_selection_delivery`: a season with only spring-training feeds; the real refresh, feature computation, training, selection at streak 0 and the real delivery chokepoint; assert no current-season parquet or refresh marker, a delivered scoreable `decision.json`, one simulated DM.
2. `test_next_day_prior_lineup_projection`: the day after, one prior regular-season game; the two teams that played get PROJECTED rows, the others get none.
3. From `test_opening_day_late_lineup_has_no_pick_to_retry`: write it as the DESIRED behaviour and mark it `@pytest.mark.xfail(strict=True, reason="S6: a day with no selected pick gets no last attempt (docs/reviews/2026-11-14-season-rollover/kickoff-draft.md)")`. Name it `test_opening_day_late_lineup_gets_a_last_attempt`. Desired: for a 19:10 one-game day with offset 60 and lineups first available at 18:25, the real `run_day` ends with a delivered pick. Today it saves nothing, so the test must fail for that reason and not for a setup error: also keep a plain passing assertion, in a separate non-xfail test, that today's checks run at 16:10, 17:10 and 18:10 and make no later feed request (`test_opening_day_late_lineup_checks_today`), so a setup break is visible.

Also port the receipt-precedence permutations from `test_receipt_precedence_and_schedule_network` into `tests/test_day_outcome_entry_sequence.py` if they are not already covered there; say which.

Do not port the 220-case builder differential (it needs a copy of the old builder).

## Limits
- Runtime: report the wall time of the new file. If the three tests together take more than 20 seconds on this machine (note it is throttled: check `python3 -c "import time;t=time.monotonic();time.sleep(0.01);print(time.monotonic()-t)"` and report the number), say so and do NOT mark anything `slow` on your own; report and stop.
- The tests must pass with `TZ=America/New_York .venv/bin/python -m pytest tests/test_cold_start_integration.py -q -p no:cacheprovider`, and must fail if `predict._refresh_season_data`'s `None` guard is removed (run that mutation, report it, restore).

## Deliverable
`.codex-review/g2-fix/handover-5.md`, under 40 lines: what was ported, runtime, the mutation result, the focused command and counts, `git status --short`. Do not commit. Print `G2F5-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
