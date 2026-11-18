# Kickoff: `opp_bullpen_hr_30g` must not depend on the same day (draft 1)

Tier A (a production model feature: `FEATURE_COLS`; it changes training inputs and so the model). Tests first, then the change, then the gate, the leakage audit and the nuclear test. A fresh adversarial review follows your handover; write for a reader who assumes you are wrong.

You work in `~/projects/bts-work/bullpen-fix`, branch `fix/bullpen-same-day`, cut from `main` at the commit named when you are started. One commit there. Do not push. No network, no box, no contest account. Python is `.venv/bin/python` in that worktree (already synced, with the model extra). Run EVERY pytest invocation and every long script as `~/bts-runs/yield_run.sh <a log file inside the worktree's .codex-review/bullpen/> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python …`, ONE wrapper run at a time (a `nice` warning is not an exit: wait for it to exit); create files with your patch tool, never a shell heredoc; never use a pattern kill. One script per family; do not edit a script after asking to run it. The sandbox cannot bind sockets: ask once for an unsandboxed wrapper run for the final fast suite. The baseline fast suite for the base is known (unsandboxed: exactly the IDs in `~/bts-runs/gate-failed.txt`): do not run a baseline; say so in the handover. Real data is on this machine: `data/processed/pa_{2017..2026}.parquet` and `data/raw/{2025,2026}` in the MAIN checkout `/Users/eric/projects/bts` (read-only for you: read it by absolute path, never write there). **Do not read 2026 outcomes for any comparison or statistic: use 2025 and earlier** (the 2026 exposure register, `CLAUDE.md` "Season 2026 wrap").

## Why
`docs/audit/2026-11-18-proptest-trial/README.md`: a property-testing trial found that the value of `opp_bullpen_hr_30g` for a plate appearance on date D exists only if the opposing (pitching) team used a reliever ON date D. In `src/bts/features/compute.py` (the "Team bullpen composite" block) the per-team daily series is built from reliever plate appearances only, the rolling mean is shifted by one, and the result is merged back on the exact `(team, date)`; a date on which the team used no reliever has no row, so every plate appearance against that team that day gets `NaN`. On real 2025 data that is 30 team-dates and 941 plate appearances (0.52% of rows), with a hit rate of 0.142 per PA against 0.221 elsewhere: they are complete games. The function's contract (its docstring; `CLAUDE.md` "Safety Rules") is that a feature for date D uses only rows dated strictly before D. A missing value that marks a low-hit game breaks it. At serving (`model/predict.py`, the `bullpen` lookup) the value is the team's latest non-missing one, so the missing-value signal never appears live: training and backtests see something serving never does.

## The rule
For every plate appearance whose pitching team is known, on date D:
`opp_bullpen_hr_30g` = the mean of the team's daily reliever hit rate (`bp_hits / bp_pas`, per date, exactly as computed today) over the team's most recent 30 dates WITH a reliever plate appearance that are strictly before D, when at least 10 such dates exist; otherwise `NaN`.
Nothing about date D or any later date enters it: not the outcomes, not which pitchers appeared, not whether a reliever appeared.

## Rows
| # | Where | Today on `main` | Required |
|---|---|---|---|
| 1 | The bullpen block in `compute_all_features` | Value merged on the exact `(team, date)` of the reliever-date series; missing on a date with no reliever plate appearance | The rule above for every row. On every row where today's code gives a value, the new value is IDENTICAL to the bit (same dates in the window, same arithmetic order). The only rows that change are rows that are `NaN` today and get a value (never a value becoming `NaN`, never a value changing) |
| 2 | Rows whose pitching team is unknown, or when the probable-pitcher lookup is empty | `NaN` | Unchanged |
| 3 | Every other column `compute_all_features` returns | | Byte-identical on real data (2024 and 2025 computed together, and each alone): compare every column except `opp_bullpen_hr_30g` with `pandas.testing.assert_frame_equal(check_exact=True)`, including row order and dtypes |
| 4 | `model/predict.py`, the `bullpen` serving lookup | The latest non-missing value per pitching team | The code is NOT changed. State in the handover what its OUTPUT now is in the one case where it differs: a team whose most recent date in the frame had no reliever plate appearance (today the lookup returns the value from that team's last reliever date; with row 1 it returns the value as of the most recent date, which includes that last reliever date's results). Show the size of the difference on real 2025 data for each of the 30 team-dates' following game |
| 5 | Everything else | | Unchanged: `FEATURE_COLS`, the other features, the model code, the policies. No new dependency (Hypothesis is not to be added) |

## Tests required (RED on the base unless named a GREEN control)
- The regression from the trial, as a plain pytest test in `tests/` (`docs/audit/2026-11-18-proptest-trial/regression_bullpen_same_day_availability.py` is the shrunk example; no Hypothesis import).
- An independent from-scratch reference: for a small hand-built frame and for a sampled real 2025 slice, compute the rule with plain loops over the team's prior reliever dates and compare with the function's output on EVERY row, including: fewer than 10 prior reliever dates; exactly 10; more than 30 (the window expires); a date with no reliever plate appearance (the value must equal what a reliever date in that position would have had); two games on one date (a doubleheader: both games' rows get the same value, and neither game's results enter it); a team's first date; a season boundary (the window crosses it, as today); a game whose probable pitcher is unknown.
- Same-day invariance on a real 2025 slice: change, on one date D, which pitcher every plate appearance is charged to (all starter; all reliever) and every `is_hit` on D: no `opp_bullpen_hr_30g` value on D changes (values AFTER D may, by design: say so).
- Row 1's identity and row 3's byte-identity on real 2024 and 2025 data, as a script under `.codex-review/bullpen/` run through the wrapper, with its summary in the handover: rows changed, rows `NaN` before and after, the maximum absolute difference on rows that had a value (must be exactly 0).
- GREEN controls: the existing feature tests.

## After the change
1. `scripts/leakage_audit.py` and the nuclear test that `CLAUDE.md` "Safety Rules" requires after any feature change (find how each is run from the repository; run them through the wrapper; if either needs hours, say so and STOP before running it, do not skip it silently). Report their results.
2. The fast suite alone on the committed tree, failing set exactly `~/bts-runs/gate-failed.txt`.
3. Do NOT run backtests, retrain artifacts, or touch `data/models/` or `data/validation/`: measuring the effect on backtest metrics is a separate step the lead will decide on.

## Known limits (state them in the handover; they are not findings)
- The model's training inputs change on about half a percent of rows, so a retrained model differs slightly; no shipped artifact is rebuilt by this change, and backtest baselines computed before it are not comparable to the last decimal.
- Serving output changes only in the case row 4 names.
- Historical backtests and audits already written used the old values.

## Handover
Write `HANDOVER-BULLPEN.md` in the worktree root (do not commit it): the commit; the files and lines; for each test whether it was RED on the base; the identity and byte-identity summary; the row-4 table; the leakage audit and nuclear test results (or why you stopped); what you ran with the counts; the known limits; anything you could not match or think is wrong. If the contract cannot be met as written (in particular if bit-identity on valued rows cannot be kept), STOP and say so instead of choosing. Then print `BULLPEN-<first 8 hex characters of the sha256 of HANDOVER-BULLPEN.md>` on its own line and `DONE`.
