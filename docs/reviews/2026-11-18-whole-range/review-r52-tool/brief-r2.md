# Round 2 (the last): the R52 comparison tool after your BLOCK

You wrote `report.md` here (BLOCK, must-fix 1 to 6). The lead rewrote `scripts/audit/bullpen_r52_compare.py` and `tests/scripts/test_bullpen_r52_compare.py` (still uncommitted, same checkout). **Assume the lead got it wrong.** Same hard rules as round 1: do NOT open, list or compute anything from `~/bts-runs/bullpen-bt/out/` (schema of one file only, if needed); you may read `~/bts-runs/bullpen-bt/progress.log` and `run_all.sh` (they hold no outcomes); synthetic probes only, `nice -n 19`, a few seconds each; the one test file directly; no wrapper, no full suite; write only here.

What changed, and one deliberate deviation:
- Must-fix 1: the CLI has no grid options; `main` always uses the fixed 8 seeds and 2021 to 2025 (a private `_grid` keyword exists for tests only). `compare` refuses duplicate seeds or seasons.
- Must-fix 2: exact `Fraction` P@1 per cell (one validated rank-1 row per day, binary outcome), checked equal to the scorecard's float; both floors compared exactly, condition 1 by your squared inequality.
- **Must-fix 3, deviation:** the RUNNER is not changed (it is running the last two backtests; editing a running zsh script corrupts it). Instead the tool binds completion to the runner's own log: a run counts only with its DONE marker, its LAST `start` line in `progress.log` followed by `end … rc=0` (a later start supersedes), and all five season files' mtimes inside that window (+5 s). Rule on whether this is enough for the 16 real runs, given how `run_all.sh` behaves (killed runs, a restart, a removed partial directory after a reboot) — read `progress.log` for the history.
- Must-fix 4: per-cell validation (columns, dates non-null in the season's year, integer ranks 1..n with n <= 10 per day, rank-1 outcomes in {0,1}), every read error as rule unmet, and the two arms of a seed must cover the same day set.
- Must-fix 5: JSON written atomically before anything is printed, exit 2 if it cannot be; streak metrics per season (no carry-over), labelled, printed, each failure reported without touching the verdict; positive `--mc-trials`.
- Must-fix 6: 27 tests; the lead's own rerun of your mutant list plus three more: 11 of 12 killed; the survivor ("first rank-1 row per day") is rejected earlier by the rank check.

Answer: (1) is each must-fix resolved (one line each)? (2) any new way to a wrong verdict or a verdict on an incomplete queue? (3) anything wrong in the deviation for THIS queue's history?

Deliverable: `report-r2.md` here, under 40 lines: `Verdict`, the must-fix table, anything new, `What I ran`, `What I did not run`. Then print `R52REV2-<first 8 hex of the sha256 of report-r2.md>` on its own line and `DONE`.
