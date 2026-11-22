# Review: the tool that decides a deploy condition (ruling R52), follow-up after two BLOCKs

You are a fresh reviewer. **Assume the author is wrong.** Tier C, the one tool that decides a result: the owner made the paired backtest a deploy condition, and this tool prints its verdict. Find every way it can print PASS when the rule says FAIL, FAIL when it says PASS, or any verdict (or exit 0 or 1) when the rule is unmet.

**The defect class this follow-up must have closed:** "the tool gives a verdict, or the wrong exit code, on inputs it has not proven complete and unaltered." Two earlier review rounds found cases of it; their reports are `earlier-r1.md` and `earlier-r2.md` in this directory. **Every finding in both is a case you must probe again** (besides your own work): grid overrides and duplicates; exact floors (138 to 137 hits on 200 days must FAIL condition 1; exactly -1.0 pt in a season must pass condition 2); stale, truncated, partial, replaced-after-DONE files; same days cut from both arms; overlapping or stray log records; invalid rank and outcome types; undecodable or impossible log lines; the JSON target inside the evidence tree; streak metrics that are null.

## Read
- The rule as written before any result was read: `docs/reviews/2026-11-18-whole-range/FINDINGS.md`, "Ruling R52".
- The tool `scripts/audit/bullpen_r52_compare.py` and its tests `tests/scripts/test_bullpen_r52_compare.py` (uncommitted, this checkout). What it calls: `src/bts/validate/scorecard.py`.
- The runner `~/bts-runs/bullpen-bt/run_all.sh` and its log `~/bts-runs/bullpen-bt/progress.log` (no outcomes in either). The writer: `src/bts/simulate/backtest_blend.py` (`save_profiles`, the profile builder).

## Hard rules
- **Do NOT open, list or compute anything from `~/bts-runs/bullpen-bt/out/`** (the real results are not seen before the tool is accepted); the DONE markers' modification times are the one exception.
- Change no tracked file; write only in `.codex-review/r52-followup/`. No network. No `~/bts-runs/yield_run.sh`, no full suite (a long job holds the machine). Synthetic probes only, `nice -n 19 .venv/bin/python …`, a few seconds each; the one test file directly the same way.

## Known and deliberate (say whether each is acceptable for THIS queue)
- The runner is not changed (it is running the last two backtests). Completion is bound to the runner's own log and file times, not to producer digests; the result records the sha256 of the bytes scored.
- The log reader accepts a start that supersedes an unfinished start of the same run (the queue's real history has two such restarts).
- A file's time must be after its run's start and no later than its DONE marker; DONE must be no later than the logged end + 2 s (the real gaps are 0.03 to 0.97 s).

Deliverable: `.codex-review/r52-followup/report.md`, under 50 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), a table of every earlier finding (still caught?), anything new, `What I ran`, `What I did not run`. When complete, print `R52FU-<first 8 hex of the sha256 of report.md>` on its own line and then `DONE`.
