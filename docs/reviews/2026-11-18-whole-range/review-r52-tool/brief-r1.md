# Review: the tool that decides a deploy condition (ruling R52), before it is run on the real outputs

You are a fresh reviewer. **Assume the author is wrong.** Tier C (a tool that decides a result): find every way this tool can print PASS when the rule says FAIL, FAIL when it says PASS, or a verdict when the rule is unmet.

## Read
- The rule, exactly as written before any result was read: `docs/reviews/2026-11-18-whole-range/FINDINGS.md`, the paragraph "Ruling R52". Owner's decision: the result is a deploy condition.
- The tool: `scripts/audit/bullpen_r52_compare.py` (uncommitted, in this checkout). Its tests: `tests/scripts/test_bullpen_r52_compare.py`.
- What it calls: `src/bts/validate/scorecard.py` (`compute_precision_at_k`, `_precision_at_k_flat`, `compute_full_scorecard`).
- The run setup it reads: `~/bts-runs/bullpen-bt/run_all.sh` (arms, seeds, output paths, DONE markers). The backtest writer: `src/bts/simulate/backtest*.py` or wherever `bts simulate backtest` writes `backtest_<season>.parquet` (find it), to learn the real file shape (columns, ranks per day, double-down rows, ties, NaN).

## Hard rules
- **Do NOT open, list the contents of, or compute anything from `~/bts-runs/bullpen-bt/out/`** (the real results are not to be seen before the tool is accepted). You may read the column NAMES of one file with `pyarrow.parquet.read_schema` and nothing else.
- Change no tracked file. Write only in `.codex-review/r52-compare/`.
- No network. No pytest through `~/bts-runs/yield_run.sh` and no full suite (a long job holds the machine). Small probes on SYNTHETIC data only, each a few seconds, run with `nice -n 19 .venv/bin/python …`; you may run the one test file directly the same way.

## Establish
1. Does the tool compute exactly R52's statistic (per-seed season-averaged paired difference; one-sided 95% t lower bound with n-1 degrees of freedom; strictly above -0.005; no season mean below -0.010, equal passing)? Quote any mismatch.
2. Is P@1 what R52 names (`compute_precision_at_k(..., k=1)`) on the real file shape? What happens with several rank-1 rows on one day, missing ranks, NaN `actual_hit`, a season file with dates from two years, an arm whose day set differs from the other's, a seed listed twice, a truncated or partial file next to a DONE marker?
3. Can a partial queue (a run not finished, a DONE missing, a file from an earlier killed run) ever produce a verdict instead of exit 2?
4. False greens in the tests: which test would still pass if the rule were implemented wrongly? Write and run at least five mutants of the tool on synthetic data and report which the tests catch.
5. The reported streak metrics (`--streak`): anything that could make them misleading next to the verdict (they are reported, not ruled)?

Deliverable: `.codex-review/r52-compare/report.md`, under 50 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Must fix` (each with the exact code change), `Notes`, `What I ran`, `What I did not run`. When complete, print `R52REV-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
