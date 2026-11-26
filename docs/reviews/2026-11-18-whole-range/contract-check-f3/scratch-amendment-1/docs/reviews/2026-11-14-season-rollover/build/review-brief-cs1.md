# Adversarial review, round 1: change set 1 of the season-boundary build (defensive reads, model-state isolation)

You are a fresh reviewer; you have seen none of the earlier rounds. **Assume the author is wrong.** This is tier A (pick-path behaviour): the code decides which streak the daily pick decision runs on.

This checkout is an isolated copy at the commit under review. The change is `git diff main...HEAD` (two commits: the author's, then one correction by the lead). Change no tracked file. Put probes, logs and your report in `.codex-review/cs1-review/` inside this copy. No network, no box, no contest account.

## The contract
`.codex-review/cs1-review/kickoff-cs1.md`: an eight-row before-and-after table. That table is what the change must do, and all it may do. The author's handover is beside it (`handover-cs1.md`) for reference only: do not rely on anything it claims; run it.

## What to establish, by running the real code
1. **Each of the eight rows.** For each: an input that the row covers, the outcome on `main` and the outcome here, from your own probe (real files on disk, corrupted the way the row says: a single `0xff` byte, `[]`, broken JSON, an unreadable file). Then look for an input the row covers that the change gets wrong, and an input NEAR the row that the change now treats differently although no row says so.
2. **What changed for anyone else.** The contract forbids changing what `load_streak`, `load_saver_available` and `load_applied_dates` return to their other callers. `contest_state` no longer calls the first two: does the decision now see a different model streak or model saver than `main` did in ANY state of `streak.json` and the saver flag file (healthy, missing, missing keys, wrong value types, a saver recorded in one place and not the other)? Every caller of `load_contest_streak_state` and `load_decision_streak_state` (find them with grep: scheduler, orchestrator, strategy, CLI, health checks, dashboard, decision records, scripts): does any now raise, or silently behave differently, on a file that `main` accepted? Which of them pass `require_contest_state=True`?
3. **The two optional fields.** Every reader of `model_streak` and `model_saver_available`, and of the new `model_state_error`: what does each do with `None` (arithmetic, comparison, formatting, JSON, the DM text, `decision.json`)? Find one that breaks or records something false.
4. **Rows 3 and 4 on realistic production files.** What does the fetch on `main` actually write to `contest_streak.json` (read the writer)? Is there a file the real fetch can write, or wrote during 2026, that this change now rejects, so that a day which picked on `main` has no pick here? What happens to the daily pick, the health check and the dashboard on such a day, and is there a way back without a code change? Dates and times: `source_date`, `as_of`, `recorded_at` each missing, empty, malformed, in the future; a file dated by one field that the code then treats as undated elsewhere.
5. **Precedence.** A valid unexpired manual override beside a corrupt automatic file; an expired or legacy manual file beside a corrupt, an invalid (row 3) or an undated (row 4) automatic file; both corrupt. State what the decision does in each and whether that differs from `main` where no row says it should.
6. **False greens.** For each new or changed test: does it fail on `main`'s code (check out `main`'s `src/bts/contest_state.py` into a scratch copy of the package, never into this checkout's tracked tree, or reason from the diff and say which)? Was any existing assertion weakened or deleted? Is any row pinned only in the direction that passes?
7. Anything else in the diff you consider wrong.

A finding about test code or evidence tooling is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this change does not touch are NOTES for a follow-up.

## Running things
- Python is `.venv/bin/python` in this copy (already synced). Use `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider … -q`.
- This machine is shared. Run any pytest invocation through the wrapper: `~/bts-runs/yield_run.sh .codex-review/cs1-review/<name>.log <command…>` (it runs the command at low priority, pauses it while the other project needs the machine, and puts the output in the log; read the log). Small probe scripts may be run directly. Do not run the whole fast suite: the gate already did (4,938 passed; the 53 known environment failures only). Run the files that touch the changed module and its callers.
- Never use a pattern kill (`pkill`, `killall`); stop only a process whose pid you started.

## Deliverable
`.codex-review/cs1-review/report.md`, under 150 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Notes`, `False greens and changed assertions`, `Rows 1 to 8` (one line each: holds or not, and the probe), `What I ran`. A BLOCK names a concrete input or state, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. Keep your probe scripts: they will be adopted as regression tests. When the file is complete, print `MARKERTAG-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
