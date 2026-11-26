# Adversarial confirmation: the last fixes on change set 2 (tier A, pick-path behaviour)

You are a fresh reviewer; you have seen none of the earlier rounds. **Assume the author is wrong.** Change no tracked file. Probes, logs and your report go in `.codex-review/cs2-confirm/` inside this copy. No network, no box, no contest account.

## Scope
This checkout is at `ee9fb870`. Two earlier review rounds covered `d8eb50d0...a26958dc`. **You review only `git diff a26958dc HEAD`** (one commit: `src/bts/scheduler.py` and two test files), and whether it closes the finding below without changing anything else. Earlier code is in scope only where this commit changes how it behaves.

## The finding it must close
`.codex-review/cs2-confirm/report-r2.md`, "Blocking findings", B1: the change set added four season reads in `scheduler.py` that parsed the date with `date.fromisoformat`, which rejects an unpadded date such as `2027-3-24`. Before the change set (`d8eb50d0`) the shadow one-shot command and three display branches of `run_result_polling` accepted that form. The contract is `.codex-review/cs2-confirm/kickoff-cs2.md` (the twelve-row table and its amendment): rows 6 and 8 say those reads use the season of the date; nothing in it narrows which date strings the scheduler accepts.

## Establish, by running the real code
1. B1 is closed at all four sites, through the real entry points where you can (the reviewer's probes `shadow_date_cli.py` and `date_read_edges.py` are in `.codex-review/cs2-confirm/` for reference; write your own): the shadow command with a padded and an unpadded date, and `run_result_polling` on its all-void, final-peer and early-peer branches. For each: the outcome at `d8eb50d0`, at `a26958dc` and here.
2. **What `strptime("%Y-%m-%d")` accepts or rejects differently from what each of those four call paths accepted at `d8eb50d0`.** For every date string a path accepted at the base (unpadded month or day, surrounding whitespace, a time part, other separators, a `datetime.date` object, `None`), is the outcome here the base's? Is there a string the base REJECTED early that now gets further and does something (writes a file, scores a pick) before failing?
3. The other season reads the change set added outside `scheduler.py` (`cli.py` near the void read, `strategy.select_pick`, `picks.update_streak`): the earlier round judged them correct as they are (`update_streak` is strict by contract; the CLI rejects the same strings upstream at the base). Confirm or refute for the unpadded form.
4. The test changes in this commit: the two parametrised scheduler tests (do the unpadded cases fail on `a26958dc`'s `scheduler.py`, checked in a scratch copy of the package, never this checkout's tracked tree?), the clock pin in `test_row6_cli_void_display_reads_scored_season` (does it weaken the test? does the test now pass under a clock in 2028?), and the tightened `match=` in the season-type test.
5. Anything else in this one commit you consider wrong.

A finding about test code is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this commit does not touch are NOTES.

## Running things
`.venv/bin/python` in this copy is synced. Use `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1`. Run every pytest invocation through `~/bts-runs/yield_run.sh .codex-review/cs2-confirm/<name>.log <command…>` (shared machine: low priority, output in the log). Small probe scripts may run directly. Do not run the whole fast suite; the gate is running separately. Never use a pattern kill.

## Deliverable
`.codex-review/cs2-confirm/report.md`, under 80 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `B1` (closed or not, by what you ran), `Blocking findings`, `Notes`, `False greens`, `What I ran`. A BLOCK names a concrete input, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `CS2RC3-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
