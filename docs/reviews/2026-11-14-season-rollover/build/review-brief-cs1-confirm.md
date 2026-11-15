# Adversarial confirmation: one fix on change set 1 (tier A, pick-path behaviour)

You are a fresh reviewer; you have seen none of the earlier rounds. **Assume the author is wrong.** Change no tracked file. Probes, logs and your report go in `.codex-review/cs1-confirm/` inside this copy. No network, no box, no contest account.

## Scope
This checkout is at `60cece20`. Two earlier review rounds covered `main...acea53f8`. **You review only `git diff acea53f8 HEAD`** (one commit: `src/bts/contest_state.py` and a new test file), and whether it closes the finding below without changing anything else. Earlier code is in scope only where this commit changes how it behaves.

## The finding it must close
`.codex-review/cs1-confirm/report-r2.md`, section "Blocking findings", B2: `json.loads` raises a plain `ValueError` (not `JSONDecodeError`) for a JSON integer longer than the interpreter's conversion limit, and that escaped the read handlers. The contract is `.codex-review/cs1-confirm/kickoff-cs1.md` (the eight-row table and its amendments): rows 1, 2, 5, 6 and 7 say what a corrupt file must do. The commit also treats `RecursionError` (a body nested too deeply) the same way; the earlier reviewer did not ask for that.

## Establish, by running the real code
1. B2 is closed at every read site in `contest_state.py`, for `streak.json`, a date-named pick file, `contest_streak.json` and `contest_streak.manual.json`, with and without `require_contest_state`, with a manual and with an automatic observation.
2. **What the wider catch now swallows that it should not.** `ValueError` is broad. Is there any input that on `acea53f8` produced a correct result or a correct, more specific error, and now produces a different outcome? Any `ValueError` or `RecursionError` raised by OUR code inside those `try` blocks (not by the read or the decode) that is now silently turned into "skip this file" or "model state corrupt"? Read each `try` body.
3. Other exceptions a read or decode of an arbitrary file can still raise past these handlers on this Python (name the input and run it): for example `MemoryError`, `OverflowError`, a float or exponent form, a huge string, NUL bytes, a BOM, a file replaced by a FIFO or a directory, a symlink loop. Say for each whether the contract's rows cover it.
4. `RecursionError`: is catching it safe here (state after the catch, a second read in the same process), and does the nested-body test really exercise `RecursionError` on this interpreter rather than passing for another reason?
5. False greens: do all 12 new tests fail on `acea53f8`'s `contest_state.py` (use a scratch copy of the package, never this checkout's tracked tree)? Does any pass for the wrong reason?

A finding about test code is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this commit does not touch are NOTES.

## Running things
`.venv/bin/python` in this copy is synced. Use `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1`. Run every pytest invocation through `~/bts-runs/yield_run.sh .codex-review/cs1-confirm/<name>.log <command…>` (shared machine: low priority, output in the log). Small probe scripts may run directly. Do not run the whole fast suite; the gate is running separately. Never use a pattern kill.

## Deliverable
`.codex-review/cs1-confirm/report.md`, under 80 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `B2` (closed or not, by what you ran), `Blocking findings`, `Notes`, `False greens`, `What I ran`. A BLOCK names a concrete input, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `CS1RC3-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
