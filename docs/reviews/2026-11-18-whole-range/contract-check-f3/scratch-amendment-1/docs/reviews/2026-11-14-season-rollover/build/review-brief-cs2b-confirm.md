# Adversarial confirmation: the last fix on change set 2, second attempt (tier A, pick-path behaviour)

You are a fresh reviewer; you have seen none of the earlier rounds. **Assume the author is wrong.** Change no tracked file. Probes, logs and your report go in `.codex-review/cs2b-confirm/` inside this copy. No network, no box, no contest account.

## Scope
This checkout is at `0a34b7cf`. Two review rounds covered `d8eb50d0...c9c1e342`. **You review only `git diff c9c1e342 HEAD`** (one commit: six lines of `src/bts/picks.py` and two test files), and whether it closes the finding below without changing anything else. Earlier code is in scope only where this commit changes how it behaves.

## The finding it must close
`.codex-review/cs2b-confirm/report-r2.md`, "Blocking findings", B2. The contract (`.codex-review/cs2b-confirm/kickoff-cs2b.md`, row L and its amendment at the end) says: only an exact built-in `str` is parsed as a date; anything else has no season; and a dated `update_streak` refuses a date with no season with `ValueError` before any I/O. At `c9c1e342` the `ValueError`'s message was built with `type(date).__name__`, which a metaclass can turn into caller code (raise, exit, write, stall), so the promised `ValueError` did not hold for such a value. The fix builds the message from `repr(date)` only when `type(date) is str`, and from fixed text otherwise.

## Establish, by running the real code
1. B2 is closed: the reviewer's attacks in `.codex-review/cs2b-confirm/r2_attack.py` (the metaclass `__getattribute__` trap, the `__name__` that is a `str` subclass with a hostile `__format__`, the one that writes a file, the one that sleeps), each through the real `update_streak` on a real directory: `ValueError`, no caller code run, no state read, bytes unchanged. Write your own attacks too.
2. **Between entering `update_streak(results, picks_dir, date=D)` with `D` not `None` and the `ValueError` being raised, is there ANY operation on `D` other than `type(D) is str` and, for an exact `str`, the two parsers and `repr`?** Read the code path line by line (`update_streak`, `season_of_date`) and list every expression that touches `D`. For an exact built-in `str`, can `repr`, `date.fromisoformat` or `datetime.strptime` run caller code or raise something that escapes (name the input and run it; rule interpreter-wide conditions such as an asynchronous signal or memory exhaustion in or out with a reason)?
3. The same question for the five read sites that call `season_of_date` (the four in `scheduler.py`, the one in `cli.py`, the one in `strategy.py`): does this commit change anything there? (It should not.)
4. `results` and `picks_dir` are not dates and are out of this commit's scope; say so if you find something there, as a NOTE.
5. The two test changes: does the new `type-name-trap` case fail on `c9c1e342`'s `picks.py` (a scratch copy of the package, never this checkout's tracked tree) for the reason stated; is the adopted racing-scorers test in `tests/test_streak_season_cs2b_review.py` sound now that its peer runs once (does it still interleave a second scorer, and are its assertions the same); is the header's corrected claim true?

A finding about test code is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this commit does not touch are NOTES. A finding about a value that cannot be produced by decoding command-line text, a file or JSON, and that no caller in `src` or `scripts` constructs, is a NOTE unless it shows the amended row L is still false as written at this commit.

## Running things
`.venv/bin/python` in this copy is synced. Use `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1`. Run every pytest invocation through `~/bts-runs/yield_run.sh .codex-review/cs2b-confirm/<name>.log <command…>`, one script and one invocation per family of checks, and do not edit a script after asking to run it. Small probe scripts may run directly. Do not run the whole fast suite; the gate is running separately. Never use a pattern kill.

## Deliverable
`.codex-review/cs2b-confirm/report.md`, under 70 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `B2` (closed or not, by what you ran), `Every expression that touches the date`, `Blocking findings`, `Notes`, `What I ran`. A BLOCK names a concrete input, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `CS2BRC3-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
