# Adversarial review, round 2 of 2 (final): change set 2, second attempt

Same rules as round 1: assume the author (and now the lead) is wrong; change no tracked file; probes, logs and the report go in `.codex-review/cs2b-review/`; pytest through `~/bts-runs/yield_run.sh`, one script and one invocation per family of checks, and do not edit a script after asking to run it; no network, box or contest account.

This checkout is now at `c9c1e342`. The change under review is still everything since its base: `git diff d8eb50d0 HEAD`. What changed since your round 1: `git diff dd915ac9 HEAD` (two commits by the lead).

## What happened after round 1
1. **Your B1 and R1** (a value whose `__class__` raises; a `str` subclass whose `__len__` raises `SystemExit` or blocks). Accepted. Your R1 asked for a rule decision, and it is made: `.codex-review/cs2b-review/kickoff-cs2b-amended.md`, the section "Amendment to row L after review round 1" at the end. **Only an exact built-in string is parsed: `type(date) is str`.** Anything else, a `str` subclass included, has no season: L1 to L3 read as at the base, and a dated `update_streak` refuses it with `ValueError` before any I/O. Nothing on the caller's value is consulted, so nothing of the caller's can run inside the helper. Commit `57d17de8`. **Attack it:**
   - Is there any value for which `season_of_date` can still raise, exit, stall, or return something that is not an exact `int` or `None`? Any value for which `type(x) is str` is true and the two parsers can run caller code or fail with something `except Exception` does not catch (`MemoryError`, `RecursionError`, `KeyboardInterrupt` from a signal are yours to rule in or out with a reason)?
   - Rerun your two counterexamples through every path you ran in round 1 (the three polling read branches, the shadow selection, `select_pick`, the dated update): is each outcome now the base's, or the allowed `ValueError` before I/O?
   - The dated update's `ValueError` message: can building it run caller code?
   - Does the amendment change any cell of your 344-cell matrix other than the hook rows and the plain `str` subclass rows? Rerun the matrix and say.
2. **A second reviewer (another model family) returned SIGN** with notes; verify, do not rely on them:
   - Two tests in `tests/test_cli_integration.py` tagged their model-streak fixture 2026 while the command reads the wall clock, so in a later year the model streak read 0 and no longer competed with the contest streak. They now tag the fixture with the current ET year (`_current_et_year()`). Is that sound, and does it weaken either test?
   - Its 134 adversarial tests are adopted as `tests/test_streak_season_cs2b_review.py`; three expectations for `str` subclasses were changed to the amended row L (marked in the file). Check the file for false greens and for assertions that pin behaviour the amended contract does not want.
   - Its rule observation, accepted as it stands: `strptime` matches some non-ASCII digits in the year and day, so a few such strings have a season; every invariant still holds. Say if you think that is wrong to leave.
3. Not changed, recorded as follow-ups: the same day scored under two spellings of its date is applied once per spelling, as at the base (your round-1 result `6,7,7` at both versions); the export and regenerate tools drop the season tag; the reconcile summary samples its clock again at the ET New Year.

## Also establish
- Your round-1 results still hold at this commit (rerun your probes, exit paths, boundary race and the clock runs with your `tick2` harness).
- Anything else in `git diff d8eb50d0 HEAD` you now consider wrong.

A finding about test code or evidence tooling is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this change does not touch are NOTES. A finding about an input that cannot be produced by decoding command-line text, a file or JSON, and that no caller in `src` or `scripts` constructs, is a NOTE unless it shows the amended row L is false as written.

## Deliverable
`.codex-review/cs2b-review/report-r2.md`, under 100 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-1 findings` (closed or not, by what you ran), `Blocking findings`, `Notes`, `False greens and changed assertions`, `What I ran`. A BLOCK names a concrete input or state, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `CS2BRC2-<first 8 hex characters of the sha256 of report-r2.md>` on its own line and then `DONE`.
