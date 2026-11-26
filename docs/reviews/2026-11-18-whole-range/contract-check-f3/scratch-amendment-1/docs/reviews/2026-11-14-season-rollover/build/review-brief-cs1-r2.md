# Adversarial review, round 2 of 2 (final): change set 1

Same rules as round 1: assume the author (and now the lead) is wrong; change no tracked file; probes, logs and the report go in `.codex-review/cs1-review/`; pytest through `~/bts-runs/yield_run.sh`; no network, box or contest account.

This checkout is now at `acea53f8`. The change under review is still everything since `main`: `git diff main...HEAD`. What changed since your round 1: `git diff 7cff54d5 HEAD` (two commits).

## What happened after round 1
1. **Your B1 (row 4).** You offered two ways out: restore the looser check, or revise the contract before signing the stricter behaviour. The lead revised the contract; the code stands. The amended row and the reason are in `.codex-review/cs1-review/kickoff-cs1-amended.md` (the section "Amendment to row 4" at the end; the table above it is unchanged). In short: row 4 now requires a `recorded_at` that PARSES as a timestamp, because the row exists so that an undated observation cannot drive a required live decision. Attack that ruling on its merits: is there a file that a real producer on `main` (the fetch, the manual command, the dashboard, a 2026 file still on the box) writes or wrote, whose only date evidence is a `recorded_at` string that `_parse_dt` rejects? Is there a timestamp form the fetch writes that `_parse_dt` rejects on this Python? If the ruling is sound, B1 is closed by the amendment, and your two table-based regressions are superseded by `test_unparseable_recorded_at_does_not_date_an_observation`.
2. **Your N6** (a pick object whose `result` is a list or an object raised `TypeError` out of `_resolved_pick_dates`) is fixed in `e3fd5eef`, test first. Check the fix, that `_has_unconfirmed_miss` needs none, and whether any other value type still raises in either scan.
3. **A second reviewer (another model family) returned SIGN** and its 19 adversarial tests were adopted verbatim as `tests/test_contest_state_cs1_review.py` (`acea53f8`). Check those tests for false greens and for assertions that pin behaviour the contract does not want.

## The second reviewer's notes: verify or refute each by running it, and say whether it is a regression of THIS change that should block
- (a) Row 3 runs for every caller, so an out-of-range automatic file now makes `web.saver_transition_response` (documented "Never raises", called unguarded in `do_POST`) raise `ContestStateError` through `_saver_season`. On `main`, did the same call already raise for a malformed automatic file? What does the dashboard user see in each case?
- (b) With such a file the dashboard shows "Streak State Error" while displaying the model replay streak from `streak.json` as the main number.
- (c) `health.contest_state.check` now emits CRITICAL for a row-3 file even with `expected=False`, where `main` emitted nothing.
- (d) `bts fetch-contest-streak` reads the previous `contest_streak.json` before its `try` block (`cli.py` near line 2032), so a `0xff` or `[]` file cannot be overwritten by the next fetch. Pre-existing? Does it make any recovery text in the contract or its amendment false?
- (e) `streak.json` containing `{"streak": null}` with no contest observation and `require_contest_state=False` returns `streak=None` and the policy raises `TypeError`. Unchanged from `main`?

## Also establish
- Your round-1 notes N1 to N5 still hold at this commit (rerun your probe).
- Anything else in `git diff main...HEAD` you now consider wrong.

A finding about test code or evidence tooling is a NOTE, never a BLOCK, unless it makes a false claim about behaviour that nothing else covers. Findings in code this change does not touch are NOTES for a follow-up change set; say which.

## Deliverable
`.codex-review/cs1-review/report-r2.md`, under 120 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-1 findings` (closed or not, by what you ran), `Blocking findings`, `Notes` (including (a) to (e)), `False greens and changed assertions`, `What I ran`. A BLOCK names a concrete input or state, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `CS1RC2-<first 8 hex characters of the sha256 of report-r2.md>` on its own line and then `DONE`.
