# Kickoff: change set 1 of the season-boundary build (defensive reads and model-state isolation)

Tier A (pick-path behaviour): tests first and RED, then the change, then the gate. A fresh adversarial review follows your handover; write for a reader who assumes you are wrong.

You work in `~/projects/bts-cs1`, branch `sb-cs1`, cut from `main`. Commit there. Do not push. No network, no box, no contest account. Use `UV_CACHE_DIR=/tmp/uv-cache uv run …` (the environment is already synced).

## Read first
1. `docs/reviews/2026-11-14-season-rollover/design/design-v2.md`, section 6 item 1 (CS1). The rest of that document describes later change sets: do not build them.
2. `docs/reviews/2026-11-14-season-rollover/design/design-attack-codex.md`: finding F1, the U1 row of the coverage table, and in F4 the paragraph on the undated "bare fixture". Its probe script is `docs/reviews/2026-11-14-season-rollover/design/probes-codex/probe.py` (cases `manual_not_universal_recovery`, `u1_bad_best_reachable_after_corruption`, `legacy_fixture_exception_is_live`): turn these into regression tests.
3. `src/bts/contest_state.py` in full, and `load_streak`, `load_saver_available`, `load_applied_dates` in `src/bts/picks.py`.

## The contract
The specification's CS1 line is short. Where it could be read two ways, this table decides; the choices marked (sequencing) are the lead's, made so that this change set can land before the later ones change the file schemas. If you cannot make the code match a row, or a row looks wrong against the real code, STOP and say so in the handover. Do not choose.

| # | Situation | Today on `main` | Required after CS1 |
|---|---|---|---|
| 1 | A date-named pick file (`YYYY-MM-DD.json`) scanned by `_resolved_pick_dates` or `_has_unconfirmed_miss` is unreadable (`OSError`), not UTF-8 (`UnicodeDecodeError`), not JSON, or JSON that is not an object | `json.JSONDecodeError` and `OSError` are caught in one scan; `UnicodeDecodeError` and a non-object body raise out of the loader | The file is skipped in both scans; no exception leaves the loader; the other files are still read |
| 2 | A contest state file (`contest_streak.json` or `contest_streak.manual.json`) is not UTF-8 | `UnicodeDecodeError` raises out of `_parse_state_file` | `ContestStateError`, as for malformed JSON today. The existing precedence stays: an unexpired manual override is returned before the automatic file is parsed |
| 3 | The AUTOMATIC file (`contest_streak.json`) parses, but its stored `best_streak` is an int below its streak, or its streak or best is above 57 | Accepted; only the best is marked untrusted later by `classify_best_streak` | `ContestStateError` naming the file and the values. (sequencing) The MANUAL file keeps today's behaviour in this change set |
| 4 | The automatic file has neither a parseable `source_date` (or `as_of`) nor a `recorded_at` string, and the caller passed `require_contest_state=True` | Accepted with status `stale` | `ContestStateError` saying the observation is undated. (sequencing) With `require_contest_state=False` behaviour is unchanged. A dated version-1 file stays accepted: rejecting version 1 belongs to a later change set |
| 5 | `streak.json` is unreadable, not UTF-8, not JSON, or not an object, AND a usable contest observation exists (manual or automatic) | `load_streak` raises before the contest observation is even looked at | The contest observation is returned exactly as it would be with a healthy `streak.json`, with `model_streak=None`, `model_saver_available=None`, and a new field on `DecisionStreakState` carrying a short description of the model-state error (`None` when healthy). One warning is logged. Nothing is written to disk |
| 6 | `streak.json` is unreadable as in row 5 and NO contest observation exists, `require_contest_state=False` (the model fallback) | Raises whatever the read raised | `ContestStateError` saying the model state is corrupt and naming the file; no invented streak |
| 7 | `streak.json` is unreadable as in row 5, no contest observation, `require_contest_state=True` | Raises the read error | `ContestStateError` for the missing contest state, as today for a healthy `streak.json`; the message may add that the model state is also unreadable |
| 8 | `streak.json` is a JSON object without the expected keys (for example `{}`) | whatever `load_streak` returns today | Unchanged. Pin today's result in a test |

Not in this change set: any season logic, the saver mapping, new schemas or files, alert routing, the fetch, the CLI. Do not change what `load_streak`, `load_saver_available` or `load_applied_dates` return to their other callers (the scorers); isolate the failure where `contest_state` calls them.

`model_streak` and `model_saver_available` become optional. Find every reader of those two fields (source, dashboard, decision records, tests) and make sure none breaks on `None`; list each one in the handover with what you did.

## Tests
- RED first: write each test, run it, and record the failure, before the change.
- One test per table row, in the direction of the failure: for rows 1, 2 and 5 corrupt the real file on disk with invalid UTF-8 (a single `0xff` byte) and with a non-object (`[]`), not only with broken JSON. For row 3 write a valid file and then change the stored value on disk. For row 5 use a valid manual override in one test and a valid automatic file in another.
- The three probe cases named above, as regression tests.
- Existing tests that relied on the old behaviour of rows 3 and 4 (for example an undated automatic fixture used with `require_contest_state=True`): give the fixture real metadata where the test is about something else; where the test asserted the old behaviour itself, change it and say so. List every changed test with the reason.
- Before you start and after you finish, run every test file that mentions `contest_state`, `load_streak` or `load_saver_available` (find them with grep, do not rely on a list), then the fast suite: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q`. Report the counts before and after.

## Handover
Write `HANDOVER-CS1.md` in the worktree root (do not commit it): the commits; for each table row the file and lines that implement it and the test that pins it; every existing test you changed and why; every reader of the two optional fields; what you ran with the counts; anything you could not match or think is wrong. Then print `CS1A-<first 8 hex characters of the sha256 of HANDOVER-CS1.md>` and `DONE`.

## Amendment to row 4 (lead session on the studio, 2026-11-15, after review round 1)
Row 4 as written says "nor a `recorded_at` string". The branch's second commit (`7cff54d5`, the previous lead's correction) requires a `recorded_at` that PARSES as a timestamp: an empty or malformed string does not date an observation. The Codex reviewer (marker `CS1RC-58327cbb`, finding B1) correctly reported that the code rejects more than the table authorizes, and offered two ways out: restore the looser check, or revise the contract.

**The contract is revised; the code stands.** Row 4 now reads: the automatic file has neither a parseable `source_date` (or `as_of`) nor a `recorded_at` that parses as a timestamp. Reason: the row exists so that an undated observation cannot drive a required live decision (the attack's "bare fixture"), and `""` or `"not-a-time"` is not a date. The reviewer's own check of the real writer (its note N1) found that the fetch has written a valid UTC `recorded_at` since June 2026, so no file the fetch produces is affected; only a hand-edited or damaged file is.

Cost if wrong: with `require_contest_state` on, a file whose only date field is a damaged `recorded_at` refuses the pick with an alert where `main` picked on it as stale; the way back is the next fetch or a manual observation, with no code change.

Also taken into this change set from the same review (its note N6): a date-named pick object whose `result` is not a string (a list or an object) raised `TypeError` out of `_resolved_pick_dates` on `main` and on the branch. Row 1 says no exception leaves the loader; such a file is now skipped like the other malformed pick files.

## After review round 2 (2026-11-15): one confirmation round, by the owner's decision
Round 2 (Codex, marker `CS1RC2-5e45e3ae`) closed the round-1 findings and blocked on one gap (B2): an over-long JSON integer raises a plain `ValueError` past the new handlers. The work policies cap a change at two review rounds. Put to Eric with three options (apply the fix and allow one narrow confirmation; restart with a fresh author; merge on the gate alone), he chose the first ("1"). The fix is on `sb-cs1` after `acea53f8`, test first, with the reviewer's probes adopted; a fresh Codex reviewer checks only that delta. If the confirmation blocks, the change set stops and goes back to him.
