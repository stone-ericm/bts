# Kickoff: change set F4, the morning report says when `streak.json` cannot be read (2026-11-25, version 1; base `main` at `5e0167b5` for the contract check; the author builds on `main` after F3 merges, since both change `src/bts/day_outcome.py`)

Finding BL-O3 of the closing test, VISIBILITY ONLY (`CLOSING-TEST.md`; `closing-test/codex-blind/report.md`, "Local grading corruption can persist behind a clean entry report", fixture `closing-test/codex-blind/evidence/interactions-refusal-model_list.json`; ruling R60). Tier A (it changes what the operator is told; read-only). Session `bts-wr-fresh13` (drafted by `bts-wr-fresh12`). Branch `fix/wr-f4`.

## What is wrong on the base
A `streak.json` that holds a JSON list (or cannot be decoded) makes every scorer stop with the file intact, by design (change set 2: `picks._load_streak_season` raises): the 01:00 `check-results`, the 02:00 and 07:40 `bts reconcile`, and the scheduler's end-of-day scoring all fail every night. The end-of-day health check `streak_validation` would say so (on a list it raises inside `_safe_run`, which reports a CRITICAL `health check 'streak_validation' crashed`), but the closing test showed the health run does not get there when scoring fails first, and the scheduler's restarts fail too. The 08:30 `bts day-outcome` reads only the decision record and the entry receipts, so it reports a clean day. Nothing the operator reads says the model state is broken.

## The design (read-only; `src/bts/day_outcome.py` only)
1. **One more read in `gather`:** `picks_dir / "streak.json"`, through the project's non-blocking regular-file reader (`bts.util.read_regular_file_text`), then `json.loads`. The file is a **problem** when it exists and: it is not a regular file, cannot be opened or read, is not valid UTF-8 JSON, is not a JSON object, or has a `streak` member that is not an exact non-negative `int` (a `bool` counts as not an int). A missing file is NOT a problem (every scorer starts from 0 on a missing file). The contract check confirms that each listed problem is exactly a state in which the scorers or reconcile fail, and lists any it does not cover.
2. **The outcome carries it without hiding the day's state:** `DayOutcome` gains `streak_problem: str | None` (default `None`); `ok` is `state in OK_STATES and streak_problem is None`; `as_dict()` gains the key `streak_problem` (its value or `null`); `classify_day` is unchanged (the day's state is computed exactly as on the base). `gather` sets `streak_problem` to one of the short reasons `not a regular file`, `cannot be read (<ExceptionType>)`, `not valid JSON`, `not a JSON object`, `streak is not a non-negative integer`.
3. **The text (R64).** `message()` appends, when `streak_problem` is set: ` Also: streak.json <reason>. Nightly grading and reconcile stop until it is repaired, so results are not being scored. Do not edit, move or delete it: preserve it and hand it to the supervised BTS lead session (bts-lead) or Eric.` On a day whose state is clean, the head of the message becomes `BTS day check D: STREAK FILE UNREADABLE` in place of the state's name, followed by the base's detail for the clean state, so the DM is not titled as a clean day.
4. The exit codes are the base's: 1 when not `ok` (so a clean day with a broken `streak.json` exits 1 and DMs), 2 when the check itself cannot run (unchanged; a broken `streak.json` never makes it 2).

## Must still catch (identical to the base)
| # | State | Must still happen |
|---|---|---|
| K1 | `streak.json` absent, or a valid object with an int `streak` (any season tag, any other members, `applied_dates` of any shape) | every output byte for byte the base's except the added `"streak_problem": null` key in the stdout JSON |
| K2 | every day state (all of `OK_STATES` and every problem state) with a good `streak.json` | `state`, `detail`, `pick`, `ok`, exit code and DM text exactly the base's |
| K3 | a problem day state (`not_entered`, `no_decision`, …) with a broken `streak.json` | the base's state and head; the "Also" sentence appended; exit 1, one DM |
| K4 | the command's "could not run" path (exit 2) | unchanged; a broken `streak.json` alone never produces it |
| K5 | a named pipe or a directory at the path | reported as `not a regular file`, never blocks |

## Must change (RED on the base)
| # | State | Base | F4 |
|---|---|---|---|
| M1 | the closing-test fixture: `streak.json` holds `[]`, the day was `entered` | `entered`, exit 0, no DM | `streak_problem` `not a JSON object`, exit 1, the DM headed `STREAK FILE UNREADABLE` |
| M2 | undecodable bytes; invalid UTF-8; a JSON string; `{"streak": "3"}`; `{"streak": true}`; `{"streak": -1}` on a clean day | exit 0 | exit 1 with the matching reason |
| M3 | a broken `streak.json` on a `deliberate_skip` day and on a `no_games` day | exit 0 | exit 1 |

## Standing rows
- **An interrupt mid-step:** read-only.
- **A malformed or non-regular file:** M2, K5.
- **Midnight ET and the year boundary:** the file is read as it is at 08:30; a `streak.json` for last season is readable and not a problem (change set 2 reads it as fresh).
- **A failed or unconfirmed send:** the base's DM handling.
- **A restart mid-step:** stateless.

## The author's work (tests first; ONE commit)
`src/bts/day_outcome.py`, a new `tests/test_wr_f4_streak_file.py`. The existing files that pin `day_outcome` (found by searching the tests for `day_outcome`, `day-outcome`, `classify_day`, `DayOutcome`, `as_dict`: the contract check lists them by running) pass unchanged, except any assertion that compares the whole stdout JSON, which may gain the one key; the contract check lists those by running; anything else is a STOP.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN.

## Known and not fixed
1. The report comes at 08:30; the night's grading has already failed.
2. Only `streak.json` is read; other scorer inputs (a malformed pick or decision file) are not covered here.
3. A `streak.json` that is readable but wrong (a wrong count) is not detected.
4. The fix of the file itself stays a supervised repair (the 02:00 reconcile's full replay is the only rebuild path).

## Note for the contract check (version 1)
Change set F3 (on `fix/wr-f3`, being built) adds a problem state `skip_state_review` to `day_outcome.py` on the base `deliberate_skip` path and is merged before F4's author starts. F4's rows apply to it like any problem state (K3: its head and detail, with the "Also" sentence appended).
