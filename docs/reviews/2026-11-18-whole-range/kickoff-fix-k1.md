# Kickoff: change set K1, the live contest-state refusal DM fits the chat limit, and the four in-day alerts dedup under the slate's date (2026-11-26, version 1; base `main` at `a5fa7466`)

Findings F2-1 (blind B2) and C3-2 (Carbon slice 3 B1) of the closing test re-run (`CLOSING-TEST-2.md`, `closing-test-2/repro/dm_len.py`); ruling R72 (K1). Tier A (it changes what the operator is told and when an alert is suppressed). Session `bts-wr-fresh15`. Branch `fix/wr-k1`. R70 applies (concrete examples before the author; a contract check that runs them with the existing tests pinning the touched functions; bounded round 2). Operator text is judged under R64.

## What is wrong on the base
1. **F2-1.** `scheduler._alert_contest_state_failure(config, error)` sends `str(error)` as the CRITICAL's whole message. For the live refusal (`season_contract.live_automatic_refusal`, raised by `contest_state.load_decision_streak_state(require_contest_state=True)`), that text is the reason line plus a dated template and a repair procedure that repeats `--picks-dir <path>` several times. With the box's path (`/home/bts/projects/bts/data/picks`) the DM (header `BTS health CRITICAL alert(s):` plus `- [contest_state] `) is 1,146 characters for an absent 2027 switch and 1,147 for a revoked one (746 for "no automatic observation for this activation"). The project treats 1,000 as the chat limit (`day_outcome.DM_MAX_CODE_POINTS`); `dm.send_dm` neither truncates nor splits. The send fails, the status file records `failed`, and the failure banner (`health.alert._resurface_note`, about 70 characters) makes the next DM longer still. After a late start or a restart (first check under 10 minutes away: no morning check, 6b-1) or a switch revoked after the morning check, no contest-state DM reaches the operator during the entry window.
2. **C3-2.** `_alert_contest_state_failure`, `_alert_missed_pick`, `_refuse_late_delivery` and `_refuse_void_game_delivery` call `health.alert.dispatch_dm_for_health_alerts` without `now_et_date`, so the dedup record in `health_state/health_dm_delivery_status.json` takes the WALL-CLOCK ET date. An in-day CRITICAL for slate D sent after 00:00 ET is recorded under D+1; a CRITICAL of the same source for slate D+1 later that day is then suppressed as "already covered by today's DM" (a lost alert), and D's end-of-day health run (which dedups on the processed date D, `runner.py`) does not dedup against it (a repeat). C2's unknown-outcome alert and the health runner already pass the slate's date. It needs an in-day alert after midnight, so a slate-D game starting after about 00:15 ET (or a fallback running past midnight).

## The design (`src/bts/scheduler.py` only)
1. **The short refusal DM.** `_alert_contest_state_failure(config, error, *, slate_date=None)` sends ONE CRITICAL, source `contest_state` (no incident key: the dedup key `CRITICAL:contest_state` is the base's), whose message is exactly:
   `No pick was sent for {D} at this check: a contest-state check refused it. Reason: {R} Run bts season-contract status --picks-dir DIR (DIR is the absolute scheduler picks path) for the full reason and the ways to repair it; the scheduler journal has the same text. A later check or fallback can still pick if one remains and the state is usable then.`
   - `{D}` is the slate date string the caller passes (`date`), or `today` (the literal word) if none was passed.
   - `{R}` is the FIRST LINE of `str(error)` with surrounding whitespace stripped; if it is longer than 400 code points, its first 397 code points followed by `...`; if `str(error)` raises, the exception's type name; if empty, `(no reason text)`. Nothing here may raise.
   - The fixed text is at most 400 code points, so with `{R}` at most 400, the header (48) and the failure banner (about 70) the DM body stays under 1,000 code points for EVERY error, whatever its length.
   - The FULL text stays in the journal: every caller already prints `CONTEST STATE ERROR — …: {e}` or `FALLBACK BLOCKED — contest state invalid: {e}` to stderr before the alert (five call sites: `_deliver_and_lock_pick` twice, `run_single_check`, `run_day` twice). Those print lines are unchanged. `bts season-contract status` prints the full live refusal under `live decision now: refused; …` (unchanged).
2. **The slate's date for dedup.** The four functions pass `now_et_date=<the slate's date>` to the dispatcher: `_alert_contest_state_failure` from `slate_date` (each of the five call sites passes `slate_date=date`); `_alert_missed_pick` from a new keyword `slate_date` passed by `_maybe_alert_missed_pick` (`date`); `_refuse_late_delivery` and `_refuse_void_game_delivery` from their existing `date` parameter. The string is parsed with `datetime.date.fromisoformat`; if it is missing or does not parse, `now_et_date` is omitted (the base's wall-clock date), never an exception.
3. **Nothing else changes:** the other three DMs' bodies (`missed_pick`, `late_delivery`, `postponed_pick`) are byte for byte the base's; the status file path, its format, the failure banner rule, the sources and keys, the archive and state writes of the refusals, the return values and the journal lines are the base's.

## Concrete examples (R70 item 2): must change (RED on the base)
| # | Sequence | Base | K1 |
|---|---|---|---|
| E1 | Absent 2027 switch, picks dir `/home/bts/projects/bts/data/picks`, the real live loader's error into `_alert_contest_state_failure` (captured `send_dm`) | 1,146-character DM | one DM under 1,000 code points, naming `Live decision refused: season 2027 switch is absent.` and `bts season-contract status` |
| E2 | Revoked 2027 switch, same path | 1,147 | under 1,000 |
| E3 | EVERY refusal reason the live read can raise, with the box's path: corrupt switch; legacy manual selected; switch absent, for another season, revoked; no automatic observation; automatic schema, season, activation id; each `ContestStateError` raise site in `contest_state.py` reachable with `require_contest_state=True` (malformed, wrong types, undated, model fallback errors); the three `picks.saved_pick_refusal` texts (no record, preview record, other observation); a synthetic 100,000-character one-line error; a multi-line error whose first line is empty | some over 1,000 | every DM under 1,000 code points, ALSO with a prior `failed` status present (the banner) |
| E4 | Slate D = 2027-06-14. `_alert_contest_state_failure` at 2027-06-15 00:02 ET (sent); then at 2027-06-15 08:00 the same function for slate 2027-06-15 | the second is suppressed (record says 2027-06-15) | the first is recorded under 2027-06-14; the second is SENT |
| E5 | E4 for each of `_alert_missed_pick` (via `_maybe_alert_missed_pick`), `_refuse_late_delivery`, `_refuse_void_game_delivery` | suppressed | sent |
| E6 | Year boundary: slate 2027-12-31, alert at 2028-01-01 00:02, then a same-source alert for slate 2028-01-01 at 09:00 | suppressed | sent |

## Concrete examples: must still happen (identical to the base)
| # | Sequence | Must still happen |
|---|---|---|
| S1 | Two contest-state refusals for slate D at 14:00 and 15:00 on D | one DM (the second deduplicated) |
| S2 | A contest-state CRITICAL then a `missed_pick` CRITICAL the same day D | both sent (a new source) |
| S3 | A prior `failed` status, then a refusal | the banner is prepended and the DM is sent (and is under 1,000: E3) |
| S4 | `missed_pick` (live and not-live texts), `late_delivery`, `postponed_pick` | bodies byte for byte the base's |
| S5 | No `dm_recipient` | the base's `skipped_no_recipient` record, no send |
| S6 | `send_dm` raises | the base's `failed` record; nothing escapes the alert function that the base did not let escape |
| S7 | A slate date that does not parse (`"bad"`, `None`, `""`) | the alert is sent and recorded under the wall-clock date (the base's behaviour), no exception |
| S8 | Each call site's journal line | unchanged text, printed before the DM |
| S9 | C2's unknown-outcome alert, the morning check, the stale-reading warnings, the health runner | untouched |

## A consequence to state, not a defect
After K1, D's end-of-day health run (dedup on D) treats an in-day `contest_state` CRITICAL for D sent after midnight exactly as one sent before midnight: if its alert set is covered by the record, it is suppressed (base: repeated). This is the base's same-day behaviour extended to the slate; the in-day DM for D was delivered.

## Standing rows
- **An interrupt mid-step:** the dispatcher's handling is the base's; K1 adds only string work that cannot raise and a date parse inside `try`.
- **A malformed or non-regular file:** the status file is read by 6b-2's helper (unchanged); a malformed slate date is S7.
- **Midnight ET and the year boundary:** E4 to E6.
- **A failed or unconfirmed send:** S3, S6; the short DM is what makes the first send fit.
- **A restart mid-step:** stateless apart from the status file (unchanged format).

## Known and not fixed
1. The DM no longer carries the repair commands: the operator runs `bts season-contract status` (read-only), which prints the full live refusal; for a saved-pick refusal (5a) status shows the live state accepted, and the reason is in the DM's `Reason:` and the journal.
2. `{R}` is cut at 400 code points; the full text is in the journal and status.
3. The status file holds one day: an in-day alert for D written after a record for D+1 already exists replaces it, so a D+1 alert of a source already sent can repeat (never lost).
4. The end-of-day digest (`runner.py`) has no total length bound (as `CLAUDE.md` records); K1 does not touch it.
5. The 1,000 limit is the project's assumption (graphemes counted as code points, an upper bound), not re-measured against Bluesky.

## The author's work (tests first; ONE commit)
Files: `src/bts/scheduler.py`, a new `tests/test_wr_k1_refusal_dm.py`. Tests first, RED on the base for E1 to E6, through the real live loader and the real CLI (`season-contract enable` / `revoke`), pinned ET clocks, temporary picks directories, a captured `send_dm` that REJECTS a body over 1,000 code points (as a real over-length send would fail). Existing tests that pin the touched functions are found BY SEARCH (`grep -rlE "_alert_contest_state_failure|_alert_missed_pick|_refuse_late_delivery|_refuse_void_game_delivery|health_dm_delivery_status|dispatch_dm_for_health_alerts|contest_state.*CRITICAL" tests scripts`) and pass. Existing assertions that may change are ONLY those the contract check lists by running (expected: test doubles of `_alert_contest_state_failure` that do not accept the new keyword, a source-line search for the old call text in `tests/test_alert_status_callers_cs6b2.py`, any assertion of the old full contest-state DM text); anything else is a STOP.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds (round 2 bounded, R70); merge on a Codex SIGN or SIGN WITH EDITS applied.
