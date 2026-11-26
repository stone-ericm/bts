# Kickoff: change set F5b, the entry watchdog all day with early runs that nothing trusts (2026-11-26, version 1; base `main` at `bf78b642`, F3 and F4 merged; lead ruling R68)

Finding BL-O4 of the closing test (`CLOSING-TEST.md`; Eric 2026-11-24: "widen the window to get every game"). F5 stopped after round 2 (`STOP-F5.md` on `fix/wr-f5`, way forward 1). Tier A (entry; timing). Session `bts-wr-fresh13`. Branch `fix/wr-f5b` from `main`.

## What is wrong on the base
The watchdog runs `*/15 10-23`: a pick whose earliest first pitch is before about 11:15 ET gets fewer entry checks than the window allows, and one before 10:05 gets none; the next morning pages ENTRY UNVERIFIABLE even when the entry was right.

## The design
1. **Round 1's schedule and gate, unchanged** (`fix/wr-f5` `5a80f0cc`, carried onto this branch): the line `*/15 * * * *` with F1b's lock and limit and `--cron`; with `--cron`, a start before 10:00 ET whose committed selection has a 10:00 to 23:45 quarter-hour inside its window returns `base_slot_available` before any account read, marker write or DM. Round 1's text edits and round 2's (`review-fix-f5-codex/report.md` and `review-fix-f5-codex-r2/report.md` "Edits") are applied.
2. **An early run keeps its own marker.** A `--cron` run that starts before 10:00 ET and goes past the gate (an "early run") reads and writes ONLY `health_state/pick_entry_check.early.json` (same format as the live marker, keyed by its `date`; its own confirmation is terminal for the next early run, and its tiers count for the next early run, so retries inside an early-only window are as stable as on round 1). An early run never reads, writes or replaces the live marker `health_state/pick_entry_check.json`.
3. **Every other run is the base.** Original-hours `--cron` runs and manual runs read and write only the live marker, with the base's code path unchanged: nothing an early run wrote is terminal for them, spends their tiers, or is displaced by them.
4. **The end-of-day health check** (`health.pick_entry.check`) for processed date D: the live marker exactly as on the base (every content, including a JSON list raising and an undecodable file); only when the live marker's date is not D (or it is absent) does it read the early marker, with a bounded non-blocking regular-file read, and apply the base's rules to it when its date is D (a malformed, non-object or non-regular early marker gives no alert). So an early-only window's failed alert DM still pages at end of day.
5. **The morning report** (`day_outcome`): a receipt from an early run (it carries `early_cron_run: true`, written only when true) counts as evidence for `entered`, `entered_game_unverified` or `not_entered` only when the day's final saved pick has NO original-schedule slot (a 10:00 to 23:45 ET quarter-hour on the slate date at which `picks.SUBMISSION_CUTOFF_MIN < minutes to pitch <= 75`, the early gate's own test). The final pick is read with a bounded non-blocking regular-file read; an unreadable pick counts as "no slot information" and early receipts are then NOT used (the base's count). Otherwise every receipt is used as on the base.

## Concrete examples (R70 item 2): must still catch, from both F5 rounds; each a test
| # | Sequence (round, finding) | Expected |
|---|---|---|
| R1 | (r1 B1) 09:40 commit, batter 1010, game 101 at 10:05; 09:45 batter match, no units binding (early `confirmed`); before 10:00 the pick is saved with a corrected 10:50 pitch; 10:00 capture binds game 102 | 10:00, 10:15, 10:30 read the account and warn as the base's line does; next morning `ENTRY UNVERIFIABLE` |
| R2 | (r1 B1) as R1 with a replacement commit | as R1 |
| R3 | (r1 B1) missing entry; an early run spent its tiers; revised 10:06 pitch | the 10:00 run sends the base's DM |
| R4 | (r1 B2) March 23's `dm_failed` live marker; March 24 committed at 04:55 for a 06:10 pitch; the 05:00 to 06:00 runs confirm; March 23's end-of-day health at 05:01 or later | the base's March 23 CRITICAL (the live marker is untouched) |
| R5 | (r1 test gap) a gate written to start at 10:15 | a test fails (10:06, 10:15, 10:20 pitches) |
| R6 | (r1) 10:20 pitch, 09:45 batter match, binding at 10:00 | 09:45 defers; 10:00 warns |
| R7 | (r2 #1) as R3 plus a manual check at 09:55 that sees an unavailable crosswalk; then 10:00 | the 10:00 run sends the base's one-minute warning (the manual run used the live marker only) |
| R8 | (r2 #1) early batter-only confirmation; correction adds a base slot; a manual 10:00 check after a wrong-game binding | the manual check warns as on the base |
| R9 | (r2 #2) 09:15 early wrong-game alert; 09:30 early verified correction and all-clear; pitch corrected to 10:50; 10:00 run | no second identical all-clear beyond what the base sends for its own runs (state which DMs the operator gets, in the test) |
| R10 | (r2 #4) a named pipe with no writer at the final pick path, an early `no_pick` receipt sealed | the morning report gives the base's result (NOT ENTERED, exit 1) and never blocks |
| R11 | (r2 #5) the live marker holds invalid UTF-8 | the base's health CRITICAL (`UnicodeDecodeError` through the runner) |
| R12 | (r2 #6) Dec 31 send fails; Jan 1 05:00 early run; a historical manual check for Dec 31 confirms; Jan 1 05:15 early run; Dec 31 health | no false `never confirmed` CRITICAL (early runs never touch the live marker) |
| R13 | (r2 test gaps) resetting on manual runs; preferring an early state when the live marker is for D | each fails a test |

## Must still happen
N1 to N8, M1 to M7 of the F5 contract (`kickoff-fix-f5.md` with Amendment 1) as they apply to round 1's line and gate; manual runs (no `--cron`) and original-hours `--cron` runs byte for byte the base's, including marker and receipt files; every run outside the window or with no committed pick makes no account request, DM or marker write; F1's wrong-game invariant; F1b's lock and limit; F3's and F4's morning-report states unchanged.

## Standing rows
- **An interrupt mid-step:** an early run writes only the early marker (atomic); an interrupted run leaves the previous early marker.
- **A malformed or non-regular file:** the live marker as on the base; the early marker and the final pick read bounded and non-blocking; a malformed early marker is "no prior state" for the next early run and "no alert" for health.
- **Midnight ET and the year boundary:** "before 10:00 ET" and the slot test use the run's ET date and the slate date; the early marker is keyed by its own date (a December 31 early marker is not January 1's).
- **A failed or unconfirmed send:** `dm_failed` in the early marker stays retryable for the next early run and pages at end of day (design 4).
- **A restart mid-step:** the command is one cron run; F1b's lock serialises runs.

## Known and not fixed
1. (r2 #3) A pick that gains and then loses an original-schedule slot before the morning: the morning report counts the early receipts (the final pick has no slot), so an absence seen only at an original-hours run in between can be reported as the early state.
2. An early-only window's alert history lives in the early marker; the original-hours runs and the dashboard do not show it.
3. The window is computed in ET wall time (no game starts in the clock-change hours); the morning count includes runs that made no account read (round 1's limits).
4. A pick delivered in the last minutes before the cutoff can fall between two runs (W16); a first pitch after midnight on the slate's own date is not checked after midnight; the box runs the old line until the installer is run there.

## The author's work (tests first; ONE commit)
`scripts/cron-setup-hetzner.sh`, `src/bts/cli.py`, `src/bts/entry_receipt.py` (the true-only key), `src/bts/health/pick_entry.py`, `src/bts/day_outcome.py`, `ARCHITECTURE.md`, `docs/ops/pick-entry-receipt-v1.md`, `tests/test_wr_f5b_watchdog.py` (round 1's and round 2's tests that still apply may be carried from `fix/wr-f5`), and the two existing hour assertions F5 Amendment 1 names. Every existing test file found by the F5 search stays green; any other assertion that must change is a STOP.

## Review
Codex xhigh contract check first (it runs R1 to R13 against a scratch build and the existing pinning tests); author; gate; fresh Codex xhigh and Carbon review; at most two rounds; round 2 bounded (R70 item 1); merge on a Codex SIGN. If it stops, BL-O4 goes to Eric (R68).
