# Kickoff: change set F5, the entry watchdog's cron window (2026-11-25, version 1; base `main` at `a4e8278d`, F1 and F1b merged)

Finding BL-O4 of the closing test (`CLOSING-TEST.md`; `closing-test/codex-blind/report.md`, "Enter cron misses early slates", fixture `closing-test/codex-blind/evidence/cron-early-entry.json`). Eric, 2026-11-24: "widen the window to get every game" (`FINDINGS.md`). Tier A (entry; scheduler timing). Session `bts-wr-fresh12`. Branch `fix/wr-f5`.

## What is wrong on the base
`scripts/cron-setup-hetzner.sh` installs the entry watchdog as `*/15 10-23 * * *`: it runs from 10:00 to 23:45 ET. The command itself checks only from 75 minutes before the pick's earliest first pitch until the submission cutoff (first pitch minus 5). A pick whose earliest game starts before 11:15 ET gets fewer checks than the window allows, and one that starts before 10:05 gets none: for a 06:10 opener (delivered at 05:10) the first run is 10:00, 230 minutes after first pitch; nothing checks the entry before the pitch, and the next morning `bts day-outcome` pages ENTRY UNVERIFIABLE even when the entry was right.

## The design
1. **One cron field changes.** The line becomes `*/15 * * * * $PREFIX /usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840 $UV_BIN run bts check-pick-entered …` (every 15 minutes, every hour), with the same lock, limit, command, flags, log and marker as F1b (R67: F5 keeps the lock). Nothing else in the installer changes: the line is still installed only under `entry_intent = "enter"`, `show`/`install`/`remove` behave as they do, the other lines are byte for byte the same.
2. **No in-code window changes.** The command already limits itself: with no pick file, a pick that is not committed, a time outside `(cutoff, first pitch − 75 min]`, or a day already confirmed, it returns before any network request. So the extra 40 runs a day (00:00 to 09:45) make no request to MLB unless a committed pick's first pitch is within 75 minutes; the account's traffic grows only on days with a game before 11:15 ET, by the checks that were missing. Every run still publishes one local receipt (96 a day instead of 56).
3. **Late games** are already covered and stay covered: the line already ran through 23:45, and (the session's reading, for the contract check to confirm from the schedule data) no regular-season first pitch is after about 22:40 ET.
4. **The fetch minutes are unchanged:** the watchdog still runs at :00, :15, :30, :45; it shares a minute with `fetch-contest-streak` at 10:30 and 13:30 exactly as on the base, and with no other authenticated job (the other fetches run at 01:10 and 02:10).
5. **Texts that state the hours** are corrected to "every 15 minutes, all day": the installer's header comment, `ARCHITECTURE.md` (the `bts check-pick-entered` bullet), `docs/ops/pick-entry-receipt-v1.md` ("The cron runs …"). No checklist row and no `INCIDENT.md` text is touched (they wait for January, R63).
6. **On the box the line changes only when the installer is run:** a deploy never touches crontab. The restored box's crontab still holds whatever the snapshot held (the watchdog line was commented out on 2026-09-14); the new line exists after `bash scripts/cron-setup-hetzner.sh install` under `entry_intent = "enter"`, which the season-start checklist already requires (row A5). An installed old `*/15 10-23` line is replaced, not duplicated (the installer removes every line carrying its marker first).

## Must still happen (identical to the base)
| # | State | Must still happen |
|---|---|---|
| N1 | `entry_intent = "research"` | no watchdog line; the other lines unchanged; fetch lines without `--dm-recipient` |
| N2 | `entry_intent` missing, malformed, or disagreeing with the delivery mode; the live-gate problem | `show` and `install` refuse, exit 1, crontab untouched |
| N3 | `install` over a crontab holding foreign lines, an old BTS block, a commented-out BTS line, no crontab, an unreadable crontab | exactly the base's result apart from the one field |
| N4 | `remove` | removes every marked line, the new one included |
| N5 | every other generated line | byte for byte the base's |
| N6 | the command at any time outside its own window, with no pick, an uncommitted pick, or a confirmed marker | returns before any request (no account traffic), as the base; shown for 00:00, 03:00 (the preview has just written an uncommitted pick file), 05:00, 09:45 |
| N7 | a 19:10 game | the same five checks as the base (18:00 to 19:00) |
| N8 | F1b's lock and limit (`tests/test_wr_f1b_lock.py`: contention, the limit, `840 + 30 < 900` read from the generated line, the interruption cases) | unchanged and green; the interval is still 900 seconds at every hour |

## Must change (RED on the base)
| # | State | Base | F5 |
|---|---|---|---|
| M1 | `show`/`install` under `enter` | the line starts `*/15 10-23 * * * ` | it starts `*/15 * * * * `, followed by F1b's lock and limit unchanged |
| M2 | the closing-test case: a pick committed at 05:10 for a 06:10 first pitch; cron times taken from the generated line | no run inside the command's window | the 05:00 run finds no committed pick yet; the runs at 05:15, 05:30, 05:45 and 06:00 (55, 40, 25 and 10 minutes before the pitch) are inside the window and check the entry; the 06:15 run is outside |
| M3 | a 10:05 first pitch (cutoff 10:00; the window opens at 08:50) | no run inside the window (10:00 is at the cutoff and is excluded) | the runs at 09:00, 09:15, 09:30 and 09:45 check the entry |
| M4 | `install` over a crontab that holds the old `*/15 10-23` line | (the same line again) | one watchdog line, the new one |

## Standing rows
- **An interrupt mid-step:** the installer's pipeline into `crontab -` is unchanged; an interrupted install leaves what the base leaves.
- **A malformed or non-regular file:** the crontab filter and its refusals are unchanged (N3); the command's own file handling is not touched.
- **Midnight ET and the year boundary:** the new runs at 00:00 to 00:45 read the NEW date's pick, as every run reads `now`'s date: a pick whose first pitch is after midnight on the slate's own evening is not checked after midnight (no regular-season game starts then; stated limit). On January 1 there is no pick. **The clock changes now fall inside the line's hours** (F1b's standing row said the repeated and skipped hours were outside `10-23`; that is no longer true): on the November change cron can run the 01:00 to 01:45 slots twice and on the March change it skips the 02:00 hour. Every such run is outside any entry window and returns before a request; the contract check confirms what Debian cron does and that the interval between two runs can never fall under 870 seconds in a way that matters (a run outside the window holds the lock for well under a second).
- **A failed or unconfirmed send; a restart mid-step:** not touched (the command is unchanged).

## The author's work (tests first; ONE commit)
`scripts/cron-setup-hetzner.sh` (the one field and the header comment's hours), `ARCHITECTURE.md` and `docs/ops/pick-entry-receipt-v1.md` (the hours; the receipt guide's "56 slots per day" becomes 96), a new `tests/test_wr_f5_watchdog_window.py`, and ONLY the existing assertions that pin the generated line's hours (`tests/test_entry_intent.py`: `entry[0].startswith("*/15 10-23 * * * ")`; and any assertion in `tests/test_wr_f1b_lock.py` that pins `10-23`, which the contract check lists by running). The fixtures in that file that write an OLD installed line stay as they are (they are the M4 case). If any other existing assertion must change, STOP and ask.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN.

## Known and not fixed
1. A pick delivered in the last minutes before the cutoff can still fall between two runs (W16, accepted).
2. A first pitch after midnight ET on the slate's own date is not checked after midnight.
3. The watchdog and `fetch-contest-streak` still share the minutes 10:30 and 13:30.
4. The box runs the old hours (or no watchdog line) until the installer is run there.
5. 40 more receipts a day are written outside the entry window (one per run, as every run writes one); nothing prunes them.

## Amendment 1 (2026-11-25, after the contract check `WRCHKF5-d88824e2`, BUILD WITH THESE EDITS; `contract-check-f5/`). It overrides the text above wherever they differ.

The check replayed the closing-test case (base: no slot, morning ENTRY UNVERIFIABLE; F5: the 05:15 run confirmed the right entry, the morning was clean), computed every first pitch minute by minute (no base slot disappears; only pitches from 00:06 to 11:00 gain early slots), read 2,425 final 2025 feeds (latest start 22:15 ET, none before 06:00), and found **one way the schedule change alone loses a base protection**: for a pitch between about 10:06 and 11:00, a new 09:45 run can confirm a batter match before the units capture binds the game, and the terminal `confirmed` then stops the 10:00 run that would have seen the wrong game (and the morning alert). So F5 now has a small command change.

**Adopted verbatim from the check's "Replacement texts":** "What is wrong on the base"; Design 1, Design 2, M1 and the new regression (the `--cron` option and its early gate, quoted below); Design 2's "returns before any network request" and limit 5; Design 3 and limit 2; Design 4; Design 5; Design 6 and M4; N3 and the malformed-file row; N6, N7, M2 and M3; N8 and the midnight row EXCEPT its two sentences on UTC arithmetic (below); the author's work EXCEPT the UTC arithmetic and the day-outcome sentence (below). The two existing assertions that change are exactly the check's list: `tests/test_entry_intent.py:169` (`startswith("*/15 * * * * ")`) and `tests/test_wr_f1b_lock.py:59` (the second element becomes `"*"`).

The early gate, as adopted: "Install `*/15 * * * *` under enter with F1b's lock, limit, redirects and marker. Add a watchdog CLI option `--cron` to opt the installed line into the following check. For a CLI start before 10:00 ET, after the commit and window gates, first ask whether at least one 10:00–23:45 quarter-hour on the target ET date is inside this committed selection's window. If so, record the no-attempt outcome `base_slot_available` and return before account reads, marker writes or DMs. If no base slot exists, run the existing verification and alert rules. Keep ordinary manual invocations and all other generated lines as on the base." Rows: M5, a 10:50 slate with a 09:45 batter match and a 10:00 wrong-game binding: the 09:45 run returns `base_slot_available` and the 10:00 run warns as on the base; M6, a 09:45 entry removed before 10:00: the base's alerts; M7, a 06:10 and a 10:05 pitch: no base slot, so the early runs check (the closing-test case is closed).

### Not adopted (the session's ruling), each written as a known limit
1. **No change to the command's time arithmetic (UTC instead of ET wall time).** The check showed the window is computed in wall time across a clock change (a synthetic 03:05 EDT pitch, a 01:55 pitch in the repeated hour). No regular-season game starts between 00:06 and 06:00 ET (the 2025 scan: none before 06:00), so no real window meets a clock change; changing the arithmetic would touch every run. Limit: "The window is computed in ET wall time; across a clock change a window would be off by an hour. No game starts in those hours."
2. **No change to the morning report's sentence** ("confirmed in 1 of N checks", where N now counts the no-attempt receipts of the extra runs). It is imprecise, not harmful under R64 (the state and the exit code are unchanged), and F3 and F4 change `day_outcome.py` separately. Limit: "The morning report's count of checks includes runs that made no account read."
