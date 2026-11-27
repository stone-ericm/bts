# Verdict

**BLOCK.** The move fixes the same-day cutoff miss. It can now strand end-of-day work behind a full journal pipe. In the late-restart replay, the watchdog cut off a skip decision write and a health DM that completed with the line disabled.

## 1. Cutoff, fallback and delivery

- Replayed the earlier shared-pipe checks through delivery in `probe_pick_pipe.py`. At 510, 650, 666, 672, 678, 684, 690, 696 and 704 bytes free, both runs checked the only game, refreshed at final fallback and reached the DM transport at 18:59:59 ET. The game remained enterable and the 19:00 cutoff, stored delivery and scoreable decision agreed.
- At 10:05 ET for a 13:15 one-game slate, the first check ran at 10:15 with 350 or 690 bytes free. All three checks, their cutoff inputs and the 12:40 fallback and delivery matched the line-disabled controls. At 350 bytes the pre-existing output delayed the second check to 11:27 in both runs. See `pick-pipe.log`.

## 2. Placement, later work and decision parity

- `git show ef948b99` and the `run_day` paths confirm one call in its own `try` after section 6 completes and before section 7, including when section 6 has no rechecks or breaks on a locked pick. Off-day and dry-run returns bypass it. An exception or interrupt in section 6 can also bypass it. The daemon writer may emit the line after section 7 begins.
- `probe_end_of_day_pipe.py` left 513 bytes free after the day's pick work and let the 482-byte line write first. With a 20-minute stall, the normal lookahead fetched and saved the same wake-up in both runs, then its status print blocked only with the line enabled. Result polling moved from 19:15:00 to 19:19:59 ET and saved the same suspended result. On the skip run, the final decision write, health run, test DM and idle began after midnight instead of 23:45. Normal wake-up and return still occurred at 10:00 in these finite-stall cases.
- With the same pipe pressure at 23:45 ET and a simulated 30-minute watchdog expiry, the control wrote the skip decision, ran health, sent the test DM and reached idle. The warning run did none of those. The real CLI date selection targeted the new date on restart, so the default restart would skip the old day's end-of-day work. The remembered skip candidate remained on disk. With a failed lookahead fetch, the retry message blocked *inside* the helper, leaving `next_wakeup` unset and losing the control's midnight handoff. The successful-fetch case kept the saved wake-up but never reached the sleep. See `end-of-day-pipe.log`. Systemd expiry was simulated using the unit's `WatchdogSec=1800`.
- The stated later-line delay is real, and the same-run pick protection holds. Amendment 7 omits the lost work and altered restart path that can follow a stall. The advisory can also appear in the afternoon, as the thin-slate replay showed.
- The strategy diff from `712351a5` only adds K2's day floor. The archived-base comparison matched all 15,880 sampled positive-day decisions and bars with the shipped policy files. The retained full-grid test also matched its checked-in base results. The makeup replay exercised `select_pick`, `run_and_pick` and the walk with both saver settings. The qualifying 2027-09-27 rows played with one effective day where the base skipped. Streak 20 with trusted best 25 still skipped. See `grid-*.log` and `makeup-*.log`.

## 3. R64 and the advisory

The count and statuses come from the startup fetch, even if the line appears after a game's status changes. The instruction to verify the finale and update the boundary for a later run remains safe to follow. I found no R64 blocker in the text.

## 4. Tests and scratch changes

- Ran `test_wr_k2_season_boundary.py` (54 passed), `test_wr_k1_r2.py` (45 passed) and the other files from the specified search. The 39 separate runs had 1,421 passes, 12 skips and one expected failure.
- Moving the call before the first check failed both the journal order and 690-byte cutoff tests. The cutoff test got 19:00:01 with the only game unavailable. Moving the call before the final fallback failed the journal order test. Both changes were caught in separate scratch runs. See `mutation-*.log`.

## Edits

- Journal opening and policy sentence. “SEASON BOUNDARY BEHIND SCHEDULE — {D} ET: At scheduler startup, MLB listed {n} regular-season game(s) with that official date and reported statuses other than postponed or canceled. {boundary_source} was {E} for this run, giving a raw count of zero. With that boundary, any fresh selection for {D} with an available predicted candidate uses the last-day policy.” Keep the instruction to verify the finale and update the boundary for a later run.
- Amendment 7 item 2. “A run that exits before the boundary check queues no line. The advisory uses the startup schedule and is queued after that run's pick work, which can end before evening or without a pick. Its bytes can block later writes to shared stderr, delaying lookahead state, result polling, final decisions, health checks and DMs, and the wake-up or restart. If the watchdog stops a blocked run after midnight, the default restart processes the new date and can leave the previous slate's work unfinished.”

## Notes

- The previously documented possibility of losing the advisory itself on `EAGAIN` still applies. The OS refused the requested nice priority. Runs were serial, time-bounded and set to ET. Tracked files are unchanged.

ran: source review, shared-pipe cutoff and end-of-day replays, archived-base grid spot check and makeup replay, 39 test files, and both scratch changes; did not run: full fast suite, the full independent archived-base grid, live systemd or live delivery
