# Verdict

**BLOCK.** K2b’s longer journal line can make a restart miss the only game’s entry cutoff in a shared-pipe state where round 2 reached the check on time.

## 1. The removal

- Reviewed `git diff 712351a5 0697110b -- src`. K2 adds no boundary DM, dispatch call, dispatch thread, or receipt path. The sole worker is a daemon that drains the journal queue via `os.write(2, ...)`. The shared dispatcher is untouched.
- Qualifying starts with a configured recipient in private, DM, and public modes made zero boundary sends and created zero boundary receipts. Captured configured and fallback lines matched Amendment 5, including repeat-start counts. `probe_pipes.py isolated` also verified one exact line per start from a worker thread after the sink resumed.
- Ran round 2’s `probe_shutdown.py` against archived `6f0ab421`. Its failed-send case still needed a drain after the 1.6-second limit. The K2b replay held the daemon writer on a real full stderr pipe while interrupting `run_day`. SIGINT escaped and the process exited in 0.047 seconds before any drain.
- With only the boundary writer stalled, the first checks reached 10:15, 10:15, and 18:59:59 ET with the game enterable in each case.
- **New cutoff case.** At 18:59:59 ET with one 19:05 game and 690 bytes free in the shared pipe, let the daemon write before the scheduler prints `Running lineup check...` (`src/bts/scheduler.py:3218`). Round 2’s 332-byte line left enough room for the scheduler. K2b’s 482-byte line made that scheduler write wait for the reader. A two-second stall on the pinned ET clock moved the first check to 19:00:01, and `run_single_check` received the only game in `unavailable_game_pks`. The control with the warning disabled kept it enterable. The same result held at 666, 672, 678, and 684 bytes. See `shared-edge-{round2,head}.log`.

## 2. Remaining behaviour

- Ran the shipped-artifact grid independently against `git archive 712351a5 src` and the checkout. Both runs printed and checked `bts.__file__`. All 373,188 positive-day decisions and display bars matched, digest `7a8f719bafa7017e2f24327c1b5984ca0ed368eaa2838d20906971625e8177c7`.
- On 2027-09-27 with both saver settings, real `select_pick` and `run_and_pick` with dated contest state agreed. The actions at streaks 20, 55, 56, 57, and 120 were double, double, single, double, and double with a beatable best. A trusted best of 25 still stopped streak 20. The archived base skipped all those rows. The season walk’s makeup fixture matched K2b with raw zero and one effective day.
- Bare contexts, `decide_action`, the standalone bar, and health matched the base on the sampled dates, including raw zero in configured and unconfigured years. Real morning refusal and stale-reading DM bodies matched byte for byte across 18 starts covering three modes, missing/fresh/stale state, and both sides of the morning lead cutoff.

## 3. R64 journal text

- Found no ordinary state where following the advisory exactly would send or enter a wrong pick, corrupt state, lose an alert, or lose a day. It asks the owner to verify the regular-season finale and update the exclusive boundary for a later run. In the operator replay, storing Sep 28 after confirming a Sep 27 finale gave one raw day and preserved the already-sent pick byte for byte. The shared-pipe failure occurs during the write itself.

## 4. Tests and scratch changes

- Ran all 40 files selected by the brief’s test search, including all 52 cases in `tests/test_wr_k2_season_boundary.py`, plus `tests/scripts/season_walk/test_season_walk.py`. Across 41 separate runs, 1,506 passed, 12 skipped, and one expected failure.
- Restoring a boundary WARN through the dispatcher failed all three mode tests: they captured both a send and a receipt. Writing the line on the scheduler thread failed all three stalled-journal timing tests. Removing the floor from both places in `select_pick` failed all 24 selection tests. Each scratch change was tested alone and detected.

## Edits

None. The journal text matches Amendment 5 and the blocking finding is in the write path.

## Notes

- Round 2 had already documented a cutoff miss with 510 bytes free in the shared pipe. The comparison at 690 bytes is the new failure. Amendment 5 also records that a nonblocking sink can lose the line on `EAGAIN`.
- The OS declined the requested `nice -n 19` priority. Runs completed serially under timeouts in ET. Tracked files are unchanged.

ran: source diff, isolated and shared pipes, original round-2 shutdown and K2b SIGINT, archived-base grid/raw/morning comparisons, makeup entry points and walk, operator replay, 41 test files, and three scratch changes; did not run: full fast suite or live delivery and deployment
