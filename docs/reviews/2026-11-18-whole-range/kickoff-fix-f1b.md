# Kickoff: change set F1b, F1's one follow-up (2026-11-25, version 1; base `main` at `1e69d0f0`; F1's code `1d30de86` and `15b27474` on branch `fix/wr-f1`)

Ruling R67 (lead, 2026-11-25, after `STOP-F1.md`): F1b is F1 as it stands plus a singleton lock on the `check-pick-entered` cron line, so no two cron runs can overlap; the defect class is named; a must-still-catch table holds both rounds' findings; fresh Codex xhigh and Carbon review, at most two rounds; if it stops, F1 goes to Eric. Tier A. Session `bts-wr-fresh12`. Branch `fix/wr-f1` (continued). The contract for F1's behaviour is `kickoff-fix-f1.md` with Amendments 1 to 4 and stays in force; this file adds to it.

## The defect class: two writers of the watchdog marker
`health_state/pick_entry_check.json` is read once near the start of a run and written near its end, from that early picture. Both review rounds found a wrong-game observation lost through it:
- round 1 (`WRF1REV1-d44caf34`): a later SERIAL run overwrote the only sign of the observation (closed by Amendment 4: the observation is its own field, written by the one marker writer; round 2 confirmed it over 54 serial orders);
- round 2 (`WRF1REV2-dffa862e`): an OVERLAPPING run that started earlier wrote its older picture over a newer run's marker.
**Writers of the marker (complete list; the contract check confirms it by search):** only `cli._check_pick_entered`, through `mark`. Nothing else in `src/` or `scripts/` writes the file; `health/pick_entry.py` only reads it. So two writers exist only when two invocations of `bts check-pick-entered` on the same picks directory are alive at once. On the box the command is started by ONE cron line (and by an operator's hand).

## The design
1. **The cron line gains a singleton lock and a run limit.** In `scripts/cron-setup-hetzner.sh` the watchdog line becomes
   `*/15 10-23 * * * $PREFIX /usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout -k 30 840 $UV_BIN run bts check-pick-entered --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> $LOG_DIR/cron.log 2>&1 $MARKER`
   (the hours are unchanged here; change set F5 widens them and keeps the lock and the limit). `flock -n` is the pattern of the `check-results` line: when a run is still alive, the next slot's run does not start (exit 1, nothing printed, no receipt). `timeout -k 30 840` ends a run after 14 minutes (TERM, then KILL 30 seconds later), so a run that never returns cannot hold the lock past the next slot: without it the lock would turn one stuck run into NO watchdog at all, on that day and every later one, with no alert before the morning report.
   - **The limit is shorter than the cron interval** (the lead, 2026-11-25): 840 seconds plus the 30-second kill grace is 14.5 minutes, under the 15 minutes between slots, so a run that hangs costs its own slot, not the day. Any change to the interval must keep the limit below it; a test pins `limit + grace < interval` from the generated line itself.
   - **What a killed run leaves on disk:** no half-written marker. The marker has one writer, and it writes a temporary file and renames it (`_atomic_write_json`), so the file is either the earlier complete marker or the new complete one. A run killed before its write leaves the earlier marker and no receipt; the next slot's run repeats the check (and can repeat a DM the transport had already accepted).
   - **The box has GNU `timeout`** (`/usr/bin/timeout`, Debian coreutils) and `flock` (`/usr/bin/flock`, util-linux; the `check-results` line already uses it). `-k` is GNU syntax. The tests drive both under SHIMS on any machine (a `flock` shim that takes a real advisory lock with `fcntl`, a `timeout` shim that ends its child after the given seconds), run from the generated line with the two absolute paths mapped to the shims, and with the real tools as well where the machine has them.
2. **No change to the command.** F1's code is merged as it stands (`1d30de86`, `15b27474`).
3. **Contract-text corrections from the reviews, applied to `kickoff-fix-f1.md` as its Amendment 5:** A5.4's "overlapping runs can both send" becomes the true limit below; Carbon round 1's two edits (A6.4, A6.5) replace those sentences.

## Must still catch (the findings of both rounds; each must hold on the merged tree)
| # | Finding | Must still happen | How it is shown |
|---|---|---|---|
| K1 | round 1 B1: wrong game seen, its first DM fails, then a crosswalk gap, no entry or a batter mismatch, then a batter match with an unknown binding | no `confirmed`, no all-clear; the first wrong-game warning is sent | F1's `tests/test_wr_f1_review.py` (adopted reviewer regression; the walk of 144 pairs and 1,728 triples) |
| K2 | round 2 B1: run A starts, run B starts while A is alive | under the installed cron line run B does not start: with the lock held, the line's command exits without running `bts check-pick-entered` | a test that runs the GENERATED cron line's lock-and-limit prefix with a real `flock` (where the machine has one; the deploy gate's Linux runner does) around a stand-in command: while one holder is alive a second invocation exits non-zero and its command never runs; after the holder ends the next invocation runs |
| K3 | a run that never returns | it is ended by the limit and the lock is free for the next slot | the generated line contains `timeout -k 30 840` between the lock and the command, in that order; a real-`timeout` test with a small limit shows a stand-in that sleeps is ended and the lock is released |
| K4 | the reviewer's interleaving probe itself (`review-fix-f1-codex-r2/interleaving.py`: B nested inside A's fetch, in one process) | still reproduces: the command alone does not serialize; this is the stated limit, pinned so that nobody reads the lock as a property of the command | adopted as a test that asserts the overwrite happens without the lock (an `xfail`-free characterization, named as a known limit) |
| K5 | everything F1's rounds showed to hold (N1 to N13, M1 to M13, the texts, the standing rows) | unchanged: no line of `src/` changes in F1b | the F1 test files, unchanged |

## Must still happen in the installer (identical to the base apart from the one line)
`entry_intent = "research"`: no watchdog line. Missing or disagreeing intent, the live-gate problem: refuse, crontab untouched. `install` over foreign lines, an old BTS block, a commented-out line, no crontab, an unreadable crontab; `remove`; every other generated line: byte for byte the base's. The generated watchdog line still starts `*/15 10-23 * * * ` (the existing assertion in `tests/test_entry_intent.py` stays).

## Standing rows
- **An interrupt mid-step:** the limit's TERM ends the Python process without running its `finally` blocks: no receipt for that run and no marker write (the marker write is atomic, so the earlier marker stays); a DM already accepted by the transport can repeat on the next run (as an interrupt on the base).
- **A malformed or non-regular file:** the lock file is created by `flock` if absent; a directory or an unwritable path at `/home/bts/.check-pick-entered.lock` makes every run exit without checking (stated limit; the morning `bts day-outcome` then reports ENTRY UNVERIFIABLE). A named pipe at a capture path (F1's limit) now blocks one run for at most 14.5 minutes instead of for ever, and the slots it covers are not checked.
- **Midnight ET and the year boundary:** a run started at 23:45 and ended by the limit at 23:59 changes nothing for the next day's first run.
- **A failed or unconfirmed send; a restart mid-step:** a box restart mid-run releases the lock with the process; the next slot runs normally.

## Known and not fixed (added to F1's limits)
1. **The lock is on the cron line, not in the command.** A run started by hand (or by any second cron line) while cron's run is alive can still overwrite a newer marker with its older picture and lose a wrong-game observation (K4). Do not run `bts check-pick-entered` by hand on the box while the entry window is open.
2. The lock and the limit exist on the box only after `bash scripts/cron-setup-hetzner.sh install`; a deploy never touches crontab.
3. A slot skipped because the previous run is still alive leaves no receipt and no log line.
4. `/usr/bin/flock` and `/usr/bin/timeout` are the Debian paths (util-linux and coreutils); the `check-results` line already depends on the first. Their behaviour is tested here only where the test machine has them.

## The author's work (tests first; ONE commit on `fix/wr-f1`)
`scripts/cron-setup-hetzner.sh` (the one line; one header-comment sentence), a new `tests/test_wr_f1b_lock.py` (K2, K3, K4, the installer rows), and Amendment 5's text in `kickoff-fix-f1.md`. No `src/` file. If an existing assertion must change, STOP and ask.

## Review
Gate on the committed tree; a fresh Codex xhigh review and a Carbon review of F1 and F1b together against `main`; at most two rounds; merge on a Codex SIGN. If it stops, F1 goes to Eric (R67).
