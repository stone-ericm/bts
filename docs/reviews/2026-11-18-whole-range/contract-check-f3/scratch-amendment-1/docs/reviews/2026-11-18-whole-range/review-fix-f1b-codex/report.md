# Verdict

**SIGN WITH EDITS** for F1 and F1b at `709e783f` against `1e69d0f0`. The installed watchdog closes both reported ways to lose a marker observation during completed checks. The findings below concern the operator text and test evidence. The GNU/util-linux runtime cases remain to be verified on Ubuntu.

## Blocking findings

None. With a stable shared lock, I found no new ordinary sequence that lost a completed wrong-game observation, falsely cleared one, or suppressed an alert beyond the stated interruption limits.

## Edits

- **Cron header.** At `scripts/cron-setup-hetzner.sh:17–19`, replace the watchdog description with “Every 15 minutes from 10:00 through 23:45 ET, the entry watchdog can DM about a missing, mismatching or game-unverified pick before the submission cutoff. This line is installed only for `entry_intent = "enter"`. A slot skips if another invocation holds the same lock. After acquisition, timeout sends TERM at 14 minutes and sends KILL 30 seconds later if its monitored child is still running.” The current summary only describes absence and leaves the scope of serialization implicit. [Installed line](installed-cron.txt) · [Overlap check](installed-overlap.log)
- **Receipt operations guide.** In `docs/ops/pick-entry-receipt-v1.md:6,9,24`, describe guarded hooks and independent publication explicitly. Use “The receipt hooks capture the inputs and timing. On a batter match the watchdog can run game qualification before making its decision. Publication independently reads the captures and records the resulting evidence.” For the file rule use “The watchdog attempts to seal one receipt per CLI invocation at `data/health_state/pick_entry_receipts/<ET date>/<started>-<attempt_id>.json`. The enter schedule supplies 56 slots per day. A slot skipped on lock contention produces no receipt. A hard stop before sealing can leave the run without a discoverable receipt.” The new sentence near the schema is accurate. [Interruption check](installed-interrupts.log)
- **Amendment 5, A6.5.** Replace the sentence about scheduled scoreable days with “An enter day with a scoreable single or double and at least one scheduled regular-season game remaining after postponements and cancellations reports `entry_unverifiable` when the inputs can be read and no receipt qualifies.” The real command with the same saved decision and no receipts returned a clean `no_games` when the only scheduled game was postponed. [Morning check](standing.log) · [Classification](../../src/bts/day_outcome.py#L114)
- **Nominal interval.** In F1b's adopted design item 1, add “A forward adjustment to the system wall clock can also shorten the elapsed time between cron launches. A slot reached while the prior run is still in its grace period can skip.” The stated 30-second margin assumes the 900-second spacing. A 45-second forward adjustment gives a modeled 855-second interval, inside the 870-second cap with grace. [Clock check](standing.log)

## Notes

- The first 35-second watchdog test run timed out. The full rerun took 58 seconds. The first real-uv probe exited before starting the CLI in the sandbox. Its rerun used a writable fixture cache and reached Python. The initial parity script compared random marker IDs when receipts failed to seal. The corrected comparison normalizes those references only when a sealed ID is unavailable. I kept the original logs. [First test attempt](tests-lock.log) · [First uv attempt](installed-limits.log) · [First grid](grid.log)
- The shell test runner's cleanup signals all recorded PIDs even after their wrappers have returned. It can reject stale PID records before signalling and give the wrapper, timer and child separate start records. This is a test-harness note. [Cleanup](../../tests/test_wr_f1b_lock.py#L364)
- `nice` reported that the sandbox denied the priority change. The probes and test groups stayed serial and bounded. No tracked file changed.

## 1. Scope and base comparison

`git diff 1e69d0f0 717ac97f -- src scripts docs/ops` showed the watchdog, shared slot and qualification helpers, the one receipt sentence, and the cron line and header. F1b changed no source or operator DM template from `15b27474`.

The two grid processes printed `bts.__file__` for the base from `git archive 1e69d0f0 src` and the checkout source. Compared stdout, stderr, attempted and sent DMs, fetches, marker fields and bytes, exit, sealed receipts, health and morning outcome across 1,920 clean-history pairs. 1,896 matched after receipt ID normalization. The other 24 were unreadable-capture matches covered by Amendment 3. [Grid and source paths](grid-normalized.log) · [Fixture](grid.py)

Generated `show` and `install` output from both real installers with the same fixture crontab and both intents. Research output matched byte for byte. Enter matched after removing the exact wrapper from its watchdog line. Reinstall replaced marked active and commented entries and retained unmarked jobs. Remove retained those lines as well. The requested refusal, read and filter error cases passed. [Installer replay](installed-overlap.log) · [Installer tests](tests-lock-full.log)

## 2. K1–K5

- **K1.** The installed line preserved the observation through each of a crosswalk gap, removal and batter mismatch after a failed wrong-game DM. A conflicting second capture then left the marker unresolved and sent the still-due first warning. Full proof confirmed. The six M11/M12 cases and all 144 pairs and 1,728 triples also passed. [Installed serial checks](trace-serial.json) · [F1 tests](tests-f1.log)
- **K2.** Held the installed job in its real CLI fetch. The contender exited 1 in 0.44 seconds, reached neither the timer nor watchdog and appended zero bytes. A successor acquired the lock after release and observed the wrong game. [Installed overlap check](installed-overlap.log)
- **K3.** The generated schedule gives `840 + 30 = 870 < 900`. Shortened BSD-tool runs returned 124 twice in succession and delivered TERM to Python behind real uv. A TERM-resistant controller and Python child received TERM and reached exit 137, then the lock could be acquired. The shim tests checked the TERM/KILL diagnostics in the cron redirect. [Limit runs](installed-limits-retry.log) · [PID and signal records](trace-limits.json)
- **K4.** The opt-in diagnostic with its original assertion failed as expected on a terminal all-clear. I also let an older unwrapped job finish across an install. It erased the newer observation and a later conflicting capture yielded the same all-clear. [Opt-in diagnostic](k4.log) · [Generated legacy-line replay](trace-overlap.json)
- **K5.** The F1 files passed all 83 cases for the N and M rows, messages, receipts and standing behavior. The new type cases covered `True`, `False`, `"101"` and `101.0`. [F1 tests](tests-f1.log) · [Type and lock tests](tests-lock-full.log)

## 3. The two-writer class and interruption orders

Used the installed shell command with a test lock path, `fcntl.flock`, shortened BSD timeout and captured CLI transports. Checked timeout before fetch, during fetch, during send, after acceptance before marker write, during the marker temp write, during the receipt temp write, after receipt rename and after the seal. Before marker replacement the prior complete file survived. An unrecorded observation again allowed the next unknown binding to confirm. After replacement the unresolved field survived, including with an unsealed receipt. Only the sealed receipt counted. [Cuts and successor checks](trace-interrupts.json)

A streamed offline pending response simulated 28.5 seconds between chunks. Both commands sent a missing-entry alert at 855 elapsed seconds, after the new cap and before the 900-second cutoff. At 940.5 seconds both skipped the absence alert. F1 also declined a wrong-game nag after cutoff and retained the observation. The installed short-limit run could end before the response completed, and its next nominal slot was outside the window. These are the documented limit cases. [Candidate response](trickle.log) · [Base response](base-trickle.log) · [Installed cutoff](installed-limits-retry.log)

An escaped child kept the advisory lock after the monitored wrapper returned 124. The next contender skipped, and killing that recorded child freed the lock. A stale regular lock file was reusable. The fixture opened a directory and a read-only file, and a missing parent failed with a diagnostic before the timer. The actual unmarked legacy line survived install as the contract says. [PID trace](trace-limits.json) · [Installer overlap](installed-overlap.log)

## 4. Linux tools and the test model

Both workflows use `ubuntu-latest`, install uv 0.11.6, and collect these cases via `uv run pytest -m "not slow"` when the deploy gate runs. The resolved `/usr/bin/flock` and `/usr/bin/timeout` paths and `GNU coreutils` version text are expected there. TERM uses an absolute virtualenv Python under real uv in offline/no-project mode. The 2–3 second startup bounds and 0.3-second grace look feasible for the gated fixtures. `_require_real_tools` skips silently if either path is missing or the version text differs, and checks existence alone for flock. CI can therefore appear green with only the shims. [Detection](../../tests/test_wr_f1b_lock.py#L320) · [Test workflow](../../.github/workflows/test.yml) · [Deploy workflow](../../.github/workflows/deploy.yml)

The shims exercise a real advisory lock, exit 1 for contention, process-group TERM/KILL, statuses 124/137, and the cron log redirect. The flock shim opens read/write and execs the timer. util-linux can open read-only paths and directories, and normally retains a separate waiting parent. The timer shim uses a new child session and synthesizes status and log text from outside that group. GNU timeout can signal its own group and be killed with it. BSD delivered TERM to the real uv child here. GNU signal propagation, diagnostics and release timing are still estimates until those gate cases run. [Shim code](../../tests/test_wr_f1b_lock.py#L141) · [Local trace](trace-limits.json)

## 5. Operator texts and known limits

The first warning, reminders, locked-unconfirmed warning and cutoff line passed the F1 assertions. Rollback from real `alerted`, `dm_failed`, `present_unverified` and `confirmed` markers matched A6.4's all-clear conditions. A later timely unbound receipt yielded a clean morning result while an earlier wrong-game observation stayed unresolved, as A6.5 describes. The installed and unwrapped overlap replays support A5.4's activation and manual-run instructions. The wording edits above cover the header, receipt publication, all-void schedule and nominal clock spacing. [Rollback](rollback.log) · [Morning](standing.log) · [Legacy-line check](trace-overlap.json)

## 6. Standing rows

Raising captures gave the explicit batter-only confirmation. A silently accepted malformed capture kept an already-recorded observation unresolved. A hand-edited array game ID exercised the caught qualification `TypeError` and the same batter-only result. Invalid marker shapes or encoding, and a directory at the marker destination, exercised the inherited exception paths. The F1 test file covered FIFO captures. `KeyboardInterrupt` and `SystemExit(7)` preserved the prior marker. After ambiguous DM acceptance the first warning retried, and final `dm_failed` produced CRITICAL. [Standing checks](standing.log) · [FIFO tests](tests-f1.log)

Pinned the wrapped last slot at 23:45 ET on December 31, interrupted it, and started the next day's 10:15 slot against the new season. The old marker supplied no tiers to the new date. Both DST dates had 56 separate slots with 900-second gaps at the installed hours. Process restarts after timeout reacquired the lock. Reboot release, missed-slot handling and power-loss durability were checked by inspection against the contract and atomic write sequence. [Year and restart](installed-limits-retry.log) · [DST](standing.log)

## 7. Tests and scratch changes

Ran all 65 grep-matched test files, plus the separately named game-ID file, in serial groups. The runnable groups reported **3,571 passed, 8 skipped and 1 xfailed**. The audit-driver batch reported 33 failures and 16 setup errors due to the missing GNU rsync 3.x prerequisite. The archived base produced the same result. The opt-in K4 failure is separate. Against `1e69d0f0`, tests contain four additions and zero modifications. [Exact file groups](test-files.txt) · [Audit base](base-audit.log)

One scratch change at a time, the generated-line assertion rejected dropping `-n`, dropping the timer, raising the limit to 900, dropping `--verbose`, and swapping wrapper order. The F1 checks rejected removing `game_unresolved` from `alerted` (six failures), `dm_failed` (three) and `present_unverified` (two). All eight were caught on this Mac. None required Linux to detect the change. [Scratch driver](mutation.py) · [Example output](mutation-present_unverified.log)

## 8. Other diff review

Checked the marker writers, cron invocations, receipt consumers and shared selection helper. The marker update remains one atomic replacement, and the receipt keeps independent qualification and the batter-only verifier. The code and shell diff passed `git diff --check` and the installer passed `bash -n`. [Installed artifact](installed-cron.txt) · [Base grid](grid-normalized.log)

ran: real CLI on base and candidate, installer parity and installed-line replay, shim and BSD timeout including real uv, K1–K5, standing and morning checks, all 66 requested test files, eight scratch changes; did not run: GNU timeout or util-linux flock, either Ubuntu workflow, live cron or a reboot, GNU-rsync-dependent audit assertions, network calls or the whole fast suite.
