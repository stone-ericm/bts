# Verdict

**BLOCK** for F5 `5a80f0cc` against `a4e8278d`.

## Blocking findings

**B1. An early confirmation can silence all the original checks after a correction to the saved pitch.** A 09:40 commitment names batter 1010 in game 101 at 10:05 ET. At 09:45 the account shows the batter, no units capture binds the game, and F5 records `unit_unverified` in the receipt and terminal `confirmed` in the marker. Before 10:00, save the same pick with a corrected 10:50 pitch and keep its decision. The 10:00 capture binds the account entry to game 102. With the generated base line, 10:00, 10:15 and 10:30 all read the account and warn, and the next morning pages `ENTRY UNVERIFIABLE`. F5 returns `already_confirmed` at all three and the morning exits clean as `entered_game_unverified`. A replacement commit reproduced the live loss too. With a missing entry and a revised 10:06 pitch, F5 read at 10:00 but the early marker had already spent the alert tiers, so it lost the base's only DM for the revised window.

The gate at `src/bts/cli.py:1995` sees the early snapshot. The marker read at `:2018–2033` reuses confirmation and tiers by date alone. Mark state produced by an extra early cron run and bypass that state, including its tiers, when an original-hours cron start is inside the current window. Have `day-outcome` ignore extra early observations when the final saved selection has an original slot, since `src/bts/day_outcome.py:57` binds by decision digest and the digest can survive a timing correction. Keep manual marker handling and stable early-window retries. `alert_probe.py` ran both generated lines with the archived and current CLIs and captured the DMs and receipts. The probe models a saved-file correction or replacement. The ordinary scheduler preserves its committed pick, so the MLB schedule changing on its own is insufficient.

**B2. An extra launch can erase the previous date's pending entry-health alert.** Leave March 23's `dm_failed` marker in place. With a March 24 commitment already on disk at 04:55 for a 06:10 pitch, F5's 05:00 launch confirms the new entry and replaces the single shared marker at `src/bts/cli.py:2035–2041`. If March 23 EOD health runs at 05:01, `src/bts/health/pick_entry.py:66–72` now discards the marker on date mismatch. The base has no 05:00 watchdog and sends the prior-date CRITICAL. F5 sends none. `alert_probe.py` ran the real health runner with the other sources isolated and captured dispatch under both trees. This needs a next-date commitment already present while earlier-date health is outstanding, as in a restore or a separate historical run. A single scheduler normally runs EOD before it commits the next pick. Preserve the displaced date's marker on the first early overwrite and have the health source read that date's copy when the current marker is for another day.

## Edits

- `scripts/cron-setup-hetzner.sh:17–19` and `ARCHITECTURE.md:355`. State the cadence and the gate separately. “The entry watchdog launches every 15 minutes in ET under `entry_intent = "enter"`. Before 10:00 ET it defers when a nominal 10:00–23:45 ET quarter-hour on the same date falls inside the saved selection’s entry window.” Keep the architecture bullet’s historical re-enable date attached to the re-enable event and the header’s wrapper details after the new sentence.
- `docs/ops/pick-entry-receipt-v1.md:9`. Replace “One file per run.” and the sentence after it with “Each watchdog CLI invocation attempts to publish one receipt JSON at `data/health_state/pick_entry_receipts/<ET date>/<started>-<attempt_id>.json` and its sibling `.json.sealed` file.” In the `base_slot_available` row add “nominal” before “10:00–23:45 ET quarter-hour”.
- `src/bts/cli.py:1890` help. “Before 10:00 ET, defer if a nominal 10:00–23:45 ET quarter-hour is inside today’s entry window.” At `:2002` use `check-pick-entered: early check deferred because a nominal 10:00–23:45 ET quarter-hour is inside this pick's entry window`. The `10:05:01` case deferred at 09:45 and a CLI start at 10:00:02 was already outside the window.

## Notes

- The documented wall-time arithmetic, morning receipt denominator, and inherited batter-only confirmation and mid-send retry rules were visible in the replays. R63 keeps the checklist and incident text for January.
- Pytest's opt-in K4 diagnostic failed identically on the base and F5. Unwrapped invocations can overwrite a newer observation and give an all-clear. The wrapped contention checks passed with shims.
- The sandbox declined the `nice` priority change. The bounded commands ran to completion. All review files are under `.codex-review/wr-f5-r1/`.

## 1. Scope and ordinary CLI parity

`git diff a4e8278d 5a80f0cc` included review records. The code commit itself touched the installer, CLI, two current texts, two existing schedule assertions and the new F5 test file. `installer_probe.py` compared real shell output and installed bytes in ten intent/crontab states, including both enter and research. Only the watchdog hours and flag differed. `manual_parity.py` printed the distinct `bts.__file__` paths and matched 201 invocations in 199 sequences after normalizing random receipt identifiers, temp paths and fixture revision. Compared stdout, stderr, exits, exceptions, captured transports, markers and receipts.

## 2. Early gate

`window_probe.py` ran 418,848 checks of the real branch using the installed slots. The exhaustive minute grid used singles and separated double legs in both orders. For a fixed selection, every eligible base slot remained and added reads occurred only for the 00:06–10:05 earliest-pitch range at the default width. Real CLI checks covered exactly 10:00, the strict cutoff, exactly 75 minutes and the preceding second. Tested widths from −1 to 1,000 minutes in the branch and 39/40 in the CLI. The generated line uses the default. A first commit at 09:52 was picked up at 10:00. Confirmed and alerted markers stopped or continued verification as applicable. A corrected or replaced commitment exposed B1.

## 3. Closing test and M1–M7, N1–N8

`installer_probe.py`, `alert_probe.py`, `clock_standing.py` and the F5 tests covered the table using the actual proposed/installed lines and CLI. The closing fixture read the account at 05:15 and made morning `entered` (base no read and `ENTRY UNVERIFIABLE`). With entry missing, 05:15–06:00 and 09:00–09:45 performed the M2/M3 reads. Exact-cutoff 10:00 was local for 10:05. M5/M6 returned `base_slot_available` at 09:45 and kept all three base warnings and the morning alert. The 19:10 slate retained five eligible slots, with four local receipts after an initial confirmation. Intent, replacement, foreign/legacy line preservation, refusal and removal agreed with the base. See item 5 for the platform limits on N8.

## 4. Traffic and receipts

`traffic_probe.py` replayed all 40 added slots across no pick, yesterday-only data, preview, outside-window, shared-window, confirmed, alerted, missing, mismatch, identity mismatch and login failure states. All local outcomes had zero transport, DM attempts and marker writes. A fully eligible missing/mismatch early pick had five account-read sequences and three DMs. An already-alerted one had two DMs. Wrong identity and failed login stopped before profile. The line can do `uv` setup first.

An ordinary no-pick day sealed 96 JSON/seal pairs versus 56. Lock skips and hard stops can leave no discoverable receipt. `day-outcome` counted extra no-attempt runs in its denominator and required an observed, bound receipt for account evidence. The ops backup runner captured the entire `data/health_state` path, inclusive of both files, and left them in place locally. `scripts/audit/` search and contest/reconcile reader tests found no consumer of this outcome. Pick-entry health exposed B2.

## 5. F1b wrapper and clock changes

The real generated wrapper retained `flock -n` and TERM at 840 seconds, then KILL after 30 more if the child remained alive. `clock_standing.py` applied the documented Debian wildcard rule to the ET timeline and replayed the resulting clock labels through the CLI with no pick. Spring omitted 02:xx (92 launches), fall repeated 01:xx (100), and all adjacent projected instants were 900 seconds apart. Replaying each duplicate label published separate receipts. The naive CLI pin uses the first fold for both occurrences. Synthetic spring and fall windows reproduced the documented one-hour wall-time error. F1b's shortened timeout, contention and interruption shims passed. Three native Linux-tool cases skipped on this Mac, which lacks `/usr/bin/flock` and GNU `/usr/bin/timeout`. The Debian daemon was not exercised.

## 6. Operator text

Read the generated cron header, architecture bullet and receipt guide. Captured the actual CLI help and the early deferral outcome. The ambiguities are covered in Edits under R64. None of the proposed text edits alone addresses either code finding.

## 7. Standing rows

`installer_interrupt.py` stopped the real installer after a crontab shim staged the full pipe, before publication. Both versions retained the old bytes and a retry installed the staged rows. F1b tests killed wrapped children at marker and receipt stages. `clock_standing.py` covered a malformed pick, a directory and a bounded FIFO read. With and without the flag these produced the same local exceptions and no transport or marker write. At 00:00–00:45 on Jan 1 the CLI ignored Dec 31's pick and read only Jan 1 if present. An interrupted send left no marker. An accepted send with a lost response produced `dm_failed` and a new invocation retried it. Known duplicates and receipt interruption behavior held.

## 8. Tests and scratch changes

All 25 matching test files including F5 ran in four sequential batches. `pytest-1.log` through `pytest-4.log` recorded 1,034 passed and eight skipped. The two existing one-line assertions pin the requested hours. Five extra backup/ledger files passed 95 tests. The separately enabled K4 diagnostic reproduced the base failure noted above.

The 18 F5 cases on each scratch copy caught a missing gate, application without `--cron`, a missing installer flag, `<= 10`, and the inclusive cutoff. Two scratch copies passed. Ending the search at 23:30 is equivalent here because if a before-10:00 run is already inside a continuous window that reaches 23:30, 10:00 is also inside. Starting at 10:15 is a testing gap. `mutation_witness.py` ran a 10:20 pitch with a 09:45 batter match and binding at 10:00. The real candidate deferred then warned. The passing scratch copy confirmed early and skipped the only base check. Add that case and B1/B2 to the tests.

## 9. Other diff checks

`bash -n` and `git diff --check` passed on the code commit. The wider diff has whitespace in archived review CSV/log files. No tracked file was edited during the review.

ran: real installers and pinned CLIs, paired alert and reader probes, minute and clock grids, standing cases, matching tests, extra reader tests and seven scratch changes; did not run: live Debian cron or native Linux wrappers, live MLB/Bluesky/restic/R2, or the full suite.
