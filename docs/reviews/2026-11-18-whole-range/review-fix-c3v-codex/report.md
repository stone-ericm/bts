# Verdict

**BLOCK** on `cb3bc37c` against `88b69a16`. Every compared replay return, exception and post-reconcile streak matched the base. The new diagnostic violates V4 for a filename that the inherited scan accepts.

## Blocking findings

- **B1. A trailing newline in a pick filename splits the required single stderr line.** With ET pinned to `2027-04-20 02:00`, seed `streak.json` with `{"streak":9,"season":2027}` and put `{}` in the file named `2027-04-16\n.json` (one literal LF before `.json`). Run `bts reconcile --lookback 8`. The inherited regex uses `$`, which accepts the stem before its final LF, and the new diagnostic interpolates the raw stem. [Replay scan](../../src/bts/picks.py#L845), [stderr formatting](../../src/bts/cli.py#L2958)

  ```text
  Season replay unavailable for 2027-04-16
   (no final result on the pick); streak retained, not recalculated.
  ```

  Exit was 0 and stdout ended in `Streak retained: 9`. The receipt's `blocked_date` decodes to `2027-04-16\n`. B8b's ordinary `<date>.json` path cannot identify this file by the displayed date. Escape LF as a literal `\n` at the stderr formatting site, preserving the raw receipt value and the replay scan. `hostile_probe.py` reproduced the defect on archived HEAD. `output_probe.py` reproduced it with the default lookback and verified the scratch fix against both the LF and ordinary filenames, including raw receipt and streak byte comparisons.

## Notes

- **B8b overpromises the result of a move.** With pending files on April 16 and 17, the documented move of the confirmed never-entered April 16 file leaves the next reconcile blocked on April 17. Say to rerun and expect a rebuild once the last blocker is cleared. The entry prohibition applies independently to each date. [B8b](../../docs/ops/2027-season-start.md#L32)
- **The existing CS2 test changed fixture state.** The original receipt was `not_reached` in both branches despite the test's comment. The new stub records `unavailable`. Keep both original branches and put the blocked branches in the new tests, or explicitly amend the fixture change. [Edited test](../../tests/test_streak_season_cs2.py#L385)
- The receipt schema reference describes `unavailable` but omits its two new fields. The pre-existing `outcome=completed` with a corrupt streak and a blocked replay also reproduced, with the subsequent CLI summary read exiting 1. [Receipt schema](../../docs/ops/reconcile-receipt-v1.md#L32)

## 1. Replay equivalence

Separate processes imported full archived `src/` trees and asserted the source paths of `picks`, `cli` and `reconcile_receipt`. The original 630 histories produced 462 saves, 133 refusals and 35 exceptions on each tree. The four round-one delivery cases, five prior writer sequences, eight Carbon cases and Carbon b2 all returned `None` and preserved the full streak bytes. The 17 correction cases saved their pick correction while retaining the streak.

Another 233 cases matched across the two trees on return or exception, CLI exit and raw streak bytes or absence. Inputs covered JSON types including arrays, null and numbers, malformed JSON and UTF-8, missing and non-regular paths, dangling links, permissions, nested decision records, today and past, and marker lists, nonlists, mixed entries and foreign or missing season tags. The bounded FIFO probe reached its deadline on both. A real contended scoring lock crossed ET midnight and New Year, with the post-lock dates April 19 and January 1 reported respectively. Corrupt and missing streak files behaved identically in saved and blocked cases.

The author's oracle extracts the three original modules with `git archive 88b69a16`. The fixture imports them under aliases and binds the archived CLI to archived `picks` and receipt code. The independent full-tree run bypassed those bindings. Both harnesses pin the `datetime.now(timezone.utc)` call that writes `updated` and compare full bytes. Receipt comparisons also pin the clock, run ID and producer revision. [Author oracle](../../tests/test_wr_c3v_reconcile.py#L77), [Streak writer](../../src/bts/picks.py#L734)

## 2. CLI output and receipts

With a readable streak and no corrections, all four ordinary cases exited 0 with `No scoring changes detected. Streak retained: N` after the opening line, and one stderr line in the form `Season replay unavailable for <date> (<reason>); streak retained, not recalculated.` The receipt's `replay` was `{"state":"unavailable","blocked_date":"<date>","reason":"<reason>"}` with these exact reasons.

| Fixture | Reason |
| --- | --- |
| Torn JSON, unreadable file, directory or dangling pick link | `pick file unreadable or not valid JSON` |
| Pending today with today's marker, even on a foreign-season streak tag | `today's result is applied to streak.json but not yet saved on the pick` |
| Past pick with a nonterminal result | `no final result on the pick` |
| A hit file disappears between the initial read and `load_pick` | `hit pick could not be loaded` |

Mixed blockers, two pending files and a later exception confirmed first-in-walk order. Corrections printed before `Streak retained: 10`. Saved runs matched their base stdout, stderr and receipt bytes, including `Streak recalculated: 0` after a correction. A malformed yesterday file with lookback 1 raised before replay, and a `KeyboardInterrupt` during replay also exited 1 with base `not_reached` receipt bytes. An injected receipt-write `OSError` left exit 0 and no receipt for saved and blocked runs. The blocked run still printed the replay warning in addition to the existing receipt error. The only in-tree reader is `discover`, which returned the same sealed record with the new fields. Reconcile makes scoring reads and sends no picks.

## 3. Operator step

Ran B8b's `mv` with synthetic `data/picks`. Two reruns before the move reproduced the warning and bytes. After moving the sole blocker, reconcile saved and the archive retained the exact bytes. With two blockers it saved only after the second move. The checklist requires confirmation in MLB before moving and explicitly forbids moving an entered day's pick. Its other three reasons explicitly keep the file in place.

Scanned `src/` for nested readers, globs, traversal and direct path construction. Reconcile, pick readers and export/verification use top-level files. Under dates, `daily_decision` and `skip_policy_shadow` select `decision.json`. `scheduler`, `web`, `lineup_analyze`, `analytics_artifacts_missing` and `late_delivery` select `scheduler_state.json`. `scheduler_state_integrity` lists only its quarantines and `fallback_defer` opens only `deferred_fallback_*.json`. The restore drill selects four fixed filenames and Restic can copy the directory as opaque backup data. The exercised readers performed zero opens of the archive path.

## 4. Existing test amendment

`git diff 88b69a16 HEAD --stat -- tests` shows four new files and exactly one edited existing file, `test_streak_season_cs2.py`. Exit and exact ET-year assertions were preserved and the C3v comment was added. Before, the assertions were `Streak: 0` or `Streak recalculated: 0`. After, they are `No scoring changes detected. Streak retained: 0` or `Streak retained: 0`. `output_probe.py` ran the old stub on both trees and matched both original summary branches and `not_reached` receipts. The amendment permits a wording change for an unavailable replay. The fixture change in this test goes beyond that wording change.

## 5. New tests and mutations

On the archived base, 26 failed and 633 passed. The 26 failures were the nine delivery/writer cases, eight reconcile cases and nine Carbon cases, all on the absent stderr warning. All 630 corpus histories passed on the base. HEAD passed all 968 focused tests in one process at the real clock and again under verified ET wall clocks in November 2027 and November 2028. These runs included the four new files, `test_streak_replay.py`, `test_streak_season_cs2.py`, `test_reconcile_receipt.py`, `test_lightgbm_optional.py` and `test_web_saver.py`.

| Scratch mutation | Failed | Passed |
| --- | ---: | ---: |
| Drop stderr warning | 26 | 633 |
| Report last blocker | 3 | 656 |
| Print recalculated for blocked corrections | 17 | 642 |
| Skip an unresolved past pick | 112 | 547 |
| Check the generic unresolved condition before today's exceptions | 16 | 643 |

All five mutants were caught. The first focused run had one sandbox denial for the existing loopback HTTP test. The loopback-enabled rerun and both wall-clock runs passed all 968. The mutation harness's nested `pgrep` was sandbox-denied, so I reran each mutant with a separate successful idle check before launching its wrapper. The `verified-*.log` files are those reruns.

## 6. Rest of the diff

`git diff --check` passed. I found no other production behavior change in correction reads and writes, the replay save, scheduling, health or scoring. B1 is in the new rendering independently of the inherited scan. The test amendment and the operational wording are recorded above.

## What I ran

- `git archive 88b69a16 src` and `git archive HEAD src tests pyproject.toml`, extracted under `base/` and `current/`. Compared the source, checklist and test diffs, read the C3 ruling and checked the original producer fixtures.
- `corpus_probe.py`, `writers_probe.py`, `hostile_probe.py`, `operator_probe.py` and `output_probe.py`, with raw archived-tree evidence in their `-base.json` and `-current.json` files.
- Serialized `yield_run.sh` pytest runs with `TZ=America/New_York`, `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`. Logs are `focused.log`, `focused-loopback.log`, `base-red.log`, `wall-2027.log`, `wall-2028.log` and `verified-*.log`. The wall-clock shim shifts realtime only, and the runner verified `date.today()`, `datetime.now()` and `time.time()` before collection.
- `git diff --exit-code` and `git status --short` confirmed the tracked checkout remained clean. Scope was the focused suite and synthetic probes, with no external network.
