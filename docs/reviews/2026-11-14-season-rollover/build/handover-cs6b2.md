# Change set 6b-2 handover

Commit `38d707bd0b2214a22f407abe0d7c6fc902e7d59f` on `sb-cs6b2`, based on `59efa8258203ae561d43a8319c4a501b1e4b7ee2`.

## Contract rows

| Row | Implementation and evidence |
|---|---|
| 1 | `src/bts/health/alert.py:169` checks the resolved file type and reads a single JSON object snapshot. The dispatcher calls it at line 221. `tests/health/test_alert_status_cs6b2.py:71` covers the three status lanes and the requested call modes. The FIFO probe fails on a blocked read or write. The test at line 265 counts file opens. |
| 2 | `src/bts/health/alert.py:182` preserves the exact banner for a prior failure with a positive exact integer count. The test at `tests/health/test_alert_status_cs6b2.py:147` checks the banner with a current alert and on its own. |
| 3 | `src/bts/health/alert.py:234` uses only strings from a same-ET-day sent snapshot. Line 267 writes the sorted union through the original writer. The tests at `tests/health/test_alert_status_cs6b2.py:170` and line 188 cover mixed, malformed, and stale keys, repair after one corruption, persistent write failure, and day rollover. |
| 4 | `src/bts/health/alert.py:51` is the existing writer. The dispatcher's persistence calls pass through the guard at line 96, which logs and skips persistence for special objects, including a FIFO after a successful DM. `tests/health/test_alert_status_cs6b2.py:221` archives the base source and compares disposition, DM bodies, booleans, and written bytes with a pinned clock across 36 prior-state and call combinations. |
| 5 | The production diff is limited to `src/bts/health/alert.py`. `tests/test_alert_status_callers_cs6b2.py:30` compares corrupt and missing status files through the real scheduler and runner. It checks both fallback handler sites, state-read and saved-pick refusals, missed-pick, late and void refusals, direct and classified unconfirmed branches, and the EOD CRITICAL. |

## Existing test change

`tests/test_saved_pick_paths_cs5a.py:442` now expects one health DM, a persisted `sent` status, and the absence of the failed-alert log. It still asserts refusal, unchanged pick bytes, an unlocked pick, no decision record, and no pick DM. This was the only existing test changed.

## Test evidence

Wrapper logs and exit records are in `.codex-review/cs6b2/`.

- Base RED dispatcher run. 130 failed and 171 passed. RED included non-object and deeply nested JSON, FIFO, malformed counts and keys, recovery from a corrupted file, and the migrated saved-pick test. GREEN controls included invalid syntax, invalid UTF-8, injected denied reads, directories, regular symlinks, missing files, and already contained count and dedup cases.
- Base caller, parity, and snapshot run. 14 failed, 43 passed, and 300 deselected. The 36 archived-source parity cases passed. Caller failures included three fixture problems in the missing-file controls. The final fallback initially returned no predictions during the refresh, and the late and void assertions used the parent of the actual archive directory. The corrected fixture drives the final fallback with a prediction frame and checks the actual archive paths. The single-snapshot test observed two opens on the base.
- Focused GREEN run. 358 passed. The classified-unconfirmed controls were added afterward and passed in the combined suite.
- First combined run. 546 passed, 4 skipped, and 5 failed. The new scheduler fixture loaded policy from its temporary working directory and left the resulting cache for later tests. The fixture now restores the cache after each case.
- Combined retry with local loopback and an outbound Python socket guard. 551 passed and 4 skipped in one process, including all 12 web-saver tests.
- Native 2027-07-15 and 2028-07-15 ET clocks, applied to the Python processes while leaving monotonic time alone. Each run covered all three added or changed test files and finished with 454 passed and 4 skipped.
- The final fast gate ran alone against commit `38d707bd` with the plain wrapper command and no extra plugins or environment additions. Pytest exited 1 with 9,139 passed, 37 failed, 16 errors, 23 skipped, 6 deselected, and 1 xfailed in 25m 40s. All 53 failing or errored test IDs matched `~/bts-runs/gate-failed.txt` exactly. The base fast suite was not rerun.

## Known limits

- A single corruption can repeat a covered alert once per key in a serialized, writable lane. A successful write restores coverage for the keys in that send. Persistent corruption or failed persistence can repeat alerts on every eligible call, with no daily cap. A send failure or skipped-recipient write can also reset coverage.
- An unreadable prior status provides no failure banner. Path type checks assume a stable path during the subsequent file operation. Resource exhaustion, stalled filesystems, invalid caller input, logging failures, unrelated transport failures, and interrupts are outside the status hardening.
- The dashboard retains its own reader and renderer. For example, a status value of `[]` can still break rendering.
- A malformed `decision.json` can still stop scheduler checks under ruling R37. A malformed pick file can also still stop checks.
