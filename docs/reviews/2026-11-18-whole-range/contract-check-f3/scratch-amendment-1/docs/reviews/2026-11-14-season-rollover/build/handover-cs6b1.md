# CS6b-1 morning state check

Commit `0e0c49a30a97c6961de4612f522834dc7cf82981` on `sb-cs6b1`, parent `1bf518a2`. One local commit, no push. The scheduler uses the ten-minute margin in version 2 of the kickoff.

The baseline fast suite for this base is already known (unsandboxed: exactly the 53 IDs in ~/bts-runs/gate-failed.txt; inside your sandbox four socket tests in tests/test_sd_notify.py and tests/test_web_saver.py also fail). No baseline fast suite was run.

## Contract and evidence

Test locations below are in `tests/test_morning_state_cs6b1.py`.

| Row | Implementation in `src/bts/scheduler.py` | Tests |
|---|---|---|
| 1 | L29, L115–151, L2931–2945. The call follows the existing state save. It samples the clock once for the advisory and compares it to the earliest planned target. The helper checks the decision independently of the pick by passing `None` to `is_scoreable_commit`, which calls `load_decision` internally. Advisory print failures and the outer advisory failure report are guarded. | Margin and gate cases L179–302, unreadable records L459, startup faults and stderr L361–430. |
| 2 | L153–172. One required loader call, separate CRITICAL status and incident key, stock envelope and retry banner. | Refusal and live alert ordering L74, flag and recipient combinations L234, restarts L305, isolation L361 and L433. |
| 3 | L174, L180–201. The morning WARN uses the existing 4c status and key. Its detail formatting follows 4c inline to keep the delivery function untouched. | Sent, failed and aging snapshots L127, stale byte comparison L478. |
| 4 | L174–179. Accepted state writes one journal line. | Dated and fresh automatic cases L112. |
| 5 | The base-to-commit diff adds three hunks to the production tree, all in the scheduler (constant, helpers and call site). The named existing functions, EOD invocation, account-state code, and dispatchers have no diff. | Live refusal L74, rereads after a dated correction or real captured fetch L328, shared status observed at the first prediction L433, day artifacts L478 and existing scheduler/delivery suites. |

## RED and GREEN controls

- `red.log` was run before the implementation. The replay against the exact base scheduler with the corrected fetch and commitment fixtures is in `red-final.log`. Both produced 41 expected failures and 20 passes. RED cases cover the startup refusal, accepted messages, early stale WARN and retry, margin boundary and skip lines, requiring flags and outward/recipient combinations, restart dedup and banner, two recovery paths, 12 read/dispatcher/transport/gate faults, both stderr faults, shared-file sent/failed cases, and unreadable decision/pick files. Fault tests verify the injection and the ensuing real first prediction.
- The 20 GREEN controls are the six ineligible starts, six existing commit-marker cases, the midnight CLI launch, aging into the delivery-only WARN, the no-recipient shared-file case, and five day-artifact scenarios. The last five validate the base outcomes and drive the separate byte comparison.
- No existing test was changed. The 4c stale delivery fixture starts at 18:35 ET, with its earliest outstanding regular target already due. This excludes its startup advisory and retains its existing delivery WARN expectation. The other legacy assertions were kept.

## Byte comparison

`bytes-base.log` and `bytes-head.log` each report five passing scenarios. `compare.py` compares persisted bytes and captured sends using the same fixture paths, pinning persistence clocks, monotonic check duration, and model commit provenance. It strips just the added advisory line from the comparison of stderr and removes only the lane-specific artifact and corresponding DM where the contract changes them.

| Day | Matching files | Differences |
|---|---:|---|
| Normal delivered | 7 | Accepted journal line |
| Refused | 4 | Morning refusal line, CRITICAL DM and its own status file |
| MDP skip | 5 | Accepted journal line |
| No games | 1 | None |
| Stale delivered | 8 | Morning stale line and WARN DM at 10:00 instead of 16:10, with the corresponding age, body and timestamp in the same 4c status file. Delivery record, journal line and pick DM match. |

The shared CRITICAL file's seeded bytes match at first prediction for a successful morning send, a failed send, and no recipient. The controlled day-artifact fixtures disable EOD health and capture result polling. The separate offline `eod_probe.py` enables the real EOD runner with empty predictions and a failed morning transport. Its EOD DM and shared CRITICAL status match the base byte for byte. The persistent stderr fault applies to every advisory print, including its error report. Existing scheduler-loop output uses the normal stream.

## Runs

Logs and disposable scripts are under `.codex-review/cs6b1/`. Every pytest command used the yield wrapper. The wrapper's sandboxed `nice` warning was followed by a confirmed exit.

- Focused GREEN and legacy coverage in `green-final.log` gave 238 passed and 4 skipped. An earlier focused run exposed a fixture without a season switch, so the fetch case now starts with an active switch and missing automatic observation before running the real captured fetch.
- `clock-2027.log` and `clock-2028.log` each gave 61 passed. The process-local plugin substitutes the Python wall clock at noon ET on July 15 in each year and restores it at process cleanup.
- The unsandboxed one-process `combined.log` gave 74 passed across the new file, `tests/test_lightgbm_optional.py`, and `tests/test_web_saver.py`.
- The unsandboxed standalone `fast.log` ran on the committed tree and reported 8,746 passed, 23 skipped, 6 deselected, 1 xfailed, 37 failures and 16 errors in 25m 14s. The sorted 53 failure and error IDs exactly match `~/bts-runs/gate-failed.txt`. The wrapper recorded pytest exit 1 and the comparison script exited 0.

## Known limits

- The advisory is a snapshot. Same-day manual writes or corrections, a dated lapse at midnight, a fetch, a switch or flag edit, and the 14-hour freshness boundary can all change the later result. An accepted state can still yield a skip or no delivery. An advisory can arrive after a repair. A dated observation read during revoke can pass before a later check refuses.
- Less than ten minutes to the earliest planned check skips the advisory. This includes a 06:10 ET opener, a thin slate where the first start-minus-180 target follows wake by only minutes, and restarts close to a check. An empty plan also skips.
- The dispatcher deduplicates successful, persisted same-date sends across serialized restarts within each lane. Failed sends, interrupted status writes, or missing, corrupt or unwritable status can repeat on restart. Writes are neither atomic nor locked across processes. A failed morning CRITICAL is retried with the failure banner. A WARN-only failure has no banner.
- The inline transport can spend about 195 seconds in timeouts and backoff, which fits the ten-minute margin. Filesystem and DNS stalls have no end-to-end deadline. The systemd watchdog remains the backstop.
- An all-postponed nonempty slate can produce a false alarm. Private delivery, optional contest state, and runs for another date receive no morning check.
