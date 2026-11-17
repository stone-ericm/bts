# Verdict

**SIGN** on `87c118f6`.

## B1 and row 3

- B1 is closed. Across 50 dry-run cases at HEAD, the date, exit, worker output, resource access, heartbeat, notification and sleep traces matched `09d9b51f` apart from the required warning. `cad7c5ec` diverged in 10 omitted-date cases when midnight fell inside the TOML load or between loading and the late date read. An explicit `--date` always went straight through.
- A real accepted launch without `--date` at 23:59:59.900 ET on July 15, 2027, with TOML loading through 00:00:00.100 ET, took different paths. The base fetched July 15, wrote its scheduler state and reached its late lineup check. HEAD and `cad7c5ec` fetched July 16, found an off day, looked ahead to July 17 and idled for 35,999.9 seconds. This can skip previous-day work. Row 3 expressly requires `load → gate → default date`, so the new date is the specified outcome. Crossing between the gate and lookup also selects July 16. Six of 30 accepted real launches differed from the base. All 50 real cases matched `cad7c5ec`.
- All 20 refused real runs at HEAD exited 2 on stderr, including runs with an explicit date. The loader opened only the TOML. The trace recorded zero default date reads, `run_day` calls, resource opens/stats/mkdirs, schedule fetches, notifications, sleeps or prediction calls. Accepted workers received the identical loaded mapping.

## Blocking findings

None.

## Notes

- `git diff cad7c5ec HEAD -- src` adds exactly ten lines in `src/bts/cli.py`. Six implement or explain the dry-run order and four extend the `entry-intent` help. The help command printed the new text and exited 0 with the TOML loader uncalled.
- In `09d9b51f..HEAD`, only `schedule` moves a date read relative to loading. `orchestrate` requires `--date` and still loads TOML before clock-dependent worker work. `run` requires `--date`, keeps its opening timestamp and the three state-loader call sites. `check_entry_intent` reads its mapping without a clock. `day-outcome` still resolves an omitted date before loading TOML and `gather` only adds the gate opt-out. No clock-dependent default expression moved.
- `tests/test_live_gate_cs6a_review.py:10` lists `private_mode = "false"` as a real-scheduler case. The tracked test uses `pick_delivery = "private"` with `entry_intent = "research"`. A scratch substitution of the legacy option passed the same real-scheduler assertions for local locking and zero sends.
- The hook and tracing search found child-process hooks in the incident-register tests and in the historical CS4b-2 caller control. The in-process `sys.settrace` test restores the prior tracer in `finally`. No `sys.setprofile` call or module-level socket patch was found. Existing import-time `sys.path` changes are in `tests/conftest.py` and `tests/test_audit_driver.py`. `tests/test_sync.py` sets a moto environment default at import.

## Tests

- Both lead-added functions passed at HEAD. The dry-run ordering test failed on both its accepted and warning inputs at `cad7c5ec` and with the late-date mutant. `run_day` received July 16 instead of July 15. The real-refusal test passed at `cad7c5ec` and failed when the mutant called `_today_et` before the gate. The spies intercept the actual `cli._today_et`, `orchestrator.load_config` and `scheduler.run_day` bindings used by the command.
- All eight adopted cases passed at `cad7c5ec` and HEAD. Each failed under a relevant scratch mutation. Applying the gate to day-outcome broke three classifications. Reopening TOML broke the orchestrator snapshot case. Making selection or cached delivery optional broke both truthy-flag cases. A public send broke the private-lock case. An optional Read 3, a forced Read 2 on an already-posted pick, and making a dated observation read the switch each broke the combined `bts run` case.
- The adopted module, `test_web_saver.py` (TCP loopback) and `test_sd_notify.py` (Unix datagrams) passed 25/25 in one pytest process in each order. An actual TCP connect attempted during an adopted test raised the hook's `RuntimeError`. A fresh connect succeeded after teardown in both processes. Removing `active[0] = False` reproduced failure in `test_do_post_saver_transition_socket_wiring` and kept blocking connects after teardown.
- All six added or touched test files passed with Python wall clocks set to noon ET on July 15 in 2027 and 2028. At each date I ran every file ten times alone and ten times as the final file after the other five in the same pytest process. All 240 runs passed, with 28,420 passing test instances.

## What I ran

- `schedule_round2.py` ran 100 cases per revision with the real CLI, TOML loader and `run_day`. The probe captured the schedule and prediction transports, notifications and sleeps. Filesystem traces include the heartbeat, picks and model paths. The first probe attempt asserted against the wrong cascade argument and was corrected before collecting the three-revision traces in `schedule-traces.json`.
- `clocks_round2.py` ran through `yield_run.sh` once and exited 0. The cron tests used their throwaway `HOME` and shim `crontab` and `uv`. The `nice` warning did not stop the run.
- `mutations_round2.py` ran through `yield_run.sh` once and exited 0, using archived `cad7c5ec` source and ten independent scratch source changes. Logs preserve the expected failing assertions.
- `socket_round2.py` ran through `yield_run.sh` once and exited 0. It ran outside the sandbox because the sandbox denied a local bind. Its audit guard permitted only the test loopback and Unix addresses. The harness kept `HTTPServer`'s name lookup numeric and exercised the real HTTP and datagram transports.
- Both commit-range `git diff --check` runs passed. The tracked tree is clean.
