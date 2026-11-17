# Verdict

**SIGN** on `9c13f7b0`. No blocking findings.

## Findings

- `src/bts/cli.py:2555` restores the base’s default-date lookup before the TOML load. The source diff from `87c118f6` contains only that reordering and its comment. Against `09d9b51f`, the executable `schedule` diff consists of the gate and its import.
- Across 176 cases per revision, HEAD matched the base on 64 gate-accepted cases, 48 warned dry runs after removing the exact warning prefix, and 16 malformed TOML cases. The worker date and mode, exit, output, filesystem operations and artifacts, fetches, notification calls, and sleeps matched. The worker received the same mapping the loader returned.
- When the load crossed midnight ET, the base and HEAD passed July 15 to the worker, which wrote July 15 scheduler state and reached the late lineup check. At `87c118f6` it took July 16 and idled. Midnight immediately after the gate also shifted the prior revision’s real default date. Six valid and two invalid-delivery cases shifted across these two boundaries. Midnight before the date lookup selected July 16 on HEAD. Explicit dates went through unchanged.
- All 48 real refusals at both gated revisions emitted the exact row-3 stderr and exited 2. The only file activity was Click’s validation and one TOML read. No worker, picks, scheduler state, heartbeat, schedule, prediction, notification, or sleep activity occurred. HEAD did one clock lookup before the load when the date was omitted. The prior revision did zero on refusal.
- NOTE at `tests/test_live_gate_cs6a.py:549`. The split-out warning case checks the warning and start date but drops the old explicit assertions that the loader ran and the worker received `dry_run=True`. The real-worker dry-run tests and the trace comparison cover both.

## Tests

- All eight focused cases passed on HEAD. With the current tests at `87c118f6`, the required-state and private real runs got July 16 instead of July 15, and the real-refusal order was `[load]` instead of `[date, load]`. The other five cases passed.
- Moving the date lookup after the load or after the gate each broke all four accepted ordering cases, the warned ordering case and the refusal ordering case. Looking up the date only on dry runs broke both accepted real cases and the real-refusal ordering case. Overwriting an explicit date broke its guard test. Reading state on refusal broke the resource test.
- The amended refusal tests retain the exact combined-trigger diagnostic, one real TOML load, worker/state/heartbeat/fetch/cascade spies, and picks and heartbeat path guards. The historical refusal tests failed on HEAD because they prohibited the date lookup that amendment 2 now requires. The historical dry-run cases still passed.
- All six files touched in CS6a passed under Python clocks set to July 15 at noon ET in 2027 and 2028, with 207 passes in each year.

## What I ran

- `schedule_round3.py` ran the real CLI, TOML loader and `run_day` across three revisions, four clock conditions and all date/mode combinations. Schedule, lineups, predictions, notifications and sleep were captured locally. [Log](schedule.log) · [Traces](schedule-traces.json)
- `mutations_round3.py` ran current and historical tests against archived source and five scratch mutations. `clocks_round3.py` ran the two wall-clock suites. Each family ran once through `yield_run.sh` with the network audit hook. All wrappers exited 0 after their `nice` warnings. [Mutations](mutations.log) · [Clocks](clocks.log)
- Both commit ranges passed `git diff --check`. The AST comparisons and source diff confirmed the scope. The tracked tree is clean.
