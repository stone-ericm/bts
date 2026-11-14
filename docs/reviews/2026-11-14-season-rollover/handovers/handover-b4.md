# B4 handover — base 6a4c20c

1. Chose (b): reuse only for extra attempts. run_single_check has an opt-in
include_lock_decision argument, default False; run_day supplies it only for extras.
Planned checks retain their previous return shape, calls and fallback decisions.
The 8/30 notes establish the one-cascade budget and the intended reuse branch;
enabling reuse globally nevertheless changes a real planned-check scenario:
the constructed planned overrun normally refreshes at 18:56 and refuses at 19:12;
the global-reuse experiment instead delivers at 18:56. Therefore it was rejected.
With (b), the unlocked 18:35 extra cascade finishes and delivers at 18:51 with no
second cascade. The later-pick control still waits to 19:35, refreshes and delivers
at 19:51. Existing planned-check scenario outcomes are unchanged.
2. _playable_game_pks uses _delivery_game_statuses and picks._is_void_detailed_state.
After each extra sleep, run_day performs one eligibility read for that start;
an entirely void start runs no cascade. Before an alert, one fresh read checks the
slate, including after any old alert-window wait. Both paths fail open on read
errors or missing statuses. Existing delivery-guard reads remain unchanged.
Tests cover postponed/cancelled slates, mixed and shared starts, read failures,
and games becoming void during the last cascade (no day-end alert).
3. _alert_missed_pick now passes now_et_date from the slate, for both old and new
alerts. Old direct callers with a pick infer its date; empty-day callers supply it.
Yes, the old missed-pick alert could already misfire at the base: 12dbaaa's sender
also omitted now_et_date and therefore used wall-clock dedupe. Real run_day tests
cover empty and undelivered-pick days: same-slate midnight restart sends no second
DM, leaves sent_day bound to that slate, and does not suppress tomorrow's alert.
Validation: baseline 16 passed. RED: 9 failures (one lost pick, five void-state
cases, one missing status consultation, two date cases); later-deadline control
already passed. Mutations: disable reuse → 1 failed; ignore void statuses → 2;
remove slate date → 2; enable reuse globally → planned control failed. Each source
restoration was asserted byte-for-byte. Protected functions and final fallback
are byte-identical to base. No new configuration or other source files changed.
Cold-start control asserts exact feed times {16:10,17:10,18:10,18:35} before first pitch; result polling is separately {19:20}.
Full B3 list rerun: 634 passed, 9 skipped, 0 failed across all 41 listed paths.
Command form: TZ=America/New_York .venv/bin/python -m pytest <listed files> -q -p no:cacheprovider.
Exact command, paths, per-file counts and JUnit: b4-command.txt, b4-test-files.txt,
b4-counts.txt, b4-focused.xml. Companion 66f3992 plus final S6 scheduler/tests (isolated scratch, PYTHONPATH=src): 23 passed.
Worst case: at most K extra check cascades, K = remaining distinct playable starts.
A due pick reuses its own cascade; a later deadline can add one existing refresh,
and a retained pick after delivery failure can add one final fallback refresh:
conservative total bound K+2. Existing pick/skip paths add no extra checks.
git diff --stat 6a4c20c:
```text
 src/bts/scheduler.py                 |  38 ++++++-
 tests/test_cold_start_integration.py |   3 +-
 tests/test_e3_missed_pick_alert.py   |   2 +-
 tests/test_scheduler_last_attempt.py | 202 ++++++++++++++++++++++++++++++++++-
 4 files changed, 234 insertions(+), 11 deletions(-)
```
git diff --check passed. No unmatched rows, commits, pushes, or other-worktree edits.
