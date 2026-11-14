# S6 handover — Tier A, sr-lastcheck

Changed only scheduler.py, the permitted cold-start test, and a new test file.
No commit, push, other worktree, production account, or shared-memory reads.

Control flow:
- run_day now iterates its original planned checks, then lazily offers extra
  checks only while unlocked, without a pick file or either standing skip signal.
- Extra checks use sorted distinct regular-season starts from the morning fetch;
  only deadlines still ahead qualify. The existing resolver, measured cascade
  budget, reserve floor, heartbeat and watchdog sleep apply to each deadline.
- Each extra check enters the existing check body: selection, confirmation gate,
  delivery, and the selected pick's own fallback handling. The first returned
  pick or recorded skip stops extra attempts. Planned future_runs stay intact.
- _maybe_alert_missed_pick now alerts immediately on an exhausted empty day with
  regular-season games, unless a skip/scoreable decision exists. _alert_missed_pick
  uses the existing missed_pick health DM transport with "No pick was selected today."
  Existing pick-file alert timing and delivery text are unchanged.

RED:
- Initial required baseline: 274 passed, 8 skipped, 1 strict S6 xfailed.
- Final corrected tests against original scheduler: 8 failed, 6 passed.
  Failures: late-lineup check timing and delivery; exhausted-lineup attempts;
  outage recovery; first extra attempt recording a skip; immediate no-pick alert;
  morning deadline and configured budget floor. All lacked attempts/alerts.
- Tests use synthetic HTTP, pinned clocks and fake DMs, with real training,
  run_day, selection, delivery and persisted pick/decision artifacts; no test git calls.

Mutations (each source restored byte-for-byte, asserted by the runner):
- Return immediately after planned checks: late-lineup delivery and
  test_first_extra_attempt_recording_a_skip_stops_later_attempts both failed (2 RED).
- Disable the empty-day alert: test_lineups_never_arrive_extra_attempt_then_one_immediate_alert
  failed on missing DM after its extra attempt (1 RED).
- Logs: .codex-review/sr/{red-b,mutation-no-extra-b,mutation-no-alert-b}.log.

Focused command (no whole suite):
```sh
TZ=America/New_York .venv/bin/python -m pytest tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_pickpath_review_pp1.py tests/test_pickpath_review_pp2.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/test_scheduler_decision_record_integration.py tests/test_cold_start_integration.py tests/test_scheduler_last_attempt.py -q -p no:cacheprovider --junitxml=.codex-review/sr/focused-b.xml
```
Final: 285 passed, 8 skipped; focused-b.log and focused-b.xml record the run.
Per file (passed/skipped): thin_slate 14/0; late_delivery 8/0; postponed 24/0;
pp1 4/0; pp2 59/8; incident_register 62/0; scheduler 76/0;
scheduler_decision_record_integration 24/0; cold_start 4/0; last_attempt 10/0.
All listed pre-existing files except the permitted cold-start edits are unedited.
The cold-start control retains 16:10/17:10/18:10 and now requires 18:35;
the xfail is removed. Its synthetic feed now supplies status for result polling,
which successful delivery newly reaches. No assertions about old pick paths weakened.
Existing main-loop body AST is identical; the final fallback text and all other
functions, including every prohibited target, are unchanged. git diff --check passed.

git diff --stat (tracked):
```text
 src/bts/scheduler.py                 | 42 ++++++++++++++++++++++++++++++++++--
 tests/test_cold_start_integration.py | 15 ++++++++-----
 2 files changed, 50 insertions(+), 7 deletions(-)
```
Also untracked tests/test_scheduler_last_attempt.py: 191 insertions.

Worst-case cost: U extra check cascades, U <= today's regular-season game count,
one per remaining distinct start with a deadline ahead, at start minus the resolved
and floored fallback buffer (default evening 35min; morning floor 27min).
Stop at the first pick/skip; overrun deadlines are skipped. Once a pick is selected,
unchanged handling can also refresh at its own later deadline and once in final
fallback if delivery fails and the file remains: conservative total bound U+2
additional cascades versus the old empty day. Existing pick/skip days add none.

Unmatched table rows: none. Lead's remote gate and fresh reviews remain outside this author task.
