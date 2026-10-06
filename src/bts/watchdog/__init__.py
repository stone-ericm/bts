"""The C1 rank-2 watchdog (`bts watchdog`): a separate, read-only checker that alerts, never repairs.

Registration: `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (FROZEN). Build plan:
`docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`.

- **R1, detect and alert only:** no pick, decision, streak, saver, scheduler-state or shared health-state write; no
  re-delivery, re-grading, reconcile or configuration change.
- **R5, writes:** every watchdog write, lock, log, dedup entry and temp file resolves beneath `data/watchdog/`
  (`root.OwnedRoot`); symlinks and path escapes are refused. Notification state has a short owned critical
  section; no network work holds it.
- **R6, network:** no authenticated contest request and no cookie access.

W0 (this package's skeleton): the owned root, the clock seam, check results, the job runner, and notifications.
"""
