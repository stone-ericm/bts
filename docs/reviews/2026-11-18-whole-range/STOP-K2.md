# STOP-K2: change set K2 stopped after round 2 (2026-11-26 14:45 ET, bts-wr-fresh15)

Branch `fix/wr-k2b` (pushed; built on K1's tip `4c301f7f`; code `26c621c7` round 1, `6f0ab421` round 2, records to `ec5966e1`). Not merged; `main` unchanged.

## What K2 is
Fresh selection (`select_pick`, `DecisionContext.min_days=1`) decides a slate whose stored day count is 0 as the season's last day (a makeup game after the stored finale is played, not skipped at every streak); direct resolver calls, the standalone bar and health keep the raw count; `run_day`, after the morning state check, journals and (with a recipient, lead and deadline) DMs one WARN per slate date that the stored boundary is behind the schedule. Contract `kickoff-fix-k2.md` with Amendments 1 to 4; two contract checks BUILD WITH THESE EDITS (`WRCHKK2-e39104d4`, `WRCHKK2B-0e3b8c3e`); an owner-list item (ruling 1's trust in candidate rows).

## The rounds
- Round 1, Codex xhigh **BLOCK** `WRK2REV1-1176ba94`: 373,188 positive-day decisions equal to the base; B1: the warning's journal lines on the scheduler thread let a stalled journal delay the first check. Carbon stopped (R70).
- Round 2 (all warning I/O off the scheduler thread: a queue-fed writer thread and a dispatch thread; the texts), Codex xhigh **BLOCK** `WRK2REV2-17651cb0` (archive `review-fix-k2-codex-r2/`): B1, E2 to E4 and the test gap FIXED; positive-day decisions still the base's. **B2 (introduced):** on a failed boundary send with a full stderr pipe, the shared dispatcher's failure line (`report_unknown_outcome_alert_failure`, through `sys.stderr`) runs on the dispatch thread and holds the buffered stderr lock; a SIGINT then unwinds `run_day` but the Python process does not exit until the pipe drains (interpreter shutdown waits on that lock). Amendment 4 stated a delayed line, not a process that cannot exit. Smallest fix (reviewer): give the acknowledgement dispatch an optional failure reporter and pass K2's queue, so the failure line goes through the daemon writer via `os.write`; other callers keep the default. Also: a nonblocking sink returning EAGAIN drops the observation (retry off the scheduler thread); one writer per permanently blocked dispatch in a reused process; an edit to the E1 limit's wording. Carbon round 2 stopped before its report (R70 item 5).
- Counted BLOCKs for K2: two (R57: the lead rules the one follow-up).

## Gate
On `ec5966e1`: started 14:34, running at the stop.

## Ways forward (for the lead)
1. **Recommended: K2b**, the reviewer's B2 fix only (an optional failure reporter in `health.alert._dispatch_with_acknowledgement`, default unchanged for every other caller; K2 passes a reporter that puts the line on its queue), plus the EAGAIN retry in the writer and the E1 wording edit, each with a RED test (the real-full-pipe SIGINT probe kept); one fresh Codex xhigh check limited to those changes.
2. Drop the DM from K2 and keep the journal line only (no dispatch thread at all): the floor alone fixes the lost day; the operator learns of the boundary from the journal and the 08:30 day-outcome. Cost: no push notice on a makeup day.
3. Accept K2 as it stands with B2 written as a known limit (Eric).
