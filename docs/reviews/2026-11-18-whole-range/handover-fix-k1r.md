# Handover: change set K1r (ruling R75), `bts-wr-fresh16`, 2026-11-27

Branch `fix/wr-k1-revert-dedup`: `7a3e116e` (the lead's source change) and `1eb8602f` (tests and texts, this session).

## What K1r is
R75: the focused read of FINAL3 (`WRK12READ-43f82e3d`, `closing-test-2/addendum-final3/codex-read/report.md`) found that K1's slate-date dedup suppressed an alert FINAL2 delivered when the first send for slate D came before midnight. Way (iii): revert the dedup-date half of K1 (C3-2 returns to the ledger); keep the shorter refusal DM (F2-1) and its slate label; apply the read's two text edits.

## Source (`7a3e116e`, read and kept)
- The four in-day alerts (`_alert_contest_state_failure`, `_alert_missed_pick`, `_refuse_late_delivery`, `_refuse_void_game_delivery`) no longer pass `now_et_date` to `dispatch_dm_for_health_alerts`. `git diff d0bdcdba 7a3e116e -- src/bts/scheduler.py src/bts/health/alert.py`: at these four sites the dispatch calls are FINAL2's again; what remains of K1 is the DM text (`No pick was sent for {D} at this check. Reason: …`, the slate label, the 400-code-point cut), the `slate_date` keyword the callers pass for that label, and the `999+` banner in `alert.py`.
- Note (not changed): `_alert_missed_pick` still accepts `slate_date` and no longer uses it; `_maybe_alert_missed_pick` still passes it. Harmless; left as the lead wrote it.

## Tests (`1eb8602f`)
On `7a3e116e` the two K1 files had 64 failing tests: every `test_rollover` case (24), `test_cross_midnight_wait`, both `test_e8_historical_*`, `test_s8_real_callers[final_midnight]`, both `test_real_switch_over_midnight`, the 33 alerting `test_scheduler_alert_dates` cases (the three successful-delivery controls passed) and `test_operator_overnight_fallback`. Those, and only those, were changed (R75 is the permission). On `main` `5cd58a0c` with these test files, the same 64 fail, plus the two new ones (66; run with `PYTHONPATH=<main copy>/src`, `bts.__file__` printed). Every other K1 assertion (DM length for every refusal reason, the slate label, the texts) is unchanged.

Each changed assertion:
- `test_rollover` (steps are slate, wall date, time, DMs expected): E4-E6 `(D, D+1, 00:02, 1), (D+1, D+1, 08:00, 1→0)`; E7 `(D, D, 23:30, 1), (D, D+1, 00:02, 0→1), (D, D+1, 00:02, 0)`; E9 `(D+1, D+1, 00:01, 1), (D, D+1, 00:02, 1→0), (D+1, D+1, 08:00, 1→0)`; the receipt's `sent_day` is compared with the wall date instead of the slate (`(expected, slate)` → `(expected, wall)`). The test comment (corrected in the handover commit, wording only) says E4-E6 and E9 show C3-2's suppression and E7 its repeat.
- `test_cross_midnight_wait`: `sent_day == "2027-06-14"` → `"2027-06-15"` (the wall date at the send).
- `test_e8_historical_contest_state` and `test_e8_historical_late_delivery`: `len(sent) == 1` → `2` (the slate's end-of-day digest is delivered after an after-midnight in-day send, as in FINAL2; the rerun is deduplicated).
- `test_s8_real_callers[final_midnight]`: `all(d is not None and d.isoformat() == day …)` → `all(d is None …)` (no dedup date passed).
- `test_real_switch_over_midnight`: `len(sent) == (2 if E4 else 1)` → `(1 if E4 else 2)`; `sent_day == steps[-1][0]` (slate) → `steps[-1][1]` (wall).
- `test_wr_k1_r2.py::test_scheduler_alert_dates`: `all(row[4] == slate)` → `all(row[4] is None)`; `receipt["sent_day"] == slate` → `== ` the wall date of the last dispatch, plus `(wall != slate) == overnight`. The archived probe (`review-fix-k1-codex/probe_callers.py`, not edited) decides whether to expect K1's slate date from the presence of `slate_date` on `_alert_contest_state_failure`; the test shows it the signature without that parameter (`_WallClockDedupView`), so it checks its own base (wall-clock) expectations.
- `test_operator_overnight_fallback`: `probe_operator.overnight` inlined with its one dedup assertion `record[4] == slate` → `record[4] is None`, plus `receipt["sent_day"] == later`; its text and CLI checks unchanged.

New (`tests/test_wr_k1r_dedup.py`, the read's harness `codex-read/rollover.py`, real `run_day`): two void refusals for different games of one slate at 23:58 and 00:24 ET both reach the wire; an end-of-day digest at 23:46, a restart at 23:59 and a live refusal at 00:01 (then that run's digest) all reach the wire. Both RED on `main` (the second dispatch's `wire` is False), green here.

## Texts
- B8a: "The refusal names this command." → "The refusal DM points to the scheduler journal and `bts season-contract status`. Use the dated command above when the timing and prior-entry checks below permit filing."; after "file the verified dated observation between runs.": "After the last lineup check, the final fallback needs a saved, undelivered pick. A changed filing records the request and updates the dated observation. The scheduler handles pick selection and delivery." (the check's edit replaced the read's "Filing only changes the dated observation.") Pinned by `test_k1r_b8a_refusal_and_last_check_sentences` (RED on main); `test_wr_f2_saver_flag.py::test_e7_other_b8a_text_matches_round_two` undoes both before its hash.
- `scripts/audit/season_walk/README.md`: the finale bullet replaced verbatim; the effective-day sentence now says the count is floored at one for a slate with candidates, then clamped to 180 days (matches `walk.check`: `max(1, expected) if "slate" in r`). Pinned by `test_k1r_season_walk_readme`.
- `CLAUDE.md`: K1's Landed bullet records the revert, C3-2 ledgered again, and the two edits.

## Runs
- Focused: the two K1 files, the new file, the six checklist-reading files and `tests/scripts/season_walk`: 394 passed; `tests/test_wr_f3_skip_state_review.py` (reads CLAUDE.md) 92 passed.
- Gate on `1eb8602f`: see the session report.

## Known and not fixed
**C3-2 is ledgered again.** A successful in-day dispatch for slate D evaluated after midnight ET records D+1. While that receipt covers the source, it suppresses same-source in-day alerts evaluated on D+1 for either slate, including the first one for slate D+1. A delayed check or delivery can cross midnight. A same-source in-day alert for D can repeat after midnight when the latest receipt is dated D. D's end-of-day digest can send against a D+1 receipt because the digest uses D. The read's alternative (incident keys per game and per live refusal, way (i)) is not built.
