# Kickoff 3: two test-only follow-ups from the round-2 review (SIGN)

The reviewer signed your change: `.codex-review/pp-c1-review/report-r2.md`. Read its "False greens" section. Make these two TEST-ONLY changes in `tests/test_pickpath_review_pp2.py`; do not touch production code.

1. `test_delivered_hit_plus_void_double_adds_exactly_one` passes even with `picks._slot_is_voided` disabled, because its postponed feed is Final with explicit zero at-bats and the grader's no-official-at-bat rule supplies the Pass by itself. Add the case the reviewer used (its probe file is `/tmp/test_ppc1_review_r2.py`; the mutation harness is `/tmp/ppc1_r2_disable_void_resolution.py`): the void leg's game is Preview / Postponed in the schedule with no roster entry, so the Pass can only come from schedule-based void resolution. Show it fails with `_slot_is_voided` disabled (in memory, as the reviewer did) and passes unmodified.
2. The daemon-scorer variant calls `run_result_polling(kept.game_pk, ...)`. Real `run_day` passes the PRIMARY pick's game even when the primary is the void leg. Use the production argument.

Run the kickoff's focused set plus the file you changed and paste the last line. Write `/Users/eric/projects/bts-pp-c1/.codex-review/pp-c1/handover-3.md` (under 40 lines: what changed, the RED with the mutation, the GREEN, the focused run's last line). Then print one line `PPC3-<first 8 hex characters of the sha256 of the handover file>` and `DONE`.
