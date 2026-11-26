# Change set F2 handover

Commit `2e20fdfc6bf1b406d879289f0465721fd8f2d4f7` on `fix/wr-f2`, based on `3b944d94`.

The saver disagreement uses the checked slate's ET season and dispatches as a separate WARN DM after the existing digest. It has its own status file and dedup key. The historical attention label appears before the prior-slate pointer.

## Tests first

On the unchanged production code at `3b944d94`, the new file had **11 failures and 39 passes**. The final run passed all 50 cases.

| Contract row | Base result |
| --- | --- |
| M1 | Failed. No live warning for `active`. |
| M2 | Failed at streaks 9 and 14. No live warning for `not_earned`. |
| M3 | Failed in July and across December 31. No historical warning for the processed slate. |
| M4 | Failed for both initial flags at the missing day-one warning and delivery. The final run also verified the real CLI repair, byte-for-byte preservation of the observation, and the spent result on the next automatic read. |
| M5 | Failed at the missing day-one warning for the `active` fixture. The accepted day-two automatic read returned `True` before correction on both the base and the candidate, and the final run verified `False` after repair. |
| K1 | Passed for streaks 10–15. Existing live and historical messages matched. |
| K2 | Passed. A `used` flag added no warning. |
| K3 | Passed. Unusable flags preserved the coverage alert. FIFO read bounded by a 15-second alarm. The integer and nesting decoder cases exercised the corresponding errors on the final code. |
| K4 | Passed. An `active` or out-of-zone `not_earned` observation added no warning across all flag states. |
| K5 | Passed. Expired, automatic, legacy, and missing-state cases matched the base's ordered alerts, DM, and status bytes in both modes. |
| K6 | Passed. Future dates, coverage, expired legacy override, and refusal matched the base's alerts, ordering, DM, and status bytes in both modes. |
| K7 | Passed. Invalid dated observations matched the base's alerts, DM, and status bytes in both modes. |
| Same-date reruns | Both orders failed on the base at the missing mismatch incident. The final run sent and deduplicated the two WARNs independently. |
| Chat limit | Failed on the base at the missing mismatch incident. The final run sent the original near-limit digest first with identical body and status bytes, followed by the separate WARN digest. |
| Interrupt and restart | The base failed at the missing first send. The final run simulated an interrupt before receipt replacement, checked the preserved flag and prior receipt, then verified retry and subsequent suppression. |
| Malformed flag and midnight | K3 and M3 above. The final historical run also checked replacement of either input after midnight. |
| Failed or unconfirmed send | The base failed at the missing transport attempt. The final run recorded failure, retried, and verified the normal dispatcher's treatment of an empty transport ID. |

## Rendered DMs

### Live active (384 raw characters, 446 DM characters)

```text
BTS health WARN attention:
- [saver_state] Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. The flag for season 2027 read active at this check. A later automatic decision may count a saver as available. Verify the current season's saver and prior entries in MLB before any correction. Use `bts season-contract status` for the scheduler's picks directory and follow checklist B8a for any correction. (1st observed day)
```

### Live not_earned (388 raw characters, 450 DM characters)

```text
BTS health WARN attention:
- [saver_state] Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. The flag for season 2027 read not_earned at this check. A later automatic decision may count a saver as available. Verify the current season's saver and prior entries in MLB before any correction. Use `bts season-contract status` for the scheduler's picks directory and follow checklist B8a for any correction. (1st observed day)
```

### Historical December 31 (500 raw characters, 562 DM characters)

```text
BTS health WARN attention:
- [saver_state] Prior slate 2027-12-31 ET. The dated observation recorded the saver as used. At this check the flag for season 2027 read active. (1st observed day) This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
```

## Existing tests

Ran all 78 existing files in the union of the contract index and the requested search, in sequential bounded processes with Eastern time. The sandbox refused the requested priority change, then ran the commands. A first runner test attempt exposed a boolean attention stub on the existing dispatch path. The final code preserves the original call when the mismatch is absent, and the targeted rerun passed all 881 cases across the new file, runner, alert-status callers, alert-status Carbon tests, CLI integration, and dated contest tests. The web saver socket case reached the sandbox's loopback bind denial before its assertions.

| Test file | Result |
| --- | --- |
| `tests/health/test_alert.py` | 23 passed |
| `tests/health/test_alert_status_cs6b2.py` | 337 passed |
| `tests/health/test_alert_status_cs6b2_carbon_r2.py` | 11 passed |
| `tests/health/test_alert_status_cs6b2_review.py` | 15 passed |
| `tests/health/test_contest_state.py` | 16 passed |
| `tests/health/test_f4_exception_propagation.py` | 9 passed |
| `tests/health/test_h6_resurface.py` | 4 passed |
| `tests/health/test_late_delivery.py` | 12 passed |
| `tests/health/test_runner.py` | 21 passed on rerun |
| `tests/scripts/test_state_segment_policy_candidate_screen.py` | 5 passed |
| `tests/test_alert_status_callers_cs6b2.py` | 22 passed |
| `tests/test_alert_status_cs6b2_carbon.py` | 28 passed |
| `tests/test_backup.py` | 27 passed |
| `tests/test_backup_cli.py` | 5 passed |
| `tests/test_cli_integration.py` | 70 passed |
| `tests/test_cli_schedule_date.py` | 1 passed |
| `tests/test_cold_start_integration.py` | 3 passed, 1 xfailed |
| `tests/test_contest_dated_cs4a.py` | 690 passed |
| `tests/test_contest_dated_cs4a_review.py` | 49 passed |
| `tests/test_contest_fetch.py` | 54 passed |
| `tests/test_contest_state.py` | 17 passed |
| `tests/test_contest_state_decoder_limits.py` | 12 passed |
| `tests/test_contest_state_defensive_reads.py` | 53 passed |
| `tests/test_decision_saver_fallback.py` | 3 passed |
| `tests/test_e3_missed_pick_alert.py` | 7 passed |
| `tests/test_entry_intent.py` | 52 passed |
| `tests/test_health_endpoint.py` | 12 passed |
| `tests/test_incident_2026_08_30.py` | 4 passed |
| `tests/test_incident_register_2026.py` | 62 passed |
| `tests/test_late_delivery_guard.py` | 8 passed |
| `tests/test_live_gate_cs6a.py` | 56 passed |
| `tests/test_live_gate_cs6a_review.py` | 8 passed |
| `tests/test_morning_state_cs6b1.py` | 61 passed |
| `tests/test_morning_state_cs6b1_review.py` | 17 passed |
| `tests/test_pick_chosen_on_cs5a.py` | 171 passed |
| `tests/test_pickpath_review_pp1.py` | 4 passed |
| `tests/test_pickpath_review_pp2.py` | 59 passed, 8 skipped |
| `tests/test_postponed_delivery_guard.py` | 24 passed |
| `tests/test_saved_pick_cs5a_review.py` | 48 passed |
| `tests/test_saved_pick_paths_cs5a.py` | 97 passed, 4 skipped |
| `tests/test_saver_cs3.py` | 184 passed |
| `tests/test_saver_cs3_review.py` | 22 passed |
| `tests/test_saver_state.py` | 16 passed |
| `tests/test_scheduler.py` | 76 passed |
| `tests/test_scheduler_decision_record_integration.py` | 24 passed |
| `tests/test_scheduler_eve_of_break.py` | 10 passed |
| `tests/test_season_contract_cs4b1.py` | 102 passed |
| `tests/test_season_contract_cs4b1_edges.py` | 85 passed |
| `tests/test_season_contract_cs4b1_review.py` | 7 passed |
| `tests/test_season_contract_cs4b2.py` | 1561 passed |
| `tests/test_season_contract_cs4b2_edges.py` | 20 passed |
| `tests/test_season_contract_cs4b2_review.py` | 12 passed |
| `tests/test_stale_delivery_cs4c.py` | 62 passed |
| `tests/test_stale_delivery_cs4c_carbon.py` | 20 passed |
| `tests/test_stale_delivery_cs4c_review.py` | 47 passed |
| `tests/test_streak_season_cs2.py` | 249 passed |
| `tests/test_streak_season_cs2_review.py` | 16 passed |
| `tests/test_thin_slate_checks.py` | 14 passed |
| `tests/test_web_saver.py` | 11 passed, 1 sandbox-blocked loopback bind |
| `tests/test_wr_a_calendar.py` | 8 passed |
| `tests/test_wr_a_pick.py` | 31 passed |
| `tests/test_wr_a_review_carbon.py` | 25 passed |
| `tests/test_wr_b_contract.py` | 24 passed |
| `tests/test_wr_b_edges.py` | 73 passed |
| `tests/test_wr_b_r2_review_adopt.py` | 11 passed |
| `tests/test_wr_c1_clock.py` | 5 passed |
| `tests/test_wr_c1_history.py` | 41 passed |
| `tests/test_wr_c2_carbon_v2.py` | 21 passed |
| `tests/test_wr_c2_r2_carbon.py` | 18 passed |
| `tests/test_wr_c2_receipt_identity.py` | 143 passed |
| `tests/test_wr_c2_recovery.py` | 101 passed |
| `tests/test_wr_c3_r2_entered_pick_blockers.py` | 9 passed |
| `tests/test_wr_e_texts.py` | 65 passed |
| `tests/test_wr_g_guard.py` | 44 passed |
| `tests/test_wr_h2_caps.py` | 16 passed |
| `tests/test_wr_w15_health.py` | 45 passed |
| `tests/test_wr_w15b.py` | 58 passed |
| `tests/validate/test_ope.py` | 16 passed |
