# F2 round 2 handover

Commit `3b7d5a18` on `fix/wr-f2`.

## Tests first

The focused run against the round-1 implementation at `2e20fdfc` had 13 expected failures and 46 passes. Both item 9 alias cases also failed on a repeat run after their expectations were updated. The final focused run passed all 59 cases in `tests/test_wr_f2_saver_flag.py` and `tests/test_f2_b1_stat_oserror_regression.py`.

| Row | Round-1 result | Final result |
| --- | --- | --- |
| Finding 1, unreadable flag | RED in four adopted Carbon cases. Permission denial and overlong symlink targets raised in live and historical modes. | Prior alerts preserved and no crash DM. |
| Finding 2, both receipt aliases | RED for symlink and hard link. Both changed saver flag bytes. | The DM sends, flag bytes and alias are preserved, and an eligible rerun sends again. |
| Finding 3, root override | RED. The existing digest sent and receipt naming raised before the mismatch DM. | Existing digest matched the control and sent first. The mismatch uses the default dedicated receipt. |
| Items 2 and 5, empty ID and changed body | RED. Empty ID produced a sent receipt and a changed same-day body was suppressed. | Empty ID records failure and retries. Same-day changed body sends and identical acknowledged bodies deduplicate. |
| Item 3, live text and B8a | Three live cases were RED on the prior text. The checklist case was RED on the first addition. | Complete live DMs and both checklist additions matched exactly. |
| Items 4 and 7, decoder limit and FIFO | Fixture checks passed on round 1. The decoder fixture explicitly raised `RecursionError`. Each FIFO read had its own 15-second alarm and an exception outside the reader's handlers. | The decoder-limit flag preserves K1. Both FIFO modes complete within their separate deadlines. |
| Remaining comparison and delivery rows | Passed on round 1. | Passed, including K2–K7, both same-day incident orders, consecutive days, the unchanged near-limit digest, after midnight, and base snapshots. |

## Item 8 assertion changes

- In `test_m1_m2_live_warning_and_delivery`, item 3 changes the expected live sentence and the rendered body comparison now checks the exact DM.
- In `test_failed_send_and_interrupt_before_status_can_retry`, items 2 and 5 change the empty-ID receipt expectation to failed and require an eligible retry. The suppression assertion now follows the acknowledged retry.
- Item 9 applies to the newly added alias assertions. No assertions from the base `a4e8278d` changed in this commit.

## Required existing files

Every file in the contract index and requested test search ran after the final implementation edit, in sequential bounded batches. The sandbox declined the priority change and pytest continued. The web saver socket test was blocked at the loopback bind. The full fast suite was outside the brief.

### Batch 1

453 passed in 1.64s.

- `tests/health/test_alert.py`
- `tests/health/test_alert_status_cs6b2.py`
- `tests/health/test_alert_status_cs6b2_carbon_r2.py`
- `tests/health/test_alert_status_cs6b2_review.py`
- `tests/health/test_contest_state.py`
- `tests/health/test_f4_exception_propagation.py`
- `tests/health/test_h6_resurface.py`
- `tests/health/test_late_delivery.py`
- `tests/health/test_runner.py`
- `tests/scripts/test_state_segment_policy_candidate_screen.py`

### Batch 2

949 passed, 1 xfailed in 3.47s.

- `tests/test_alert_status_callers_cs6b2.py`
- `tests/test_alert_status_cs6b2_carbon.py`
- `tests/test_backup.py`
- `tests/test_backup_cli.py`
- `tests/test_cli_integration.py`
- `tests/test_cli_schedule_date.py`
- `tests/test_cold_start_integration.py`
- `tests/test_contest_dated_cs4a.py`
- `tests/test_contest_dated_cs4a_review.py`
- `tests/test_contest_fetch.py`

### Batch 3

230 passed in 10.12s.

- `tests/test_contest_state.py`
- `tests/test_contest_state_decoder_limits.py`
- `tests/test_contest_state_defensive_reads.py`
- `tests/test_decision_saver_fallback.py`
- `tests/test_e3_missed_pick_alert.py`
- `tests/test_entry_intent.py`
- `tests/test_health_endpoint.py`
- `tests/test_incident_2026_08_30.py`
- `tests/test_incident_register_2026.py`
- `tests/test_late_delivery_guard.py`

### Batch 4

456 passed, 8 skipped in 2.22s.

- `tests/test_live_gate_cs6a.py`
- `tests/test_live_gate_cs6a_review.py`
- `tests/test_morning_state_cs6b1.py`
- `tests/test_morning_state_cs6b1_review.py`
- `tests/test_pick_chosen_on_cs5a.py`
- `tests/test_pickpath_review_pp1.py`
- `tests/test_pickpath_review_pp2.py`
- `tests/test_postponed_delivery_guard.py`
- `tests/test_private_mode_transport.py`
- `tests/test_saved_pick_cs5a_review.py`

### Batch 5

2108 passed, 4 skipped in 5.33s.

- `tests/test_saved_pick_paths_cs5a.py`
- `tests/test_saver_cs3.py`
- `tests/test_saver_cs3_review.py`
- `tests/test_saver_state.py`
- `tests/test_scheduler_decision_record_integration.py`
- `tests/test_scheduler_eve_of_break.py`
- `tests/test_season_contract_cs4b1.py`
- `tests/test_season_contract_cs4b1_edges.py`
- `tests/test_season_contract_cs4b1_review.py`
- `tests/test_season_contract_cs4b2.py`

### Batch 6

479 passed in 3.36s.

- `tests/test_season_contract_cs4b2_edges.py`
- `tests/test_season_contract_cs4b2_review.py`
- `tests/test_stale_delivery_cs4c.py`
- `tests/test_stale_delivery_cs4c_carbon.py`
- `tests/test_stale_delivery_cs4c_review.py`
- `tests/test_streak_season_cs2.py`
- `tests/test_streak_season_cs2_review.py`
- `tests/test_thin_slate_checks.py`
- `tests/test_wr_a_calendar.py`
- `tests/test_wr_a_pick.py`

### Batch 7

462 passed in 7.45s.

- `tests/test_wr_a_review_carbon.py`
- `tests/test_wr_b_contract.py`
- `tests/test_wr_b_edges.py`
- `tests/test_wr_b_r2_review_adopt.py`
- `tests/test_wr_c1_clock.py`
- `tests/test_wr_c1_history.py`
- `tests/test_wr_c2_carbon_v2.py`
- `tests/test_wr_c2_r2_carbon.py`
- `tests/test_wr_c2_receipt_identity.py`
- `tests/test_wr_c2_recovery.py`

### Batch 8

237 passed in 36.55s.

- `tests/test_wr_c3_r2_entered_pick_blockers.py`
- `tests/test_wr_e_texts.py`
- `tests/test_wr_g_guard.py`
- `tests/test_wr_h2_caps.py`
- `tests/test_wr_w15_health.py`
- `tests/test_wr_w15b.py`

### Batch 9

76 passed in 1.12s.

- `tests/test_scheduler.py`

### Batch 10

1 failed, 11 passed in 0.24s (loopback bind denied by the sandbox).

- `tests/test_web_saver.py`

### Batch 11

16 passed in 24.90s.

- `tests/validate/test_ope.py`

## Rendered messages

Lengths are Unicode code points in the raw alert and DM body, respectively.

### Live active (439 raw, 501 DM)

```text
BTS health WARN attention:
- [saver_state] Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. The flag for season 2027 read active at this check. A later automatic decision may count a saver as available. Verify the current season's saver and prior entries in MLB before any correction. Use the `bts season-contract status` command with the scheduler's picks directory as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction. (1st observed day)
```

### Live not_earned (443 raw, 505 DM)

```text
BTS health WARN attention:
- [saver_state] Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. The flag for season 2027 read not_earned at this check. A later automatic decision may count a saver as available. Verify the current season's saver and prior entries in MLB before any correction. Use the `bts season-contract status` command with the scheduler's picks directory as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction. (1st observed day)
```

### Historical December 31 (500 raw, 562 DM)

```text
BTS health WARN attention:
- [saver_state] Prior slate 2027-12-31 ET. The dated observation recorded the saver as used. At this check the flag for season 2027 read active. (1st observed day) This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
```

## Stops and rulings

The first stop concerned the round-1 empty-ID assertions. Item 8 authorised their update. The second stop concerned the acknowledgement dispatcher's refusal to replace aliased receipts. Item 9 corrected that contract language. Both were resolved before the final test runs and commit.
