# K1 handover

Commit `d4017e6e` on `fix/wr-k1`, based on `22a27d29`. Git used the local host identity `Eric <eric@studio.local>` for the committer.

## Test results

The first complete baseline run had 43 failed and 51 passed. The final K1 run had 97 passed. The final exact-body checks and real E4, E7 and E8 new-key tests were added after the first baseline run. The contract-check logs cover the base real E4 and E7 sequences. The E8 new-key rerun is a passing control for the runner’s existing deduplication.

| Row | Base | K1 |
| --- | --- | --- |
| E1 | Absent switch exceeded the limit | Passed |
| E2 | Revoked switch exceeded the limit | Passed |
| E3 | 15 failures across the E1–E3 inventory. Other cases in that run were passing controls. Malformed date and 701-digit count failures were reproduced in the contract check. | All 56 inventory cases passed, including the malformed date and large count. The expanded exact-body checks were added after the initial baseline run. |
| E4 | Contest-state receipt used the wall date and suppressed the next slate | Passed, including the real loader and CLI |
| E5 | Other three sources used the wall date and suppressed the next slate | Passed for all three |
| E6 | All four sources failed over New Year | Passed |
| E7 | Base sent a new old-slate alert after midnight | Passed for all sources, including the real observation change |
| E8 | Same-source contest-state and late-delivery digests sent | Both suppressed with receipt coverage. New-key digest sent once across two runs. |
| E9 | Base suppressed the late D alert and the repeat D+1 alert | Passed at both boundaries. All three send under the single-record rule. |
| Missed-pick wait and final fallback | Both used the wall date after midnight | Both recorded the previous slate |
| S1 | Passed | Passed |
| S2 | Passed | Passed |
| S3 | Passed | Passed |
| S4 | Passed against the four recorded base bodies | Passed |
| S5 | Passed | Passed |
| S6 | Passed | Passed |
| S7 | Passed for missing, empty and invalid dates at contest alert dispatch | Passed |
| S8 | Journal-before-send control passed at the five call sites | Passed. Diff retains all five journal statements. |
| S9 | Targeted base checks passed in the contract check | C2, morning, stale-reading and runner files passed in the batches below |
| Standing, interrupted send and restart | Passed | Passed |
| Standing, malformed JSON, FIFO and directory receipt | Passed | Passed |
| Standing, midnight and year boundary | Failed as above | Passed |
| Standing, failed and unconfirmed sends | Passed | Passed |

The dated non-object error is intercepted by the loader’s file reader before dated selection. The required-missing error is shadowed by the live authority refusal. Both model-fallback errors were exercised with the requirement off. These reachability limits match the contract check.

## Existing files

The six bounded runs passed with 1,683 passed and 12 skipped. Batch 6 took 36.30 seconds. The other batches took under six seconds each. The host denied the requested priority change and the tests ran. The sandbox process query did not expose a process list for checking Victoria 3.

### Batch 1 · 223 passed

- `tests/health/test_alert.py`
- `tests/test_alert_status_callers_cs6b2.py`
- `tests/test_late_delivery_guard.py`
- `tests/test_postponed_delivery_guard.py`
- `tests/test_stale_delivery_cs4c.py`
- `tests/test_wr_c2_carbon_v2.py`
- `tests/test_wr_f2_saver_flag.py`

### Batch 2 · 605 passed, 4 skipped

- `tests/health/test_alert_status_cs6b2.py`
- `tests/test_alert_status_cs6b2_carbon.py`
- `tests/test_morning_state_cs6b1.py`
- `tests/test_saved_pick_paths_cs5a.py`
- `tests/test_stale_delivery_cs4c_carbon.py`
- `tests/test_wr_c2_r2_carbon.py`
- `tests/test_wr_g_guard.py`

### Batch 3 · 289 passed

- `tests/health/test_alert_status_cs6b2_carbon_r2.py`
- `tests/test_contest_dated_cs4a_review.py`
- `tests/test_morning_state_cs6b1_carbon.py`
- `tests/test_saver_cs3_review.py`
- `tests/test_stale_delivery_cs4c_review.py`
- `tests/test_wr_c2_receipt_identity.py`

### Batch 4 · 230 passed

- `tests/health/test_alert_status_cs6b2_review.py`
- `tests/test_e3_missed_pick_alert.py`
- `tests/test_morning_state_cs6b1_review.py`
- `tests/test_scheduler.py`
- `tests/test_thin_slate_checks.py`
- `tests/test_wr_c2_recovery.py`

### Batch 5 · 126 passed

- `tests/health/test_h6_resurface.py`
- `tests/test_health_endpoint.py`
- `tests/test_pickpath_review_pp1.py`
- `tests/test_scheduler_decision_record_integration.py`
- `tests/test_wr_b_edges.py`
- `tests/test_wr_c3_r2_entered_pick_blockers.py`

### Batch 6 · 210 passed, 8 skipped

- `tests/health/test_runner.py`
- `tests/test_incident_2026_08_30.py`
- `tests/test_pickpath_review_pp2.py`
- `tests/test_season_contract_cs4b2_edges.py`
- `tests/test_wr_c1_history.py`
- `tests/test_wr_e_texts.py`

## Rendered messages

Lengths count Unicode code points. The failed receipt in these examples had one CRITICAL.

### Absent switch

Without banner · 536 code points

```text
BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Live decision refused: season 2027 switch is absent.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

With banner · 620 code points

```text
⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.

BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Live decision refused: season 2027 switch is absent.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

### Revoked switch

Without banner · 537 code points

```text
BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Live decision refused: season 2027 switch is revoked.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

With banner · 621 code points

```text
⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.

BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Live decision refused: season 2027 switch is revoked.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

### Saved-pick refusal

Without banner · 818 code points

```text
BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Saved pick for 2027-03-24 was not delivered because it was chosen on dated (effective date 2027-03-24, active streak 11, best streak 15, saver used). The live state is dated (effective date 2027-03-24, active streak 12, best streak 15, saver used). A later scheduled check or fallback refresh may choose a replacement, if one remains.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

With banner · 902 code points

```text
⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.

BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: Saved pick for 2027-03-24 was not delivered because it was chosen on dated (effective date 2027-03-24, active streak 11, best streak 15, saver used). The live state is dated (effective date 2027-03-24, active streak 12, best streak 15, saver used). A later scheduled check or fallback refresh may choose a replacement, if one remains.
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

### 100,000-character error

Without banner · 884 code points

```text
BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

With banner · 968 code points

```text
⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.

BTS health CRITICAL alert(s):
- [contest_state] No pick was sent for 2027-03-24 at this check. Reason: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...
See the scheduler journal for this refusal. To inspect current state, run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. For live repair, use the season-start checklist (B8a). Before changing state for a new slate, verify in MLB that prior entries settled or were absent. Use a later official pick only while its entry window is open.
```

## Completion

The staging and commit approvals were granted separately. The staged whitespace check passed. The commit includes the nine patch files. This handover remains outside the commit. Work reached the approved commit without a contract stop.
