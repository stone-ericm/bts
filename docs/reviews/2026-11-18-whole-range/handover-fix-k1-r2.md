# K1 round 2 handover

Round 2 is committed as `24962e12` on `fix/wr-k1`, based on Amendment 3 at `3329d359`. Staging and commit were approved separately. The staged whitespace check passed. Git used the local committer identity `Eric <eric@studio.local>`. The production diff against `d4017e6e` is empty. This handover is outside the commit.

## Stop and resumption

`tests/test_wr_f2_saver_flag.py::test_e7_other_b8a_text_matches_round_two` stopped the initial run because it hashes the whole B8a row after restoring the F2 text. The exact K1 wording changed the normalized hash from `21b6dd612d9019555a4436b0b3f68b5493d5751d466bda8a32dfabf5a156739d` to `4093244611f9b65adb7151e905a01210f165a82415f9bd73fb8aabef714e144e`. Batch 7 had 284 passes and that one failure, and batch 8 was pending at the stop.

Amendment 3 authorized restoring the old K1 sentence in that test before hashing. The test now reverses both the F2 and K1 edits and still uses the original expected hash. The kickoff's Amendment 2 records the corrected base comparison.

## New tests before and after the checklist edit

| Test | Before | After |
| --- | --- | --- |
| `test_scheduler_alert_dates`, 12 scenarios each on the same day, at June midnight and at New Year | 36 passed | 36 passed |
| `test_operator_late_start[absent]` | Passed | Passed |
| `test_operator_late_start[filed-between-checks]` | Passed | Passed |
| `test_operator_late_start[after-last-check]` | Passed | Passed |
| `test_operator_delivery_refusals` | Passed | Passed |
| `test_operator_overnight_fallback` | Passed | Passed |
| `test_b8a_filing_after_first_check_refusal` | Failed at the missing new wording | Passed |

The tests import the review's caller and operator replays and run their scenarios under pytest. The pre-edit run took 1.25 seconds. The caller tests assert the alert sources, forwarded date and persisted receipt date. The checklist test reads the B8a row and checks for the exact new wording and the absence of the old sentence.

## Scratch date checks

Each run used a fresh scratch package copy, one changed source line and the listed June midnight test. Each failed at the forwarded-date assertion with `chosen = None` and slate `2027-06-14`. The original source passed every case.

| `scheduler.py` line | Path | Scratch change | Failing case | Time |
| --- | --- | --- | --- | --- |
| 1362 | Saved pick refused after dated correction | `slate_date=None` | `saved-midnight` | 0.85s |
| 1355 | Delivery live-state load refused | `slate_date=None` | `loader-midnight` | 0.84s |
| 2029 | Lineup check refused | `slate_date=None` | `check-midnight` | 0.84s |
| 3351 | In-loop fallback refused | `slate_date=None` | `in_loop-midnight` | 0.86s |
| 3504 | Final fallback refused | `slate_date=None` | `final-midnight` | 0.85s |
| 328 | Missed-pick wait calls the alert | `slate_date=None` | `missed_live-midnight` | 0.83s |
| 2570 | Late-delivery alert date | Parse `None` to omit the dispatcher date | `late_initial-midnight` | 0.84s |
| 2633 | Void-game alert date | Parse `None` to omit the dispatcher date | `void-midnight` | 0.84s |

## B8a wording

Before

> **Enter it in the morning, before the scheduler's first lineup check.**

After

> **Enter it in the morning, before the scheduler's first lineup check, when possible.** If that check refused before choosing a pick, confirm this is still today's slate in ET and that MLB has settled every prior entry or confirmed its absence after its window closed. File the verified dated observation between checks while another lineup check or fallback remains.

A byte comparison confirmed this was the sole checklist edit.

## Matched test files after the Amendment 3 edit

The brief's search also found compiled Python caches. All 43 runnable source files from the search are listed below, with 1,916 passed and 12 skipped. Tests ran in serial bounded batches with ET and the requested priority. The host printed the priority warning and continued.

### Batch 1 — 411 passed in 0.76s

- `tests/health/test_alert.py`
- `tests/health/test_alert_status_cs6b2.py`
- `tests/health/test_alert_status_cs6b2_carbon_r2.py`
- `tests/health/test_alert_status_cs6b2_review.py`
- `tests/health/test_h6_resurface.py`
- `tests/health/test_runner.py`

### Batch 2 — 122 passed in 1.15s

- `tests/test_alert_status_callers_cs6b2.py`
- `tests/test_alert_status_cs6b2_carbon.py`
- `tests/test_contest_dated_cs4a_review.py`
- `tests/test_e3_missed_pick_alert.py`
- `tests/test_health_endpoint.py`
- `tests/test_incident_2026_08_30.py`

### Batch 3 — 166 passed, 8 skipped in 1.85s

- `tests/test_late_delivery_guard.py`
- `tests/test_morning_state_cs6b1.py`
- `tests/test_morning_state_cs6b1_carbon.py`
- `tests/test_morning_state_cs6b1_review.py`
- `tests/test_pickpath_review_pp1.py`
- `tests/test_pickpath_review_pp2.py`

### Batch 4 — 251 passed, 4 skipped in 2.50s

- `tests/test_postponed_delivery_guard.py`
- `tests/test_private_mode_transport.py`
- `tests/test_saved_pick_paths_cs5a.py`
- `tests/test_saver_cs3_review.py`
- `tests/test_scheduler.py`
- `tests/test_scheduler_decision_record_integration.py`

### Batch 5 — 187 passed in 2.37s

- `tests/test_season_contract_cs4b2_edges.py`
- `tests/test_stale_delivery_cs4c.py`
- `tests/test_stale_delivery_cs4c_carbon.py`
- `tests/test_stale_delivery_cs4c_review.py`
- `tests/test_thin_slate_checks.py`
- `tests/test_wr_b_contract.py`

### Batch 6 — 397 passed in 6.37s

- `tests/test_wr_b_edges.py`
- `tests/test_wr_c1_history.py`
- `tests/test_wr_c2_carbon_v2.py`
- `tests/test_wr_c2_r2_carbon.py`
- `tests/test_wr_c2_receipt_identity.py`
- `tests/test_wr_c2_recovery.py`

### Batch 7 — 285 passed in 36.41s

- `tests/test_wr_c3_r2_entered_pick_blockers.py`
- `tests/test_wr_e_texts.py`
- `tests/test_wr_f2_saver_flag.py`
- `tests/test_wr_f4_streak_file.py`
- `tests/test_wr_g_guard.py`
- `tests/test_wr_k1_r2.py`

### Batch 8 — 97 passed in 1.00s

- `tests/test_wr_k1_refusal_dm.py`

## Note (bts-wr-fresh15, 2026-11-26 13:47)
The K1 commits were first made in a bundle clone without the repository identity (`Eric <eric@studio.local>`) and pushed as `fix/wr-k1`. They were rewritten with the configured identity, trees unchanged, and pushed as `fix/wr-k1b`: c7ba2379 -> 8f3f7106, 22a27d29 -> 1dae0550, d4017e6e (round-1 code) -> 34981a1a, 01896138 -> 80e2904e, fdae83c9 -> 497121a1, 3329d359 -> 5c795488, 24962e12 (round-2 tests and B8a) -> bf3d4f02. Earlier records name the old ids.
