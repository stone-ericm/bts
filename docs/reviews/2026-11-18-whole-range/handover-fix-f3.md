# F3 author handover

Commit `dee14603` on `fix/wr-f3`, based on `966069b1`.

## New tests before and after

The base run had 47 failures and 36 passes. The implementation run had 83 passes.

| Row | Base | Implementation |
| --- | --- | --- |
| M1 | Failed. Deliberate skip, exit 0, no morning DM. | Passed. Installed streak difference, exit 1, one morning DM. |
| M2 | Failed. Deliberate skip, exit 0, no morning DM. | Passed. Revocation, exit 1, one morning DM. |
| M3 | Failed. Deliberate skip, exit 0, no morning DM. | Passed. Saver availability difference, exit 1, one morning DM. |
| M4 | Failed. Deliberate skip, exit 0, no morning DM. | Passed. Archived streak difference and revocation, exit 1, one morning DM. |
| K1 | Passed. | Passed. |
| K2a | Passed. | Passed. |
| K2b | Passed. | Passed. |
| K2c | Passed. | Passed. |
| K2d | Passed. | Passed. |
| K3 | Passed. | Passed. |
| K4 | Passed for all base outcomes in the matrix. | Passed. |
| K5 | All bad-input-only, unusable skip, and base-reader assertions passed. The 39 mixed-evidence cases failed at the assertion that valid evidence elsewhere still counts. | Passed. |
| K6 | Passed, including send failure and the could-not-run DM. | Passed. |
| K7 | Passed. | Passed. |
| Revoke, re-enable and fetch before skip | Passed. | Passed. |
| Newer archive with matching installed observation | Passed. | Passed. |
| Evidence over 64 KiB | Passed with the oversized file alone for current, archive and switch. Failed when valid evidence was added alongside it. | Passed in both configurations. |
| ET timestamp overflow | Passed with the overflowing switch alone. Failed when a valid dated mismatch was added. | Passed in both configurations. |

The equal-instant archive case with a matching archive passed on base. The other two archive-selection cases and the research-mode dated case failed on base. All four passed with the implementation. The scheduler cases reuse the offline scheduler harness from the historical Carbon checks. The report tests compare the picks directory's file bytes before and after reporting.

## Existing tests after the change

| Test file | Result |
| --- | --- |
| `tests/test_audit_f6_review_probes.py` | 3 passed |
| `tests/test_day_outcome.py` | 56 passed |
| `tests/test_day_outcome_entry_sequence.py` | 35 passed |
| `tests/test_entry_intent.py` | 52 passed |
| `tests/test_live_gate_cs6a_review.py` | 8 passed |
| `tests/test_saved_pick_paths_cs5a.py` | 93 passed, 4 skipped |
| `tests/test_season_contract_cs4b2_edges.py` | 20 passed |
| `tests/test_wr_a_review_carbon.py` | 25 passed |
| `tests/test_wr_a_schedule.py` | 24 passed |
| `tests/test_wr_b_contract.py` | 24 passed |
| `tests/test_wr_b_edges.py` | 73 passed |
| `tests/test_wr_f1_wrong_game.py` | 70 passed |

The first batch finished in 19.58 seconds with 417 passes and 4 skips. The second batch finished in 40.71 seconds with 70 passes. The sandbox denied the `nice` priority adjustment. Pytest continued under the requested timeout. Test-run logs for the new file were captured under `/tmp`.

## Rendered details and DMs

### M1

**Detail**

The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 1 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

**DM**

BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 1 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M2

**Detail**

The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The season switch records a revocation on 2027-03-24 at 16:30 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

**DM**

BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The season switch records a revocation on 2027-03-24 at 16:30 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M3

**Detail**

The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 30 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

**DM**

BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 30 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M4

**Detail**

The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. An archived dated observation for 2027-03-24 records streak 1 and policy saver availability available. The season switch records a revocation on 2027-03-24 at 16:31 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

**DM**

BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. An archived dated observation for 2027-03-24 records streak 1 and policy saver availability available. The season switch records a revocation on 2027-03-24 at 16:31 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

## Stop conditions

No existing test edits or contract stop conditions. The pre-existing untracked contract-check scratch directory was left on disk.
