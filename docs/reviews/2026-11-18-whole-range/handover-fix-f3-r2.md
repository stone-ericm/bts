# F3 round 2 handover

Commit `490c5381` on `fix/wr-f3`, based on `439dabef`. No contract blocker or test failure remained at handover.

## Before and after

Before is the round-1 implementation at `439dabef` with the new tests added. The F3 file went from 85 passed and 7 failed to 92 passed.

| Row | Before | After |
|---|---|---|
| M1 | Pass, installed streak reason | Pass, exit 1 and one DM |
| M2 | Pass, revocation reason | Pass, exit 1 and one DM |
| M3 | Pass, installed saver reason | Pass, exit 1 and one DM |
| M4 | Pass, archive and revocation reasons | Pass, exit 1 and one DM |
| K1 | Pass | Pass, clean |
| K2a | Pass, including the newer-archive control | Pass, clean |
| K2b | Pass | Pass, failed installation leaves the report clean |
| K2c | Pass, including the revoke and re-enable control with no dated archive | Pass, no revocation reason |
| K2d | Pass | Pass, clean with matching retained evidence |
| K2e | Pass, pinned round-1 behavior | Pass, archive reason only, exit 1 and one DM |
| K3 | Pass | Pass, `skip_not_deliberate` |
| K4 | Pass | Pass, other outcomes retain their expected results and evidence reads stay outside those paths |
| K5 | Pass in the existing cases | Pass, including malformed skip fields and switch timestamp overflow |
| K6 | Pass | Pass, existing exit and DM behavior |
| K7 | Pass | Pass, ET year boundary |
| E1, year 1 installed | Fail, dated review | Pass, clean |
| E1, year 1 archive | Fail, dated review | Pass, clean |
| E1, year 9999 installed | Fail, dated review | Pass, clean |
| E1, year 9999 archive | Fail, dated review | Pass, clean |
| E1, valid differing archive beside overflowing matching archive | Fail, clean | Pass, archive and revocation reasons, exit 1 and one DM |
| E2, module docstring | Fail | Pass, required wording |
| E2, CLAUDE.md | Fail | Pass, required wording |
| Byte limit | Pass, pinned round-1 behavior | Pass, at most 65,537 bytes read and clean report |

K2e reproduces the reviewer's 09:00 dated streak 1, 09:01 revoke, 09:02 enable and fetch of automatic streak 30, and 16:10 skip on 30. The archive is retained for the morning report.

## Existing test files

| File under `tests/` | Passed | Skipped |
|---|---:|---:|
| `test_audit_f6_review_probes.py` | 3 | 0 |
| `test_day_outcome.py` | 56 | 0 |
| `test_day_outcome_entry_sequence.py` | 35 | 0 |
| `test_entry_intent.py` | 52 | 0 |
| `test_live_gate_cs6a_review.py` | 8 | 0 |
| `test_saved_pick_paths_cs5a.py` | 97 | 4 |
| `test_season_contract_cs4b2_edges.py` | 20 | 0 |
| `test_wr_a_review_carbon.py` | 25 | 0 |
| `test_wr_a_schedule.py` | 24 | 0 |
| `test_wr_b_contract.py` | 24 | 0 |
| `test_wr_b_edges.py` | 73 | 0 |
| `test_wr_f1_wrong_game.py` | 70 | 0 |

The sandbox denied the requested `nice` priority adjustment. The bounded pytest commands continued.

## Rendered detail and DMs

Each row returned `skip_state_review` with exit 1 and one morning DM. The scheduler and CLI tests checked that the report left the file snapshot identical.

### M1

Detail

> The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 1 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

DM to `operator.bsky.social`

> BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 1 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M2

Detail

> The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The season switch records a revocation on 2027-03-24 at 16:30 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

DM to `operator.bsky.social`

> BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The season switch records a revocation on 2027-03-24 at 16:30 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M3

Detail

> The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 30 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

DM to `operator.bsky.social`

> BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-03-24 records streak 30 and policy saver availability available. Check MLB for unsettled entries and verify the current streak, season best and saver.

### M4

Detail

> The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. An archived dated observation for 2027-03-24 records streak 1 and policy saver availability available. The season switch records a revocation on 2027-03-24 at 16:31 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.

DM to `operator.bsky.social`

> BTS day check 2027-03-24: SKIP STATE REVIEW. The policy recorded a skip for 2027-03-24 using streak 30 and policy saver availability unavailable. An archived dated observation for 2027-03-24 records streak 1 and policy saver availability available. The season switch records a revocation on 2027-03-24 at 16:31 ET. Check MLB for unsettled entries and verify the current streak, season best and saver.
