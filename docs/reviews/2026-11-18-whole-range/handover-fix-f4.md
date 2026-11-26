# F4 author handover

Commit `b3d5170b3d951900eafc53ffb99ad127fbc69568` on `fix/wr-f4`.

## Contract results

The tests-first run on `deef7864` produced 39 expected failures and one pass. The first F4 run passed all 40 cases. The final 56-case run also passed, with separate good-file and broken-file grids and the archived replay upgraded to regenerate both receipt variants through the offline scheduler and entry checker. The original RED run used reconstructed receipts for M1 and its combined grid reached the K1/K2 assertion first.

| Row | Base | F4 |
| --- | --- | --- |
| M1 | Both reconstructed entry variants exited 0 without a DM on `[]`. | Both archived replay variants retained the entry state, detail and pick, exited 1 and sent one review DM. |
| M2 | Empty bytes, bad JSON, invalid UTF-8, non-objects, invalid counts, mode-000 and oversized files all left clean days quiet. | All diagnosed with the expected reasons and exit 1. The mode-000 test ran unprivileged. |
| M3 | The no-games and deliberate-skip fixtures had clean exit 0. The combined grid failed its first assertion at the missing key. | Both clean states retained their details with a review head and exit 1 on `[]`. |
| K1 | The null key was absent for a missing target, including a dangling symlink. | A missing target, dangling symlink, acceptable counts, omitted count, ignored metadata and a file of exactly 1 MiB all passed. |
| K2 | All 16 states reached the null-key assertion with their expected base exit and DM. | The good-file grid passed all 16 states, including `skip_state_review`. |
| K3 | Broken files added no reason or notice. The 700-character fixture sent the 833-code-point base DM. | Problem states retained their heads. Normal messages kept detail and pick. The compact case kept the full pick in JSON and sent one 260-code-point DM. |
| K4 | Missing-config control passed. The two decoder cases kept `NO DECISION` but lacked diagnostics. | Missing config still exits 2 with its original DM. The 5,000-digit integer and 12,000-level nesting retain the day detail, diagnose the respective decoder limit, exit 1 and each attempt one DM. |
| K5 | FIFO and directory were ignored. | Both diagnosed without opening the known special file, under a three-second alarm. |
| Interrupt and restart | The interrupted scorer preserved the live bytes. The morning reports lacked advisories. | Simulated interruption before scorer replacement preserved the live bytes. Retry after an unconfirmed send repeated the advisory. Snapshots were unchanged. |
| Midnight ET and year boundary | Prior-season controls were quiet and lacked the new null key. | Both 08:30 ET controls passed. A valid previous-season count was quiet and a bad count remained advisory even though the new-season scorer loaded zero. |
| Failed or unconfirmed send | The captured timeout retained exit 1 and the original stderr. The advisory was absent. | One attempt, same exit and stderr, and no file changes. |

## Regression runs

| Files under `tests/` | Result |
| --- | --- |
| `test_wr_f1_wrong_game.py` | 70 passed in 40.64 s |
| `test_audit_f6_review_probes.py`, `test_day_outcome.py`, `test_day_outcome_entry_sequence.py`, `test_entry_intent.py`, `test_wr_f3_skip_state_review.py` | 238 passed in 11.88 s |
| `test_live_gate_cs6a_review.py`, `test_saved_pick_paths_cs5a.py`, `test_season_contract_cs4b2_edges.py`, `test_wr_a_review_carbon.py`, `test_wr_a_schedule.py` | 174 passed and 4 existing conditional skips in 3.76 s |
| `test_wr_b_contract.py`, `test_wr_b_edges.py`, `test_wr_f4_streak_file.py` | 153 passed in 2.08 s, including all 56 F4 tests |

The script matches passed syntax checks. `scripts/cron-setup-hetzner.sh` invokes the command. The hits in `scripts/final_leaderboard_grab.py` and `scripts/audit/pvr_threshold_bootstrap.py` are unrelated names. Each pytest batch ran in isolation with the 110-second timeout. The shell reported that `nice` could not set the priority and proceeded with each test run.

The first F1 regression run found a second `World.morning()` result inside N11's comparison input. The authorized comparison now asserts and removes the null key from both parsed outputs in that case before hashing. All existing digests passed. The F3 shared helper adds the one null key authorized by Amendment 2. No stop condition remained.

## Captured renders

All lengths are Unicode code points. The fixture messages are ASCII, so these lengths match grapheme counts.

| Day state with `[]` | Head | Head length | Full DM length |
| --- | --- | ---: | ---: |
| `entered` | `BTS day check 2027-04-02: MODEL STREAK FILE NEEDS REVIEW` | 56 | 601 |
| `entered_game_unverified` | `BTS day check 2027-04-02: MODEL STREAK FILE NEEDS REVIEW` | 56 | 612 |
| `deliberate_skip` | `BTS day check 2027-04-02: MODEL STREAK FILE NEEDS REVIEW` | 56 | 557 |
| `no_games` | `BTS day check 2027-04-02: MODEL STREAK FILE NEEDS REVIEW` | 56 | 554 |
| `research_commit` | `BTS day check 2027-04-02: MODEL STREAK FILE NEEDS REVIEW` | 56 | 586 |
| `delivered_under_research` | `BTS day check 2027-04-02: DELIVERED UNDER RESEARCH` | 50 | 597 |
| `intent_unknown` | `BTS day check 2027-04-02: INTENT UNKNOWN` | 40 | 606 |
| `schedule_unverifiable` | `BTS day check 2027-04-02: SCHEDULE UNVERIFIABLE` | 47 | 590 |
| `no_decision` | `BTS day check 2027-04-02: NO DECISION` | 37 | 616 |
| `skip_not_deliberate` | `BTS day check 2027-04-02: SKIP NOT DELIBERATE` | 45 | 587 |
| `commit_not_scoreable` | `BTS day check 2027-04-02: COMMIT NOT SCOREABLE` | 46 | 595 |
| `contradiction` | `BTS day check 2027-04-02: CONTRADICTION` | 39 | 585 |
| `entered_late` | `BTS day check 2027-04-02: ENTERED LATE` | 38 | 601 |
| `not_entered` | `BTS day check 2027-04-02: NOT ENTERED` | 37 | 609 |
| `entry_unverifiable` | `BTS day check 2027-04-02: ENTRY UNVERIFIABLE` | 44 | 676 |
| `skip_state_review` | `BTS day check 2027-04-02: SKIP STATE REVIEW` | 43 | 810 |

The normal notice for `not a JSON object` is 472 code points including the leading space. The archived batter-only and fully bound alerts measured 616 and 605 respectively.

| Reason inserted in normal notice | Notice length |
| --- | ---: |
| `not a JSON object` | 472 |
| `not valid JSON` | 469 |
| `cannot be read (UnicodeDecodeError)` | 490 |
| `streak is not a non-negative integer` | 491 |
| `exceeds 1 MiB` | 468 |
| `JSON exceeds decoder limits (ValueError)` | 495 |
| `JSON exceeds decoder limits (RecursionError)` | 499 |
| `not a regular file` | 473 |
| `cannot be read (PermissionError)` | 487 |

**Archived fully bound DM (605)**

> BTS day check 2027-03-24: MODEL STREAK FILE NEEDS REVIEW. The entry was confirmed in 1 of 1 checks before the cutoff. Pick: Batter 1. Also: The local model streak file (streak.json) needs review (not a JSON object). Grading or replay may be affected. Preserve the files in place during triage. Follow B8b item 2 in docs/ops/2027-season-start.md before the next affected job. Give the BTS lead this alert, the host and ET run time, and any grading or reconcile errors, receipt paths and health alerts. If the lead session is unavailable, escalate through Eric. The 08:00 ET correction cutoff still applies.

**F3 problem head and appended notice (810)**

> BTS day check 2027-04-02: SKIP STATE REVIEW. The policy recorded a skip for 2027-04-02 using streak 30 and policy saver availability unavailable. The installed dated observation for 2027-04-02 records streak 1 and policy saver availability unavailable. Check MLB for unsettled entries and verify the current streak, season best and saver. Also: The local model streak file (streak.json) needs review (not a JSON object). Grading or replay may be affected. Preserve the files in place during triage. Follow B8b item 2 in docs/ops/2027-season-start.md before the next affected job. Give the BTS lead this alert, the host and ET run time, and any grading or reconcile errors, receipt paths and health alerts. If the lead session is unavailable, escalate through Eric. The 08:00 ET correction cutoff still applies.

**Compact DM (260)**

> BTS day check 2027-04-02: NOT ENTERED. Local model streak file needs review (not a JSON object). Preserve the files. See day-outcome JSON in cron.log and B8b item 2 in docs/ops/2027-season-start.md before the next affected job. Escalate through Eric if needed.

The compact fixture's combined text would have been 1305 code points. The transport accepted the compact DM, and stdout retained the full 700-character pick.
