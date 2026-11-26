# H2 contest-state caps

Commit `b12e144c` on `fix/wr-h2`. The six contest-state validation locations use `contest_state.MAX_SEASON_STREAK = 400`. The contest and policy target is 57.

## RED on the base

The first bounded run on `bc789868` failed all eight cases in `tests/test_wr_h2_caps.py`.

| Row | Test | Observed failure |
| --- | --- | --- |
| M1 | `test_m1_dated_cli_selects_delivers_and_commits` | The dated command refused best 58. |
| M2 | `test_m2_switched_fetch_trusts_high_best` | The switched fetch returned exit 2 for 58/58. |
| M3 | `test_m3_dated_high_best`, `test_m3_automatic_high_best_matches_target_actions` | The dated command refused 60/75 and the fetch refused 12/60. |
| M4 | `test_m4_required_pick_survives_reload[dated]` and `[automatic]` | The dated and switched setup paths refused 58/58. |
| M5 | `test_m5_walk_high_dated_state` | The dated setup yielded neither the finale double nor the post-finale skip. |
| K10 | `test_k10_fresh_fetch_replaces_57_and_over_cap_leaves_it_intact` | The fresh 58/58 replacement returned exit 2. |

The new H2 tests and the final season-walk test run passed. The existing K9 check ordering, switch, precedence, and saver regressions passed in the focused groups.

## Existing test changes

| File under `tests/` | Change |
| --- | --- |
| `test_contest_dated_cs4a.py` | Used 401 for the pre-I/O CLI refusal, ordered dated check 5 and 6 inputs, and the invalid automatic fixture in the dated authority control. |
| `test_contest_dated_cs4a_review.py` | Used 401 for the pairwise check 5 input and for the invalid automatic fixture in the saver isolation control. |
| `test_contest_state_best_trust.py` | Replaced the formerly over-cap best of 99 with 401. |
| `test_contest_state_cs1_review.py` | Used 401 for automatic refusal and untrusted legacy best fixtures, and for the fetch/builder/loader refusal and recovery test. Updated fetch error text to 400 and the loader diagnostic to match the new fixture. |
| `test_contest_state_defensive_reads.py` | Used 401 in the automatic-corruption and legacy trust fixtures. |
| `test_season_contract_cs4b1.py` | Used 401 in the over-cap response fixture. |
| `test_season_contract_cs4b1_edges.py` | Used 401 for the fetch refusal and 400 in the message assertion. |
| `test_season_contract_cs4b2.py` | Used 401 in the refused automatic observation fixture. |
| `test_pick_chosen_on_cs5a.py` | Used 401 in the dated best and automatic streak metadata fixtures. |
| `test_saver_cs3.py` | Used 401 in the refused automatic fixture for the dashboard and POST control. |
| `test_wr_c1_history.py` | Used 401 to exercise the exact historical active-streak diagnostic. |
| `scripts/season_walk/test_season_walk.py` | Added a regression for the stored baseline comparison and high-state results. |

## Walk and verification

The walk produced 5,037 records with zero findings. All 4,818 existing records and the year boundary results equal `walk-ea-g-h.json.gz`. The 219 new `dated/58/60/used/0.96` records contain 207 singles, three doubles from September 24 through 26, and nine post-finale skips. The results are in `scripts/audit/season_walk/results/walk-h2.txt` and `walk-h2.json.gz`.

The focused tests passed except `test_web_saver.py::test_do_post_saver_transition_socket_wiring`, which failed at the loopback bind with the known sandbox permission error. The sandbox also denied the requested `nice` priority adjustment. Bounded runs completed. The two known CS2 base failures were left untouched. No unexpected assertion changes were required.

The author brief's file allowlist excluded `CLAUDE.md` and the deploy checklist. Deploy H with H2. Before rolling back past H2, inspect **active and best** in both observations and pending pick metadata. An expired dated observation can still block a good automatic observation. Archive or replace incompatible observations with verified current state when the rollback can represent it, then obtain a fresh selection for incompatible pending metadata. Already committed picks remain scoreable.
