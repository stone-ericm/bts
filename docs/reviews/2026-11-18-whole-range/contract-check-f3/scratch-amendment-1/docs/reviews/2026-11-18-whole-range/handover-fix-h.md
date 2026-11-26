# Change set H handover

Commit `813935bccb2fd3fe2fd4596020e01077630a992e` on `fix/wr-h`.

## RED evidence

Behavior tests ran before implementation on `aee4b77d`. Test names below are in `tests/test_wr_h_longest.py` unless otherwise noted.

| Row | Test that failed on the base |
| --- | --- |
| H1 | `test_h1_h2_h3_resolver_grid[days0-True-True-single]` |
| H2 | `test_h1_h2_h3_resolver_grid[days1-True-True-double]` |
| H3 | `test_h1_h2_h3_resolver_grid[days2-False-True-single]` and `[days3-True-False-single]` |
| H4 | `test_h4_bad_artifacts_force_single`, all 11 artifact cases |
| H5 | `test_h5_live_selection_save_and_commit`, all eight orchestrator cases. The first scheduler run exposed a heartbeat stub signature error. With the stub fixed, all eight scheduler cases were rerun on archived `aee4b77d` and failed on the action assertion. |
| H6 | `test_h6_display_before_artifacts`, shipped, missing and raising loader cases |
| H7 | `test_h7_finalized_decision`, writer returned None |
| H8 | `test_h8_saved_metadata_and_delivery[cached]` and `[restart]`, missing finalized records. The unconfirmed-send record case also failed. The failed-send check passed on the base. |
| H9 | `tests/simulate/test_tail_policy.py::TestObjectiveSwitch::test_no_days_or_finished_streak_stay_with_base_policy` |

In `tests/scripts/season_walk/test_season_walk.py`, the injected skip and single were both accepted by the original checker, and the actual finale choice skipped.

## Verification

- The focused suites passed in two bounded runs of 111 and 399 tests. The sandbox denied the requested niceness change and the test processes ran to completion.
- The targeted parity grid is in `test_legacy_parity_against_archived_base`. It compared every shipped streak/raw-day/saver cell below the boundary across the probability grid and used separate grids for artifact, best, calendar and execution branches. Each process compared 640,451 contexts including all 1,511 G guard cells. Raw counts covered 0–271 and effective counts covered 0–180. The test printed and asserted both `bts.__file__` paths. The archived `0a2bec10` and worktree decision/bar digests both matched `5ca5e0fe6fec880363ef2cee5925c81750485594c59c8a97621377d19973f35d`. The same archived process parsed a finalized H record, normalized its objective to `unknown`, and found it scoreable.
- The complete season walk exited 1 and matched all 4,599 original G records, all 612 findings and the year-boundary checks. The 219 added rows produced 207 singles including 23 synthetic preseason plays, doubles on September 24–26, and nine post-finale reach-57 skips. All 210 plays had valid `chosen_on` and matching selected slots. The added rows had zero findings. A rerun on the commit matched the saved JSON byte for byte after decompression and matched the captured portion of the text result.
- The walk results are `scripts/audit/season_walk/results/walk-h.txt` and `walk-h.json.gz`.

No unresolved contract discrepancy surfaced after applying Amendment 1. The brief calls the existing assertion exception H9, and A10 also authorizes the specified G grid edit. The full gate and review are pending.
