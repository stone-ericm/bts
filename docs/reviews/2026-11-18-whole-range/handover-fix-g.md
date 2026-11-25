# Change set G handover

Commit `d6141728` on `fix/wr-g` applies the monotone skip guard and records its provenance. The commit is local.

## Tests against the base

The first focused run on the base had 24 failures and two passes.

| Contract section | Base test and result |
| --- | --- |
| Must change and rows 1 and 5 | `test_affected_rows_and_independent_clamps`, `test_raw_and_guarded_lookup_grid`, and `test_lookup_edge_shapes` failed at the missing guarded lookup. `test_regenerate_csv` passed and reproduced the committed bytes. |
| Rows 2 and 3 | The four `test_resolver_and_direct_bar_grid` cases failed at the missing guard field or the old skip above a finite bar. |
| Row 4 | `test_unaffected_controls` reached the missing guard field after comparing shared fields to base. |
| Row 6 and decision records | `test_decision_round_trip_and_shadow` failed at the missing write argument. `test_pick_readers_and_fresh_recovered_commits` failed at the base skip for the guarded fixture. |
| Row 7 | `test_guarded_play_and_unchanged_skip_share_census` originally halted on the raw skip versus recorded single. The final test also creates a primary boundary diff on the guarded play and verifies its exclusion from the follow-up count. In the fixture, skip membership falls from five records to four. |
| Journal | The unchanged control passed. The four guarded cases in `test_guard_journal` failed at the missing line. |
| Standing rows | The pick interruption, delivery, and restart tests failed at the base skip for the guarded fixture. `test_corrupt_directory_and_fifo` failed at the missing guard field after resolving the corrupt control. `test_year_boundary` failed at the January synthetic selection, which skipped on base. |

## Verified coverage

- All 341 focused tests passed across the three required groups. All named test paths were present.
- The lookup grid exercised 104,980 cells at three probabilities each for 314,940 calls per implementation. The 1,511 affected cells accounted for 4,533 overrides.
- The resolver grid exercised 109,040 date, streak, saver, and bin combinations at three probabilities each for 327,120 calls per implementation. It compared direct bars at 21,808 date, streak, and saver combinations. It checked 4,701 overrides, including 168 calls on the seven extra capped dates.
- Under Amendment 1, the CSV test exercised all 1,511 rows on their own dates and 56 repeat cases on March 24–30. It exercised the two independent clamps on all 184 inherited doubles. The full grid covered effective days 0–180 and verified unchanged behavior on days 0, 1, 2, 3, 4, and 31.
- The FIFO probe observed the base loader waiting at the unwritten pipe and interrupted it with a 150 ms deadline.

## Season walk

Compared all 4,599 records with the saved baseline by date and state. The 252 skip-to-single changes comprise 229 flagged in-season skips and 23 unflagged preseason skips. Exactly 612 warnings remain, all on in-season all-skip states. Zero other records changed and the boundary results matched. The walk exited 1 as expected. The committed evidence is in `scripts/audit/season_walk/results/walk-g.txt` and `walk-g.json.gz`.

The amplified census fixture and direct recommit byte-preservation assertion were added after the initial RED run. They passed in the final focused run.
