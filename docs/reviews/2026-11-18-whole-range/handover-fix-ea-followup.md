# WR-Ea follow-up handover

Commit `988708df1a4e2c202c9cc091e0aa952e39709e40` on `fix/wr-ea` contains the four authorized files. No push was made.

## Tests first

The first run against scheduler source at `c511c724` produced 14 failures and 49 passes. All names below are from `tests/test_wr_e_texts.py`.

| Item | Tests that first failed |
| --- | --- |
| 8. Delivery write condition | `test_stale_warning_maximum_detail[delivery]` |
| 9. Saver wording for both warnings | `test_stale_warning_maximum_detail[delivery]` and `test_stale_warning_maximum_detail[morning]` |
| 10. Streak wording for legacy summaries | `test_skip_streak_basis[missing]`, `test_skip_streak_basis[unknown]` and eight `test_skip_surfaces` cases |
| 11. Wording in stored skip text | `test_skip_stored_text_surfaces` and `test_run_single_check_skip_journal[stored-reason]` |

`test_b8a_follow_up` passed with the previously applied checklist wording for C22 through C26. The six requested focused targets passed, with 252 distinct tests. The runs used the requested ET timezone, pytest arguments, and 110-second bounds. The sandbox denied the `nice` priority adjustment but ran pytest sequentially.

## Warning measurements

The maximum-detail tests checked actual DM bodies, including the envelope. Delivery measured **892 characters** and morning measured **889 characters**. Each used the 177-character maximum detail.

## C27 restorations

| Scratch restoration | Test that failed |
| --- | --- |
| Old delivery write condition | `test_stale_warning_maximum_detail[delivery]` |
| Old delivery saver date | `test_stale_warning_maximum_detail[delivery]` |
| Old morning saver date | `test_stale_warning_maximum_detail[morning]` |
| Old streak wording for a summary with no basis | `test_skip_streak_basis[missing]` |
| Drop the stored-text replacement | `test_skip_stored_text_surfaces` |

Each restoration was run in isolation and reverted immediately.

## Season walk

The committed tree completed 4,599 runs with zero problems. The 841 formerly flagged records split into 229 K3 and 612 K4. Decoded output matched `scripts/audit/season_walk/results/walk-e.json.gz` in full. The earlier baseline matched on day, state, action, bar and day counts. The round-2 handover described the comparison. The season-walk README supplied the direct command, `.venv/bin/python -m scripts.audit.season_walk.walk --out <file.json>`. The saved results required no update.

No stop condition was reached.
