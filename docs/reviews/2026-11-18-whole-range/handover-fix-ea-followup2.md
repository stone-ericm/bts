# WR-Ea second follow-up

Commit `3f7257790862e5d168cbbacf97a22a730c60bd87` on `fix/wr-ea` contains only the three authorized files. No push was made.

| Item | Tests that first failed |
| --- | --- |
| 12. Delivery warning | `test_stale_warning_maximum_detail[delivery]` and all four `test_warning_date_is_emission_date` cases, including sends that crossed midnight. |
| 13. Whole words in stored skip text | `test_skip_stored_text_surfaces[longer-word]` and `test_run_single_check_skip_journal[longer-word]`. The existing phrasing with whole words still rendered as “under the.” |
| 14. B8a wording | `test_b8a_follow_up` failed on the starting commit because the checklist already had the new wording and the test still expected the old wording. It passed after both expectations were updated. The test pins the between-checks instruction in C28 and the date and settlement conditions in C29. |

All 254 focused tests passed in two sequential runs. The sandbox denied the requested `nice` priority adjustment. Pytest still ran with the requested ET timezone and 110-second bound.

The maximum-detail delivery and morning warnings measured **896** and **889** characters respectively, including the DM envelopes. Both used the 177-character maximum detail.

| C31 scratch restoration | Test that detected it |
| --- | --- |
| Restore the old delivery write sentence | `test_stale_warning_maximum_detail[delivery]` |
| Restore the plain `.replace("below the", "under the")` | `test_skip_stored_text_surfaces[longer-word]` |

Both restorations were tested independently and reverted immediately.

The season walk ran on the tree that was committed. It completed **4,599 runs**, with **229 K3**, **612 K4** and **zero problems**. The decoded results matched `walk-e.json.gz` exactly. The earlier baseline matched on day, state, action, bars and day counts. The saved results required no update.

No stop condition was reached.
