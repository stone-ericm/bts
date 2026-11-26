# Change set E round 2

Commit `256ac9580d32546628945b9a8f0053ca8c89f80e` is the sole new commit on `1e03ab40`. It includes the requested co-author trailer. The branch was not pushed.

## Amendment 2

1. The morning warning now conditions the dated observation on prior entries being settled or absent. `test_stale_warning_maximum_detail[morning]` was RED on the exact message comparison. The final measured lengths, including the actual DM envelope and the longest accepted detail fixture, were **882 characters for morning** and **881 for delivery**, both under 900. The RED log shows a preliminary morning length expectation of 879 that was corrected to 882. Its RED failure occurred at the earlier exact message assertion.
2. The display parser accepts only probabilities and bars in the closed range from 0 to 1. All ten cases in `test_out_of_range_skip_numbers_are_missing` were RED. The four endpoint cases in `test_skip_number_edges_are_valid` passed on the starting head.
3. Skip reasons are trimmed after line breaks are converted to spaces. The blank reason and padded nonblank reason cases in `test_skip_surfaces` were RED.
4. `test_skip_surfaces` pins K1 precedence over K5 and over a finite bar, the unverified label in K2, and HTML escaping of the team and reason. `test_run_single_check_skip_journal` asserts the complete K1, K3 and K4 lines from the scheduler. These coverage cases passed on the starting head.
5. The two legacy-summary tests now name the absence of a probability comparison, and the scheduler comment describes checks that produce predictions without a pick. The assertions already passed on the starting head.
6. The commit is scoped to the four files listed below. The handover remains in the worktree.

## Diff stat against `db52bd02`

The full comparison includes the intervening review and documentation commits. It has 77 files changed, 9,526 insertions, and 33 deletions. The source and test portion is also the entire round-2 commit.

```text
 src/bts/scheduler.py                    | 16 ++++-----
 tests/test_morning_state_cs6b1.py       |  3 +-
 tests/test_scheduler_skip_visibility.py |  4 +--
 tests/test_wr_e_texts.py                | 59 +++++++++++++++++++++++++++++++--
 4 files changed, 69 insertions(+), 13 deletions(-)
```

## Verification

RED on `1e03ab40` — 13 failed and 42 passed in `.codex-review/wr-e-r2/red.log`.

GREEN before commit — all five required groups passed in order. Output is in `.codex-review/wr-e-r2/green.log`.

| Test group | Passed |
| --- | ---: |
| `test_wr_e_texts.py`, `test_lightgbm_optional.py`, `test_web_saver.py` | 68 |
| `test_scheduler_skip_visibility.py`, `test_scheduler_tail_objective.py` | 48 |
| `test_stale_delivery_cs4c.py` | 62 |
| `test_morning_state_cs6b1.py` | 61 |
| `tests/scripts/season_walk/` | 18 |

The initial first-group run hit the sandbox's loopback bind restriction in the web saver socket test. The full group passed on the approved rerun with loopback access.
