# G2 fourth author handover

Base `dbb78b1`, branch `g2-fix`. No commit. Only `src/bts/data/build.py` and `tests/test_data_build_empty.py` changed; all other product and test files untouched.

## Diff summary
- Restored the parent's two `.get` extraction lines and membership test verbatim. The entire loop block supplied in kickoff-4 was checked for exact inclusion in the final source; there is no exception handler around it.
- Renamed bookkeeping to `untyped_files`. Only excluded, hashable values that are not nonblank strings are logged and recorded. Without a requested-type feed, recorded untyped files cause the existing clear ValueError; otherwise the normal build proceeds.
- Updated the docstring: missing/hashable untyped metadata is distinguished from other malformed shapes, which retain their AttributeError/TypeError behavior. Missing directory and readable excluded-type no-ops remain intact.
- Split test inputs into nine untyped shapes and six fatal shapes. Every shape is exercised with companions None/S/R, both without an output parquet and with an existing valid parquet.
- Untyped + R: all nine shapes are compared with a copied directory omitting the bad file, using real CLI builds and `pd.testing.assert_frame_equal`; both outputs must contain the expected two rows. These comparisons run in both existing-artifact states (18 comparisons).
- Untyped without R and all fatal cases assert no writes and preservation of existing parquet bytes/mtime. Fatal cases require the parent's exception class and no informational skip. No-op controls cover S/E/X, absent/empty directories and weather-only directories, with and without existing artifacts.
- Deleted `parent_build` and the importlib/subprocess/Path imports it required. The test file contains no git subprocess or branch-name dependency; comparison inputs use `shutil.copytree`, not a moving branch reference.

## RED / GREEN
- Before edits, `baseline-4.log`: `85 passed in 3.37s`.
- RED before any builder change, on clean-source `dbb78b1`, `red-4.log`: `36 failed in 3.11s`. This is six fatal shapes × three companion states × two artifact states. All 12 R-companion cases failed because the old code incorrectly succeeded; all 24 other cases failed the required exception-class check.
- Targeted GREEN, `green-4.log`: `139 passed in 4.51s` (110 CLI build tests, 25 existing data-build tests, 4 schema tests).
- Additional source mutations were each killed and restored: swallowing AttributeError → `test_cli_fatal_shapes_raise_without_writes[True-R-null_body]`; bypassing membership for non-strings → the same test `[True-R-list_type]`. Exact requested source restoration was verified before the final run. Logs: `mutations-4.log`, `mutant-4-*.log`.

## Final focused run
Command, exit 0, log `focused-4.log`:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_day_outcome.py tests/test_entry_intent.py tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/health tests/test_refresh_memo.py tests/test_data_build_empty.py tests/test_day_outcome_entry_sequence.py tests/test_build_schema_assert.py tests/data/test_build.py -q -p no:cacheprovider`
Last line: `======================== 849 passed in 63.62s (0:01:03) ========================`
`git diff --check` passes.

## git diff --stat
```text
 src/bts/data/build.py          |  28 +++++----
 tests/test_data_build_empty.py | 126 ++++++++++++++++++++++-------------------
 2 files changed, 82 insertions(+), 72 deletions(-)
```

No requested table row remains unmatched. Evidence uses synthetic feeds; no production execution or deployment was performed. Review and landing remain the lead's work.
