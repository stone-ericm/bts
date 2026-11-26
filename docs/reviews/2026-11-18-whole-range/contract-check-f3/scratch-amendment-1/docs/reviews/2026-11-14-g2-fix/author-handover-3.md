# G2 correction to B

Worktree `/Users/eric/projects/bts-pp-c1`, branch `g2-fix`; changes remain uncommitted. No commit, push, stash, branch change, deployment, agent launch or real-account access.
This round changed only `src/bts/data/build.py`, `tests/test_data_build_empty.py`, and `tests/test_refresh_memo.py`. Kickoff-2 A, C and D stand; `plan_run_times` still matches main byte-for-byte.

## Implementation
- `build_season` now logs one stderr skip line per missing/unreadable game-type file, records its name, and continues scanning.
- With any requested-type feed present, those metadata failures do not prevent the normal build. The requested-type counter still precedes the seven-inning exclusion, preserving the existing schema-error path if no rows survive.
- With no requested-type feed and any unreadable metadata, raise a clear error naming every unreadable file; no parquet is written or replaced.
- Return None without writing only for missing/no-feed directories or entirely readable excluded types. Weather sidecars remain excluded. Truncated JSON and regular-feed row/schema failures retain their existing errors.

## Tests and RED/GREEN
- Baseline before this round's edits, `baseline-3.log`: `65 passed in 7.36s`.
- RED, `red-3.log`: `46 failed, 20 passed in 7.81s`.
- GREEN, `green-3.log`: `95 passed in 7.97s` (56 CLI build tests, 10 refresh tests, 25 existing data-build tests, 4 schema tests).
- Retained all 12 malformed-metadata cases and exercised each alone, beside spring training, and beside regular feeds. Regular companions build two rows with the named skip line; the other cases fail without writing.
- `test_cli_typeless_feed_preserves_parent_rows_with_regular_feed`: `{}` and missing-type bodies, before/after good feeds. A module-scoped scratch import loads the actual `git show main:src/bts/data/build.py`; four regular-companion cases compare the resulting parquet frames exactly with the parent's output. Spring-only companions fail and preserve the existing parquet's bytes and mtime.
- `test_cli_without_regular_feeds_names_every_unreadable_file` checks both error filenames and both skip lines.
- `test_regular_season_with_one_bad_cached_feed_refreshes_and_writes_marker` runs real `_refresh_season_data`/builder with only the pull transport stubbed: regular feeds plus cached `{}` produce the parquet and `.refreshed_2027_through_2027-06-15`; the bad cached file remains intact.

## Mutation checks
All four were applied to the source separately, killed by named tests, then restored; restoration was verified before the final run. Ledger: `mutations-3.log`; individual outputs: `mutant-3-*.log`.
- `skip-with-regular`: raise immediately instead of skipping → `test_cli_typeless_feed_preserves_parent_rows_with_regular_feed[000000.json-R-bad_feed0]` fails.
- `raise-without-regular`: disable the unreadable-file error → `test_cli_skips_unreadable_type_but_requires_a_regular_feed[None-feed0]` fails.
- `no-op-must-remember-unreadable`: stop recording unreadable files, wrongly permitting no-op → the same test `[S-feed0]` fails.
- `readable-nonregular-no-op`: remove the clean return → `test_cli_empty_season_is_success_without_writes[non_regular]` fails.

## Final focused run
Exit 0; log `focused-3.log`:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_day_outcome.py tests/test_entry_intent.py tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/health tests/test_refresh_memo.py tests/test_data_build_empty.py tests/test_day_outcome_entry_sequence.py tests/test_build_schema_assert.py tests/data/test_build.py -q -p no:cacheprovider`
Last line: `======================== 795 passed in 91.69s (0:01:31) ========================`
`git diff --check` passes. Nothing requested is unfinished. Parent parity is measured for the requested `{}`/typeless mixed-directory cases; evidence uses synthetic feeds and no real model training or production execution. Review/commit/landing remain the lead's work.
