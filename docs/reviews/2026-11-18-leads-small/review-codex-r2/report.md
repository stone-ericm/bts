# Verdict

SIGN

## Blocking findings

None.

## Notes

Confirmed `fb1acd4c` against `df454725`. The diff contains only `src/bts/features/compute.py` and `tests/test_leads_small_fixes.py`. The `src/bts/state/` diff is empty.

1. Base and HEAD returned identical lookups and produced identical cache bytes on the first build, incremental build, and repeat with no new feeds. With no raw directory, both returned the existing lookup and preserved its cache bytes, or returned `{}` with no cache file.
2. HEAD propagated each injected `OSError`, `KeyboardInterrupt`, and `SystemExit` during the temp write and immediately before and after `os.replace`. The old cache stayed intact until replacement, the new cache was complete afterward, and all 18 cases left no temp file. Before the first successful replacement, the fresh-cache path stayed absent. The base left truncated JSON in all six direct-write fault cases.
3. Truncated JSON, invalid UTF-8, and a JSON array each emitted exactly one warning on stderr and nothing on stdout. HEAD rebuilt the lookup from the raw fixture and rewrote the cache. Both versions stayed silent on subsequent reads with no new feeds and preserved the bytes, inode, and mtime.
4. Found no pick-impacting dependency on empty stderr or cache inode identity. The scheduler and local orchestrator use in-process results. Cron redirects stderr to logs, and the health checks use heartbeat, unit state, or specific fatal journal messages. With a synthetic pipeline invoking the real cache builder and a stubbed SSH transport, `predict-json` exited zero with the warning on stderr and the orchestrator accepted its stdout prediction. R2 sync and audit readers open the cache by path. R2 sync still uses separate reads for hashing and upload.
5. The base failed `test_interrupted_cache_write_leaves_the_previous_cache_intact`, `test_cache_write_replaces_the_file_atomically`, and `test_unreadable_cache_is_reported_and_rebuilt`. The readable-cache test passed. All four scratch mutations were caught. Write in place failed the interrupted-write and replacement tests. No cleanup failed the leftover-file assertion. No warning failed the stderr assertion. Swallowing the write error failed the `pytest.raises` assertion.
6. With no usable raw feeds, a corrupt cache stays on disk and each call says it is rebuilding. This preserves the base cache behavior, though the new message overstates what took place in that case.

## What I ran

- `git rev-parse`, `git status`, `git diff` for scope and whitespace, and repository searches for the cache path, builder, feature and prediction callers, and stderr handling.
- `.venv/bin/python .codex-review/leads-r2/prepare_sources.py` archived `src` from both commits with `git archive` and prepared the four independent mutants. `.venv/bin/python .codex-review/leads-r2/probe.py` ran the synthetic cache and prediction checks recorded in `probe.log`.
- `~/bts-runs/yield_run.sh` ran each pytest invocation sequentially with `-p no:cacheprovider -q --tb=short`. `head.log` records 21 passes across the new cache, `predict-json`, and orchestrator tests. `cache-callers.log` records seven passes across the sidecar and mocked R2 sync cases.
- The four new tests ran against each scratch source. `base.log` records three failures, `write-in-place.log` records two, and `no-cleanup.log`, `no-warning.log`, and `swallow-write-error.log` each record one. The wrapper emitted a `nice` warning and recorded the expected pytest exit statuses in the `.watch` logs.
