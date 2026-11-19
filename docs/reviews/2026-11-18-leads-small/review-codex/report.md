# Verdict

**BLOCK** on `b0c11135` against `df454725`.

## Blocking findings

- **A late void can take the next day's result and roll back the restored streak.** In a fake author feed, May 1's pick was still open when May 2's pick was posted. May 1's correctly threaded `Void. Streak: 9` then arrived before May 2's `✅ Streak: 10`. Base regeneration wrote `[null, hit]` and streak `10`. HEAD wrote `[hit, void]` and streak `9`, and export accepted both incorrect results as settled. `bts state verify` reported two result mismatches and a streak mismatch on HEAD. The new parser at `src/bts/state/regenerate.py:228` feeds the void into the newest-unresolved matcher at `:322` after `:85` discards the parent URI. Carry the parent URI into `ParsedPost` and attach a void only to the unresolved pick it references. [Probe](void_probe.py) · [Base](void-base.log) · [HEAD](void-head.log) · [Code](../../src/bts/state/regenerate.py)

## Notes

- The interrupted-write guarantee covers exceptions that unwind Python. SIGKILL halfway through the branch's temp write left the old cache intact and an abandoned temp on disk. A reused PID overwrote an abandoned regular temp and removed it on success. Concurrent calls in two threads sharing a PID collided on the same temp: the first published the second's bytes and the second raised `FileNotFoundError`. The scheduler and a manual CLI invocation are separate processes. Host power-loss durability was outside the fault injection, and there is no file or directory `fsync`. [Cache log](cache.log)
- With a `0555` models directory and a writable existing cache, the base updated in place. HEAD raised `PermissionError` and kept the old file. HEAD requires a writable models directory for publication. The checked-in prediction setup also writes blend files there. [Cache log](cache.log) · [Prediction setup](../../src/bts/cli.py)
- HEAD accepts decimal Unicode digits and leading zeros in a void streak. `١٢`, `𝟙２` and `0012` all resolve to 12. This widens the formatter's spelling for manually authored replies in the configured author's feed. The scorer emits ordinary ASCII digits without leading zeros. [Void log](void-head.log)
- A saver already spent in the snapshot was restored as available on both revisions when the first feed pick was void. With only a skip post, the written streak and saver came from the snapshot on both. This is the existing timeline precedence. [Base](void-base.log) · [HEAD](void-head.log)

## The cache

- Synthetic feeds and six copied 2025 feeds gave identical return values and cache bytes on first build, incremental build, no-op, and missing raw directory. The real-cache seed replay kept all 5,095 entries after replacing the six sampled entries from copied feeds. The complete cache was also readable with raw absent. Existing IDs were skipped and a missing cache with no raw stayed absent. [Cache log](cache.log)
- Injected ENOSPC, `KeyboardInterrupt` and `SystemExit` before and during the write and before and after replacement. HEAD propagated the injected failures, kept the old bytes through pre-rename failures, and kept the full new bytes after rename. Every unwinding case removed the temp. The base published truncated bytes on all three mid-write failures. [Cache log](cache.log)
- Two processes used disjoint feeds. On HEAD, the reader saw seed ID `100` during a half-write. The writers used different temp paths and published `[100, 101]` then `[100, 102]`. On base, the same reader got an empty lookup, and the second writer had read the first writer's truncated cache and finished with `[102]`. Atomic replacement prevents a torn read in this schedule. A last writer can still lose another writer's new IDs, as it could before. [Cache log](cache.log)
- Base followed a cache symlink in place. HEAD replaced the symlink with a regular file and left a second alias pointing at the stale target. The real `sync_to_r2` with fake storage uploaded the updated primary pathname on both revisions. With umask `022`, HEAD replaced a `0600` linked cache with a `0644` file. With umask `077`, HEAD replaced a regular `0644` file with `0600`. Both modes permit access by the scheduler and cron's shared `bts` user. [Cache log](cache.log) · [Sync](../../src/bts/data/sync.py) · [Deployment](../../.github/workflows/deploy.yml)
- Torn JSON, invalid UTF-8, array/null, non-numeric key, and a simulated read error discarded the entire cache on both revisions. A numeric key with a scalar value survived silently on both and can still make feature computation fail when it calls `.get()` on that value. [Cache log](cache.log) · [Feature consumer](../../src/bts/features/compute.py)
- HEAD emitted exactly one stderr warning per bad read and zero stdout bytes. A successful rebuild stopped further warnings. Without usable raw feeds, it returned the empty lookup and repeated the warning next call. The feature builder calls the cache once per `run_pipeline` pass. Scheduler retries, fallback, shadow or another invocation can add passes. The SSH orchestrator parses `predict-json` stdout alone, `run` and `preview` print human-readable output, the dashboard reads persisted state, and cron merges streams into logs. [Cache log](cache.log) · [Pipeline](../../src/bts/model/predict.py) · [Orchestrator](../../src/bts/orchestrator.py) · [Cron](../../scripts/cron-setup-hetzner.sh)

## The void reply

- The real formatter round-tripped all hit, miss and void variants for streaks 0–57 on HEAD. A miss always formats and parses as streak 0. Leading/trailing whitespace worked. Extra text, case changes, a negative number and embedded newlines stayed unrecognized without emoji. An appended ✅ or ❌ retained its prior hit or miss interpretation. [Void log](void-head.log)

| Feed case | Base results, final streak, saver | HEAD results, final streak, saver |
|---|---|---|
| First day void at 7 | `null`, 0, available | `void`, 7, available |
| Void between hits | `hit/null/hit`, 13, available | `hit/void/hit`, 13, available |
| First void at each streak 10–15, then miss | `null/miss`, 0, available | `void/miss`, 0, spent |
| Last day void after a saver-consuming miss | `hit/miss/null`, 0, spent | `hit/miss/void`, 0, spent |
| Both double-down slots void at 12 | `null`, 0, available | `void`, 12, available |

- All five HEAD cases in the table passed verification against independent live fixtures and exported their `void` as a settled Pass. Base verification found drift and export refused the unresolved picks. `_record_to_pick_file` wrote `void` directly and `load_pick` read it with both double-down batter records intact. A double with one hit and one void grades as a hit. [Rule](../../CLAUDE.md) · [Probe](void_probe.py) · [Base](void-base.log) · [HEAD](void-head.log)

## Tests and false greens

- Base new-file results were 7 failures and 7 passes. Failures covered torn published bytes, missing `os.replace`, missing warning, all three void round trips, and an unresolved void in the timeline. [Base pytest log](base-tests.log)
- Six scratch mutations were caught. Writing in place failed two cache tests. Removing cleanup, silencing the warning, and swallowing the write error each failed a cache test. Dropping the void streak failed three round trips and timeline replay. Making a void spend the saver failed timeline replay. [Mutation matrix](mutations.log)
- The substring matcher survived all 14 tests. The unrelated-text fixtures contain no exact `Void. Streak: N` substring. The independent parser probe covers `re: Void. Streak: 12` and `Void. Streak: 12 extra`. The new timeline fixture also covers only same-day replies, leaving the late-void block undetected. [Surviving mutant](mut-substring-void.log) · [Void probe](void_probe.py)
- Current-clock combined run had 486 passes and one failure. The sandbox refused the dashboard socket test's `bind(127.0.0.1)`. The archived base reproduced the same failure before the handler ran. All cache-patching tests and both real-data bullpen checks ran and passed in the combined process. [Combined](combined.log) · [Base dashboard](base-web.log)
- On November 18 of 2027 and 2028 the new file, state regeneration, optional LightGBM, web saver, cache, sync and feature tests shared a process under each simulated wall clock. Each run had 188 passes and the same socket-bind failure. All 14 new tests passed in both years. The macOS shim shifted process realtime and the runner verified `date.today()`, `datetime.now()` and `time.time()` before collection. [2027](clock-2027.log) · [2028](clock-2028.log) · [Clock probe](clock_probe.py) · [Shim](review_clock.c)

## What I ran

- Archived base with `git archive df454725 src`. Ran the cache probe against both source trees and the public-feed probe separately on each source tree with a fake Bluesky client. Took local copies of the real cache, six feeds and the 2025 PA parquet. [Cache probe](cache_probe.py) · [Void probe](void_probe.py)
- Used `~/bts-runs/yield_run.sh` serially for the pytest jobs. The combined process included the new file, state regeneration, optional LightGBM, web saver, cache and sync tests, C2 framing, and the feature tests. No tracked files changed. [Combined runner log](combined.log.watch)
