# W1.2 benchmark bridge: results memo (DRAFT SKELETON)

**Status:** skeleton written 2026-10-04 before the full run finished. Sections marked *to fill* take numbers only from the accepted run's `summary.json` / `table.parquet`. Analysis deliverable: at most two Codex rounds, then freeze with the limits stated.
**Design:** `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md` rev 3, FROZEN. **Exposure:** X-21 (`e09a7b7`), predeclared before any outcome-bearing execution.
**Use:** descriptive only, under D3 = RESERVE. Nothing here nominates or tests a candidate.

## 1. The run
- **Accepted run:** *to fill:* `data/validation/w12_bridge/<sha>-<stamp>/`, code commit, registered dates, input manifest hash.
- **Pre-output fixes.** Neither attempt below wrote any output, and none was read.
  1. **Memory (`24129fd`).** The first full attempt was OOM-killed under its 11G cap while training C-frozen. The runner held every slate game's full v1.1 feed in memory: about 4.2 MB each, measured on the box, across about 1,575 games. It now keeps `gameData` without its `players` map. The bridge reads only teams, weather, officials, venue, datetime and status. A test compares slots, start times and final status on full vs slim feeds.
  2. **Gzipped unit captures (`4e4bf5a`).** The static capture writer switched to `.json.gz` on 2026-07-10, but verified eligibility globbed only `units/*.json`. It therefore read 187 of 2,335 unit captures, and verified eligibility came out unknown for later dates: 1,854 of 26,399 candidate rows were verified-eligible in the stopped attempt. Both forms are now read. The fix is confirmed in the failure direction: the restarted run reports 14,094 verified-eligible rows.
  - **Operational changes, results unaffected:** the memory cap was raised from 11G to 13G at runtime (per-date scoring peaks hit the cap with no OOM kill), with `OOMScoreAdjust=1000` so the kernel kills this job before any production process.
- **The 03:00 data rebuild.** The production cron rebuilds `data/processed/pa_2026.parquet` at 03:00 ET. The run read that file at start (features) and reads it again at the end (outcome labels), and its manifest hashes the file at the end. Pre-03:00 sha256: `06264f2f…`. *To fill:* the post-03:00 sha256, and whether the two reads saw identical bytes.

## 2. Reproduction (C-served vs D), first
*to fill:* classes (exact_final_feed / inferred_weather_absent_at_serve / unexplained), rows and dates, residual quantiles, by pool.

## 3. Coverage and denominators
*to fill:* registered dates vs the season's game dates (missing fraction), candidate rows, pools (verified / surrogate / all), selection states, served status per date, exclusions per surface.

## 4. Surfaces and the chain
*to fill:* per surface and pool: top-1 hit / no_hit / no_pa / unknown, equal-date AUC, row-weighted Brier and log loss, mean stated−realized, reliability. The chain pairs with paired score changes, reranked rank-1 changes, discordant dates and the date-block interval. Stated sign: W1.2's `top1_diff_b_minus_a` is later minus earlier.

## 5. Prior results carried (not re-derived)
From the plan: the May PA-basis memo; M3's null; the June 29 PA-tilt; the May Gate B headline.

## 6. Limits
*to fill when frozen;* includes the design's §10 limits.
