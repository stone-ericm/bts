# Verdict

**BLOCK** `153ff8c2` against `3d49308d`. The ordinary pick, delivery and scoring comparison matched. Four findings crossed their bounds by suppressing alerts or accepting invalid capture evidence.

## Blocking findings

1. **F4 — Missing contest coverage can be `INFO` well into the season.** On July 15, 2027, a fresh automatic observation with `source_date: null` and only a January 2 test pick marked hit yields `INFO` on HEAD and `CRITICAL` on base. The same occurs with only a March 24 settlement. Fresh writes with an empty ledger can keep the alert suppressed until a third settlement. Bound the exception to a short interval measured from the first eligible settlement, using the real opening slate when available. Replayed both July files and the opening controls through the actual health check. `src/bts/health/contest_state.py:76–98`, `probe-contest`.
2. **F5 — A prior-season outage is treated as an offseason gap.** A table whose source stopped June 15, 2026, with a fresh manifest and successful producer status on March 26, 2027, gives no alert on HEAD and `CRITICAL` on base. The prior-year branch discards all previous-season lag. Admit carryover grace only for coverage from the immediately previous season that was within the normal freshness threshold at its close. Ran the actual freshness check with the stopped table and the September carryover control. `src/bts/health/park_drag_freshness.py:77–100`, `probe-park`.
3. **F7 — The retry can begin scoring after an outcome is visible and can publish after one arrives.** With a failed nine-row capture and the original unresolved pick, I wrote a real PA parquet row at exporter entry, after the first guard. Both scoring calls ran with the file visible. The publication check preserved the old capture. I then inserted the same row at the first rename after the second guard. The CLI verifier accepted the staged export, the wrapper returned `recaptured_after_failed_verification`, and the capture health check returned no alert. Fence both start and acceptance with the existing research first-pitch deadline for automatic retry, or serialize with both outcome publishers through the swap. Preserve or restore the prior artifact on a failed acceptance. `scripts/live_forward_capture_once.py:397–469,643–699` and `src/bts/health/analytics_artifacts_missing.py:305`, `probe-capture`.
4. **F7 — A failed recheck can erase the protection for a formerly verified capture.** A verified full capture fails a recheck after its profile becomes unreadable. A changed pick causes `stale_pick_snapshot` and overwrites the previous-success flag. Restore the original unresolved pick and the new retry replaces the previously verified capture and clears the health alert. The existing backup contains the original. Persist successful verification independently of tick status, keyed to the captured evidence, or preserve the flag on every subsequent status write. Replayed all four ticks with the actual verifier and health check. `scripts/live_forward_capture_once.py:576–606,706–720`, `probe-capture`.
5. **F8 — Default cron drops the existing aged-contradiction warning at rollover.** A September 27, 2026, shadow record whose authoritative decision has since changed to `single` is still detected by `find_aged_contradictions`. In March 2027 the default update emits the warning on base. HEAD drops it from stdout and from the status JSON. Explicit `--season 2026` exposes it. Keep the full contradiction set for operational warnings and use the season only for the statistical sample. Invoked the actual update command against the same decision rewrite on both trees. `src/bts/skip_policy_shadow.py:425–436` and `src/bts/cli.py:3148–3153`, `probe-shadow`.

## Notes

- The existing P-03 closeout script calls the newly scoped status builder without a season while defaulting to a frozen 2026 snapshot. Its recorded side will use the wall-clock year, while its raw-feed backfill still spans the snapshot. Pass the snapshot season explicitly. `scripts/audit/p03_context_shadow_closeout.py:36,56–74`.
- The automatic opening exception and park carryover are independent of the separate contest fetch and park cron liveness alerts. A healthy process can still write an empty ledger or a stale export.
- The leaderboard handle remains an owner decision. The cron change leaves the service file alone. `deploy/systemd/bts-leaderboard.service:15`.

## F4 — Contest-state health

- The opening `INFO` cases reproduced for March 24–25, a moved April 20–21 opener and April 2028. A third pick, a reading over 36 hours old, a last-ET-year or future reading, a future settlement, zero current-year settlements and 100 settled picks all stayed `CRITICAL`. Exactly 36 hours still gave `INFO`.
- An active legacy manual file with no source date stayed `CRITICAL`. An active dated CLI observation supplied coverage and produced no alert. A revoked live contract appended its `CRITICAL` last, even with the opening `INFO` first.
- With the last successful reading at 13:30 ET on March 24, opening EOD and the next EOD are `INFO`, then the first check past 01:30 ET on March 26 is `CRITICAL`, even with only two settlements. The separate failed-fetch DM can fire sooner. The July probes establish the case where the bound fails.

## F5 — Park drag

- A September 27 carryover is quiet March 23–26, `WARN` March 27–29, and `CRITICAL` March 30. The April 20 moved opener gave April 23/26. The first refresh that can see Opening Night runs March 25 because the query ends yesterday. These thresholds give two days from that first reachable refresh to `WARN` and five to `CRITICAL`.
- Unconfigured 2028 uses the full gap. A stale same-year source kept the original `CRITICAL`, October–February were silent, and an old generated timestamp still warned. The carryover branch loads strategy and its NumPy/Pandas dependencies, and succeeded with LightGBM imports explicitly denied.
- Base fails on the header-only pitch store both before Opening Night and after its first game. HEAD completed both, captured 30 pitches from the first game, and completed with zero rows and no export when static data was also empty. On populated refreshes with and without new pitches the export, manifest, status, metadata and compressed pitch store matched byte for byte with timestamps pinned.

## F6 — Cron and fetch

- Under a throwaway home and real `show`, enter output matched line for line after replacing fixture paths. Research removed exactly four `--dm-recipient` arguments. Installation/removal plumbing and every other cron line are unchanged.
- Fourteen handled failures all returned 2, wrote the error to stderr and status with the expected category, and sent zero DMs. Covered cookies, auth outage/rate limit/network, both identity guards, profile 429/503/403/network/shape, empty or unreachable rounds and corrupt switch.
- A syntactically valid JSON array in the prior observation raises `AttributeError`, exits 1 and leaves stderr and status empty. Both revisions reproduced this preexisting failure. `src/bts/cli.py:2352`.

## F7 — Live-forward capture

- Used the real capture wrapper, CLI materializer and verifier, with captured predictions. Nine scored rows with `top_n=10` verified on HEAD, with `effective_top_n=9`. An eighteen-batter tick reverified the frozen profiles without exporting again. Base failed the nine-row count twice. Legacy ten-row captures passed and legacy nine-row captures failed identically.
- A failed same-pick verification retried before outcomes and backed up the old artifact. An already settled pick or existing PA prevented scoring. Pick replacement during export failed the publication check and preserved the old artifact.
- Requested top-N, schema, provenance, locked-snapshot and null outcome field checks remain. The widened acceptance in these probes comes from the retry and status handling, including the gap between the last outcome check and the swap.

## F8 — Shadow seasons

- Mixed 29 dates from 2026 and one from 2027. The three default outputs used one date on HEAD and 30 on base. Explicit 2026 reports matched base on a 2026-only folder, including both status calculations. For `check-results`, the processing date selected 2026. Only generated times and fixture paths were normalized.
- A real CLI grading pass produced byte-identical production hit, shadow miss and streak files. The policy update recorded one missing decision, pruned one recent supersession and reconciled two pending records across years on both trees. Dry-run shadow backfill visited all 30 dates on both.
- January 15, February 15 and March 23 showed `no_shadow_files (0/30)` from `shadow-status`, with an artifact action to verify shadow configuration. Policy update showed zero divergent days and `insufficient_n`, and the dashboard omitted its section. `check-results` printed the empty status artifact path and `No pick found`. EOD analytics health checks locked picks and their artifacts directly. I found no automated health consumer of the empty status outputs. The status action can mislead a human before Opening Day.

## The pick path

- The diff spans 21 files. The production changes are in `cli.py`, `experiment/artifacts.py`, `features/park_drag_producer.py`, `health/{analytics_artifacts_missing,contest_state,park_drag_freshness}.py`, `shadow_eval.py`, `skip_policy_shadow.py`, `strategy.py`, `cron-setup-hetzner.sh` and `live_forward_capture_once.py`. It also changes one ops doc and nine test files. See `diff-stat.log` for the complete list.
- No selector or delivery function changed. Strategy adds only a table and `season_opener_date`, called by park health. Both changed health `check` functions and the capture health status set run through the EOD runner after scoring. The park producer `refresh` is a cron/CLI writer. Park drag is in the shadow context columns and outside the production feature list.
- The scheduler dispatches capture after lock. `capture_once` and `refresh_stale_artifact` call the experiment exporter and verifier for observational files. `materialize_live_candidate_profile_pair` invokes scoring of the capture slate through the prediction pipeline, and `verify_candidate_artifact_pair` checks the artifacts. None of these functions supplies a selection, delivery, lock or pick result back to the scheduler.
- `check_results` is a grading entry point. Its changed nested status writer reads results into a separate JSON and catches errors. That output is never read back into commit checks, slot resolution or streak updates. `shadow_status`, `skip_policy_shadow_update`, `build_shadow_cycle_status`, `load_decision_records` and `write_status` are monitoring functions. The update command's existing hypothetical outcome reconciliation precedes and is independent of the scoped write.
- Real `run_day` on July 15, 2027, using the saved-pick fixture transports, selected and delivered the Kwan/Soto double, scored the hit and ran real EOD health. Pick, decision, streak, pick DM and health DM matched byte for byte. Neither revision attempted a public post or reply. The sample used the fixture's local fallback policy, an injected final-game response and a stale park health fixture to exercise DM delivery.

## Tests and false greens

- Replayed the changed tests on the archived base. Expected failures were F4 five, F5 four, F6 one, F7 three, F8 two. The F7 full-width control fails on base only because the new manifest key is missing. The thin case fails verification, and the retry case fails because no export runs. The F6 no-recipient controls and F7 preexisting-outcome controls were green on base.
- Forty separate scratch mutants. Seventeen failed the relevant tests. Twenty-three survived. F4 survivors include the automatic-file restriction and some redundant settlement guards. F5 survivors include the source-year restriction and the both-empty output branch. F7 survivors include effective-count validation, prior-verification protection, snapshot identity and every publication recheck predicate. F8 survivors include both CLI date overrides, contradiction scoping, and an intentionally wrong 2026 default in the context status builder because its test asserts counts without dates.
- The test diff preserves the existing result expectations. F6 replaces an equality assertion with equality after the intended four removals and adds the schedule checks. F7 parameterizes the original full-width case. F8 pins four existing tests to 2026 through arguments or clock.

## What I ran

- `git diff 3d49308d HEAD --stat`, production and test diffs, call-site search, and `git diff --check`. Base used `git archive 3d49308d src scripts`. No tracked files changed.
- `review_runner.py probe {pick,contest,park,cron,capture,shadow}` on both revisions. Transport doubles, DNS and non-loopback guards, and all fixture outputs are in this review directory. `review_probes.py` contains the executable reproductions.
- `yield_run.sh` ran the nine changed test files with `test_lightgbm_optional.py` and `test_web_saver.py` in one process per shifted clock. Both ET wall clocks were verified. Each year finished **975 passed, 1 skipped**. The skipped park producer oracle requires the absent external analysis fixture. The web saver used local loopback and a local reverse-lookup shim. See `focused-final.log`.
- `yield_run.sh` ran the base replay and scratch mutations sequentially. See `base-and-mutants-final.log` and `review_mutations.py`.
