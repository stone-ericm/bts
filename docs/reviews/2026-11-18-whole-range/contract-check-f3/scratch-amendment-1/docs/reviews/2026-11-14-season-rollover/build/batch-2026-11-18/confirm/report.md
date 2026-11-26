# Verdict

**BLOCK** `8cac32ea`. The five batch reproductions and the frozen-snapshot regression are corrected on the tested inputs. Three remaining cases hide a real alert or accept incomplete capture evidence. The ordinary pick, delivery and scoring replay matched the base.

## Blocking findings

1. **F4 — Sparse midseason history still suppresses the coverage alarm.** After a July restore with an established contest account but only a July 15 settled pick copied locally, a fresh automatic observation with `source_date: null` gave `INFO` on July 15 and just after midnight July 16 ET. An empty midseason account ledger needs the coverage alarm even when earlier local picks are absent. The branch uses the oldest local settlement as its only time anchor. Also bound grace to the actual season opener when available. The configured March 24 opener suffices to reject this July case, and the moved-opener and 2028 controls need a reliable opening date. [contest_state.py:81](../../src/bts/health/contest_state.py#L81), [HEAD probe](probe-contest/head/summary.json).
2. **F5 — Carryover clears a table already critical at the season close.** With the default six-day threshold, the same September 22, 2026 source was `CRITICAL` on September 28 and quiet on March 26, 2027 despite no new source data. It was only `WARN` on March 27 and returned to `CRITICAL` March 30. The grace branch admits equality while the alert starts at equality. Use a strict upper bound on the closing gap and require a configured prior-season close before granting grace. The fallback can also hide a closing outage. With March 24, 2029 as the configured opener and a September 23, 2028 source, the missing 2028 end silently uses September 28 and grants grace. Configuring October 1 for the same fixture restores `CRITICAL`. [park_drag_freshness.py:88](../../src/bts/health/park_drag_freshness.py#L88), [strategy.py:56](../../src/bts/strategy.py#L56), [HEAD probe](probe-park/head/summary.json).
3. **F7 — Both exports can drop scored rows and verify.** The offline prediction transport returned ten scored batters per variant with `top_n=10`. A fault injection dropped the last row from both profile conversions before manifest generation. The real exporter wrote `effective_top_n=9` and both manifest row counts as nine. The base verifier rejected both profiles, while HEAD verified them and the real wrapper reported `existing_verified` with no capture alert. Calculate each variant’s capped count from the scored prediction rows before profile conversion and require the written profiles to match that independent count. [artifacts.py:464](../../src/bts/experiment/artifacts.py#L464), [verifier](../../src/bts/experiment/artifacts.py#L1356), [base probe](probe-capture/base/summary.json), [HEAD probe](probe-capture/head/summary.json).

## Notes

- A fresh reading with an empty ledger can still be normal after an opening off day. In the opener/off day/second game sequence, the time cutoff begins March 26 at 00:00 ET, before the second game. Fresh reads do not extend grace. The alert stays critical for as long as `source_date` is absent, with no separate expiry. The probe recovered when coverage through the opener arrived. A January test settlement alongside the real opener also causes an opening-night critical. [Contest probes](probe-contest/head/summary.json).
- The real P-03 snapshot and its raw cache were unavailable locally. The comparison used a frozen two-day 2026 fixture with local raw feeds.
- The mutation survivors below include reachable verifier coverage gaps and guards that are redundant in the tested path.

## F4

- July 15, 2027 with only January 2 or March 24 settled went from `INFO` on `153ff8c2` to `CRITICAL` on HEAD. Adding a July 15 settlement to either old-history fixture stayed critical on HEAD. March 24–25, a moved April 20–21 opener and April 3–4, 2028 stayed `INFO`.
- Yesterday’s first settled pick at 00:01 ET stayed `INFO`. Two games represented by one daily pick with two distinct game IDs counted once. An opening-night void counted as a settlement and stayed `INFO`, including the next day with the first non-void result. [Before and after](contest.log).

## F5

- On March 26, 2027 the June 15, 2026 and September 28, 2025 sources went from quiet on `153ff8c2` to `CRITICAL` on HEAD. September 27, 2026 carryover was quiet March 26, warned March 27 and was critical March 30. September 22 qualified at six days, while September 21 stayed critical. The default close for an unconfigured year is September 28. [Before and after](park.log), [case outputs](probe-park/head/summary.json).

## F7

- The wrapper and capture-health file are byte-identical to `3d49308d`. On `153ff8c2`, a PA appearing at exporter entry allowed scoring to begin, a PA at swap allowed replacement, and a failed recheck plus transient pick change allowed replacement of formerly verified evidence. On HEAD each failure stayed frozen and raised `CRITICAL`. [Empty diff](f7-restored.diff), [prior reproductions](probe-capture/before/first-review/summary.json).
- Nine genuinely scored rows verified with `top_n=10`, and a later tick with a wider slate reverified without exporting. Legacy ten passed and legacy nine failed on base and HEAD. Floats, booleans, zero, negatives, over-cap counts, counts unequal to stored rows and mismatched variants failed. A resolved copy inherited nine and verified. An unreadable Parquet file and readable lost rows against an intact manifest both failed.
- The verifier compares each stored profile’s length to its declared `row_counts` and its per-date length to `effective_top_n`. The exporter takes that effective count from the larger of the two already converted profiles. Neither check sees the upstream scored-row count. [Case outputs](probe-capture/head/summary.json).

## F8

- The real default March 26, 2027 update on `153ff8c2` omitted the aged September 27, 2026 contradiction from stdout and status JSON. HEAD restored the exact base warning and JSON date list. The mixed default sample contained only March 24, 2027. HEAD `--season 2026` on mixed data reproduced the base’s 2026-only status, including dates, outcomes and statistics. [CLI evidence](probe-shadow/head/summary.json).

## P-03 closeout

- Under a 2027 process wall clock, the real closeout produced byte-identical stdout and JSON on base and HEAD with two evaluable days and zero API calls on the frozen fixture. The same run on `153ff8c2` yielded zero recorded evaluable days. [Closeout output](probe-p03/head/closeout.stdout).

## The pick path

- `git diff 153ff8c2 HEAD --stat` shows 14 files, 366 insertions and 98 deletions. Production edits are confined to the closeout, capture wrapper, contest/park/capture health, and policy-shadow status writer. The F6 cron file has an empty fix diff. `git diff --check` passed. [Stat](diff-stat.log), [cron diff](f6-cron.diff).
- The real July 15, 2027 `run_day` picked the Kwan/Soto double using the fixture’s local policy fallback, sent one pick DM, scored a hit and ran EOD health with a stale park feed. Pick JSON, decision record, streak JSON and both captured DMs matched byte for byte on `3d49308d` and HEAD. Neither run called the captured public-post or reply transport. [Replay](pick.log), [sends](probe-pick/head/sends.json).

## Tests

- The ten files changed since `3d49308d`, plus `test_lightgbm_optional.py` and `test_web_saver.py`, ran together per ET wall clock. **2027 and 2028 each had 1,002 passed, 1 skipped.** The skip is the absent park reference dataset. The first sandboxed run failed only on the real HTTP server’s loopback bind. The approved rerun passed with the local reverse-lookup shim and outbound guards. [Focused results](focused-loopback.log).
- Applied 17 of the 23 prior survivors, adapting F4’s folded guards. Twelve were caught. Six former retry mutants were inapplicable because the path was removed. Survivors were F4 timestamp-parsed and explicit settlement presence (loader and count make them redundant here), F5 clamp and empty physics (same outputs in these fixtures), and F7 live-kind restriction (historical-manifest coverage gap).
- Of five extra mutants, the F4 time and future guard removals were caught. F4’s upper count survived because two permitted calendar dates cannot contain three distinct root-level daily files. F7 bool-only and negative-only mutants survived, though the direct verifier probes reject those values. [Mutation results](mutations.json), [pytest evidence](mutations.log).
- The old F7 export-count and retry exit-code assertions were replaced by repeated frozen failed rechecks with an unexpected export treated as an error. The moved-opener park fixture changed from September 20 to September 22, with separate boundary assertions. The future-pick case and shadow count assertions remain covered. I found no other existing result assertion removed or loosened. [Test diff](tests.diff).

## What I ran

- Archived `3d49308d` and `153ff8c2` via `git archive <revision> src scripts`. Ran the contest, park, capture, shadow, P-03 and `run_day` probe families through the local runner, with the first reviewer’s capture cases replayed on `153ff8c2`. [Runner](review_runner.py), [probes](confirmation_probes.py).
- Ran the focused and sequential mutation families through `~/bts-runs/yield_run.sh`. The Python network guards blocked DNS and non-loopback connects, and the MLB schedule, prediction, outcome and send transports used local fixtures. No box, contest account or external network was used. [Network guard](sitecustomize.py).
- All review changes are under `.codex-review/confirm-1118/`. No tracked files changed.
