# W15 Review Report (Round 2 — Carbon)

## Verdict
**SIGN**

## Blocking findings
None.

## Notes
- **N1 — `warn()` under `2>&-` (`src/bts/features/probable_pitcher_cache.py:37`).** If fd `2` is closed before Python starts (`sys.stderr is None`), `fd < 0` calls `print(line, file=None, flush=True)`, which writes to `sys.stdout` (`probe_r2_item1c_dm_transport.py`). No production caller closes fd `2` (`systemd`, `cron >> cron.log 2>&1`, `orchestrator.py` SSH, and `DEVNULL` keep fd `2` open), and base `f02e6b9c` behaved identically.
- **N2 — `D+1` streak label (`src/bts/health/probable_pitcher_cache.py:80`).** Because R4 embeds `<processed date>` in `incident_key` (`probable_pitcher_cache:<today>:<sha16>`), `attention.py` treats `D+1`'s repeat summary as a fresh key and appends `(1st observed day)` rather than `(2nd consecutive day)`. Delivery is unaffected (`ALWAYS_ATTENTION_WARN_SOURCES` sends on streak 1; `alert.py` dedups per `sent_day`).
- **N3 — Ruling W15-S2 (R3).** Missing-member entries drop the whole object (`0.5` → `NaN` without feed/alias) as ruled and ledgered in `CLAUDE.md:251`.

## 1. Round-1 blocking findings on original inputs
Ran `probe_r2_item1_fixes.py` and `probe_r2_item1c_dm_transport.py` against `d7c66234` and archived `f02e6b9c`:
- **Descriptor pressure (Codex 1 & Carbon B1, `probable_pitcher_status.py:35`):** (i) Codex 1's exact fixture (`0001.json`–`0064.json` directories before valid `109.json`, `RLIMIT_NOFILE=48`) scanned and persisted `109`, returned `opp_bullpen_hr_30g = 0.5`, emitted no stderr/incident, and kept open fds unchanged (`0 → 0`, matching base). (ii) Carbon B1's 70-call daemon loop (`RLIMIT_NOFILE=64`) with directories at `probable_pitcher_lookup.json`, `rejected/...json`, and `health_state/probable_pitcher_cache/2026-12-31.dir.json` returned `0.5` on all `70/70` calls with `0` leaked fds.
- **Closed-stderr child exit code (Carbon B2, `features/probable_pitcher_cache.py:28-43`):** Child `compute_all_features` runs with a closed-reader `os.pipe()` (`EPIPE`) on `stderr` completed (`FEATURES_OK rows=2`) and exited `0` on both single-warning (`invalid_entries`) and triple-warning (`unreadable_cache` + failed preserve + failed status) incidents.
- **1,000-grapheme chat cap (Codex 3, `health/probable_pitcher_cache.py:64-108`):** Through real `run_all_checks`, `apply_warn_attention_policy`, `format_health_dm_body`, `dispatch_dm_for_health_alerts`, and `bts.dm.send_dm` over a captured `retry_urlopen` rejecting `> 1,000` chars: persistent `{"100": null}` sent wire lengths `D=635`, `D+1=636`; same-day `invalid_entries` + `unreadable_cache` pair sent `D=648`, `D+1=648`; 16 records (15 valid + 1 corrupt) sent `671` (`alert=598 <= 600`), with zero backticks.

## 2. Scope, DM vs journal split, `incident_key` dedup, and log routing
Inspected `git diff a924d748 d7c66234` and ran `probe_r2_item2_3_journal_and_texts.py`:
- **Changes beyond Amendment 2/2a:** None in `src/`; `CLAUDE.md:251` also clarified *"scans or rescans a usable game"* (Round-1 N3). Nothing regressed.
- **DM vs per-record details:** The single DM keeps count `<k>`, `<kinds>`, `<status dir>`, the keep-copies rule, the `bts-lead`/Eric route, and both urgent actions (preview/pick review before next selection; R2 verification before next sync). Timestamps, per-bucket counts/IDs, and `<preserved>` paths live in `<status dir>` JSONs and journal logs, which `bts-lead` and Eric both access directly; on a `D+1` repeat after repair, they check `<status dir>`/journal timestamps rather than judging from the DM alone.
- **`incident_key` dedup:** Verified suppression on same-day repeats (`status_unreadable` or valid), fresh DM on same-day repair (`path.name` → 64-hex `identity`), fresh DM on same-day added record, fresh DM on `D+1` catch-up, and fresh DM when `D+1` adds a new recurrence of `D`'s identical `identity` (`[id]` → `[id, id]`). Verified `len == 600` keeps custom `<status dir>` and `601` falls back to `data/health_state/probable_pitcher_cache` (`524` chars).
- **Where `log.warning` goes:** Tested in a standalone child importing `bts.cli` and `bts.scheduler` (without `pytest` `caplog`). `root`, `bts`, and module loggers have `handlers == []` (no `NullHandler`), so `log.warning` routes via Python's `logging.lastResort` (`_StderrHandler`) to `sys.stderr`, emitting all per-record A7 warnings, `status unreadable` warnings, and `log_alerts`'s `[probable_pitcher_cache WARN]` summary. Under `bts-scheduler.service` (default `StandardError=journal`), `sys.stderr` lands in `journalctl --user -u bts-scheduler`. No cron job in `scripts/cron-setup-hetzner.sh` calls `run_all_checks`; any cron invocation using `>> $LOG_DIR/cron.log 2>&1` writes those `stderr` lines to `~/logs/cron.log` rather than `journald`.

## 3. New and changed operator texts
| Text | Worst state if followed literally |
|---|---|
| Summary WARN (`health/probable_pitcher_cache.py:68-74`, normal `<status dir>`) | Combines `D` + `D-1` observations (including post-repair `D+1` repeats) without timestamps or IDs; operator keeps all copies intact and forwards to `bts-lead`/Eric, who inspect `<status dir>` and preview/R2 before acting. Safe. |
| Summary WARN with `> 600`-char `<status dir>` fallback (`line 76-77`) | Very long custom `picks_dir` prints default `data/health_state/probable_pitcher_cache` in DM while journal logs the real path; production uses the default path (`<= 528` chars), and `bts-lead`/Eric have journal access. Safe. |
| Plain-text logged A7 & `status unreadable` lines (`lines 24, 33, 38-61`) | Identical to Round-1 A7/P6 semantics with backticks removed (R6); retrospective to `recorded_at`, directs review/preservation only, never manual deletion or entered-pick edits. Safe. |
| `CLAUDE.md:251` updated W15 bullet | Documents R3 missing-member loss, journal vs 600-char single-WARN DM split, and inherited total-digest chat cap across other sources; mandates stopping scheduler and active runs before repair cutover. Safe. |

## 4. Tests and mutation analysis
After 04:25 ET (`pgrep -f yield_run.sh` empty), ran single-process `pytest` via `yield_run.sh` (`r2/pytest_combined.log`) across `test_wr_w15_{corpus,incidents,health}.py`, `test_probable_cache_review.py`, `test_leads_small_fixes.py`, `tests/health/test_{runner,attention}.py`, `test_lightgbm_optional.py`, and `test_web_saver.py`: **`256 passed, 3 skipped in 14.67s`**. Re-ran all Round-1 mutants, all 3 R5 mutants, and 6 new Round-2 mutants (`probe_r2_mutations.py` → `r2/mutations.log`): **`20 killed, 0 survived`**.

## What I ran / What I did not run
- **Ran:** `probe_r2_item1_fixes.py`, `probe_r2_item1c_dm_transport.py`, `probe_r2_item2_3_journal_and_texts.py` (direct child/transport probes against `d7c66234` and `f02e6b9c`); wrapped `r2/pytest_combined.log` (259 collected) and `r2/mutations.log` (20 mutants) via `~/bts-runs/yield_run.sh`.
- **Did not run:** Full fast suite, live network/Bluesky/R2 calls, production systemd/cron jobs, or full historical backtests.
