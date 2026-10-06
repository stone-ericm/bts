# Reconcile receipt, `bts_reconcile_receipt_v1`

**What it is:** the producer prerequisite **P2** of the C1 rank-2 watchdog build plan (`docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`). It closes I-207's evidence gap: registration line 54 and the reconcile contract at line 59 of `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md`. Before this, `bts reconcile` always rewrote `streak.json`, printed "No scoring changes" and kept no record, so an empty corrections list was the only evidence. It establishes nothing.
- **Producer:** `bts reconcile` (`src/bts/cli.py`).
  - `bts.picks.reconcile_results(..., receipt=)` records each date and slot.
  - `resolve_daily_slot_results(..., receipt=)` records each slot.
  - A context-local source observer in `bts.picks._fetch_json` hands each grading payload's exact bytes to the receipt.
- **Receipt code:** `src/bts/reconcile_receipt.py`. Publication: `src/bts/receipt_io.py`, shared with P1.
- **Tests:** `tests/test_reconcile_receipt.py`. They run the real grading path with only `retry_urlopen` faked by URL.
- **What it does not change:**
  - **Requests, results, writes and the streak:** identical with or without a receipt; a test compares the request list, corrections, slot results and streak.
  - **Without a receipt:** `reconcile_results` calls `resolve_daily_slot_results(daily, d)` exactly as before.
  - **Failures:** every recording method is guarded; a failure is listed in `degraded` and never reaches the run.
  - **Retries and cadence** are unchanged.

## Where
- **One file per run:** `data/health_state/reconcile_receipts/<run start ET date>/<started %Y%m%dT%H%M%S%f>-<run_id>.json`. The cron runs at 02:00 and 07:40, so two small files a day. The restic `ops` set backs them up, and nothing prunes them.
- **Publication** (`receipt_io.publish`): the new directory levels' parents are fsynced, then the file is written atomically and its directory fsynced. On failure the files are withdrawn or tombstoned (`<name>.failed`), and the run prints `reconcile receipt unavailable`.
- **Discovery** (`reconcile_receipt.discover(picks_dir, run_date)`) returns only receipts without a tombstone. A missing or tombstoned receipt is unavailable evidence.

## Fields
| Field | Meaning |
|---|---|
| `schema`, `run_id` | `bts_reconcile_receipt_v1`; 32-hex uuid4 |
| `producer` | `{command: "reconcile", revision}`. The revision is sampled at publication |
| `lookback_days` | the run's lookback (default 8) |
| `started_at`, `finished_at`, `published_at` | tz-aware ET, from the run's own clock |
| `outcome`, `error` | `completed`, `raised` (with the exception type) or `incomplete` |
| `days[]` | one entry per target date, newest first |
| `corrections` | the run's corrections list (the CLI output), or null |
| `replay` | `saved` (`streak`, `saver_available`), `unavailable` (the replay refused incomplete history; the streak file was kept) or `not_reached` |
| `degraded` | recording methods that failed |

**Each `days[]` entry:**
- `date`, `cutoff_at` (08:00 ET the next day), `state`, and `detail`;
- `selection`: the slots (`slot`, `batter_id`, `batter_name`, `game_pk`) and the pre-run `result_before` / `slot_results_before`;
- `status_source`: the schedule fetch (`ok`, `url`, `sha256`, `completed_at`), or `{ok: false, error}`;
- `slots[]`, `write`.

| Day `state` | Meaning |
|---|---|
| `past_cutoff` | the run started at or after the cutoff: final, not fetched |
| `no_pick_file` | no pick file |
| `not_graded` | the pick has no hit, miss or void result yet (`detail.result`) |
| `pending` | a slot is not final; the remaining slots are `not_attempted` |
| `failed` | a slot's fetch raised. The run raised too (unchanged behaviour), and the receipt is still published with `outcome: raised` |
| `late_answer` | the day's answer arrived at or after its cutoff: discarded, with no coverage |
| `observed` | every slot resolved before the run's arrival check |

| Slot `state` | Meaning |
|---|---|
| `observed` | `result` (hit, miss or void) and its `basis`: `final_feed`, `final_feed_fallback_search` (the batter was found in another Final game that day), `schedule_void_state:<state>` or `suspended_no_evaluable_pa` |
| `pending` | not final |
| `failed` | `error` is the exception type |
| `not_attempted` | an earlier slot was pending or failed |

**Each slot also records:**
- `sources[]`: every payload the grader parsed for it (`url`, `sha256` of the exact bytes, `completed_at`);
- `response_completed_at`: the slot's last payload, or the schedule fetch for a schedule void;
- `covered`.

| `write.state` | Meaning |
|---|---|
| `correction_applied` | `old_result` → `new_result`, written under the scoring lock |
| `slot_results_updated` | the day result is unchanged but the slot results changed |
| `unchanged` | observed, and nothing to change |
| `refused_after_cutoff` | the write would land at or after the cutoff |
| `skipped_under_lock` | the pick was gone or ungraded when re-read under the lock |

## Rules for a consumer
- **Coverage** for a date's slot is established only by a discoverable receipt whose slot is `observed`, with `covered` true. That means `response_completed_at` strictly before `cutoff_at`, and the day not `late_answer`.
- **What cannot establish coverage:** `pending`, `failed`, `not_attempted`, `past_cutoff`, `not_graded`, a late slot, an empty `corrections` list, or a process exit of 0.
- **Coverage and corrections are separate facts.** `write.state` says whether a correction was applied or refused. `unchanged` is positive evidence that the observed result matched the stored one.
- **Clock steps:** a slot answered after the cutoff stays uncovered even if a wall-clock step back made the run's own arrival check pass. The run's behaviour in that case is unchanged; the receipt keeps the actual response time.

## Re-certification (registration line 82), for the producer set P1, P2, P4 and C1/C2
The following certified current-defence mutant patches (`results-f453283`) target functions in files these changes touched. Each function's body is unchanged, and each patch still applies at an offset. Re-run them with the frozen runner at the deploy candidate before claiming them current.

| Patches | File | Function |
|---|---|---|
| `D-I043-1`, `D-I043-4`, `I-0813-a` | `src/bts/picks.py` | `_classify_unposted_game_status` |
| `D-I043-2` | `src/bts/picks.py` | `resolve_pick_slot_result` |
| `D-I043-3` | `src/bts/scheduler.py` | `save_nonterminal_result` |
| `D-I063-2`, `I-0811-b` | `src/bts/cli.py` | `fetch_contest_streak` |
