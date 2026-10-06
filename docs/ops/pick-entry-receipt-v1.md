# Pick-entry receipt, `bts_pick_entry_receipt_v1`

**What it is:** the producer prerequisite **P1** of the C1 rank-2 watchdog build plan (`docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`). It implements registration R4 and the producer receipt contract in `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md`. It is revised after producer review r1 (C1–C6).
- **Producer:** `bts check-pick-entered` (`src/bts/cli.py`), with the receipt code in `src/bts/entry_receipt.py` and publication in `src/bts/receipt_io.py`.
- **Tests:** `tests/test_entry_receipt.py`.
- **What it does not change:** the run's DMs, marker statuses, exit codes, retry policy, authenticated requests and cadence. The existing `TestCheckPickEntered` tests pass unchanged. The marker gains one key, `receipt`.

## Where
- **One file per run:** `data/health_state/pick_entry_receipts/<ET date>/<started %Y%m%dT%H%M%S%f>-<attempt_id>.json`, every run, whatever its outcome. The cron runs `*/15 10-23` under `entry_intent = "enter"`, so about 56 small files a day.
- **Backup and retention:** the restic `ops` set (`data/health_state`) backs it up. Nothing prunes it.
- **Publication** (`receipt_io.publish`):
  - each missing directory is created one level at a time, and its parent is fsynced;
  - a temp file is written and fsynced, renamed to the final name, then the directory is fsynced.
  - **On any failure,** the temp and final files are removed and the removal fsynced. If removal fails, a `<name>.failed` tombstone is written. The run prints `entry receipt unavailable` and is otherwise unaffected.
- **Discovery** (`entry_receipt.discover(picks_dir, date)` / `receipt_io.discover`) returns only `*.json` files **without** a `.failed` tombstone. **Consumers must use that rule:** a tombstoned or missing receipt is unavailable evidence, never success. The one unhandled case is a filesystem that refuses the removal **and** the tombstone write; that is the stated limit.
- **The marker** `data/health_state/pick_entry_check.json` names the receipt of the run that wrote it (`receipt`).

## Behaviour neutrality (review r1 C5)
- **During the run,** the receipt's hooks only store references and clock readings, each behind a guard; a guard failure is listed in `degraded`.
- **In `publish`:** every extraction, hash, qualification and serialization happens there, after the run's own decisions, DMs and marker writes. A failure there publishes nothing and changes nothing.

## Fields
| Field | Meaning |
|---|---|
| `schema` | `bts_pick_entry_receipt_v1` |
| `attempt_id` | 32-hex uuid4, one per run |
| `producer` | `{command: "check-pick-entered", revision: <git HEAD of the running checkout or null>}`. The revision is sampled at publication: a code version, not proof of the deployed artifact |
| `et_date`, `season` | the contest date checked (ET) and its year |
| `started_at` | run start (tz-aware ET). It is the same instant the run uses for its decisions |
| `outcome` | see below |
| `detail` | set only when the run raised: `raised <ExceptionType>` |
| `selection` | **expected identity**, parsed from the exact bytes the run read once (C2): `slots` (`role` pick / double_down, `batter_id`, `batter_name`, `game_pk`, `game_time`), `delivery` (channel, notification id, Bluesky uri, `delivered_at`, flags), `commit` (the gated decision's `decision_sha256`, `decision_valid`, `delivery_status`, `scoreable`), `pick_file_sha256` |
| `cutoff_at`, `minutes_to_pitch` | the submission cutoff (the **earliest** selected leg's first pitch − `SUBMISSION_CUTOFF_MIN`) and the minutes to that first pitch at `started_at` |
| `account` | **verified account**, once the login session returned: `expected_username` (the `--expected-username` option), `user_id`, `username` |
| `observation` | only for `observed`. **`response_completed_at`:** the instant the last fetch returned, before verification (C4). **`before_cutoff`:** strictly earlier than `cutoff_at`. **`sources`:** the sha256 of each consumed payload (`profile`, `pending`, `rounds`, `crosswalk`; canonical JSON) plus `units`, the local `units.json` capture used (path, sha256, capture time) or null. **`rows`:** the target date's rows, `profile` slots and `pending` rows, restricted to the typed allowlist `roundId`, `unitId`, `playerId`, `number`, `result`; a mistyped value is nulled and named in `mistyped` (C3). **`entered_bts_ids`, `resolved_mlb_ids`** |
| `verifier` | only for `observed`: the unchanged legacy verifier's `ok`, `reason` (`match` / `present_unverified` / `no_pick` / `mismatch`), `required_mlb_ids`, and `game_qualified: false`. It is batter-only |
| `qualification` | only for `observed` (C1). Per selection slot, `state`: `confirmed` (the batter's single target-date row names a unit whose captured `feedId` is the slot's game in the same round), or `missing`, `player_unverified`, `ambiguous`, `unit_unverified` (no capture, unknown unit, no `feedId`, or conflicting captures), `wrong_round`, `wrong_game`. Plus `all_confirmed` |
| `references` | only for `already_confirmed`: `{confirmed_by: <attempt_id of the confirming receipt>}`. It is `null` when the marker predates receipts |
| `error`, `failed_at` | only for `fetch_failed`: the exception type and the HTTP status, never the message (messages can carry URLs, headers or tokens), plus the failure time |
| `marker_status` | the marker status this run wrote (`confirmed`, `present_unverified`, `alerted`, `dm_failed`), or null |
| `exit_code` | the run's exit code (null if it raised) |
| `degraded` | names of receipt hooks that failed (their fields may be missing) |
| `published_at` | when the receipt was built |

**Clocks:** in production every time is the wall clock. Under the `--now-et` test override, times are the anchored instant, advanced by process time after the run's own fetch start.

**Never stored:** cookies, the session id (`xsid`), any token, or any exception message.

## Outcomes
| Outcome | Kind | Meaning |
|---|---|---|
| `no_pick_file` | no attempt | no pick file for the date |
| `not_committed` | no attempt | the pick is not a scoreable commit (a preview, or deferred) |
| `outside_window` | no attempt | outside `(cutoff, window]` before the earliest leg's first pitch; `selection` and `cutoff_at` are set |
| `already_confirmed` | no fetch | the marker already says confirmed. It references the confirming receipt and makes **no new observation and no freshness claim** |
| `identity_mismatch` | attempted | the session belongs to another username; `account` records what was seen; no observation |
| `fetch_failed` | attempted | an auth, HTTP or shape failure; no observation |
| `observed` | successful observation | the account was read; `verifier` and `qualification` give the results |
| `incomplete` | — | the run raised before setting an outcome |

## Rules for a consumer (W-entry)
- **Positive confirmation** needs a discoverable `observed` receipt that satisfies all of these:
  - `qualification.all_confirmed` (**not** the legacy `verifier.reason == "match"`);
  - an `account` equal to the configured account;
  - `et_date` and `season` equal to the contest date;
  - `selection` equal to the current committed selection (the same slots, or the same `pick_file_sha256` and `decision_sha256` as the current files);
  - `observation.before_cutoff` true;
  - an empty `degraded`.

  A confirmation proves the matching entry **at `response_completed_at`**, not continuous presence through the cutoff. A later selection, account or date change invalidates it.
- **Not confirmation:** legacy `present_unverified`, and any qualification state other than `confirmed`.
- **Absence** (`no_pick`, `mismatch`, a `missing` slot) comes only from an `observed` receipt. Every no-attempt, `fetch_failed`, `identity_mismatch`, `incomplete`, tombstoned or missing receipt is **unverifiable**, never "not entered".
- **A late observation** (`before_cutoff` false) is reported as late, with its actual completion time.
- **`already_confirmed`** is resolved through `references.confirmed_by` to the original receipt and judged on that. With `null`, it is unverifiable.

## Re-certification (registration line 82)
- **What this touches:** `check_pick_entered` in `src/bts/cli.py`, plus new modules. The loaders were split into read-once plus parse; `load_pick`, `load_decision` and `is_scoreable_commit` behave as before.
- **The certificates to re-run** for the whole producer set (P1, P2, P4, C1/C2) are listed in `docs/ops/reconcile-receipt-v1.md` § Re-certification. Every one still applies at an offset. Re-run them with the frozen runner at the deploy candidate before claiming them current.
