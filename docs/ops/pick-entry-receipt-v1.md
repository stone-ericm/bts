# Pick-entry receipt, `bts_pick_entry_receipt_v1`

**What it is:** the producer prerequisite **P1** of the C1 rank-2 watchdog build plan (`docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`). It implements registration R4 and the producer receipt contract in `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md`.
- **Producer:** `bts check-pick-entered` (`src/bts/cli.py`), with the receipt code in `src/bts/entry_receipt.py`.
- **Tests:** `tests/test_entry_receipt.py`.
- **What it does not change:** the run's DMs, marker statuses, exit codes, retry policy, authenticated requests and cadence. The existing `TestCheckPickEntered` tests pass unchanged. The marker gains one key, `receipt`.

## Where
- **One file per run:** `data/health_state/pick_entry_receipts/<ET date>/<started %Y%m%dT%H%M%S%f>-<attempt_id>.json`, every run, whatever its outcome. The cron runs `*/15 10-23` under `entry_intent = "enter"`, so about 56 small files a day.
- **Writing:** atomic and durable (a temp file, fsync, rename, then an fsync of the directory). The restic `ops` set (`data/health_state`) backs it up. Nothing prunes it.
- **Publication failure:** prints `entry receipt unavailable` on stderr and changes nothing else. A missing receipt is **unavailable evidence**, never success.
- **The marker** `data/health_state/pick_entry_check.json` now names the receipt of the run that wrote it (`receipt`).

## Fields
| Field | Meaning |
|---|---|
| `schema` | `bts_pick_entry_receipt_v1` |
| `attempt_id` | 32-hex uuid4, one per run |
| `producer` | `{command: "check-pick-entered", revision: <git HEAD of the running checkout or null>}` |
| `et_date`, `season` | the contest date checked (ET) and its year |
| `started_at` | run start (tz-aware ET). It is the same instant the run uses for its decisions |
| `outcome` | see below |
| `detail` | set only when the run raised: `raised <ExceptionType>` |
| `selection` | **expected identity**, once the pick is a committed selection: `slots` (`role` pick / double_down, `batter_id`, `batter_name`, `game_pk`, `game_time`), `delivery` (channel, notification id, Bluesky uri, `delivered_at`, flags), `commit` (`decision.json` sha256, `delivery_status`, `scoreable`), `pick_file_sha256` |
| `cutoff_at`, `minutes_to_pitch` | the submission cutoff (earliest selected leg's first pitch − `SUBMISSION_CUTOFF_MIN`) and the minutes to that first pitch at `started_at` |
| `account` | **verified account**, once the login session returned: `expected_username` (the `--expected-username` option), `user_id`, `username` |
| `observation` | only for `observed`: `response_completed_at` (after all four fetches), `before_cutoff`, the sha256 of each consumed source payload (`profile`, `pending`, `rounds`, `crosswalk`; canonical JSON), the target date's raw `rows` (`profile` predictions and `pending` rows whose round maps to the date), `entered_bts_ids`, `resolved_mlb_ids` |
| `verifier` | only for `observed`: the unchanged verifier's `ok`, `reason` (`match` / `present_unverified` / `no_pick` / `mismatch`) and `required_mlb_ids` |
| `references` | only for `already_confirmed`: `{confirmed_by: <attempt_id of the confirming receipt>}`. It is `null` when the marker predates receipts |
| `error`, `failed_at` | only for `fetch_failed`: the exception type and the HTTP status, never the message (messages can carry URLs or headers), plus the failure time |
| `marker_status` | the marker status this run wrote (`confirmed`, `present_unverified`, `alerted`, `dm_failed`), or null |
| `exit_code` | the run's exit code (null if it raised) |
| `published_at` | when the receipt was written |

**Clocks:** in production every time is the wall clock. Under the `--now-et` test override, times are the anchored instant, advanced by process time after the run's own fetch start.

**Never stored:** cookies, the session id (`xsid`) or any token.

## Outcomes
| Outcome | Kind | Meaning |
|---|---|---|
| `no_pick_file` | no attempt | no pick file for the date |
| `not_committed` | no attempt | the pick is not a scoreable commit (a preview, or deferred) |
| `outside_window` | no attempt | outside `(cutoff, window]` before the earliest leg's first pitch; `selection` and `cutoff_at` are set |
| `already_confirmed` | no fetch | the marker already says confirmed. It references the confirming receipt and makes **no new observation and no freshness claim** |
| `identity_mismatch` | attempted | the session belongs to another username; `account` records what was seen; no observation |
| `fetch_failed` | attempted | an auth, HTTP or shape failure; no observation |
| `observed` | successful observation | the account was read. `verifier` gives the result |
| `incomplete` | — | the run raised before setting an outcome |

## Rules for a consumer (W-entry)
- **Positive confirmation** needs an `observed` receipt that satisfies all of these:
  - `verifier.ok` with `reason == "match"`;
  - an `account` equal to the configured account;
  - `et_date` and `season` equal to the contest date;
  - `selection` equal to the current committed selection (the same slots, or unchanged pick-file and decision hashes);
  - `observation.before_cutoff` true.

  A confirmation proves the matching entry **at `response_completed_at`**, not continuous presence through the cutoff. A later selection, account or date change invalidates it.
- **Not confirmation:** `present_unverified`, which means rows exist but the crosswalk cannot prove identity.
- **Absence** (`no_pick`, `mismatch`) comes only from an `observed` receipt. Every no-attempt, `fetch_failed`, `identity_mismatch`, `incomplete` or missing receipt is **unverifiable**, never "not entered".
- **A late observation** (`before_cutoff` false) is reported as late, with its actual completion time.
- **`already_confirmed`** is resolved through `references.confirmed_by` to the original receipt and judged on that. With `null`, it is unverifiable.

## Re-certification (registration line 82)
- **What this touches:** only `check_pick_entered` in `src/bts/cli.py` (plus the new module). P4, the `entry-intent` command, also added lines to `cli.py`.
- **Affected certificates:** two certified current-defence mutant patches target `fetch_contest_streak` in the same file, `D-I063-2` and `I-0811-b` (`results-f453283`). That function is unchanged, and both patches still apply at an offset (76 lines with P1).
- **Before claiming them current:** re-run them with the frozen runner at the deploy candidate. No other certified patch targets `cli.py`.
