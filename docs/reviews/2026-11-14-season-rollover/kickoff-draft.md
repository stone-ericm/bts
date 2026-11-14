# Kickoff draft: season rollover of the streak state, and three small hardenings

Status: DRAFT written by the lead on 2026-11-14, not yet handed to an author. Tier A (pick-path state). RED-first tests, the gate run, a fresh adversarial Codex review the day it is written, Gemini in parallel. Source of the findings: round 2 of the g2 reviews (`docs/audit/2026-11-14-g2-fixes-codex-r2.md`, `-gemini-r2.md`) and the roadmap ledger entry "Season-start hazards".

Each item is stated as a table of states. Anything not in a table keeps today's behaviour exactly; if the code cannot be made to match a row, stop and say so instead of choosing.

## S1. A prior-season observation must not supply the decision streak
`src/bts/contest_state.py::load_decision_streak_state`. Today the decision streak is `contest.streak` whenever a contest file loads; nothing checks its season, and a 2026 file even reads `fresh` in March 2027 when no newer pick has settled.

Definitions: `now_year` is the America/New_York year of `now`. An observation's season year is `source_date.year` when it has a `source_date`, otherwise the New York year of `recorded_at` when that parses, otherwise unknown.

| Observation selected by `load_contest_streak_state` | Today | Required |
|---|---|---|
| season year equals `now_year` | used | unchanged |
| an unexpired manual override, any dates | used | unchanged (an operator's explicit, time-bounded choice) |
| auto file or expired manual file, season year before `now_year` | used; can read `fresh` | treated as absent: with `require_contest_state=True` raise `ContestStateError` whose message says the observation is from season YYYY; otherwise the model-only fallback |
| season year unknown | used, status `stale` | unchanged |
| no `source_date`, `recorded_at` in `now_year` (the first fetch of a season) | used, status `stale`, streak as recorded | unchanged |

Model-only fallback, same function: the decision's model streak is `load_streak(picks_dir)` only when at least one root-level pick file dated in `now_year` has a settled result (`hit`, `miss`, `void`); otherwise 0. `streak.json` itself is not rewritten here; reconcile's season replay already corrects it after the first result. The saver and the season-best classification are already season-aware and do not change.

## S2. Health: say what is wrong at a season's start
`src/bts/health/contest_state.py::check`, with `expected=True`:

| State | Today | Required |
|---|---|---|
| observation from a season before `now_year` (S1's definition) | no alert when it reads fresh | CRITICAL naming the season and telling the operator to run the fetch for the current season |
| no `source_date`, `recorded_at` in `now_year`, streak 0, and no settled root-level pick dated in `now_year` | CRITICAL "investigate the fetch" | no CRITICAL (a new season's profile has no settled round yet) |
| no `source_date` in any other case | CRITICAL | unchanged |

## S3. Weather sidecars in the probable-pitcher scan
`src/bts/features/compute.py`, the loop that does `pk = int(f.stem)` over every `*.json` under `data/raw/<season>/`: skip a file whose stem is not all digits. Today a `<gamePk>_weather.json` sidecar (written by the manual `bts data enrich-weather`) raises `ValueError` and stops feature computation. No other behaviour changes; the cache contents for numeric stems must be identical.

## S4. A feed download is validated and written atomically
`src/bts/data/pull.py::download_game_feed`. Today the response bytes are written straight to `<gamePk>.json` and an existing file is never re-downloaded, so a truncated or non-JSON body persists and fails every later build and prediction.

| Response | Today | Required |
|---|---|---|
| bytes that parse as a JSON object | written | written to a temporary file in the same directory, then renamed over the target (`os.replace`); final bytes identical to today's |
| bytes that do not parse, or parse to something that is not an object | written and kept | not written; counts as a failed attempt in the existing three-attempt loop; after the last attempt the existing exception path runs (`pull_feeds` logs SKIP and continues) |
| target file already exists | returned without a request | unchanged |

No temporary file may be left behind on any path, and the temporary name must not match `*.json` (the builder and the feature scan glob that pattern).

## S5. The pre-season no-op applies only while the season file does not exist
`src/bts/data/build.py::build_season`, the `eligible_games == 0` branch (landed with the g2 fixes). Today a directory with no requested-type feed and no untyped feed returns `None` whether or not `pa_<season>.parquet` already exists. Once a season file exists the season has started, and a build that finds no regular-season feed is a fault (a renamed game-type code, a partial restore), which the base reported by raising.

| State with no requested-type feed and no untyped feed | Today | Required |
|---|---|---|
| `output_path` does not exist | `None`, nothing written | unchanged |
| `output_path` exists | `None`, the existing file kept, the nightly chain continues to `preview` and `sync-to-r2` | raise (a clear message naming the season and the directory); the existing file untouched |
| the season path exists but is not a directory, or cannot be listed (`PermissionError`), whether or not `output_path` exists | `None`: `Path.glob` swallows the error, so even a directory full of regular-season feeds reads as "no games yet" | raise the `OSError` (`NotADirectoryError`, `PermissionError`); nothing written. List the directory in a way that propagates the error (`iterdir`), keep the missing-directory case a no-op |

The last row was found by Gemini 4 Carbon's known-answer review and reproduced on `main` (`probes/lead_unreadable_season_dir.py`, `probes/carbon_build_parity.py`).

`predict._refresh_season_data` needs no change: it already lets a build exception propagate.

## S6. A day with no selected pick gets a last attempt before each remaining cutoff
`src/bts/scheduler.py::run_day`, the block after the lineup checks ("5. Fallback"), and the missed-pick alert near the top of the file. Found by the final g2 review (`docs/audit/2026-11-14-g2-fixes-codex-final.md`, cold start, item 1) and confirmed by reading the code: the end-of-checks fallback only acts on an existing pick file, so when every check returned no predictions the day ends with no pick, no decision and no alert until the next morning's `bts day-outcome`. On a team's first game of a season there is no prior lineup to project, so this happens whenever the lineups post after the last check: the reviewer's run of the real `run_day` for a 19:10 one-game Opening Night with offset 60 and lineups first available at 18:25 ran checks at 16:10, 17:10 and 18:10 and saved nothing.

| State when the planned checks are exhausted and the pick is not locked | Today | Required |
|---|---|---|
| a pick file exists, undelivered, its fallback deadline ahead, no standing skip | wait for the deadline, then the fallback | unchanged |
| a standing skip (`final_skip_candidate` or `skip_summary`) | no wait | unchanged |
| no pick file, no standing skip, and at least one of today's regular-season games still has its fallback deadline ahead | the day ends silently | for each remaining distinct game start, in order: sleep until that start's fallback deadline (the same `resolve_fallback_deadline_min` and floor the existing wait uses), run one more check, and deliver through the existing fallback path if it selects a pick; stop at the first check that selects a pick or records a skip |
| no pick file and no game with its deadline ahead | the day ends silently | the day ends, as today |

Missed-pick alert: when no pick file exists, no skip is recorded and a regular-season game is still ahead, the alert fires at that game's start minus `missed_pick_alert_min` and says no pick has been selected (today the function returns without alerting when there is no pick file).

RED test to start from: `tests/test_cold_start_integration.py::test_opening_day_late_lineup_gets_a_last_attempt` (an expected failure adopted from the reviewer's probe).

## Out of scope, on purpose
- Rewriting `streak.json` or `contest_streak.json` at rollover; deleting last season's files.
- The early lock on a team's first game (ruling R18).
- An existing corrupt feed file: it still fails the build, as today.
