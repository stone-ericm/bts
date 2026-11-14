# Kickoff A: season rollover of the streak state, and four data-path hardenings (S1 to S5)

You are a fresh author. Work only in this worktree (`~/projects/bts-sr-a`, branch `sr-state`, created from `main`). Background: two review rounds on 2026-11-14 found that nothing in the code checks which season the saved streak state belongs to, and found four smaller hazards in the data path. Sources: `docs/audit/2026-11-14-g2-fixes-codex-r2.md`, `-gemini-r2.md`, `-codex-final.md`, `docs/audit/2026-11-14-gemini-carbon-known-answer.md`. Project rules: `CLAUDE.md`.

Files you may change: `src/bts/contest_state.py`, `src/bts/health/contest_state.py`, `src/bts/features/compute.py`, `src/bts/data/pull.py`, `src/bts/data/build.py`, and tests. **Do not touch `src/bts/scheduler.py`, `src/bts/strategy.py` or `src/bts/model/predict.py`** (another author is working on the scheduler in parallel).

## Rules
- Tier A (pick-path behaviour). RED first: write each failing test, run it, record that it fails for the stated reason, then implement. Report the RED counts.
- Each item is a table of states. A row marked "unchanged" must behave exactly as today; if you cannot make the code match a row, stop and say so in the handover instead of choosing.
- Smallest change that satisfies the tables. No refactors, no renames, no new configuration keys, no new files under `src/` unless a row cannot be met without one.
- Tests must not use the network, must not shell out to git, must not depend on a branch name, and must pin `today=`/`now=` (never `date.today()` or the wall clock).
- Run Python as `.venv/bin/python`; focused tests: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite (this machine is throttled; the lead runs the gate elsewhere).
- For each item, break the new behaviour once on purpose (a mutation), show a test fails, and restore the source exactly. Report which test caught which mutation.
- Do not commit, push, or touch any other worktree. Do not read `~/projects/claude-shared` or `~/bts-runs`.
- The reviewers' probes that demonstrated these defects are in `docs/reviews/2026-11-14-season-rollover/probes/` (written against older commits and scratch paths; raw material for your RED tests, not tests themselves).

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

## Deliverable
`.codex-review/sr/handover-a.md`, under 80 lines: per item, what changed (file and function), the RED count and reason, the mutation and the test that caught it; the focused test command and counts; `git diff --stat`; any table row you could not match. When the file is complete, print `SRA-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
