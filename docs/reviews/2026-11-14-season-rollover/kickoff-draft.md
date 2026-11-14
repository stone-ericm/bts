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

## Out of scope, on purpose
- Rewriting `streak.json` or `contest_streak.json` at rollover; deleting last season's files.
- The early lock on a team's first game (ruling R18).
- An existing corrupt feed file: it still fails the build, as today.
