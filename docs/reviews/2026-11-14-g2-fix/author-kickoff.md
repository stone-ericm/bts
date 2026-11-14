# Kickoff: fresh author for four defects found by the Gemini review (g2)

You are a fresh AUTHOR session. Work only in this worktree: `/Users/eric/projects/bts-pp-c1` (branch `g2-fix`, at `main`). Test first, focused tests only, no commit/push/stash/branch change; the lead lands the change after a fresh review.

## Read first
1. `CLAUDE.md` sections "Work policies" and "2027 season".
2. `docs/audit/2026-11-14-gemini-known-answer-tests.md` (the table under Test 2) and the findings B3, B4, notes 1 and 2 in `docs/audit/2026-11-14-gemini-known-answer-pickpath-report.md`. The reviewer's probes that reproduce them are in `.codex-review/g2-fix/gemini_probe_*.py` (they import from another checkout by path; fix the path or adapt them).
3. `src/bts/day_outcome.py`, `tests/test_day_outcome.py`, `docs/ops/pick-entry-receipt-v1.md` (the qualification states and the consumer rules).
4. `src/bts/data/build.py` (`build_season`), the `data build` command in `src/bts/cli.py`, `scripts/cron-setup-hetzner.sh` (the 03:00 line and the two `day-outcome` lines), `tests/test_entry_intent.py` (the cron tests).
5. `src/bts/scheduler.py`: `compute_run_times`, `plan_run_times`; `tests/test_thin_slate_checks.py`.
6. `src/bts/health/late_delivery.py`.

## The four fixes and their contracts. Keep each as small as stated; if a contract cannot be met without new apparatus, STOP and say so.

### F1. The nightly chain must not abort before the season's first game
Today `bts data build --seasons 2027` raises when `raw/2027` is missing (`FileNotFoundError`) or holds no regular-season game yet (`RuntimeError: Schema drift`), and the 03:00 cron line is `pull && build && { preview ; sync-to-r2 ; }`, so preview and the R2 sync never run from the box's restore until the day after the opener.
**Contract:** for a requested season with NO regular-season games available (the season's raw directory is missing, or it contains no regular-season game that the build would use), `bts data build` prints one line to stderr saying so, writes nothing, leaves any existing parquet of that season untouched, and exits 0; other requested seasons in the same call still build. A season that HAS regular-season games and fails to build still raises exactly as now. Do not change the cron line's `&&` structure.
**Tests:** through the real CLI (`click.testing.CliRunner`): missing directory; empty directory; a directory holding only non-regular-season feeds (adapt `tests/conftest.py`'s sample feed); a directory with one regular-season game builds as before; two seasons in one call where one is empty; an existing parquet is not touched by the no-op; a season with regular-season games and a genuine build error still exits non-zero.

### F2. `plan_run_times` must not schedule two runs within `cluster_min` of each other
On a 13:10 / 14:25 slate it returns `12:25 -> [101]` and `12:25 -> [202]`: two cascades back to back.
**Contract:** on a thin slate, an early check within `cluster_min` minutes of another planned run is merged into ONE run (the earlier time, the union of `game_pks`, each game listed once). A slate above the thin threshold is still planned exactly as `compute_run_times` plans it, and a one-game slate is unchanged (16:10, 17:10, 18:10 for a 19:10 start at offset 60).
**Tests:** the 13:10 / 14:25 case; no adjacent gap under `cluster_min` across a grid of two-game start times; the existing tests in `tests/test_thin_slate_checks.py` stay green without edits to their expectations, except where they pin the duplicate.

### F3. `bts day-outcome` must not alarm on three ordinary situations
(a) **A batter-only confirmation.** `check-pick-entered` marks the day confirmed when its legacy verifier says `match`, even if the unit capture has not yet bound the game (`qualification` slot state `unit_unverified`); every later run then writes only `already_confirmed`. Today day-outcome reports `entry_unverifiable` and DMs twice. **Contract:** add ONE clean state, `entered_game_unverified`: there is a receipt that binds to this day exactly as `entered` requires (observed, date and season, decision sha, account, empty `degraded`, `before_cutoff` true) whose `verifier.reason == "match"` and whose every `qualification.slots[*].state` is `confirmed` or `unit_unverified`. It is ok (exit 0, no DM); its detail says the game was not verified. `entered` (all_confirmed) still wins when both exist. Every other qualification state (`missing`, `player_unverified`, `ambiguous`, an unsupported duplicate) keeps today's behaviour.
(b) **Every scheduled game postponed or cancelled.** Today the schedule's game count includes void games, so such a day with no decision record reports `no_decision`. **Contract:** when the schedule has games and ALL of them are void (`bts.picks._is_void_detailed_state` on each game's `status.detailedState`), the state is `no_games` with a detail naming how many were postponed, whether or not a decision record exists. A schedule with zero games keeps today's behaviour, including `contradiction` for a commit.
(c) **Timing.** The scheduler can write the end-of-day skip record as late as 23:55 (doubleheader re-checks), after the 23:50 run. **Contract:** the two cron lines both check YESTERDAY, at `30 1 * * *` and `30 8 * * *` (same arguments as the existing 08:30 line). Update the cron test and the comments that name 23:50.
**Tests:** classifier tests for each state and each excluded qualification state; a command-level test with real receipt files for (a) reproducing the reviewer's sequence (an `observed` batter-only receipt, then `already_confirmed` receipts); a command-level test for (b) with a schedule of void games; the cron test for (c).

### F4. The health check must not name the batter of a refused postponed-game pick
`late_delivery.check` renders a `game_not_playable` refusal as `delivery REFUSED for <day>: <batter> (<label>): <games>; ...`. The refusal's own DM deliberately names the game and not the batter, so nothing can be read as a pick. **Contract:** the health message for that reason names the label and the game(s) and no batter. Late-cutoff refusals keep their wording.

## Not yours
Gemini's B5 (an in-loop fallback wait on split-start thin slates) is being ledgered as a limit, not fixed. Do not touch the day loop in `run_day`.

## How to work
- RED before GREEN for every contract line; keep the failing output. Start from the reviewer's probes.
- Mutation check each new behaviour: break it in the source, show the named test that fails, restore.
- Focused runs only: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Before handing over run and paste the last line of: `tests/test_day_outcome.py tests/test_entry_intent.py tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/health` plus every test file you add or change and the data-build tests (`ls tests | grep -i build`).
- Edit only: `src/bts/day_outcome.py`, `src/bts/cli.py` (the `day-outcome` and `data build` commands), `src/bts/data/build.py`, `src/bts/scheduler.py` (`plan_run_times` only), `src/bts/health/late_delivery.py`, `scripts/cron-setup-hetzner.sh`, and tests. Say why if you need another file. Do not read `~/projects/claude-shared`, `~/projects/bts-c2-framing-results` or `~/bts-runs`.

## Deliverable
Write `/Users/eric/projects/bts-pp-c1/.codex-review/g2-fix/handover.md`: per fix, files and functions changed, the tests with RED and GREEN lines, the mutants and what kills each; the last line of the focused run; anything not done or uncertain. Under 120 lines. Then print one line `G2F1-<first 8 hex characters of the sha256 of the handover file>` and `DONE`.
