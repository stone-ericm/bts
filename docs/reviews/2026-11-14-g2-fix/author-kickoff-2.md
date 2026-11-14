# Kickoff 2: after the two reviews of the four fixes (both BLOCK)

Same session, same worktree and rules as `.codex-review/g2-fix/kickoff.md`. Your commit `282c438` is on branch `g2-fix`; keep working on top of it, uncommitted (the lead commits).

## Read first
- `.codex-review/g2-review/report-codex-r1.md` (Codex, xhigh): B1, B2, N1, N5, False greens. Its probes: `/tmp/g2_probe.py`, `/tmp/g2_more_probe.py`, `/tmp/g2_final_probe.py`, `/tmp/g2_health_probe.py`.
- `.codex-review/g2-review/report-gemini-r1.md` (Gemini): B1, Notes 1–4, False greens. Its probes: `.codex-review/g2-fix/gemini_probe_f1_season_start.py`, `gemini_probe_f3_day_outcome.py` (they import from another checkout by path; adapt).

Both reviewers confirmed F3(a), F3(b) and F4 against their contracts. What changes:

## A. Remove F2 completely (the lead's contract was wrong)
Codex B1 shows the merge loses a delivery the unmerged plan makes (prediction transport down until 12:08; the old plan delivers at 12:10; the merged plan's next check is after the cutoff). Any merge removes one of two attempts, and some outage window always makes that attempt the one that mattered. Two cascades five minutes apart on a rare thin-slate day cost one extra cascade and lose nothing.
**Do:** restore `scheduler.plan_run_times` byte-for-byte to `git show main:src/bts/scheduler.py`'s version and delete the four tests you added for the merge. Add ONE test that pins the retained behaviour with Codex's case as its reason: a 13:10 / 14:05 slate at offset 60 keeps both the 12:05 and the 12:10 run.

## B. F1: an unreadable or unrecognised feed is not "no games yet"
Codex B2 / Gemini note 1: a feed whose body is `{}` or lacks `gameData.game.type` is skipped by the game-type filter, so `eligible_games` stays 0 and the build exits 0 where the parent raised.
**Contract:** the no-op applies ONLY when the season directory is missing, or it contains no game feed at all, or EVERY game feed in it carries a readable game type that is positively not a regular-season type. A game feed whose game type is missing or unreadable raises a clear error naming the file, whatever else is in the directory. A regular-season feed that the seven-inning rule or the row extraction then excludes still leads to the existing error path, not to the no-op. (`*_weather.json` sidecars are not game feeds.)
**Tests:** `{}` body; missing `gameData.game.type`; a mix of one spring-training feed and one typeless feed; all spring-training feeds (no-op); a regular-season feed excluded by the seven-inning rule only (raises as the parent did); truncated JSON (still raises). Each through the real CLI.

## C. F1b: the prediction path must survive a season with no games yet (Opening Night)
Codex N1 / Gemini B1: `src/bts/model/predict.py::_refresh_season_data` calls `len(df)` on `build_season`'s return. With no regular-season game yet for the prediction date's year, `bts preview` logs `Failed: object of type 'NoneType' has no len()` and `run_and_pick` returns no predictions on every check: no pick on Opening Night. Before your change the same call failed with the build's exception, so this has always been broken; it is now yours to fix.
**Contract:** when `build_season` returns `None`, `_refresh_season_data` prints one stderr line saying the season has no regular-season games yet, does NOT write the `.refreshed_*` marker, and returns normally; `run_pipeline` then loads the historical `pa_*.parquet` files and proceeds. Nothing else in the refresh changes.
**Also trace and report (fix only if it breaks the cold start, and say so):** every other reader of the current year's parquet or of `build_season`'s return on the prediction path, starting with `src/bts/orchestrator.py` around line 125 (`current_pa`), `predict_local`, `run_and_pick`, and `bts preview`. For each: what it does on a date whose year has no `pa_<year>.parquet`.
**Tests:** `_refresh_season_data` with a season of only non-regular-season feeds (no exception, no marker, returns); the same through `run_pipeline` far enough to show the historical frames are loaded and feature computation is reached (stub the model boundary as Gemini's probe `[3f]` did); `bts preview` through the CLI with the downstream stubbed: no `Failed:` line. Keep `tests/test_refresh_memo.py` green and add the `None` case there.

## D. F3(c): one cron run, at 08:30, for yesterday
Gemini note 4: with two unconfirmed doubleheader game 2s the end-of-day skip record can be written after 01:30 too. **Do:** remove the `30 1 * * *` line; keep `30 8 * * * … --date $YESTERDAY`. Update the cron test and comments. F3(a) and F3(b) stay as they are.
Add one test that documents F3(a)'s accepted limit (Gemini note 3 / false green 2): after a batter-only confirmation, a later unit capture that binds a DIFFERENT game still yields `entered_game_unverified`, because the entry checker does not re-observe; name the test so it reads as a limit, not a guarantee.

## E. Mutation check and focused run
Mutation-check B and C (each guard broken, the named test fails, restored). Run the kickoff-1 focused set plus `tests/test_refresh_memo.py`, the data-build tests and every file you changed; paste the last line.

## Deliverable
`/Users/eric/projects/bts-pp-c1/.codex-review/g2-fix/handover-2.md`, under 100 lines: per item A–D what changed, RED and GREEN lines, mutants; the trace asked for in C; the focused run's last line; anything not done. Then print `G2F2-<first 8 hex characters of the sha256 of the handover file>` and `DONE`.
