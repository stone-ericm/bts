# Adversarial review, one round, final: pre-season data build, the Opening Night refresh, the day-outcome check, one health message

Tier A (pick-path behaviour). You are a fresh reviewer with no part in writing this and no knowledge of earlier review rounds; none of their reports is in this checkout, on purpose. Work read-only here. The change under review is everything between the base commit `e3edbf5` and `HEAD`: `git diff e3edbf5 HEAD`. Do not read `~/projects/claude-shared`, `~/bts-runs`, or any other checkout of this repository.

This code runs a real contest account's daily pick in 2027 and the nightly data chain it depends on. Assume the author is wrong somewhere and find where. A false alarm or a lost delivery matters as much as a wrong delivery.

## Background
- Project rules and rulings: `CLAUDE.md` ("Work policies", "2027 season"); the rulings ledger in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`. Do not re-litigate a ledgered ruling unless it produces a wrong outcome the ledger does not name.
- The season being prepared starts 2027-03-24 with a one-game slate. Until the day after the first regular-season game there is no `data/processed/pa_2027.parquet`, and `data/raw/2027/` holds spring-training feeds only (game type `S`) or does not exist.

## The contracts
- **Builder** (`src/bts/data/build.py::build_season`, CLI `bts data build`). Everything is the base's behaviour except two no-op cases:

  | Feed shape | With a regular-season feed beside it | With only non-regular or no other feeds |
  |---|---|---|
  | untyped: `{}`, `{"gameData": {}}`, `{"gameData": {"game": {}}}`, a `game` object without `type`, `type` in `None`, `7`, `True`, `""`, `"   "` | skipped with one stderr line; the output equals a build of the same directory without that file (the base skipped these silently) | raises, naming the file; nothing written; an existing parquet untouched (the base also failed here, with a different exception) |
  | fatal on the base: JSON `[]`, JSON `null`, `{"gameData": null}`, `{"gameData": {"game": null}}`, `type` `[]`, `type` `{}` | raises as the base does; nothing written; an existing parquet untouched | the same |
  | readable non-requested type only (`"S"`, `"E"`, …), or no feed, or no season directory | (not applicable) | **no-op: returns `None`, writes nothing, the CLI prints one stderr line and exits 0** (the base raised) |

  The nightly cron line `pull && build && { preview ; sync-to-r2 ; }` is unchanged.
- **Refresh** (`src/bts/model/predict.py::_refresh_season_data`): when the build returns `None` it prints one stderr line, writes no `.refreshed_*` marker and returns, so `run_pipeline` predicts from the earlier seasons' parquets. On the base the refresh raised on every prediction for a date whose season had no regular-season game yet.
- **Scheduler:** `src/bts/scheduler.py` must be byte-identical to the base.
- **Day outcome** (`src/bts/day_outcome.py`, CLI `bts day-outcome`): (a) a new clean state `entered_game_unverified` = a receipt bound to this day exactly as `entered` requires, before the cutoff, with legacy `verifier.reason == "match"` and every qualification slot `confirmed` or `unit_unverified`; `entered` still wins. Accepted limit: this does not prove the right game of a doubleheader. (b) A schedule whose games are ALL postponed or cancelled is `no_games`, with or without a decision record; zero games is unchanged. (c) One cron line, `30 8 * * *`, checking yesterday (ruling R17).
- **Health:** the `late_delivery` message for a refusal with reason `game_not_playable` names the label and the game(s), not the batter.

## What to establish
1. **Builder against the base** (`git show e3edbf5:src/bts/data/build.py`): any directory state where the result (rows written, exception or exit status, files touched, an existing parquet's bytes) differs from the base other than the no-op cells. Any state where the nightly chain now reaches `preview` or `sync-to-r2` on data the base would have refused, or where a prediction cascade now fails that the base's did not.
2. **The cold start.** With the refresh no longer raising, run the real path as far as you can for a date whose season has no regular-season game yet, and for the following day with one prior game: `run_pipeline`, feature computation, training, lineups, selection and the policy lookup at streak 0, `decision.json`, delivery. Say which steps you executed for real and which you only read. Findings in code this change does not touch are NOTES.
3. **Day outcome:** a receipt history where a clean state is reported but the pick was not entered, beyond the accepted limit; a day wrongly called `no_games`; precedence between `entered`, `entered_game_unverified`, `entered_late`, `not_entered`; which contest date `$YESTERDAY` is at 08:30 box time around the DST changes.
4. **False greens:** a test added or changed here that would still pass with the behaviour it names broken. Also: the tests must pass in a checkout that has no `main` branch, and no test may shell out to git.

Write and run your own adversarial probes against the real functions and the real CLI (`click.testing.CliRunner`), with only the network, the clock and the DM transport faked. Run Python as `.venv/bin/python`; focused tests: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite. Do not edit tracked files, commit or push. Probes go under `.codex-review/final/scratch/`.

## Rules for findings
A BLOCK names a concrete input or state, the wrong production outcome, the file and line, and the smallest fix, and says how you demonstrated it. A finding about test-code quality is a NOTE unless the test makes a false claim about behaviour that nothing else covers.

## Deliverable
`.codex-review/final/report.md` with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Cold start: what runs and what fails next`, `Notes`, `False greens`, `What I ran`. Under 130 lines. When the file is complete, print `G2FINAL-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
