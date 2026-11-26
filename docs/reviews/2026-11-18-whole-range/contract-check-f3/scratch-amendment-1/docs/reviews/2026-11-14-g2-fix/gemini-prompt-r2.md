# Adversarial review, round 2 of 2 (final): the g2 fixes after both round-1 reviews

Same rules as round 1. The change under review is now the top TWO commits of this checkout over base commit `e3edbf58`: `git diff e3edbf58 HEAD` (the second commit, `git show HEAD`, is what changed since your round 1).

## What changed since round 1, and why
Two reviewers blocked round 1 (Codex at xhigh and Gemini; archived on `main` as `docs/audit/2026-11-14-g2-fixes-codex-r1.md` and `-gemini-r1.md`; if this checkout lacks them, the points are summarised here).
- **A. F2 is removed.** `scheduler.plan_run_times` is byte-identical to the base `e3edbf58` again (check it). Reason: merging two nearby checks removes one attempt, and a prediction outage ending between them then loses a pick the unmerged plan delivers. Ruling R16 ledgers the duplicate cascade as accepted.
- **B. The pre-season no-op in `data/build.py::build_season` is narrowed.** No-op (return None, write nothing) ONLY when the season directory is missing or has no game feed, or every game feed has a readable game type and none is a requested type. A feed with a missing or unreadable game type is skipped beside requested-type feeds exactly as the parent skipped it (plus one stderr line), and raises when there is no requested-type feed. Everything else is meant to be the parent's behaviour.
- **C. `model/predict.py::_refresh_season_data` handles the `None` return:** one stderr line, no `.refreshed_*` marker, return; `run_pipeline` then predicts from the historical parquets. Before this, the refresh raised on any prediction date whose season had no regular-season game yet, so there were no predictions on Opening Night.
- **D. `bts day-outcome` runs once, at 08:30, for yesterday** (ruling R17). F3(a), F3(b), F4 are unchanged from round 1.

## What to establish
1. Your round-1 findings: closed, by running the code?
2. **B against the parent** (`git show e3edbf58:src/bts/data/build.py`): any directory state where the result (rows written, exit status, files touched) differs from the parent other than the two no-op cases. Any state where the nightly chain now proceeds to `preview` / `sync-to-r2` on data the parent would have refused to publish, or where a prediction cascade now fails that the parent's would not.
3. **C, the cold start itself.** With the refresh no longer raising, what is the NEXT thing that fails on a prediction date whose season has no games yet (Opening Night 2027-03-24, one game) and on the following day (one prior game)? Follow the real path as far as you can run it: `run_pipeline` after the refresh, feature computation for players with no current-season rows, lineups (`_fetch_prior_lineup` has no prior game), the blend cache, selection and the policy lookup (`bts.strategy` with streak 0 and a full season of days left), `decision.json`, delivery. Say which steps you executed for real and which you could only read.
4. **Beyond the diff (findings here are NOTES for a follow-up, not blockers of this change):** anything else on the scoring, streak, health-check or cron paths that breaks or alarms on the first days of a season: no prior pick files, no `pa_<year>.parquet`, an empty `raw/<year>`, last season's `streak.json`/`applied_dates`, checks with look-back windows.
5. False greens in the tests changed since round 1.

Write and run your own probes; do not rely on the author's handovers.

## Deliverable
Sections: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-1 findings`, `Blocking findings`, `Cold start: what runs and what fails next`, `Notes`, `False greens`, `What I ran`. Under 150 lines. A BLOCK names a concrete input or state, the wrong production outcome, file and line, the smallest fix and how you demonstrated it.

This checkout is detached at the commit under review; "main" in the text above means the base commit `e3edbf58`. Your own round-1 report is `.codex-review/agy-g3/report-g3.md`. The other reviewer's round-1 blocking findings were: (B1) merging two nearby thin-slate checks lost a delivery the unmerged plan made when a prediction outage ended between them; (B2) a feed with missing game-type metadata was treated as "no regular-season games" and let the nightly chain publish.

Report file: `/Users/eric/projects/bts-agy-g3/.codex-review/agy-g3/report-g3-r2.md`. Probes go under `/Users/eric/projects/bts-agy-g3/.codex-review/agy-g3/scratch/`. Do not commit, push, or edit tracked files. When the file is complete, print `AGYG3R2-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
