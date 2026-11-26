# Cold-start rehearsal: predicting the first days of a season (2026-11-14)

Real-artifact check for the g2 fixes (merge `f53c019`). Scripts: `scripts/audit/cold_start/rehearse.py` and `run_coldstart.sh`. Logs: `docs/audit/2026-11-14-cold-start-rehearsal/`. Reviewed in the daily batch review (`docs/audit/2026-11-14-batch-review-codex.md`: SIGN WITH EDITS, all notes); this version of the note and the scripts carries its corrections, and the logs at the top of the folder come from the corrected scripts.

It is a rehearsal of prediction and selection from a cold data directory. It is not a rehearsal of a live 2027 pick, and it is not an automated acceptance gate.

## The question
On Opening Night the data directory has no file for the new season, and the season's raw directory holds only games that are not regular-season games, or does not exist. Does the real prediction path produce predictions from that state, and on the following day when one game has been played?

## What ran
A scratch checkout on the Mac Studio with `data/processed` holding `pa_2017.parquet` to `pa_2025.parquet`. Before each first date the orchestration script (`rerun-1114.sh` in the logs folder) deleted `data/raw`, any 2026 or 2027 season file, the refresh markers, cached blends and the probable-pitcher cache. `rehearse.py --require-cold` then refused to start unless the season had no parquet, no marker and no raw directory, and printed what it found. For each date the script calls the real `predict.run_pipeline` (the season refresh with its public schedule and feed reads, the season build, feature computation, training of the single model and the twelve-model blend, lineups, prediction) and then `strategy.select_pick` with the tracked policy files.

Results with the corrected scripts on the pushed commit `07780d3` (its `src/` is identical to the merge `f53c019`):

| Date | Start state (as logged) | Result |
|---|---|---|
| 2026-03-25 | seasons 2017–2025; no `data/raw/2026`; no marker; no probable-pitcher cache | The refresh fetched 70 feeds, none eligible for a regular-season build (counted in that directory after the runs: 65 spring-training and 5 exhibition feeds, plus the one regular-season game the second date added). No season file and no marker were written; stderr says the season has no regular-season game yet. 18 prediction rows for one game, all with a finite probability. Policy: double; with one predicted game the selection is a single. Pipeline 168 s. Exit 0. |
| 2026-03-26 | as left by the first date (70 feeds, the cache now present) | The refresh added one feed (the regular-season game of 03-25) and built a 74-PA, one-game `pa_2026.parquet` with its marker. 222 prediction rows over 11 games. A double. Pipeline 113 s. Exit 0. |
| 2027-03-24 | seasons 2017–2026 (the studio's `pa_2026.parquet`, sha256 `50b39b11…`, copied in; both hashes are in `run.log`); no `data/raw/2027`; the 2026 raw feeds still present | No final game in 2027-03-20 to 03-23, so no 2027 raw directory, season file or marker. Zero prediction rows. The season end used was 2027-09-26, and the policy at streak 0 chose to play at each of the five probabilities tried (0.70 to 0.88), from the base table (both policy files loaded). Pipeline 109 s. Exit 0. |

Why zero rows on 2027-03-24: the logs do not show the cause. Checked separately the same day, the public schedule endpoint lists no game of any type on 2027-03-24 and 15 regular-season games on 03-25 (`schedule-2027-03-20_27.txt`); MLB's schedule release (mlb.com/news/mlb-2027-schedule-released) names an Opening Night game on 03-24 with the matchup undecided. Zero rows would also result from a listed game with no posted lineup.

Earlier runs with the first version of the scripts gave the same picks and counts: on the branch commit `dbb78b1` and on the merge `f53c019` (`first-runs/`).

## Failure direction
The opener on the pre-fix commit `e3edbf5`, from the same logged start state (seasons 2017–2025, no raw directory, no marker, no cache; the same checkout and environment, reset by the same script): the refresh fetches the same 70 feeds, `build_season` raises `RuntimeError: Schema drift in build_season`, `run_pipeline` raises, and the script exits 2 with `RESULT: failed (the pipeline raised; expected prediction rows, got none)` (`prefix-e3edbf5-rehearse-2026-03-25.log`). Both reviewers reported the same failure from synthetic feeds in their own runs (`docs/audit/2026-11-14-g2-fixes-codex-r2.md` and `-gemini-r2.md`, the cold-start sections).

## What this does not show
- **An enterable pick.** The script gives every predicted game the status "Scheduled"; the real games are final, and the real selector returns no eligible pick for a final game. Selection here is a counterfactual on eligibility. Postponements, warm-up, started games and status-read failures are not exercised.
- **The real streak state.** Streak 0, saver available, doubles allowed and an empty picks directory are supplied as arguments. `load_decision_streak_state`, its freshness rules, existing-pick reuse and the season rollover of the real files are skipped (roadmap, season-start hazard 1).
- **What was known before the game.** The feeds are completed games: actual starting pitchers, weather and boxscore batting orders stand in for pregame information, and a boxscore lists substitutes in a batting slot beside the starter, so the row counts are not a pregame nine a side. On a real opener there are no rows at all until a lineup posts, because no prior lineup exists to project (hazard 6).
- **Everything after selection.** The optional post-pipeline calibration, the confirmation and gap gate, the cutoff filter, the delivery-time status, cutoff and state checks, saving, retries, the DM and the contest entry. The reviewers ran selection through the delivery chokepoint on synthetic feeds; those runs are theirs and are recorded in their reports, not here.
- **The box.** The park-drag table is absent at the scratch checkout's default path (it feeds the shadow model only). The box's probable-pitcher cache, streak files and contest state were not present.
- **Outcomes.** No pick was graded or compared with its outcome. Completed feeds were read for retrospective prediction inputs and, on day two, as the prior day's training rows.
- **The run conditions.** That the runs were at nice 19 under the yield wrapper with no Victoria 3 process and no hold file is the operator's report; `run.log` records only start and exit.

## Observations
- The displayed batters carry a long-rest display flag on these dates (`IL? (178d rest)` to `IL? (205d rest)` in the twelve rows printed per date), because rest is counted from last season. The pick path reads only the `PROJECTED` flag string; days of rest is itself a model feature, and past openers are in the training data.
- The top probabilities on the two 2026 dates were 0.73 and 0.77, below the policy's lowest bin boundary (0.796; the known scale mismatch, sweep candidate C1). At streak 0 with the saver available the policy plays in every bin.

## Limits of the tool
- `run_coldstart.sh` leaves the first season's raw feeds in place for the third date; the third date is cold for its own season only.
- The check for a cold start covers the season parquet, the refresh marker and the season's raw directory. The probable-pitcher cache and the park-drag table are reported, not required to be absent; the orchestration script removed the cache before each first date.
- The scripts print file names and counts, not content hashes, except for the swapped season file.

## Reproduce
```
# on the run host; the checkout at ~/bts-runs/coldstart/repo with data/processed holding the earlier seasons only,
# rehearse.py and run_coldstart.sh copied beside it
zsh ~/bts-runs/yield_run.sh run.log ~/bts-runs/coldstart/run_coldstart.sh 2026-03-25 2026-03-26 2027-03-24
```
The first date must start cold and must produce predictions and a pick, the second must produce predictions and a pick, and any stage that raises fails the run; the script exits non-zero otherwise.
