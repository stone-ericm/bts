# Cold-start rehearsal: predicting the first days of a season (2026-11-14)

Tier A real-artifact check for the g2 fixes (merge `f53c019`). Scripts: `scripts/audit/cold_start/rehearse.py` and `run_coldstart.sh`. Logs: `docs/audit/2026-11-14-cold-start-rehearsal/`. Run on the Mac Studio under the EoR terms (nice 19, the yield wrapper, no Victoria 3 process and no hold file at each start).

## The question
On Opening Night the data directory has no file for the new season, and the raw directory holds spring-training feeds or nothing. Does the real prediction path produce a pick from that state, and on the following day when one game has been played?

## What ran
A scratch checkout of the commit under test with `data/processed` holding seasons 2017–2025 only (copy-on-write clones of the studio's files) and no `data/raw`. For each date the script calls the real `predict.run_pipeline` (the season refresh with its public schedule and feed reads, the season build, feature computation, training of the single model and the twelve-model blend, lineups, prediction), then the real `strategy.select_pick` with the tracked policy files at streak 0.

| Date | State | Result on `f53c019` |
|---|---|---|
| 2026-03-25, the 2026 Opening Night, one game | seasons 2017–2025, no raw directory | The refresh pulled 70 spring-training feeds, found no regular-season game, wrote no season file and no refresh marker, and said so on stderr. 18 predictions for the one game, all with a game probability. Policy: double; one game on the slate, so a single. 156 s. |
| 2026-03-26, day two, eleven games | as left by the first run | The refresh pulled the one played game and built a 74-PA `pa_2026.parquet` and its marker. 222 predictions over 11 games. A double. 108 s. |
| 2027-03-24 | seasons 2017–2026 (the frozen 2026 file swapped in), no raw directory | No feeds for 2027-03-20 to 03-23 were final, so no raw directory, no season file, no marker. Zero predictions, because the public schedule lists no game on that date yet: the Opening Night matchup is unannounced and 15 regular-season games are listed on 03-25 (checked 2026-11-14). The season end used was 2027-09-26, and the policy at streak 0 plays at every probability tried (0.70 to 0.88). 109 s. |

The same three runs on the branch commit `dbb78b1`, before the final correction, gave the same outcomes.

## Failure direction
The opener run on the pre-fix commit `e3edbf5`, from the same clean state: `build_season` raises `RuntimeError: Schema drift in build_season` inside the refresh, `run_pipeline` raises, no predictions (`prefix-e3edbf5-rehearse-2026-03-25.log`). That is the Opening Night failure both reviewers reproduced on synthetic data.

## What this does not show
- **Lineups before they post.** The 2026 games are final, so the feeds carry the actual lineups. On a real opener there are no predictions until a lineup posts, because no prior lineup exists to project (roadmap, season-start hazard 6).
- **Delivery, the contest account, the scheduler's day loop.** Nothing was saved, sent or entered. The reviewers ran selection through the delivery chokepoint on synthetic feeds.
- **The box's files.** The park-drag table is absent on the studio (a warning; it feeds the shadow model only), and the box's streak and contest state were not present (hazard 1).
- **Results.** No pick was graded and no outcome was read or analysed; the run reads the played 2026-03-25 game only as the pipeline does, as training rows for the next day.

## Observations
- Every batter carries the display flag `IL? (178d rest)` on the first days, because rest is counted from last season. Only the `PROJECTED` flag is read by the pick path.
- The top probabilities on the two 2026 dates were 0.73 and 0.77, below the policy's lowest bin boundary (the known scale mismatch, sweep candidate C1). At streak 0 the policy plays regardless.

## Reproduce
```
# on the run host, in a scratch checkout with data/processed holding the earlier seasons only
zsh ~/bts-runs/yield_run.sh run.log scripts/audit/cold_start/run_coldstart.sh 2026-03-25 2026-03-26 2027-03-24
```
`run_coldstart.sh` expects the checkout at `~/bts-runs/coldstart/repo` and `rehearse.py` beside it, and swaps the frozen 2026 season file in after the second date.
