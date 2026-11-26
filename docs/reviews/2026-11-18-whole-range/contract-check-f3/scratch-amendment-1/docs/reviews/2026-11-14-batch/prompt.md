# Daily adversarial batch review, 2026-11-14 (tier C, one round)

You are a fresh reviewer. Work read-only in this checkout (it is at the commit under review). Do not read `~/projects/claude-shared`, `~/bts-runs`, or any other checkout. Do not read `.codex-review/final/` (another review's files).

## What is under review
One evidence tool written today and the note that reports its result:
- `scripts/audit/cold_start/rehearse.py` and `scripts/audit/cold_start/run_coldstart.sh`
- `docs/audit/2026-11-14-cold-start-rehearsal.md` and the logs in `docs/audit/2026-11-14-cold-start-rehearsal/` (three runs on the merge `f53c019`, one run of the first date on the pre-fix commit `e3edbf5`)

The note is used as the real-artifact check for a pick-path change: before a season's first game, `bts data build` returns a no-op and `predict._refresh_season_data` predicts from the earlier seasons (`src/bts/data/build.py::build_season`, `src/bts/model/predict.py::_refresh_season_data`). The runs happened on another machine with the real data; you do not have that data and are not asked to rerun them.

Assume the author is wrong. The question is whether the note claims more than the tool and the logs establish.

## What to establish
1. **Claim by claim:** for each factual sentence in the note's table, "Failure direction", "What this does not show" and "Observations", is it supported by a line in the logs or by the code? List every sentence that is unsupported, overstated or wrong, with the log line or code line that contradicts it.
2. **False green:** could the three passing logs have been produced with the fix absent or with the cold state contaminated? Trace what `run_coldstart.sh` and `rehearse.py` do to the working directory between dates (raw feeds left from the previous date, refresh markers, the season file swap, the `data/models` blend cache, the park-drag table, relative paths resolved from the working directory). Is the first date really a cold start, given what the log prints before and after? Is the pre-fix run red for the reason the note states, and from the same state?
3. **What the script substitutes:** it gives every predicted game the status "Scheduled" and passes streak 0 with the saver available to `select_pick`. Name anything in the real path between `run_pipeline` and a delivered pick that this skips and the note does not list under "What this does not show".
4. **The script itself:** any way it could report a pipeline failure as a success (exit status, swallowed exceptions, a log line printed before the thing it describes), or print a value that is not what its label says.

You may write and run small probes against the real functions with synthetic inputs (`.venv/bin/python`, `TZ=America/New_York`, probes under `.codex-review/batch-1114/scratch/`). Do not use the network. Do not edit tracked files, commit or push.

## Rules for findings
This is evidence tooling: a finding is a NOTE unless the note makes a false claim about behaviour that no other check covers; say which it is. Name the sentence, the evidence against it, and the smallest correction.

## Deliverable
`.codex-review/batch-1114/report.md`, under 80 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Unsupported or wrong claims`, `False-green analysis`, `Substitutions not disclosed`, `Script defects`, `What I ran`. When the file is complete, print `BATCH1114-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
