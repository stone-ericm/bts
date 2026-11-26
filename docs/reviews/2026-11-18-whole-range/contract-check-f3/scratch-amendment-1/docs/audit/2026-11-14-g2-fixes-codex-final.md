# Verdict

**SIGN WITH EDITS** for `e3edbf5..c339361a`. No new production BLOCK reproduced.
The changes match the stated contracts in the cases executed. Address the test-isolation
and documentation notes below; this is not a claim that Opening Night is fully rehearsed.

# Blocking findings

None.

- Builder: 220 directory/input combinations compared the actual base implementation with
  HEAD, then exercised the real `data build` CLI. Only the specified no-op/untyped cases
  changed. Fatal bodies/types, truncated JSON, empty regular feeds and seven-inning-only
  regular feeds still failed; failure paths preserved existing output bytes and mtimes.
  Successful frames matched the base. Untyped feeds beside regular feeds were logged.
- The unchanged nightly `&& build && { preview ; sync-to-r2 ; }` therefore advances on the
  newly authorized no-op cases. No additional successful CLI case was found in that matrix.
- `scheduler.py` is byte-identical to the base: Git blob
  `7e05b6db7689b0feefceacfb4e7d882119cddb7a` for both.
- Receipt precedence was independent of input ordering across 33 permutations:
  `entered` > `entered_game_unverified` > `entered_late` > `not_entered`.
  Existing focused tests also reject wrong bindings and disallowed qualification states.
  No false clean beyond the declared game/observation-time limits was reproduced.
- Real `day-outcome` CLI calls through the real schedule reader correctly filtered spring
  games, accepted postponed/cancelled-only slates, and kept Suspended, Delayed Start and
  Final games distinct from `no_games`. Zero-game precedence remains unchanged.
- The health-message test passed: label and both games appear; batter names do not.

# Cold start: what runs and what fails next

**Executed**, with synthetic feeds, real shipped policy NPZs, and only network/clock/DM
substitutions in the functional probes:

- Built historical 2025/2026 parquets from 32 small feeds (64 PAs total).
- March 24, spring-only current-season cache: real pull discovery, no-op rebuild, parquet
  loading, feature computation, single-model plus all twelve blend fits, feature lookups
  and live-slot parsing ran. No current-season parquet or refresh marker was produced.
- Before posted lineups, predictions were empty because neither team had a prior 2027
  regular-season lineup. With posted lineups, real `run_and_pick` selected a single from
  the shipped MDP at streak 0; the real delivery chokepoint saved the pick, scheduler lock
  and scoreable v3 `decision.json`, with delivery status `delivered` and a simulated DM.
- March 25, one prior regular game: real refresh built its two PA rows and marker; feature
  computation, all thirteen fits, selection and simulated delivery succeeded again.
  A separate real slot-fetch probe also produced projected lineups for the two teams that
  played March 24. This does not provide prior lineups for the other teams' season openers.
- The measured synthetic top probability was about 0.956; it is not model-quality evidence.

**What fails next, both pre-existing NOTES:**

1. **Late first-day lineups can still lose the day.** For a 19:10 Opening Night start,
   offset 60 and lineups first available at 18:25, the complete real `run_day` executed
   checks at 16:10, 17:10 and 18:10, then made no further game-feed request. It saved no
   pick or decision. `model/predict.py:382` has no prior same-season lineup to project;
   `scheduler.py:3095` requires an existing pick to arm the final fallback, and
   `scheduler.py:106` also returns before the missed-pick warning when no pick exists.
   Smallest follow-up: retry a slate with no selected pick before the cutoff, even when
   there is no cached pick. Evidence: `scratch/late_lineup_day.json` and its named probe.
   The synthetic run disabled end-of-day health checks; no claim is made about their DMs.
2. **Weather sidecars break feature computation.** Adding `9001_weather.json` after a
   successful pipeline made the next real pipeline raise `ValueError` at
   `features/compute.py:74` (`int(f.stem)`, outside the exception guard).
   The builder skips that file correctly. Skip sidecars before parsing the stem here too.
   This feature code is unchanged, so it is not a regression introduced by this diff.

The successful delivery probe exercised the chokepoint directly after `run_and_pick`;
only the no-pick scenario exercised the entire day loop. No production historical
parquets, live contest account, real DM transport, box cron daemon or grading run was used.

# Notes

- **The no-Git-in-tests requirement is not literally met.** Auditing subprocess launches
  while running `tests/test_day_outcome_entry_sequence.py:95` observed six
  `git -C <this checkout> rev-parse HEAD` calls. They originate in production receipt
  provenance (`entry_receipt.py:305` -> `picks.py:55`), not explicit test Git commands.
  Use deterministic test provenance if the requirement includes indirect calls. This is
  a test-isolation NOTE, not a production blocker. Evidence: `scratch/false_green.json`.
- **Documentation lags the cron change.** `CLAUDE.md` and the roadmap still describe
  23:50 plus 08:30, and the checked-in ledger has no R17 row. Update them to the explicit
  prompt's once-daily 08:30 contract; the installed-line tests passed under both intents.
- **DST:** on a correctly configured ET box, 08:30 on 2027-03-14 checks **2027-03-13**;
  08:30 on 2027-11-07 checks **2027-11-06**. Adjacent-day checks also select the preceding
  contest date. `scratch/dst.py` verified both civil-day and absolute-24-hour subtraction;
  the possible one-hour difference cannot cross midnight at 08:30. GNU `date` was not
  available on this Mac, so the literal box executable was not run.

# False greens

- `tests/test_refresh_memo.py:143-146` replaces training, lookups and prediction. Both new
  cold-start/preview tests still passed when I replaced the real `_fetch_game_slots` with
  an unconditional exception. Their feature-loading claims are supported, but their green
  status cannot establish usable first-day predictions or delivery. Retain an integration
  case with a posted lineup and the real downstream path, such as the probe supplied here.
  Evidence: `scratch/false_green.py` and `scratch/false_green.json`. NOTE, not BLOCK.
- This checkout has only `g2-final-review`, with no `main` branch. All 252 focused tests
  passed in that state; no branch-dependent failure was hidden by a local `main`.

# What I ran

- `TZ=America/New_York .venv/bin/python -m pytest` on the seven requested changed test
  files: **252 passed**, with `-q -p no:cacheprovider`; no whole-suite run.
- `scratch/test_adversarial.py`: **5 passed**, including 220 builder comparisons and CLI
  runs, cold training/cascade/delivery, next-day projection, late-lineup day loop and
  sealed-receipt/schedule CLI probes. The only warning was joblib's CPU-count fallback.
- `scratch/false_green.py`: two cold-start tests stayed green under the injected fault;
  two real receipt CLI tests passed while their Git subprocesses were recorded.
- `scratch/dst.py`, base/current source inspection, module-path verification and scheduler
  blob comparison. Outputs and runnable probes are under `.codex-review/final/scratch/`.
- Tracked files were not edited; no commit, push, external send or paid call was made.
