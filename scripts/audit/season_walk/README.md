# The 2027 season walk (closing test of the whole-range review, part d)

`walk.py` runs every calendar date from 2027-03-01 to 2027-10-05 through the scheduler's decision call,
`orchestrator.run_and_pick`, with the shipped policy artifacts (`data/models/mdp_policy.npz` sha
`66d15471…`, `data/models/mdp_tail_policy.npz` sha `dc5d0c99…`, both asserted per record), for 21
synthetic contest states per date: 18 dated observations (`bts set-contest-streak --date`) and 3
automatic ones (`bts season-contract enable`, `bts fetch-contest-streak` with stubbed MLB calls,
`bts saver-state --init`), each installed through the real CLI in a fresh temporary picks directory
under a pinned ET clock. Candidates: three synthetic batters in three different games, the top one
at 0.96 or 0.72. No network (sockets refuse), no real data, no 2026 or 2027 outcomes.

Run from the repository root: `.venv/bin/python -m scripts.audit.season_walk.walk --out <file.json>`
(about 15 seconds; exit 1 when it reports a problem). Tests: `tests/scripts/season_walk/`.

## What it checks
- days left equals the calendar dates from the date through the finale 2027-09-26 inclusive (0 after),
  and the policy's effective count is that clamped to the table's 180 days;
  `SEASON_END_DATES[2027]` is the day after the finale;
- every date yields a pick or a NAMED skip (a skip carrying the policy's structured decision), never an
  error or an empty without a decision; the decision source is the MDP (the shipped artifacts loaded);
- the saver the policy sees follows the change set 3 rule; a pick records the state kind it was chosen on;
- after the finale every action is a skip; on the finale streak 56 plays;
- in season, a skip's operator notice (`scheduler.format_skip_dm` over `build_skip_summary`, built as
  `run_single_check` builds it from the same decision) never says the best candidate is below a bar it
  clears;
- `boundary_checks`: the 2028 fallback boundary (09-28) and its `tail_policy` WARN, no WARN in 2027, the
  day counts at the year boundary, the `tail_policy` health check silent with the shipped artifacts on
  2027-09-10, and the 2027 automatic switch accepted at 23:59 ET on 2027-12-31 and refused after
  midnight ET.

## Limits (what it does not show)
- It calls `run_and_pick`, not `scheduler.run_day`: lineup timing, delivery, the cutoff, fallbacks,
  restarts and the schedule fetch are outside it. Pre- and post-season dates are run as if candidates
  existed; on the box the regular-season schedule has no games then, so the scheduler never selects.
- Synthetic candidates in three separate scheduled games; no doubleheader, postponement or void.
- 21 states per date, two candidate strengths; the policy's behaviour between them is not sampled.
- The day count is calendar days, as the policy counts them; MLB off-days are not modelled.

## Result on `bc143131` (`results/walk-bc143131.txt`, records in `results/walk-bc143131.json.gz`)
4,599 runs; every check passes except one: 841 in-season skip notices say a 96% candidate is "below the
streak-N bar of 82.5%" or "below the ~80% bar" (streaks 43 to 56, the shipped table skipping its top
quality bin while it plays the bin below, or skipping every bin). See the closing test report.

## Extension for the closing test re-run (`--extended`; branch `audit/season-walk-h2g`, not merged)
`--extended` walks `DEFAULT_STATES` plus `EXTENDED_STATES` (17 more: automatic 57/57, 58/60 and 120/120; dated 57,
59, 100/150 and the 400 cap edge; candidate-strength pairs 0.96/0.72 at 41, 45, 53 dated and 50, 55 automatic) and adds:
- `guard_pair_problems`: in season, for the same date and contest state, the 0.96 candidate never skips where the
  0.72 candidate plays (G, on the walk's own decisions);
- `table_guard_check`: every cell of the shipped table (streak 0-56, days 1-180, both savers, every bin through its
  midpoint) through `mdp.lookup_action_guarded`: no bin skips while a weaker bin plays, playing cells unchanged, and
  the number of guarded cells (G's 1,511); the unguarded lookup reports exactly those 1,511 (failure direction);
- `cap_checks`: H2's 400 on both writers: dated and automatic 400 accepted by the live loader, 401 refused;
- `sequence_57_58`: H and H2 together in one picks directory on the automatic path: MLB 55, 56, 57, 58, 59 on
  2027-08-01..05 through the real fetch, the decision call, the pick saved and read back from disk and judged by the
  live saved-pick check, the health contest-state check, and the model streak file graded past 57. Without H2 (caps
  patched back to 57) the walk reports problems at 58 (test).
Result on FINAL2 `d0bdcdba`: `results/walk-final2-extended.txt` (8,760 runs, 0 problems). Limits as above; the
sequence's MLB streak is synthetic (the policy skipped at 55 and 56, so the numbers are not a real run of hits); the
health check is the contest-state check only; `bts day-outcome` is not run here.
