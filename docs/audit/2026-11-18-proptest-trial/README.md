# Property-testing trial, 2026-11-18 (one lane, discovery only)

Eric's request (`request.md`, relayed from his MacBook session) after reading https://recursion.wtf/posts/agents-and-property-tests/. One Codex agent, one target (`features.compute.compute_all_features`), on branch `discovery/proptest-features` (studio worktree `~/projects/bts-work/proptest`), which is NOT merged. Hypothesis 6.151.12 was installed into that worktree's environment only; it is not a project dependency. The files here named `harness_*` and `regression_*` are copies for the record: they are not collected by pytest (no `test_` prefix) and `harness_*` need Hypothesis.

## Cost
9 minutes 12 seconds of agent time (14:51 to 15:01 ET), of which 287 seconds were the 2,000-example run. Token usage was not visible to the agent. The lead spent about ten minutes on set-up and approvals and five on the measurement below.

## What it found
- **The prefix-invariance property holds** on 2,000 generated schedules (two seeds), with fixed matchup structure, after the property was shown to fail on three planted leaks in three features (`batter_hr_7g`, `batter_pitcher_shrunk_hr`, `pitcher_catcher_framing`).
- **Independent from-scratch reference calculations agree** for the seven-game-date mean, the rookie shrinkage rule, the expanding platoon rate and the batter-pitcher shrinkage (100 examples plus three boundary cases).
- **One confirmed violation: `opp_bullpen_hr_30g`** (a production feature, `FEATURE_COLS`). Its value for a plate appearance on date D exists only if the opposing team used a reliever ON date D (`compute.py`, the team bullpen block: the rolling series is built over reliever dates only and merged back on the exact date). So whether the value is missing depends on what happened that day. Regression: `regression_bullpen_same_day_availability.py`.

## How much it matters (lead's measurement on real 2025 data, `measure_bullpen_2025.py`)
Of 4,493 team-dates after the ten-date warm-up, **30 (0.67%) have the value missing only because the pitching team used no reliever that day: 941 plate appearances, 0.52% of the 2025 rows.** The hit rate on those rows is **0.142 per PA against 0.221** on the rest: they are complete games. So in training and in backtests a missing value marks a small set of rows where hits were much scarcer, which the model can learn; at serving the value is never missing (`predict.py` uses the team's latest value), so the learned split never fires live. The effect is a small optimistic bias in backtests and a train/serve difference, not a wrong live pick. Not yet measured: the size of the bias in P@1.

## Not covered by the trial
Windows longer than 34 game-dates (the 60 and 120 windows never expire in the generated frames); real artifact contents; serving; the request's other leads (see `request.md`). The agent's suggestion for a second run: the tied-quantile case in `simulate/quality_bins.py`.
