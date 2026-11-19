# Property-testing lane 2 (policy tables, streak rules) and the leads check, 2026-11-18

Eric, 2026-11-18: "work on the property test targets and any other available open items". Two Codex lanes, about ten minutes each. Lane 2's report is `report.md` here (discovery branch `discovery/proptest-policy`, never merged; its harnesses and plain regressions are copied here as reference, they are NOT in `tests/` and most of them fail on purpose). The leads check is `../2026-11-18-leads-check/report.md`.

## What was found, and what the lead decided (ruling R42)

| # | Finding | Touches a live pick? | Decision |
|---|---|---|---|
| P1 | **The reach-57 policy's late-season columns are looked up with the early-season cutpoints.** `pooled_policy.build_pooled_policy` fits the late phase (the last 30 dates) on its OWN quintile cutpoints, `solve_mdp` solves the late days on those bins, and `MDPSolution.save` stores only the early cutpoints. Serving classifies every day with the stored (early) ones. | Yes: the last 30 days of a season, while the objective is still reach-57 | **For Eric** (a policy matter; the shipped policy cannot be re-solved, its profiles are lost). Measured below. Not changed. |
| P2 | Bins that end up empty (tied quantiles) give a policy table narrower than the lookup indexes; training can also abort when the two phases keep different numbers of bins | No: the shipped tables are full width with strictly increasing cutpoints | Ledgered. To fix when a policy is next rebuilt (the reopened longest-streak work, 4b) |
| P3 | The tail rebuild cannot save an artifact when its cutpoints tie; and the tail optimizer can choose a skip that the artifact validator rejects (the validator requires play whenever the best can still be beaten) | No: the shipped tail artifact has one bin | Ledgered with P2. The second half is a real question of intent for a multi-bin tail policy |
| P4 | `streak.json` remembers the last 30 applied dates; a date re-applied after 30 later ones counts twice | Only through `--allow-stale-scoring` or a direct call (the stale-date guard refuses dates more than 2 days old) | Ledgered, not fixed |
| P5 | Out-of-order scoring is arrival-ordered; the fixed exact chain forgives repeated misses | Known and documented (`CLAUDE.md` stale-scoring guard; `exact.build_transition_matrix` docstring) | None |
| — | The streak and saver rules otherwise AGREE across `update_streak`, the season replay, the MDP transitions, Monte Carlo, the exact evaluators and the tail solver: 530 generated comparisons, each shown able to fail with a planted fault | | Confidence, no change |
| L1 | Scheduler state after a rollback (`load_state` quarantines a file with an unknown field) | A repeated skip notice at worst; a delivered pick and its decision survive. No field was added since the last deployed commit `619abda7` | Ledgered, not fixed (a new reader cannot protect a rollback to code without it) |
| L2 | A void result reply is not read back by the feed regeneration | Only the manual recovery tool, and DM picks are not on the feed | **Fixed** (branch `fix/leads-small`) |
| L3 | The probable-pitcher cache is written in place; a torn file is silently read as empty; an entry is never refreshed | A model input (the bullpen feature), through lost history | **Atomic write and a warning: fixed** (same branch). Refresh on a changed feed: NOT built (the raw downloader never rewrites an existing feed, so a stale entry needs an operator re-pull; then delete `data/models/probable_pitcher_lookup.json`) |
| L4 | A fired skip-policy shadow checkpoint can change when an older outcome arrives late | A shadow statistic only | Ledgered, not fixed. It needs a row left pending past day 4 (the checker failing for days, or a cron outage of five or more days) right where a checkpoint is crossed; the only complete fix is to persist each fired look's membership, and the exclusion of aged contradictions that also moves a look is deliberate (`write_status`, round-3 F3). After any multi-day cron outage, read a checkpoint verdict fired near it as provisional |
| L5 | `parse_daily_pick` drops `shadow_model_version` | No | Not a defect |
| — | Wind parsing differs between `compute.py` and `predict.py` ("out to l f") | No: the string never occurs in 2023 to 2025 data, and the feature is shadow-only | Closed, no change |

## P1 measured (lead, on this machine)
The shipped policy's own profiles are lost, so its late cutpoints cannot be recovered. On the saved single-seed 2021 to 2025 backtest predictions (`data/simulation/backtest_*.parquet`; rank-1 `p_game_hit` only, no outcome read), split exactly as `split_by_phase_pooled` does:

| | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| Shipped cutpoints (early) | 0.7960 | 0.8115 | 0.8252 | 0.8407 |
| Early, recomputed here | 0.7954 | 0.8118 | 0.8243 | 0.8412 |
| Late (last 30 dates), recomputed here | 0.7887 | 0.8050 | 0.8201 | 0.8405 |

The recomputed early cutpoints match the shipped ones to 0.001, so these profiles are a fair stand-in. Late cutpoints sit 0.004 to 0.007 lower in the bottom three. With the early cutpoints applied to late days, **18.7% of late days land in a different bin than the one the solver trained that column for** (24.7% on the second saved set, `data/validation/mdp_dd_guardrail_profiles_2026-05-30`), always the adjacent LOWER bin: serving treats those days as one quintile worse than the solver assumed, so in September it is slightly more conservative (more skips, fewer doubles) than the solved policy. The size of the effect on reach-57 probability is NOT measured. It applies only while `streak + 2*days_left >= 57`; below that the tail policy (one bin) decides.

What a fix would take: store both cutpoint vectors in the artifact and classify by phase at serving, which needs a re-solved policy (a new artifact, D7 territory), or re-solve with ONE set of cutpoints for both phases. Neither is possible for the shipped sha `66d15471…`.
