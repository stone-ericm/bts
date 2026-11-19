# Property-testing discovery · policy tables and streak rules

Started **2026-11-18 20:52:26 EST**. Ended **2026-11-18 21:03:11 EST**. The local branch is `discovery/proptest-policy`, based on `df454725`. Discovery commits are `4ba7a928`, `400ecbf1`, and `7b513123`. Product code and dependencies were unchanged. All examples were synthetic. The only checked-in data read for this discovery was the two policy artifacts. No network, box, account, or push was used.

## What ran

Hypothesis 6.151.12, Python 3.12.16, seed **18112026** for every run, and a disabled Hypothesis example database in the harnesses. All pytest runs used `~/bts-runs/yield_run.sh` serially with the kickoff environment. The sandbox denied the wrapper's requested priority adjustment. The wrapper still ran pytest and recorded its exit status.

- **Policy tables.** Generated constant, tiny, few-level, outlier, and distinct profiles. Covered single-phase, pooled early and late, and late-only tail training. Exposed exact columns with sentinel actions for `QualityBins.classify`, base lookup, tail classification and lookup, and both adapters used by the live decision path. Tested probes on and just to either side of each boundary. Checked training assignments by the nonempty bins' observed ranges. Base persistence used fixed-boundary bins to keep empty columns and preserve a meaningful action comparison. Tail persistence used a supported two-bin control with different actions across the boundary.
- **Streak rules.** Ran one reference fold over synthetic hit, miss, double, mixed, void and skip outcomes. Exercised saver boundaries, repeated misses, duplicated dates, season rollover and out-of-order application. Used an isolated temporary directory per scoring or replay example. Where the evaluator returns a probability or an optimal value, computed the expected result from the same fold.

In the final unmodified run, **11 checks passed and 17 failed** in 38.16 seconds of pytest time (41 seconds elapsed in the wrapper). The 17 failures comprised six generated checks and 11 plain regression cases. Five earlier checks with harness errors were superseded and deselected. The counts below are valid trials including generation and shrinking, excluding invalid draws and final replay of a failure. They count repeated calls made while shrinking.

| Target and check | Passing | Failing | Hypothesis generation and shrink time |
| --- | ---: | ---: | ---: |
| Policy widths, ordinary and pooled single-phase | 6 | 12 | 0.30 s |
| Policy pooled phase width | 6 | 860 | 30.45 s |
| Policy lookup monotonicity, assignment and base save/load | 180 | 0 | 1.12 s |
| Policy tail quantile rebuild | 5 | 47 | 1.30 s |
| Policy tail save/load, unconstrained two-bin input | 0 | 17 | 0.39 s |
| Policy tail save/load, admissible two-bin control | 16 | 0 | 0.48 s |
| Streak update, replay, MDP, Monte Carlo, fixed exact and tail solver | 480 | 0 | 1.76 s |
| Streak exact policy-table evaluation | 50 | 0 | 0.32 s |
| Streak date retries after marker eviction | 0 | 16 | 0.93 s |

Target totals were **1,149 policy trials** and **546 streak trials**. Logs are under `.codex-review/proptest2/`. The standalone regressions use pytest and synthetic inputs with no Hypothesis import.

## Scratch faults and red results

Each mutation replaced one source fragment in a temporary copy, compiled that copy, and patched the function in the test process. These mutations left the checkout intact.

| Property or comparison | Fault and red result |
| --- | --- |
| Policy A · table width | Removed the last solver column with fixed-boundary control bins. Both trainers' width checks went red at `[0.1]` with 4 columns for 5 indices. |
| Policy B · monotonicity and bounds | Reversed the tail boundary predicate. The ordered probes for `[0.1]` moved backwards. |
| Policy C · assignment | Changed base lookup from `>=` to `>`. Tied confidence 0.1 was assigned to 4 and the base lookup and live base adapter returned 0. |
| Policy D · save/load | Shifted loaded boundaries by 0.01. Base control changed single to skip at the cutpoint. The admissible tail control changed double to single for the boundary from `[0.1, 0.6]`. |
| Persisted streak | Made any all-hit day add one. The first double produced streak 1 instead of 2. |
| Season replay | Reset on void. Four doubles and a void replayed to 0 instead of 8. |
| MDP transitions | Raised saver lower bound to 11. Five double hits followed by a miss reset a streak of 10 and left the saver available. |
| Monte Carlo | Changed double hits to +1. Four doubles gave best 4 instead of 8. |
| Fixed exact matrix | Changed double hits to +1. Both one-row and whole-sequence checks went red. |
| Exact policy-table evaluation | Preserved the saver after a caught double miss. Eight certain single hits, three stochastic doubles, and the closing singles gave 0.5 versus 0.375. |
| Tail expected-best solver | Preserved the saver after a caught miss. Starting at streak and best 8 with four days and rates 0.75/0.5 gave 10.5 versus 10.4375. |

`bins-mutants.log` produced five red cases in 14.86 s. `streak-mutants.log` produced six valid red cases and one fixture broadcast error in 7.32 s. The corrected exact-policy mutation produced its assertion failure in 1.35 s (22 valid trials including shrinking). The first tail round-trip mutation run stopped at the real validation conflict detailed below. The admissible control produced its assertion failure in 1.57 s (17 valid trials). The first two mutation logs did not request trial statistics. The incorrect phase trainer name, missing pooled season column and exact-policy broadcast were harness errors. Corrected versions ran from new files so the originally run scripts stayed intact.

## Findings and regressions

All regression paths below are relative to `.codex-review/proptest2/regressions/`.

1. **Reach-57 empty bins produce an unusable policy.** `test_reachability_bins.py::test_single_profile_policy_can_be_used_after_reload` covers both builders, and `test_pooled_phase_contract.py::test_phase_table_matches_lookup_width` covers phase training. One synthetic top probability of 0.1 gives four identical boundaries, one bin labeled 4, and a table with one physical column. Saving and loading succeeds, then a normal lookup indexes column 4 and raises. Different occupied counts in early and late phases can also abort training with incompatible dot-product dimensions. Hypothesis shrank to early `[0.1]` and late `[0.1, 0.1, 0.2]`; the plain cases use one day versus two days and exercise both directions. This is an unambiguous trainer/consumer contract defect. It can stop offline training or a live lookup if such a base artifact is installed. The shipped width and boundaries pass this structural check.

2. **Late phase uses independently fitted columns with one shared serving boundary vector.** `test_pooled_phase_contract.py::test_late_column_uses_late_cutpoints` trains with early confidences `[0.1, 0.2]` and late `[0.8, 0.9]`. Both have two occupied bins. At streak 55 on the last day, the late trainer's lower bin doubles for confidence 0.8. The persisted early cutpoint of 0.15 routes 0.8 to the upper bin, which skips. This is an unambiguous training/serving disagreement for a phase-aware file. `evaluate_mdp_policy` explicitly evaluates relative tiers, so that offline evaluator can report a different choice from absolute-confidence serving. Git history identifies the shipped base file as pooled v2. The artifact has only one boundary vector, so the original late cutpoints and live impact could not be measured here.

3. **Tail rebuild rejects tied boundaries at save.** `test_tail_artifacts.py::test_tied_tail_quantiles_produce_saveable_policy` uses one confidence of 0.1 with three requested bins. The rebuild retains the correct width with two zero-frequency bins, then the validator rejects `[0.1, 0.1]`. The strict boundary rule is deliberate. The offline builder's inability to emit an artifact for this input is real, with ambiguity over whether ties should be rejected earlier or supported. The current one-bin artifact cannot hit this tie.

4. **Tail optimizer can produce an action that the artifact validator rejects.** `test_tail_artifacts.py::test_tail_optimizer_and_artifact_validator_agree_on_stop_rule` uses two distinct confidences `[0.1, 0.6]` and rates of either a certain miss or a certain double. At streak and best 1, with two days and saver spent, the poor bin should skip to preserve the better expected best. The optimizer does so. The validator demands play whenever improving is arithmetically possible and rejects the table. This is a real solver/validator contract conflict in offline rebuild. The intended precedence of optimal expected best and the mandatory play rule is ambiguous. The regression establishes the rejected rebuild, with no claim about live behavior of today's artifact.

5. **The 30-entry date marker can expire before a retry.** `test_streak_disagreements.py::test_evicted_date_is_still_idempotent` applies 31 dated hits and retries the first. The streak advances to 32. It violates the unqualified at-most-once contract in `update_streak` and CLAUDE.md L04 for a direct caller. The bound is explicit in `save_streak`, so whether this is actionable for normal scoring is ambiguous. The CLI stale-date guard limits exposure, and a manual override or a direct call can still reach it. This affects model streak state.

6. **Out-of-order scoring is arrival-ordered.** `test_streak_disagreements.py::test_dated_update_is_chronological` applies an April 2 hit then an April 1 miss. It returns 0, while chronological replay would end at 1. The marker list is also in insertion order. This is the documented hazard in CLAUDE.md's stale-scoring guard, with chronological backfill a caller obligation. The regression records a real scoring/replay disagreement against the chronological expectation, without treating that expectation as a new contract for the low-level writer. A repeat while its marker is retained returns the current streak with byte-for-byte unchanged state. Changing the date to another season starts streak and saver fresh and replaces the previous marker list.

7. **The fixed exact chain forgives repeated misses.** `test_streak_disagreements.py::test_fixed_exact_and_mdp_agree_after_two_misses_at_ten` compares two forced misses at streak 10 with the saver initially enabled. Fixed exact stays at 10 and full MDP resets to 0. `exact.build_transition_matrix` explicitly documents the missing saver-used coordinate. This is a real known simulator disagreement, with no ambiguity about its cause. Use of that scalar approximation as a baseline can alter P(57), while the tested policy-table exact evaluator uses the one-use saver. No live streak is written by either exact routine.

## Shipped arrays

Read with NumPy from the worktree. `data/models/mdp_policy.npz` stores boundaries `[0.7959788446395174, 0.8114907020399261, 0.8252474190317034, 0.8407401966749947]` and shape `(58, 181, 2, 5)`. All four cutpoints are strictly increasing. `data/models/mdp_tail_policy.npz` stores `[]` and shape `(58, 58, 29, 2, 1)`. Neither has tied or non-increasing boundaries.

## Reference rules and comparison domains

- A single hit adds one. A double with both active legs hit adds two. Any active miss makes the day a miss. Sources are `update_streak` and `picks.effective_daily_result`.
- Voids are passes, per CLAUDE.md Incident repairs L01 and L03 and `picks.active_streak_results`. Discard each void leg. A remaining hit adds one, a remaining miss is a miss, and an all-void day leaves state unchanged. A skip leaves state unchanged, per `mdp.transition_outcomes` and `monte_carlo.simulate_season`. Replay represents skips by absent pick files. The update test calls the low-level writer with empty results to check its streak semantics. The regular CLI and scheduler slot-result callers skip that write when no active results remain.
- A miss at streak 10 through 15 inclusive consumes an available saver and holds the streak. Every other miss resets the streak and keeps the previous saver flag. Sources are CLAUDE.md “Streak saver tracked” and `update_streak`. Best is the maximum streak reached so far, which is the quantity reported by Monte Carlo and optimized by the tail solver.
- CLAUDE.md's 2027 change set 2 notes and the season-rollover contract specify a fresh zero and available model saver on a dated write in a different season, with fresh markers. CLAUDE.md L02 and L04 specify that terminal today is included in replay and that applying a date is idempotent. Actual account saver acquisition and the separate account flag are outside the local-model fold.
- MDP, Monte Carlo, fixed exact, exact policy-table and tail model hit/miss and skip without a void leg. Those comparisons use the explicitly generated no-void domain. The MDP transition compares each prefix's active and best streak and saver, with best computed from observed streaks; Monte Carlo exposes best, count of plays and saver used. The local updater and replay expose streak and saver, and their observed prefix streaks give best. Fixed exact tests one saver-enabled transition and full sequences with the saver disabled. Its documented repeated-saver discrepancy is preserved as a red case. Exact policy-table compares first passage to 57 against enumerated forced action outcomes. Tail compares optimal expected best against the fold on stochastic branches with a finite horizon and generated states below the cap. Live `strategy.py` delegates classification and action lookup, and has no independent streak transition.

## Reproduction and limits

The exact final selection is in `.codex-review/proptest2/final.log.watch`. The five deselected historical checks are the original exact-policy test, both earlier generated phase-width checks, and the two phase regressions in `test_reachability_bins.py` that lacked a season column. Their replacements are `test_exact_policy_retry.py`, `test_pooled_phases.py` and `regressions/test_pooled_phase_contract.py`. The discovery run is expected to return pytest exit 1.

Coverage did not include external grading feeds, operator backfill workflows, concurrent scoring, historical file spellings, broad multi-seed runs, or Monte Carlo calibration on observed outcomes. The full live selection path's state acquisition and action clamps were read but the generated bin-column tests exercised its two lookup adapters directly. Impact on real-season decisions and probability estimates was not measured. A handful of corrected follow-up files were needed to preserve the original scripts after execution. The two main harnesses and the reference fold remain the canonical starting points.

**Was it worth it?** For quality bins, yes. Small generated profiles exposed malformed reloadable reach-57 policies, phase-specific failures even with equal widths, and two independent tail rebuild limits. The shipped boundary check identified which structural failures are absent in today's files. For streaks, yes, with a narrower payoff. The red mutations and more than 500 passing comparisons increased confidence in the shared one-use state machine. The investigation gave a concrete retry-window counterexample and independently reproduced the known chronological and fixed-matrix disagreements.
