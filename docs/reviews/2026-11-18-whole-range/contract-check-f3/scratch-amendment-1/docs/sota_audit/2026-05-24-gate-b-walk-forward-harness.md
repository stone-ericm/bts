# Gate B Walk-Forward Harness Plan (2026-05-24)

**Status:** methodology/tooling. No production behavior change and no policy
artifact swap.

## Scope

This is the real Gate B decision path after the PA-basis screen. The screen
showed an in-sample positive signal, but a policy change still needs a
production-PA-basis walk-forward evaluation.

The harness isolates this question:

Does re-binning and re-solving the MDP on a production-comparable probability
basis beat the deployed policy when both are evaluated on the same held-out
future profile stream?

It does not claim to fully replay production lineup availability. Historical
profiles still use the actual batter universe and actual lineup slot. That is
acceptable for this Gate B PA-basis decision because it holds the batter/slot
surface constant while changing the probability basis. Projected-lineup
availability remains a separate production replay caveat.

## Profile Generation Rail

`bts simulate backtest` now has an opt-in `--game-probability-mode
estimated_pa` mode. The default remains `actual_pa`.

The estimated mode rides the existing `blend_walk_forward` loop:

1. Models are the same per-day/per-fold models already trained on prior data
   only.
2. Candidate rows are starter-matchup rows. A batter-game without a row against
   the batting side's first pitcher is excluded from the estimated-mode
   candidate universe rather than scored off a reliever row.
3. Estimated PA volume uses the production lineup-slot map:
   `{1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}`.
4. Starter and reliever PA split matches production: starter capped at `2.5`
   PAs, reliever gets the remainder.
5. Reliever score uses a baseline-model copy of the starter row with
   `pitcher_hr_30g` set to the training-window league-average reliever hit
   rate and `pitcher_entropy_30g` set to the training-window entropy mean.
6. Game probability uses the production product form:
   `1 - (1 - p_starter) ** starter_pas * (1 - p_reliever) ** reliever_pas`.

Estimated-mode profiles also record the daily count and fraction inputs needed
to audit batter-games dropped because no starter-matchup row was available.

The legacy actual-PA mode is guarded by a regression test that compares the
default call against explicit `game_probability_mode="actual_pa"` and verifies
the historical realized-PA product aggregation.

## Policy Evaluation Rail

`scripts/gate_b_walk_forward_policy_eval.py` consumes estimated-PA profiles and
uses expanding-origin folds:

| Holdout | Candidate train seasons |
| ---: | --- |
| 2022 | 2021 |
| 2023 | 2021-2022 |
| 2024 | 2021-2023 |
| 2025 | 2021-2024 |

The first fold is intentionally thin and must be reported as such.

For each fold:

1. Fit candidate bins and solve the MDP on prior seasons only.
2. Classify holdout rows with the candidate training boundaries.
3. Classify the same holdout rows with the deployed policy boundaries.
4. Forward-evaluate the candidate policy and deployed policy on their respective
   fixed-boundary holdout bins.
5. Report per-season P(57), gap, and bin support.
6. Report starter-matchup drop counts/fractions overall and by season.

## Pre-Registered Screen Rule

The harness can only produce a Gate B screen verdict:

- Positive screen: candidate P(57) is at least the deployed baseline in every
  reported holdout season and aggregate mean gap is positive.
- Mixed screen: aggregate mean is positive but at least one holdout season is
  negative.
- Null screen: aggregate mean is non-positive.

Even a positive screen is not deploy-ready. A policy swap would still require a
separate re-baseline, `scripts/leakage_audit.py`, the nuclear test, a reversible
policy artifact, and an explicit deploy gate.

## Example Commands

Generate estimated-PA profiles off-host:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked bts simulate backtest \
  --data-dir /Users/stone/projects/bts/data/processed \
  --output-dir data/simulation_estimated_pa \
  --seasons 2021,2022,2023,2024,2025 \
  --game-probability-mode estimated_pa
```

Evaluate the walk-forward policy screen:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/gate_b_walk_forward_policy_eval.py \
  --profiles-dir data/simulation_estimated_pa \
  --output data/validation/gate_b_walk_forward_policy_eval_2026-05-24.json \
  --date 2026-05-24
```
