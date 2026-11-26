# Gate B Estimated-PA Walk-Forward Result (2026-05-24)

**Status:** measurement-only. No production behavior change, no policy artifact
swap, and no deploy claim.

## Question

The Gate B walk-forward harness asks whether re-binning and re-solving the MDP
on an estimated-PA probability basis beats the deployed policy when both are
evaluated on the same held-out future profile stream.

This is narrower than full production replay. The generated profiles still use
the historical batter universe and historical lineup slot. The result isolates
the probability-basis and pitcher-exposure question; projected-lineup
availability remains a separate caveat.

## Inputs

Estimated-PA profiles were generated with:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked bts simulate backtest \
  --data-dir /Users/stone/projects/bts/data/processed \
  --output-dir /tmp/bts_gate_b_estimated_pa_profiles \
  --seasons 2021,2022,2023,2024,2025 \
  --game-probability-mode estimated_pa
```

The output profiles were:

| Season | rows | rank-1 days | `p_game_hit_basis` | mean `p_game_hit` |
|---:|---:|---:|---|---:|
| 2021 | 1,820 | 182 | `estimated_pa` | `0.749548` |
| 2022 | 1,790 | 179 | `estimated_pa` | `0.745520` |
| 2023 | 1,820 | 182 | `estimated_pa` | `0.743350` |
| 2024 | 1,850 | 185 | `estimated_pa` | `0.741526` |
| 2025 | 1,840 | 184 | `estimated_pa` | `0.737985` |

The walk-forward evaluation was run with:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/gate_b_walk_forward_policy_eval.py \
  --profiles-dir /tmp/bts_gate_b_estimated_pa_profiles \
  --output /tmp/gate_b_walk_forward_policy_eval_2026-05-24.json \
  --date 2026-05-24
```

The evaluation artifact reports `production_deploy_claim=false` and
`writes_policy_artifact=false`.

## Result

The emitted decision was:

`WALK_FORWARD_SIGNAL_POSITIVE_REQUIRES_REBASELINE`

Per the pre-registered screen rule, the candidate beat the deployed baseline in
all reported holdout seasons and had a positive mean gap.

| Holdout | Train seasons | Candidate P(57) | Deployed baseline P(57) | Gap |
|---:|---|---:|---:|---:|
| 2022 | 2021 | `0.0004620581` | `0.0000433585` | `+0.0004186996` |
| 2023 | 2021-2022 | `0.0000855952` | `0.0000071290` | `+0.0000784662` |
| 2024 | 2021-2023 | `0.0002318280` | `0.0000628553` | `+0.0001689726` |
| 2025 | 2021-2024 | `0.0000027595` | `0.0000005263` | `+0.0000022332` |

Overall:

- mean gap: `+0.0001670929`
- gap standard deviation: `0.0001810551`
- nonnegative folds: `4 / 4`
- negative folds: `0 / 4`

## Interpretation

This is a positive-direction walk-forward screen, not a trustworthy magnitude
estimate and not deploy evidence.

The result is worth recording because it agrees with the in-sample PA-basis
screen: moving the MDP onto an estimated-PA scale changes the policy surface in
the favorable direction. It also corroborates the probability-scale finding
that the deployed MDP boundaries are out of domain for production-style
estimated-PA probabilities.

The result must be downgraded for five reasons:

1. Absolute P(57) is at the floor. Both candidate and deployed-baseline values
   imply the streak almost never completes under this replay.
2. The mean gap is not stable relative to fold noise. The gap standard
   deviation is larger than the mean, and the positive mean is heavily
   influenced by the 2022 fold.
3. The deployed baseline is structurally disadvantaged on this scale. Its saved
   boundaries are `[0.7959788, 0.8114907, 0.8252474, 0.8407402]`, above most of
   the estimated-PA rank-1 probability distribution, so the deployed holdout
   bins collapse toward the bottom bin. Because the comparator is
   bottom-collapsed by out-of-domain projection, the `4 / 4` fold unanimity
   overstates robustness; the pre-registered every-season rule assumed a fair
   baseline, while this comparison partly rewards any correctly scaled
   discriminating candidate.
4. Candidate P(57) swings by roughly 170x across holdout seasons, so the sign
   is more informative than the magnitude.
5. The candidate policy is discriminating, but non-monotone and noise-reactive.
   Some higher estimated-probability bins are mostly skipped when their
   empirical `p_hit` or `p_both` is weak, so the candidate's own magnitude
   rests on thin per-bin reward estimates rather than a clean probability
   ordering.

The deployed holdout bin counts show the out-of-domain projection directly:

| Holdout | Deployed holdout bin counts |
|---:|---|
| 2022 | `[126, 37, 10, 5, 1]` |
| 2023 | `[135, 27, 13, 6, 1]` |
| 2024 | `[137, 29, 14, 3, 2]` |
| 2025 | `[154, 18, 7, 2, 3]` |

By comparison, candidate holdout bins are better supported on the estimated-PA
scale. The minimum candidate holdout bin counts are `16`, `21`, `31`, and `20`
for the 2022-2025 folds.

## Candidate Policy Shape

The candidate policies are not collapsed. Recomputed policy tables use all
three actions, although the action surface is noisy and non-monotone.

| Holdout | skip | single | double |
|---:|---:|---:|---:|
| 2022 | `56.8%` | `26.2%` | `17.0%` |
| 2023 | `45.8%` | `27.3%` | `26.9%` |
| 2024 | `54.7%` | `30.1%` | `15.2%` |
| 2025 | `55.1%` | `28.4%` | `16.6%` |

At the start state (`streak=0`, `days_remaining=180`, saver available), every
fold chooses `double` in every bin. Away from the start state, actions vary
substantially by streak, days remaining, saver state, and bin.

The 2025 fold illustrates the non-monotone shape:

| Candidate bin | skip | single | double |
|---:|---:|---:|---:|
| 0 | `40.7%` | `44.9%` | `14.4%` |
| 1 | `68.9%` | `1.2%` | `29.9%` |
| 2 | `81.7%` | `4.7%` | `13.6%` |
| 3 | `9.2%` | `73.1%` | `17.7%` |
| 4 | `74.7%` | `18.0%` | `7.3%` |

This supports the direction-only interpretation: the policy is not simply
collapsed to one action, but it is sensitive to noisy empirical bin hit and
double-hit rates.

## Starter-Matchup Drops

Estimated-PA mode excludes batter-games that have no row against the batting
side's first pitcher. Across the five generated profile seasons:

- profile days: `912`
- dropped starter-matchup batter-games: `27,591`
- total batter-games: `243,533`
- dropped fraction: `11.33%`

By season:

| Season | dropped | total batter-games | dropped fraction |
|---:|---:|---:|---:|
| 2021 | 6,838 | 49,259 | `13.88%` |
| 2022 | 4,716 | 48,135 | `9.80%` |
| 2023 | 5,454 | 48,696 | `11.20%` |
| 2024 | 5,306 | 48,699 | `10.90%` |
| 2025 | 5,277 | 48,744 | `10.83%` |

The drops are not mostly pure opener games. They are concentrated in
normal or near-normal first-pitcher spans, consistent with late entrants,
substitutions, or batters who only faced relievers.

| First-pitcher PA bucket | team-games | dropped batter-games | share of all drops | within-bucket drop rate |
|---|---:|---:|---:|---:|
| `<=3 PA` | 114 | 838 | `3.0%` | `71.9%` |
| `4-9 PA` | 741 | 2,716 | `9.8%` | `35.8%` |
| `10-18 PA` | 3,094 | 3,583 | `13.0%` | `11.4%` |
| `19-27 PA` | 18,752 | 19,613 | `71.1%` | `10.3%` |
| `28+ PA` | 1,335 | 841 | `3.0%` | `6.4%` |

The median first-pitcher PA count among team-games with any dropped
batter-game was `22`. Opener and very-short-start games have high per-game drop
rates, but they explain only a small share of the total dropped batter-games.

This makes the exclusion less concerning than the whole-universe `11.33%`
figure looks in isolation. Production scores the projected starting lineup
against the projected opposing starter; reliever-only substitutes and late
entrants are not normal production pick candidates. The decision-relevant
rank-1/rank-2 profile rows are top retained starter-matchup candidates, so the
effective impact on the evaluated picks is likely far below the whole-universe
drop rate. That inference should still be measured directly before treating the
drop set as fully harmless.

## Next Gate

This result does not justify swapping `data/models/mdp_policy.npz`.

The next fair comparison should re-baseline on the estimated-PA scale instead
of comparing against the deployed policy projected through out-of-domain
boundaries. The fair comparator is a policy re-solved on the same estimated-PA
scale and binning convention, so the next test can isolate whether there is
signal beyond fixing the boundary scale.

Any production policy change still requires:

1. a separate estimated-PA-scale re-baseline,
2. `scripts/leakage_audit.py`,
3. the nuclear test,
4. a reversible policy artifact, and
5. an explicit deploy gate.
