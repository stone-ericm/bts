# MDP Double-Down Guardrail Result (2026-05-30)

**Status:** result artifact only. No production behavior change, no policy
artifact write, no deploy claim, and no guardrail implementation.

## Verdict

Do **not** build the proposed double-down probability floor guardrail.

The regenerated primary surface clears the schema gate, but the exact
row-stream sweep rejects the only floor with enough trigger overlap:

| Floor | Label | Trigger Days | Seasons >=5 Triggers | Changed Dates | Mean E[max] Gap |
|---:|---|---:|---:|---:|---:|
| `0.40` | `UNDERPOWERED_TRIGGER_OVERLAP` | `0 / 902` | `0` | `0` | `0.000000` |
| `0.50` | `UNDERPOWERED_TRIGGER_OVERLAP` | `0 / 902` | `0` | `0` | `0.000000` |
| `0.55` | `UNDERPOWERED_TRIGGER_OVERLAP` | `2 / 902` | `0` | `2` | `-0.002375` |
| `0.60` | `REJECT` | `71 / 902` | `5` | `71` | `-0.066200` |

For floor `0.60`, every evaluated season is negative:

| Season | Current E[max] | Guardrail E[max] | Gap | Changed Dates |
|---:|---:|---:|---:|---:|
| `2021` | `23.795018` | `23.771624` | `-0.023394` | `7` |
| `2022` | `23.718322` | `23.649837` | `-0.068485` | `17` |
| `2023` | `23.649616` | `23.590073` | `-0.059543` | `15` |
| `2024` | `23.335818` | `23.242614` | `-0.093204` | `17` |
| `2025` | `23.531160` | `23.444785` | `-0.086375` | `15` |

Support-rung gaps for floor `0.60` are also all negative:

| Metric | Aggregate Gap |
|---|---:|
| `P(reach >= 10)` | `-0.000069` |
| `P(reach >= 20)` | `-0.004745` |
| `P(reach >= 30)` | `-0.001810` |
| `P(reach >= 40)` | `-0.000318` |
| `P(reach >= 57)` | `-0.000071` |

Selection is therefore `none`; `auto_enable_authorized=false`.

## Artifacts

Primary regenerated surface:

```text
data/validation/mdp_dd_guardrail_profiles_2026-05-30/
```

Key files:

```text
data/validation/mdp_dd_guardrail_profiles_2026-05-30/manifest.json
data/validation/mdp_dd_guardrail_profiles_2026-05-30/evaluation.json
data/validation/mdp_dd_guardrail_profiles_2026-05-30/production_p_both_summary.json
```

The manifest freezes:

- exact generation command;
- git SHA `c68c7c044a84f45938cc9f4dce483cd79ec7d416`;
- all 10 input PA parquet SHA-256 values;
- per-season output SHA-256 values;
- required schema and null counts;
- row counts by season and date;
- doubleheader duplicate `(date, batter_id)` counts;
- `BTS_LGBM_RANDOM_STATE=42(default)`; and
- `BTS_LGBM_DETERMINISTIC=0(default)`.

The generated surface has `game_pk` at generation time and clears the primary
schema gate:

| Season | Rows | Dates | Missing Required Columns | Required Nulls | Duplicate `(date,batter_id)` Rows |
|---:|---:|---:|---:|---:|---:|
| `2021` | `1820` | `182` | `0` | `0` | `0` |
| `2022` | `1790` | `179` | `0` | `0` | `4` |
| `2023` | `1820` | `182` | `0` | `0` | `8` |
| `2024` | `1850` | `185` | `0` | `0` | `14` |
| `2025` | `1840` | `184` | `0` | `0` | `6` |

The duplicate `(date,batter_id)` rows are doubleheader appearances with
distinct `game_pk`; there are no duplicate `(date,batter_id,game_pk)` rows.
The production strategy's current double-down eligibility rule is
`game_pk`-only, so the evaluator matches production behavior.

## Method

The evaluator is exact dynamic programming over the historical daily row
stream, not Monte Carlo.

Implementation safeguards:

- `bts.simulate.mdp.solve_mdp` now uses the shared
  `transition_outcomes` helper.
- `scripts/evaluate_mdp_dd_guardrail.py` imports the same helper, so solver and
  evaluator share the transition kernel by construction.
- Saver behavior is unchanged: a miss at streak `10..15` holds the streak and
  consumes the saver.
- First-passage evaluation absorbs into `{streak >= k}` and counts a double
  jump crossing the target threshold, such as `8 -> 10` for `k=9`.
- `E[max streak]` is computed by layer cake:
  `sum(P(reach >= k) for k in 1..57)`.

Focused tests:

```text
PYTHONPATH=src /Users/stone/projects/bts/.venv/bin/python -m pytest \
  tests/simulate/test_mdp.py \
  tests/scripts/test_evaluate_mdp_dd_guardrail.py \
  tests/scripts/test_generate_mdp_dd_guardrail_profiles.py \
  -q
```

Result:

```text
26 passed
```

Claude independently reviewed the manifest, recomputed trigger overlap, and
reviewed the evaluator/test logic before this memo was written.

## Regime Mismatch

The regenerated backtest surface still does not reproduce the current
production collapsed probability regime:

| Surface | n | Mean | Min | q10 | Median | q90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Backtest primary surface | `902` | `0.651294` | `0.523787` | `0.605055` | `0.651646` | `0.697924` | `0.789162` |
| Production picks, 2026-04-02..2026-05-29 | `58` | `0.544749` | `0.474423` | `0.498582` | `0.547558` | `0.581607` | `0.622455` |

This matters for interpretation:

- The backtest can meaningfully evaluate floor `0.60`.
- Floors `0.40`, `0.50`, and `0.55` are not exercised enough historically to
  prove benefit or harm under the frozen trigger-overlap rule.
- The backtest shows the override is harmful where measurable, roughly around
  the high end of current production pair probabilities.
- It does **not** prove the guardrail is harmful on production-typical
  `p_both ~= 0.50..0.55` days, because the historical backtest barely enters
  that regime.

The practical conclusion is still negative: the available evidence does not
support implementing the floor guardrail. The weak-DD symptom should remain
with Gate-B probability-scale / policy-bin reconciliation, not be patched with
a direct `p_both` floor.

## Recommendation

Close the DD guardrail branch as a negative result:

1. Do not add a runtime DD floor guardrail.
2. Do not change `data/models/mdp_policy.npz`.
3. Do not change pick ranking, lock timing, posting, or deploy behavior.
4. Preserve the evaluator and artifacts as evidence for why the direct floor
   patch was rejected.
5. Continue addressing weak-DD behavior through Gate-B reconciliation and
   policy calibration work.

## Caveats

- The generated surface uses shipped defaults:
  `BTS_LGBM_RANDOM_STATE=42(default)` and
  `BTS_LGBM_DETERMINISTIC=0(default)`.
- The artifact is SHA-pinned, but the generation path is not guaranteed
  bit-reproducible on rerun.
- Seed `42` is a known outlier default in BTS historical experiments.
- Because both arms are evaluated on the same regenerated surface, the
  guardrail-vs-current gap is less exposed to seed variance than absolute
  performance, but trigger-overlap counts can still move on a different
  regenerated surface.
