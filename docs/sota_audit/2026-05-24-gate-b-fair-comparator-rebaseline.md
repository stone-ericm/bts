# Gate B Fair-Comparator Re-baseline (2026-05-24)

**Status:** measurement-only. No production behavior change, no policy artifact
swap, and no deploy claim.

## Question

The estimated-PA walk-forward result showed a positive direction when a policy
re-solved on estimated-PA bins was compared against the deployed policy
projected through its old actual-PA boundaries.

That comparison changed two things at once:

1. the bin boundaries, from deployed actual-PA-scale boundaries to
   estimated-PA-scale boundaries; and
2. the action table, from the deployed action table to a freshly re-solved
   action table.

This follow-up isolates the second lever. It asks:

If both arms use the same corrected estimated-PA boundaries, does re-solving
the action table beat the deployed action structure?

## Method

The new harness `scripts/gate_b_fair_comparator_rebaseline.py` consumes the
same estimated-PA profile directory used by the walk-forward result.

For each expanding-origin fold:

1. Fit equal-frequency boundaries on prior estimated-PA profile seasons.
2. Classify the held-out season with those same prior-season boundaries.
3. Evaluate Arm A: a fresh MDP action table solved on the prior estimated-PA
   bins.
4. Evaluate Arm B: the deployed `data/models/mdp_policy.npz` action table,
   applied by bin index to the same estimated-PA holdout bins.

The deployed policy's saved boundaries are recorded for provenance but are not
used to classify the holdout rows. This is the key difference from the prior
Gate B walk-forward result.

Arm B tests whether the deployed relative action structure transfers to the
corrected estimated-PA bin scale. It does not claim the deployed action table
was originally optimized on these lower absolute probabilities.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked python scripts/gate_b_fair_comparator_rebaseline.py \
  --profiles-dir /tmp/bts_gate_b_estimated_pa_profiles \
  --output /tmp/gate_b_fair_comparator_rebaseline_2026-05-24.json \
  --date 2026-05-24
```

The output artifact reports `production_deploy_claim=false` and
`writes_policy_artifact=false`.

## Result

The emitted decision was:

`MIXED_RE_SOLVE_ACTION_TABLE_SIGNAL_REQUIRES_REVIEW`

| Holdout | Train seasons | Re-solved candidate P(57) | Deployed action structure P(57) | Gap |
|---:|---|---:|---:|---:|
| 2022 | 2021 | `0.0004620581` | `0.0000675242` | `+0.0003945339` |
| 2023 | 2021-2022 | `0.0000855952` | `0.0001005667` | `-0.0000149716` |
| 2024 | 2021-2023 | `0.0002318280` | `0.0001416532` | `+0.0000901748` |
| 2025 | 2021-2024 | `0.0000027595` | `0.0000068597` | `-0.0000041002` |

Overall:

- mean gap: `+0.0001164092`
- gap standard deviation: `0.0001913331`
- nonnegative folds: `2 / 4`
- negative folds: `2 / 4`

## Interpretation

This decomposition narrows the Gate B finding:

- The boundary-scale fix is robust.
- Re-solving the action table on top of that fix is not robust.

The prior walk-forward result showed that a correctly scaled estimated-PA
policy beat the old deployed policy projection. This run shows that once the
deployed action table is also given the corrected estimated-PA boundaries, the
residual value of re-solving the action table is mixed and sign-flipping.

The re-solved action table wins in 2022 and 2024, but loses in 2023 and 2025.
The positive mean gap is smaller than the fold-to-fold standard deviation, so
the re-solve residual is not distinguishable from noise in this screen.

This is consistent with the prior action-shape inspection: the re-solved
candidate policy is discriminating, but non-monotone and sensitive to thin
per-bin empirical `p_hit` and `p_both` estimates.

## Scale Fix Decomposition

The fair comparator also confirms that most of the original Gate B improvement
came from correcting the bin scale. Arm B uses the deployed action table, but
with estimated-PA boundaries. Compared with the old mis-projected deployed
baseline from the previous walk-forward result, Arm B improves every fold:

| Holdout | Old deployed projection P(57) | Deployed actions on estimated-PA bins P(57) |
|---:|---:|---:|
| 2022 | `0.0000433585` | `0.0000675242` |
| 2023 | `0.0000071290` | `0.0001005667` |
| 2024 | `0.0000628553` | `0.0001416532` |
| 2025 | `0.0000005263` | `0.0000068597` |

That is the cleaner Gate B signal: the deployed boundaries are on the wrong
probability scale for production-style estimated-PA probabilities. Giving the
deployed action table correctly scaled bins recovers most of the value.

## Action Table Comparison

The re-solved and deployed action tables differ materially, but the value
effect is unstable.

| Holdout | same-action fraction |
|---:|---:|
| 2022 | `0.645` |
| 2023 | `0.576` |
| 2024 | `0.646` |
| 2025 | `0.657` |

The tables differ on roughly one third to two fifths of compared states, yet
the P(57) residual is tiny and sign-flipping. That implies the differing
actions mostly land in states that do not change the reachability objective
reliably under this profile replay.

## Decision

Do not swap `data/models/mdp_policy.npz`.

Do not advance a freshly re-solved action table toward deployment from this
result.

The useful finding is scope-narrowing: future Gate B work should focus on the
boundary-scale mismatch and treat action-table re-solving as unproven and
possibly overfit unless stronger evidence appears.

This still does not make a boundary-only policy artifact deployable. The
absolute P(57) values remain floor-level, the replay still uses historical
lineup slots and batter universe, and this is an evidence-only local artifact.

Any production policy change still requires:

1. a production-metric re-baseline on the estimated-PA surface,
2. `scripts/leakage_audit.py`,
3. the nuclear test,
4. a reversible policy artifact, and
5. an explicit deploy gate.

## Verification

Focused tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked pytest tests/scripts/test_gate_b_fair_comparator_rebaseline.py -q
```

Adjacent Gate B script tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_EXCLUDE_NEWER=2026-04-12T04:00:00Z \
  uv run --locked pytest \
    tests/scripts/test_gate_b_fair_comparator_rebaseline.py \
    tests/scripts/test_gate_b_walk_forward_policy_eval.py \
    tests/scripts/test_measure_pa_basis_rebin_gate.py \
    tests/scripts/test_measure_raw_rebin_gate.py \
    -q
```

Both passed locally before this memo was written.
