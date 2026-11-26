# MDP Double-Down Guardrail Surface Inventory (2026-05-30)

**Status:** read-only inventory. No production behavior change, no evaluator
result, no policy artifact write, and no deploy claim.

## Question

Before implementing the pre-registered double-down guardrail evaluator, confirm
whether the existing historical ranked profile surface can support the primary
gate:

- stable `game_pk` for enforcing the different-game double-down rule;
- enough rows per day to select the first eligible different-game candidate;
- enough low-pair-probability trigger overlap to test the production symptom;
- no ambiguous post-hoc game attribution that would make the primary surface
  look more precise than it is.

## Tracked Surface

The tracked files are:

```text
data/simulation/backtest_2021.parquet
data/simulation/backtest_2022.parquet
data/simulation/backtest_2023.parquet
data/simulation/backtest_2024.parquet
data/simulation/backtest_2025.parquet
```

They contain `9120` rows across `912` profile days. Each file has columns:

```text
date, rank, batter_id, p_game_hit, actual_hit, n_pas
```

They do **not** contain `game_pk` or `season`. `season` can be inferred from the
filename, but `game_pk` cannot be recovered from the file itself.

The current code path is ahead of those tracked files:
`src/bts/simulate/backtest_blend.py` now defines `PROFILE_COLUMNS` as:

```text
date, rank, batter_id, game_pk, p_game_hit, actual_hit, n_pas
```

That means the missing `game_pk` is a stale-surface problem, not a current code
schema limitation.

## Post-Hoc Enrichment Check

For inventory only, I joined the tracked profiles against PA data from the main
BTS checkout:

```text
/Users/stone/projects/bts/data/processed/pa_2017.parquet
...
/Users/stone/projects/bts/data/processed/pa_2026.parquet
```

The join key was `(date, batter_id) -> game_pk`, matching the older
`scripts/phase7_same_game_double.py` diagnostic approach.

Coverage is superficially complete:

| Metric | Value |
|---|---:|
| profile rows | `9120` |
| unique profile days | `912` |
| rows with joined `game_pk` | `9120 / 9120` |

But the join is not primary-gate quality. Same-date batter mappings are
ambiguous on doubleheader days:

| Ambiguity Surface | Count |
|---|---:|
| ambiguous represented rows | `185` |
| ambiguous represented days | `73` |
| ambiguous rank-1 rows | `22` |
| ambiguous rank-2 rows | `18` |
| ambiguous rank <= 3 rows | `56` |
| ambiguous rank <= 10 rows | `185` |

Because the guardrail being evaluated is explicitly about different-game
double-down eligibility, those rank-1/rank-2 ambiguities are not harmless.
The post-hoc join is acceptable as a diagnostic inventory, but it should not be
treated as the primary deployment gate.

## Trigger-Overlap Inventory

Using the post-hoc enriched surface only to estimate trigger support:

| Surface | Eligible Days | p_both Mean | p_both Min | p_both Max |
|---|---:|---:|---:|---:|
| rank-2 proxy | `912` | `0.657431` | `0.526335` | `0.827460` |
| first different-game candidate | `900` | `0.650883` | `0.526335` | `0.763588` |

The first different-game candidate differs from rank-2 on `350 / 912` days;
the selected different-game double-down rank averages `2.84` and can be as
late as rank `10`.

Trigger counts on the different-game diagnostic surface:

| Floor | Trigger Days | Fraction | Seasons With >=5 Triggers | By Season |
|---:|---:|---:|---:|---|
| `0.40` | `0 / 900` | `0.000` | `0` | `2021:0, 2022:0, 2023:0, 2024:0, 2025:0` |
| `0.50` | `0 / 900` | `0.000` | `0` | `2021:0, 2022:0, 2023:0, 2024:0, 2025:0` |
| `0.55` | `3 / 900` | `0.003` | `0` | `2021:0, 2022:3, 2023:0, 2024:0, 2025:0` |
| `0.60` | `70 / 900` | `0.078` | `4` | `2021:4, 2022:20, 2023:13, 2024:19, 2025:14` |

Under the pre-registration gate, floor `0.55` is clearly
`UNDERPOWERED_TRIGGER_OVERLAP` on this surface. Floor `0.60` has enough raw
trigger days and season spread to be measurable, but it still does not resemble
the production collapsed regime well: production through 2026-05-28 had
`p_both` mean around `0.544`, while this diagnostic surface has mean around
`0.651`.

## Conclusion

The primary gate is **not cleared** by the tracked `data/simulation` files.

Do not implement or run a deployment-facing DD guardrail evaluator on the
current tracked profiles. If Eric chooses to pursue this secondary DD item,
the correct next input is a regenerated historical profile surface from the
current code path, preserving `game_pk` at generation time rather than
backfilling it through an ambiguous `(date, batter_id)` join.

That regeneration is necessary but not sufficient. It can clear the schema
gate, but it will not by itself fix the regime mismatch: this diagnostic
backtest surface centers around `p_both ~= 0.651`, while production through
2026-05-28 centers around `p_both ~= 0.544`. Even after a clean rebuild, expect
floors `<= 0.55` to remain underpowered and floor `0.60` to be at best
measurable but still regime-mismatched. The realistic best case for this
historical route is likely a no-harm screen plus an explicit human decision,
not a backtest proof that the guardrail improves production.

The regeneration should freeze:

- exact command;
- git SHA;
- data directory and PA parquet checksums;
- output directory;
- profile schema;
- row counts by season and date;
- `game_pk` null counts;
- duplicate `(date, batter_id)` / doubleheader ambiguity counts; and
- SHA-256 for each generated profile parquet.

Only after that surface exists, and only after Eric explicitly chooses to keep
pursuing the secondary DD guardrail path, should the exact row-stream evaluator
be implemented or run.

## Operational Note

The existing pre-registration remains valid. This inventory sharpens its input
gate: the DD guardrail question is not blocked by concept, but it is blocked by
the currently tracked historical surface and remains secondary to the
early-lock concern that is already closed.
