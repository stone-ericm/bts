# Realized-picks top-of-slate refresh

**Date**: 2026-05-10T00:43Z snapshot
**Branch**: `codex/realized-top-slate-calibration`
**Predecessors**: `2026-05-04-realized-picks-calibration.md`, `2026-05-05-realized-picks-attribution-p1.md`, `2026-05-05-realized-picks-fdr.md`
**Artifacts**:
- `data/validation/realized_picks_canonical_2026-05-10_p1.parquet`
- `data/validation/realized_picks_fdr_2026-05-10.json`

The `_p1` suffix preserves the existing P1 attribution schema lineage from the May 5 realized-picks work; this refresh does not introduce a new `top_of_slate_calibration_v1` artifact schema.

**Scope**: refreshed realized-picks monitoring only. No production deploy claim.

## Headline

The strict current-model (`post_bpm`) sample grew from 7 resolved rows in the May 5 P1 memo to **15 resolved rows**. It is still underpowered, but it now answers the most important watch question from May 5:

> The historical `post_pooled_mdp_pre_bpm` DD x not_park_driven x Q4 overconfidence warning does **not** currently reproduce in the strict `post_bpm` rows.

Current strict-model DD x not_park_driven x Q4 is:

| Regime | Slot | Env | Skill | n | hits | rate | mean_p | gap |
|---|---|---|---|---:|---:|---:|---:|---:|
| `post_bpm` | double_down | not_park_driven | Q4 | 5 | 4 | 0.800 | 0.726 | -0.074 |
| `post_pooled_mdp_pre_bpm` | double_down | not_park_driven | Q4 | 8 | 3 | 0.375 | 0.731 | +0.356 |

The May 5 watch item remains a historical pre-BPM warning, not a current-model finding. With `post_bpm` n=5, the correct action is still tracking, not a strategy edit.

## Refreshed Counts

Canonical artifact:

| Rows | Resolved | Pending |
|---:|---:|---:|
| 80 | 76 | 4 |

Resolved regime summary:

| Regime | n | hits | rate | mean_p | gap | Brier | BSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `post_bpm` | 15 | 10 | 0.667 | 0.737 | +0.070 | 0.2283 | -0.027 |
| `post_pooled_mdp_pre_bpm` | 30 | 19 | 0.633 | 0.739 | +0.105 | 0.2479 | -0.068 |
| `pre_pooled_mdp` | 31 | 23 | 0.742 | 0.735 | -0.007 | 0.1920 | -0.003 |

`post_bpm` fixed-bin reliability:

| Bin | n | mean_p | mean_y | gap | Wilson 95% |
|---|---:|---:|---:|---:|---|
| [0.70, 0.75) | 11 | 0.727 | 0.636 | +0.090 | [0.354, 0.848] |
| [0.75, 0.80) | 4 | 0.764 | 0.750 | +0.014 | [0.301, 0.954] |

This remains too wide for a calibration verdict.

## Slot And Env Read

Strict `post_bpm` by slot:

| Slot | n | hits | rate | mean_p | gap |
|---|---:|---:|---:|---:|---:|
| primary | 8 | 6 | 0.750 | 0.745 | -0.005 |
| double_down | 7 | 4 | 0.571 | 0.728 | +0.156 |

Strict `post_bpm` by env:

| Env | n | hits | rate | mean_p | gap | Wilson 95% |
|---|---:|---:|---:|---:|---:|---|
| park_driven | 3 | 2 | 0.667 | 0.747 | +0.081 | [0.208, 0.939] |
| not_park_driven | 12 | 8 | 0.667 | 0.734 | +0.067 | [0.391, 0.862] |

No env separation is supportable. The strict-model DD park-driven cell is still n=0, so the DD park-vs-not-park question remains untestable on production behavior.

## Strategic-Question Status

The original strategic hypothesis was: low-skill, park-driven, top-of-slate picks at predicted 0.65-0.80 may realize higher than predicted. That target cell remains unpopulated:

| Regime | Q1 x park_driven rows |
|---|---:|
| `post_bpm` | 0 |
| `post_pooled_mdp_pre_bpm` | 0 |

So the hypothesis is still **untestable on production picks**, not falsified.

## FDR Refresh

The refreshed Cut C family has:

| Family size | Pending excluded | NA-key rows excluded |
|---:|---:|---:|
| 24 | 4 | 19 |

No cell has an FDR-adjusted signal. All `q_bh = q_by = 1.000`. The smallest p-value remains the historical pre-BPM watch cell:

| Regime | Slot | Env | Q | n | hits | p_two_sided | q_BH | Tail |
|---|---|---|---:|---:|---:|---:|---:|---|
| `post_pooled_mdp_pre_bpm` | double_down | not_park_driven | 4 | 8 | 3 | 0.0734 | 1.000 | overconfidence |

The new strict-model DD x not_park_driven x Q4 cell is in the opposite direction and has `p_two_sided = 1.000`.

## Interpretation

1. Current-model primary picks look roughly calibrated so far: n=8, rate 0.750, mean_p 0.745.
2. Current-model double-down picks show point overconfidence overall: n=7, rate 0.571, mean_p 0.728. This is a watch item, not a verdict.
3. The historical DD x Q4 x not_park warning does not carry into strict current-model rows at this snapshot.
4. The intended low-skill park-driven hypothesis remains untested because production still has no Q1 x park-driven rows.
5. No FDR-adjusted Cut C cell is significant or deploy-relevant.

## Reproducibility

Inputs were copied read-only from production:

```
rsync -a --include='2026-*.json' --exclude='*' \
  bts-hetzner:/home/bts/projects/bts/data/picks/ \
  /tmp/realized_picks_input_2026-05-10/

rsync -a \
  bts-hetzner:/home/bts/projects/bts/data/processed/pa_2026.parquet \
  /tmp/bts_realized_refresh/pa_2026_fresh.parquet
```

Canonical artifact:

```
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/canonicalize_realized_picks.py \
  --picks-dir /tmp/realized_picks_input_2026-05-10 \
  --pa-path /tmp/bts_realized_refresh/pa_2026_fresh.parquet \
  --output data/validation/realized_picks_canonical_2026-05-10_p1.parquet \
  --summary
```

FDR artifact:

```
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_realized_picks_fdr.py \
  --input data/validation/realized_picks_canonical_2026-05-10_p1.parquet \
  --output data/validation/realized_picks_fdr_2026-05-10.json
```

## Next

Re-run after either:

- strict `post_bpm` resolved rows reach roughly n=30, or
- strict `post_bpm` double_down x not_park_driven x Q4 reaches n=10-15, or
- any Q1 x park_driven cell finally appears.

Until then, no model or strategy change is supported by this realized-picks surface.
