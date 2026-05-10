# Live-forward surface export

**Date**: 2026-05-10 ET
**Scope**: read-only export of live-forward ranked-slate profiles into the
ranked-surface parquet contract consumed by leaderboard mechanism mining.
**Script**: `scripts/live_forward_surface_export.py`
**Status**: Phase 3 bridge tooling; no production policy, model, scheduler,
dashboard, or deploy change.

## Purpose

The mechanism-mining script can already accept ranked model surfaces via:

```bash
scripts/leaderboard_mechanism_mining.py --surface NAME=PATH
```

This exporter supplies that `PATH` from live-forward artifact manifests. The
output schema is deliberately checked against
`scripts.leaderboard_backfilled_model_audit.load_ranked_surfaces`, which is the
same reader used by mechanism mining. This prevents drift between the supplier
and consumer schemas.

## Operational Check

Claude recommended attempting operational live-forward logging before building
more bridge tooling. A read-only check found that a valid 2026-05-10 official
fresh-target artifact could not be generated after the fact:

- `/home/bts/projects/bts-live-forward` is still at the frozen candidate commit
  `5004b1c8b093da0f8acb11bd728430ebacbf92d3`.
- That frozen worktree's `export-live-candidate-artifacts` CLI does not expose
  `--production-pick-file`, so it cannot create a parity-guarded official
  artifact as-is.
- `/home/bts/projects/bts` at `986fa36cd71b400f941c7df509b2f898c44ba8cf`
  does expose `--production-pick-file`, but `data/picks/2026-05-10.json`
  already has `result = hit`. A new `live_forward_preoutcome` artifact after
  the result is known would not be methodologically valid.

So this PR does not attempt to backfill a fake official pre-outcome artifact.
It instead builds the exporter needed once valid parity-guarded artifacts
exist.

The structural resolution options are: re-cut the candidate freeze at a SHA
that includes the parity-guard CLI and document any prediction drift; establish
an operational protocol where post-freeze CLI tooling loads frozen candidate
model artifacts with matching hashes; or cherry-pick the parity-guard CLI onto
the frozen worktree without changing candidate prediction logic.

## Contract

The exported parquet must include, at minimum:

- `date`
- `rank`
- `batter_id`
- `p_game_hit`
- `actual_hit`

It also preserves `game_pk`, `n_pas`, and provenance metadata such as source
artifact path, source run kind, source git commit, verification status,
snapshot presence, and readiness flags.

The exporter has two eligibility modes:

| Mode | Meaning |
|---|---|
| default | include artifacts that are `at_lock_ranked_surface_joinable`; useful for exploratory top-N coverage. |
| `--require-official-ready` | include only artifacts that pass the official fresh-target readiness gate, including verifier and production-pick snapshot. |

The default mode can consume the existing 2026-05-09 artifact for exploratory
rank coverage. The official-ready mode correctly returns no rows until a
parity-guarded artifact exists.

## Remote Smoke

Read-only smoke was run on `bts-hetzner` using branch scripts copied to `/tmp`
and outputs written to `/tmp`:

```bash
PYTHONPATH=/tmp/bts_live_forward_surface:/home/bts/projects/bts/src \
  /home/bts/.local/bin/uv run python \
  /tmp/bts_live_forward_surface/scripts/live_forward_surface_export.py \
  --artifact-root /home/bts/projects/bts/data/validation/decision_weighted_lgbm_v0_live_forward \
  --resolved-root /home/bts/projects/bts/data/validation/decision_weighted_lgbm_v0_live_forward_resolved \
  --variant production \
  --output /tmp/bts_live_forward_production_surface.parquet \
  --manifest-output /tmp/bts_live_forward_production_surface.json
```

Result:

| Metric | Value |
|---|---:|
| Rows | 10 |
| Dates | 1 |
| Date range | 2026-05-09 to 2026-05-09 |
| Max rank | 10 |
| Actual-hit null rows | 10 |

The exported surface was then passed into mechanism mining as:

```bash
--surface live_forward_production=/tmp/bts_live_forward_production_surface.parquet
```

The mechanism-mining reader accepted the surface with `rows=10`,
`joinable_rows=10`, `max_rank=10`, and no duplicate date/batter collapse.
For the 2026-05-09 fixed cohort, top-10 coverage was `0.5` across the two
date-slot units. Outcome metrics remained null because the supplied surface is
pre-outcome by construction.

## Next Step

The operational gap is daily capture, not the exporter. Before tomorrow's slate
locks, either:

1. update the live-forward logging path so the frozen candidate code plus
   production-pick snapshot guard can run together, or
2. run the parity-guarded export from a consciously chosen checkout and record
   that provenance explicitly.

Do not count any artifact as official fresh-target evidence unless
`verify-candidate-artifacts --require-live-preoutcome --require-production-pick-snapshot`
passes before outcomes are known.
