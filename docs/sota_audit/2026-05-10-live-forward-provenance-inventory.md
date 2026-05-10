# Live-forward provenance inventory

**Date**: 2026-05-10 ET
**Scope**: read-only inventory of live-forward ranked-slate artifacts for
leaderboard top-N coverage and later resolved miscalibration joins.
**Script**: `scripts/live_forward_provenance_inventory.py`
**Status**: Phase 2 provenance surface; no production policy, model, scheduler,
dashboard, or deploy change.

## Question

Phase 1 leaderboard mechanism mining can only use ranked-surface variables
(`consensus_model_rank_bin`, `consensus_model_probability_bin`) when an artifact
is genuinely at-lock or has manifest-proven lock-time provenance. This inventory
checks which live-forward artifacts meet that bar.

## Contract

The inventory is mutation-free. It reads live-forward artifact directories and
writes JSON/parquet reports only. It does not export artifacts, resolve
outcomes, write production picks, edit model state, touch scheduler/dashboard
state, or push `deploy`.

The script reports three distinct readiness flags:

| Flag | Meaning |
|---|---|
| `at_lock_ranked_surface_joinable` | The artifact is a live pre-outcome, research-only ranked surface with readable production/candidate top-N profiles and null outcomes. It can support leaderboard top-N coverage and rank/probability annotations. |
| `resolved_outcome_joinable` | A matching resolved artifact has complete non-null outcomes and can support outcome calibration/miscalibration joins. |
| `official_fresh_target_ready` | The artifact is verifier-passing and includes the required production-pick snapshot, so it can count under the fresh-target parity-guard protocol. |

These flags intentionally do not collapse into one boolean. An artifact can be
useful for exploratory rank coverage while still failing the official fresh
target protocol.

## Remote Smoke

Read-only smoke was run on `bts-hetzner` using the branch script copied to
`/tmp` and outputs written to `/tmp`:

```bash
PYTHONPATH=/tmp/bts_live_forward_inventory:/home/bts/projects/bts/src \
  /home/bts/.local/bin/uv run python \
  /tmp/bts_live_forward_inventory/scripts/live_forward_provenance_inventory.py \
  --artifact-root /home/bts/projects/bts/data/validation/decision_weighted_lgbm_v0_live_forward \
  --resolved-root /home/bts/projects/bts/data/validation/decision_weighted_lgbm_v0_live_forward_resolved \
  --output /tmp/bts_live_forward_provenance_inventory.json \
  --rows-output /tmp/bts_live_forward_provenance_inventory.rows.parquet
```

Smoke summary:

| Metric | Value |
|---|---:|
| Artifact count | 1 |
| Date range | 2026-05-09 to 2026-05-09 |
| At-lock ranked-surface joinable | 1 |
| Resolved outcome joinable | 0 |
| Official fresh-target ready | 0 |
| Missing verifier | 0 |
| Missing production-pick snapshot | 1 |

The lone artifact is:

| Field | Value |
|---|---|
| Date | `2026-05-09` |
| Run kind | `live_forward_preoutcome` |
| Candidate | `decision_weighted_lgbm_v0` |
| Top N | `10` |
| Frozen git commit | `5004b1c8b093da0f8acb11bd728430ebacbf92d3` |
| Verification | `ok=true`, `failure_count=0` |
| Production-pick snapshot | missing |

## Read

The 2026-05-09 live-forward artifact is usable for exploratory leaderboard
top-N coverage and model rank/probability annotation because it is a verified
pre-outcome ranked surface with null outcomes. It is not official
fresh-target-ready because it predates, or was not generated with, the
production-pick snapshot parity guard.

There is no resolved outcome artifact currently joinable under this inventory,
so leaderboard miscalibration against live-forward outcomes remains blocked
until resolved artifacts exist and pass the same provenance check.

## Next Step

For future official fresh-target logging, keep using the runbook command with
`--production-pick-file data/picks/YYYY-MM-DD.json` and require
`verify-candidate-artifacts --require-live-preoutcome --require-production-pick-snapshot`.
Once at least one post-parity artifact and its resolved copy exist, rerun this
inventory and then feed the joinable ranked surface into
`scripts/leaderboard_mechanism_mining.py` as a `--surface NAME=PATH` input.
