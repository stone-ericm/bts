# Void-Aware Live-Forward Resolver Pre-Registration

**Date**: 2026-05-11

**Scope**: resolver-side handling for live-forward ranked-slate artifacts when
an original source-date game is postponed or cancelled and therefore has no
processed PA rows.

## Decision

The live-forward resolver remains strict by default. A missing outcome row is
still treated as pending evidence and fails closed unless the caller opts into
terminal void handling with:

```bash
bts experiment resolve-live-candidate-artifacts --treat-void-games-as-terminal
```

The source of truth for postponed/cancelled detection is
`bts.picks.get_game_statuses_detailed(date)`. No parallel schedule parser or
string heuristic is introduced.

## Outcome Status Contract

New resolved artifacts write schema version
`bts_candidate_ranked_slate_pair_v2` and add a per-row `outcome_status` column:

- `resolved`: PA evidence exists; `actual_hit` and `n_pas` are observed.
- `void_postponement`: the game is terminally void due to postponement;
  `actual_hit` and `n_pas` remain null.
- `void_cancellation`: the game is terminally void due to cancellation;
  `actual_hit` and `n_pas` remain null.
- `pending`: evidence is still missing; official verification must fail.

Void rows are not misses. They are also not evidence of model quality. They
contribute zero rows to model-comparison denominators.

## Manifest Contract

Resolved v2 manifests add:

- `source_schema_version`
- `outcome_status_values`
- `outcome_status_counts`
- `outcome_status_counts_by_variant`
- `outcome_terminal_void_enabled`
- `outcome_terminal_void_total`
- `outcome_missing_total`
- `outcome_status_semantics`
- `outcome_terminal_void_semantics`
- `profile_schema_columns` including `outcome_status`

The verifier accepts legacy v1 manifests for backward compatibility, but new
resolution writes v2. For v2 resolved artifacts, null `actual_hit`/`n_pas` is
accepted only when `outcome_status` is `void_postponement` or
`void_cancellation`. `pending` rows fail official verification.

## Runner Contract

The guarded runner enables `--treat-void-games-as-terminal` and reports:

- `resolved_verified` when all rows have observed outcomes and verification
  passes.
- `resolved_with_voids` when verification passes and at least one terminal
  void row exists.
- `pending_outcomes` when missing rows are not known terminal voids.

Existing resolved manifests are verified idempotently rather than rewritten.
When converting a previously pending postponed slate, use `--overwrite` as a
separate explicit backfill step.

## Downstream Policy

Downstream scorecards and mechanism mining must exclude rows whose
`outcome_status != "resolved"` from denominators. Dates without resolved rank 1
and rank 2 rows are excluded from streak scorecards, because the combined BTS
strategy cannot be evaluated honestly from a partially void top pair. Candidate
and production scorecards must then be evaluated on the common remaining date
set so the comparison stays paired.

The 2026 fresh-target slate floor remains a comparison-denominator constraint.
A date with terminal void rows can provide resolved row evidence, but void rows
do not count as hits, misses, or comparable observations.

## Acceptance Tests

- Strict mode still fails on missing outcome rows.
- Opt-in mode converts a synthetic postponed game to
  `outcome_status=void_postponement` with null `actual_hit` and `n_pas`.
- Verifier accepts known void-null rows and rejects pending-null rows.
- Runner invokes the resolver with `--treat-void-games-as-terminal` and reports
  terminal void counts.
