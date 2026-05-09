# Shadow Context Backfill Quality Report (Pre-Apply Evidence)

- Generated: 2026-05-09 03:04 UTC (2026-05-08 23:04 ET)
- Scope: existing context-stack shadow model using `CONTEXT_COLS`
- Status: research diagnostic and backfill-readiness evidence only
- Production deploy claim: `false`
- Shadow promotion claim: `false`
- Evaluation code SHA: `96b6e4f`
- Production apply addendum: completed 2026-05-08 23:32 ET

## Scope Boundary

This report covers the existing context-stack shadow model scored from
`CONTEXT_COLS` shadow picks. It does not evaluate or recommend
`decision_weighted_lgbm_v0`, the separate #16 candidate cycle, or any new
production policy.

The baseline comparator is the deployed production pick record for the same
calendar dates. The shadow rows are research diagnostics until a separately
approved promotion process freezes a candidate and evaluates it live-forward on
fresh targets.

## Inputs

This report summarizes a read-only local dry run against a production `data/picks`
snapshot copied from `bts-hetzner`. The production host was still at `a3bc4d3`
when checked, so PRs #56, #57, and #58 were not deployed.

Evidence sources:

- `data/picks`: local tar snapshot from production.
- `data/raw`: local and production cached raw game JSON where available.
- MLB API fallback: used only for the two missing May 8 game feeds.
- PR #56: corrected shadow result reconciliation to score DD-aware shadow
  candidates rather than the legacy primary-only row.
- PR #57: added the dry-run/apply backfill command and paired quality metrics.
- BTS bus forensic note [591]: identified that historical shadow files were
  being recorded with primary-only legacy result semantics, motivating this
  backfill audit.

No production files were changed. No deploy was performed.

## Backfill Result

The dry run covered all 29 shadow files from `2026-04-10` through
`2026-05-08`.

| Metric | Value |
| --- | ---: |
| Shadow files | 29 |
| Resolved | 29 |
| Unresolved | 0 |
| Errors | 0 |
| Would change | 28 |
| New results | 27 |
| Changed existing results | 1 |
| Unchanged existing results | 1 |
| Skipped | 0 |

The single changed row is `2026-04-11`: the legacy shadow result was `hit`,
but DD-aware recomputation is `miss`. Luis Arraez hit; Steven Kwan missed.

The single unchanged existing row is `2026-04-24`: the legacy shadow result was
`hit`, and DD-aware recomputation is also `hit`. Masataka Yoshida and Trea
Turner both hit.

The `2026-05-08` row resolved through MLB API fallback because raw feeds
`824768` and `822904` were absent from the local and production raw caches.
Yandy Diaz and Nico Hoerner both missed, so the DD-aware shadow result is
`miss`.

## Quality

All 29 paired production-vs-shadow days were evaluable.

| Metric | Production | Shadow |
| --- | ---: | ---: |
| Day hit count | 12 / 29 | 15 / 29 |
| Day hit rate | 41.4% | 51.7% |
| Wilson 95% CI | [25.5%, 59.3%] | [34.4%, 68.6%] |

Paired comparison:

- Shadow minus production hit-rate gap: `+10.3pp`.
- Paired bootstrap 95% CI: `[-6.9pp, +27.6pp]`.
- Sign-test two-sided p-value: `0.453`.
- Paired outcomes: both hit `10`, both miss `12`, production-only hit `2`,
  shadow-only hit `5`.
- Production recorded-result mismatches: `0`.

## Interpretation

The point estimate is positive for the context-stack shadow model, but it must
not be read as a standalone improvement claim. The paired bootstrap confidence
interval is `[-6.9pp, +27.6pp]`, includes zero, the sign test fails to reject
the null (`p=0.453`), and the sample is only 29 days.

Operationally, the dry run is useful: it verifies that the backfill tooling can
resolve every historical shadow row, identify the one legacy DD-aware flip, and
surface no production scoring mismatches. Methodologically, it does not justify
promoting the shadow model or changing production policy.

Any shadow-model promotion still needs a separate pre-registered live-forward
cycle with a frozen candidate, fresh-target evaluation, and explicit production
gates.

## Local Apply Test

A separate `/private/tmp` apply sandbox was created from the production pick
snapshot. Applying the manifest there produced:

- Applied files: 28.
- Skipped files: 1 (`2026-04-24`, reason `no_change_or_not_eligible`).
- Backup files: 28.
- Non-result JSON differences between backups and applied files: 0.

The post-apply idempotency dry run returned:

| Metric | Value |
| --- | ---: |
| Shadow files | 29 |
| Resolved | 29 |
| Would change | 0 |
| Unchanged | 29 |
| New | 0 |
| Changed | 0 |
| Skipped | 0 |
| Errors | 0 |

## Reproduction

The pre-apply dry-run manifest was regenerated with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts shadow-backfill-results \
  --picks-dir /private/tmp/bts-shadow-dryrun-prod-snapshot/data/picks \
  --raw-dir /private/tmp/bts-shadow-dryrun-prod-snapshot/combined_raw \
  --output /private/tmp/bts-shadow-dryrun-prod-snapshot/shadow_backfill_dryrun_rerun_2026-05-09.json \
  --bootstrap 10000
```

The local apply sandbox was verified with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts shadow-backfill-results \
  --picks-dir /private/tmp/bts-shadow-apply-sandbox/data/picks \
  --raw-dir /private/tmp/bts-shadow-dryrun-prod-snapshot/combined_raw \
  --output /private/tmp/bts-shadow-apply-sandbox/shadow_backfill_apply_manifest_2026-05-09.json \
  --apply \
  --backup-dir /private/tmp/bts-shadow-apply-sandbox/backup_shadow_2026-05-09 \
  --bootstrap 10000
```

Post-apply idempotency was verified with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts shadow-backfill-results \
  --picks-dir /private/tmp/bts-shadow-apply-sandbox/data/picks \
  --raw-dir /private/tmp/bts-shadow-dryrun-prod-snapshot/combined_raw \
  --output /private/tmp/bts-shadow-apply-sandbox/shadow_backfill_post_apply_idempotency_2026-05-09.json \
  --bootstrap 10000
```

## Future Evaluation

Promotion should remain out of scope until the shadow model has a
pre-registered live-forward cycle with a frozen candidate and enough paired days
to make the interval informative. A practical gate is to revisit promotion only
after a fresh live-forward sample is large enough that the paired bootstrap
confidence interval can exclude zero under the pre-specified metric. Until then,
new rows should be treated as monitoring data and reported with intervals, not
as evidence that the shadow model beats production.

## Production Gate

Production apply remains blocked until explicit authorization. The apply should
not use the live production checkout, which was still at `a3bc4d3` when checked.
A safe apply sequence should be:

1. Create a separate worktree on `bts-hetzner` at `96b6e4f`.
2. Run a production read-only dry run from that worktree against the live
   `/home/bts/projects/bts/data/picks` directory and compare counts with this
   report.
3. Apply with a fresh backup directory only after explicit authorization.
4. Verify backup count, result-only diffs, and idempotency.
5. Separately deploy the shadow reconciliation/backfill code only when the
   normal production deploy window is authorized.

## Production Apply Addendum

Production apply was authorized and executed on 2026-05-08 at 23:32 ET from a
separate `bts-hetzner` worktree pinned to `96b6e4f`. The live production
checkout was not used for the apply.

Production manifests:

- Read-only dry run:
  `/home/bts/projects/bts/data/validation/shadow_backfill_dryrun_2026-05-09.json`
- Apply:
  `/home/bts/projects/bts/data/validation/shadow_backfill_apply_2026-05-09.json`
- Post-apply idempotency:
  `/home/bts/projects/bts/data/validation/shadow_backfill_post_apply_idempotency_2026-05-09.json`
- Backup directory:
  `/home/bts/projects/bts/data/picks/backup_shadow_2026-05-09`

The production dry run matched the pre-apply evidence counts: 29 shadow files,
29 resolved, 0 unresolved, 0 errors, and 28 would change. The change-class
breakdown was new `27`, changed `1`, unchanged `1`, skipped `0`, and error `0`.
The single changed row remained `2026-04-11`.

The apply wrote 28 `.shadow.json` files and skipped the unchanged
`2026-04-24` row. The backup directory contains 28 `.shadow.json` files.
Backup-vs-applied verification found no non-result JSON differences
(`non_result_violations=[]`), so the apply changed only the `result` field.

Post-apply idempotency returned 29 resolved files, 0 would change, and
29 unchanged rows. The applied `2026-05-08.shadow.json` result is `miss` for
Yandy Diaz plus Nico Hoerner.

The deploy bundle containing PRs #56, #57, #58, and #59 was deployed via the
`deploy` branch at `293ce51` on 2026-05-08 at 23:34 ET. A later narrow health
hotfix deployed `52db4d7`; it did not change shadow selection, scoring, or
backfill behavior.

The 2026-05-08 shadow result was already populated by the apply, so the next
empirical going-forward cron verification is the 2026-05-10 01:00 ET check for
the 2026-05-09 shadow row.
