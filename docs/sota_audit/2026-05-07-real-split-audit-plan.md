# Real Split-Audit Plan

- **Generated**: 2026-05-07
- **Status**: `candidate_selected_pending_profile_launcher_and_canaries`
- **Depends on**: `docs/sota_audit/2026-05-07-candidate-generation-closeout.md`

## Verdict

Do not run the real split audit yet, but the candidate intake blocker is now
resolved.

After Codex/Claude review and Eric's 2026-05-07 budget authorization, the audit
candidate is the pooled-policy methodology on a determinism-certified raw
profile surface. The current verdict is
`candidate_selected_pending_profile_launcher_and_canaries`, not deploy-ready.
The audit becomes runnable only after the raw profile launcher is merged, the
OCI one-seed retrieval canary passes, and the provider/provenance inventory is
recorded for the exact run.

The original remote-screen orchestration gap is now closed for `scripts/audit_driver.py`: the driver passes `--selection-seasons` and `--outer-eval-seasons` through remote Phase 1 screening and records split metadata. The pooled-policy path needs raw `bts simulate backtest` artifacts, not `bts experiment screen` score JSONs, so the next orchestration slice is the `--run-kind profiles` launcher that writes and retrieves `data/simulation_seed*/backtest_*.parquet` with provider/determinism metadata.

## Selected Candidate

Candidate stack: pooled-policy methodology, not pooled prediction.

Pre-registered implementation target:

- generate deterministic per-seed `bts simulate backtest` profiles with `BTS_LGBM_RANDOM_STATE=<seed>` and `BTS_LGBM_DETERMINISTIC=1`
- retain `backtest_*.parquet` and `pa_predictions_*.parquet` under `simulation_seed<seed>/`
- tag each seed by provider, box, region, run kind, queue mode, deterministic-env intent, and profile seasons
- build pooled policy bins from seed-tagged rank-1/rank-2 profiles
- solve/evaluate the pooled policy table against the current production policy table
- keep production unchanged until separate production gates clear

Reason for choosing it: the saved pooled-policy artifacts are the only current positive signal large enough to justify a SOTA audit:

| Artifact | Signal |
| --- | --- |
| `data/validation/pooled_policy_ab_24seed_consolidated.json` | leave-one-out mean P(57) gap `+0.019290`; `24/24` seeds positive |
| `data/validation/pooled_policy_gap_ci_2026-05-06.json` | seed-bootstrap CI `[+0.014468, +0.024308]`; exact sign-test two-sided p `1.192e-07` |
| `data/validation/pooled_policy_ab_trackd_crosspath.json` | cross-path mean P(57) gap `+0.015846`; `8/8` seeds positive |

Caveat: the existing positive raw surfaces did not embed complete determinism/provider metadata, so they nominate the candidate but do not certify deployment. The new audit surface must be generated with metadata embedded before cross-provider pooling is interpreted as seed variation.

## Candidate Intake Checklist

The audit can start only when these inputs are fixed before looking at outer-evaluation outcomes:

| Input | Required decision |
| --- | --- |
| Candidate stack | Pooled-policy methodology on deterministic raw backtest/profile surfaces. No post-hoc additions after outer evaluation starts. |
| Commit | Git SHA containing the candidate and runner code. |
| Selection seasons | Seasons used for Phase 1/2 keep/drop and any threshold or feature decisions; must be earlier than the outer-evaluation seasons. |
| Outer-evaluation seasons | Disjoint later seasons evaluated once after selection. |
| Seed family | Pre-registered seed list and stopping rule. |
| Compute budget | Eric authorized up to `$1000` on 2026-05-07 for audit/canary planning. Target plan is `$700`-`800` after canaries, not a blank check for deploy. |
| Provider allocation | Default tri-provider plan after canaries: 48 Hetzner seeds, 24 OCI seeds, 24 Vultr seeds; fall back to Hetzner+Vultr or Hetzner-only if OCI canary fails. |
| Artifact roots | Output paths for selection artifacts, outer-evaluation artifacts, resource inventory, and final memo. |
| Metrics | Pre-registered metrics: P@1, P(57) MDP/exact, #12 proper-scoring metrics, and any candidate-specific diagnostic. |
| Gate family | Pre-registered test family for BH/BY or future e-value control. |
| Verdict ladder | What counts as `survives_outer_eval`, `inconclusive`, and `falsified`. |

If any row is missing, stop before compute.

## Cloud Resource Inventory

Do the resource/provenance inventory before setting seed count, parallelism, deadline, or provider allocation. The inventory should be saved as a durable artifact, for example `docs/sota_audit/<date>-split-audit-cloud-inventory.md` plus an optional JSON summary.

Compute tier options are scoped in `docs/sota_audit/2026-05-07-split-audit-budget-options.md`.

Current local knowledge:

| Provider | Driver default | Existing local surfaces | Current use in plan |
| --- | --- | --- | --- |
| Hetzner | `cpx62`, Ubuntu 24.04, locations `fsn1`, `nbg1`, `hel1`, `sin` | 4-box 48-seed scorecard surface in `data/hetzner_results/audit_full_48seed_v2`; 16-seed Phase 1 in `data/hetzner_results/audit_phase1`; 24-seed raw pooled-bin surface across `pooled_bins_run` and `pooled_bins_run_trackc` | Best default for reproducible large runs. User-provided current cap is `5` machines; quota increase request possible in about `4` days. |
| Vultr | `voc-c-16c-32gb-300s-amd` then `voc-c-16c-32gb-500s-amd`, European fallback regions | 26-box / 52-`phase1_seed*` extended surface in `data/vultr_results/audit_ext_n100_v4` | Best burst capacity fallback. User-provided current cap is `30` machines / `$2500`; record actual boxes and check deterministic-runtime overhead against the spend ceiling. |
| OCI | `VM.Standard.E5.Flex`, 8 OCPU / 32 GB, Ubuntu 24.04 x86_64 | `data/oci_results/audit_n48` has 4 Ashburn boxes in `batch_01/boxes.json` and 31 queued seeds, but no retrieved `phase1_seed*` scoring artifacts locally | Eric authorized read-only OCI verification on 2026-05-07. OCI CLI checks verified `83` E5 OCPU and `1250` GB E5 memory available with `0` used in each of 3 Ashburn ADs. At the planned shape, quota supports 10 boxes per AD / 30 boxes total. Treat OCI as a serious accelerator only after the subnet OCID is supplied through Keychain/env, a one-seed retrieval canary passes, and the driver's AD-rotation/fallback path is live-verified before using more than about 10 OCI boxes. |

The docs plan scopes the inventory only. The OCI service-limit checks above were read-only and did not launch instances. Any provisioning, canary launch, or provider spend is still a separate authorized operational slice.

Eric's 2026-05-07 capacity notes plus read-only OCI quota checks are partial inventory only: Hetzner `5` machines with an increase request possible in about `4` days, Vultr `30` machines / `$2500`, and OCI E5 quota verified at 30 planned boxes across Ashburn. The full inventory still needs launch canaries, credential/subnet readiness, cost re-quotes, artifact retrieval checks, live verification of larger OCI multi-AD launches, and provenance recording before compute.

The inventory must record:

- provider credentials present and scoped correctly
- live quota or practical max boxes available per provider
- actual instance shape/plan obtained, not only requested shape
- expected cost and wall-clock time at the chosen seed count
- whether provider/model determinism metadata is embedded in produced artifacts
- whether remote seed artifacts record provider, box, region, and deterministic-env intent before any cross-provider pooling
- whether each existing artifact surface is selection evidence, outer-evaluation evidence, raw policy-bin evidence, or only operational provenance
- teardown/retrieve safety status, including any preserved boxes after partial retrieval

## Orchestration Work Before Compute

1. Add the raw profile launch path to `scripts/audit_driver.py`: `--run-kind profiles` should run `bts simulate backtest`, not `bts experiment screen`.
2. Fail closed if split-mode profile seasons omit any selection or outer-evaluation season.
3. Persist the resolved split metadata beside every remote seed output.
4. Persist provider, box, region, run kind, queue mode, deterministic-env intent, and `cross_provider_pooling_validated=false` beside every seed output.
5. Write `boxes.json` before launch and record actual boxes obtained.
6. Preserve boxes on partial retrieve. Do not tear down a box unless its retrieve status is `ok`.
7. Add command-rendering tests proving the remote command uses `bts simulate backtest`, `BTS_LGBM_DETERMINISTIC=1`, `--log-pa-predictions`, and `simulation_seed*` retrieval.

This is separate from `bts experiment screen/select`, which already has the season-level split API. The gap is cloud orchestration.

`scripts/audit_driver.py` currently orchestrates remote Phase 1 screening only. `bts experiment select` remains local; if a future audit needs remote select orchestration, it should use the same split-flag pass-through and metadata pattern rather than falling back to legacy `--test-seasons`.

## Load-Bearing Gates

The real split audit inherits these constraints from the SOTA closeout cycle:

- Split season sets must be disjoint. Mixed legacy `--test-seasons` plus split flags must fail closed.
- Split artifacts currently carry `production_deploy_claim=false`; the split result is methodology evidence, not a deploy approval.
- The audit-verdict FDR retrospective is a p-value BH/BY baseline only. It is not e-BH or online FDR.
- The current conformal-gate v2 artifact has `ship_set=[]`; no conformal lower-bound path is production-cleared yet.
- The #10 pooled-policy uncertainty layer is blocked on a determinism-state precondition. Existing artifacts cannot analytically clear it.
- #16/#17 are parked candidate-generation directions unless Eric names a candidate stack and compute budget.
- Full #13/#14/#15 SOTA variants remain conditional. Use the existing falsification harness unless the candidate creates a deployment-grade policy comparison the v1 harness cannot answer.
- The production-equivalence claim is still required: the picks scored by research must be the picks production would have made with the same information at decision time.

## Audit Protocol Once Unblocked

1. Freeze the candidate and commit SHA.
2. Write a pre-registration memo naming candidate, seasons, seeds, providers, gates, and output paths.
3. Run selection on `selection_seasons` only.
4. Run the final selected stack exactly once on `outer_eval_seasons`.
5. Aggregate across seeds only after each seed respects the temporal split.
6. Apply the pre-registered p-value family correction or future e-value/e-process gate.
7. Run supporting gates that match the candidate's claim: proper scoring, OPE/falsification harness, rare-event MC, dependence caveats, conformal v2, and production data-lineage checks.
8. Produce a final memo with separate sections for selection evidence, outer-evaluation evidence, lockbox status, provider/resource provenance, and deployment posture.

No split-audit artifact should claim production deployability by itself. The current split implementation labels artifacts with `production_deploy_claim=false`; keep that posture unless separate production gates explicitly change it.

Separate production gates means at least:

- non-empty conformal-gate v2 `ship_set` on the candidate stack's predictions
- #12 proper-scoring evidence on selectable or decision-bucket rows that does not regress versus current production
- audit-verdict FDR baseline that survives at `q<=0.05`, with enough season-level evidence to be testable
- OPE/falsification-harness result that does not contradict the deploy claim
- production data-lineage check showing research and production information sets match at decision time
- Eric's explicit deploy go-ahead

## Initial Season Split

Use a conservative first pass unless Eric chooses otherwise:

- Selection: `2023,2024`
- Outer evaluation: `2025`
- Lockbox/deploy claim: none from this artifact

This mirrors the existing split-design memo and keeps the outer span later than the selection span. If the candidate needs a different temporal split, pre-register the reason before running.

## Expected Current Outcome

With pooled-policy selected but canaries still pending, the plan should stop at:

```json
{
  "verdict": "candidate_selected_pending_profile_launcher_and_canaries",
  "candidate": "pooled_policy_methodology",
  "reason": "Candidate intake is complete, but the raw profile launcher, OCI canary retrieval, and provider/provenance inventory must pass before scaled cloud spend."
}
```

## Verification

Docs-only plan. Verified locally:

```bash
git diff --check
```
