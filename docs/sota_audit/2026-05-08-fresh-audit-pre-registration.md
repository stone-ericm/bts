# Fresh Audit Pre-Registration: Decision-Aware Learning Candidate Cycle

- **Generated**: 2026-05-08
- **Status**: `candidate_frozen_live_forward_logging_ready`
- **Prior cycle**: `cycle_closed_no_deployable_candidate`
- **Production deploy claim**: `false`
- **Current launch posture**: candidate training is frozen at commit
  `5004b1c8b093da0f8acb11bd728430ebacbf92d3`; official pre-outcome
  live-forward logging also requires the 2026-05-10 production-pick parity
  guard before any artifact counts toward the fresh target. This clears
  research artifact logging only, not cloud compute, production changes, or
  deploy branch changes.

## 1. Cycle Context

PR #50 closed the pooled-policy cycle without a deployable candidate. The
100-seed Hetzner-plus-OCI Phase D temporal split falsified the pooled-policy
candidate on 2025, and the follow-up recency, hybrid, and state-segment screens
did not produce a clean frozen patch. The 2021-2025 surface is now consumed for
that candidate family.

This pre-registration starts a new candidate-generation cycle. It does not
reuse the pooled-policy result as deployment evidence, and it does not reopen
candidate mining on 2021-2025 as if those years were fresh. Historical
2021-2025 data can still be used for development, implementation debugging,
and pre-registered selection rules, but any deployment-grade evaluation must be
post-registration or otherwise demonstrably untouched.

The concrete candidate direction is SOTA tracker item #16:
decision-aware learning. The narrow first implementation should be a
sensitivity-weighted LightGBM training variant, not a full end-to-end SPO
surrogate. The reason is pragmatic: a sample-weighted candidate is implementable
inside the existing LightGBM and experiment-runner structure, while still
testing the core #16 hypothesis that PA-level binary cross-entropy is
misaligned with downstream BTS policy value.

## 2. Candidate Stack Specification

Candidate code name: `decision_weighted_lgbm_v0`.

Candidate variant: #16a, sensitivity-weighted LightGBM. This means training
rows are weighted by a fixed downstream policy-sensitivity map approximating
the absolute change in BTS policy value from changing a PA hit probability near
the decision boundary. The first implementation must not use an SPO surrogate
or decision-bucket calibration-error reweighting unless this memo is amended
before any evaluation launch.

The implemented v0 weighting mode is `top_slate_v0`:

- train an in-window probe LightGBM model with the same feature columns,
  LightGBM parameters, and random state as the final model;
- predict probe PA hit probabilities on the same training window;
- aggregate probe PA probabilities to batter-game hit probabilities;
- rank batter-games within each historical date;
- upweight PA rows attached to the top `10` projected daily batter-games with
  `1 + alpha * exp(-(rank - 1) / rank_scale) * 4 * p_game * (1 - p_game)`;
- use `alpha=2.0`, `rank_scale=3.0`, `clip_min=0.25`, and `clip_max=4.0`;
- normalize final PA weights to mean `1.0`.

The probe/final coupling is intentional in v0: the probe uses the same feature
columns, hyperparameters, random state, and training window as the final model.
Because the probe predicts on its own training data, its absolute probabilities
may be overfit. The v0 guardrail is that these probabilities only define a
within-date rank-and-sensitivity weight, with clipping and mean normalization;
they are not reported as evaluation forecasts. A later v1 can replace this with
cross-fitted out-of-fold probe predictions if the v0 candidate survives local
screening and needs a cleaner estimator.

Baseline reference:

- current production 12-model LightGBM blend;
- current shipped production MDP policy table;
- current production information set at pick time;
- no deploy branch changes.

Candidate hypothesis:

> Rows near downstream policy decision boundaries should matter more than rows
> whose predicted hit probability cannot plausibly change the daily BTS action.
> Training the PA model with pre-registered policy-sensitivity weights may
> improve downstream P(57) without requiring a new solver or a wholesale
> policy-table replacement.

Implementation boundary before any launch:

1. Freeze an exact sample-weight function in code.
2. Freeze the feature columns and blend membership.
3. Freeze the LightGBM parameter differences from production.
4. Freeze the seed list and deterministic training settings.
5. Freeze the command line that emits candidate and production predictions
   before outcomes are known.
6. Freeze the analysis script that compares candidate against production.

The first candidate variant should be intentionally small:

| Name | Description | Allowed before launch? |
| --- | --- | --- |
| `decision_weighted_lgbm_v0` | Production blend with PA sample weights derived from a fixed policy-sensitivity map. | yes, after code freeze |
| `decision_weighted_lgbm_v0_ablation` | Same code path with uniform weights, used only to prove the wrapper does not change production by accident. | yes, diagnostic only |
| Any additional weighting curve | Not part of the primary launch unless explicitly added before candidate freeze. | no |

The sample-weight function must not be fit on the fresh evaluation target. If
it uses historical estimates of state/action sensitivity, those estimates must
be computed only from development seasons named before launch.

This memo is launch-ready only for pre-outcome research artifact logging after
the commit freeze recorded below. It is not a production deployment memo.

## 2a. Implementation Freeze Status

Frozen launch commit SHA for `decision_weighted_lgbm_v0` research logging:

- `5004b1c8b093da0f8acb11bd728430ebacbf92d3` (PR #54, merged 2026-05-08).
- This SHA includes the #16 `top_slate_v0` training hook, historical
  candidate-vs-production artifact schema and comparison command, and
  live-forward pre-outcome export command.
- `production_deploy_claim=false` remains in force. This SHA is the research
  logging freeze, not a deploy branch or production-pick change.

Training hook frozen in the implementation PR:

- `src/bts/simulate/backtest_blend.py` adds `decision_weight_mode=top_slate_v0`
  for LightGBM classifier configs only.
- `src/bts/experiment/models.py` registers
  `decision_weighted_lgbm_v0`, which rewrites the existing production 12-model
  blend configs instead of appending a 13th model.
- Production defaults are unchanged because the hook is inactive unless an
  experiment config supplies `decision_weight_mode`.
- The fast-path experiment runner rejects `decision_weighted_lgbm_v0` because it
  modifies baseline blend configs rather than adding a standalone side model;
  this is covered by `test_model_swap_eligibility_rejects_ineligible`.

Artifact schema and historical comparison path frozen in the next implementation
PR:

- `bts_candidate_ranked_slate_pair_v1` is the paired production/candidate
  ranked-slate schema.
- Each profile parquet carries `artifact_schema_version`, `run_kind`, `variant`,
  `model_name`, `generated_at`, `git_commit`, `date`, `season`, `rank`,
  `batter_id`, `game_pk`, `p_game_hit`, `actual_hit`, and `n_pas`.
  For `live_forward_preoutcome` artifacts, `actual_hit` and `n_pas` are null
  until outcomes are joined later.
- `manifest.json` records schema version, git commit, run kind, season list,
  retraining cadence, top-N, environment seed/determinism variables, profile
  paths, row counts, day counts, `production_deploy_claim=false`, and whether
  the artifact is a fresh-target claim.

Frozen local historical screen command:

```bash
BTS_LGBM_DETERMINISTIC=1 bts experiment export-candidate-artifacts \
  --candidate decision_weighted_lgbm_v0 \
  --seasons 2024,2025 \
  --output-dir data/validation/decision_weighted_lgbm_v0_historical_local \
  --retrain-every 7 \
  --top-n 10
```

The `2024,2025` season list is the frozen default local screen. Operators may
override `--seasons` for wider historical diagnostics, including
`2021,2022,2023,2024,2025`, but those widened runs remain development evidence,
not fresh-target deployment evidence.

Frozen historical comparison command:

```bash
bts experiment compare-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_historical_local \
  --save data/validation/decision_weighted_lgbm_v0_historical_local/comparison.json
```

This comparison emits production and candidate scorecards plus deltas. Its
primary field is the `p_57_mdp` delta when the scorecard can compute it; Monte
Carlo streak metrics are ancillary. It does not yet compute bootstrap CIs,
family-control statistics, or the full `survives_fresh_target` verdict. Those
belong to the fresh-target live-forward slice.

Frozen fresh-target live-forward logging command:

```bash
BTS_LGBM_DETERMINISTIC=1 bts experiment export-live-candidate-artifacts \
  --date YYYY-MM-DD \
  --candidate decision_weighted_lgbm_v0 \
  --output-dir data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD \
  --data-dir data/processed \
  --production-pick-file data/picks/YYYY-MM-DD.json \
  --top-n 10 \
  --no-refresh-data
```

The live command writes only research artifacts under `--output-dir`: paired
production/candidate pre-outcome ranked-slate parquets plus `manifest.json`.
It does not write `data/picks`, `data/models`, production posts, cloud assets,
or the `deploy` branch. The default `--no-refresh-data` assumes routine data
refresh has already completed, so the normal invocation is after the production
daily data refresh has produced the data snapshot. Operators can pass
`--refresh-data` only when they intentionally want this logging command to
refresh current-season data. Use a separate `--output-dir` per date, as shown
above, because the v0 live manifest is one date per directory and would be
overwritten if multiple dates reused the same directory.

The `--production-pick-file` argument is required for official fresh-target
logging after the 2026-05-10 parity-guard amendment. It snapshots the locked
production pick JSON into the manifest so later analysis can distinguish the
candidate ranked slate from the production decision actually submitted to BTS.
The snapshot schema is `production_pick_snapshot_v1`; it includes the full
locked pick JSON inline plus a SHA-256 of the source file so retention policy or
post-resolution edits cannot silently erase the decision context.
If export is re-run for an already logged date, the parity guard captures the
production pick file as of the re-export run; downstream comparison is anchored
on `source_sha256`.
Do not count a live-forward artifact toward the fresh target unless the
post-export verifier passes:

```bash
bts experiment verify-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD \
  --expected-run-kind live_forward_preoutcome \
  --expected-candidate decision_weighted_lgbm_v0 \
  --expected-date YYYY-MM-DD \
  --expected-top-n 10 \
  --require-live-preoutcome \
  --require-production-pick-snapshot
```

Candidate training remains frozen at
`5004b1c8b093da0f8acb11bd728430ebacbf92d3`; official live-forward logging must
also include the 2026-05-10 parity guard above once it is merged. Fresh-target
research logging may begin only for eligible slates whose artifacts are
generated after both the candidate freeze and the parity-guard merge, using the
command above and a distinct date output directory. The first eligible calendar
date remains `2026-05-09`; if no eligible slate is generated after both gates on
that date, use the first later eligible regular-season slate.

## 3. Family-Control Rule

The primary confirmatory family is the set of frozen candidate variants compared
against the production reference on the fresh target. For the first launch the
family should be `m=1`: `decision_weighted_lgbm_v0` versus production.

If the team adds more candidate variants before candidate freeze, all variants
must be listed here before any fresh-target outcomes are inspected. In that
case, report both:

- classical one-sided p-value BH and BY q-values across the frozen candidate
  family, using the cycle's standard `q<=0.05` threshold; and
- the unadjusted primary effect for each candidate, labeled exploratory unless
  it survives the pre-registered family rule.

Post-hoc subgroups, state segments, feature cuts, late-phase cuts, and
provider-only slices are diagnostics only unless they are explicitly promoted
into the family before launch. The prior state-segment screen ended in
`E3_over_survival_revisit_family_control_before_conclusions`; do not use that
result to silently define a smaller fresh family after outcomes arrive.

True e-BH or online FDR is not claimed here. If the cycle wants sequential
anytime-valid stopping, valid e-values or e-processes must be designed before
launch and added as a separate pre-registration amendment.

## 3a. Primary Fresh-Target Comparison Rule

The primary fresh-target estimand is the paired candidate-minus-production
delta in `p_57_mdp` from `bts experiment compare-candidate-artifacts` after
live-forward artifacts are resolved. This keeps the evaluation aligned with
the BTS objective while using the existing scorecard contract.

The comparison unit is the resolved live-forward slate date. The primary
comparison uses the paired production and candidate ranked-slate artifacts,
not the production pick snapshot directly. The production pick snapshot is a
parity/audit guard: it records the actual decision submitted to BTS and must be
available for every official slate, but it does not replace the ranked-slate
scorecard until a future PR adds an explicit candidate policy-selection
surface.

Void-aware resolution amendment, 2026-05-11: resolved v2 artifacts can mark
postponed/cancelled source-date rows with `outcome_status=void_postponement`
or `outcome_status=void_cancellation`. These rows keep null `actual_hit` and
`n_pas`; they are neither hits nor misses and are excluded from comparison
denominators. Pending rows are still a verification failure. See
`docs/sota_audit/2026-05-11-void-aware-resolver-prereg.md`.

A deployment-supporting result requires all of the following before any
production claim:

1. At least `120` eligible resolved slate dates unless this floor is amended
   before outcomes are inspected.
2. Positive point delta in `p_57_mdp`.
3. A one-sided candidate-better-than-production date-paired block-bootstrap
   test whose lower confidence bound is above `0`; use expected block length
   `7`, at least `1000` bootstrap replicates, and random seed `57016` unless
   amended before the first resolved comparison.
4. Family-control survival for the frozen candidate family (`m=1` for
   `decision_weighted_lgbm_v0` unless another variant is added before launch).
5. No material proper-scoring regression on the same resolved slate set. Rank-1
   Brier and log-loss deltas are guardrails: any candidate degradation larger
   than `0.01` absolute Brier or `0.02` log loss blocks a production claim even
   if the `p_57_mdp` point estimate is positive.

Secondary diagnostics are pre-registered but not success criteria: paired P@1
delta, rank-2-on-rank-1-miss behavior, top-10 calibration, candidate-vs-locked
production pick overlap, and slot-level outcomes from the production pick
snapshot. These may explain a result but must not redefine the primary metric
after outcomes are known.

Stopping and falsification rules:

- Do not inspect paired `p_57_mdp` deltas, bootstrap intervals, or guardrail
  deltas before `120` eligible resolved slates. Artifact-existence checks,
  verifier status, and missing-outcome counts may be monitored operationally.
- If the point delta is positive but the lower confidence bound is `<= 0` at
  `120` slates, state is `E4_fresh_target_inconclusive`; continue logging only
  under this same rule or decline to deploy.
- If the point delta is negative and the upper confidence bound is `< 0` at
  `120` slates, reject `decision_weighted_lgbm_v0` for this cycle and use the
  result as falsification evidence for the next #16 candidate iteration.
- If the point delta is non-positive but the interval crosses `0`, make no
  production claim. Continue logging only under this same rule or close as
  inconclusive at season end.
- No interim re-tests between `120` resolved slates and season end are
  pre-registered here. Any interim look at a larger n requires a separate
  amendment before that look happens.

Eligibility and void rules:

- A slate date counts only if both production and candidate resolved artifacts
  have complete outcomes for the ranked rows used by the scorecard.
- If the production pick snapshot has any `slot_results` value of `void`, drop
  that date from the primary paired comparison. A partial-void day has different
  contest semantics and must not be coerced into an active two-slot comparison.
- If the locked production pick snapshot disagrees with the production ranked
  slate rank-1 row, or if candidate rank-1 differs from the locked production
  pick, record the mismatch as a parity diagnostic. The mismatch does not
  redefine the primary ranked-slate scorecard.
- Post-hoc slot, regime, environment, skill, or mismatch-subgroup tests are
  diagnostics only unless they were added to the candidate family before launch;
  any separately tested subgroup family must use the project's BH/BY FDR
  baseline rather than silently borrowing the primary `m=1` rule.

## 4. Fresh Evaluation Target

Primary fresh target:

- 2026 live-forward regular-season slates on or after `2026-05-09` whose
  prediction artifacts are generated after this pre-registration is accepted,
  after the candidate commit is frozen, and after the production-pick parity
  guard is merged.

Pre-memo 2026 data is not primary evidence unless a later audit can prove it
was not used by any candidate-generation or diagnostic script. The default
position is conservative: pre-memo 2026 is development or sanity-check data,
not a lockbox.

Local verification on 2026-05-08 found pre-memo 2026 artifacts and script
references, including `data/processed/pa_2026.parquet`,
`data/simulation/backtest_2026.parquet`, `data/validation/*2026-05-*.json`,
`data/validation/realized_picks_canonical_2026-05-*.parquet`, and 2026 pick
artifacts under `data/picks/`. That is enough to treat pre-memo 2026 as touched
for audit purposes unless a narrower future proof says otherwise.

The clean target is therefore:

```text
fresh_target_start = max(2026-05-09, first eligible slate after candidate freeze)
fresh_target_end = 2026 regular-season end, or the pre-registered analysis date
target_primary_days = all remaining eligible regular-season slates
minimum_primary_days = 120 eligible slates, unless an amendment lowers the floor before outcomes are inspected
```

If the candidate freezes too late to reach the minimum by the 2026 regular
season end, the cycle state is `E4_fresh_target_inconclusive` rather than a
failed or successful deployment claim. The team can then pre-register a 2027
continuation before looking at the accumulated target as confirmation evidence.
The `120`-slate floor is chosen to fit the remaining 2026 regular-season
window while still forcing a season-scale live-forward evaluation; raising the
floor is allowed only by amendment before outcomes are inspected.

Every candidate and production artifact used for the fresh target must be
written before game outcomes are known. The artifact must include:

- generation timestamp;
- git commit SHA;
- feature-data snapshot or source manifest;
- production-equivalence claim for the information set;
- candidate name;
- seed or deterministic setting;
- complete ranked slate, not only the final picked player.

If the team cannot produce pre-outcome ranked slates for both production and
candidate, the target cannot support a deployment-grade policy comparison.

## 5. Provider And Provenance Plan

No provider spend is authorized by this memo alone. The first work is local
implementation and local verification.

Provider posture inherited from the closeout:

| Provider | Current use | Constraint |
| --- | --- | --- |
| Hetzner | Default serious audit provider. | User-reported cap is 5 machines; Phase C obtained 4/5 because one request hit `server_limit`. |
| OCI | Validated for raw profile generation at small multi-AD scale. | Read-only quota math looked larger, but Phase C practical launch degraded; use 4 boxes unless re-verified. |
| Vultr | Burst fallback. | API IP allowlist blocked this session; do not include until fixed and canaried. |

For any cloud run:

1. Use `BTS_LGBM_DETERMINISTIC=1`.
2. Preserve provider, box, region, seed, run kind, queue mode, and commit SHA
   in every artifact.
3. Keep provider tags visible in summaries.
4. Treat cross-provider pooling as valid only if deterministic settings and
   artifact metadata are present.
5. Retrieve before teardown; preserve workers with partial retrieval.

Provider allocation should be selected only after the candidate passes local
implementation tests. The expected path is:

1. Local smoke run.
2. 4-16 seed cloud smoke if local signal is non-negative and code is stable.
3. 48-100 seed provider-diverse run only if the smoke result is positive or
   strategically inconclusive.

## 6. Acceptance And Rejection Thresholds

`survives_fresh_target` requires all of:

- primary P(57) or policy-value gap is positive versus production;
- uncertainty interval excludes zero under the pre-registered resampling rule;
- the result survives the family-control rule;
- proper-scoring diagnostics do not show a material regression on selectable
  or decision-bucket rows;
- production-equivalence and data-lineage checks pass;
- no provider or seed diagnostic explains the result as an artifact.

Production equivalence means the research-ranked slates must match what
production would have produced from the same information at the same decision
time, except for the pre-registered candidate-vs-production model difference.

`inconclusive` means at least one of:

- point estimate is positive but uncertainty overlaps zero;
- proper scoring improves while policy value is flat, or policy value improves
  while proper scoring regresses;
- provider or seed diagnostics are mixed;
- sample size is below the primary minimum.

`falsified` means at least one of:

- primary point estimate is non-positive;
- uncertainty excludes zero in the wrong direction;
- family-control rule fails;
- proper-scoring or data-lineage diagnostics contradict the policy claim.

`production_deploy_ready` is not a possible verdict from this memo alone.
Deployment still requires separate conformal, realized-picks, proper-scoring,
OPE/falsification, data-lineage, operational, and explicit Eric go-ahead gates.

## 7. Execution Budget

Current authorized ceiling from the prior audit planning cycle is `$1000`, but
this memo should consume little or none of it before code freeze.

Recommended budget ladder:

| Stage | Purpose | Spend cap | Stop rule |
| --- | --- | ---: | --- |
| Local implementation | Candidate code, unit tests, dry-run artifacts. | `$0` cloud | Stop if production-equivalence or basic tests fail. |
| Local historical screen | Confirm the candidate is not obviously broken on consumed development data. | `$0` cloud | Stop if P(57), P@1, or Brier are materially worse. |
| Cloud smoke | 4-16 deterministic seeds across Hetzner/OCI. | `$50` | Stop if mean policy gap is non-positive. |
| Serious audit | 48-100 deterministic seeds, provider tagged. | `$200` planning cap before re-quote | Escalate only if smoke survives. |
| Luxury addendum | Provider diversity or richer uncertainty. | remaining cap, explicit go-ahead required | Do not launch silently. |

If provider pricing or quotas are re-checked more than 30 days after the
2026-05-07 budget memo, refresh the estimate before provisioning.

## 8. Verdict Enum And Stop Rules

The cycle should use explicit stop states:

| State | Meaning | Next action |
| --- | --- | --- |
| `E0_candidate_unfrozen` | Candidate implementation or analysis script is not fixed. | Do not launch; finish code and tests. |
| `E1_local_falsified` | Local historical screen is materially worse than production. | Stop the candidate; write closeout. |
| `E2_cloud_smoke_falsified` | Small deterministic cloud smoke is non-positive. | Stop before serious spend. |
| `E3_family_control_blocked` | Multiple variants or subgroups survive in a way the family rule did not anticipate. | Revisit family control before conclusions. |
| `E4_fresh_target_inconclusive` | Fresh target is positive but underpowered or diagnostically mixed. | Continue only if pre-registered sample-size rules allow. |
| `E5_fresh_target_survives` | Fresh target clears thresholds without contradictory diagnostics. | Start separate production-gate review. |

Any analyst-visible fresh outcome outside the pre-registered monitoring path
ends the confirmatory status of the affected target and requires an amendment
before further launch.

## 9. Operational Discipline

No production or deployment branch changes are part of this cycle until a later
explicit production-gate decision. Pushing to `main` is not a deployment; pushing
to `deploy` is a deployment trigger and is out of scope here.

Before candidate launch:

1. Land candidate code and tests in a PR.
2. Record the launch commit SHA in this memo or a dated amendment.
3. Run `git diff --check`.
4. Run focused tests for the candidate code and the audit driver path.
5. Post the frozen candidate and launch command to the BTS bus for Codex/Claude
   review.

During launch:

1. Use exact seed files, not `--seeds N`, when combining providers.
2. Record `boxes.json` before remote execution.
3. Retrieve artifacts before teardown.
4. Preserve boxes on partial retrieval.
5. Do not extend deadlines or add seeds after seeing fresh-target results.

After launch:

1. Produce a final memo separating development evidence from fresh evidence.
2. Keep `production_deploy_claim=false` unless separate production gates clear.
3. If the candidate is falsified, stop the family and do not mine the fresh
   target for a replacement candidate.

## 10. References

- `docs/sota_audit/2026-05-08-pooled-policy-cycle-synthesis.md`
- `docs/sota_audit/2026-05-08-pooled-policy-postmortem-next-candidates.md`
- `docs/sota_audit/2026-05-08-state-segment-policy-candidate-screen.md`
- `docs/sota_audit/2026-05-07-real-split-audit-plan.md`
- `docs/sota_audit/2026-05-07-split-audit-budget-options.md`
- `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md`
