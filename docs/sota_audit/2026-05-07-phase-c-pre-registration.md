# Phase C Pooled-Policy Audit Pre-Registration

- **Generated**: 2026-05-07
- **Status**: `phase_d_outer_eval_falsified`
- **Runner commit**: `ca71f064f5597cfd01cdf70f32f87fbecf1ca189`
- **Depends on**:
  - `docs/sota_audit/2026-05-07-real-split-audit-plan.md`
  - `docs/sota_audit/2026-05-07-oci-profile-canary-result.md`
  - `docs/sota_audit/2026-05-07-oci-scaling-canary-result.md`

## Candidate

The candidate is pooled-policy methodology on deterministic raw backtest
profile surfaces, not pooled prediction.

This audit generates raw `bts simulate backtest` surfaces only. It does not
deploy, select a production policy, or make a production-deploy claim.

## Allocation

Vultr is excluded from this launch because the B2 Vultr canary failed before
spend with an API IP allowlist error:

```text
Vultr HTTP 401: {"error":"Unauthorized IP address: 2600:4041:5976:5800:e82e:1bd3:c1f2:1210","status":401}
```

The pre-registered Phase C allocation is:

| Provider | Boxes | Seed file | Seeds | Output root |
| --- | ---: | --- | ---: | --- |
| Hetzner | 5 | `scripts/audit_seeds_default48.txt` | 48 | `data/hetzner_results/phase_c_pooled_policy_profiles_2026-05-07` |
| OCI | 4 | `scripts/audit_seeds_extension_n100.txt` | 52 | `data/oci_results/phase_c_pooled_policy_profiles_2026-05-07` |

Total: `100` unique disjoint seeds.

This uses existing seed files only. No new seeds are generated for this audit.
The reason is pre-registration discipline: the marginal precision gain from a
new 144-seed extension does not justify introducing a newly generated seed
family immediately before launch.

## Seed Manifest

Seed file hashes:

| File | Count | Unique count | SHA-256 |
| --- | ---: | ---: | --- |
| `scripts/audit_seeds_default48.txt` | 48 | 48 | `2d4a0d8c5e01d6857ed68718efce8bdaa1790348f0ca650c4e4c316a300e4b8d` |
| `scripts/audit_seeds_extension_n100.txt` | 52 | 52 | `60c3b5142a9cbf3e140ded34d8fea3c012f86b2debee64786b28a1bd0e431173` |

Combined seed count: `100`.

Combined unique seed count: `100`.

Duplicate seeds across the two files: none.

The provider split is file-level:

- all `default48` seeds run on Hetzner
- all `extension_n100` seeds run on OCI

Do not use `--seeds N` for Phase C. Separate provider commands must use
`--seeds-file` to avoid accidentally reusing the same first-N default seeds
across providers.

## Temporal Split

Selection seasons:

- `2021`
- `2022`
- `2023`
- `2024`

Outer-evaluation seasons:

- `2025`

Profile seasons:

- `2021`
- `2022`
- `2023`
- `2024`
- `2025`

The split mode is `season_level_selection_outer_eval`. The raw profile surfaces
must keep `production_deploy_claim=false`.

## Launch Commands

Hetzner:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/audit_driver.py \
  --run-kind profiles \
  --provider hetzner \
  --boxes 5 \
  --seeds-file scripts/audit_seeds_default48.txt \
  --selection-seasons 2021,2022,2023,2024 \
  --outer-eval-seasons 2025 \
  --label bts-phasec-profile-hetzner \
  --out data/hetzner_results/phase_c_pooled_policy_profiles_2026-05-07 \
  --poll-interval 600 \
  --deadline-hours 24
```

OCI:

```bash
OCI_SUBNET_OCID=ocid1.subnet.oc1.iad.aaaaaaaanknhmzaeamsc7hkda2crn2a4zr7bggrm4ma6wbiuedpxgef2qzeq \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/audit_driver.py \
  --run-kind profiles \
  --provider oci \
  --boxes 4 \
  --seeds-file scripts/audit_seeds_extension_n100.txt \
  --selection-seasons 2021,2022,2023,2024 \
  --outer-eval-seasons 2025 \
  --label bts-phasec-profile-oci \
  --out data/oci_results/phase_c_pooled_policy_profiles_2026-05-07 \
  --poll-interval 600 \
  --deadline-hours 12
```

Both commands should use the default `--log-pa-predictions` behavior.

## Completion Outcome

Hetzner and OCI both completed the pre-registered profile generation without
production deployment or production-policy changes.

Hetzner:

- requested `5` boxes and obtained `4` because of the provider server limit;
- completed all `48` `default48` seeds;
- retrieved all profile artifacts cleanly;
- deleted all `4` exact audit workers.

OCI:

- ran the clean relaunch label `bts-phasec-profile-oci-r2`;
- completed all `52` `extension_n100` seeds before the `12` hour deadline;
- retrieved all profile artifacts cleanly;
- deleted all `4` exact audit workers;
- exact-ID provider verification after teardown reported all four OCI instances
  as `TERMINATED`.

Combined completed seed count: `100` unique disjoint seeds.

## Deadline Recovery Contingency

Do not interrupt a live Phase C driver only because one worker is behind.
The preferred recovery path is to let `audit_driver.py` reach its deadline.
The driver retrieves complete boxes, tears down boxes whose retrieval is clean,
and logs `PRESERVED` lines for any worker with partial results.

If the OCI run exits with a preserved profile worker, re-attach only to the
preserved box name so the original full-box seed assignment is retained:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/audit_attach.py \
  --provider oci \
  --run-kind profiles \
  --out data/oci_results/phase_c_pooled_policy_profiles_2026-05-07 \
  --seeds-file scripts/audit_seeds_extension_n100.txt \
  --only-box bts-phasec-profile-oci-r2-4 \
  --deadline-hours 24 \
  --poll-interval 600
```

Replace `bts-phasec-profile-oci-r2-4` with the exact worker name from the
driver's `PRESERVED` line if a different box is preserved. Do not run broad
provider cleanup commands.

## Expected Cost And Runtime

Planning estimate:

- Hetzner: `48 * 1.02h * $0.0953/h = $4.66` raw compute
- OCI: `52 * 0.67h * $0.3984/h = $13.88` raw compute
- subtotal: about `$18.54`
- with 50% overhead: about `$28`
- conservative operator budget: `$40`-`45`

Expected wall clock is Hetzner-bound at roughly `10`-`11` hours because the
current Hetzner run obtained `4` boxes. OCI is also capped to `4` boxes for the
clean relaunch after a 12-box attempt exposed a practical OCI launch limit.

## Metrics And Gates

The Phase D analysis is pre-registered to evaluate:

- P@1 and P(57) gap for pooled policy versus the current production policy
- exact/MDP P(57) diagnostics where applicable
- seed-bootstrap uncertainty over the 100 disjoint seeds
- provider-tagged sensitivity, with Hetzner and OCI kept visible in summaries
- proper-scoring diagnostics for any selectable or decision-bucket rows produced
  by the downstream analysis path
- p-value BH/BY audit-verdict baseline if a family of audit claims is tested

Classical p-value BH/BY is a baseline only. This audit does not claim true
e-BH, online FDR, conformal production readiness, or deployment clearance.

## Verdict Ladder

`survives_outer_eval`:

- pooled-policy P(57) gap is positive on the pre-registered 2025 outer
  evaluation surface;
- the 100-seed bootstrap interval excludes zero;
- provider-tagged diagnostics do not show the result is driven by only one
  provider;
- no proper-scoring or falsification diagnostic contradicts the deploy-relevant
  interpretation.

`inconclusive`:

- point estimate is positive but uncertainty overlaps zero;
- provider-tagged diagnostics materially disagree;
- downstream gates are unavailable or underpowered.

`falsified`:

- pooled-policy gap is non-positive on the outer-evaluation surface;
- uncertainty or provider diagnostics contradict the candidate claim;
- downstream falsification/proper-scoring evidence shows the policy value signal
  is not deploy-relevant.

`production_deploy_ready` is not a possible verdict from this artifact alone.
Production deployment still requires separate conformal, realized-picks,
proper-scoring, data-lineage, OPE/falsification, and explicit Eric go-ahead
gates.

## Phase D Outcome

Phase D was run on 2026-05-08 with
`scripts/phase_d_pooled_policy_outer_eval.py`, producing
`data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json`.

The pooled-policy candidate improved the 2021-2024 selection surface but failed
the pre-registered 2025 outer-evaluation surface:

| Surface | Production P(57) | Pooled candidate P(57) | Gap |
| --- | ---: | ---: | ---: |
| Selection, 2021-2024 | `0.067049` | `0.096378` | `+0.029329` |
| Outer eval, 2025 | `0.127678` | `0.064691` | `-0.062987` |

The primary provider-stratified seed bootstrap over 100 seeds produced a 95%
CI of `[-0.065250, -0.060757]`; all `100/100` seed-level gaps were negative.
Provider summaries agreed: Hetzner mean gap `-0.062095`, OCI mean gap
`-0.063810`.

Final Phase D verdict: `falsified`, not deploy-ready.

See `docs/sota_audit/2026-05-08-phase-d-pooled-policy-outer-eval.md`.

## Verification Before Launch

Verified locally before launch:

```bash
git rev-parse HEAD
shasum -a 256 scripts/audit_seeds_default48.txt scripts/audit_seeds_extension_n100.txt
python3 -c "..."
git diff --check
```
