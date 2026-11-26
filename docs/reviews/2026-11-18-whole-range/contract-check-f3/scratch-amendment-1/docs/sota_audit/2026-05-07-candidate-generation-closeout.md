# Candidate-Generation Closeout Before Split Audit

- **Generated**: 2026-05-07
- **Tracker**: `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md`
- **Scope**: SOTA items #16 and #17, plus the compute-resource prerequisite for a real split audit.

## Verdict

Do not manufacture a new candidate stack just to exercise the split-audit flags.

The existing legacy Phase 1 candidate and model-class artifacts are not viable split-audit nominees. They were produced before the explicit selection/outer-evaluation split and do not show a positive enough signal to justify spending the next cycle on deployment-grade evaluation. If Eric does not name a concrete candidate stack and compute budget, the split-audit plan should report `plan_blocked_no_viable_candidate`.

## Legacy Candidate Evidence

The Phase 1 audit-verdict FDR retrospective records the main candidate-generation surfaces as negative or zero on the saved artifacts:

| Candidate | Mean P@1 delta | P(57) MDP delta | P(57) exact delta | Direction |
| --- | ---: | ---: | ---: | --- |
| `catboost` | `-0.010825` | `-0.012309` | `-0.008799` | negative |
| `lambdarank` | `-0.021710` | `-0.044887` | `-0.021741` | negative |
| `xendcg` | `-0.010825` | `-0.032922` | `-0.017444` | negative |
| `decision_calibration` | `-0.021680` | `-0.038885` | `-0.021523` | negative |
| `quantile_gated_skip` | `0.000000` | `0.000000` | `0.000000` | zero |
| `quantile_q10` | `0.000000` | `0.000000` | `-0.024859` | zero |

The same retrospective tested `m=24` Phase 1 verdict artifacts and found `0` positive candidates surviving BH at `q<=0.05` and `0` surviving BY. This does not prove that decision-aware learning or alternative model classes are useless; it says the saved legacy artifacts do not supply the candidate stack for the real split audit.

## #16 Closeout Posture

Decision-aware learning remains a plausible future candidate-generation direction. The right implementation path is still a lightweight sensitivity-weighted or SPO-style training experiment, measured under proper scoring and downstream policy value.

For the current closeout, #16 should be parked as post-audit candidate generation unless Eric names it as the next candidate and allocates compute. Starting a new SPO implementation now would create a fresh exploratory candidate before the project has chosen the real audit target.

## #17 Closeout Posture

The model-class challenge remains a useful rebuttal to the "LightGBM by habit" critique, but the saved CatBoost, LambdaRank, and XE-NDCG artifacts do not justify nomination. A future bakeoff should be run only as a candidate-generation cycle with pre-registered proper-score and downstream-value metrics.

For the current closeout, #17 should be parked as post-audit candidate generation unless Eric chooses a model-class stack as the next candidate and budgets the run.

## Cloud Resource Prerequisite

Before drafting or running the real split audit, inventory the cloud compute resources and artifact surfaces across Hetzner, Vultr, and OCI. The audit plan should not set seed count, parallelism, deadline, or provider allocation until this inventory exists.

Known local surfaces:

| Provider/surface | Local evidence | Current implication |
| --- | --- | --- |
| Hetzner Phase 1 | `data/hetzner_results/audit_phase1` has 16 `profiles_seed*` directories | Early Phase 1 provenance exists. |
| Hetzner 48-seed audit | `data/hetzner_results/audit_full_48seed_v2` has 48 `phase1_seed*` directories | Broad scorecard surface exists, but it is not the raw policy-bin surface. |
| Hetzner pooled raw bins | `data/hetzner_results/pooled_bins_run` plus `pooled_bins_run_trackc` contain 24 `simulation_seed*` directories and 120 backtest parquets | This is the raw pooled-bin surface used by the DR-MDP screens. |
| Vultr extended audit | `data/vultr_results/audit_ext_n100_v4` has 52 `phase1_seed*` directories | Large Vultr Phase 1 surface exists and needs provenance review before reuse. |
| OCI n48 audit | `data/oci_results/audit_n48` has `batch_01/boxes.json`, `batch_01_seeds.txt`, and `seeds_done.txt`, but no retrieved `phase1_seed*` artifacts locally | OCI capacity was exercised or planned, but local artifact retrieval is incomplete for audit scoring. |
| Cross-provider LambdaRank | `data/lambdarank_only/{hetzner,vultr,oci}` has 10 `phase1_seed*` directories | A cross-provider model-class slice exists for LambdaRank only. |
| Post-cutover screens | `data/screen_postcutover/{hetzner,vultr,oci}` and `finalize/` contain provider-specific screen/finalize outputs | These are candidate-screening artifacts, not automatically valid outer-evaluation evidence. |

Operational entrypoint:

- `scripts/audit_driver.py` already supports `--provider hetzner`, `--provider vultr`, and `--provider oci`, with `--boxes`, `--seeds`, `--seeds-file`, `--experiments`, `--label`, `--out`, and two-stage screening flags.
- Default compute shapes are intended to be roughly comparable 16-vCPU / 32-GB AMD boxes: Hetzner `cpx62` in `fsn1`, `nbg1`, `hel1`, or `sin`; Vultr `voc-c-16c-32gb-300s-amd` or `voc-c-16c-32gb-500s-amd` in European fallback regions; OCI `VM.Standard.E5.Flex` at 8 OCPU / 32 GB.
- Credentials are provider-specific: Hetzner and Vultr token entries are read from macOS Keychain; OCI requires `~/.oci/config` or OCI keychain entries plus `oci-subnet-ocid`, and the OCI Python SDK.
- OCI is capacity-gated by service limits for `VM.Standard.E5.Flex` AMD OCPUs; a real audit plan must verify the live limit before assigning OCI work.
- The driver handles graceful degradation when fewer boxes are available than requested, so the split-audit plan must record actual boxes obtained, not only requested parallelism.

## Next Work

1. Treat #16 and #17 as parked candidate-generation directions, not pre-audit blockers.
2. Before the split-audit plan, create a resource/provenance inventory for Hetzner, Vultr, and OCI: available quotas, usable instance shapes/plans, live credentials, expected cost/runtime, local artifact completeness, and which surfaces are selection evidence versus outer-evaluation evidence.
3. If no concrete candidate stack is named after closeout, stop at `plan_blocked_no_viable_candidate` instead of running a ceremonial split audit.

## Verification

Docs-only slice; `git diff --check` passes.
