# BTS State-of-the-Art Audit — Master Tracker

**Date created**: 2026-05-01
**Last updated**: 2026-05-08 (fresh audit pre-registration)
**Status**: Active; 17 audit areas. Falsification-harness path shipped v1 + v2.5 attribution + v2.6 CI piece (PR #8 merged at `1a0eefb` on 2026-05-04); full-SOTA variants for #13/#14/#15 remain conditional/deferred rather than immediate blockers. #12 phase 1/2/3 surfaces have shipped; strict current-model realized-picks evidence is still underpowered. #1 solver-side DR-MDP is measurement-gated and produced no production-change signal on the explicit 2021-2025 canonical surface. #5 has opt-in season-level split support for experiment runner adoption. #10 pooled-policy candidate generation is closed under `cycle_closed_no_deployable_candidate`: Phase D falsified the 24-seed C0 signal under a 100-seed temporal split, and the follow-up recency/hybrid/segment screens did not produce a deployable candidate. #11 conformal-gate v2 exists and currently blocks deploy because `ship_set=[]`, not because binary-y validation is unimplemented. #16 is reactivated as the next concrete candidate-cycle direction via `docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`; the `top_slate_v0` training-weight hook is frozen at `5004b1c8b093da0f8acb11bd728430ebacbf92d3`, while official fresh-target logging also requires the 2026-05-10 production-pick parity guard before any slate counts toward the fresh target. #17 remains parked unless Eric names a specific model-class stack and compute budget. Real split-audit planning is documented at `docs/sota_audit/2026-05-07-real-split-audit-plan.md`; after the pooled-policy closeout it now depends on frozen #16 live-forward evidence, not on mining the consumed 2021-2025 surface again.
**Origin**: `project_bts_state_of_art_audit_2026_05_01.md` — Eric committed to project-wide SOTA audit after observing pattern of Claude defaulting to "existing codebase" or "Eric-friendly" rather than state-of-the-art.

This document is the operating tracker for the audit. It's structured for rolling updates: as each area is brainstormed/scoped/implemented, append status notes here.

## Status update — 2026-05-04 (post-v2.5/v2.6 reconciliation)

**PR #8 merged** at commit `1a0eefb` on 2026-05-04 — completes the v2.5 nested-factorial attribution + v2.6 block-bootstrap CI ablation cycle.

**v2.6 outcome (within the falsification-harness path only)**:
- Profile-level paired hierarchical block-bootstrap (Politis–Romano stationary, expected_block_length=7, n=500) added to `corrected_audit_pipeline` for `corrected_pipeline_p57`.
- Gate-class transition collapsed at the current half-headline=0.04085 threshold: under block-bootstrap, the v1 BROKEN classification was a percentile-CI artifact (ci_upper 0.0375 was 0.7pp below threshold on a CI whose grid resolution is 0.83pp). All 6 ablation cells gate REDUCED.
- v2.5 point-estimate attribution survives with precise framing: B (per-bin rho_pair) and C (per-fold MDP solve) each shift +1.67pp single-mode; combined B+C shift +2.50pp via one extra 2023 fold success. A (fold-local params) has no observable effect at current resolution AND is ~3.5× slower in this runner — keep A as leakage-hygiene methodology for final audits, use pooled for exploratory screens.
- See `docs/sota_audit/2026-05-03-harness-v2.5-attribution.md` for full memo + v2.6 addendum.

**What v2.6 did NOT close**: areas #13 (OPE), #14 (rare-event MC), and #15 (PA/cross-game dependence modeling) shipped **v1 simplifications** inside the Task 13 falsification harness, but their full SOTA variants remain open conditionally: **#13 sequential DR/FQE remains deferred; #14 richer per-step/per-action CE-IS or subset-simulation variants remain deferred; #15 fuller out-of-fold PA/cross-game residual-dependence modeling remains deferred.** These should be implemented when a new policy candidate or split audit produces a deployment-grade question the v1 harness cannot answer. #5 now has opt-in experiment-runner split support; adoption on real candidate artifacts is still pending. #11 has conformal-gate v2 infrastructure and a current `NO_PRODUCTION_DEPLOY` result. #12 is no longer unstarted: phase 1 proper scoring, phase 2 realized-picks calibration, phase 3 realized-picks attribution, and the realized-picks FDR baseline have shipped, but the current-model sample remains too small for a deployable calibration verdict.

**Current production recommendation**: keep policy as-is. The harness work has not produced grounds to redeploy.

**Strategic question reframe**: avoid framing the next project as "distribution-shift remediation." The strategic-gaps memo flagged that the original production-overconfidence diagnosis was iteration-contaminated. The current open production question is **whether current-model top-of-slate under-confidence is real and exploitable** — and #12 (probabilistic forecast evaluation suite) is the cheapest foundation for testing that.

**Next active SOTA item**: **bin-side / multi-seed pooling measurement**, using #12's proper-scoring and realized-picks artifacts as the honesty layer. The immediate solver-side DR-MDP screen did not justify production solver work on the explicit 2021-2025 canonical surface.

## Status update — 2026-05-06 (MDP objective audit + DR-MDP gap screen)

**PR #26 merged** at commit `c02142b` on 2026-05-06 — documents that `solve_mdp` already optimizes reachability P(57), not expected streak length. CVaR-over-streak is ruled out as the wrong objective topology for the BTS win condition. The only solver-side candidate left by that memo is DR-MDP over bin-parameter ambiguity, and it must be measured before any production solver work.

**PR #27 merged** at commit `d96388a` on 2026-05-06 — adds `scripts/dr_mdp_gap_measure.py`, focused tests, and the DR-MDP measurement plan. The first durable run is recorded at `data/validation/dr_mdp_gap_2021_2025.json` and summarized in `docs/sota_audit/2026-05-06-dr-mdp-gap-result.md`.

**DR-MDP screen outcome on explicit 2021-2025 canonical profiles**:
- `point_p57=0.039960`
- `max_delta_p57=0.035678`
- inherited v2.6 cell-111 CI half-width: `0.083333`
- both finite rectangular constructions stayed within that CI half-width
- the intended 24-seed pivoted profile files were not present locally, so this closes only the explicit single-seed 2021-2025 surface

**Current production recommendation**: keep policy as-is. No deploy and no MDP solver change are justified by this screen.

**Next active SOTA item**: bin-side / multi-seed pooling. #12 is no longer an implementation blank; it is now an evidence layer to use when measuring whether bin pooling or multi-seed surfaces improve the top-of-slate decision problem.

## Status update — 2026-05-06 (pooled-seed inventory)

**Pooled-seed inventory added** as `docs/sota_audit/2026-05-06-pooled-seed-inventory.md` plus `data/validation/pooled_seed_inventory_2026-05-06.json`, generated by `scripts/inventory_pooled_seed_surfaces.py`.

**Key correction**: the 24-seed raw policy-bin surface is present locally, but not under the `data/simulation/profiles_seed*_season*.parquet` path shape expected by the DR-MDP memo. It is split across `data/hetzner_results/pooled_bins_run` (16 seeds, 80 backtest parquets) and `data/hetzner_results/pooled_bins_run_trackc` (8 seeds, 40 backtest parquets). `data/hetzner_results/audit_full_48seed_v2` has 48-seed scorecards but no raw policy-bin backtest parquets.

**Current evidence**: pooled-policy A/B already has a strong 24-seed leave-one-out screen (`+1.929pp` mean P(57), pooled wins `24/24`), distinct from the rejected pooled-prediction overconfidence fix. The raw profile parquets do not embed seed columns, so consumers must tag seed from the path before pairing rank-1/rank-2 rows.

**Next active SOTA item**: add a v2.6-style uncertainty layer to the existing 24-seed pooled-policy gap before using it as a deployment argument. If rerunning `scripts/dr_mdp_gap_measure.py` on the 24-seed raw surface, first add seed tagging to that loader or feed it a seed-tagged profile frame.

## Status update — 2026-05-06 (pooled-policy gap screen)

**Artifact-level pooled-policy gap screen added** as `docs/sota_audit/2026-05-06-pooled-policy-gap-ci.md` plus `data/validation/pooled_policy_gap_ci_2026-05-06.json`, generated by `scripts/pooled_policy_gap_ci.py`.

**Result**: on the saved 24-seed A/B artifact, the leave-one-out pooled policy has mean P(57) gap `+0.019290`, paired-seed bootstrap CI `[+0.014468, +0.024308]`, and `24/24` positive seed gaps (`exact_sign_p_two_sided=1.19e-07`). Within-pool is similar: mean gap `+0.020816`, CI `[+0.016026, +0.025737]`.

**Scope caveat**: this is not a v2.6 profile block-bootstrap. It resamples saved seed-level gaps only. It does not recompute bins or policies under day-block resamples, and it does not resolve the raw-profile determinism/provenance caveat. Because iid seed resampling is narrower than a proper profile block-bootstrap, this check can falsify the saved-gap screen if it crosses zero, but exclusion of zero is only `positive_screen_unchanged`, not deployment evidence.

**Next active SOTA item**: decide whether to spend the heavier compute/implementation on a profile-level bootstrap check or move to the next bin-side lever. If profile-level work proceeds, it must tag seed from raw-profile paths before pairing rank-1/rank-2 rows.

## Status update — 2026-05-06 (24-seed raw-surface DR-MDP screen)

**Seed-tagged raw-surface ingestion added** as shared pooled-policy loader code and wired into `scripts/dr_mdp_gap_measure.py`. The loader now auto-populates or validates `seed` from `seedN`/`simulation_seedN` path segments when profile paths contain seed markers, preventing silent cross-seed rank pairing when raw pooled parquets omit seed metadata. `--derive-seed-from-path` remains available to force strict path-seed parsing.

**24-seed raw pooled-bin surface measured** as `data/validation/dr_mdp_gap_pooled_24seed_raw_2026-05-06.json`, summarized in `docs/sota_audit/2026-05-06-dr-mdp-gap-pooled-24seed.md`. The run consumed `120` backtest parquets from `data/hetzner_results/pooled_bins_run` plus `data/hetzner_results/pooled_bins_run_trackc`, yielding `218750` profile rows, `21888` rank-1/rank-2 pair rows, and `24` path-derived seeds.

**Result**: point P(57) is `0.052882`. The largest finite-candidate robust gap is the paired-day bootstrap construction: robust P(57) `0.009665`, delta `0.043218`, policy disagreement `0.105240`. That delta remains below the inherited v2.6 CI half-width `0.083333`, so no production DR-MDP solver change is justified by this screen.

**Scope caveat**: this still depends on local untracked raw parquet provenance and path-derived seed identity. It measures solver-side robust sensitivity on the pooled raw surface; it does not replace a profile block-bootstrap for the pooled-policy A/B gap.

## Status update — 2026-05-06 (pooled-policy determinism-bound screen)

**Existing-artifact determinism screen added** as `docs/sota_audit/2026-05-06-pooled-gap-determinism-bound.md` plus `data/validation/pooled_gap_determinism_bound_2026-05-06.json`, generated by `scripts/determinism_gap_bound.py`.

**Result**: current artifacts do not provide a direct paired bound on nondeterminism inside the pooled-policy seed gaps. The n=100 deterministic baseline shows no detectable P(57) distribution shift versus the prior non-deterministic baseline summary (`mean_delta=-0.000003`, `z_vs_prior_std=-0.000204`), but that is a distribution-level screen, not the C0 pooled-policy A/B estimand. The post-cutover deterministic feature screen also shows substantial deterministic seed variation: median per-experiment `delta_p_57_mdp` std `0.015771`, with `21/32` feature experiments at or above the C0 LOO gap std `0.012666`.

**Verdict**: C0 remains `positive_screen_unchanged`, but the determinism/provenance caveat is not resolved. The iid-seed determinism contribution is `not_evaluable_from_existing_artifacts`; a direct bound requires paired same-seed deterministic/non-deterministic reruns on the same pooled-policy gap estimand or embedded deterministic/provider metadata in the raw surfaces.

## SOTA closeout posture before real split audit — 2026-05-06, amended 2026-05-07

Before running a real split audit on a deployable candidate stack, finish or explicitly park the remaining SOTA items that can change the audit's evidentiary standard. The split audit only produces useful output if a concrete candidate Phase 2 stack exists to evaluate against the new selection/outer-evaluation split. Prior Phase 2 winners under legacy `--test-seasons` are legacy selection-on-evaluation-span evidence; if no candidate stack is in flight after closeout, the split-audit plan should report `plan_blocked_no_viable_candidate` instead of manufacturing a candidate to exercise the flags. Split-audit planning also requires a cloud resource/provenance inventory across Hetzner, Vultr, and OCI before setting seed count, parallelism, deadline, or provider allocation.

1. **#10 pooled-policy uncertainty layer**: blocked on a determinism-state precondition. The existing saved 24-seed gap remains `positive_screen_unchanged`, but PR #33 established that further analytical work on existing artifacts cannot resolve whether iid seed variation or provider/model nondeterminism drives the gap. The next move is one of: generate a determinism-certified paired same-seed surface and re-run the C0/DR-MDP screens, explicitly park #10 at the current verdict, or treat #10 only as candidate-generation evidence.
2. **#7 audit-level multiple-testing control**: realized-picks BH/BY and Phase 1 audit-verdict permutation/FDR baselines exist. Treat them as historical p-value truth-up; true e-BH/online-FDR remains deferred until valid e-values/e-processes are designed before a future audit cycle.
3. **#16/#17 candidate-generation methods**: parked as post-audit candidate-generation work unless Eric names a concrete candidate stack and compute budget. Legacy CatBoost, LambdaRank, XE-NDCG, decision-calibration, and quantile variants are negative or zero on the saved Phase 1 artifacts and cannot seed the real split audit. See `docs/sota_audit/2026-05-07-candidate-generation-closeout.md`.
4. **Conditional full-SOTA variants for #13/#14/#15**: keep deferred unless #10, #16, #17, or another candidate creates a deployment-grade policy comparison the v1 falsification harness cannot answer.

**Initial split-audit plan (2026-05-07)**: `docs/sota_audit/2026-05-07-real-split-audit-plan.md` records current verdict `plan_blocked_no_viable_candidate`. The plan also identifies an orchestration blocker: `bts experiment screen/select` support `--selection-seasons` and `--outer-eval-seasons`, but `scripts/audit_driver.py` still launches remote screening with legacy `--test-seasons 2024,2025`. A real cloud split audit needs split-flag pass-through or a dedicated split-audit launcher before compute.

## Status update — 2026-05-08 (pooled-policy Phase C/Phase D closeout)

**Cycle verdict**: `cycle_closed_no_deployable_candidate`. The determinism/provenance-certified 100-seed Phase C surface completed across Hetzner (48 seeds) and OCI (52 seeds), but the deploy-relevant Phase D temporal split falsified the pooled-policy candidate: shipped production mean P(57) on 2025 was `0.127678`, selection-built pooled candidate mean P(57) was `0.064691`, mean gap was `-0.062987`, provider-stratified CI was `[-0.065250, -0.060757]`, and `0/100` seeds were positive. Hetzner and OCI gaps agreed closely (`-0.062095` and `-0.063810`), so the result is not a provider artifact. The earlier 24-seed C0 result (`+1.929pp`, `24/24` positive, `exact_sign_p=1.19e-7`) and Phase D answer different questions; Phase D is the stricter deployment-relevant estimand.

**Follow-up screens**: recency-weighted, last-season, production-anchored hybrid, and state-segment candidates were evaluated only as candidate-generation diagnostics on the consumed 2021-2025 surface. The final 45-cell state-segment family had `9/45` BH survivors and `9/45` BY survivors at `q<=0.05`, triggering `E3_over_survival_revisit_family_control_before_conclusions`; no segment patch is cleared or frozen for deployment. Stop mining 2021-2025 for this cycle. Primary closeout memo: `docs/sota_audit/2026-05-08-pooled-policy-cycle-synthesis.md`; supporting artifacts: `data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json`, `data/validation/phase_d_pooled_policy_postmortem_2026-05-08.json`, `data/validation/rolling_origin_policy_candidate_screen_2026-05-08.json`, and `data/validation/state_segment_policy_candidate_screen_2026-05-08.json`.

**Infrastructure delivered**: the cycle leaves behind split-aware audit methodology, provider/provenance metadata, OCI inclusion-rule verification, and `audit_attach` recovery tooling for profile runs. Cloud spend was approximately `$30-35` of the `$1000` authorized cap. Future work should start with a fresh pre-registration that fixes the candidate, family-control rule, cloud budget/provider split, and fresh evaluation target before looking at new outcomes.

## Status update — 2026-05-08 (fresh #16 pre-registration draft)

**Fresh-audit pre-registration draft added** as `docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`. The draft chooses #16 decision-aware learning as the next concrete candidate-cycle direction, with candidate code name `decision_weighted_lgbm_v0`.

**Launch posture**: `candidate_frozen_live_forward_logging_ready`. No cloud run, production change, or deploy branch action is cleared by this memo. Candidate training is frozen at `5004b1c8b093da0f8acb11bd728430ebacbf92d3`; official pre-outcome research logging also requires the 2026-05-10 production-pick parity guard.

**Fresh target**: post-registration 2026 live-forward slates generated after candidate freeze, with `2026-05-09` as the earliest eligible calendar date and a `120` eligible-slate minimum unless amended before outcomes are inspected. Pre-memo 2026 data is conservative development/sanity-check data because local `pa_2026`, `backtest_2026`, pick, realized-picks, and validation artifacts already exist.

**Budget posture**: spend `$0` cloud for the initial live-forward logging launch. Use local tests and historical screens first; only escalate to 4-16 deterministic cloud smoke if the candidate is stable, non-negative locally, and separately authorized.

**Implementation slice**: `decision_weighted_lgbm_v0` now has a frozen
candidate training hook: `decision_weight_mode=top_slate_v0` uses an
in-window probe LightGBM model to upweight PA rows attached to top projected
daily batter-games. The probe uses the same feature columns, hyperparameters,
random state, and training window as the final model; v0 accepts same-window
probe overfit risk because it only uses the probe for rank-based, clipped,
mean-normalized training weights. Production defaults remain unchanged because
the hook is inactive unless an experiment config opts in, and the fast-path
experiment runner rejects this experiment because it rewrites baseline blend
configs. The candidate-training freeze is now recorded at
`5004b1c8b093da0f8acb11bd728430ebacbf92d3`. The historical artifact schema/comparison
slice is `bts_candidate_ranked_slate_pair_v1`, using
`bts experiment export-candidate-artifacts` followed by
`bts experiment compare-candidate-artifacts`. The comparison emits scorecard
deltas for local screening; bootstrap CIs, family-control statistics, and the
full fresh-target verdict remain post-collection analysis. The fresh-target
pre-outcome logging command is `bts experiment export-live-candidate-artifacts`;
it writes research artifacts only and does not write picks, cached production
models, posts, cloud assets, or `deploy`. Run it after the production daily
data refresh and use one output directory per date because the v0 manifest is
single-date. As of the 2026-05-10 parity guard, official runs must also pass a
locked `--production-pick-file` and verify with `--require-production-pick-snapshot`.
This clears pre-outcome research logging only; production deploy
remains blocked by fresh-target evidence plus the separate production gates.

## Audit framework

For each area, we capture:

- **Current approach** — what BTS does today
- **SOTA target** — the literature-recommended technique with citation
- **Speculative P(57) impact** — rough order-of-magnitude estimate; refine via brainstorm
- **Implementation effort** — relative scale (S/M/L/XL)
- **Prerequisites** — what must be true before starting
- **Status** — `unstarted` / `in-brainstorm` / `in-implementation` / `shipped` / `parked`
- **Next concrete action** — what to do FIRST when this area is picked up
- **Notes** — running scratch space

**Methodological orientation (added 2026-05-01 evening, after Codex review):** Treat each area not just as "a technique to install" but as "a claim to falsify." The goal is not to ornament the system with respectable methods; it's to find out whether the headline numbers (8.17% pooled P(57), 16-feature blend gain, etc.) survive honest decision-level scrutiny. When the literature SOTA P(57) is ~0.5%, being at 8.17% creates a burden of proof, not a baseline.

**Cross-cutting claim — production equivalence / data lineage (added 2026-05-04 per Codex post-v2.6 review):** In addition to the per-area concerns above, the audit must defend the claim that "the picks the research backtest scores are the picks production would have made on the same dates with the same information." This means: research data lineage = production data lineage at evaluation time (no future leakage, no projected-lineup vs confirmed-lineup mismatch, no upstream feature drift between training and serving). This claim cuts across area **#5 (nested validation must reflect production information sets at fold boundaries)** and area **#13 (OPE must use frozen-at-decision-time information sets)**. It is a defended claim the harness assumes; it is NOT a separate numbered area — it is a constraint on how #5 and #13 must be implemented.

## Prioritization heuristics

Order areas by **expected_P(57)_honesty_or_impact / weeks_of_work**, with hard prerequisites respected. Highest expected EV starts first.

Per Eric's stated lens (2026-05-01 brainstorm): "the best anyone could possibly make." Don't compromise rigor for speed; do compromise scope when SOTA is ill-defined.

**Deployability constraint (added 2026-05-04 per Codex post-v2.6 review):** SOTA areas are gated by deployable constraints — cost, latency, and provider determinism — that decide whether a method can run daily in production. These are operational constraints on what's deployable, not statistical-validity constraints. A method that is statistically defensible but cannot run within the production daily window (morning lineup-lock to game-start) cannot be deployed regardless of its SOTA-ness. This adds a third axis to the prioritization heuristic: expected EV / weeks-of-work, gated by daily-runnability.

---

## Audit area inventory (17 items)

### 1. MDP decision layer — distributional DP / DR-MDP / CVaR  [⏳ partial — objective audit + DR-MDP screen shipped 2026-05-06]

- **Status update 2026-05-06**: PR #26 audited the current MDP and established that `solve_mdp` already maximizes reachability P(57), not E[streak]. CVaR-over-streak is ruled out. PR #27 added `scripts/dr_mdp_gap_measure.py`, a finite-candidate rectangular DR-MDP measurement screen. Initial run on explicit 2021-2025 canonical profiles produced `point_p57=0.039960`, `max_delta_p57=0.035678`, below the inherited v2.6 CI half-width of `0.083333`. No production solver change is justified on this surface. The stronger 24-seed profile surface is absent locally and would need regeneration before making a stronger solver-side claim.
- **Status update 2026-05-02**: The two-knob mean correction (PA dependence + cross-game pair) on the existing tabular MDP IS now live in `bts.validate.dependence.build_corrected_transition_table` and exercised by the falsification harness. CVaR-MDP and full DR-MDP are NOT yet implemented; the corrected transitions feed the SAME vanilla value iteration. Real-data run produced corrected_pipeline_p57 = 0.83% [0, 3.75%] vs headline 8.17% — verdict HEADLINE_BROKEN. See `data/validation/falsification_harness_2026-05-02.json` and `project_bts_2026_05_02_task13_verdict.md`.
- **Current**: Vanilla value iteration on point estimates of P(hit). Single policy table indexed by (streak, days_remaining, saver, quality_bin). Last solved 2026-04-15.
- **SOTA target** (sequence, immediate to advanced):
  1. **Exact distributional DP** + **robust value iteration over calibrated probability intervals** — keeps the tabular structure, adds tail-risk awareness without deep-RL machinery. This is the immediate target.
  2. **Distributionally Robust MDP** (Iyengar 2005, Wiesemann et al. 2013) — explicit parameter-uncertainty handling.
  3. **CVaR-MDP** (Chow & Ghavamzadeh 2014) — explicit tail-risk objective.
  4. (Far end) Distributional RL (C51 / QR-DQN / IQN, Bellemare et al. 2017, Dabney et al. 2018) — only if abandoning the tabular MDP, likely overkill given 103K states.
- **Speculative ΔP(57)**: unknown; initial DR-MDP screen found no solver-side production-change signal on the explicit 2021-2025 canonical surface.
- **Effort**: L
- **Prerequisites**: (a) Fresh post-bpm 24-seed pooled backtest; (b) **Offline policy evaluation infrastructure (#13) for honest comparison against current vanilla VI**.
- **Status**: measurement-gated; current solver objective is correct for P(57), CVaR-over-streak is ruled out, and DR-MDP did not clear the CI bar on the explicit 2021-2025 canonical surface.
- **Next action**: Do not build production DR-MDP unless a regenerated 24-seed profile surface or a materially different bin manifold clears the `scripts/dr_mdp_gap_measure.py` screen. Prefer bin-side / multi-seed pooling first.

### 2. Calibration — Beta / Venn-Abers / spline + binary-y conformal

- **Current**: Isotonic regression (`src/bts/model/calibrate.py`, default OFF behind `BTS_USE_CALIBRATION`). Validated as DROP at n=61 on 2026-05-01.
- **SOTA target**: **Cross-fitted comparison** of Platt / Beta (Kull, Filho, Flach 2017) / spline / isotonic / **Venn-Abers** under proper scores + top-bin calibration. Venn-Abers (Vovk & Petej 2014) is more relevant than Beta if probability *intervals* are needed. Plus binary-y conformal proper (#11).
- **Speculative ΔP(57)**: +0.3–0.5pp (helps MDP threshold decisions when activated)
- **Effort**: S (any single calibrator) + M (cross-fitted comparison + binary-y validation)
- **Prerequisites**: n>=200 resolved picks (currently 61). Estimated late June 2026.
- **Status**: unstarted; deferred until n threshold reached.
- **Next action**: When n reaches 200, brainstorm Beta-vs-isotonic-vs-Venn-Abers comparison under proper-scoring-rule eval (#12). Separately: per-bucket coverage tests for binary-y conformal validation (#11) unblocks #8 below.

### 3. Feature attribution / interpretability — TreeSHAP + ALE + model reliance

- **Current**: One-feature-at-a-time sensitivity analysis (used today on Beck investigation). Misses interaction effects.
- **SOTA target**: **TreeSHAP** (Lundberg & Lee 2017) for local attribution + **ALE plots** (Apley & Zhu 2020) for marginal effects without correlation artifacts + **conditional PFI** + **SHAP interactions** + **model reliance** (Fisher, Rudin, Dominici 2019). FastTreeSHAP is a speed upgrade, not a conceptual one.
- **Speculative ΔP(57)**: 0.0pp (interpretability tool, not a P(57) lever; high value for diagnosis when uncomfortable picks land).
- **Effort**: S (drop-in for TreeSHAP) + S (ALE).
- **Prerequisites**: None.
- **Status**: unstarted; **demoted in execution order** — Codex flagged this as comfortable-but-low-leverage. Still worth doing eventually, just not first.
- **Next action**: After validation/scoring work lands, add `shap` + ALE library to deps, write `src/bts/model/attribution.py` wrapper. Retro Beck pick using TreeSHAP. ~2-4 hours.

### 4. Audit experiment design — e-values / e-processes (Safe Anytime-Valid Inference)

- **Current**: Fixed-n with t-stat threshold + dual-stratum dual-split protocol. Each Phase 2 audit runs to completion regardless of mid-stream signal.
- **SOTA target**: **Safe Anytime-Valid Inference** framing (Ramdas & Grünwald 2024) — e-values, e-processes, confidence sequences. Howard et al. 2021 is the same lineage but the SAVI framing is the current one. Plus **e-values for combining** (Vovk & Wang 2021).
- **Speculative ΔP(57)**: 0.0pp directly; ~30-50% audit-cycle compute reduction (compounds: more audits per dollar).
- **Effort**: M (research + integrate into `bts.experiment` framework).
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Read Ramdas & Grünwald 2024 SAVI paper. Design how to integrate e-process testing into `audit_driver.py` — specifically: per-experiment early-stop criterion + protocol for downstream pooling.

### 5. Validation methodology — nested purged blocked CV + lockbox

- **Status update 2026-05-06**: #5 is no longer accurately described as unstarted. The manifest path shipped as `bts validate split-manifest` / `src/bts/validate/splits.py`, and manifest-bound validators (`scorecard --manifest`, `conformal-gate`, `policy-value-eval`, `rare-event-ce-is`) carry `lockbox_held_out` + `manifest_metadata`. A current split inventory is recorded at `data/validation/nested_cv_lockbox_inventory_2026-05-06.json`; summary memo: `docs/sota_audit/2026-05-06-nested-cv-lockbox-scoping.md`. Remaining #5 work is uniform adoption and nested tuning discipline, especially `bts experiment screen/select` and the falsification harness.
- **Status update 2026-05-06 (PR #37)**: `bts experiment screen/select` now support an opt-in season-level split via `--selection-seasons` + `--outer-eval-seasons`. The implementation fails closed on overlapping season sets, rejects mixing split flags with legacy `--test-seasons`, keeps Phase 1/2 decisions on the selection span, evaluates the final selected stack once on the outer span, and labels split artifacts with `production_deploy_claim=false`. This closes the immediate implementation next action from `docs/sota_audit/2026-05-06-experiment-runner-nested-split-design.md`; it does not complete #5 adoption until real audit artifacts consume the split and the remaining day-level/manifest paths are addressed.
- **Current**: Manifest-based rolling-origin CV + lockbox infrastructure exists and is tested, and `bts experiment screen/select` now have an opt-in season-level selection/outer-evaluation split. Not every validation/audit path consumes the manifest or split contract yet, so #5 remains adoption-incomplete.
- **SOTA target**: **Nested rolling-origin CV with purging/embargo** as the more general target (per Codex). **Combinatorial Purged CV** (López de Prado 2018) is finance-derived; useful but not the only SOTA. Plus **reverse-CV diagnostics** for selection-bias detection. Plus **untouched lockbox** (final season/month never seen during audit) for honest final assessment. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) for selection-bias correction across audit batches.
- **Speculative ΔP(57)**: 0.0pp directly; better honesty about which "shipped" features are real signal vs selection bias. **May shrink the historical 8.17% claim** when prior audits get re-evaluated.
- **Effort**: M
- **Prerequisites**: None (but compounds badly if started after lots of audits already executed — plan now).
- **Status**: shipped opt-in / adoption pending.
- **Next action**: First adoption decision pending. Keep legacy `--test-seasons` as the default path, but when a real candidate audit or deployable stack is in flight, run `bts experiment screen/select` with explicit `--selection-seasons` and `--outer-eval-seasons`, save the selection and outer-evaluation artifacts, and summarize whether the selected stack survives the outer span. Treat that result as methodology evidence only (`production_deploy_claim=false`), then decide whether the next #5 slice should be day-level manifest integration for `bts experiment` or adoption of the split contract in the remaining falsification harness.

### 6. Distribution shift handling — BOCPD + drift-aware health check

- **Current**: Walk-forward retraining only (daily blend cycle). No explicit drift detection.
- **SOTA target**: **Bayesian Online Changepoint Detection** (Adams & MacKay 2007) for adaptive segmentation; **MMD / KS-CUSUM** on features and residuals; **online calibration drift** monitoring; **label-lag-aware loss monitors**. ADWIN/DDM (2004/2007) are older comparable approaches. Drift detection only matters if it triggers a policy change.
- **Speculative ΔP(57)**: +0.2–0.5pp (helps adaptation to mid-season regime shifts).
- **Effort**: M
- **Prerequisites**: Decide what drift signal does — re-train? Switch to a fallback policy? Fire alert?
- **Status**: unstarted.
- **Next action**: Read Adams & MacKay BOCPD. Decide policy-side response. Design where in the predict pipeline drift signal is computed.

### 7. Multiple testing across audits — e-BH / online FDR

- **Status update 2026-05-06**: A p-value BH/BY baseline shipped for realized-picks attribution cells in PR #23 (`src/bts/validate/fdr.py`, `tests/validate/test_fdr.py`, `scripts/run_realized_picks_fdr.py`, `docs/sota_audit/2026-05-05-realized-picks-fdr.md`, `data/validation/realized_picks_fdr_2026-05-05.json`). That artifact tested the Cut C family (`m=22`) and found all `q_BH = q_BY = 1.000`. This is useful as a classical FDR baseline, but it does **not** close the #7 e-BH / online-FDR target: `1/p` is not a valid e-value construction here, and no sequential audit-level e-process has been designed.
- **Status update 2026-05-06 (audit-verdict FDR)**: A Phase 1 audit-verdict p-value FDR retrospective shipped as `scripts/run_audit_verdict_fdr.py`, `tests/scripts/test_run_audit_verdict_fdr.py`, `docs/sota_audit/2026-05-06-audit-verdict-fdr.md`, and `data/validation/audit_verdict_fdr_2026-05-06.json`. It uses exact paired sign-flip permutation p-values over per-season P@1 deltas from `experiments/results/phase1/*/diff.json`, then applies BH/BY across the family. Result: `m=24`, `0` positive candidates survive BH at q<=0.05, `0` survive BY, and all q-values are `1.0000`.
- **Current**: Classical p-value FDR support exists for realized-picks attribution families and Phase 1 audit verdicts. True audit-level e-BH / online-FDR is still absent because no valid e-values or e-processes are constructed.
- **SOTA target**: **e-BH / online FDR** for sequential audits (Wang & Ramdas 2022). **Knockoffs** (Barber & Candès 2015) attractive in principle but hard under temporal dependence. **Randomization/permutation tests** around the whole audit pipeline. Classical BH (1995) and Storey q-values (2002) are baselines, not SOTA.
- **Speculative ΔP(57)**: 0.0pp directly; honest interpretation of which features ARE legitimate KEEPs vs noise. May trigger re-investigation of historically-shipped features.
- **Effort**: S (BH/eBH is one function call given an array of p-values or e-values).
- **Prerequisites**: None.
- **Status**: p-value baselines shipped for realized-picks attribution and Phase 1 audit verdicts / e-BH and online-FDR deferred.
- **Next action**: Treat the existing p-value FDR artifacts as historical truth-up only. Do not claim e-BH complete from either baseline. If future audit cycles need sequential control, design valid e-values/e-processes before running the cycle and pre-register the tested family before looking at outer-evaluation outcomes.

### 8. Streaming calibration alert — ACI / RCPS

- **Current**: `realized_calibration` health check uses fixed-window thresholds; just patched today (since-deploy filter).
- **SOTA target**: **Adaptive Conformal Inference** (Gibbs & Candès 2021); **Risk-Controlling Prediction Sets** (Bates et al. 2021); **Strongly-adaptive online conformal** (Bhatnagar et al. 2023); **NexCP / weighted conformal** (Barber et al. 2022). Lei et al. 2018 was the wrong target.
- **Speculative ΔP(57)**: 0.0pp; early detection of calibration drift.
- **Effort**: M
- **Prerequisites**: Conformal v1 working (currently parked at branch `feature/conformal-lower-bounds`, gate failed). Need to fix binary-y validation methodology (#11) first.
- **Status**: BLOCKED on conformal v1 unblock.
- **Next action**: When conformal v1 unblocks, read ACI + RCPS papers and design online update path.

### 9. Feature engineering — sequence / transformer / GNN models for batter-pitcher

- **Current**: Tabular features (16 in FEATURE_COLS) + Bayesian shrinkage on bpm.
- **SOTA target** (verified 2026-05-01 via independent web search):
  - **Neural Sabermetrics with World Model** (arxiv 2602.07030, Feb 2026) — LLM continuously pretrained on 10+ years MLB tracking data; ~64% next-pitch accuracy, 78% swing-decision accuracy. Pitch-level not BTS-level.
  - **The Impacts of Increasingly Complex Matchup Models on Baseball Win Probability** (arxiv 2511.17733, Nov 2025) — pitcher+batter neural matchup outcome distributions over 9 outcomes (K/BB/HBP/GO/FO/1B/2B/3B/HR).
  - **Pitcher Performance Prediction via Temporal Fusion Transformer** (ScienceDirect S1546221825005028, 2025).
  - **Singlearity** (Baseball Prospectus) — older NN PA-outcome model; established baseline.
  - **Kevin Garnett "Chasing $5.6M with ML"** (Medium, Feb 2026) — directly addresses BTS via ML; this is the SOTA P@500=77% / P@100=85% benchmark already cited in BTS ARCHITECTURE.md.
  - **REMOVED 2026-05-01**: previous "Mehta & Rao 2023, Sabermetric Sequence Models" — this citation was unverifiable on independent web search and is concluded to be hallucinated. See `data/external/codex_reviews/2026-05-01-sota-audit.md`.
- **Speculative ΔP(57)**: +0.5–1.5pp speculative; high uncertainty; may be more useful as one model voice in the existing 12-blend than as a replacement.
- **Effort**: XL
- **Prerequisites**: GPU compute (Mac MPS or cloud); training data pipeline that surfaces PA sequences instead of aggregated features; #12 proper-scoring suite for honest comparison vs Garnett's benchmark.
- **Status**: unstarted.
- **Next action**: Read Neural Sabermetrics paper + Matchup Models paper + Garnett's writeup. Scope a brainstorm for adding a sequence-model output as one feature in the existing 12-blend. Compare against Garnett's reported P@500=77% as external SOTA benchmark.

### 10. Distribution-aware ensemble — predictive stacking

- **Status update 2026-05-08**: The pooled-policy candidate-generation cycle is closed with no deployable candidate. The 100-seed Phase D temporal split falsified the 24-seed C0 screen under a stricter deployment-relevant estimand, and local follow-up screens did not recover a clean candidate. The state-segment FDR screen ended in `E3_over_survival_revisit_family_control_before_conclusions`, so no segment patch is frozen for deployment. See `docs/sota_audit/2026-05-08-pooled-policy-cycle-synthesis.md`.
- **Status update 2026-05-06**: pooled-seed inventory added. Do not conflate two different hypotheses: pooled prediction was rejected as an overconfidence fix on 2026-04-29, but pooled policy / pooled bins has a separate positive decision-layer screen (`data/validation/pooled_policy_ab_24seed_consolidated.json`: leave-one-out mean P(57) gap `+1.929pp`, wins `24/24`). Raw 24-seed policy-bin parquets are locally present under `data/hetzner_results/pooled_bins_run` plus `data/hetzner_results/pooled_bins_run_trackc`, but determinism metadata is not embedded and seed identity is path-derived. Next work is uncertainty screening of the pooled-policy gap, not a production pooled-prediction cutover.
- **Status update 2026-05-06 addendum**: artifact-level paired-seed screen on the saved 24-seed pooled-policy gap is positive: leave-one-out mean P(57) gap `+1.929pp`, paired-seed bootstrap CI `[+1.447pp, +2.431pp]`, `24/24` positive seed gaps. This leaves the positive screen standing under an iid-seed assumption; it is not profile block-bootstrap or deployment proof.
- **Current**: Single-seed production model (BTS_LGBM_RANDOM_STATE=42). Multi-seed pooling REJECTED 2026-04-29 for the wrong reason (overconfidence framing was iteration noise).
- **SOTA target**: **Predictive stacking of distributions** (Yao, Vehtari, Simpson, Gelman 2018 "Using Stacking to Average Bayesian Predictive Distributions") — out-of-fold log-score weighted, more honest than Bayesian Model Averaging when models are misspecified. **Conformal ensemble** with calibrated weights as alternative. **Stack against downstream MDP value** not generic PA accuracy (decision-aware, see #16).
- **Speculative ΔP(57)**: +0.2–0.5pp variance reduction; potentially more if combined with #1 (CVaR-MDP) which can directly use ensemble variance.
- **Effort**: M
- **Prerequisites**: Compute budget for multi-seed daily training (~10x current cost); #12 proper-scoring suite for stacking weights.
- **Status**: pooled-policy candidate closed as `cycle_closed_no_deployable_candidate`; production cutover not cleared; predictive-stacking implementation still unstarted.
- **Next action**: Do not spend more analytical work on the consumed 2021-2025 pooled-policy surface as if it can clear deployment evidence. A future #10 cycle must be newly pre-registered with a fresh evaluation target, explicit family-control rule, provider/provenance plan, and acceptance thresholds. Keep predictive stacking / pooled-prediction cutover separate until proper-scoring evidence clears the 2026-04-29 Brier failure.

### 11. Validation methodology for binary classification calibrators

- **Status update 2026-05-06**: #11 is no longer accurately described as the old `scripts/validate_conformal.py` anti-pattern. The shipped path is `bts validate conformal-gate` / `src/bts/validate/conformal_gate.py`, schema `conformal_validation_v2`, which replaces per-row binary-y coverage with bucket-level lower-bound validity plus tightness. A fresh current-surface run is recorded at `data/validation/conformal_gate_v2_2026-05-06.json` with manifest `data/validation/split_manifest_conformal_2026-05-06.json`: all 6 method/alpha cells failed, `ship_set=[]`, verdict `NO_PRODUCTION_DEPLOY`. Parked conformal-v1 artifacts exist under `data/conformal/`, but production clearance remains blocked pending a non-empty v2 `ship_set` and #12 calibration diagnostics on decision/selectable rows. Summary memo: `docs/sota_audit/2026-05-06-conformal-gate-v2-refresh.md`.
- **Current**: v2 conformal-gate infrastructure exists and is tested, but no conformal lower-bound method/alpha is production-cleared on the current canonical profile surface.
- **SOTA target**: **Reliability diagrams with uncertainty bands**; **Brier decomposition** (reliability/resolution/uncertainty); **top-bin calibration**; **class-conditional / Mondrian conformal diagnostics** (Sesia & Romano 2021); **Venn-Abers intervals**; **decision-bucket calibration**; **conditional coverage diagnostics** (Romano et al. 2020). Per-bucket coverage is necessary but too narrow on its own — the binary-y validation problem needs the full probabilistic-validation toolkit.
- **Speculative ΔP(57)**: 0.0pp directly; unblocks the conformal v1 ship which then enables #8 + parts of #1.
- **Effort**: M
- **Prerequisites**: None (the parked conformal v1 branch has the calibrator infrastructure ready).
- **Status**: conformal-gate v2 shipped / production deploy blocked by empty `ship_set`.
- **Next action**: Do not revisit the old per-row coverage gate. Re-run this area only when a non-empty method/alpha candidate appears or a new calibration method is proposed; use #12 proper-scoring and decision/selectable-row diagnostics as the supporting evidence before unblocking any conformal lower-bound deploy.

---

### 12. (NEW, 2026-05-01 evening) Probabilistic forecast evaluation suite

- **Status update 2026-05-06**: #12 is no longer unstarted. Phase 1 shipped as `src/bts/validate/proper_scoring.py`, `tests/validate/test_proper_scoring.py`, and scorecard integration (`bts validate scorecard` includes `proper_scoring`). Phase 2 shipped realized-picks calibration (`docs/sota_audit/2026-05-04-realized-picks-calibration.md`, `data/validation/realized_picks_canonical_2026-05-04.parquet`). Phase 3 shipped realized-picks attribution P0/P1 and the p-value BH/BY FDR baseline (`docs/sota_audit/2026-05-05-realized-picks-attribution*.md`, `docs/sota_audit/2026-05-05-realized-picks-fdr.md`, `data/validation/realized_picks_fdr_2026-05-05.json`). Strict current-model realized-picks evidence remains underpowered; re-run phase 2/3 after more post-bpm picks resolve.
- **Status update 2026-05-10**: Phase 2/3 realized-picks refresh reran on production picks through the 2026-05-09 slate against a fresh 2026 PA frame (`data/validation/realized_picks_canonical_2026-05-10_p1.parquet`, `data/validation/realized_picks_fdr_2026-05-10.json`, `docs/sota_audit/2026-05-10-realized-picks-refresh.md`). Strict current-model (`post_bpm`) evidence is now n=15 resolved rows (10/15 hits, mean_p=0.737, gap=+7.0pp) and remains underpowered. The historical `post_pooled_mdp_pre_bpm` DD x not_park_driven x Q4 overconfidence watch item does not currently reproduce in strict `post_bpm` rows (n=5, 4/5 hits, mean_p=0.726, gap=-7.4pp). Q1 x park_driven still has n=0, so the original low-skill park-driven hypothesis remains untestable. Refreshed Cut C FDR family is m=24 with all q_BH=q_BY=1.000. No strategy/model edit is supported by this refresh.
- **Current**: Proper-scoring-rule evaluation is implemented and wired into scorecard output. Realized-picks calibration/attribution artifacts exist, but the strict current-model sample is still too small for a deployable calibration verdict. External benchmark reconciliation remains only partially addressed and should be folded into future scoring surfaces.
- **SOTA target**: **Proper scoring rules** (Gneiting & Raftery 2007 "Strictly Proper Scoring Rules, Prediction, and Estimation"); **Brier decomposition** into reliability/resolution/uncertainty (Murphy 1973); **top-decile calibration** specifically (the picks live there); **sharpness-vs-reliability framework** (Gneiting et al. 2007); **CRPS** for ranked outputs. Plus **decision-bucket calibration** (calibration restricted to days where the pick is actually selectable as rank-1).
- **Scope expansion (added 2026-05-04 per Codex post-v2.6 review)**: External benchmark reconciliation is part of #12. Garnett 2026 / lokikg-style P@K comparisons must be evaluated under identical temporal guardrails, proper scoring rules, and decision-bucket calibration to be honest; headline P@1 / P(57) reported in different papers under different temporal-leak rules and different calibration conditions are NOT directly comparable. #12 should produce the apples-to-apples external-benchmark reconciliation as part of its first deliverable, not as a downstream task.
- **Speculative ΔP(57)**: 0.0pp directly; foundational for honest model comparison and for replacing P@1 in tuning loops with decision-aware scoring (#16).
- **Effort**: S (most of these are one-pass calculations on existing OOF predictions).
- **Prerequisites**: None.
- **Status**: shipped in phases 1/2/3 for proper scoring + realized-picks calibration/attribution; external benchmark reconciliation remains an open analysis obligation when a comparable surface is available.
- **Next action**: Re-run realized-picks phase 2/3 when strict `post_bpm` resolved rows reach roughly n=30, strict `post_bpm` DD x not_park_driven x Q4 reaches n=10-15, or any Q1 x park_driven strict-model row appears. Use the proper-scoring and realized-picks artifacts as the evidence layer for bin-side / multi-seed pooling work.

### 13. (NEW, 2026-05-01 evening) Offline policy evaluation (OPE)  [✅ shipped 2026-05-02 — falsification harness Task 13]

- **Status (2026-05-02)**: SHIPPED as part of the Task 13 falsification harness. `bts.validate.ope` module includes `audit_fixed_policy` (frozen-policy held-out), `audit_pipeline` (LOSO refit + re-solve per fold), `corrected_audit_pipeline` (LOSO with global corrected policy), paired hierarchical block bootstrap, and policy regret table. Real-data run on 24-seed × 5-season backtest verdict: HEADLINE_BROKEN. v1 simplification: terminal-reward MC, not full sequential DR; documented inline.


- **Current**: `bts.validate.ope` now provides the v1 falsification-harness OPE path: frozen-policy held-out audit, LOSO pipeline refit/re-solve, corrected-pipeline audit, paired hierarchical block bootstrap, and policy regret tables. This is stronger than the old `evaluate_mdp_policy` self-score, but it remains terminal-reward MC rather than full sequential DR-OPE or per-decision IS.
- **SOTA target**: **Doubly-robust OPE** (Jiang & Li 2016 "Doubly Robust Off-policy Value Evaluation"); **per-decision IS estimators** (Precup et al. 2000); **Q-evaluation with held-out fitted Q**; **policy regret against baseline policies**; **uncertainty intervals around policy value** (bootstrap or bayes). The MDP layer is a fully offline batch-RL problem and should be evaluated as such.
- **Speculative ΔP(57)**: 0.0pp directly; foundational. Currently we **can't honestly compare** a CVaR-MDP policy to vanilla VI without OPE infra. Also: this is the right place to find out if the 8.17% claim is real (component of the falsification harness).
- **Effort**: M
- **Prerequisites**: None.
- **Status**: v1 falsification harness shipped / full sequential DR-OPE deferred.
- **Next action**: Only implement full sequential DR-OPE or per-decision IS if a policy candidate needs deployment-grade comparison beyond the current harness. Until then, use the existing `bts.validate.ope` harness and keep its v1 simplification documented in any production argument.

### 14. (NEW, 2026-05-01 evening) Rare-event Monte Carlo with variance reduction  [✅ shipped 2026-05-02 — falsification harness Task 13]

- **Status (2026-05-02)**: SHIPPED as `bts.simulate.rare_event_mc` — direct deterministic-theta CE-IS sampler (bypassing the planned LatentFactorSimulator after a structural-bug discovery; documented), unbiasedness gate validated against `bts.simulate.exact`. Real-data verdict from harness: rare_event_ce_p57 = 0.0034 [0.0025, 0.0045], independently corroborates the HEADLINE_BROKEN verdict (the CE-IS estimate is even lower than the corrected pipeline estimate). v1 fits only theta_0 constant logit shift; per-step / per-action tilt deferred to v1.5.


- **Current**: `bts.simulate.rare_event_mc` provides a deterministic-theta CE-IS sampler and unbiasedness tests against `bts.simulate.exact`. Naive MC and analytical absorbing-chain estimates still exist; richer variance-reduction variants are not implemented.
- **SOTA target**: **Cross-entropy importance sampling** (Rubinstein 1997, Rubinstein & Kroese 2017); **subset simulation** (Au & Beck 2001); **multilevel Monte Carlo** (Giles 2008) if applicable. P(57) is an extreme survival event — naive MC needs ~10^4-10^5 trials to estimate with reasonable variance, and correlated game-day outcomes inflate variance further.
- **Speculative ΔP(57)**: 0.0pp directly; provides honest CIs around the 8.17% number. **Critical component of the falsification harness** — if the honest CI on 8.17% is `[2pp, 14pp]` rather than the implied tight band, that changes the audit posture entirely.
- **Effort**: M
- **Prerequisites**: None.
- **Status**: CE-IS v1 shipped / advanced tilts deferred.
- **Next action**: Add per-step/per-action tilt, subset simulation, or multilevel MC only if the current CE-IS CI or methodology sensitivity becomes a blocker for the real split audit or a deployment-grade policy comparison.

### 15. (NEW, 2026-05-01 evening) PA-independence and cross-game dependence modeling  [✅ shipped 2026-05-02 — falsification harness Task 13 (v1) + Issue #7 (v2)]

- **Status (2026-05-04 v2.6)**: PR #8 merged at `1a0eefb`. v2.6 added profile-level paired hierarchical block-bootstrap CI (Politis–Romano stationary, expected_block_length=7, n=500) for `corrected_pipeline_p57` within the harness path. Gate-class transition collapsed at half-headline=0.04085 threshold: under block-bootstrap, the v1 BROKEN classification was a percentile-CI artifact (ci_upper 0.0375 was 0.7pp below threshold on a CI whose grid resolution is 0.83pp); all 6 ablation cells gate REDUCED. v2.5 point-estimate attribution survives with precise framing (B/C each shift +1.67pp single-mode; combined +2.50pp via 2023 fold synergy; A no observable effect at current resolution AND ~3.5× slower in this runner). Full v2.6 addendum: `docs/sota_audit/2026-05-03-harness-v2.5-attribution.md`. **Note**: v2.6 is a CI-methodology piece on top of v1 dependence-modeling simplifications; fuller out-of-fold PA/cross-game residual-dependence modeling remains scoped but unstarted.

- **Status (2026-05-03 v2.5 evening — partial attribution)**: v2.5 SHIPPED via 6-cell nested factorial ablation. **Headline finding (Codex matrix-reviewed): in this six-cell nested ablation, Change A (fold-local parameter estimation) has no observable effect on the corrected_pipeline_p57 point estimate conditional on per-fold policy, while Changes B and C each independently produce most of the v1→v2 verdict shift.** A_effect_given_per_fold = 0.00pp; nested AB interaction = 0.00pp; both at metric resolution of 1/120 = 0.0083pp. Cells V010 = V001 = V101 = 0.0250 (any single one of B-alone, C-alone, or A+C produces same coarse scalar); V011 = V111 = 0.0333. **Caveat**: point estimates are coarse (1/120, 3/120, 4/120 successes); same-scalar across cells doesn't establish mechanism equivalence. Path-sum residual is 17% of total (decomposition is descriptive, not additive). **Defensible**: A is below detection in measured per-fold contrasts; B/C drive the shift. **Not defensible**: "A is methodology theater," "B/C substitutable," "deploy cell 010 as production policy" — those are interpretive jumps beyond what 6 coarse scalars establish. See `docs/sota_audit/2026-05-03-harness-v2.5-attribution.md`. v2.6 priorities (historical, as recorded 2026-05-03): (1) block-bootstrap CI replacing 5-fold percentile, (2) cell 101 full-rep verification (cheap), (3) mechanism inspection (does V010=V001 reflect same fold patterns?), (4) distribution shift remediation as strategic priority. **Note (2026-05-04)**: priorities (1)-(3) shipped via v2.6; priority (4) framing superseded — see 2026-05-04 status update at top of tracker for under-confidence/top-of-slate reframe.

- **Status (2026-05-02 v2 evening)**: v2 SHIPPED via Issue #7 — closes v1's two methodology gaps (later refuted by v2.5 attribution; see above). Per-rank-1-bin `rho_pair_per_bin` correction (5-element vector) and within-fold dependence-parameter estimation (rho_PA, tau, rho_pair refit per LOSO fold's 4 training seasons). New diagnostic 5×5 lower-triangular heatmap via `pair_residual_correlation_per_cell`. v2 verdict: `corrected_pipeline_p57 = 0.0333 [0.0000, 0.1167]` → `HEADLINE_REDUCED`. **However (per Codex round 1 review of memo)**: v2 point estimate is *still below* half-headline (0.0408); the gate-class transition from v1's `BROKEN` to v2's `REDUCED` is CI-driven, not point-estimate driven. Q4 sign reversed (v1 antagonism → v2 cooperative in 2/5 folds, near-zero in 3/5) — heterogeneous, not "artifact." See `docs/sota_audit/2026-05-02-harness-v2-comparison.md` and `data/validation/falsification_harness_v2_2026-05-03.json`.

- **Status (v1, 2026-05-02)**: SHIPPED as `bts.validate.dependence` — Pearson residuals + within-batter-game residual correlation via cluster bootstrap, logistic-normal random-intercept fit (cross-pair products + brentq inversion, NOT the textbook `tau^2 ≈ var-1` which Codex round 2 caught as backwards), cross-game pair-residual permutation test, and `build_corrected_transition_table` (two-knob mean correction). Real-data findings: rho_PA_within_game = 0.0012 [0.0009, 0.0015] (small but nonzero), rho_pair_cross_game = -0.0074 [-0.0607, 0.0476] (essentially zero). The PA-correction collapses corrected_pipeline_p57 by ~10× — this is the dominant signal that drove the v1 HEADLINE_BROKEN verdict, partly relaxed in v2 to HEADLINE_REDUCED.


- **Current**: v1/v2/v2.5/v2.6 dependence harnesses test and model parts of PA and cross-game dependence: residual correlation diagnostics, within-batter-game bootstrap, logistic-normal random-intercept correction, cross-game pair-residual permutation tests, per-bin `rho_pair`, fold-local parameter estimation, factorial attribution, and profile-level block-bootstrap CI. The remaining open target is fuller out-of-fold PA/cross-game residual-dependence modeling, not basic assumption testing.
- **SOTA target**: **Test PA independence empirically** — fit a within-game-residual covariance model, compare to independent-baseline log-likelihood. **Test cross-game dependence** — same weather slate, same modeling errors, correlated bullpen availability, cross-game park effects on the same day. Methods: copula approaches; conditional residual models; permutation tests for independence (Romano 1989). Decision implication: if dependence is non-trivial, the double-down policy under-weights correlation risk and CVaR-MDP becomes more important.
- **Speculative ΔP(57)**: -0.5 to +0.5pp depending on direction. May reduce the headline number (good — honest) and shift policy toward more conservative doubles.
- **Effort**: M
- **Prerequisites**: None.
- **Status**: v1/v2/v2.5/v2.6 shipped / full residual-dependence model deferred.
- **Next action**: Build fuller out-of-fold residual/covariance modeling only if it blocks a new candidate audit or materially changes a deployment argument. Otherwise use the existing harness conclusions and carry the v1/v2 simplification caveats forward.

### 16. (NEW, 2026-05-01 evening) Decision-aware learning

- **Current**: PA model is optimized for binary cross-entropy on hit/no-hit. The contest objective is a nonlinear tail event (P(57)). Training and decision metrics are decoupled.
- **SOTA target**: **Smart Predict-then-Optimize** (Elmachtoub & Grigas 2022 "Smart Predict, then Optimize"); **end-to-end loss surrogates** that target eventual MDP value; **decision-focused learning** (Wilder et al. 2019); **reweighting by downstream policy sensitivity** (train with weights proportional to how much each PA's prediction affects MDP decisions).
- **Speculative ΔP(57)**: +0.2-1.0pp speculative; high uncertainty. **The decoupling between PA-Brier and downstream-MDP-value may explain part of why feature-engineering returns are diminishing** (today's morning verdicts).
- **Effort**: M (reweighting) to L (full SPO surrogate).
- **Prerequisites**: #12 probabilistic-scoring-suite + #13 OPE for honest measurement.
- **Status update 2026-05-07**: Parked as post-audit candidate generation in `docs/sota_audit/2026-05-07-candidate-generation-closeout.md`. #16 remains plausible, but starting a new SPO/reweighting implementation now would create a fresh exploratory candidate before the project has named the real split-audit target.
- **Status update 2026-05-08**: Reactivated as the next concrete candidate-cycle direction after pooled-policy closeout. Pre-registration draft: `docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`. The draft names `decision_weighted_lgbm_v0` as a sensitivity-weighted LightGBM candidate. The v0 training hook is `top_slate_v0`: train an in-window probe LightGBM model using the same feature columns, hyperparameters, random state, and training window as the final model; aggregate probe PA probabilities to batter-game probabilities; rank batter-games within date; and upweight PA rows attached to top projected daily candidates before fitting the final model. Same-window probe overfit risk is accepted for v0 because the probe is used only for clipped, rank-based training weights, not evaluation probabilities. The historical paired-artifact schema is `bts_candidate_ranked_slate_pair_v1`, exported with `bts experiment export-candidate-artifacts` and scored with `bts experiment compare-candidate-artifacts`; the comparison path is a local scorecard-delta screen and intentionally does not claim fresh-target CI/family-control survival. The fresh-target logging command is `bts experiment export-live-candidate-artifacts`, which emits pre-outcome production/candidate slates without writing production picks or model caches.
- **Status update 2026-05-10**: Fresh-target logging is gated on production-pick parity. Official `decision_weighted_lgbm_v0` live-forward artifacts must snapshot the locked `data/picks/YYYY-MM-DD.json` via `--production-pick-file` and pass `bts experiment verify-candidate-artifacts --require-live-preoutcome --require-production-pick-snapshot` before counting toward the fresh target. The snapshot schema is `production_pick_snapshot_v1` and embeds the full locked pick JSON plus source SHA-256. This preserves the production decision actually submitted to BTS alongside the paired ranked slates and prevents candidate-vs-production analysis from drifting away from the locked pick context.
- **Status update 2026-05-10 (comparison rule)**: The fresh-target primary estimand is pre-registered as paired candidate-minus-production `p_57_mdp` delta from resolved ranked-slate artifacts, with one-sided candidate-better block bootstrap (`expected_block_length=7`, `n_bootstrap>=1000`, `seed=57016`), rank-1 Brier/log-loss guardrails, no-peeking before `120` eligible resolved slates, explicit inconclusive/rejection states, and drop-rather-than-coerce handling for production-void slates. Production pick snapshots are parity/audit guards and slot diagnostics, not a replacement for the ranked-slate scorecard until a candidate policy-selection surface exists.
- **Status update 2026-05-10 (leaderboard clue audit)**: Public leaderboard data now has an exploratory clue memo and prospective candidate-join artifact surface in `docs/sota_audit/2026-05-09-leaderboard-clue-audit.md`. The post-hoc consensus result is explicitly survivorship-biased and supports no production edit; the forward protocol is gated on pre-lock visibility, a fixed cohort, `data/validation/leaderboard_clue_audit_<DATE>.json` artifacts, and at least 30 future resolved disagreement date-slot units before first evaluation.
- **Status**: candidate training frozen at `5004b1c8b093da0f8acb11bd728430ebacbf92d3`; official research live-forward logging is additionally gated on the 2026-05-10 production-pick parity guard; production deploy not claimed.
- **Next action**: Run the frozen pre-outcome logging command on eligible 2026 slates after production daily data refresh, then join outcomes and analyze only under the pre-registered CI/family-control rules. Do not launch cloud compute, production changes, or deploy branch updates from this memo alone.

### 17. (NEW, 2026-05-01 evening) Model-class challenge

- **Current**: LightGBM-only, default hyperparameters, 12-model blend with rotating Statcast feature.
- **SOTA target**: **CatBoost ordered boosting** (Prokhorenkova et al. 2018) — handles target leakage in categorical/target-encoded features which BTS has many of (e.g., bpm). **NGBoost** (Duan et al. 2020) — natively probabilistic outputs (predictive distribution, not point), feeds into #14 directly. **Explainable Boosting Machines** (InterpretML, Nori et al. 2019) — interpretability+accuracy without the SHAP layer. **Monotone-constrained XGBoost/LightGBM** where baseball monotonicities are defensible (e.g., higher bpm → higher P(hit) all else equal).
- **Speculative ΔP(57)**: -0.2 to +0.5pp. Mostly an honest-comparison move — if LightGBM is genuinely best, Codex's "too comfortable" critique gets formally rebutted.
- **Effort**: M
- **Prerequisites**: #5 nested CV + #12 proper scoring rules for honest comparison.
- **Status update 2026-05-07**: Parked as post-audit candidate generation in `docs/sota_audit/2026-05-07-candidate-generation-closeout.md`. Saved CatBoost, LambdaRank, and XE-NDCG surfaces are negative on the legacy Phase 1 artifacts; they do not nominate a model-class stack for the real split audit.
- **Status**: parked as post-audit candidate generation unless Eric names a specific model-class stack and compute budget.
- **Next action**: If reactivated, run a pre-registered model-class bakeoff under proper-score and downstream-value metrics, then nominate a winner only if it clears that candidate-generation gate. Otherwise keep parked.

---

## Suggested execution order (revised 2026-05-01 evening, post-Codex review)

The original execution order put TreeSHAP first as a quick win. Codex's review re-prioritized aggressively: feature-engineering returns are diminishing (today's morning verdicts confirm), and the 16× gap between our 8.17% pooled P(57) and published SOTA ~0.5% means the audit's first job is to **defend the headline number**, not extend it.

Revised order:

1. **Falsification harness for the 8.17% claim** = #13 OPE + #14 rare-event MC + #15 dependence modeling, designed and built together. Goal: try to break the 8.17% number with honest decision-level evaluation under correlated rare-event variance. ~~First concrete area to execute.~~ **(2026-05-04 update: v1 harness path shipped + v2.5 attribution + v2.6 block-bootstrap CI; full-SOTA #13/#14/#15 remains open.)**
2. **#12 Probabilistic forecast evaluation suite** — replace P@1-centric evaluation. Foundation for everything else. **(2026-05-06 update: phase 1/2/3 surfaces shipped; realized-picks verdict remains sample-size-limited; external benchmark reconciliation remains open.)**
3. **#11 Binary-y validation methodology** — unblocks parked conformal v1.
4. **#5 Nested rolling-origin CV + lockbox** — methodology foundation; should happen before any further model-class or feature audits.
5. **#1 MDP robustness (distributional DP / robust VI)** — measurement-gated; current objective is already P(57), CVaR-over-streak is ruled out, and DR-MDP did not clear the CI bar on the explicit 2021-2025 canonical surface.
6. **#16 Decision-aware learning** — parked as post-audit candidate generation unless selected as the next concrete candidate stack.
7. **#10 Predictive stacking** — variance reduction with proper-score weighting.
8. **#4 e-values / e-processes for sequential audits** — audit-compute reduction.
9. **#17 Model-class challenge** — parked as post-audit candidate generation unless a specific stack is selected and budgeted.
10. **#7 e-BH / online FDR retrospective** — truth-up on past audits.
11. **#6 BOCPD drift detection** — production monitoring.
12. **#3 TreeSHAP + ALE** — interpretability after the heavy lifting; use for diagnosis when uncomfortable picks land.
13. **#2 Beta / Venn-Abers calibration** — deferred until n>=200 resolved picks (~late June 2026).
14. **#8 ACI / RCPS online conformal** — blocked on #11.
15. **#9 Transformer / GNN feature** — major research; do after Garnett-comparable benchmarks are in place via #12.

---

## Day 1 retro (2026-05-01)

The audit was kicked off today. Concrete deliverables:
- This tracker doc.
- `project_bts_state_of_art_audit_2026_05_01.md` — pattern identification.
- `project_bts_conformal_v1_validation_gate_failed_2026_05_01.md` — first concrete output (parked).
- Two feedback memos: `feedback_aim_for_state_of_the_art.md` + `feedback_dont_truncate_for_session_length.md` (operating principles for future sessions).
- Conformal v1 implementation through Task 7 (parked branch `feature/conformal-lower-bounds`, commits 5ad4145 → ab59628). Calibrator math is correct; validation methodology needs the binary-y redesign before reattempt.

What was learned:
- **Validation methodology is itself a SOTA-audit area**, not just calibrator math. Per-row coverage tests don't transfer to binary outcomes.
- **The "best anyone could make" framing helps catch quiet quality regressions.** Today's bucket-Wilson-instead-of-conformal retreat was reverted; today's sensitivity-analysis-instead-of-TreeSHAP wasn't caught at the time.
- **First concrete output was rejected by validation, which is success.** The gate worked. Implementation infrastructure (calibrator math, dataclass extensions, predict_local wiring, refit script) is reusable when the validation methodology is fixed.

## Day 1 evening update — Codex adversarial review (2026-05-01 ~22:30 ET)

Eric authorized a Codex (GPT-5.5, high reasoning) adversarial review of this tracker doc via the `consulting-codex` skill, with explicit instruction to use GPT-5.5 wherever it adds value regardless of cost. Goal: catch SOTA blind spots that came from Claude's training-distribution defaults. The review surfaced substantive disagreement. Full output preserved at `data/external/codex_reviews/2026-05-01-sota-audit.md`.

Concrete changes absorbed into this doc:

1. **One fictional citation removed.** Area #9 cited "Mehta & Rao 2023, Sabermetric Sequence Models" — this paper does not exist (verified by independent web search 2026-05-01 evening: nothing matching that author/title combination, only generic sabermetrics pages and an unrelated medical-foundation report). Replaced with verified real references: Neural Sabermetrics with World Model (arxiv 2602.07030, Feb 2026), Matchup Models paper (arxiv 2511.17733, Nov 2025), TFT pitcher performance (ScienceDirect 2025), Singlearity (Baseball Prospectus), Garnett's BTS-direct ML piece (Medium Feb 2026).
2. **Six new audit areas added (#12-#17):** probabilistic forecast evaluation, offline policy evaluation, rare-event MC, dependence modeling, decision-aware learning, model-class challenge. None of these were in the original 11.
3. **SOTA targets updated for areas where the named technique was outdated or wrong:** #1 (exact distributional DP / robust VI before deep RL), #2 (Venn-Abers added), #4 (e-values / e-processes / SAVI framing), #6 (BOCPD over ADWIN/DDM), #7 (e-BH / online FDR over classical BH), #8 (ACI / RCPS over Lei 2018), #10 (predictive stacking over BMA), #11 (full probabilistic-validation toolkit, not just per-bucket coverage).
4. **Execution order rewritten.** TreeSHAP demoted from #1 to #12. New first concrete area: "8.17% falsification harness" combining #13 + #14 + #15.
5. **Methodological orientation added to "Audit framework" section:** treat each area as "a claim to falsify" not just "a technique to install." Codex's full reframe (5 claims A-E) was considered but the area-inventory structure was kept per Eric's preference; the falsification posture is absorbed into the framing.

Codex's full top-3 moves (verbatim summary, ordering matches revised execution order):
1. Build the decision-level validation and rare-event harness — first job is to break the 8.17% claim.
2. Replace P@1-centric evaluation with probabilistic + decision-aware scoring.
3. Robust calibrated policy optimization (cross-fitted calibrators feed robust DP).

This maps directly to revised execution order items 1, 2, and 5 above.

## Open questions for next session

- Does the falsification-harness scope break out cleanly into a single design spec, or three (one per #13/#14/#15)?
- Where to put per-area brainstorm outputs — separate spec docs in `docs/superpowers/specs/` (one per area), or extend this tracker as we go?
- Cadence — one area per session, or grouped (e.g., "validation week" covering #5, #11, #12, #13)?
- Budget for Codex consultations on each individual area's literature scan — at ~$1-2 each, the 17-area inventory implies ~$17-34 of consulting cost over the audit lifetime. Worth it given today's review surfaced one factual error and 6 missing areas.
