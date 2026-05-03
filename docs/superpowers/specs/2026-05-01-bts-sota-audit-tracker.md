# BTS State-of-the-Art Audit — Master Tracker

**Date created**: 2026-05-01
**Last updated**: 2026-05-01 evening (Codex adversarial review absorbed)
**Status**: Active; 17 audit areas. First concrete area: **8.17% falsification harness** (combines areas 13, 14, 15).
**Origin**: `project_bts_state_of_art_audit_2026_05_01.md` — Eric committed to project-wide SOTA audit after observing pattern of Claude defaulting to "existing codebase" or "Eric-friendly" rather than state-of-the-art.

This document is the operating tracker for the audit. It's structured for rolling updates: as each area is brainstormed/scoped/implemented, append status notes here.

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

## Prioritization heuristics

Order areas by **expected_P(57)_honesty_or_impact / weeks_of_work**, with hard prerequisites respected. Highest expected EV starts first.

Per Eric's stated lens (2026-05-01 brainstorm): "the best anyone could possibly make." Don't compromise rigor for speed; do compromise scope when SOTA is ill-defined.

---

## Audit area inventory (17 items)

### 1. MDP decision layer — distributional DP / DR-MDP / CVaR  [⏳ partial — corrected transitions shipped 2026-05-02 via Task 13 falsification harness]

- **Status update 2026-05-02**: The two-knob mean correction (PA dependence + cross-game pair) on the existing tabular MDP IS now live in `bts.validate.dependence.build_corrected_transition_table` and exercised by the falsification harness. CVaR-MDP and full DR-MDP are NOT yet implemented; the corrected transitions feed the SAME vanilla value iteration. Real-data run produced corrected_pipeline_p57 = 0.83% [0, 3.75%] vs headline 8.17% — verdict HEADLINE_BROKEN. See `data/validation/falsification_harness_2026-05-02.json` and `project_bts_2026_05_02_task13_verdict.md`.
- **Current**: Vanilla value iteration on point estimates of P(hit). Single policy table indexed by (streak, days_remaining, saver, quality_bin). Last solved 2026-04-15.
- **SOTA target** (sequence, immediate to advanced):
  1. **Exact distributional DP** + **robust value iteration over calibrated probability intervals** — keeps the tabular structure, adds tail-risk awareness without deep-RL machinery. This is the immediate target.
  2. **Distributionally Robust MDP** (Iyengar 2005, Wiesemann et al. 2013) — explicit parameter-uncertainty handling.
  3. **CVaR-MDP** (Chow & Ghavamzadeh 2014) — explicit tail-risk objective.
  4. (Far end) Distributional RL (C51 / QR-DQN / IQN, Bellemare et al. 2017, Dabney et al. 2018) — only if abandoning the tabular MDP, likely overkill given 103K states.
- **Speculative ΔP(57)**: +1.0–1.5pp (highest single-area direct impact)
- **Effort**: L
- **Prerequisites**: (a) Fresh post-bpm 24-seed pooled backtest; (b) **Offline policy evaluation infrastructure (#13) for honest comparison against current vanilla VI**.
- **Status**: unstarted; deprioritized in execution order behind validation work (Codex critique: don't jump to deep RL on a 103K-state tabular problem).
- **Next action**: Build OPE infra first (#13). Then design brainstorm for robust VI vs CVaR vs full distributional.

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

- **Current**: Walk-forward CV (single fold-per-day) + multi-seed pooling for audits.
- **SOTA target**: **Nested rolling-origin CV with purging/embargo** as the more general target (per Codex). **Combinatorial Purged CV** (López de Prado 2018) is finance-derived; useful but not the only SOTA. Plus **reverse-CV diagnostics** for selection-bias detection. Plus **untouched lockbox** (final season/month never seen during audit) for honest final assessment. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) for selection-bias correction across audit batches.
- **Speculative ΔP(57)**: 0.0pp directly; better honesty about which "shipped" features are real signal vs selection bias. **May shrink the historical 8.17% claim** when prior audits get re-evaluated.
- **Effort**: M
- **Prerequisites**: None (but compounds badly if started after lots of audits already executed — plan now).
- **Status**: unstarted.
- **Next action**: Read López de Prado 2018 ch. 7 + 14. **Carve a recent 2025 month-block as untouched lockbox before any further audit work.** Design retrofit into `bts.experiment.runner`.

### 6. Distribution shift handling — BOCPD + drift-aware health check

- **Current**: Walk-forward retraining only (daily blend cycle). No explicit drift detection.
- **SOTA target**: **Bayesian Online Changepoint Detection** (Adams & MacKay 2007) for adaptive segmentation; **MMD / KS-CUSUM** on features and residuals; **online calibration drift** monitoring; **label-lag-aware loss monitors**. ADWIN/DDM (2004/2007) are older comparable approaches. Drift detection only matters if it triggers a policy change.
- **Speculative ΔP(57)**: +0.2–0.5pp (helps adaptation to mid-season regime shifts).
- **Effort**: M
- **Prerequisites**: Decide what drift signal does — re-train? Switch to a fallback policy? Fire alert?
- **Status**: unstarted.
- **Next action**: Read Adams & MacKay BOCPD. Decide policy-side response. Design where in the predict pipeline drift signal is computed.

### 7. Multiple testing across audits — e-BH / online FDR

- **Current**: Nothing applied. Phase 1's "8 unshipped KEEPs all DROPped against bpm-baseline" is partly explainable as multiple-testing inflation that wasn't controlled for.
- **SOTA target**: **e-BH / online FDR** for sequential audits (Wang & Ramdas 2022). **Knockoffs** (Barber & Candès 2015) attractive in principle but hard under temporal dependence. **Randomization/permutation tests** around the whole audit pipeline. Classical BH (1995) and Storey q-values (2002) are baselines, not SOTA.
- **Speculative ΔP(57)**: 0.0pp directly; honest interpretation of which features ARE legitimate KEEPs vs noise. May trigger re-investigation of historically-shipped features.
- **Effort**: S (BH/eBH is one function call given an array of p-values or e-values).
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Run e-BH retrospectively on the audit verdicts collected in `experiments/results/`. Surface findings: how many shipped features survive FDR correction at q=0.05?

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

- **Current**: Single-seed production model (BTS_LGBM_RANDOM_STATE=42). Multi-seed pooling REJECTED 2026-04-29 for the wrong reason (overconfidence framing was iteration noise).
- **SOTA target**: **Predictive stacking of distributions** (Yao, Vehtari, Simpson, Gelman 2018 "Using Stacking to Average Bayesian Predictive Distributions") — out-of-fold log-score weighted, more honest than Bayesian Model Averaging when models are misspecified. **Conformal ensemble** with calibrated weights as alternative. **Stack against downstream MDP value** not generic PA accuracy (decision-aware, see #16).
- **Speculative ΔP(57)**: +0.2–0.5pp variance reduction; potentially more if combined with #1 (CVaR-MDP) which can directly use ensemble variance.
- **Effort**: M
- **Prerequisites**: Compute budget for multi-seed daily training (~10x current cost); #12 proper-scoring suite for stacking weights.
- **Status**: unstarted; rejection from 2026-04-29 should be re-evaluated.
- **Next action**: Re-read the rejection memo (`project_bts_2026_04_29_pooled_prediction_rejected.md`). Brainstorm whether predictive stacking (vs simple multi-seed mean) addresses the rejection logic.

### 11. Validation methodology for binary classification calibrators

- **Current**: `scripts/validate_conformal.py` uses per-row coverage `(actual_hit ≥ bound).mean()`. For binary y, this collapses to empirical hit rate regardless of bound — not what we want.
- **SOTA target**: **Reliability diagrams with uncertainty bands**; **Brier decomposition** (reliability/resolution/uncertainty); **top-bin calibration**; **class-conditional / Mondrian conformal diagnostics** (Sesia & Romano 2021); **Venn-Abers intervals**; **decision-bucket calibration**; **conditional coverage diagnostics** (Romano et al. 2020). Per-bucket coverage is necessary but too narrow on its own — the binary-y validation problem needs the full probabilistic-validation toolkit.
- **Speculative ΔP(57)**: 0.0pp directly; unblocks the conformal v1 ship which then enables #8 + parts of #1.
- **Effort**: M
- **Prerequisites**: None (the parked conformal v1 branch has the calibrator infrastructure ready).
- **Status**: surfaced 2026-05-01 evening when validation gate fired 0/6 SHIP. unstarted.
- **Next action**: Brainstorm session on binary-y validation methodology specifically. Read Sesia & Romano 2021 + Vovk's class-conditional conformal. Redesign `evaluate_fold` to use per-bucket coverage + reliability diagram + Brier decomposition. Re-run gate; if non-empty SHIP set, unblock conformal v1 deploy.

---

### 12. (NEW, 2026-05-01 evening) Probabilistic forecast evaluation suite

- **Current**: Primary metric is P@1 game-level accuracy. No proper-scoring-rule evaluation, no Brier decomposition, no sharpness-vs-reliability framework. P@1 is too blunt for a chained probabilistic decision system.
- **SOTA target**: **Proper scoring rules** (Gneiting & Raftery 2007 "Strictly Proper Scoring Rules, Prediction, and Estimation"); **Brier decomposition** into reliability/resolution/uncertainty (Murphy 1973); **top-decile calibration** specifically (the picks live there); **sharpness-vs-reliability framework** (Gneiting et al. 2007); **CRPS** for ranked outputs. Plus **decision-bucket calibration** (calibration restricted to days where the pick is actually selectable as rank-1).
- **Speculative ΔP(57)**: 0.0pp directly; foundational for honest model comparison and for replacing P@1 in tuning loops with decision-aware scoring (#16).
- **Effort**: S (most of these are one-pass calculations on existing OOF predictions).
- **Prerequisites**: None.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review).
- **Next action**: Read Gneiting & Raftery 2007. Implement `bts.validate.proper_scoring` module: log loss, Brier, Brier decomposition, reliability diagram with bootstrap bands, sharpness-vs-reliability scatter, decision-bucket calibration. Add to `bts validate scorecard` output.

### 13. (NEW, 2026-05-01 evening) Offline policy evaluation (OPE)  [✅ shipped 2026-05-02 — falsification harness Task 13]

- **Status (2026-05-02)**: SHIPPED as part of the Task 13 falsification harness. `bts.validate.ope` module includes `audit_fixed_policy` (frozen-policy held-out), `audit_pipeline` (LOSO refit + re-solve per fold), `corrected_audit_pipeline` (LOSO with global corrected policy), paired hierarchical block bootstrap, and policy regret table. Real-data run on 24-seed × 5-season backtest verdict: HEADLINE_BROKEN. v1 simplification: terminal-reward MC, not full sequential DR; documented inline.


- **Current**: MDP policy is offline-trained on walk-forward predictions; "evaluation" is `evaluate_mdp_policy` which uses the same point-estimate value function the policy was solved against (not honest cross-validated policy value).
- **SOTA target**: **Doubly-robust OPE** (Jiang & Li 2016 "Doubly Robust Off-policy Value Evaluation"); **per-decision IS estimators** (Precup et al. 2000); **Q-evaluation with held-out fitted Q**; **policy regret against baseline policies**; **uncertainty intervals around policy value** (bootstrap or bayes). The MDP layer is a fully offline batch-RL problem and should be evaluated as such.
- **Speculative ΔP(57)**: 0.0pp directly; foundational. Currently we **can't honestly compare** a CVaR-MDP policy to vanilla VI without OPE infra. Also: this is the right place to find out if the 8.17% claim is real (component of the falsification harness).
- **Effort**: M
- **Prerequisites**: None.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review). **Hard prerequisite for area #1 and the falsification harness.**
- **Next action**: Read Jiang & Li 2016. Design DR-OPE estimator over the existing walk-forward backtest profiles. Include policy-regret bounds against (a) "always-skip" baseline, (b) "always-rank1" baseline, (c) the heuristic strategy.

### 14. (NEW, 2026-05-01 evening) Rare-event Monte Carlo with variance reduction  [✅ shipped 2026-05-02 — falsification harness Task 13]

- **Status (2026-05-02)**: SHIPPED as `bts.simulate.rare_event_mc` — direct deterministic-theta CE-IS sampler (bypassing the planned LatentFactorSimulator after a structural-bug discovery; documented), unbiasedness gate validated against `bts.simulate.exact`. Real-data verdict from harness: rare_event_ce_p57 = 0.0034 [0.0025, 0.0045], independently corroborates the HEADLINE_BROKEN verdict (the CE-IS estimate is even lower than the corrected pipeline estimate). v1 fits only theta_0 constant logit shift; per-step / per-action tilt deferred to v1.5.


- **Current**: P(57) estimated via straightforward Monte Carlo (`bts.simulate.monte_carlo`) and analytical absorbing Markov chain (`exact.py`). Bootstrap CIs reported but use naive sampling that may undercount variance for an extreme survival probability.
- **SOTA target**: **Cross-entropy importance sampling** (Rubinstein 1997, Rubinstein & Kroese 2017); **subset simulation** (Au & Beck 2001); **multilevel Monte Carlo** (Giles 2008) if applicable. P(57) is an extreme survival event — naive MC needs ~10^4-10^5 trials to estimate with reasonable variance, and correlated game-day outcomes inflate variance further.
- **Speculative ΔP(57)**: 0.0pp directly; provides honest CIs around the 8.17% number. **Critical component of the falsification harness** — if the honest CI on 8.17% is `[2pp, 14pp]` rather than the implied tight band, that changes the audit posture entirely.
- **Effort**: M
- **Prerequisites**: None.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review). Component of the falsification harness.
- **Next action**: Read Rubinstein-style CE methods + Au & Beck subset simulation. Implement `bts.simulate.rare_event_mc` with cross-entropy IS for P(57). Compare CIs against current naive-MC CIs.

### 15. (NEW, 2026-05-01 evening) PA-independence and cross-game dependence modeling  [✅ shipped 2026-05-02 — falsification harness Task 13 (v1) + Issue #7 (v2)]

- **Status (2026-05-03 v2.5 evening — partial attribution)**: v2.5 SHIPPED via 6-cell nested factorial ablation. **Headline finding (Codex matrix-reviewed): in this six-cell nested ablation, Change A (fold-local parameter estimation) has no observable effect on the corrected_pipeline_p57 point estimate conditional on per-fold policy, while Changes B and C each independently produce most of the v1→v2 verdict shift.** A_effect_given_per_fold = 0.00pp; nested AB interaction = 0.00pp; both at metric resolution of 1/120 = 0.0083pp. Cells V010 = V001 = V101 = 0.0250 (any single one of B-alone, C-alone, or A+C produces same coarse scalar); V011 = V111 = 0.0333. **Caveat**: point estimates are coarse (1/120, 3/120, 4/120 successes); same-scalar across cells doesn't establish mechanism equivalence. Path-sum residual is 17% of total (decomposition is descriptive, not additive). **Defensible**: A is below detection in measured per-fold contrasts; B/C drive the shift. **Not defensible**: "A is methodology theater," "B/C substitutable," "deploy cell 010 as production policy" — those are interpretive jumps beyond what 6 coarse scalars establish. See `docs/sota_audit/2026-05-03-harness-v2.5-attribution.md`. v2.6 priorities: (1) block-bootstrap CI replacing 5-fold percentile, (2) cell 101 full-rep verification (cheap), (3) mechanism inspection (does V010=V001 reflect same fold patterns?), (4) distribution shift remediation as strategic priority.

- **Status (2026-05-02 v2 evening)**: v2 SHIPPED via Issue #7 — closes v1's two methodology gaps (later refuted by v2.5 attribution; see above). Per-rank-1-bin `rho_pair_per_bin` correction (5-element vector) and within-fold dependence-parameter estimation (rho_PA, tau, rho_pair refit per LOSO fold's 4 training seasons). New diagnostic 5×5 lower-triangular heatmap via `pair_residual_correlation_per_cell`. v2 verdict: `corrected_pipeline_p57 = 0.0333 [0.0000, 0.1167]` → `HEADLINE_REDUCED`. **However (per Codex round 1 review of memo)**: v2 point estimate is *still below* half-headline (0.0408); the gate-class transition from v1's `BROKEN` to v2's `REDUCED` is CI-driven, not point-estimate driven. Q4 sign reversed (v1 antagonism → v2 cooperative in 2/5 folds, near-zero in 3/5) — heterogeneous, not "artifact." See `docs/sota_audit/2026-05-02-harness-v2-comparison.md` and `data/validation/falsification_harness_v2_2026-05-03.json`.

- **Status (v1, 2026-05-02)**: SHIPPED as `bts.validate.dependence` — Pearson residuals + within-batter-game residual correlation via cluster bootstrap, logistic-normal random-intercept fit (cross-pair products + brentq inversion, NOT the textbook `tau^2 ≈ var-1` which Codex round 2 caught as backwards), cross-game pair-residual permutation test, and `build_corrected_transition_table` (two-knob mean correction). Real-data findings: rho_PA_within_game = 0.0012 [0.0009, 0.0015] (small but nonzero), rho_pair_cross_game = -0.0074 [-0.0607, 0.0476] (essentially zero). The PA-correction collapses corrected_pipeline_p57 by ~10× — this is the dominant signal that drove the v1 HEADLINE_BROKEN verdict, partly relaxed in v2 to HEADLINE_REDUCED.


- **Current**: Game-level aggregation `1 - prod(1 - p_PA)` assumes conditional PA independence given features. Double-down policy treats two games as independent. Neither assumption is tested.
- **SOTA target**: **Test PA independence empirically** — fit a within-game-residual covariance model, compare to independent-baseline log-likelihood. **Test cross-game dependence** — same weather slate, same modeling errors, correlated bullpen availability, cross-game park effects on the same day. Methods: copula approaches; conditional residual models; permutation tests for independence (Romano 1989). Decision implication: if dependence is non-trivial, the double-down policy under-weights correlation risk and CVaR-MDP becomes more important.
- **Speculative ΔP(57)**: -0.5 to +0.5pp depending on direction. May reduce the headline number (good — honest) and shift policy toward more conservative doubles.
- **Effort**: M
- **Prerequisites**: None.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review). Component of the falsification harness.
- **Next action**: Design a within-game PA residual covariance test on backtest data. Then a cross-game day-correlation test on rank-1/rank-2 picks. Quantify the dependence and feed it into MDP simulation as variance inflation.

### 16. (NEW, 2026-05-01 evening) Decision-aware learning

- **Current**: PA model is optimized for binary cross-entropy on hit/no-hit. The contest objective is a nonlinear tail event (P(57)). Training and decision metrics are decoupled.
- **SOTA target**: **Smart Predict-then-Optimize** (Elmachtoub & Grigas 2022 "Smart Predict, then Optimize"); **end-to-end loss surrogates** that target eventual MDP value; **decision-focused learning** (Wilder et al. 2019); **reweighting by downstream policy sensitivity** (train with weights proportional to how much each PA's prediction affects MDP decisions).
- **Speculative ΔP(57)**: +0.2-1.0pp speculative; high uncertainty. **The decoupling between PA-Brier and downstream-MDP-value may explain part of why feature-engineering returns are diminishing** (today's morning verdicts).
- **Effort**: M (reweighting) to L (full SPO surrogate).
- **Prerequisites**: #12 probabilistic-scoring-suite + #13 OPE for honest measurement.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review).
- **Next action**: Read Elmachtoub & Grigas 2022 SPO. Try the lightweight version first: weight training rows by `|MDP-value-gradient w.r.t. p_PA|` from a simulator pass. Compare downstream P(57) to baseline.

### 17. (NEW, 2026-05-01 evening) Model-class challenge

- **Current**: LightGBM-only, default hyperparameters, 12-model blend with rotating Statcast feature.
- **SOTA target**: **CatBoost ordered boosting** (Prokhorenkova et al. 2018) — handles target leakage in categorical/target-encoded features which BTS has many of (e.g., bpm). **NGBoost** (Duan et al. 2020) — natively probabilistic outputs (predictive distribution, not point), feeds into #14 directly. **Explainable Boosting Machines** (InterpretML, Nori et al. 2019) — interpretability+accuracy without the SHAP layer. **Monotone-constrained XGBoost/LightGBM** where baseball monotonicities are defensible (e.g., higher bpm → higher P(hit) all else equal).
- **Speculative ΔP(57)**: -0.2 to +0.5pp. Mostly an honest-comparison move — if LightGBM is genuinely best, Codex's "too comfortable" critique gets formally rebutted.
- **Effort**: M
- **Prerequisites**: #5 nested CV + #12 proper scoring rules for honest comparison.
- **Status**: unstarted; surfaced 2026-05-01 evening (Codex review).
- **Next action**: After #5 + #12 are in place, run a single-pass model-class bake-off: LightGBM (current) vs CatBoost ordered vs NGBoost vs EBM under proper-scoring evaluation. Time-bounded to one week; if no model beats LightGBM under proper scores, stop.

---

## Suggested execution order (revised 2026-05-01 evening, post-Codex review)

The original execution order put TreeSHAP first as a quick win. Codex's review re-prioritized aggressively: feature-engineering returns are diminishing (today's morning verdicts confirm), and the 16× gap between our 8.17% pooled P(57) and published SOTA ~0.5% means the audit's first job is to **defend the headline number**, not extend it.

Revised order:

1. **Falsification harness for the 8.17% claim** = #13 OPE + #14 rare-event MC + #15 dependence modeling, designed and built together. Goal: try to break the 8.17% number with honest decision-level evaluation under correlated rare-event variance. **First concrete area to execute.**
2. **#12 Probabilistic forecast evaluation suite** — replace P@1-centric evaluation. Foundation for everything else.
3. **#11 Binary-y validation methodology** — unblocks parked conformal v1.
4. **#5 Nested rolling-origin CV + lockbox** — methodology foundation; should happen before any further model-class or feature audits.
5. **#1 MDP robustness (distributional DP / robust VI)** — once OPE infra is in place.
6. **#16 Decision-aware learning** — try lightweight SPO/reweighting; depends on #12 + #13.
7. **#10 Predictive stacking** — variance reduction with proper-score weighting.
8. **#4 e-values / e-processes for sequential audits** — audit-compute reduction.
9. **#17 Model-class challenge** — depends on #5 + #12.
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
