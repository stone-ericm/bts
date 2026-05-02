# BTS State-of-the-Art Audit — Master Tracker

**Date created**: 2026-05-01
**Status**: Active; populated with 11 audit areas. First area not yet selected.
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

## Prioritization heuristics

Order areas by **expected_P(57)_impact / weeks_of_work**, with hard prerequisites respected. Highest expected EV starts first.

Per Eric's stated lens (2026-05-01 brainstorm): "the best anyone could possibly make." Don't compromise rigor for speed; do compromise scope when SOTA is ill-defined.

---

## Audit area inventory (11 items)

### 1. MDP decision layer — CVaR / DR-MDP

- **Current**: Vanilla value iteration on point estimates of P(hit). Single policy table indexed by (streak, days_remaining, saver, quality_bin). Last solved 2026-04-15.
- **SOTA target**: **CVaR-MDP** (Chow & Ghavamzadeh 2014) — explicit tail-risk objective. **Distributionally Robust MDP** (Iyengar 2005, Wiesemann et al. 2013) — handles parameter uncertainty.
- **Speculative ΔP(57)**: +1.0–1.5pp (highest single-area impact; tail-risk control directly addresses the streak-survival concern)
- **Effort**: L
- **Prerequisites**: Fresh post-bpm 24-seed pooled backtest (the same blocker that gates plain MDP re-solve). ~$50 cloud cycle.
- **Status**: unstarted
- **Next action**: Design brainstorm. Key decisions: (a) full CVaR vs hybrid, (b) at which streak threshold the risk-aversion kicks in, (c) how to validate against current value-iteration policy.

### 2. Calibration — Beta calibration + binary-y conformal

- **Current**: Isotonic regression (`src/bts/model/calibrate.py`, default OFF behind `BTS_USE_CALIBRATION`). Validated as DROP at n=61 on 2026-05-01.
- **SOTA target**: **Beta calibration** (Kull, Filho, Flach 2017) — empirically superior to isotonic for binary classification (smoother, handles small calibration sets better). Plus: research on binary-y conformal proper (today's parked branch surfaced that per-row coverage metrics don't transfer cleanly from continuous regression).
- **Speculative ΔP(57)**: +0.3–0.5pp (helps MDP threshold decisions when activated)
- **Effort**: S (Beta calibration alone) + M (binary-y validation methodology)
- **Prerequisites**: n>=200 resolved picks (currently 61). Estimated late June 2026.
- **Status**: unstarted; deferred until n threshold reached.
- **Next action**: When n reaches 200, brainstorm Beta-vs-isotonic comparison. Separately: research per-bucket coverage tests for binary-y conformal validation (this unblocks #8 below).

### 3. Feature attribution / interpretability — TreeSHAP

- **Current**: One-feature-at-a-time sensitivity analysis (used today on Beck investigation). Misses interaction effects.
- **SOTA target**: **TreeSHAP** (Lundberg & Lee 2017) — exact Shapley values for tree models, captures interactions. Available in `shap` library; trivial integration with LightGBM.
- **Speculative ΔP(57)**: 0.0pp (interpretability tool, not a P(57) lever; high value for diagnosis when uncomfortable picks land)
- **Effort**: S (drop-in)
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Add `shap` to deps, write `src/bts/model/attribution.py` wrapper. Retro Beck pick using TreeSHAP and compare to today's sensitivity analysis. ~1-2 hours of work.

### 4. Audit experiment design — always-valid sequential p-values

- **Current**: Fixed-n with t-stat threshold + dual-stratum dual-split protocol. Each Phase 2 audit runs to completion regardless of mid-stream signal.
- **SOTA target**: **Always-valid sequential p-values** (Howard, Ramdas, McAuliffe, Sekhon 2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences"). Allows early-stopping audits when signal is unambiguous, reducing compute by 30-50%.
- **Speculative ΔP(57)**: 0.0pp directly; ~30-50% audit-cycle compute reduction (compounds: more audits per dollar).
- **Effort**: M (research + integrate into `bts.experiment` framework)
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Read Howard et al. 2021. Design how to integrate sequential testing into `audit_driver.py` — specifically: per-experiment early-stop criterion + protocol for downstream pooling.

### 5. Validation methodology — Combinatorial Purged CV

- **Current**: Walk-forward CV (single fold-per-day) + multi-seed pooling for audits.
- **SOTA target**: **Combinatorial Purged CV** (Lopez de Prado 2018, "Advances in Financial Machine Learning") — handles temporal structure properly, embargoes around test periods, generates many resampling paths for robust aggregation. **Deflated Sharpe Ratio** for selection-bias correction across audit batches.
- **Speculative ΔP(57)**: 0.0pp directly; better honesty about which "shipped" features are real signal vs selection bias.
- **Effort**: M
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Read Lopez de Prado 2018 chapters 7 (CV) and 14 (DSR). Design retrofit into `bts.experiment.runner` — specifically how CPCV interacts with the dual-stratum dual-split protocol.

### 6. Distribution shift handling — drift detection

- **Current**: Walk-forward retraining only (daily blend cycle). No explicit drift detection.
- **SOTA target**: **ADWIN** (Bifet & Gavaldà 2007) for adaptive window sizing; **DDM/EDDM** (Gama et al. 2004) for explicit drift signaling. Combined with importance-weighted training to handle covariate shift directly in the model.
- **Speculative ΔP(57)**: +0.2–0.5pp (helps adaptation to mid-season regime shifts that walk-forward alone might lag on)
- **Effort**: M
- **Prerequisites**: Pull request to add a drift-monitoring health check (similar pattern to `realized_calibration` and `pitcher_sparsity`).
- **Status**: unstarted.
- **Next action**: Read ADWIN + EDDM papers. Design where in the predict pipeline drift signal is computed (per-batter rolling residuals?). Brainstorm health-check thresholds.

### 7. Multiple testing across audits — FDR control

- **Current**: Nothing applied. Phase 1's "8 unshipped KEEPs all DROPped against bpm-baseline" is partly explainable as multiple-testing inflation that wasn't controlled for.
- **SOTA target**: **Benjamini-Hochberg FDR control** across the dozens of feature audits. **Storey q-values** for FDR-adjusted p-values. Apply both prospectively (new audits) and retrospectively (re-evaluate Phase 1 verdicts).
- **Speculative ΔP(57)**: 0.0pp directly; honest interpretation of which features ARE legitimate KEEPs vs noise. May trigger re-investigation of historically-shipped features.
- **Effort**: S (BH is one function call given an array of p-values)
- **Prerequisites**: None.
- **Status**: unstarted.
- **Next action**: Run BH retrospectively on the audit verdicts collected in `experiments/results/`. Surface findings: how many shipped features survive FDR correction at q=0.05?

### 8. Streaming calibration alert — online conformal

- **Current**: `realized_calibration` health check uses fixed-window thresholds; just patched today (since-deploy filter).
- **SOTA target**: **Anytime-valid sequential testing** (Howard et al. 2021 — same framework as #4) for streaming calibration. **Online conformal updates** (Lei et al. 2018) so calibrator improves with each new resolved pick without re-running full K-fold validation.
- **Speculative ΔP(57)**: 0.0pp; early detection of calibration drift.
- **Effort**: M
- **Prerequisites**: Conformal v1 working (currently parked at branch `feature/conformal-lower-bounds`, gate failed). Need to fix binary-y validation methodology (#2 second sub-area) first.
- **Status**: BLOCKED on conformal v1 unblock.
- **Next action**: When conformal v1 unblocks, read Howard et al. + Lei et al. and design online update path.

### 9. Feature engineering — sequence/transformer/GNN models for batter-pitcher

- **Current**: Tabular features (16 in FEATURE_COLS) + Bayesian shrinkage on bpm.
- **SOTA target**: **Transformer-based PA-sequence models** (recent baseball analytics literature: see Mehta & Rao 2023, "Sabermetric Sequence Models"). **Graph Neural Networks** on player-pitcher-context graphs (e.g., DGL-PyTorch).
- **Speculative ΔP(57)**: +0.5–1.5pp speculative; high uncertainty.
- **Effort**: XL
- **Prerequisites**: GPU compute (Mac MPS or cloud); training data pipeline that surfaces PA sequences instead of aggregated features.
- **Status**: unstarted.
- **Next action**: Survey 2024-2026 baseball-analytics literature for transformer/sequence-model results. If credible improvements documented, scope a brainstorm for a Phase-1-style audit experiment that adds a sequence-model output as a feature in the existing 12-blend.

### 10. Distribution-aware ensemble — Bayesian model averaging

- **Current**: Single-seed production model (BTS_LGBM_RANDOM_STATE=42). Multi-seed pooling REJECTED 2026-04-29 for the wrong reason (overconfidence framing was iteration noise).
- **SOTA target**: **Bayesian model averaging** with proper posterior over hyperparameters. **Conformal ensemble** with calibrated weights across multiple seed/blend variants.
- **Speculative ΔP(57)**: +0.2–0.5pp variance reduction; possibly more if combined with #1 (CVaR-MDP) which can directly use ensemble variance.
- **Effort**: M
- **Prerequisites**: Compute budget for multi-seed daily training (~10x current cost; possibly cheaper via subsampled-bootstrap variants).
- **Status**: unstarted; rejection from 2026-04-29 should be re-evaluated.
- **Next action**: Re-read the rejection memo (`project_bts_2026_04_29_pooled_prediction_rejected.md`). Brainstorm whether the rejection logic still applies given today's reframing of distribution shift.

### 11. (NEW) Validation methodology for binary classification calibrators

- **Current**: `scripts/validate_conformal.py` uses per-row coverage `(actual_hit ≥ bound).mean()`. For binary y, this collapses to empirical hit rate regardless of bound — not what we want.
- **SOTA target**: **Per-bucket coverage tests** that aggregate (predicted_p, actual) pairs by bucket and ask "does the bucket's realized rate exceed the bucket's lower bound at the claimed coverage?". Plus class-conditional Mondrian conformal techniques for proper binary-y bounds (Sesia & Romano 2021 "Adaptive Conformal Prediction Intervals"). Plus Venn-Abers extensions for multi-confidence-level outputs.
- **Speculative ΔP(57)**: 0.0pp directly; unblocks the conformal v1 ship which then enables #8 + parts of #1.
- **Effort**: M
- **Prerequisites**: None (the parked conformal v1 branch has the calibrator infrastructure ready).
- **Status**: surfaced 2026-05-01 evening when validation gate fired 0/6 SHIP. unstarted.
- **Next action**: Brainstorm session on binary-y validation methodology specifically. Read Sesia & Romano 2021 + Vovk's class-conditional conformal. Redesign `evaluate_fold` in validate_conformal.py to use per-bucket coverage. Re-run gate; if non-empty SHIP set, unblock conformal v1 deploy.

---

## Suggested execution order

Ordered by EV-per-week:

1. **#3 TreeSHAP** — Quick win. Drop-in replacement; ~2 hours; foundation for all future "why did it pick X" investigations.
2. **#7 FDR control** — Quick win + truth-up. Re-evaluating Phase 1 verdicts under FDR control may shift our entire interpretation of which features are "real."
3. **#11 Binary-y validation methodology** — Unblocks the parked conformal v1.
4. **#1 CVaR-MDP / DR-MDP** — Highest direct P(57) impact. Substantial effort but the payoff lines up with the goal.
5. **#5 Combinatorial Purged CV** — Methodology foundation. Should happen before more audits.
6. **#4 Always-valid sequential testing** — Audit-compute reduction. Should happen before more audits.
7. **#10 Bayesian ensemble** — Reframe of rejected idea given today's distribution-shift findings.
8. **#6 Drift detection** — Production monitoring upgrade.
9. **#2 Beta calibration** — Deferred until n>=200 resolved picks (~late June 2026).
10. **#8 Online conformal** — Blocked on #11.
11. **#9 Transformer/GNN features** — Major research; lowest priority given uncertainty.

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

## Open questions for next session

- Which area to start with (suggested: #3 TreeSHAP for quick foundation, then #7 FDR for retrospective truth-up)?
- Audit cadence — one area per session, or grouped?
- Where to put per-area brainstorm outputs — separate spec docs in `docs/superpowers/specs/` (one per area), or extend this tracker as we go?
