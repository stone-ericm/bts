# Codex Adversarial Review — BTS SOTA Audit Tracker

**Date**: 2026-05-01 evening (~22:30 ET)
**Model**: GPT-5.5 (high reasoning) via OpenAI Codex CLI
**Trigger**: Eric authorized adversarial review of the SOTA audit tracker via the consulting-codex skill, with explicit instruction to use GPT-5.5 where it adds value regardless of cost.
**Cost**: ~$1-2 of high-reasoning tokens
**Outcome**: Substantive disagreement; led to citation fix + 6 new audit areas + execution-order rewrite. See `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md` for absorbed changes.

---

## Codex output (verbatim)

**1. Coverage Gaps**

The biggest missing area is **probabilistic forecast evaluation**. BTS is not mainly a classifier; it is a chained probabilistic decision system. P@1 is too blunt. You need log loss, Brier, Brier decomposition into reliability/resolution/uncertainty, top-decile calibration, sharpness vs reliability, and decision-bucket calibration. Gneiting & Raftery's proper scoring-rule framework and Murphy's Brier decomposition should be first-class audit areas, not calibration footnotes. See [Gneiting & Raftery 2007](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised) and [Murphy/Brier decomposition](https://search.r-project.org/CRAN/refmans/SpecsVerification/html/BrierDecomp.html).

Second: **offline policy evaluation**, not just MDP design. The MDP is fully offline and trained on historical predictions. Before CVaR-MDP, you need OPE-style validation: frozen historical information sets, doubly robust / model-based policy evaluation, policy regret against baselines, and uncertainty intervals around policy value. See Jiang & Li's [doubly robust OPE for RL](https://proceedings.mlr.press/v48/jiang16.html).

Third: **rare-event estimation**. P(57) is an extreme survival probability. A claimed 8.17% versus published ~0.5% is not "promising"; it is a red-alert target for leakage, dependence error, or selection bias. Add rare-event Monte Carlo with variance reduction: cross-entropy importance sampling, subset simulation, and sensitivity to correlated game outcomes. See rare-event cross-entropy / importance-sampling work such as [Rubinstein-style CE methods](https://www.sciencedirect.com/science/article/abs/pii/B9780444538598000023) and [subset simulation](https://authors.library.caltech.edu/records/m7b6w-kbc81).

Fourth: **dependence modeling**. The aggregation assumes conditional PA independence inside a game. Double-pick policy also depends on cross-game correlation: same weather slate, same run environment, same modeling errors, late lineup info, postponements, bullpen availability. This is probably more important than TreeSHAP.

Fifth: **decision-aware learning**. The PA model is optimized for prediction, but the contest objective is a nonlinear tail event. Add predict-then-optimize / SPO-style training or reweighting by downstream policy sensitivity. See [Elmachtoub & Grigas, Smart Predict-then-Optimize](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922).

Sixth: **model-class challenge set**. LightGBM-only is too comfortable. Missing: CatBoost ordered boosting for leakage-prone categorical/target-stat features, NGBoost for probabilistic outputs, EBMs as interpretable accuracy baselines, monotone-constrained XGBoost/LightGBM where baseball monotonicities are defensible. See [CatBoost ordered boosting](https://papers.neurips.cc/paper/7898-catboost-), [NGBoost](https://proceedings.mlr.press/v119/duan20a), and [EBM/InterpretML](https://interpret.ml/docs/ebm.html).

**2. Wrong SOTA Targets**

#1 CVaR / DR-MDP: directionally right, but too old and too abstract. Since the state/action space is small, do not jump to deep offline RL. Use exact distributional dynamic programming, robust value iteration over calibrated probability intervals, and stress-tested policy regret. Distributional RL methods like C51/QR-DQN/IQN are relevant conceptually, but likely overkill unless you abandon the tabular MDP. See [Bellemare et al. distributional RL](https://proceedings.mlr.press/v70/bellemare17a.html) and [IQN](https://proceedings.mlr.press/v80/dabney18a.html).

#2 Beta calibration: not SOTA as a blanket target. Beta is a useful binary calibrator, especially when isotonic overfits small n, but the real target should be cross-fitted Platt/beta/isotonic/spline/Venn-Abers comparison under proper scores and top-bin calibration. Venn-Abers is more relevant than beta if you want probability intervals. See [Beta calibration](https://proceedings.mlr.press/v54/kull17a.html) and [Venn-Abers](https://deepai.org/publication/venn-abers-predictors).

#3 TreeSHAP: useful, not SOTA as an audit priority. It explains the fitted model, not whether the feature causally or decision-wise improves BTS. Add permutation feature importance, conditional PFI, ALE plots, SHAP interactions, and model-class reliance. FastTreeSHAP is a speed upgrade, not a conceptual upgrade. See [ALE](https://tripartio.github.io/ale/index.html) and [model reliance](https://jmlr.org/beta/papers/v20/18-760.html).

#4 Always-valid p-values: good, but phrase the target as **e-values/e-processes/confidence sequences**, not just Howard-style p-values. The 2024 safe anytime-valid inference literature is the current framing. See [Ramdas & Grünwald 2024](https://nejsds.nestat.org/journal/NEJSDS/article/76/read) and [Vovk & Wang e-values](https://pure.royalholloway.ac.uk/en/publications/e-values-calibration-combination-and-applications/).

#5 CPCV: finance-derived, not automatically SOTA for MLB. Use nested rolling-origin CV with purging/embargo, plus reverse-CV diagnostics and a final untouched season/month lockbox. CPCV helps selection-bias honesty, but "nested purged blocked CV" is the more general target.

#6 ADWIN/DDM: old. Better targets: Bayesian online changepoint detection, MMD/KS-CUSUM on features and residuals, online calibration drift, and label-lag-aware loss monitors. But drift detection only matters if it triggers a policy change. See [Adams & MacKay BOCPD](https://lips.cs.princeton.edu/bibliography/adams2007changepoint/).

#7 BH/Storey: baseline, not SOTA. For sequential audits use online FDR/e-BH/e-values. For feature discovery, knockoffs are attractive but hard under temporal dependence. Add randomization/permutation tests around the whole audit pipeline. See [FDR with e-values](https://academic.oup.com/jrsssb/article/84/3/822/7056146).

#8 Online conformal: Lei 2018 is not the right target. Use ACI, strongly adaptive online conformal, NexCP/weighted conformal, and risk-controlling prediction sets. See [ACI](https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html), [Bhatnagar et al. 2023](https://proceedings.mlr.press/v202/bhatnagar23a.html), and [RCPS](https://people.eecs.berkeley.edu/~angelopoulos/blog/posts/rcps/).

#9 Transformer/GNN: the cited "Mehta & Rao 2023, Sabermetric Sequence Models" appears unverified. I would remove it until you have a real citation. Public baseball sequence work exists, but it is mostly pitch-type/pitch-sequence modeling, not directly BTS PA hit probability. A recent public example is [PitchPredict xLSTM](https://huggingface.co/baseball-analytica/pitchpredict-xlstm), but that is a model artifact, not peer-reviewed SOTA.

#10 BMA: outdated target if all models are misspecified. Prefer stacking of predictive distributions / super learner weighting by out-of-fold log score or downstream policy value. See [Yao, Vehtari, Simpson & Gelman on stacking](https://research.monash.edu/en/publications/using-stacking-to-average-bayesian-predictive-distributions-with-/). Deep ensembles matter if you move to neural sequence models; for LightGBM seeds, stacked/blended OOF predictions are enough.

#11 Per-bucket coverage: necessary but too weak. The target should be binary probabilistic validation: reliability diagrams with uncertainty bands, Brier decomposition, top-k/top-bin calibration, class-conditional/Mondrian diagnostics, Venn-Abers intervals, and decision-bucket calibration. Conformal "lower bound coverage" for binary y is the wrong mental model.

**3. Prioritization**

TreeSHAP first is wrong. It is comfortable, cheap, and low leverage. FDR second is also too retrospective. The blocker is not interpretability; it is whether the claimed P(57) and feature gains survive an honest decision-level evaluation.

I would reorder:

1. End-to-end validation harness: nested purged rolling backtest, frozen information timestamps, lockbox, rare-event confidence intervals, policy-value CIs.
2. Probabilistic scoring suite: log loss, Brier decomposition, top-bin calibration, game-level calibration, decision-bucket calibration.
3. Binary calibration / Venn-Abers / conformal-risk methodology.
4. MDP robustness: robust DP, probability perturbation, dependence stress tests, policy regret.
5. Ensemble/stacking and deterministic reproducibility.
6. Sequential testing/FDR/e-values for future audits.
7. Interpretability.
8. Feature/model-class expansion.

The eight failed feature candidates make this more obvious: feature engineering is no longer the main frontier. The frontier is **honesty of probability estimates and decision value under tail-event amplification**.

**4. Top 3 Moves**

1. **Build the decision-level validation and rare-event harness.**
   Recompute P(57) with nested rolling-origin splits, no post-selection leakage, frozen lineups/info, dependence stress, and rare-event Monte Carlo CIs. The 8.17% claim is the first thing I would try to break.

2. **Replace P@1-centric evaluation with probabilistic + decision-aware scoring.**
   Add log loss, Brier decomposition, calibration by probability bucket, calibration among actually selectable top candidates, and policy regret. Then tune/stack models against downstream MDP value, not generic PA accuracy.

3. **Implement robust calibrated policy optimization.**
   Cross-fitted calibrators or Venn-Abers intervals feed a robust/distributional DP. The policy should know when the top pick is fragile to calibration error, dependence error, or lineup uncertainty. This is more likely to improve real P(57) than another Statcast feature.

**5. Audit Framing**

"Audit-then-implement area-by-area" is the wrong organizing frame. It encourages method shopping.

Use a **claim-based adversarial benchmark** instead:

Claim A: PA probabilities are honest. Test with proper scores, calibration, Brier decomposition, drift, and provider determinism.

Claim B: game probabilities are honest. Test PA aggregation, within-game dependence, batting-order/lineup censoring, and game-level calibration.

Claim C: policy value is real. Test frozen historical simulations, policy regret, rare-event uncertainty, and sensitivity to probability perturbations.

Claim D: improvements survive selection. Test nested CV, randomization/permutation, e-values/online FDR, and lockbox replay.

Claim E: production picks match research picks. Test deterministic builds, timestamped data lineage, cloud/provider reproducibility, and exact replay.

That frame is harsher and better aligned with "best anyone could possibly make." The current tracker reads like a list of respectable ML techniques. BTS needs a falsification harness for an extreme sequential decision claim.

---

## Verification notes (Claude, post-review)

- **Mehta & Rao 2023 citation**: confirmed fictional via independent web search 2026-05-01 evening. No matches for "Mehta Rao Sabermetric Sequence Models 2023" in any indexed source. The form is a plausible-looking academic citation but the paper does not exist. Replaced in the tracker with verified real references.
- **Real baseball-sequence work** that surfaced via the same search: Neural Sabermetrics with World Model (arxiv 2602.07030, Feb 2026); The Impacts of Increasingly Complex Matchup Models on Baseball Win Probability (arxiv 2511.17733, Nov 2025); Pitcher Performance Prediction via Temporal Fusion Transformer (ScienceDirect S1546221825005028, 2025); Singlearity (Baseball Prospectus); Kevin Garnett "Chasing $5.6M with ML" (Medium, Feb 2026).
- **Codex's pushback was substantive enough to warrant absorption.** Eric chose option B (keep tracker structure, add new areas + reorder execution) over option A (rewrite around 5 claims) — see updated tracker doc.
