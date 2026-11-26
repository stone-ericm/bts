# Frontier Experiment Framework Design

## Context

The BTS model (12-model LightGBM blend, 15 baseline features, MDP-optimal strategy) achieves P@1=86.2% and P(57)=8.91%. Extensive research into frontier math/statistics identified 23 potential improvements across feature engineering, model architecture, blend strategy, and decision calibration.

The exponential sensitivity law (Cramér's theorem) means P(57) ∝ p^57 — each +1pp in P@1 multiplies P(57) by ~1.77x. Even marginal improvements compound dramatically across 57 games, justifying thorough backtesting of every candidate.

## Goal

Build a declarative experiment framework that systematically backtests all 23 candidates through a 3-phase pipeline: diagnostics → independent screening → forward stepwise selection. Produce a final combined model that maximizes MDP P(57).

## Three-Phase Pipeline

### Phase 0 — Diagnostics → Review → Adapt

Six diagnostic analyses that produce reports, not scorecards. Results are reviewed before Phase 1 begins. Diagnostics may add, modify, or remove Phase 1 experiments.

**Diagnostics:**

1. **Stability selection** — Run LightGBM on 100 bootstrap samples per season. Compute per-feature stability scores. Features stable in ALL seasons are robust; features unstable across seasons are candidates for replacement or removal.

2. **Wasserstein drift audit** — Compute per-feature 1D Wasserstein distances between each season pair using `scipy.stats.wasserstein_distance`. Cross-reference with SHAP importance. High-importance + high-drift features are primary instability sources.

3. **Streak-length dependence** — Stratify historical backtest P@1 by current streak length at prediction time. If P@1 degrades at high streak lengths (after controlling for feature quality), streak length should become a direct model feature.

4. **AFT Weibull shape parameter** — Fit accelerated failure time model on historical streak termination data. Shape > 1 = streaks get harder over time (pressure/fatigue). Shape = 1 = geometric (current assumption holds). Shape < 1 = hot-hand self-stabilization.

5. **AutoGluon ceiling estimate** — Run AutoGluon on one holdout season (2025) as a one-time diagnostic. Measures how much headroom exists above the handcrafted blend. Not for production.

6. **Knockoff filter (KOBT)** — Model-X knockoffs with LightGBM+SHAP feature statistics. FDR-controlled identification of which features are genuinely predictive vs noise. Uses `knockpy` with second-order Gaussian knockoffs.

7. **ADWIN changepoint on historical Brier score** — Apply ADWIN drift detector to the model's rolling Brier score across historical seasons. Identifies within-season calibration drift points (trade deadline, roster expansion, September). Uses `river` library.

**Review gate:** After diagnostics complete, results are reviewed together. Phase 1 experiment list is adapted based on findings. Examples:
- Stability selection shows a feature is never stable → add "drop feature" experiment
- Wasserstein audit reveals a specific feature drives instability → targeted fix experiment
- Streak-length dependence detected → add streak_length as feature experiment
- AutoGluon ceiling is close to current blend → temper expectations

### Phase 1 — Independent Screening

Each experiment modifies one thing via hooks, runs a full blend walk-forward on test seasons 2024+2025, and is evaluated against the baseline scorecard.

**Pass criterion (either condition):**
1. P@1 improves on BOTH 2024 and 2025 (the both-seasons test)
2. P@1 is neutral on both seasons (no more than -0.3pp drop on either) AND MDP P(57) improves (the bullpen precedent)

**23 Experiments by category:**

#### Feature experiments (8):

1. **EB shrinkage on rolling windows** — Replace raw rolling averages (batter_hr_7g/30g/60g/120g) with beta-binomial empirical Bayes shrunken estimates. Estimate (α, β) from population of all batters with ≥20 PA. Callups automatically get population mean.

2. **Entropy → KL divergence** — Replace `pitcher_entropy_30g` with `d_FR(pitcher_mix_30g, batter_comfort_zone_60g)` — the Fisher-Rao geodesic distance between the pitcher's pitch-type distribution and the batter's historical pitch-type distribution faced. Formula: `2 * arccos(Σ√(pᵢqᵢ))`. O(K) compute per matchup.

3. **Batting Heat Index (Q)** — New feature: weighted combination of consecutive-game hit runs and batting average during those runs. Weights streakiness more heavily than simple rolling average. Based on Green & Zwiebel's 25-AB hot-hand window.

4. **Groundball-rate platoon** — New feature: interaction of batter/pitcher handedness with groundball rate. Same-handed matchups produce more ground balls via sinkers, suppressing hits beyond what standard platoon split captures.

5. **Hit-type-specific park factors** — Replace single `park_factor` with separate factors for singles, doubles, triples. More granular than current expanding BPF.

6. **Venn-ABERS interval width** — New feature: fit two isotonic regressions (imputing hit=0 and hit=1) on calibration data. Width [p₀, p₁] measures epistemic uncertainty. Hypothesis: wide intervals correlate with miss days.

7. **Quantile q10 as skip signal** — Train LightGBM quantile model at α=0.10 alongside blend. Use q10 estimate as additional skip signal in strategy (if q10 < threshold, skip regardless of median).

8. **Streak-length feature** — Add current streak length as a direct model feature. Only if Phase 0 streak-length dependence diagnostic reveals degradation at high streaks.

#### Model experiments (4):

9. **LambdaRank blend member** — Train one blend member with `objective="lambdarank"`, `lambdarank_truncation_level=1`. Game-day level data (one row per batter per day, binary hit label, daily groups). Only blend member optimizing top-1 ranking directly.

10. **CatBoost with `has_time=True`** — Train CatBoost model with ordered boosting (temporal gradient leakage prevention). Drop-in test as blend member alongside existing LightGBM models.

11. **XE-NDCG blend member** — Train one blend member with `objective="rank_xendcg"`. Faster LambdaRank alternative, convex upper bound on NDCG. Same game-day data structure.

12. **V-REx season reweighting** — Iteratively upweight worst-performing season during LightGBM training. Penalizes cross-season loss variance via `sample_weight` loop. No new model architecture.

#### Blend experiments (3):

13. **FWLS contextual stacking** — Replace equal 1/12 blend weights with Feature-Weighted Linear Stacking. Meta-learner receives 12 model predictions + context features (handedness, park, pitcher type). Ridge-penalized linear regression. Walk-forward meta-training.

14. **Fixed-Share Hedge** — Replace equal blend weights with online-adaptive weights updated daily via Hedge algorithm with Fixed-Share mixing (α=0.05). Tracks best model across the season.

15. **Copula-adjusted double selection** — Replace `P(both) = P(A) × P(B)` with Gaussian copula `P(both) = Φ₂(Φ⁻¹(pA), Φ⁻¹(pB); ρ)` where ρ estimated from historical daily random effect. Only affects double-down scoring.

#### Strategy experiments (2):

16. **Decision calibration at skip threshold** — Apply isotonic regression calibration to game-level blend output, targeting the MDP skip-threshold region. Addresses underconfidence at top (model predicts 82% when actual is 90%). Calibration map trained on holdout season.

17. **Quantile-gated skip** — MDP skip/play decision uses q10 lower bound instead of point estimate. More conservative on uncertain days. Only triggers when q10 diverges significantly from median.

#### Diagnostic-informed experiments (added after Phase 0):

18-23. **TBD** — Reserved slots for experiments added or modified based on Phase 0 diagnostic results. Examples: drop unstable features, targeted EB shrinkage on high-drift features, structural break handling for shift-ban era.

### Phase 2 — Forward Stepwise Selection + Backward Elimination

1. Sort Phase 1 winners by individual MDP P(57) improvement (descending)
2. Start with baseline model
3. Add best winner. Run full walk-forward. Keep if combined MDP P(57) improves.
4. Add next winner. Run full walk-forward. Keep if combined MDP P(57) improves.
5. Repeat until all winners tested
6. Backward elimination: from final stack, remove each component one at a time. Drop any whose removal doesn't decrease MDP P(57).
7. Final combined model gets full scorecard + MDP solve

## Experiment Definition Model

```python
@dataclass
class ExperimentDef:
    name: str                          # unique identifier, e.g. "eb_shrinkage"
    phase: int                         # 0=diagnostic, 1=screening, 2=forward-select
    category: str                      # "feature", "model", "blend", "strategy", "diagnostic"
    description: str                   # human-readable summary
    dependencies: list[str] = field(default_factory=list)
    
    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add/replace/remove features. Return modified DataFrame."""
        return df
    
    def modify_blend_configs(self, configs: list[tuple]) -> list[tuple]:
        """Override to add/replace blend model configs."""
        return configs
    
    def modify_training_params(self, params: dict) -> dict:
        """Override to change LightGBM params or training loop."""
        return params
    
    def modify_strategy(self, profiles_df, quality_bins) -> tuple:
        """Override to change strategy/calibration/MDP inputs."""
        return profiles_df, quality_bins
    
    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        """Override for Phase 0 diagnostics. Return report dict."""
        raise NotImplementedError
    
    def feature_cols(self) -> list[str] | None:
        """Override to change FEATURE_COLS for this experiment. None = use default."""
        return None
```

Each experiment subclasses `ExperimentDef` and overrides only the hooks it needs. Most experiments override a single hook.

## Runner Architecture

### CLI Commands

```
bts experiment diagnostics              # Phase 0
bts experiment screen [--subset a,b,c]  # Phase 1 (optional subset for parallelism)
bts experiment select                   # Phase 2
bts experiment summary                  # Print results table across all phases
```

### Execution Flow

**Phase 0:**
1. Load PA data, compute baseline features once
2. Run each diagnostic's `run_diagnostic()` hook
3. Save reports to `experiments/results/phase0/<name>.json`
4. Print summary table

**Phase 1:**
1. Compute and cache baseline scorecard (or load if exists)
2. Group experiments: feature-touching (need fresh `compute_all_features`) vs non-feature (share baseline DataFrame)
3. For each experiment:
   a. Apply hooks (features, blend configs, training params)
   b. Run `blend_walk_forward()` for test seasons 2024, 2025
   c. Compute `compute_full_scorecard()`
   d. Compute `diff_scorecards(baseline, variant)`
   e. Determine pass/fail against acceptance criteria
   f. Save results to `experiments/results/phase1/<name>/`
4. Print summary table

**Phase 2:**
1. Load Phase 1 results, filter to passing experiments
2. Sort by MDP P(57) delta descending
3. Forward selection loop:
   a. Apply next winner's hooks on top of current combined model
   b. Run walk-forward + scorecard
   c. If combined P(57) improves: keep, else discard
   d. Log decision
4. Backward elimination:
   a. For each component in combined model, try removing it
   b. If removal doesn't decrease P(57): drop it
   c. Log decision
5. Save final combined scorecard + diff vs original baseline

### Parallelism

Phase 1 experiments are independent. The `--subset` flag allows splitting across machines:

```bash
# Mac
bts experiment screen --subset eb_shrinkage,kl_divergence,batting_heat_q,...

# Alienware (via SSH from Pi5 or direct)
bts experiment screen --subset lambdarank,catboost,xendcg,...
```

Results merge into the same `experiments/results/phase1/` directory. `bts experiment summary` reads all results regardless of where they were computed.

### Result Storage

```
experiments/
  registry.py                              # All ExperimentDef subclasses
  runner.py                                # Phase orchestration + CLI integration
  diagnostics/                             # Phase 0 implementations
  screening/                               # Phase 1 experiment implementations
  results/
    phase0/
      stability_selection.json
      wasserstein_drift.json
      streak_length_dependence.json
      aft_shape.json
      autogluon_ceiling.json
      knockoff_filter.json
    phase1/
      baseline_scorecard.json
      <experiment_name>/
        scorecard.json
        diff.json
        summary.txt                        # one-line: P@1 Δ, P(57) Δ, pass/fail
    phase2/
      forward_selection_log.json
      backward_elimination_log.json
      final_scorecard.json
      final_diff.json
```

## Acceptance Criteria

### Phase 1 (Screening)

Pass if EITHER:
- P@1 improves on both 2024 AND 2025
- P@1 drops no more than 0.3pp on either season AND MDP P(57) improves

### Phase 2 (Selection)

Each forward-selection step passes if combined MDP P(57) improves over the previous step. Each backward-elimination step confirms that removing a component decreases MDP P(57).

### Reporting

Summary table after each phase:

```
Phase 1 Results — N experiments vs baseline (P@1=86.2%, P(57)=8.91%)
──────────────────────────────────────────────────────────────────────
Experiment              Category   P@1 2024  P@1 2025  P(57) MDP  Pass
──────────────────────────────────────────────────────────────────────
eb_shrinkage            feature    +0.3%     +0.2%     +0.8%      ✓
kl_divergence           feature    +0.1%     +0.4%     +1.2%      ✓
lambdarank_blend        model      -0.1%     +0.5%     +0.6%      ✓
catboost_blend          model      +0.0%     -0.2%     -0.1%      ✗
...
──────────────────────────────────────────────────────────────────────
Winners: K/23 passed screening
```

## Dependencies

### Python packages (new):
- `knockpy` — knockoff filter
- `scipy` — Wasserstein distance, beta distribution (already a dep)
- `catboost` — CatBoost experiments (optional, model machines only)
- `river` — ADWIN changepoint detection (lightweight)
- `autogluon.tabular` — ceiling estimate diagnostic only

### Existing infrastructure used:
- `blend_walk_forward()` from `bts.simulate.backtest_blend`
- `compute_full_scorecard()` from `bts.validate.scorecard`
- `diff_scorecards()` from `bts.validate.scorecard`
- `compute_all_features()` from `bts.features.compute`
- `BLEND_CONFIGS` from `bts.model.predict`
- `solve_mdp()` from `bts.simulate.mdp`

## Out of Scope

- Production deployment of winning experiments (separate task after results are validated)
- Changes to the scheduler, dashboard, or orchestration
- Cloud VPS provisioning
- Real-time model monitoring (ADWIN in production)
