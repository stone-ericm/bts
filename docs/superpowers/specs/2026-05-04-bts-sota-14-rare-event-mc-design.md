# SOTA #14 — Rare-event Monte Carlo (CE-IS) — Phase 0 design memo

**Date**: 2026-05-04
**Branch**: `feature/sota-14-ce-is-design`
**Base**: main `397f729` (post-PR #13 OPE merge)
**Tracker entry**: §14 (full SOTA rare-event MC; one leg of the falsification harness for the 8.17% claim)
**Author**: Claude (read-only inspection + design)
**Reviewer**: Codex (pre-implementation; bus #119–#127 sequencing)

This memo is a Phase 0 design/preflight. **No implementation code is in this PR.**
Writing this memo is the entire deliverable; an implementation PR is gated on
Codex sign-off.

**Revision history**: this memo went through three review rounds with Codex
before this docs-only PR. v1 missed six structural issues caught in #123
(estimand/horizon misframing — calling a 102-day fold-holdout result
"season-level P57"; CE math description that didn't match the actual v1
code — memo described standard LR-weighted CE while v1 actually fits theta_0
from elite hit-count quantile; oracle overclaim — current tests use naive
MC, not `exact_p57`; diagnostics referring to a `proposal_event_rate` field
that `CEISResult` doesn't expose; an overclaim-prone `comparison_to_v_pi`
block; and inaccurate harness-driver wording). v2 fixed those. v3 (per
#125) corrected the oracle test parameterization (`exact_p57` is hard-coded
to absorbing state 57; new setup uses p=0.95 / season_length=70 /
streak_threshold=57), dropped a test that depended on private internals,
replaced a conceptually-backwards "CE improves weight diagnostics" test with
a black-box-safe "theta moves + strict-JSON-safe" test, and made the
"no public tune-only API" implementation contract explicit.

## TL;DR

- **P0/P1 estimand: fixed-window** P(max consecutive rank-1 hits ≥ streak_threshold) over the **ordered fold-holdout date sequence**, under independent Bernoulli at the holdout's rank-1 p_game_hit values. Horizon = `n_holdout_dates`, NOT season_length. The output **is NOT directly comparable to #13's season-horizon V_pi** under any framing — different estimands, different horizons.
- **First honest estimator**: existing `bts.simulate.rare_event_mc.estimate_p57_with_ceis` (v1; fits theta_0 only via elite hit-count quantile selection). #14 P0/P1 wraps it in a #5-manifest-aware "fit theta on fold-train profiles, evaluate IS on fold-holdout profiles" structure. theta is described as **proposal tuning**, not model fitting.
- **Oracle**: a new implementation test must compare theta=0 CE-IS to `exact_p57` on a simple constant-probability one-bin setup. For heterogeneous fixed sequences (the production case), `exact_p57` over `QualityBins` is only an approximate / different-distribution cross-check — NOT a strict oracle.
- **Diagnostics**: report only what `CEISResult` already exposes (`ess`, `max_weight_share`, `log_weight_variance`, `n_final`, `theta_final`). Anything else (e.g., proposal_event_rate) requires source-side work in `rare_event_mc.py`, which is **out of scope for P0/P1**.
- **No `comparison_to_v_pi` block** in P0/P1 output (per Codex #123). Cross-method synthesis belongs in a separate memo after #14 lands.

## Background: what already exists on main

After grepping post-#13 main:

| Artifact | Module | Status | Used where |
|---|---|---|---|
| `estimate_p57_with_ceis(profiles, strategy=None, n_rounds, n_per_round, n_final, theta, seed, streak_threshold)` | `bts.simulate.rare_event_mc` | implemented v1 | called from `scripts/run_falsification_harness.py` Task 13 with `strategy=None` (no strategy-aware wrapping) |
| `cross_entropy_tilt_step(paths, weights, elite_quantile=0.95)` | same module | implemented v1 | inner CE round; **`weights` arg is ignored** (v1 fits theta_0 from elite hit-count, not LR-weighted) |
| `CEISResult` (point, ci_lower, ci_upper, ess, max_weight_share, log_weight_variance, n_final, theta_final) | same module | implemented | result wrapper |
| `LatentFactorSimulator` | same module | implemented but **v1 bypasses it** | unbiasedness oracle test only |
| `exact_p57(strategy: Strategy, bins: QualityBins, season_length=180) -> float` | `bts.simulate.exact` | implemented | the (constrained) oracle — accepts `Strategy` only, builds transition matrix from `QualityBins` |
| `tests/simulate/test_rare_event_mc.py` | tests | `TestCEISUnbiasedness::test_unbiased_at_theta_zero_matches_naive_mc` validates theta=0 vs naive MC. **Does NOT compare to `exact_p57`.** | — |

**Verified facts (read-only inspection)**:
- The v1 CE inner loop selects "elite" paths whose hit-count is at or above the 95th percentile, then fits theta_0 = `logit(elite_rate) - logit(overall_rate)`. The `weights` argument to `cross_entropy_tilt_step` is **never used**. This is a simplified Rubinstein-Kroese-style elite-set fitting, NOT the standard LR-weighted MLE refit.
- The current harness driver in `scripts/run_falsification_harness.py:180` calls `estimate_p57_with_ceis(ceis_profiles, strategy=None, n_final=n_final, seed=seed)`. **No strategy-aware replayer wraps the estimator.** The "always-play, no-doubles" event indicator is what drives the realized `rare_event_ce_p57 = 0.0037 [0.0031, 0.0044]`.

## Codex's #120/#123 questions, answered (revised)

### 1. Estimand & horizon (Codex #123 main blocker)

**P0/P1 estimand**: P(max consecutive hits ≥ `streak_threshold`) over the ordered fold-holdout date sequence, under independent Bernoulli rank-1 hits with per-day probability = holdout's rank-1 `p_game_hit`. **Horizon = `n_holdout_dates`** (typically ~102 for a 5-fold split of 5-season backtest). This is **NOT** a season-level P57 estimate.

For `streak_threshold=57` on a 102-day holdout, the empirical event probability is mechanically near zero unless rank-1 hit rates are extremely high; running CE-IS lets us estimate that small probability with variance reduction over naive MC. The audit-relevant signal is the **fold-distribution of these fixed-window estimates**, NOT a single comparable-to-anything number.

**This estimate is not directly comparable to #13's V_pi**:
- #13 V_pi: model-based forward eval of an MDP-optimal `policy_table` against fold-holdout BIN DISTRIBUTION over season_length=180.
- #14 CE-IS estimate: fixed-window IS estimate over the actual ordered 102-day holdout sequence, always-play.

Different estimand, different horizon, different policy semantics. The output schema name (`rare_event_ce_is_v1`) and field names (`fixed_window_estimate`, `horizon=n_holdout_dates`) call this out explicitly. **Strategy-aware variant deferred to #14 P1.5+.**

### 2. Input shape (Codex #123 #2)

Per fold:
- `train_df` → `train_profiles: list[dict[str, float]]` with `p_game = rank=1's p_game_hit` per date.
- `holdout_df` → `holdout_profiles: list[dict[str, float]]` same construction.

**Do NOT switch to `1 - prod(1 - p_PA)`** — that requires PA-level data, which is out of scope per Codex #123.

### 3. CE-rounds-on-train discipline (Codex #123 #3)

**Fit theta on fold-train profiles, run final IS estimate on fold-holdout profiles.** This is honest audit discipline (no test-set leakage into CE proposal). **Phrase theta as proposal tuning, not model fitting** per Codex's correction — the goal is variance reduction in IS, not parameter inference about the world.

**Implementation-contract note (per Codex #125)**: there is no public `run_ce_rounds_on_train` function in `bts.simulate.rare_event_mc`. The wrapper has to call `estimate_p57_with_ceis` on train, read `theta_final`, discard the train point estimate, and then call `estimate_p57_with_ceis` again on holdout with `n_rounds=0, theta=train_theta`:

```python
# Per fold (P0/P1 contract — no source changes to rare_event_mc.py):
result_train = estimate_p57_with_ceis(
    train_profiles, strategy=None,
    n_rounds=n_rounds_train,
    n_per_round=n_per_round_train,
    n_final=n_final_train,        # incurred but DISCARDED
    seed=seed,
    streak_threshold=streak_threshold,
)
train_theta = result_train.theta_final
# train point estimate is NOT used; only theta_final is

result_holdout = estimate_p57_with_ceis(
    holdout_profiles, strategy=None,
    n_rounds=0,                    # use train_theta as-is
    n_final=n_final_holdout,
    theta=train_theta,
    seed=seed,
    streak_threshold=streak_threshold,
)
# fixed_window_estimate = result_holdout.point_estimate
```

The training call **incurs a final-estimation pass** because the current API has no tune-only function. To keep the train-side cost reasonable while still allowing CE rounds to converge, P0/P1 default `n_final_train=2000` (much smaller than `n_final_holdout=20000`). Surfacing a tune-only API is **out of P0/P1 scope** (deferred to P1.5+).

### 4. CE math (Codex #123 second blocker)

**Actual v1 algorithm** (per `cross_entropy_tilt_step` source code):

1. Sample `n_per_round` paths from `q_theta`: each day's outcome is independent Bernoulli with `sigmoid(logit(p_t^target) + theta_0)`.
2. Score each path by hit-count = `sum(y_t)`.
3. Define elite set as paths with score ≥ 95th percentile of hit-count distribution.
4. Refit `theta_0 = logit(elite_hit_rate) - logit(overall_hit_rate)`. The other 3 entries of theta stay at 0 (placeholders for v1.5+).

This is a simplified Rubinstein-Kroese-style elite-set fitting, NOT standard LR-weighted MLE refit. The `weights` argument to `cross_entropy_tilt_step` is **literally ignored** — that's a v1 simplification documented in code comments.

**P0/P1 stays consistent with v1's actual algorithm.** Any change to the CE math (e.g., to standard LR-weighted refit) would require a source-code edit to `cross_entropy_tilt_step`, which is **out of P0/P1 scope**. P1.5+ may revisit.

**Likelihood-ratio per path** (used in the IS estimate, not the elite refit):
```
LR = prod_t [p_t^target / q_theta(y_t=1)]^y_t * [(1 - p_t^target) / q_theta(y_t=0)]^(1 - y_t)
```
The IS-weighted estimate is `(1/N) * sum_paths LR_path * 1{event}`. This part of v1 is standard and stays unchanged in P0/P1.

### 5. Oracle (Codex #123 third blocker)

**`exact_p57` is NOT a current-test oracle for v1.** The existing `TestCEISUnbiasedness::test_unbiased_at_theta_zero_matches_naive_mc` compares CE-IS at theta=0 to **naive MC**, NOT to `exact_p57`.

**P0/P1 must add an implementation test** that compares theta=0 CE-IS to `exact_p57` on a simple constant-probability one-bin setup. **Important constraint (per Codex #125)**: `exact_p57` is hard-coded to absorbing state 57 — it does NOT accept an arbitrary streak_threshold. The oracle test must therefore use `streak_threshold=57` end-to-end:

- Build `Strategy_always_play = Strategy(skip_threshold=None, double_threshold=None, streak_saver=False, streak_config=None)`.
- Build `QualityBins` with one bin at constant `p_hit=0.95`.
- Compute `exact = exact_p57(Strategy_always_play, bins, season_length=70)` (Codex verified locally: ≈ 0.08866).
- Build a synthetic `profiles = [{"p_game": 0.95}] * 70`.
- Run `ceis = estimate_p57_with_ceis(profiles, strategy=None, n_rounds=0, theta=zeros(4), n_final=20000, seed=42, streak_threshold=57)`.
- Assert `abs(ceis.point_estimate - exact) < 3 * (ceis.ci_upper - ceis.ci_lower) / 4`. (Loose tolerance; n_final=20k or 30k is plenty for this scale.)

**For the production heterogeneous-sequence case, `exact_p57` is only an approximate cross-check**: the production rank-1 p_game_hit varies across dates, while `exact_p57` builds a transition matrix from a single `QualityBins` representation that aggregates dates. The production estimate IS comparable to the approximate exact value for a uniform-sequence approximation, but not strictly. This caveat must be in the schema or PR body.

### 6. Diagnostics (Codex #123 fourth blocker)

**Report only what `CEISResult` already exposes**:
- `ess` (effective sample size = `(sum w)^2 / sum(w^2)`)
- `max_weight_share` (max weight / total weight)
- `log_weight_variance` (variance of log-weights)
- `n_final` (echo)
- `theta_final` (the learned tilt vector)

**Removed from P0 schema**: `proposal_event_rate`. The v1 `CEISResult` doesn't expose this; computing it would require a source change to surface raw event indicators from the final IS batch, which is **out of P0/P1 scope**. Add to P1.5+ if useful.

**`verdict_flag`** computed from public diagnostics:
- `"OK"` if `ess > min_ess_threshold` AND `max_weight_share < max_weight_threshold`
- `"IS_DIAGNOSTIC_WARNING"` otherwise (defaults: `min_ess=1000`, `max_weight=0.1`)

The flag is diagnostic, not gating.

### 7. #5 manifest composition + output schema

```json
{
  "schema_version": "rare_event_ce_is_v1",
  "created_at": "...",
  "estimand": {
    "name": "p_max_streak_ge_threshold_over_holdout_window",
    "description": "P(max consecutive rank-1 hits >= streak_threshold over the ordered fold-holdout date sequence), independent Bernoulli baseline. NOT a season-level P57 estimate; horizon = n_holdout_dates per fold.",
    "horizon_basis": "n_holdout_dates",
    "streak_threshold": 57
  },
  "estimator": {
    "primary": "estimate_p57_with_ceis",
    "v1_simplifications": [
      "fits theta_0 only (constant logit shift); per-action/per-day tilt deferred to #14 P1.5+",
      "elite-set hit-count refit, NOT LR-weighted MLE",
      "always-play event indicator (strategy arg ignored); strategy-aware tilt deferred to P1.5+"
    ],
    "n_rounds_train": 8,
    "n_per_round_train": 5000,
    "n_final_holdout": 20000
  },
  "manifest_metadata": {...},
  "lockbox_held_out": true,
  "lockbox": {...},
  "n_folds": 5,
  "fold_results": [
    {
      "fold_idx": 0,
      "n_train_dates": 365,
      "n_holdout_dates": 102,
      "fixed_window_estimate": 0.0037,
      "ci_lower": 0.0029,
      "ci_upper": 0.0046,
      "theta_train": [0.42, 0.0, 0.0, 0.0],
      "diagnostics": {
        "ess": 8421.3,
        "max_weight_share": 0.0023,
        "log_weight_variance": 1.18,
        "n_final": 20000,
        "verdict_flag": "OK"
      }
    }
  ],
  "aggregate_deferred": true,
  "thresholds": {
    "min_ess": 1000,
    "max_weight_share": 0.1,
    "n_rounds_train": 8,
    "n_per_round_train": 5000,
    "n_final_holdout": 20000,
    "seed": 42,
    "streak_threshold": 57
  }
}
```

**No `comparison_to_v_pi` block** in P0/P1 output (per Codex #123 fifth blocker). Cross-method synthesis goes in a separate memo after #14 lands.

### 8. CLI

`bts validate rare-event-ce-is --manifest <path> --output <path>` with optional flags `--n-rounds-train`, `--n-per-round-train`, `--n-final-holdout`, `--seed`, `--streak-threshold`, `--min-ess`, `--max-weight-share`.

JSON written with `allow_nan=False` (artifact contract fails closed; mirrors #13/#11 pattern).

### Tests

**Module/wrapper tests** in `tests/validate/test_rare_event_mc_eval.py` (NEW; doesn't conflict with `tests/simulate/test_rare_event_mc.py`):

- `test_theta_zero_matches_exact_p57_on_constant_probability_one_bin` — the new oracle test required by Codex #123. Uses `Strategy_always_play` + one-bin `QualityBins` (p=0.95) + constant-p profiles + season_length=70 + streak_threshold=57 (must match `exact_p57`'s hardcoded absorbing state). Runs CE-IS at theta=0 and asserts agreement with `exact_p57` (≈ 0.08866) within loose MC tolerance.
- `test_theta_zero_matches_naive_mc` — keep the existing oracle invariant lifted to the eval surface.
- `test_deterministic_seed_reproducibility` — same `seed` + same hyperparams + same input → same `fixed_window_estimate` exactly.
- `test_strict_json_round_trip` — `json.dumps(result, allow_nan=False)` succeeds on real-data smoke output.
- `test_ce_tuning_changes_theta_and_returns_strict_json_safe_diagnostics` — black-box-safe (per Codex #125): on an easy synthetic where CE has signal, run CE rounds → assert (a) `theta_final[0] != 0` (CE moved theta away from zero), (b) all diagnostic fields are finite (`json.dumps(..., allow_nan=False)` succeeds), (c) `max_weight_share < max_weight_share_threshold`. **Does NOT claim to prove CE improves event frequency** — that requires `proposal_event_rate` exposure (deferred to P1.5+).

**Removed from P0** (per Codex #125):
- `test_likelihood_ratio_sums_to_one_under_target` — the LR helper is nested inside `estimate_p57_with_ceis` (not public). Adding the test would require a source-side helper extraction, which v2 explicitly said is out of P0/P1 scope.
- `test_synthetic_ce_improves_over_theta_zero_without_weight_collapse` (the previous wording) — at theta=0 the public weight diagnostics are mathematically perfect (ESS=n, max_weight_share=1/n, log_weight_variance=0), so a "theta-tuned beats theta=0 on weight diagnostics" assertion is conceptually backwards. The replacement test above tests what's actually verifiable from the public API.

**Manifest integration tests**:
- `test_evaluate_ceis_on_manifest_basic_shape`
- `test_lockbox_held_out_true`
- `test_aggregate_deferred_true`
- `test_n_folds_matches_manifest`
- `test_each_fold_has_required_diagnostics_block`
- `test_fixed_window_estimate_ci_bounds_consistent` (lower ≤ point ≤ upper)

**CLI tests** (CliRunner):
- `test_cli_runs_basic_returns_v1_schema`
- `test_cli_n_rounds_zero_collapses_to_naive_mc_at_theta_zero`
- `test_cli_unknown_arg_raises_usage_error`

## Proposed P0/P1 boundaries

**In scope**:
- New `bts.validate.rare_event_mc_eval` module with `evaluate_ceis_on_manifest(...)`.
- Per-fold "fit theta on fold-train profiles, evaluate IS on fold-holdout" structure.
- Output schema `rare_event_ce_is_v1` with horizon=n_holdout_dates, fixed-window estimand, public-only diagnostics.
- New oracle test against `exact_p57` for constant-probability one-bin case.
- CLI `bts validate rare-event-ce-is` with strict-JSON guard.
- No source-side changes to `rare_event_mc.py` (treat as a black-box library).

**Out of scope (deferred — explicit follow-ups, not "forgotten")**:
- **#14 P1.5: per-action / per-day tilt** (full theta vector active beyond theta_0; requires source change to `cross_entropy_tilt_step`)
- **#14 P1.5: strategy-aware CE-IS** (consumes `policy_table` from #13 or `Strategy` adapter; oracle adapter for `policy_table` representation)
- **#14 P1.5: `proposal_event_rate` and other internals** as diagnostics (requires `CEISResult` field expansion + estimator source change)
- **#14 P2: latent-factor / cross-day dependence tilts** (re-attempt `LatentFactorSimulator` path; composes with #15)
- **#14 P1.5 (uncertainty)**: block-bootstrap CI on CE-IS estimate
- **`comparison_to_v_pi` cross-method synthesis** (separate memo)
- **Lockbox cert run**

## Resolved review decisions

All design questions have been resolved across the #119–#127 review cycle. Captured here for downstream reference:

1. **Fixed-window estimand at horizon=n_holdout_dates** is the P0 commitment (Codex #123 option A). Synthetic-season construction (option B) deferred to P1.5+ if needed.
2. **Oracle test setup**: p=0.95, season_length=70, streak_threshold=57 (matches `exact_p57`'s hard-coded absorbing state); exact ≈ 0.08866 per Codex #125's local check.
3. **theta as "proposal tuning" not "model fitting"** is the docstring/comment convention.
4. **`v1_simplifications` list in estimator block is kept** — useful for downstream readers and prevents overclaim about theta_0-only / elite-set / strategy-agnostic.
5. **Diagnostic thresholds (NOT gates)**: `min_ess=1000`, `max_weight_share=0.1` are provisional defaults. Public weight diagnostics at theta=0 are mathematically perfect (ESS=n, max_weight_share=1/n), so these thresholds only fire under degenerate post-CE proposals.
6. **Real-data smoke targets the canonical default `streak_threshold=57`** — fixed_window_estimate is expected near-zero (mechanically: ~0.78^57 ignoring resets). Smaller-threshold cases live in synthetic CLI tests, NOT the canonical real-data smoke.
7. **`comparison_to_v_pi` block omitted from P0/P1** to prevent overclaim across estimands. Cross-method synthesis goes in a separate memo after #14 lands.
8. **No public `run_ce_rounds_on_train` function** — the wrapper has to call `estimate_p57_with_ceis` twice (train then holdout) per the contract in §3, with `n_final_train=2000` defaulting smaller than `n_final_holdout=20000` since the train point estimate is discarded.

## What this memo does NOT propose

- No math change to `estimate_p57_with_ceis` or `cross_entropy_tilt_step`.
- No new `CEISResult` field (proposal_event_rate or otherwise).
- No new state encoder.
- No oracle adapter for `policy_table` representations (deferred to P1.5+).
- No revival of `LatentFactorSimulator` (deferred to #14 P2).
- No `Strategy → policy_table` adapter (consistent with #11/#13 deferral pattern).
- No coupling to #13's V_pi output.
- No code changes outside `docs/superpowers/specs/`.

## Phase 0 acceptance (per Codex #120 + #123 + #125 + #127)

- [x] This design memo with API boundaries
- [x] All Codex #120 design questions answered
- [x] All Codex #123 blockers addressed (estimand/horizon explicit, CE algorithm matches v1 source, oracle requires `exact_p57` implementation test, diagnostics use only public `CEISResult` fields, `comparison_to_v_pi` block removed, harness-driver wording corrected)
- [x] All Codex #125 fixes applied (oracle test parameters that work with `exact_p57`'s threshold=57 hardcoding, dropped tests requiring private internals, replaced conceptually-backwards CE-improvement test with black-box-safe equivalent, no-public-tune-only-API contract explicit)
- [x] Codex #127 docs cleanup (version metadata, revision note generalized, resolved-review-decisions section, phase acceptance updated)
- [ ] Codex review of final memo
- [ ] (Implementation PR) gated on Codex sign-off
