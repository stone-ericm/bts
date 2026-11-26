# BTS Falsification Harness v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close v1's two methodology gaps — per-rank-1-bin `rho_pair` correction + within-fold dependence-parameter estimation — to tighten the `HEADLINE_BROKEN` verdict before any production-deploy decision.

**Architecture:** Per-rank-1-bin `rho_pair` (5-element vector) drives the v2 correction; per-cross-bin-cell heatmap (15 lower-triangular cells) is *diagnostic only*, never feeds the verdict. Within each LOSO fold, refit `rho_PA` / `tau` / `rho_pair` on that fold's 4 training seasons. `tau` estimator gets explicit stability metrics rather than silent fallback to pooled (which would replicate v1's contamination).

**Tech Stack:** Python 3.12, pandas, numpy, statsmodels (existing). No new deps.

---

## Pre-registered v2 estimand

> **Primary v2 metric:** LOSO corrected `P(57)`, with bins **and** all dependence parameters (`rho_PA`, `tau`, `rho_pair_per_bin`) estimated only on each fold's 4 held-in seasons. `rho_pair` correction is per-rank-1-bin (5-element vector applied bin-by-bin in `build_corrected_transition_table`).

Three result categories, kept structurally distinct in the verdict JSON:

1. **Correction (drives the verdict):** per-rank-1-bin rho_pair, fold-local PA + cross-game parameters
2. **Diagnostic (informs v3 decision):** per-cross-bin-cell 5×5 lower-triangular heatmap with empirical p_both, synthetic p1×p2, rho estimate, cell n, bootstrap CI per cell
3. **Sensitivity (robustness checks):** pooled-parameter run (the v1 status quo) — reported as a number but **never replaces the primary v2 verdict**. (tau-grid sweep removed from v2 scope per Codex round 1; add as v2.5 if needed.)

This structure is enforced by Codex synthesis (2026-05-02 design consult) — without it, v2 risks expanding into a full dependence-modeling project rather than tightening the v1 verdict.

## Plan amendment rule

If the diagnostic heatmap reveals surprising structure (e.g., Q4 negative gap doesn't survive fold-local estimation, or a different bin shows the dominant gap), **file a v3 issue with the finding — do not mutate the primary v2 estimand mid-run.** Mid-run estimand mutation is exactly the methodology fishing the harness was built to prevent.

## JSON serialization helpers (used by Tasks 6, 7)

`fold_metadata` and the diagnostic heatmap contain numpy arrays and possibly NaN cells. Verdict JSONs must be valid for strict parsers (`jq`, etc.). Add to `scripts/run_falsification_harness.py`:

```python
def _to_jsonable(obj):
    """Recursively convert numpy types + NaN to JSON-safe Python types.

    NaN → None (so jq doesn't choke on raw NaN literals which aren't legal JSON).
    """
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
```

## Non-goals (explicitly out of scope for v2)

- Per-cross-bin-cell *correction* (15-element matrix in `build_corrected_transition_table`). Diagnostic only in v2.
- Full GLMM MLE replacement of the existing method-of-moments tau estimator. The MoM estimator already has graceful tau-floor handling; we add stability metrics, not a different estimator.
- Bias-corrected rho estimates as primary. Use bootstrap to *quantify* finite-sample bias; only add a correction if simulation shows it materially changes the verdict.
- Production policy replacement. v2 produces a verdict + corrected-policy artifact; *deployment* is a manual decision after v2 lands.
- Distribution-shift remediation (project_bts_strategic_gaps_2026_04_30.md item #1). Larger thread; v2 informs but does not subsume.

## Cost estimate

- **Codex consults:** 2 rounds at gpt-5.5-high-reasoning ~$3 (post-plan adversarial review + post-implementation review). Already used ~$1.50 on the design consult.
- **Compute:** zero new cloud spend — re-runs harness on existing Task 13 parquets (`data/simulation/profiles_seed*_season*.parquet`, `pa_predictions_seed*_season*.parquet`).
- **Wall clock:** ~3-5 hours of subagent-driven implementation + 2 hours of Codex review iteration + ~30min for the verdict re-run.

## File structure

| File | Action | Why |
|---|---|---|
| `src/bts/validate/dependence.py` | Modify (~line 212-261, 264-354, 87-209) | per-bin rho_pair return, per-cell diagnostic, per-bin rho_pair input to build_corrected_transition_table, tau stability metrics |
| `src/bts/validate/ope.py` | Modify (~line 481-550) | corrected_audit_pipeline within-fold parameter estimation |
| `scripts/run_falsification_harness.py` | Modify (~line 101-296) | wire v2 path + emit diagnostic heatmap |
| `scripts/build_corrected_mdp_policy.py` | Modify (full file) | use v2 per-bin rho_pair |
| `tests/validate/test_dependence.py` | Modify (add ~10 tests) | new return shapes + stability metrics + per-cell diagnostic |
| `tests/validate/test_ope.py` | Modify (add ~3 tests) | within-fold parameter estimation in corrected_audit_pipeline |
| `tests/scripts/test_run_falsification_harness.py` | Modify (add ~2 tests) | v2 harness output schema |
| `docs/sota_audit/2026-05-02-harness-v2-comparison.md` | Create | post-run analysis memo |
| `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md` | Modify | mark Area #15 v2 ship status |

---

### Task 1: per-rank-1-bin rho_pair vector in `pair_residual_correlation`

**Files:**
- Modify: `src/bts/validate/dependence.py:212-261` (existing `pair_residual_correlation`)
- Test: `tests/validate/test_dependence.py`

**Why:** v1 returned a single global `rho_pair_cross_game` scalar. Aggregate ≈ 0 hid Q4's 8pp pair-correlation gap. Per-rank-1-bin breaks the aggregate apart so each bin's cooperative/antagonistic structure can drive its own MDP transition correction.

**Backward compatibility:** When `bin_assignment=None` (default), function preserves existing scalar return shape. Existing v1 callers don't break.

**New signature:**

```python
def pair_residual_correlation(
    df: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    bin_assignment: pd.Series | np.ndarray | None = None,
    expected_bin_indices: np.ndarray | list | None = None,
) -> tuple[float, float, float, float] | dict:
    """Stratified permutation test on rank-1/rank-2 Pearson residuals.

    [existing docstring]

    When bin_assignment is None: returns (rho_hat, ci_lo, ci_hi, p_value).

    When bin_assignment is provided (one rank-1 bin index per row in df):
    returns dict with arrays indexed by `expected_bin_indices` (when given) or
    by sorted unique values of bin_assignment otherwise.

    **CRITICAL** (caught by Codex round 1 review): caller MUST pass
    `expected_bin_indices=np.arange(n_bins)` when the consumer indexes the
    output by `bin.index` (as `build_corrected_transition_table` does).
    Otherwise output[i] silently means different bins on different folds when
    a fold's data lacks bin labels.

    Returned dict keys:
        rho_per_bin: shape-(K,) array; rho_per_bin[k] is rho for bin k
            where K = len(expected_bin_indices) if given, else len(unique(bin_assignment))
        ci_lo_per_bin: shape-(K,)
        ci_hi_per_bin: shape-(K,)
        p_value_per_bin: shape-(K,)
        n_per_bin: shape-(K,) — bin observation counts (0 means bin absent in this fold's data)
        bin_indices: shape-(K,) — the bin indices each output row corresponds to
        global_rho, global_ci_lo, global_ci_hi, global_p_value: aggregate scalars

    Empty-bin behavior: when n_per_bin[k] < 2, rho_per_bin[k]=0.0,
    ci_lo/ci_hi=0.0, p_value=1.0. Caller should check n_per_bin and treat
    rho=0 as "uninformative for this bin in this fold."
    """
```

- [ ] **Step 1: Write failing test for per-bin rho return shape**

Add to `tests/validate/test_dependence.py`:

```python
def test_pair_residual_correlation_per_bin_returns_dict_with_correct_shapes():
    """Per-bin rho returns shape-(K,) arrays for K unique bins."""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "p_rank1": rng.uniform(0.5, 0.95, n),
        "y_rank1": rng.binomial(1, 0.7, n),
        "p_rank2": rng.uniform(0.4, 0.85, n),
        "y_rank2": rng.binomial(1, 0.6, n),
    })
    bin_assignment = pd.Series(rng.integers(0, 5, n))  # 5 bins

    result = pair_residual_correlation(
        df, n_permutations=100, bin_assignment=bin_assignment,
    )

    assert isinstance(result, dict)
    assert result["rho_per_bin"].shape == (5,)
    assert result["ci_lo_per_bin"].shape == (5,)
    assert result["ci_hi_per_bin"].shape == (5,)
    assert result["p_value_per_bin"].shape == (5,)
    assert result["n_per_bin"].shape == (5,)
    assert isinstance(result["global_rho"], float)
    # Per-bin n_per_bin should sum to total
    assert int(result["n_per_bin"].sum()) == n


def test_pair_residual_correlation_scalar_path_preserved_for_back_compat():
    """When bin_assignment is None, return tuple (rho, ci_lo, ci_hi, p) as before."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "p_rank1": rng.uniform(0.5, 0.95, 200),
        "y_rank1": rng.binomial(1, 0.7, 200),
        "p_rank2": rng.uniform(0.4, 0.85, 200),
        "y_rank2": rng.binomial(1, 0.6, 200),
    })
    result = pair_residual_correlation(df, n_permutations=100)
    assert isinstance(result, tuple)
    assert len(result) == 4
    rho, ci_lo, ci_hi, p = result
    assert isinstance(rho, float)


def test_pair_residual_correlation_missing_bin_returns_zero_at_correct_index():
    """CRITICAL: when bin 2 is missing from data but expected_bin_indices=[0..4],
    output[2] is the empty-bin slot — NOT the next observed bin's value.

    This protects against the silent indexing-shift bug where bin labels in
    output don't match bin labels the consumer indexes by.
    """
    rng = np.random.default_rng(42)
    n = 400
    # Build df where rank-1 bin is in {0,1,3,4} (no bin 2).
    df = pd.DataFrame({
        "p_rank1": rng.uniform(0.5, 0.95, n),
        "y_rank1": rng.binomial(1, 0.7, n),
        "p_rank2": rng.uniform(0.4, 0.85, n),
        "y_rank2": rng.binomial(1, 0.6, n),
    })
    bin_assignment = pd.Series(rng.choice([0, 1, 3, 4], n))

    result = pair_residual_correlation(
        df, n_permutations=50,
        bin_assignment=bin_assignment,
        expected_bin_indices=np.arange(5),  # [0,1,2,3,4]
    )
    # Output is shape-(5,) with bin 2 empty.
    assert result["rho_per_bin"].shape == (5,)
    assert result["n_per_bin"][2] == 0  # bin 2 absent
    assert result["rho_per_bin"][2] == 0.0  # empty bin → rho=0
    assert result["p_value_per_bin"][2] == 1.0  # empty bin → p=1
    # Bins 0, 1, 3, 4 should have non-zero counts.
    for k in [0, 1, 3, 4]:
        assert result["n_per_bin"][k] > 0
    # bin_indices should match what was passed in.
    np.testing.assert_array_equal(result["bin_indices"], np.arange(5))
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py::test_pair_residual_correlation_per_bin_returns_dict_with_correct_shapes tests/validate/test_dependence.py::test_pair_residual_correlation_scalar_path_preserved_for_back_compat -v`

Expected: 1 fails (per-bin not implemented), 1 passes (scalar path already exists).

- [ ] **Step 3: Implement per-bin path**

Update `pair_residual_correlation` body. Insert after the existing scalar-path computation, before `return rho_hat, ci_lo, ci_hi, p_value`:

```python
def pair_residual_correlation(
    df: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    bin_assignment: pd.Series | np.ndarray | None = None,
):
    rng = np.random.default_rng(seed)
    e1 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank1"], df["p_rank1"])])
    e2 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank2"], df["p_rank2"])])

    rho_hat = float(np.mean(e1 * e2))

    null_distribution = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(e2)
        null_distribution[k] = float(np.mean(e1 * shuffled))
    p_value = float(np.mean(np.abs(null_distribution) >= abs(rho_hat)))

    n = len(e1)
    bs = np.empty(n_permutations)
    for k in range(n_permutations):
        idx = rng.integers(0, n, n)
        bs[k] = float(np.mean(e1[idx] * e2[idx]))
    ci_lo = float(np.quantile(bs, 0.025))
    ci_hi = float(np.quantile(bs, 0.975))

    if bin_assignment is None:
        return rho_hat, ci_lo, ci_hi, p_value

    # Per-bin path. bin_assignment must have len == len(df).
    bins_arr = np.asarray(bin_assignment)
    assert len(bins_arr) == n, (
        f"bin_assignment length {len(bins_arr)} != df length {n}"
    )
    # Use caller-supplied expected indices if given (safe contract); else fall
    # back to sorted unique values (legacy behavior).
    if expected_bin_indices is not None:
        bin_indices = np.asarray(expected_bin_indices)
    else:
        bin_indices = np.sort(np.unique(bins_arr))
    K = len(bin_indices)
    rho_per_bin = np.zeros(K)
    ci_lo_per_bin = np.zeros(K)
    ci_hi_per_bin = np.zeros(K)
    p_per_bin = np.ones(K)  # default p=1.0 for empty bins
    n_per_bin = np.zeros(K, dtype=int)

    for k, bin_idx in enumerate(bin_indices):
        mask = bins_arr == bin_idx
        e1_b = e1[mask]
        e2_b = e2[mask]
        n_b = len(e1_b)
        n_per_bin[k] = n_b
        if n_b < 2:
            # Empty/singleton bin: rho=0, p=1.0 (uninformative); rest already
            # zeroed at init. Caller should check n_per_bin.
            continue
        rho_b = float(np.mean(e1_b * e2_b))
        rho_per_bin[k] = rho_b
        # Permutation null within this bin.
        null_b = np.empty(n_permutations)
        for j in range(n_permutations):
            shuffled_b = rng.permutation(e2_b)
            null_b[j] = float(np.mean(e1_b * shuffled_b))
        p_per_bin[k] = float(np.mean(np.abs(null_b) >= abs(rho_b)))
        # Bootstrap CI within this bin.
        bs_b = np.empty(n_permutations)
        for j in range(n_permutations):
            idx_b = rng.integers(0, n_b, n_b)
            bs_b[j] = float(np.mean(e1_b[idx_b] * e2_b[idx_b]))
        ci_lo_per_bin[k] = float(np.quantile(bs_b, 0.025))
        ci_hi_per_bin[k] = float(np.quantile(bs_b, 0.975))

    return {
        "rho_per_bin": rho_per_bin,
        "ci_lo_per_bin": ci_lo_per_bin,
        "ci_hi_per_bin": ci_hi_per_bin,
        "p_value_per_bin": p_per_bin,
        "n_per_bin": n_per_bin,
        "bin_indices": bin_indices,
        "global_rho": rho_hat,
        "global_ci_lo": ci_lo,
        "global_ci_hi": ci_hi,
        "global_p_value": p_value,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py -v -k pair_residual`

Expected: all pair_residual tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/validate/dependence.py tests/validate/test_dependence.py
git commit -m "feat(validate.dependence): per-rank-1-bin rho_pair vector when bin_assignment given"
```

---

### Task 2: `build_corrected_transition_table` accepts per-bin `rho_pair` vector

**Files:**
- Modify: `src/bts/validate/dependence.py:264-354`
- Test: `tests/validate/test_dependence.py`

**Why:** v1 used scalar `rho_pair_cross_game` to apply the same correction to every bin's `p_both`. v2 needs each bin's correction to use its own `rho_pair` value. Backward-compat: scalar input still works.

- [ ] **Step 1: Write failing test for per-bin rho_pair input**

Add to `tests/validate/test_dependence.py`:

```python
def test_build_corrected_transition_table_accepts_per_bin_rho_vector():
    """Passing a length-K rho_pair vector applies per-bin correction."""
    from bts.simulate.quality_bins import QualityBins, QualityBin
    bins = QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.0, 0.5), p_hit=0.3, p_both=0.10, frequency=0.2),
            QualityBin(index=1, p_range=(0.5, 0.6), p_hit=0.55, p_both=0.30, frequency=0.2),
            QualityBin(index=2, p_range=(0.6, 0.7), p_hit=0.65, p_both=0.42, frequency=0.2),
            QualityBin(index=3, p_range=(0.7, 0.8), p_hit=0.75, p_both=0.55, frequency=0.2),
            QualityBin(index=4, p_range=(0.8, 1.0), p_hit=0.85, p_both=0.72, frequency=0.2),
        ],
        boundaries=[0.0, 0.5, 0.6, 0.7, 0.8, 1.0],
    )
    rho_per_bin = np.array([0.0, 0.0, 0.0, -0.10, +0.05])  # Q4 negative, Q5 positive
    corrected = build_corrected_transition_table(
        bins,
        rho_PA_within_game=0.0,
        tau_squared=0.0,
        rho_pair_cross_game=rho_per_bin,
        n_pa_per_game=5,
    )
    # Q1-Q3 use rho=0 → p_both should equal p1*p2 (within FH bounds).
    for i in [0, 1, 2]:
        b_orig = bins.bins[i]
        b_new = corrected.bins[i]
        expected_pboth = b_orig.p_hit ** 2
        assert abs(b_new.p_both - expected_pboth) < 1e-9, (
            f"Q{i+1}: rho=0 should give p_both = p_hit^2"
        )
    # Q4 uses rho=-0.10 → p_both should be BELOW p1*p2.
    b4_orig = bins.bins[3]
    b4_new = corrected.bins[3]
    p1 = p2 = b4_orig.p_hit
    sigma = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    expected_q4 = p1 * p2 + (-0.10) * sigma
    expected_q4 = max(0.0, p1 + p2 - 1.0) if expected_q4 < max(0.0, p1 + p2 - 1.0) else expected_q4
    expected_q4 = min(p1, p2) if expected_q4 > min(p1, p2) else expected_q4
    assert abs(b4_new.p_both - expected_q4) < 1e-9


def test_build_corrected_transition_table_scalar_input_still_works():
    """Backward compat: scalar rho_pair_cross_game applies to all bins."""
    from bts.simulate.quality_bins import QualityBins, QualityBin
    bins = QualityBins(
        bins=[QualityBin(index=i, p_range=(i*0.2, (i+1)*0.2), p_hit=0.5, p_both=0.25, frequency=0.2) for i in range(5)],
        boundaries=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    corrected = build_corrected_transition_table(
        bins,
        rho_PA_within_game=0.0,
        tau_squared=0.0,
        rho_pair_cross_game=0.05,  # scalar
        n_pa_per_game=5,
    )
    for b in corrected.bins:
        assert abs(b.p_both - (0.5*0.5 + 0.05*0.25)) < 1e-9
```

- [ ] **Step 2: Run failing**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py -v -k "build_corrected_transition_table"`

Expected: per-bin test fails (numpy array not accepted), scalar test already passes.

- [ ] **Step 3: Implement per-bin acceptance**

Modify `build_corrected_transition_table` body in `src/bts/validate/dependence.py:264-354`:

Change the type annotation:
```python
def build_corrected_transition_table(
    bins,
    *,
    rho_PA_within_game: float,
    tau_squared: float,
    rho_pair_cross_game: float | np.ndarray,
    n_pa_per_game: int = 5,
):
```

Inside the loop over `bins.bins`, replace:

```python
new_p_both = p1 * p2 + rho_pair_cross_game * np.sqrt(...)
```

with:

**Outside the loop**, normalize input once via `np.asarray` (Codex round 1: handles list/tuple/array uniformly):

```python
# At top of build_corrected_transition_table, after the imports:
rho_arr = np.asarray(rho_pair_cross_game, dtype=float).ravel()
if rho_arr.size == 1:
    # Scalar input → broadcast to all bins via the same scalar value.
    pass
elif rho_arr.size != len(bins.bins):
    raise ValueError(
        f"rho_pair_cross_game length {rho_arr.size} != number of bins "
        f"{len(bins.bins)}; pass scalar or per-bin vector"
    )
```

Then **inside the loop**, replace the `new_p_both = p1 * p2 + rho_pair_cross_game * np.sqrt(...)` line with:

```python
# Per-bin rho_pair: when scalar, broadcast; when array, index by bin position.
# b.index is in [0, K-1] matching QualityBin's contract.
rho_for_bin = float(rho_arr[0]) if rho_arr.size == 1 else float(rho_arr[b.index])
new_p_both = p1 * p2 + rho_for_bin * np.sqrt(
    p1 * (1.0 - p1) * p2 * (1.0 - p2)
)
```

Also update the docstring to document per-bin acceptance + array coercion behavior (lists, tuples, and ndarrays all work).

- [ ] **Step 4: Run tests pass**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py -v -k "build_corrected"`

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/validate/dependence.py tests/validate/test_dependence.py
git commit -m "feat(validate.dependence): build_corrected_transition_table accepts per-bin rho_pair vector"
```

---

### Task 3: small-sample stability metrics for `tau` estimator

**Files:**
- Modify: `src/bts/validate/dependence.py:87-209` (existing `fit_logistic_normal_random_intercept`)
- Test: `tests/validate/test_dependence.py`

**Why:** Codex flagged that silently falling back to pooled tau on small-fold instability replicates v1's contamination. The fix is **not** an optimizer hierarchy (the existing estimator is method-of-moments — no MLE convergence to fail) but **explicit stability metrics** so the harness driver can flag fold-local tau as uncertain. The estimator already has graceful tau-floor behavior (rho_hat ≤ 0 → tau=0, brentq ValueError → tau=0); we add observability.

**New return signature:** existing returns `(tau_hat, integrate_fn)`. v2 returns `(tau_hat, integrate_fn, stability)` where `stability` is a dict.

- [ ] **Step 1: Write failing test for stability dict shape**

Add to `tests/validate/test_dependence.py`:

```python
def test_fit_logistic_normal_random_intercept_returns_stability_dict():
    """3-tuple return: (tau, integrate_fn, stability_dict)."""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "group_id": rng.integers(0, 100, n),
        "p_pred": rng.uniform(0.2, 0.4, n),
        "y": rng.binomial(1, 0.3, n),
    })
    result = fit_logistic_normal_random_intercept(df)
    assert len(result) == 3
    tau, integrate_fn, stability = result
    assert isinstance(stability, dict)
    assert "n_groups" in stability
    assert "total_pair_n" in stability
    assert "rho_hat" in stability
    assert "small_sample_warning" in stability
    assert isinstance(stability["n_groups"], int)
    assert stability["n_groups"] > 0


def test_stability_warning_fires_on_small_pair_count():
    """Few groups with tiny within-group n → small_sample_warning=True."""
    df = pd.DataFrame({
        "group_id": [0, 0, 1, 1],  # 2 groups, 1 pair each = total_pair_n=2
        "p_pred": [0.3, 0.4, 0.5, 0.6],
        "y": [0, 1, 1, 0],
    })
    _, _, stability = fit_logistic_normal_random_intercept(df)
    assert stability["small_sample_warning"] is True
    assert stability["total_pair_n"] < 100  # threshold
```

- [ ] **Step 2: Run failing**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py -v -k "stability"`

Expected: both fail (returns 2-tuple).

- [ ] **Step 3: Implement stability dict + 3-tuple return**

Update `fit_logistic_normal_random_intercept`. After computing `rho_hat` and `tau_hat`, add stability tracking. Change last line:

```python
SMALL_SAMPLE_PAIR_THRESHOLD = 100

stability = {
    "n_groups": int(len(counts)),
    "total_pair_n": int(total_pair_n),
    "rho_hat": float(rho_hat),
    "small_sample_warning": bool(total_pair_n < SMALL_SAMPLE_PAIR_THRESHOLD),
    "estimator": "method_of_moments_pearson_pair_inversion",
}

return tau_hat, integrate_fn, stability
```

This breaks all existing callers — they need to unpack 3 values now. Update them in this same task to keep build green. **Exhaustive caller list** (verified via `grep -rn "fit_logistic_normal_random_intercept" --include='*.py'` 2026-05-02):

- `tests/validate/test_dependence.py:103` — `tau_hat, integrate_fn = ...` → `tau_hat, integrate_fn, _ = ...`
- `tests/validate/test_dependence.py:126` — same
- `tests/validate/test_dependence.py:148` — same
- `scripts/build_corrected_mdp_policy.py:75` — `tau_hat, _ = ...` → `tau_hat, _, _ = ...`
- `scripts/run_falsification_harness.py:165` — `tau_hat, _ = ...` → `tau_hat, _, _ = ...`

Note: `corrected_audit_pipeline` in `src/bts/validate/ope.py` does NOT directly call `fit_logistic_normal_random_intercept` — it currently uses a pre-built `corrected_policy` dict. Task 4 will add a new call site there which will use the 3-tuple natively.

Quick fix in non-test callers (scripts): `tau, integrate_fn, _ = fit_logistic_normal_random_intercept(df)` for now; Task 4 + Task 6 will use `stability` properly.

- [ ] **Step 4: Run tests pass — and full suite green**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/ -v`

Expected: all dependence/ope tests pass. The 3-tuple unpacking change is wired through.

- [ ] **Step 5: Commit**

```bash
git add src/bts/validate/dependence.py src/bts/validate/ope.py scripts/run_falsification_harness.py scripts/build_corrected_mdp_policy.py tests/validate/test_dependence.py
git commit -m "feat(validate.dependence): fit_logistic_normal_random_intercept returns stability dict"
```

---

### Task 4: `corrected_audit_pipeline` refits parameters within each LOSO fold

**Files:**
- Modify: `src/bts/validate/ope.py:481-550` (existing `corrected_audit_pipeline`)
- Test: `tests/validate/test_ope.py`

**Why:** v1's `corrected_audit_pipeline` accepts a *pre-built* `corrected_policy` dict (built once on full data), violating the audit boundary on parameters. v2: each fold builds its own corrected policy from its own 4 training seasons.

**New signature:** drops `corrected_policy` arg (v1 would still work via overload, but cleaner to break it for v2 — explicit > implicit). Adds `pa_df` arg (PA-level data needed for re-fitting `rho_PA` and `tau` per fold) and `mdp_solve_fn` (callable that takes corrected QualityBins and returns the action_table dict).

```python
def corrected_audit_pipeline(
    profiles: pd.DataFrame,
    pa_df: pd.DataFrame,
    *,
    fold_seasons: list[int],
    mdp_solve_fn: Callable[[QualityBins], dict],
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_pa_per_game: int = 5,
    rho_pair_n_permutations: int = 300,
) -> "DROPEResult":
    """LOSO audit with FOLD-LOCAL dependence parameters and corrected policy.

    For each held-out season: trains bins + dependence params (rho_PA, tau,
    rho_pair_per_bin) on the 4 training seasons, builds corrected QualityBins,
    solves MDP, and replays on held-out season.

    Returns DROPEResult with verdict point + fold-percentile CI plus, per fold:
        fold_metadata (list of dicts): rho_PA, tau, rho_pair_per_bin, stability
        warnings, n_groups for each fold.
    """
```

- [ ] **Step 1: Write failing tests including a no-leakage assertion**

Helpers (place at module level in `tests/validate/test_ope.py`, NOT as `@pytest.fixture` per Codex round 1 — fixtures called as `()` are inconsistent with their decorator):

```python
def _synthetic_profiles_5_seasons(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        for day_idx in range(150):
            date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day_idx)
            for s in [42, 43]:  # 2 seeds → 1500 rows per season, 7500 total
                p1 = rng.uniform(0.6, 0.9)
                p2 = rng.uniform(0.4, p1)
                rows.append({
                    "season": season, "date": date, "seed": s,
                    "top1_p": p1, "top1_hit": int(rng.random() < p1),
                    "top2_p": p2, "top2_hit": int(rng.random() < p2),
                })
    return pd.DataFrame(rows)


def _synthetic_pa_5_seasons(seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        for day_idx in range(150):
            date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day_idx)
            for batter_id in range(20):
                bg_id = f"{season}-{day_idx}-{batter_id}"
                p_pa = rng.uniform(0.15, 0.35)
                for pa_num in range(4):
                    rows.append({
                        "season": season, "date": date,
                        "batter_game_id": bg_id, "pa_num": pa_num,
                        "p_pa": p_pa, "actual_hit": int(rng.random() < p_pa),
                    })
    return pd.DataFrame(rows)


def _all_skip_policy(bins) -> np.ndarray:
    """Fake solver returning the contract that _trajectory_dataframe_from_profiles
    expects: np.ndarray[57, season_length+1, 2, n_bins]. ACTION_SKIP=0.

    Codex round 1 caught: a dict-of-tuples fake returned wrong shape and would
    have made tests pass-by-error. The real solver returns this np.ndarray.
    """
    n_bins = len(bins.bins)
    season_length = 200  # enough for any realistic test season
    return np.zeros((57, season_length + 1, 2, n_bins), dtype=int)
```

The actual tests:

```python
def test_corrected_audit_pipeline_refits_parameters_per_fold(monkeypatch):
    """corrected_audit_pipeline calls dependence estimators once per fold AND
    each call's input excludes the held-out season (no leakage).
    """
    from bts.validate import ope as ope_mod

    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    leakage_violations = []  # collect held_out values that appeared in passed-in df

    def make_no_leak_wrapper(real_fn, fn_name):
        # The body of corrected_audit_pipeline tracks which season is held_out
        # in a `current_held_out` local. We can't directly observe that from
        # the wrapper, but we CAN observe that no single call's input df should
        # contain ALL 5 seasons — if it does, fold-local slicing failed.
        # Stronger check: every call's input should have len(unique seasons) == 4.
        def wrapper(df, *args, **kwargs):
            if "season" in df.columns:
                seasons_in_df = set(df["season"].unique())
                if len(seasons_in_df) != 4:
                    leakage_violations.append(
                        (fn_name, sorted(seasons_in_df))
                    )
            return real_fn(df, *args, **kwargs)
        return wrapper

    from bts.validate import dependence as dep_mod
    monkeypatch.setattr(
        dep_mod, "pa_residual_correlation",
        make_no_leak_wrapper(dep_mod.pa_residual_correlation, "pa_residual_correlation"),
    )
    monkeypatch.setattr(
        dep_mod, "fit_logistic_normal_random_intercept",
        make_no_leak_wrapper(dep_mod.fit_logistic_normal_random_intercept, "fit_lnri"),
    )
    # pair_residual_correlation takes pair_df (different column shape) — wrap
    # using a bin-assignment-aware no-leak helper:
    real_pair = dep_mod.pair_residual_correlation
    def no_leak_pair(pair_df, **kwargs):
        # pair_df doesn't have a 'season' column directly; check via 'date' if needed.
        # For this test, just count calls to verify per-fold-ness.
        no_leak_pair.call_count += 1
        return real_pair(pair_df, **kwargs)
    no_leak_pair.call_count = 0
    monkeypatch.setattr(dep_mod, "pair_residual_correlation", no_leak_pair)

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
    )

    # Per-fold count: pa_residual_correlation called 5x, fit_lnri 5x, pair 5x.
    assert no_leak_pair.call_count == 5
    # Critical: NO leakage violations should have been recorded.
    assert leakage_violations == [], f"Held-out leaked into estimator input: {leakage_violations}"


def test_corrected_audit_pipeline_returns_fold_metadata_with_per_bin_rho():
    """fold_metadata has per-fold rho_PA, tau, rho_pair_per_bin (shape 5),
    rho_pair_per_bin_ci, n_per_bin, stability dict.
    """
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
    )
    assert hasattr(result, "fold_metadata")
    assert len(result.fold_metadata) == 5
    for fold_meta in result.fold_metadata:
        assert "held_out_season" in fold_meta
        assert "rho_PA" in fold_meta
        assert "tau" in fold_meta
        assert "rho_pair_per_bin" in fold_meta
        assert fold_meta["rho_pair_per_bin"].shape == (5,)
        assert "rho_pair_per_bin_ci_lo" in fold_meta
        assert "rho_pair_per_bin_ci_hi" in fold_meta
        assert "rho_pair_n_per_bin" in fold_meta
        assert fold_meta["rho_pair_n_per_bin"].shape == (5,)
        assert "stability" in fold_meta
        assert "small_sample_warning" in fold_meta["stability"]
```

- [ ] **Step 2: Run failing**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_ope.py -v -k corrected_audit`

Expected: 2 failures (signature mismatch, fold_metadata missing).

- [ ] **Step 3a: Add `fold_metadata` field to `DROPEResult`**

`DROPEResult` is a plain mutable `@dataclass` (verified at `src/bts/validate/ope.py:63-72`, neither frozen nor slotted). The right pattern is a default-factory field, NOT post-construction mutation:

```python
@dataclass
class DROPEResult:
    """Result of one DR-OPE evaluation."""

    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_trajectories: int = 0
    nuisance_v_hat: float | None = None
    bootstrap_distribution: np.ndarray | None = None
    fold_metadata: list[dict] = field(default_factory=list)  # NEW (v2)
```

The `field` import: `from dataclasses import dataclass, field` (replace existing `from dataclasses import dataclass`).

- [ ] **Step 3b: Implement v2 `corrected_audit_pipeline`**

Replace the body of `corrected_audit_pipeline` in `src/bts/validate/ope.py:481-550`:

```python
def corrected_audit_pipeline(
    profiles: pd.DataFrame,
    pa_df: pd.DataFrame,
    *,
    fold_seasons: list[int],
    mdp_solve_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_pa_per_game: int = 5,
    rho_pair_n_permutations: int = 300,
    pa_n_bootstrap: int = 300,
) -> "DROPEResult":
    """LOSO audit with FOLD-LOCAL dependence parameters and corrected policy.

    For each held-out season:
      1. Slice training data to the 4 held-in seasons (no leakage).
      2. Fit fold-local bins from training profiles.
      3. Fit fold-local rho_PA, tau, rho_pair_per_bin from training PA + profiles.
      4. Build corrected QualityBins with the fold-local parameters.
      5. Solve MDP on corrected bins (via mdp_solve_fn).
      6. Replay on held-out season; compute terminal-reward MC point estimate.

    mdp_solve_fn contract (Codex round 1 fix): callable that takes
    `corrected_bins: QualityBins` and returns either an `np.ndarray`
    policy_table of shape (57, season_length+1, 2, n_bins) OR an object with
    `.policy_table` attribute (e.g., MDPSolution from solve_mdp). Adapter logic
    below normalizes both.

    Returns DROPEResult with mean across folds + percentile CI when n_folds >= 5,
    plus fold_metadata list populated with rho_PA, tau, rho_pair_per_bin (shape K
    indexed by bin.index 0..K-1), rho_pair_per_bin_ci, rho_pair_n_per_bin,
    stability dict, and per-fold P(57).
    """
    from bts.validate.dependence import (
        build_corrected_transition_table,
        fit_logistic_normal_random_intercept,
        pa_residual_correlation,
        pair_residual_correlation,
    )

    fold_estimates = []
    fold_metadata = []

    for fold_idx, held_out in enumerate(fold_seasons):
        # Fold-specific RNG seed (Codex round 1 minor): different fold gets
        # different bootstrap/permutation realizations.
        fold_seed = seed + fold_idx

        # Within-fold training data: 4 held-in seasons.
        train_profiles = profiles[profiles["season"] != held_out].copy()
        train_pa = pa_df[pa_df["season"] != held_out].copy()
        test_profiles = profiles[profiles["season"] == held_out].copy()

        # Fold-local bins.
        train_bins = _compute_bins_from_direct_profiles(train_profiles)
        n_bins = len(train_bins.bins)

        # Fold-local dependence parameters.
        rho_PA, rho_PA_lo, rho_PA_hi, _ = pa_residual_correlation(
            train_pa, n_bootstrap=pa_n_bootstrap, seed=fold_seed,
        )
        train_pa_for_lnri = train_pa.rename(columns={
            "batter_game_id": "group_id",
            "p_pa": "p_pred",
            "actual_hit": "y",
        })
        tau_hat, _, lnri_stability = fit_logistic_normal_random_intercept(train_pa_for_lnri)

        # Fold-local per-bin rho_pair. CRITICAL: pass expected_bin_indices to
        # guarantee the returned vector is indexed by bin.index 0..K-1, NOT
        # by sorted-unique-of-data (which silently shifts when a bin is empty).
        pair_df = train_profiles[["date", "top1_p", "top1_hit", "top2_p", "top2_hit"]].rename(
            columns={
                "top1_p": "p_rank1", "top1_hit": "y_rank1",
                "top2_p": "p_rank2", "top2_hit": "y_rank2",
            }
        )
        bin_assignment = pair_df["p_rank1"].apply(train_bins.classify)
        rho_result = pair_residual_correlation(
            pair_df,
            n_permutations=rho_pair_n_permutations,
            bin_assignment=bin_assignment,
            expected_bin_indices=np.arange(n_bins),  # CRITICAL: stable indexing
            seed=fold_seed,
        )

        # Build corrected bins for this fold (rho_per_bin indexed by bin.index).
        corrected_bins = build_corrected_transition_table(
            train_bins,
            rho_PA_within_game=rho_PA,
            tau_squared=tau_hat ** 2,
            rho_pair_cross_game=rho_result["rho_per_bin"],
            n_pa_per_game=n_pa_per_game,
        )

        # Solve MDP on this fold's corrected bins. Adapter normalizes the two
        # legitimate return shapes (Codex round 1 fix):
        solver_out = mdp_solve_fn(corrected_bins)
        if hasattr(solver_out, "policy_table"):
            policy_table = solver_out.policy_table
        elif isinstance(solver_out, np.ndarray):
            policy_table = solver_out
        else:
            raise TypeError(
                f"mdp_solve_fn returned unsupported type {type(solver_out)}; "
                f"must be np.ndarray or have .policy_table attribute"
            )

        # Replay on held-out season. _trajectory_dataframe_from_profiles takes
        # the np.ndarray policy_table and the corrected bins.
        traj_df = _trajectory_dataframe_from_profiles(
            test_profiles, policy_table, corrected_bins,
        )
        fold_result = _run_terminal_r_mc_bootstrap(
            traj_df, n_bootstrap=n_bootstrap, seed=fold_seed,
        )
        fold_estimates.append(fold_result.point_estimate)

        fold_metadata.append({
            "held_out_season": int(held_out),
            "rho_PA": float(rho_PA),
            "rho_PA_ci_lo": float(rho_PA_lo),
            "rho_PA_ci_hi": float(rho_PA_hi),
            "tau": float(tau_hat),
            "rho_pair_per_bin": rho_result["rho_per_bin"],         # shape (n_bins,)
            "rho_pair_per_bin_ci_lo": rho_result["ci_lo_per_bin"],  # shape (n_bins,)
            "rho_pair_per_bin_ci_hi": rho_result["ci_hi_per_bin"],
            "rho_pair_per_bin_p_value": rho_result["p_value_per_bin"],
            "rho_pair_n_per_bin": rho_result["n_per_bin"],
            "rho_pair_global": float(rho_result["global_rho"]),
            "bin_indices": rho_result["bin_indices"],
            "stability": lnri_stability,
            "fold_p57": float(fold_result.point_estimate),
        })

    point = float(np.mean(fold_estimates))
    if len(fold_estimates) >= 5:
        ci_lo = float(np.quantile(fold_estimates, 0.025))
        ci_hi = float(np.quantile(fold_estimates, 0.975))
    else:
        ci_lo, ci_hi = None, None

    return DROPEResult(
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_trajectories=len(fold_estimates),
        fold_metadata=fold_metadata,
    )
```

**Important:** the verdict-JSON writer in Task 6 must run `_to_jsonable(result.fold_metadata)` (helper defined at top of plan) before serialization — `numpy.ndarray` and `numpy.float64` aren't JSON-native.

- [ ] **Step 4: Run tests pass**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_ope.py -v -k corrected_audit`

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bts/validate/ope.py tests/validate/test_ope.py tests/validate/conftest.py
git commit -m "feat(validate.ope): corrected_audit_pipeline refits dependence parameters within each LOSO fold"
```

---

### Task 5: `pair_residual_correlation_per_cell` for diagnostic 5×5 lower-triangular heatmap

**Files:**
- Modify: `src/bts/validate/dependence.py` (add new function near line 261)
- Test: `tests/validate/test_dependence.py`

**Why:** Codex synthesis: per-cross-bin-cell rho is *diagnostic only*, not correction. Output is a 5×5 lower-triangular grid that informs whether v3 needs the cross-bin matrix in the correction layer.

**Convention (Codex round 1 fix — must be explicit):**
- BOTH `rank1_bin_assignment` AND `rank2_bin_assignment` are computed against the **rank-1-derived bin boundaries** (the QualityBins fitted on `top1_p`). This is the natural choice given that `p_rank2 ≤ p_rank1` always holds, so rank-2 lives in a subspace of rank-1's bin scheme. The function does NOT classify rank-2 against rank-2-derived boundaries.
- Output matrix: `rho_matrix[r1, r2]` is the rho for rank-1 in bin `r1` paired with rank-2 in bin `r2`. By the `p_rank2 ≤ p_rank1` invariant, cells with `r2 > r1` should have zero observations — these are output as `np.nan` for `rho_matrix` and `0` for `n_matrix`. The JSON writer in Task 6 converts NaN → null.

**New function signature:**

```python
def pair_residual_correlation_per_cell(
    df: pd.DataFrame,
    *,
    rank1_bin_assignment: pd.Series | np.ndarray,
    rank2_bin_assignment: pd.Series | np.ndarray,
    expected_bin_indices: np.ndarray | list,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Per-cross-bin-cell rho_pair for diagnostic heatmap (lower-triangular).

    Convention: BOTH bin_assignment arrays are computed against rank-1-derived
    bin boundaries. cells where rank-2 bin > rank-1 bin are expected to be empty
    by the p_rank2 <= p_rank1 invariant; populated only with NaN/0.

    Returns dict with keys (all shape (K, K) where K = len(expected_bin_indices)):
        rho_matrix: NaN for empty cells (n < 2), otherwise rho estimate
        n_matrix: int observation counts
        ci_lo_matrix, ci_hi_matrix: bootstrap CIs (NaN for empty cells)
        empirical_p_both_matrix: realized P(y_rank1=1 AND y_rank2=1) per cell (NaN for empty)
        synthetic_p1p2_matrix: mean(p_rank1) * mean(p_rank2) per cell (NaN for empty)
        bin_indices: shape-(K,) — what each row/col index means
    """
```

- [ ] **Step 1: Write failing test**

```python
def test_pair_residual_correlation_per_cell_returns_lower_triangular_dict():
    rng = np.random.default_rng(42)
    n = 2000  # Codex round 1: bigger n to populate off-diagonal cells
    # Generate data respecting the rank invariant p_rank2 <= p_rank1.
    p1_arr = rng.uniform(0.5, 0.95, n)
    p2_arr = np.array([rng.uniform(0.4, p) for p in p1_arr])
    df = pd.DataFrame({
        "p_rank1": p1_arr,
        "y_rank1": (rng.random(n) < p1_arr).astype(int),
        "p_rank2": p2_arr,
        "y_rank2": (rng.random(n) < p2_arr).astype(int),
    })
    # Use a real QualityBins boundaries-style 5-bin assignment.
    boundaries = np.array([0.0, 0.6, 0.7, 0.8, 0.9, 1.0])
    rank1_bins = np.clip(np.digitize(p1_arr, boundaries[1:-1]), 0, 4)
    rank2_bins = np.clip(np.digitize(p2_arr, boundaries[1:-1]), 0, 4)

    result = pair_residual_correlation_per_cell(
        df,
        rank1_bin_assignment=rank1_bins,
        rank2_bin_assignment=rank2_bins,
        expected_bin_indices=np.arange(5),
        n_permutations=50,
    )
    assert result["rho_matrix"].shape == (5, 5)
    # By the p_rank2 <= p_rank1 invariant, upper-triangular cells (r2 > r1)
    # should have n == 0 and rho == NaN.
    for r1 in range(5):
        for r2 in range(r1 + 1, 5):
            assert result["n_matrix"][r1, r2] == 0
            assert np.isnan(result["rho_matrix"][r1, r2])
    # Lower-triangular cells with non-zero counts should have non-NaN rho.
    n_populated_cells = 0
    for r1 in range(5):
        for r2 in range(r1 + 1):
            if result["n_matrix"][r1, r2] >= 2:
                assert not np.isnan(result["rho_matrix"][r1, r2])
                n_populated_cells += 1
    # At least 5 of the 15 lower-triangular cells should be populated at n=2000.
    assert n_populated_cells >= 5, f"Only {n_populated_cells}/15 cells populated"


def test_pair_residual_correlation_per_cell_invariant_violation_warning(caplog):
    """When p_rank2 > p_rank1 in any row, log a warning (data violates invariant)."""
    df = pd.DataFrame({
        "p_rank1": [0.5, 0.6],
        "y_rank1": [1, 1],
        "p_rank2": [0.7, 0.5],  # row 0 violates p_rank2 <= p_rank1
        "y_rank2": [0, 1],
    })
    rank1_bins = np.array([0, 0])
    rank2_bins = np.array([1, 0])  # row 0 has rank2_bin > rank1_bin
    with caplog.at_level("WARNING", logger="bts.validate.dependence"):
        pair_residual_correlation_per_cell(
            df,
            rank1_bin_assignment=rank1_bins,
            rank2_bin_assignment=rank2_bins,
            expected_bin_indices=np.arange(2),
            n_permutations=10,
        )
    assert any("invariant" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run failing**

- [ ] **Step 3: Implement**

```python
import logging

logger = logging.getLogger(__name__)


def pair_residual_correlation_per_cell(
    df, *, rank1_bin_assignment, rank2_bin_assignment, expected_bin_indices,
    n_permutations=500, seed=42,
):
    rng = np.random.default_rng(seed)
    n = len(df)
    r1_arr = np.asarray(rank1_bin_assignment)
    r2_arr = np.asarray(rank2_bin_assignment)
    bin_indices = np.asarray(expected_bin_indices)
    K = len(bin_indices)

    # Invariant check: p_rank2 <= p_rank1 implies r2_bin <= r1_bin.
    n_invariant_violations = int((r2_arr > r1_arr).sum())
    if n_invariant_violations > 0:
        logger.warning(
            "pair_residual_correlation_per_cell: %d/%d rows violate the rank invariant "
            "(rank2_bin > rank1_bin). Heatmap upper-triangular cells will be non-empty.",
            n_invariant_violations, n,
        )

    e1 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank1"], df["p_rank1"])])
    e2 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank2"], df["p_rank2"])])

    rho_matrix = np.full((K, K), np.nan)
    n_matrix = np.zeros((K, K), dtype=int)
    ci_lo_matrix = np.full((K, K), np.nan)
    ci_hi_matrix = np.full((K, K), np.nan)
    empirical_p_both_matrix = np.full((K, K), np.nan)
    synthetic_p1p2_matrix = np.full((K, K), np.nan)

    for r1 in bin_indices:
        for r2 in bin_indices:
            mask = (r1_arr == r1) & (r2_arr == r2)
            n_cell = int(mask.sum())
            n_matrix[r1, r2] = n_cell
            if n_cell < 2:
                continue
            e1_c = e1[mask]
            e2_c = e2[mask]
            rho_matrix[r1, r2] = float(np.mean(e1_c * e2_c))

            # Bootstrap CI within cell.
            bs = np.empty(n_permutations)
            for j in range(n_permutations):
                idx = rng.integers(0, n_cell, n_cell)
                bs[j] = float(np.mean(e1_c[idx] * e2_c[idx]))
            ci_lo_matrix[r1, r2] = float(np.quantile(bs, 0.025))
            ci_hi_matrix[r1, r2] = float(np.quantile(bs, 0.975))

            # Empirical p_both and synthetic p1*p2.
            y1_c = df["y_rank1"].to_numpy()[mask]
            y2_c = df["y_rank2"].to_numpy()[mask]
            empirical_p_both_matrix[r1, r2] = float(np.mean(y1_c * y2_c))
            p1_c = df["p_rank1"].to_numpy()[mask]
            p2_c = df["p_rank2"].to_numpy()[mask]
            synthetic_p1p2_matrix[r1, r2] = float(np.mean(p1_c) * np.mean(p2_c))

    return {
        "rho_matrix": rho_matrix,
        "n_matrix": n_matrix,
        "ci_lo_matrix": ci_lo_matrix,
        "ci_hi_matrix": ci_hi_matrix,
        "empirical_p_both_matrix": empirical_p_both_matrix,
        "synthetic_p1p2_matrix": synthetic_p1p2_matrix,
        "bin_indices": bin_indices,
    }
```

- [ ] **Step 4: Run passing**

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/validate/test_dependence.py -v -k per_cell`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(validate.dependence): pair_residual_correlation_per_cell — 5x5 diagnostic heatmap with rank-invariant check"
```

---

### Task 6: wire v2 into `run_falsification_harness` + emit diagnostic heatmap JSON

**Files:**
- Modify: `scripts/run_falsification_harness.py:101-296`
- Test: `tests/scripts/test_run_falsification_harness.py`

**Why:** End-to-end driver needs to (a) call v2 `corrected_audit_pipeline` with the new signature, (b) build the diagnostic heatmap, (c) compute pooled-parameter sensitivity, (d) write to verdict JSON via `_to_jsonable`.

**Changes:**

1. **Replace v2 corrected_audit_pipeline call site** with the new signature:

```python
from bts.simulate.mdp import solve_mdp

def _solve_for_v2(corrected_bins):
    """Closure that solves MDP on a fold's corrected bins. Returns MDPSolution."""
    return solve_mdp(corrected_bins, season_length=153, late_phase_days=30)

corrected_v2 = corrected_audit_pipeline(
    profiles, pa_df,
    fold_seasons=fold_seasons,
    mdp_solve_fn=_solve_for_v2,
    n_bootstrap=n_bootstrap,
    rho_pair_n_permutations=n_permutations,
    pa_n_bootstrap=n_permutations,
)
```

2. **Pooled-parameter sensitivity (simplified)**: the v1 verdict number (`0.0083 [0, 0.0375]`) already exists at `data/validation/falsification_harness_2026-05-02.json`. Read it directly into the v2 verdict JSON for side-by-side comparison rather than re-running a v1-style audit (which would add ~30min compute + significant code complexity for marginal value). The comparison memo (Task 8) tells the story; the verdict JSON includes a `v1_reference_p57` field referencing the historical v1 number plus a pointer to the v1 JSON file.

```python
import json
v1_path = Path("data/validation/falsification_harness_2026-05-02.json")
v1_p57 = "<not-available>"
if v1_path.exists():
    v1_data = json.loads(v1_path.read_text())
    v1_p57 = v1_data.get("corrected_pipeline_p57", "<missing>")
```

3. **Compute diagnostic heatmap** on full pooled data (it's diagnostic, doesn't need audit boundary):

```python
from bts.validate.dependence import pair_residual_correlation_per_cell
from bts.validate.ope import _compute_bins_from_direct_profiles

full_bins = _compute_bins_from_direct_profiles(profiles)
n_bins = len(full_bins.bins)
pair_df_full = profiles[["date", "top1_p", "top1_hit", "top2_p", "top2_hit"]].rename(
    columns={"top1_p": "p_rank1", "top1_hit": "y_rank1",
             "top2_p": "p_rank2", "y_rank2": "y_rank2"}
)
rank1_assign = pair_df_full["p_rank1"].apply(full_bins.classify)
rank2_assign = pair_df_full["p_rank2"].apply(full_bins.classify)
heatmap = pair_residual_correlation_per_cell(
    pair_df_full,
    rank1_bin_assignment=rank1_assign,
    rank2_bin_assignment=rank2_assign,
    expected_bin_indices=np.arange(n_bins),
    n_permutations=n_permutations,
    seed=seed,
)
```

4. **Write verdict JSON via `_to_jsonable`** (NaN → null, ndarray → list):

```python
verdict_path = Path(out_path)
heatmap_path = verdict_path.with_name(verdict_path.stem + "_heatmap.json")

verdict = {
    # ... existing keys ...
    "corrected_pipeline_p57": _format_estimate(corrected_v2.point_estimate, corrected_v2.ci_lower, corrected_v2.ci_upper),
    "v1_reference_p57": v1_p57,  # historical v1 verdict from existing JSON, for comparison
    "v1_reference_path": str(v1_path) if v1_path.exists() else None,
    "fold_metadata": _to_jsonable(corrected_v2.fold_metadata),
    "diagnostic_heatmap_path": str(heatmap_path.relative_to(verdict_path.parent)),
    "verdict_basis": "pipeline_loso_v2",
    # verdict gating logic uses corrected_v2.point_estimate / ci_upper as before
}
verdict_path.write_text(json.dumps(verdict, indent=2))
heatmap_path.write_text(json.dumps(_to_jsonable({
    **heatmap,
    "bin_labels": [f"Q{i+1}" for i in range(len(heatmap['bin_indices']))],
}), indent=2))
```

5. **Update verdict CLI stdout output**: print fold-by-fold rho_pair_per_bin + stability warnings (e.g., a small markdown table). Keep this lightweight — not a critical output, just useful for the operator.

- [ ] **Step 1: Write failing test for v2 schema**

Add to `tests/scripts/test_run_falsification_harness.py`:

```python
def test_v2_verdict_json_includes_fold_metadata_and_heatmap_path(tmp_path, monkeypatch):
    """Run harness on tiny synthetic data; verdict JSON has v2 keys."""
    # ... use existing test fixtures + assert keys present.
    out_path = tmp_path / "verdict.json"
    run_harness(
        profiles_paths=[tmp_path / "profiles.parquet"],
        pa_paths=[tmp_path / "pa.parquet"],
        out_path=out_path,
        n_bootstrap=10,
        ...
    )
    verdict = json.loads(out_path.read_text())
    assert "corrected_pipeline_p57" in verdict
    assert "fold_metadata" in verdict
    assert len(verdict["fold_metadata"]) >= 1
    assert "rho_pair_per_bin" in verdict["fold_metadata"][0]
    assert "diagnostic_heatmap_path" in verdict
```

- [ ] **Step 2: Run failing**

- [ ] **Step 3: Implement** — refactor `run_harness` body. Must keep the verdict gating logic (`HEADLINE_BROKEN` / `HEADLINE_INCONCLUSIVE` / `HEADLINE_DEFENDED`) intact, just feeding from the v2 `corrected_pipeline_p57`.

- [ ] **Step 4: Run passing + spot check on existing data**

`UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_falsification_harness.py --profile-glob 'data/simulation/profiles_seed*_season*.parquet' --pa-glob 'data/simulation/pa_predictions_seed*_season*.parquet' --out data/validation/falsification_harness_v2_dryrun.json --n-bootstrap 30`

Expected: writes the JSON; prints fold-by-fold rho_pair_per_bin. Numbers may be wide-CI at n_bootstrap=30; that's fine for spot-check.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(scripts): run_falsification_harness v2 — fold-local params + diagnostic heatmap"
```

---

### Task 7: Re-run harness on existing Task 13 data; capture v2 verdict

**Files:** none modified (this task runs the script + writes outputs to `data/validation/`)

- [ ] **Step 1: Verify existing data**

```bash
ls data/simulation/profiles_seed*_season*.parquet | wc -l
ls data/simulation/pa_predictions_seed*_season*.parquet | wc -l
```

Expected: 24 profile parquets + 24 PA parquets (5 seasons × 24 seeds, but pivoted-merged shape may vary — check Task 13 layout).

- [ ] **Step 1.5: Smoke run at low n_bootstrap** (Codex round 1: prove the pipeline works before burning a 30-60min full run)

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_falsification_harness.py \
  --profile-glob 'data/simulation/profiles_seed*_season*.parquet' \
  --pa-glob 'data/simulation/pa_predictions_seed*_season*.parquet' \
  --out /tmp/v2_smoke.json \
  --n-bootstrap 30 \
  --n-permutations 30
```

Expected: completes in ~2-5 min. Stdout shows fold-by-fold rho_pair_per_bin vectors; verdict JSON has all expected keys; heatmap JSON written. If anything errors, fix before Step 2.

- [ ] **Step 2: Run v2 harness at production n_bootstrap**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_falsification_harness.py \
  --profile-glob 'data/simulation/profiles_seed*_season*.parquet' \
  --pa-glob 'data/simulation/pa_predictions_seed*_season*.parquet' \
  --out data/validation/falsification_harness_v2_$(date +%Y-%m-%d).json \
  --n-bootstrap 300 \
  --n-permutations 300
```

Expected: ~30-60 min runtime. Stdout prints fold-by-fold params + verdict.

- [ ] **Step 3: Spot-check verdict numbers**

```bash
cat data/validation/falsification_harness_v2_$(date +%Y-%m-%d).json | jq '.corrected_pipeline_p57, .verdict, .fold_metadata[].rho_pair_per_bin, .fold_metadata[].stability.small_sample_warning'
```

Verify:
- Verdict is one of HEADLINE_BROKEN / HEADLINE_INCONCLUSIVE / HEADLINE_DEFENDED
- Per-fold rho_pair_per_bin vectors look stable across folds (no fold has wildly different signs from the others)
- No fold has `small_sample_warning: true` (would indicate too few groups for stable tau)

- [ ] **Step 4: Commit data artifacts**

```bash
git add data/validation/falsification_harness_v2_*.json data/validation/falsification_harness_v2_*_heatmap.json
git commit -m "data: v2 verdict on Task 13 backtest (24 seeds × 5 seasons)"
```

---

### Task 8: comparison memo + SOTA tracker update

**Files:**
- Create: `docs/sota_audit/2026-05-02-harness-v2-comparison.md`
- Modify: `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md`

**Why:** v1 vs v2 comparison memo is the artifact that tells future-you (and any downstream production-deploy decision) what the verdict number now is and how to read it. SOTA tracker updates close out Issue #7 cleanly.

- [ ] **Step 1: Write comparison memo**

Template:

```markdown
# Falsification Harness v2 — verdict comparison memo (2026-05-02)

## Verdict comparison

| Metric | v1 | v2 | Change |
|---|---|---|---|
| corrected_pipeline_p57 | 0.0083 [0, 0.0375] | <FILL> | <delta> |
| in-sample corrected (build_corrected_mdp_policy) | 0.1183 | <FILL> | <delta> |
| in-sample vs CV gap | +0.1100 | <FILL> | <closes? widens?> |
| verdict | HEADLINE_BROKEN | <FILL> | |

## Per-fold rho_pair_per_bin

[table or short narrative]

## Diagnostic heatmap takeaway

[one-paragraph summary of the 5x5 heatmap; whether Q4 negative gap survives at fold-local estimation]

## Production policy implication

[recommendation: ship corrected policy / keep v1 production / pause for v3]
```

- [ ] **Step 2: Update SOTA tracker**

In `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md`, find Area #15 (PA + cross-game dependence). Mark as v2-shipped with a one-line summary including the v2 verdict number.

- [ ] **Step 3: Commit**

```bash
git add docs/sota_audit/2026-05-02-harness-v2-comparison.md docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md
git commit -m "docs: harness v2 comparison memo + tracker update"
```

---

## Final review (after all 8 tasks)

- [ ] **Run full test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m "not slow" -q
```

Expected: 990+ passing, 0 failing.

- [ ] **Send all v2 code changes to Codex for adversarial review**

```bash
git diff main..HEAD -- src/ scripts/ tests/ > /tmp/codex-v2-diff.patch
```

Then submit via consulting-codex skill (gpt-5.5, sandbox=read-only, /tmp). Frame: "What's wrong with this v2 implementation? Particularly look for: per-fold parameter estimation correctness, per-bin rho_pair indexing bugs, heatmap diagnostic shape errors, fold_metadata serialization issues."

Apply any catches from Codex review as a follow-up commit.

- [ ] **Open PR**

```bash
gh pr create --title "feat: harness v2 — per-bin rho_pair + within-fold parameter estimation" \
  --body-file <(cat <<EOF
## Summary
Closes #7. v1 verdict was HEADLINE_BROKEN; v2 tightens it via per-rank-1-bin rho_pair correction + fold-local dependence-parameter estimation. v2 verdict: <FILL>.

## What changed
[bulleted list of the 8 tasks]

## Test plan
- [x] 990+ tests pass
- [x] Codex round 5 review applied
- [x] Verdict JSON schema validated

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)
```

---

## Notes for the implementer subagent

- Each task starts with a failing test. Don't skip the red→green→refactor cycle even when "obvious."
- Backward compatibility: the v1 scalar paths must continue to work. Don't delete the scalar branches; they're sensitivity-analysis tools.
- When extending `DROPEResult` to carry `fold_metadata`, prefer adding the field as `Optional[list]` rather than mutating after construction. If the dataclass is `frozen=True`, unfreeze; if `slots=True`, add to slots.
- Numeric tolerance in tests: use `1e-9` for exact computations (no Monte Carlo), `0.02` for permutation/bootstrap statistics at n=100, `0.005` at n=1000.
- `pa_df` columns (per Task 13 layout): `season`, `date`, `batter_game_id`, `pa_num`, `p_pa`, `actual_hit`. Verify with one of the existing parquets before assuming.
- `profiles` columns (per Task 13 layout): `season`, `date`, `seed`, `top1_p`, `top1_hit`, `top2_p`, `top2_hit`. Same verification.
