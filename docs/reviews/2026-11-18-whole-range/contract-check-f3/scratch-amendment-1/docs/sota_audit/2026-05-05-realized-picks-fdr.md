# Realized-picks FDR baseline — SOTA #7 P0

**Date**: 2026-05-05
**Branch**: `feature/realized-picks-fdr`
**Predecessors**: [α P0 attribution](./2026-05-05-realized-picks-attribution.md), [α P1 attribution](./2026-05-05-realized-picks-attribution-p1.md)
**Tracker**: [SOTA audit tracker](../superpowers/specs/2026-05-01-bts-sota-audit-tracker.md), area #7
**Module under test**: `bts.validate.fdr`
**Application script**: `scripts/run_realized_picks_fdr.py`
**Output artifact**: `data/validation/realized_picks_fdr_2026-05-05.json`
**Scope**: docs + new module + new script + JSON artifact. Not a deploy authorization. **p-value FDR baseline only — NOT e-BH.**

> Per Codex agent-bus #225 / #227: PR #22 flagged a calibration warning at the post_pooled_mdp_pre_bpm × DD × not_park_driven × Q4 cell (n=8) and explicitly disclaimed "multi-comparisons context is unfavorable" without correcting for it. This memo closes that methodology gap by computing per-cell BH/BY-adjusted q-values over the Cut C family. e-BH was the initial proposal but was rejected on validity grounds: 1/p has infinite expectation under Uniform(0,1) null and is NOT a valid universal p-to-e calibrator (Wang & Ramdas 2022). Genuine e-values remain deferred until either a likelihood-ratio construction with a prespecified alternative, or a documented calibrator family such as κ·p^(κ-1), is introduced.

## Headline

**No cell in the Cut C family achieves FDR-adjusted significance at any conventional threshold.** The most extreme cell — DD × not_park_driven × Q4 in the post_pooled_mdp_pre_bpm stratum (n=8, 3 hits, p_two_sided = 0.0734) — has q_BH = q_BY = 1.000 under family size m = 22.

The arithmetic is direct: BH q_(1) = (m / 1) × p_(1) = 22 × 0.0734 = 1.61, capped at 1.000. All other cells inherit ≥ this via the cumulative-min step. BY is strictly ≥ BH, also 1.000.

This is the right result. PR #22 had to soften "calibration warning" framing because the n=8 cell with mean_p outside Wilson CI was a single post-hoc-selected cell from a multi-cell cut. With FDR adjustment now applied, the warning remains a watch item — there is no cell that crosses any conventional FDR cutoff (0.05, 0.10, 0.20). The DD × Q4 cell stays a watch item in the same way.

## Methodology

### Why BH/BY (and not e-BH)

The Wang & Ramdas (2022) e-BH procedure provides anytime-valid FDR control via e-values, but it requires e-values where E_H0[e] ≤ 1 (Markov bound for FDR). The naive 1/p calibration violates this constraint: under a Uniform(0,1) null, E[1/p] = ∞. The paper does discuss valid p-to-e calibrators (e.g., the family f_κ(p) = κ·p^(κ-1) for κ ∈ (0,1)) but these introduce loss of power and require explicit choice of κ. e-BH on a chosen calibrator family is a possible follow-up; it is NOT this P0.

This P0 is a **p-value FDR baseline**:
- Per-cell p-values from a Poisson-binomial null (heterogeneous Bernoulli rates).
- Family-wise adjustment via BH (Benjamini-Hochberg 1995) for PRDS dependence and BY (Benjamini-Yekutieli 2001) for arbitrary dependence (with c(m) = Σᵢ 1/i harmonic penalty).

### Cell p-values via Poisson-binomial

Each cell C contains rows {(p_i, y_i)} where p_i = p_game_hit and y_i = actual_hit. Under H0 = "the model's predictions are calibrated for this cell," the observed total hits X = Σ y_i follows a Poisson-binomial distribution with parameters (p_1, ..., p_n).

This is strictly more informative than treating the cell as iid Bernoulli with mean = mean_p. Wilson intervals on the realized rate (used in PR #22) lose the row-level prediction information: they ask "is the realized rate consistent with some unknown rate?" while the Poisson-binomial test asks "is the realized count consistent with the *specific* row-level predictions the model emitted?"

For each cell:
- p_lower = P(X ≤ x | H0) — the **overconfidence tail** (observed hits LOW vs expected ⇒ model overconfident on this cell).
- p_upper = P(X ≥ x | H0), inclusive via sf(x − 1).
- p_two_sided = min(1, 2 · min(p_lower, p_upper)) — standard discrete double-the-smaller convention; not an exact-optimal two-sided test.
- tail_direction = "overconfidence" iff p_lower < p_upper, "underconfidence" iff p_upper < p_lower, else "balanced".

Implementation uses `scipy.stats.poisson_binom` (scipy 1.17+) for exact PMF/CDF.

### Tail-direction labeling

Per Codex's correction (#227): overconfidence ⇔ observed hits LOW vs expected ⇔ p_lower < p_upper. Concretely: the model predicted high probabilities, the batters didn't hit, the lower tail (low observed counts) is the more extreme one, and that's the overconfidence direction.

### Family scope

Tested family = ALL non-empty Cut C cells across ALL three regimes (post_bpm, post_pooled_mdp_pre_bpm, pre_pooled_mdp), keyed on (regime, slot, is_park_driven, batter_skill_quartile). Excluded:
- 6 pending rows (result_status ≠ "resolved").
- 19 NA-key rows (skill-NA in pre_pooled_mdp, early-season picks below MIN_PRIOR_PA).

Net family size **m = 22** non-empty cells.

The family scope choice ("all reported cells, all regimes") is per Codex #227: the tested family is what the memo reports. Restricting to a single regime would deflate m and produce less-conservative q-values, but at the cost of FDR validity over the full reported set. We report the conservative thing.

## Per-cell q-values

```
regime                       slot          env              Q   n hits     p2   q_bh   q_by tail
------------------------------------------------------------------------------------------------------
post_bpm                     double_down   not_park_driven  Q3  1    0  0.5305 1.0000 1.0000 overconfidence
post_bpm                     double_down   not_park_driven  Q4  2    2  1.0000 1.0000 1.0000 underconfidence
post_bpm                     primary       park_driven      Q3  1    0  0.4933 1.0000 1.0000 overconfidence
post_bpm                     primary       park_driven      Q4  1    1  1.0000 1.0000 1.0000 underconfidence
post_bpm                     primary       not_park_driven  Q3  1    1  1.0000 1.0000 1.0000 underconfidence
post_bpm                     primary       not_park_driven  Q4  1    1  1.0000 1.0000 1.0000 underconfidence
post_pooled_mdp_pre_bpm      double_down   park_driven      Q3  1    1  1.0000 1.0000 1.0000 underconfidence
post_pooled_mdp_pre_bpm      double_down   not_park_driven  Q1  1    0  0.5394 1.0000 1.0000 overconfidence
post_pooled_mdp_pre_bpm      double_down   not_park_driven  Q2  3    2  1.0000 1.0000 1.0000 overconfidence
post_pooled_mdp_pre_bpm      double_down   not_park_driven  Q3  2    2  1.0000 1.0000 1.0000 underconfidence
post_pooled_mdp_pre_bpm      double_down   not_park_driven  Q4  8    3  0.0734 1.0000 1.0000 overconfidence
post_pooled_mdp_pre_bpm      primary       park_driven      Q3  2    1  0.9069 1.0000 1.0000 overconfidence
post_pooled_mdp_pre_bpm      primary       park_driven      Q4  2    1  0.8852 1.0000 1.0000 overconfidence
post_pooled_mdp_pre_bpm      primary       not_park_driven  Q2  2    2  1.0000 1.0000 1.0000 underconfidence
post_pooled_mdp_pre_bpm      primary       not_park_driven  Q3  1    1  1.0000 1.0000 1.0000 underconfidence
post_pooled_mdp_pre_bpm      primary       not_park_driven  Q4  8    6  1.0000 1.0000 1.0000 overconfidence
pre_pooled_mdp               double_down   not_park_driven  Q2  1    0  0.5905 1.0000 1.0000 overconfidence
pre_pooled_mdp               double_down   not_park_driven  Q3  3    2  1.0000 1.0000 1.0000 overconfidence
pre_pooled_mdp               double_down   not_park_driven  Q4  3    3  0.8113 1.0000 1.0000 underconfidence
pre_pooled_mdp               primary       park_driven      Q3  1    1  1.0000 1.0000 1.0000 underconfidence
pre_pooled_mdp               primary       not_park_driven  Q3  1    1  1.0000 1.0000 1.0000 underconfidence
pre_pooled_mdp               primary       not_park_driven  Q4  3    1  0.2595 1.0000 1.0000 overconfidence
```

## Interpretation

### What the FDR adjustment tells us

The smallest two-sided p-value across 22 cells is 0.0734 (DD × not_park_driven × Q4 in the post_pooled_mdp_pre_bpm stratum). At family size m = 22, BH gives q_BH = min(1, 22 × 0.0734) = 1.000 for that cell, and ≥ 1.000 for all others. **No cell achieves any conventional FDR cutoff** (0.05, 0.10, 0.20) under either BH or BY.

This is consistent with PR #22's narrative caveat. The DD × Q4 cell is overconfident at the cell level (mean_p = 0.731 vs observed_rate = 0.375; p_two_sided = 0.0734) but does not survive multiple-comparisons adjustment at family size 22.

### The DD × Q4 watch item, post-FDR

PR #22 framed this cell as a "watch item" rather than a formal calibration test. The FDR analysis here doesn't change that framing — it adds a concrete number to it: under BH at m=22, q_BH = 1.000. The cell is still worth tracking as more post-bpm picks resolve, because:
- The n=8 sample is small enough that q_BH is dominated by the family-size penalty (m × p_(1)), not the cell's own evidence.
- If the cell grows to n=20+ at the same observed rate (~0.375), the corresponding p_two_sided would shrink to ~0.001 territory, at which point even BH at m=22 would yield q_BH ≈ 0.022 — significant at FDR 0.05.
- But that's a *forecast based on the observed rate persisting*, which is precisely what we're trying to test. We don't claim it; we wait.

### Tail-direction summary

11 cells lean overconfidence (p_lower < p_upper); 11 cells lean underconfidence. The split is even, suggesting no systemic directional bias at the cell level — consistent with "the model is roughly calibrated overall, with cell-level noise."

The five cells with the smallest two-sided p-values (most extreme):
1. post_pooled_mdp_pre_bpm × DD × not_park_driven × Q4: p2=0.0734, overconfidence (the watch item)
2. pre_pooled_mdp × primary × not_park_driven × Q4: p2=0.2595, overconfidence
3. post_bpm × primary × park_driven × Q3: p2=0.4933, overconfidence
4. post_bpm × DD × not_park_driven × Q3: p2=0.5305, overconfidence
5. post_pooled_mdp_pre_bpm × DD × not_park_driven × Q1: p2=0.5394, overconfidence

The first cell is the same DD × Q4 watch item PR #22 surfaced; everything else is far above any sensible cutoff even before FDR adjustment.

## What this memo does NOT say

- It does NOT propose a deploy change. q-values are reported; no binary deploy verdict is set per Codex #227 D.
- It does NOT close the e-BH or SAVI methodology lanes. Genuine e-values and sequential anytime-valid testing remain deferred (separate P).
- It does NOT change PR #22's substantive findings. The DD × Q4 watch item is unchanged in framing; the FDR adjustment confirms it numerically.
- It does NOT apply BH/BY to the v2.5/v2.6 ablation cells (those were 5 cells × multiple metrics, a separate family). That's a P1 follow-up if the methodology stack adopts FDR throughout.

## What this memo establishes

- A reusable `bts.validate.fdr` module with `bh_qvalues`, `by_qvalues`, `cell_pvalue` (Poisson-binomial under heterogeneous H0), and `cell_pvalues_from_artifact` for Cut C extraction. 25 unit tests cover boundary behavior, dtype validation, NaN rejection, hand-computed PMFs, and the iid-reduces-to-binomial sanity check.
- A standalone application script (`scripts/run_realized_picks_fdr.py`) that takes the canonical realized-picks artifact and produces a JSON with per-cell q-values plus methodology metadata (input file SHA, scipy version, git head, exclusion counts, generated_at). Reproducible from a single command:
  ```
  UV_CACHE_DIR=/tmp/uv-cache uv run --extra model python scripts/run_realized_picks_fdr.py
  ```
- A first-pass FDR-adjusted picture of Cut C: no cell crosses any conventional cutoff at family size m=22; the DD × Q4 watch item is confirmed as a watch item, not a formal finding.

## What's next (recommendations, not commitments)

1. **Re-run when post-bpm n grows.** The DD × Q4 cell's p_two_sided will tighten if the observed rate persists at ~0.375; whether it crosses an FDR threshold depends on family size at the time. The same script + canonical regeneration command produces a fresh FDR table.

2. **Consider valid-e-value path (e-BH with κ-calibrator) only if the cell-level signal grows.** At n=8, the family size penalty dominates regardless of method choice; the e-BH advantage (anytime-valid evidence accumulation) is most useful when individual cells start carrying real evidence. Defer.

3. **Apply to v2.5/v2.6 ablation cells (P1).** Those 5 cells × multiple metrics are reported with cell-level CIs but no family-wise adjustment. A follow-up PR could re-run BH/BY across that family. Out of scope here.

4. **Methodology-stack uplift: integrate FDR into the conformal-gate / OPE / CE-IS validators.** Each of those modules currently reports cell-level intervals; an FDR-aware reporting layer would prevent the "one cell out of many is suspicious" pattern from propagating into deploy decisions. Larger scoping; defer until a deploy candidate exists.

5. **No methodology-stack PRs needed for the α track.** This P0 closes the explicit methodology gap that PR #22 flagged. The α track stays parked at "infrastructure complete, awaiting n growth."
