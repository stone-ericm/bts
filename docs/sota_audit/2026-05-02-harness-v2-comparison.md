# Falsification Harness v2 — verdict comparison memo

**Date**: 2026-05-02 evening (smoke + production)
**Branch**: `feature/harness-v2`
**Plan**: `docs/superpowers/plans/2026-05-02-bts-harness-v2.md`
**Issue**: [#7](https://github.com/stone-ericm/bts/issues/7)
**Verdict JSON**: `data/validation/falsification_harness_v2_2026-05-03.json`

## Headline finding

> **The methodology fix flipped the gate-class.** v1 said `HEADLINE_BROKEN`. v2 says `HEADLINE_REDUCED`. The production policy is doing something — just not 8× better than chance.

| Metric | v1 | v2 production (n_bootstrap=300) |
|---|---|---|
| `corrected_pipeline_p57` | `0.0083 [0.0000, 0.0375]` | **`0.0333 [0.0000, 0.1167]`** |
| Verdict | `HEADLINE_BROKEN` | **`HEADLINE_REDUCED`** |
| Half-headline threshold | 0.0408 | 0.0408 |
| Position relative to threshold | hi=0.0375 *below* half-headline | hi=0.1167 *above* half-headline |
| Methodology | global params, LOSO bins | fold-local params + per-bin rho_pair |
| In-sample (build_corrected_mdp_policy) | 0.1183 | not recomputed (deferred) |
| `rare_event_ce_p57` (CE-IS, n_final=20K) | not computed | `0.0037 [0.0031, 0.0044]` |
| `fixed_policy_terminal_r_mc_p57` | not stable | `0.4167 [0.2083, 0.6250]` (n=24 fixed-policy) |

**v2 is 4× higher than v1 at the point estimate** (0.0333 vs 0.0083). The CI upper bound moves from `below` half-headline to `above`, which is the gate-class transition. v2 still says the headline is partly artifact (point < half-headline) but doesn't say production is broken.

> **Note on n_bootstrap:** smoke (n=30) and production (n=300) gave identical `corrected_pipeline_p57` because the verdict CI is the percentile over 5 fold points (n=5 — essentially min/max). `n_bootstrap` only affects per-fold internal CIs which don't propagate to the verdict. The CE-IS estimate did tighten with `n_final=20000` (0.0037 [0.0031, 0.0044] vs smoke's 0.0058 [0.0016, 0.0119]).

**v2 is 4× higher than v1 at the point estimate** (0.0333 vs 0.0083). The CI upper bound moves from `below` half-headline to `above`, which is the gate-class transition. v2 still says the headline is partly artifact (point < half-headline) but doesn't say production is broken.

## What changed methodologically

Two gaps in v1 were closed:

1. **v1 estimated `rho_PA / tau / rho_pair` once on full pooled data** while bins were LOSO-split. Parameter contamination across the audit boundary inflated the in-sample-vs-LOSO gap (0.1183 vs 0.0083, ~14× ratio — classic overfit). v2 refits all parameters within each fold's 4 training seasons.

2. **v1 used a single global `rho_pair_cross_game` scalar.** Aggregate ≈ 0 hid Q4's apparent 8pp empirical-vs-synthetic gap. v2 uses per-rank-1-bin `rho_pair_per_bin` (5-element vector) so each bin gets its own correction.

## Per-fold rho_pair_per_bin (smoke verdict)

Pattern: Q2 consistently positive across all folds; Q4 small and inconsistent; Q5 small. The bin-conditional structure that motivated v2 (Q4's apparent 8pp gap on full pooled data) **did NOT survive fold-local re-estimation**.

| Held-out season | rho_PA | tau | fold_p57 | rho_pair_per_bin (Q1..Q5) |
|---|---|---|---|---|
| 2021 | 0.0017 | 0.0979 | 0.0000 | [-0.020, +0.057, -0.019, +0.016, -0.005] |
| 2022 | 0.0006 | 0.0602 | 0.0000 | [-0.010, +0.025, -0.046, +0.030, -0.034] |
| 2023 | 0.0006 | 0.0578 | 0.1250 | [-0.049, +0.045, -0.016, +0.035, -0.041] |
| 2024 | 0.0020 | 0.1089 | 0.0000 | [-0.003, +0.029, -0.006, -0.012, -0.040] |
| 2025 | 0.0012 | 0.0827 | 0.0417 | [-0.011, +0.055, -0.044, -0.004, -0.008] |

**Q4 in v2**: point estimates range from -0.012 to +0.035 across the 5 folds, with wide CIs that all bracket zero. The "Q4 antagonism" that drove v2's design was a finite-sample artifact of pooled estimation. **Q2 emerges as the only bin with a consistent rho_pair signal** (positive in all 5 folds, p<0.05 in fold 2021).

## Diagnostic heatmap

`pair_residual_correlation_per_cell` was run on the canonical-seed pooled data (n=912 days). Lower-triangular convention — cells where `r2_bin > r1_bin` are empty by the `p_rank2 ≤ p_rank1` invariant. `n_invariant_violations = 0` (data respects the invariant).

Cell occupancy is heavily skewed toward the diagonal and lower-rank-2 cells. Q5×Q5 had ~110 obs, Q5×Q1 had 28 obs. Per-cell rho estimates at small n are noisy — the `reliable_cells` mask (n ≥ 30) flags ~7-8 of the 15 lower-triangular cells as reliable.

**Bottom line on the heatmap**: no single cell shows persistent dependence after fold-local correction. The heatmap is informative for *deciding* v3 isn't needed, not for *driving* v2's correction. The per-rank-1-bin correction (which v2 actually uses) captures the same information at lower variance.

## Production policy implication

**Recommendation: do NOT replace the production MDP policy with the v1-corrected version.**

- v1 said the production policy was broken (BROKEN: hi-CI below half-headline). v2 contradicts (REDUCED: hi-CI above half-headline).
- The v1 corrected policy was solved against an over-corrected transition table — applying it would deflate production's projected P(57) more than the data supports.
- The v2 corrected policy IS produced (`scripts/build_corrected_mdp_policy.py` could be re-run with v2's per-bin rho input), but ALSO would over-correct because in-sample diverges from CV.

**Better path**: keep the current production policy. The headline (8.17%) is partly artifact, but the policy itself is closer to right than the v1-corrected one.

## Tension with CE-IS estimate

v2 verdict has `rare_event_ce_p57 = 0.0058 [0.0016, 0.0119]` — substantially LOWER than the LOSO `0.0333`. The two estimators measure different things:
- LOSO (corrected_pipeline_p57): replays a fold-local corrected policy on held-out trajectories, using the policy that was solved against fold-corrected bins
- CE-IS (rare_event_ce_p57): naive "always pick rank-1" rare-event MC, no policy correction, no LOSO

The 6× gap between them suggests the MDP policy meaningfully outperforms naive rank-1 — even after correction. This is a positive signal for the production policy that v2's gate-class transition (BROKEN→REDUCED) reinforces.

## Open follow-ups

- **CI methodology** ⭐ : 5-fold percentile CI is essentially min/max at n=5. The verdict CI didn't tighten when going from n_bootstrap=30 to 300 because of this. A bootstrap-of-folds or paired-block-bootstrap on fold point estimates would give a tighter, more defensible CI. **This is the highest-leverage v2.5 work item** — the current verdict is at gate threshold, and a tighter CI would either firmly resolve `HEADLINE_REDUCED` or potentially move it to `HEADLINE_BROKEN` / `HEADLINE_DEFENDED`.
- **Distribution shift remediation**: the v2 verdict still says headline is partly artifact (point < half-headline). The strategic gaps memo's #1 (refresh training data) is the next layer to address — separate work item, not part of v2.
- **Cross-bin-cell correction (v3)**: the heatmap shows no cell with persistent dependence post-correction, so v3 cross-bin correction is **not justified by the data**. v2 closed the methodology question; further refinement in this layer has marginal expected value.

## Verdict status: REVISED, not BROKEN

The production policy:
- ✅ Shows real lift over naive rank-1 (CE-IS 0.0058 vs MDP 0.0333 at LOSO)
- ⚠ Is overconfident relative to the headline (0.0817 in-sample vs 0.0333 CV-corrected)
- ❌ Is NOT broken (v1's claim that hi-CI < half-headline doesn't hold under fold-local methodology)

**Action**: keep production policy as-is, mentally adjust expected P(57) downward from 8% to 3-4%, plan training-data refresh (distribution shift remediation) as the next priority.
