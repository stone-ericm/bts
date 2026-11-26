# Falsification Harness v2 — verdict comparison memo

**Date**: 2026-05-02 evening (smoke + production)
**Branch**: `feature/harness-v2`
**Plan**: `docs/superpowers/plans/2026-05-02-bts-harness-v2.md`
**Issue**: [#7](https://github.com/stone-ericm/bts/issues/7)
**Verdict JSON**: `data/validation/falsification_harness_v2_2026-05-03.json`

## Headline finding (revised after Codex round 1 review)

> **Conservative claim, supported by data:** v1's `HEADLINE_BROKEN` verdict does not reproduce under v2. The v2 verdict is `HEADLINE_REDUCED` — but the gate-class transition is CI-driven, not point-estimate driven.

| Metric | v1 | v2 production (n_bootstrap=300) |
|---|---|---|
| `corrected_pipeline_p57` | `0.0083 [0.0000, 0.0375]` | `0.0333 [0.0000, 0.1167]` |
| Verdict | `HEADLINE_BROKEN` | `HEADLINE_REDUCED` |
| Half-headline threshold | 0.0408 | 0.0408 |
| Position relative to threshold | hi=0.0375 *below* half-headline (BROKEN) | **point=0.0333 still below half-headline; only hi=0.1167 is above (REDUCED)** |
| Methodology | global params, LOSO bins, single corrected policy | fold-local params + per-bin rho_pair, fold-local corrected policies |
| In-sample (build_corrected_mdp_policy) | 0.1183 | **NOT recomputed in v2** — see open question |
| `rare_event_ce_p57` (CE-IS, n_final=20K) | not computed | `0.0037 [0.0031, 0.0044]` |
| `fixed_policy_terminal_r_mc_p57` | not stable | `0.4167 [0.2083, 0.6250]` (n=24 fixed-policy) |

**The honest read:** v2's point estimate (0.0333) is *still below* half-headline. The reason the verdict isn't BROKEN is that the upper-CI bound (0.1167) widened enough that we can no longer **certify** BROKEN. This is "uncertainty got large enough that BROKEN is no longer guaranteed," not "production policy was rehabilitated."

> **Note on n_bootstrap:** smoke (n=30) and production (n=300) gave identical `corrected_pipeline_p57` because the verdict CI is the percentile over 5 fold points (n=5 — essentially min/max). `n_bootstrap` only affects per-fold internal CIs which don't propagate to the verdict. The CE-IS estimate did tighten with `n_final=20000` (0.0037 [0.0031, 0.0044] vs smoke's 0.0058 [0.0016, 0.0119]).

## CI fragility — what's actually happening

The 5 fold point estimates are: `[0.0, 0.0, 0.125, 0.0, 0.0417]`. Three folds returned exactly zero successes; two had small positive rates. The percentile CI on n=5 values is essentially the min/max, plus interpolation noise. This is **not a defensible uncertainty interval** for the underlying probability.

**Binomial perspective**: the held-out replays produced ~4 successes out of ~120 trajectory-attempts (5 folds × 24 seeds). Naive binomial 95% interval on 4/120 ≈ [0.009, 0.083]. That's a tighter, more defensible interval than the fold-percentile [0, 0.117], but it ignores fold-level variation. **Either bound contains 0.0408 (half-headline)**, so neither cleanly classifies as BROKEN or REDUCED. The current verdict is at the gate threshold.

## What changed methodologically (and what we DO NOT know)

Three things changed between v1 and v2:

1. **v1 estimated `rho_PA / tau / rho_pair` once on full pooled data** while bins were LOSO-split. Parameter contamination across the audit boundary. v2 refits all parameters within each fold's 4 training seasons.

2. **v1 used a single global `rho_pair_cross_game` scalar.** v2 uses per-rank-1-bin `rho_pair_per_bin` (5-element vector).

3. **v1 used a single pre-built corrected policy** replayed across folds. v2 solves a fold-local MDP per fold.

**What we do NOT know without ablations** (Codex round 1 catch): which of (1), (2), (3) is responsible for how much of the 4× point-estimate change. The fair claim is "under the v2 harness, the verdict moves from BROKEN to REDUCED" — NOT "the methodology fix flipped it." Without ablations (v1 policy under v2 replay; v2 solver with v1 global params; fold-local scalar rho_pair; etc.) we can't attribute. **Highest-priority v2.5 work item.**

## Per-fold rho_pair_per_bin (production verdict)

| Held-out season | rho_PA | tau | fold_p57 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|---|---|
| 2021 | 0.0017 | 0.0979 | 0.0000 | -0.020 | +0.057 | -0.019 | +0.016 | -0.005 |
| 2022 | 0.0006 | 0.0602 | 0.0000 | -0.010 | +0.025 | -0.046 | **+0.030*** | -0.034 |
| 2023 | 0.0006 | 0.0578 | 0.1250 | -0.049 | +0.045 | -0.016 | **+0.035*** | -0.041 |
| 2024 | 0.0020 | 0.1089 | 0.0000 | -0.003 | +0.029 | -0.006 | -0.012 | -0.040 |
| 2025 | 0.0012 | 0.0827 | 0.0417 | -0.011 | +0.055 | -0.044 | -0.004 | -0.008 |

\* = bootstrap CI excludes zero (CI semantically distinct from permutation p-value, see note below).

### Q4 finding (corrected after Codex round 1 review)

**My initial claim was wrong.** Q4 is NOT ≈0 across all folds. Q4 has positive estimates with bootstrap CIs that exclude zero in 2022 and 2023 (rho ≈ +0.03, CI [+0.010, +0.052] and [+0.013, +0.058]). The other three folds have Q4 near zero.

**The real finding**: Q4 sign **reverses** between v1 and v2. v1 (pooled) showed Q4 antagonism (negative gap). v2 (fold-local) shows Q4 *cooperative* in 2/5 folds (positive rho excluding zero). Either:
- The v1 pooled estimate was driven by a strong antagonistic signal in 2024/2025 (Q4 negative there) that v2 averages away with the cooperative seasons
- Or Q4's true dependence is genuinely heterogeneous across seasons

Both readings argue against a confident **global** Q4 antagonism correction (v1's framing). Neither says Q4 has zero dependence universally.

### p-value vs CI tension (defined here, not just casually invoked)

The reported `rho_pair_per_bin_p_value` is from a **two-sided permutation test** (n_permutations=300): `mean(|null_dist| >= |observed|)` under shuffles of e2 within bin. CIs are from a separate paired bootstrap. The two can disagree because:
- Permutation test is two-sided around 0; CIs are around the point estimate
- Permutation null can have heavy tails making absolute test statistic insensitive
- n_permutations=300 has minimum detectable p of ~0.003

**The CI is the more reliable signal** for "is rho different from zero in this fold." The p-values shown should be read as "didn't reach permutation significance" rather than "no effect."

## Diagnostic heatmap

`pair_residual_correlation_per_cell` was run on the canonical-seed pooled data (n=912 days). Lower-triangular convention. `n_invariant_violations = 0`. ~7-8 of the 15 lower-triangular cells have n ≥ 30 (the `reliable_cells` mask threshold).

The heatmap shows no single cell with persistent dependence post-correction across the reliable subset. This argues against v3 cross-bin-cell correction being justified by current data.

## Production policy implication (revised after Codex round 1)

**Recommendation: do NOT replace the production MDP policy YET — but the reasons are weaker than I initially claimed.**

The v1-corrected policy is still inappropriate (it's based on global parameter estimation contaminated across the audit boundary). But my initial argument that "v2 corrected policy would also over-correct because in-sample diverges from CV" is **not established** — v2 in-sample was not recomputed in this round. Without v2 in-sample numbers, we can't say whether the in-sample-vs-CV gap closed or persists.

**The defensible recommendation**:
- Don't apply v1's corrected policy to production (v1's contamination story is real)
- Don't yet apply v2's corrected policy without first running same-evaluator policy comparisons (v1 policy vs v2 policy on identical held-out trajectories with identical replay mechanics)
- Recompute v2 in-sample to confirm the in-sample-vs-CV gap behavior changed

Until those two ablations land, "keep current production policy" is the safest move, but it should be framed as "wait for the comparison" not "v2 also over-corrects."

## CE-IS vs LOSO — gap is suggestive but underidentified (Codex round 1)

v2 production verdict has `rare_event_ce_p57 = 0.0037 [0.0031, 0.0044]` — substantially LOWER than the LOSO `0.0333`. My initial claim that this gap proves MDP policy lift over naive rank-1 was **overclaim**.

The two estimators differ in: policy (CE-IS uses always-pick-rank-1, LOSO uses fold-local corrected MDP), data split (CE-IS uses pooled canonical-seed data, LOSO uses per-fold held-out), replay mechanics, denominators, terminal-r MC variance behavior, and event definition.

The gap is **consistent with** policy lift, but not isolation evidence for it. To confirm policy lift, run both policies through the same evaluator on the same held-out folds with the same replay mechanics (a v2.5 ablation).

## Open follow-ups (priority-ordered after Codex round 1)

1. **Attribution ablations** ⭐ : run minimal-pair ablations to isolate which v1→v2 change drove the verdict shift. Specifically: v1 policy under v2 replay (controls solver/replay path); v2 with global rho_pair scalar (controls #2 from "What changed"); v2 with v1 global params but per-bin rho_pair (controls #1). Without these, we know "v2 ≠ v1 verdict" but not why.

2. **v2 in-sample recompute** ⭐ : recompute the in-sample corrected P(57) using v2's per-bin rho_pair via `build_corrected_transition_table`. v1 in-sample was 0.1183. If v2 in-sample is also high (>0.05 say), the in-sample-vs-CV gap is mostly the same — meaning fold-local estimation didn't fully close the overfit. If v2 in-sample is closer to v2 LOSO (0.03), v2 actually closed the gap. **This is the cleanest single test of v2's methodological claim.**

3. **Same-evaluator policy comparison**: run v1 policy and v2 policy on identical held-out trajectories with identical replay. If they produce ≈ same P(57), policy doesn't matter; if different, we can quantify lift.

4. **CI methodology**: 5-fold percentile CI is not adequate. Block-bootstrap or hierarchical bootstrap that separates season variation from within-season trajectory uncertainty. Also report binomial interval over pooled held-out successes (4/120) as a sanity check.

5. **Distribution shift remediation**: a higher-leverage strategic question — the v2 verdict still says headline is partly artifact (point < half-headline). Refresh training data per `project_bts_strategic_gaps_2026_04_30.md` item #1.

## Verdict status (revised)

The production policy under v2 evaluation:
- ⚠ Not affirmatively rehabilitated (point estimate 0.0333 still below half-headline 0.0408)
- ⚠ Not affirmatively broken (CI upper 0.1167 above half-headline)
- ❓ Genuine signal vs noise unresolved (5-fold CI is essentially min/max; need block-bootstrap or binomial interval to firm up)
- ❓ Methodology attribution unresolved (don't know which of v2's three changes drove the gate-class transition)

**Honest action**: keep production policy. Do NOT broadcast v2 as "production rehabilitated." Schedule v2.5 ablations + in-sample recompute as the next priority, ahead of distribution-shift remediation.
