# Swing-screen Escalation — Design Spec

**Date:** 2026-06-14
**Status:** DRAFT (Codex-designed gpt-5.5; build on `swing-escalation` worktree)
**Author:** Claude (Opus 4.8) + Codex, delegated by Eric

## Why

The amendment-#3 residual-stacking swing screen ran 900/900 (2026-06-13) and
**failed its power gate**: the soft-oracle (a ~+0.005 calibrated leak) came back
+0.0028 vs a permuted-null band of 0.0171 (p=0.245) → **underpowered**, families
(~+0.003) below the noise floor. Verdict: inconclusive, not a discovery.

Power math: per-day rank-AUC noise gives null ≈ 0.0171 at 88 eval days, scaling
~1/√(days). Even using ALL covered-era days (~520) → null ≈ 0.007, still above the
candidate ≈0.003–0.005. So **more days alone cannot fix it.** Codex's two levers:

## The escalation (Codex recommendation)

### 1. Statistic — the main power lever
Replace the **equal-weighted mean of daily rank-AUC** with a **pair-count-weighted
within-day AUC delta**:

```
T = Σ_day [ within-day (pos,neg) pair-wins(arm) − pair-wins(baseline) ]
        / Σ_day [ within-day (pos,neg) pairs ]
```

This preserves the daily-ranking target (still within-day, not a cross-day pooled
AUC — which would test the wrong thing) but stops tiny-slate days from dominating
the variance. Report season-stratified; significance via week-blocked sign
permutation (or block wild bootstrap) on the paired per-day contributions.

### 2. Walk-forward, past-only refit (not a frozen window)
Replace the frozen train(2019-01..2024-06)→eval(2024-07..11) with a **walk-forward
over 2024-07 → latest 2026**, refit per weekly/monthly fold:
- prod-prior trained strictly on data before the fold;
- covered-layer training inputs via nested past-only OOF;
- baseline + swing arms trained on past covered rows only;
- coverage + rolling features computed as-of game time only;
- score the next fold identically for baseline and candidate.

Avoids the staleness of a 2024-frozen model predicting 2026 (which would add
noise that swamps a +0.003 effect) and matches the deployable question. 2025/2026
are the decision-grade portion.

## Decision rule (pre-registered)
1. **Re-run the soft-oracle through the NEW pipeline first.** Count the screen as
   powered ONLY if the soft-oracle: delta ≈ near the injected size (`≥ +0.004` for
   a +0.005 sentinel), clears the blocked permutation test (one-sided p<0.05), and
   the null half-width is `≤ 0.005–0.006`.
2. **Then the real candidate is signal** iff: pair-count-weighted within-day delta
   `≥ ~+0.003`, one-sided blocked p clears the pre-set threshold (after family
   correction), and season/half-season estimates aren't driven by one short block.
3. **Final negative** if the oracle passes but the candidate is ~0 to +0.002 /
   non-significant / its upper CI excludes a practical +0.005.
4. **Honest backstop:** if the soft-oracle STILL fails under the stronger
   statistic + walk-forward, the conclusion is **"the covered era cannot resolve a
   +0.003–0.005 daily-rank-AUC effect with this screen — stop chasing variants"**
   (not "no swing effect"). This is a valid, decision-useful outcome.

## Implementation (build on the amendment-#3 base, `ed126a6`)
- `src/bts/experiment/swing_screen.py`:
  - Add `paired_within_day_auc_delta(per_day_arm, per_day_base)` (pair-count-weighted)
    + per-day pair counts (n_pos·n_neg) already derivable from the slate.
  - Add a walk-forward driver path: fold the covered era; per fold, reuse
    `build_prod_prior_oof` (past-only) + `run_residual_arm` scoring the fold.
  - Keep the existing frozen path for comparison/back-compat; gate the new path
    behind a `--mode walkforward` (or a new driver).
- `scripts/swing_screen_residual_driver.py`: window = 2024-07→2026, fold cadence
  flag; emit per-day {day, season, arm/base within-day pairs, pair-wins}.
- `scripts/swing_screen_report.py`: compute T (pair-count-weighted) + the
  decision-rule gates above; soft-oracle power gate first.
- TDD: unit-test the pair-count weighting (a synthetic slate where equal-weight
  and pair-count-weight diverge), and the walk-forward leakage guard (no fold
  trains on its own/future dates).
- Codex-review the implementation before any box run.

## Out of scope
- Changing the candidate feature set (same swing families).
- Any ship decision (Eric's call; this only produces a powered verdict).

## Risks
- Walk-forward leakage (per-fold past-only is the guard; unit-test it).
- The pair-count weighting must still be a *within-day* statistic (not pooled
  cross-day) — assert/test that.
- Compute: walk-forward refits per fold are heavier than the frozen single fit;
  size the box + fold cadence accordingly.
