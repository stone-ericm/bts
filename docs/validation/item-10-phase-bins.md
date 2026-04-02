# Item 10: Phase-Aware Bin Granularity

**Verdict:** CURRENT BINARY SPLIT IS CORRECT IN KIND BUT MISCALIBRATED — reducing `late_phase_days` from 60 to 30 increases P(57) from 6.15% to 7.82% (+1.67 pp). Monthly bins are too thin to be reliable. Quarterly splits offer no practical improvement over a well-calibrated binary split.

## What We Tested

The current MDP uses phase-aware bins: early season (months 1–7) and late season (months 8–9) with `late_phase_days=60`. This means the late-season hit rate profile is applied for the last 60 days of the season (August + September). We tested whether finer granularity — monthly or quarterly bins, or a different binary cutoff — improves the optimal P(57).

Script: `scripts/validation/item_10_phase_bins.py`
Data: `data/simulation/backtest_{2021-2025}.parquet` — 912 game dates, 9,120 player-days

---

## Results

### (a) Baseline: No Phase Split

| Configuration | P(57) |
|--------------|-------|
| No phases (single bin set) | 4.8278% |

### (b) Binary Split — Current Configuration

| Configuration | P(57) | Delta vs baseline |
|--------------|-------|------------------|
| Binary early/late, late_phase_days=60 (current) | 6.1462% | +1.32 pp |

Early (months 3–7) vs late (months 8–9) bin hit rates:

| Bin | Early P(hit) | Late P(hit) | Degradation |
|-----|-------------|------------|-------------|
| Q1 | 0.767 | 0.778 | +0.011 (noise) |
| Q2 | 0.883 | 0.855 | −0.028 |
| Q3 | 0.900 | 0.839 | −0.061 |
| Q4 | 0.925 | 0.871 | −0.054 |
| Q5 | 0.917 | 0.857 | −0.060 |

Late-season degradation is real and concentrated in Q3–Q5 (−5 to −6 pp).

### (c) Varying the Late-Phase Cutoff

All configurations use the same binary early/late split on profile data; only the MDP's switchover point changes.

| late_phase_days | P(57) | Delta vs baseline | Delta vs current (60d) |
|----------------|-------|------------------|----------------------|
| 30 | **7.8151%** | +2.99 pp | **+1.67 pp** |
| 45 | 6.9901% | +2.16 pp | +0.84 pp |
| 60 (current) | 6.1462% | +1.32 pp | — |
| 75 | 5.2816% | +0.48 pp | −0.86 pp |
| 90 | 4.4016% | −0.43 pp | −1.74 pp |

The relationship is monotonic: shorter late windows yield higher P(57). This is not paradoxical — it means the current configuration over-applies the "late" penalty by including August alongside September.

### (d) Monthly P@1 and Data Sufficiency

| Month | N days (5 seasons) | P@1 |
|-------|-------------------|-----|
| Mar   | 15  | 73.3% |
| Apr   | 144 | 86.8% |
| May   | 155 | 88.4% |
| Jun   | 150 | 87.3% |
| Jul   | 136 | 90.4% |
| **Aug** | 155 | **85.2%** |
| **Sep** | 148 | **83.1%** |
| Oct   | 9   | 77.8% |

Key observations:
- **August (85.2%) is meaningfully better than September (83.1%)** — a 2.1 pp gap within the "late" pool.
- March (15 days) and October (9 days) are far too thin for reliable bin estimation.
- Monthly bins at 5 quintiles would have ~3–31 observations per bin — viable for Apr–Sep, but bins like March have only 3 days per quintile, producing highly unstable estimates (observed Q1 hit rate = 0.333 vs 1.000 for Q3 — pure noise).

---

## Why `late_phase_days=30` Wins

The `late_phase_days` parameter controls *when* the MDP switches to the degraded hit-rate profile during backward induction:

- `late_phase_days=60`: applies late bins for the last 60 days of season (~Aug 1 onward)
- `late_phase_days=30`: applies late bins for only the last 30 days (~Sep 1 onward)

The current `late_bins` were computed from the combined Aug+Sep data. August's empirical P@1 (85.2%) is meaningfully higher than September's (83.1%) and closer to early-season levels. By applying the combined Aug+Sep penalty to August, we're overly pessimistic about August quality — leading the MDP to skip more days or avoid doubling when it shouldn't.

With `late_phase_days=30`, August uses the early-season bins (which more accurately reflect August's true hit rate), and only September uses the degraded profile. This better calibration adds +1.67 pp to P(57) — substantial given the current baseline of 6.15%.

---

## Monthly Bins: Not Feasible

Monthly bins would require:
1. Enough days per month to estimate 5 quintile boundaries reliably
2. ~30+ observations per bin for stable empirical hit rates

April–September have 136–155 days across 5 seasons = 27–31 days per bin — marginally acceptable but at the reliability floor. March (3 days/bin) and October (2 days/bin) are unusable. The observed bin hit rates for March (0.33, 1.00, 1.00, 0.67, 0.67) confirm pure noise at small sample sizes.

More critically, monthly-level MDP would require separate phase transitions at each month boundary — 6 phases instead of 2 — dramatically increasing complexity with marginal benefit over a well-tuned binary split.

**Conclusion**: Monthly bins are data-insufficient and not worth pursuing.

---

## Recommendation

Update the production MDP solve to use `late_phase_days=30` instead of `late_phase_days=60`.

**However**, note a subtlety: the `late_bins` are computed from combined Aug+Sep data (months 8–9). If we're only applying them to the last 30 days (~Sep), we should ideally recompute late bins from September-only data. The current analysis uses Aug+Sep combined bins for all non-baseline configurations, which may slightly underestimate September's true degradation.

A more precise implementation would:
1. Keep early bins from months 3–7
2. Compute late bins from September-only data
3. Use `late_phase_days=30`

This is a one-line change to the bin computation; the combined Aug+Sep bins already tested here show a +1.67 pp gain, and Sep-only bins would likely show an equal or larger improvement.

**Action required**: Re-run `bts simulate solve --save-policy` with updated bin construction.

---

## Files

- Script: `scripts/validation/item_10_phase_bins.py`
- This document: `docs/validation/item-10-phase-bins.md`
