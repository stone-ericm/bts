# Item 11: Miss-Day Pattern Analysis

**Verdict:** NO ACTIONABLE PATTERN — misses are mostly random noise with weak signals below the threshold needed for safe skip rules.

## Overview

Analyzed 912 rank-1 daily picks across backtest seasons 2021–2025 (123 misses, 789 hits, overall P@1 = 86.5%). Six dimensions examined: confidence signal, confidence gap, rank-2 correlation, month effects, and miss clustering.

---

## A. Basic Stats by Season

| Season | Days | Hits | Misses | P@1   |
|--------|------|------|--------|-------|
| 2021   | 182  | 159  |  23    | 87.4% |
| 2022   | 179  | 157  |  22    | 87.7% |
| 2023   | 182  | 158  |  24    | 86.8% |
| 2024   | 185  | 156  |  29    | 84.3% |
| 2025   | 184  | 159  |  25    | 86.4% |
| **ALL**| **912**| **789**| **123** | **86.5%** |

2024 was the weak year (84.3%), consistent with general model degradation that season.

---

## B. Confidence Signal: p_game_hit on Hit vs Miss Days

The model's rank-1 confidence is statistically higher on days the pick actually hits vs misses, but the gap is small:

- Hit days mean p_game_hit: **0.8198** (n=789)
- Miss days mean p_game_hit: **0.8091** (n=123)
- Gap: **0.0106** — t=3.91, p=0.0001

**Interpretation:** Statistically significant (3.9σ), but practically tiny. The model "knows" something on miss days, but not enough to act on. The bulk of misses are in the 0.78–0.84 range — the model's most common operating zone.

Miss distribution by confidence:
- <0.80: 43% of misses
- 0.80–0.84: 41% of misses
- 0.84+: 16% of misses

There is no clean confidence threshold that separates miss days from hit days.

---

## C. Confidence Gap (rank-1 minus rank-2)

If misses concentrate on days when rank-1 barely edges out rank-2, a gap-based skip rule could help.

- Hit days mean gap: **0.0160** (median 0.0106)
- Miss days mean gap: **0.0140** (median 0.0094)
- Gap difference: 0.002 — t=1.33, **p=0.184** (not significant)

**Skip threshold simulation** (skip days where rank-1 gap < threshold):

| Threshold | Days Skipped | Skip % | Remain P@1 | Delta |
|-----------|-------------|--------|------------|-------|
| 0.002     | 110         | 12.1%  | 87.03%     | +0.52% |
| 0.005     | 251         | 27.5%  | 86.99%     | +0.48% |
| 0.010     | 441         | 48.4%  | **87.26%** | +0.75% |
| 0.020     | 638         | 70.0%  | 87.23%     | +0.71% |

The "best" threshold (0.010) requires **skipping 48% of all days** to gain +0.75% P@1. The miss rate across gap bins is flat (7–15%) with no clean break:

| Gap range     | Miss rate |
|--------------|-----------|
| 0.000–0.005  | 14.7%     |
| 0.005–0.010  | 13.7%     |
| 0.010–0.020  | 12.7%     |
| 0.020–0.050  | 13.1%     |
| 0.050–0.100  | 7.1%      |

**Interpretation:** The gap is not a useful filter. Miss rate is roughly flat across gap sizes. You'd skip nearly half of all playable days for a marginal P@1 improvement. More critically, the MDP model already accounts for skipping days — a hard filter on top of that is not additive.

---

## D. Rank-2 Hit Rate on Rank-1 Miss Days

- Rank-2 hit rate when rank-1 **misses**: **87.0%** (n=123)
- Rank-2 hit rate when rank-1 **hits**: **85.2%** (n=789)
- Overall rank-2 rate: **85.4%**

t-test: t=0.53, p=0.595 — **not significant**

Correlation(rank-1 hit, rank-2 hit): **−0.018**

**Interpretation:** Rank-2 slightly outperforms on rank-1 miss days, but this is noise (p=0.60). This confirms the prior finding (ARCHITECTURE.md §9): rank-1 and rank-2 outcomes are independent. P(both) = P1 × P2 is correct. There is no "consolation" signal in rank-2.

---

## E. P@1 by Month

| Month | Days | Hits | P@1   |
|-------|------|------|-------|
| Mar   |  15  |  11  | 73.3% |
| Apr   | 144  | 125  | 86.8% |
| May   | 155  | 137  | 88.4% |
| Jun   | 150  | 131  | 87.3% |
| Jul   | 136  | 123  | 90.4% |
| Aug   | 155  | 132  | 85.2% |
| Sep   | 148  | 123  | 83.1% |
| Oct   |   9  |   7  | 77.8% |

**Pattern:** July is the peak month (90.4%). Performance degrades Aug–Sep (83–85%), consistent with the known phase-aware bin degradation already captured in the MDP. March has a small sample (15 days) and should be discounted.

The phase split already in the model (early: Mar–Jul, late: Aug–Sep) correctly captures the main effect. There is no additional monthly granularity worth adding — the within-phase variation season-to-season swamps the signal.

---

## F. Miss Clustering

Are misses random or do they cluster in streaks?

Per-season inter-miss gaps:
- 2021: avg 7.9d, median 7d
- 2022: avg 8.6d, median 5d
- 2023: avg 7.3d, median 5d
- 2024: avg 6.5d, median 4d
- 2025: avg 8.0d, median 5d

Combined (912 days, 123 misses):
- Expected mean gap (Poisson): 7.4 days
- Observed mean gap: **7.6 days**
- Expected variance (Poisson): 55.0
- Observed variance: **47.3**
- Variance ratio: **0.860** — slightly underdispersed (more regular than random)

KS test vs Exponential: stat=0.123, **p=0.050** — borderline, consistent with random.

**Interpretation:** Misses are not clustered — they are slightly *more* evenly spaced than a pure Poisson process. No "cold streaks" or "hot streaks" to predict. Longest hit runs between misses ranged from 16 (2021) to 34 (2025) days, consistent with geometric distribution at ~13.5% daily miss rate.

---

## Verdict

| Signal | Effect | Actionable? |
|--------|--------|-------------|
| Low confidence on miss days | +0.0106 gap, p=0.0001 | No — too small, no clean threshold |
| Small rank-1/rank-2 gap | Not significant (p=0.18) | No — flat miss rate across all gap bins |
| Rank-2 correlation with miss | r=−0.018, p=0.60 | No — confirms independence |
| Month (July peak, Sep dip) | Real but already captured | No — phase bins handle this |
| Miss clustering | Slight underdispersion | No — effectively random |

**No actionable miss-day filter exists.** The strongest signal (absolute confidence, p=0.0001) is statistically real but operationally useless — the 0.0106 gap has no clean cutoff. Gap-based skip rules require sacrificing 12–48% of play days for 0.5–0.75% P@1 improvement, which does not justify the strategy disruption and compounds negatively with MDP skip logic.

The current approach — MDP-optimal play with phase-aware bins — already captures the most important structure. Miss days are fundamentally random conditional on the model's top prediction.

---

*Generated: 2026-04-02 | Script: `scripts/validation/item_11_miss_analysis.py`*
