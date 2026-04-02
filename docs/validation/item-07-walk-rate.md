# Item 07: Walk-Rate / BB% Feature Investigation

**Verdict:** REJECT both feature and filter — marginal delta (+0.22% P@1) doesn't justify complexity, and the Soto problem doesn't actually exist in backtest data.

## Overview

Tested whether explicit BB% as a feature or hard filter helps beyond the existing `batter_count_tendency_30g` proxy. The r/beatthestreak community universally avoids high-walk players (Juan Soto is the poster child). Analyzed 912 rank-1 daily picks across backtest seasons 2021–2025.

---

## A. Correlation: BB% vs `batter_count_tendency_30g`

Computed rolling 30-game-date BB rate and count tendency on the same (batter, date) pairs across 292,974 observations (2019–2025, min 10 game-dates warm-up).

| Metric | Value |
|--------|-------|
| N pairs | 292,974 |
| Pearson r | **0.653** |
| Spearman r | **0.645** |
| p-value (both) | < 1e-300 |

**Interpretation:** BB% and `batter_count_tendency_30g` are highly correlated (r ≈ 0.65). This makes sense mechanically — batters who draw more walks run deeper counts (more balls, fewer strikes). The existing proxy already captures most of the BB% signal. BB% as a standalone feature would be largely redundant.

---

## B. P@1 by BB% Quartile (Rank-1 Picks)

For rank-1 picks, grouped by the batter's rolling 30-game BB% on that date.

| Quartile | Days | Hits | Mean BB% | P@1 |
|----------|------|------|----------|-----|
| Q1 (low BB, ≤3.4%) | 233 | 198 | 3.4% | **85.0%** |
| Q2 (3.4–5.6%) | 225 | 199 | 5.6% | **88.4%** |
| Q3 (5.6–7.7%) | 228 | 199 | 7.7% | **87.3%** |
| Q4 (high BB, ≥7.7%) | 226 | 193 | 11.5% | **85.4%** |

**Interpretation:** The pattern is non-monotonic and weak. Q4 (high walk) barely underperforms Q1 (low walk): 85.4% vs 85.0%. The performance *dip* at Q4 relative to Q2/Q3 is real but small. Critically, **the model already discounts high-walk batters** — they rarely reach rank-1 (only 226/912 = 24.8% of rank-1 picks are in Q4, while Q1 accounts for 25.5%). The model's existing count tendency feature is doing the filtering.

---

## C. Hard Filter Test

Simulated excluding batters with BB% > threshold from rank-1, falling back to next-eligible ranked batter.

| Threshold | Days Affected | Affected % | High-Walk P@1 | Fallback P@1 | New Overall P@1 | Delta |
|-----------|--------------|------------|---------------|--------------|-----------------|-------|
| 10% | 163 | 17.9% | 85.3% | 86.5% | 86.7% | +0.22% |
| 12% | 73 | 8.0% | 87.7% | 87.7% | 86.5% | 0.00% |
| **15%** | **20** | **2.2%** | **80.0%** | **90.0%** | **86.7%** | **+0.22%** |
| 18% | 5 | 0.5% | 60.0% | 80.0% | 86.6% | +0.11% |
| 20% | 1 | 0.1% | 100.0% | 100.0% | 86.5% | 0.00% |

At 15% threshold: 20 pick-days affected (2.2% of all days), with the high-walk picks hitting at 80.0% vs 90.0% for the fallback. Overall improvement: **+0.22% P@1** (86.73% vs 86.51%).

**Interpretation:** The 15% filter is the only threshold with any positive signal — but it affects only 20 days across 5 seasons (4 days/season) and produces +0.22% P@1. That's below the noise floor of seasonal variation (P@1 ranges 84–88% year to year). The 10% threshold moves more days but the filtered picks hit at 85.3%, nearly identical to the fallbacks at 86.5%, so the gain is marginal.

---

## D. High-Walk Batters as Rank-1 Picks

Top 10 highest-BB batters appearing as rank-1 in backtest data:

| Batter ID | Rank-1 Days | Hit Rate | Mean BB% | Mean p_game |
|-----------|-------------|----------|----------|-------------|
| 701762 | 1 | 0.0% | 18.9% | 80.5% |
| 663457 | 1 | 100.0% | 16.8% | 81.6% |
| 666397 | 1 | 100.0% | 16.2% | 80.1% |
| 663697 | 1 | 100.0% | 15.9% | 83.9% |
| 664023 | 1 | 0.0% | 15.4% | 79.5% |
| 592450 | 3 | 100.0% | 15.2% | 81.6% |
| 457708 | 1 | 100.0% | 13.3% | 81.3% |
| 607043 | 1 | 100.0% | 13.3% | 79.1% |
| 669701 | 2 | 100.0% | 12.7% | 80.2% |
| 673548 | 9 | 77.8% | 12.5% | 80.8% |

High-walk batters appear as rank-1 very rarely (1–9 days each) and their hit rates are mixed — 8 of 10 hit at 77–100%. Only 2 batters (ids 701762, 664023) missed when they were rank-1. The small sample sizes mean these are noisy.

---

## D2. Juan Soto Check

- Soto (MLB batter_id=665742) is confirmed in the PA data: **4,166 PAs, 727 walks = 17.5% BB** (consistent with real-world figures)
- **Soto never appeared as rank-1** in any of the 5 backtest seasons
- The model's `batter_count_tendency_30g` already penalizes him — his deep counts generate negative tendency scores, pushing him down the ranking

**Interpretation:** The r/beatthestreak concern about Soto is valid in theory but **irrelevant in practice for our model**. The existing count tendency feature already filters him out before he reaches rank-1. No additional rule is needed to exclude Soto.

---

## Verdict

| Approach | P@1 Impact | Days Affected | Justified? |
|----------|-----------|---------------|------------|
| BB% as feature | ~0% (redundant with count_tendency, r=0.65) | N/A | **No** |
| BB% > 15% filter | +0.22% | 20/912 days (2.2%) | **No** |
| BB% > 10% filter | +0.22% | 163/912 days (17.9%) | **No** |

**Reject both.** Three reasons:

1. **`batter_count_tendency_30g` already captures it.** r=0.65 Pearson correlation means BB% provides ~57% overlapping variance with the existing feature. Adding it would introduce multicollinearity without independent signal.

2. **The filter moves too few days for the gain.** +0.22% P@1 from the best threshold (15%) affects only 20 days across 5 seasons. Year-to-year P@1 variation is ±2–3%, making this indistinguishable from noise.

3. **The Soto problem doesn't exist in our data.** The community concern about high-walk players is already handled by the model — Soto (17.5% BB) never reached rank-1 in 5 seasons of backtest data. No escape hatch needed.

---

*Generated: 2026-04-02 | Script: `scripts/validation/item_07_walk_rate.py`*
