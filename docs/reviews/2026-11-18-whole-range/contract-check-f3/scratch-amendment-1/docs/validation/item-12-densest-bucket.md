# Item 12: Densest-Bucket Strategy Validation

**Verdict:** TOLERABLE COST — densest-bucket displacement at realistic rates (<15%) costs less than 10% of P(57), and the benefit (avoiding lineup uncertainty) must exceed ~15% displacement rate to break even.

## What We Tested

The production `strategy.py` applies a "densest bucket" filter: it groups games by time window (early / prime / west coast), picks from the window with the most games, and only overrides if the overall rank-1 player exceeds 0.78 probability. The rationale: more simultaneous games = better odds the rank-1 player is actually in the lineup.

Since backtest profiles use confirmed lineups, lineup uncertainty can't be tested directly. This analysis measures the **cost function**: if densest-bucket sometimes forces us to pick rank-2 instead of rank-1, how much does P@1 drop?

Script: `scripts/validation/item_12_densest_bucket.py`  
Data: `data/simulation/backtest_{2021-2025}.parquet` — 912 game dates, 9,120 player-days

---

## Results

### Rank-1 vs Rank-2 P@1 (Overall, 2021–2025)

| Metric | Value |
|--------|-------|
| Rank-1 P@1 | 86.51% |
| Rank-2 P@1 | 85.42% |
| Gap | 1.10 percentage points |
| Baseline P(57) | 0.0259% |

The rank gap is real but modest: 1.10 pp across 912 dates.

### By-Season Breakdown

| Season | Rank-1 P@1 | Rank-2 P@1 | Gap | N days |
|--------|-----------|-----------|-----|--------|
| 2021 | 87.36% | 88.46% | −1.10 pp | 182 |
| 2022 | 87.71% | 83.80% | +3.91 pp | 179 |
| 2023 | 86.81% | 85.16% | +1.65 pp | 182 |
| 2024 | 84.32% | 84.86% | −0.54 pp | 185 |
| 2025 | 86.41% | 84.78% | +1.63 pp | 184 |

Two seasons (2021, 2024) show rank-2 beating rank-1. This is consistent with the well-known fragility of P@1: top-ranked players have near-identical probabilities, and noise dominates the gap in individual seasons. The 5-season average is what matters.

### P(57) Impact at Various Displacement Rates

| Displacement Rate | Blended P@1 | P(57) | vs Baseline |
|------------------|-------------|-------|-------------|
| 0% (baseline) | 86.51% | 0.000259 | — |
| 5% | 86.46% | 0.000250 | −3.5% |
| 10% | 86.40% | 0.000241 | −7.0% |
| 15% | 86.35% | 0.000233 | −10.3% |
| 20% | 86.29% | 0.000224 | −13.5% |
| 30% | 86.18% | 0.000209 | −19.5% |

**Threshold**: A -10% P(57) drop requires a 14.6% displacement rate.

### Probability Gap Distribution

The model's rank-1 vs rank-2 probability margin is small most days:

| Condition | Count | Fraction |
|-----------|-------|---------|
| prob gap < 0.01 | 441 / 912 | 48.4% |
| prob gap < 0.02 | 638 / 912 | 70.0% |

Mean rank-1 p_game_hit: 0.8183. Mean rank-2: 0.8026. The median gap in predicted probability is only 0.0104 — confirming that rank-1 and rank-2 are nearly interchangeable on most days.

---

## Interpretation

### Cost is modest at realistic displacement rates

At 10% displacement (roughly "1 in 10 days, densest-bucket forces rank-2"), P(57) drops 7%. At 15%, it drops 10.3%. These are real but not disqualifying.

### The break-even question

The densest-bucket filter is worth keeping **if and only if** lineup uncertainty causes expected rank-1 P@1 to drop by more than the displacement penalty. Specifically:

- If lineup DNP risk for a random rank-1 player ≈ 5%, the filter needs to reduce realized DNP exposure by enough to offset displacement costs.
- From item 8b data (if/when available): what fraction of days does the rank-1 player actually sit?

As a rough benchmark: the filter breaks even if it prevents even **one** rank-1 DNP per ~7 games (given a 10% displacement rate and 7% P(57) cost per unit).

### Small probability gaps make displacement cheap

Because rank-1 and rank-2 have nearly identical model scores on ~70% of days (gap < 0.02), forcing rank-2 on those days costs almost nothing in P@1. The displacement cost is primarily driven by the minority of days where the gap is large — and on those days, rank-1 is genuinely better, so the filter should ideally not override it (the 0.78 override threshold is meant to catch exactly this).

### Season-level instability is expected

Two of five seasons show rank-2 beating rank-1. This is noise, not signal — at 182 game-days per season, the standard error on P@1 is about 2.5 pp, larger than the 1.10 pp gap. The 5-season average is the reliable estimate.

---

## Connection to Item 8b

This analysis provides the cost function. Item 8b should measure the benefit (how often does densest-bucket avoid a confirmed DNP?). Together they give a full ROI picture:

```
Net P(57) impact = P(57 | benefit) - P(57 | cost)
                 = (fewer DNPs × value per avoided DNP) - (displacement_rate × 7% per 10%)
```

---

## Recommendation

Keep the densest-bucket filter, but verify item 8b. The 0.78 override threshold is well-placed: it protects against large-gap days where rank-1 is clearly dominant. If item 8b shows fewer than ~1 avoided DNP per 15 games, reconsider whether the filter adds net value.
