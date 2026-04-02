# Item 05: Implied Run Total Investigation

**Status: REJECT — no signal beyond existing features**

## Motivation

Deep_Slice875's r/beatthestreak regression found that batting order + implied run total >> all other features combined. We previously rejected *player-level* hit props (odds API). This test checks *team-level* implied offensive strength as a market-consensus context variable capturing pitcher quality + park + weather + lineup strength in one number.

## Data Source

The v2 odds dataset (`data/external/odds/v2/`) does not contain game totals (over/under) or team totals. It contains only `batter_hits` player props (0.5, 1.5, 2.5 over/under lines per player across multiple bookmakers).

**Proxy used:** Implied team hit rate = median implied P(hit>=1) across all listed players for a team on a given date. Derived from 0.5-over American odds using `P = |odds| / (|odds| + 100)` for favorites, `P = 100 / (odds + 100)` for underdogs. Mediated across bookmakers to reduce vig noise.

This is a defensible proxy: the 0.5 hit prop market prices in the same context factors (opposing pitcher, park, weather, lineup strength) that a game total would reflect.

## Data Coverage

- **v2 odds files:** 418 dates (2023-09-01 → 2025-09-28)
- **Backtest seasons covered:** 2023, 2024, 2025 (551 rank-1 picks total)
- **Name mapping:** 91.6% of player names in odds matched to batter_ids via raw game JSON scan
- **Game matching:** 5,232 / 5,316 odds events matched to game_pks (98.4%)
- **Overall rank-1 coverage:** 367 / 551 picks (66.6%)

| Season | Covered | Total | Coverage |
|--------|---------|-------|----------|
| 2023   | 31      | 182   | 17.0%    |
| 2024   | 183     | 185   | 98.9%    |
| 2025   | 153     | 184   | 83.2%    |

2023 is nearly unusable (odds data only starts Sept 2023). Primary analysis relies on 2024 (183 picks) and 2025 (153 picks).

## Results

### (b) Correlation with actual hit outcome

| Metric | Value | p-value |
|--------|-------|---------|
| r(implied_hit_rate, actual_hit) | **-0.0007** | 0.990 |
| r(implied_hit_rate, p_game_hit) | +0.316 | <0.001 |
| r(p_game_hit, actual_hit) | +0.071 | — |

The implied team hit rate has **essentially zero correlation** with whether the rank-1 pick actually got a hit (r=-0.0007). In contrast, it has moderate correlation with our model's own prediction (r=0.316), indicating our model already incorporates the same offensive context signals the market encodes.

Team implied hit rates ranged from 0.585 to 0.816, mean 0.691 ± 0.044 — sensible MLB batting average territory.

### (c) P@1 by Implied Hit Rate Quartile

| Quartile | Avg Implied HR | P@1 | n |
|----------|---------------|-----|---|
| Q1 (low) | 0.640 | 0.848 | 92 |
| Q2       | 0.675 | 0.848 | 92 |
| Q3       | 0.699 | 0.901 | 91 |
| Q4 (high)| 0.750 | 0.837 | 92 |

**Q4 vs Q1 z-test: z=-0.202, p=0.840 (not significant)**

Q3 has the highest P@1 (0.901), but Q4 (highest implied rate) is *lower* than Q1. There is no monotone relationship between team implied hit rate and rank-1 pick success. The quartile ordering is essentially flat with random variation around 0.858.

### (d) Residual Signal (after controlling for p_game_hit)

**Partial r(implied_hit_rate, actual_hit | p_game_hit) = -0.024, p=0.641**

After removing variance explained by our model's own prediction, the residual correlation is -0.024 — not distinguishable from zero.

Logistic regression 5-fold CV AUC: baseline 0.566 vs +implied rate 0.389. The implied rate *hurts* the logistic regression, consistent with it adding noise on a 367-sample dataset with highly skewed outcomes (88% hit rate).

### (e) High vs Low Implied Rate Split

| Half | P@1 | n |
|------|-----|---|
| High (≥ median 0.686) | 0.870 | 184 |
| Low (< median 0.686) | 0.847 | 183 |

z=0.620, p=0.535 — not significant. The 2.3 percentage point gap in P@1 (high vs low) is well within noise.

Per-season, the direction flips: +8.9pp in 2024 (favor high), -3.8pp in 2025 (favor low). No consistent effect.

### (f) Model Already Captures the Signal

Across teams: r(avg implied hit rate, avg p_game_hit) = **0.404, p=0.045**.

Our model's predictions are moderately correlated with the betting market's team-level implied offensive strength. This confirms the model's pitcher, park, and weather features already encode the same context the betting market uses.

## Key Finding

The betting market's implied team hit rate is **not predictive of rank-1 pick outcomes** beyond our existing model. The likely explanation:

1. Our model's `pitcher_hr_30g`, `park_factor`, and `weather_temp` features already capture the major components of what Vegas implies.
2. The r=0.316 between our predictions and market-implied rates shows we're tracking the same signal.
3. The residual (what the market "knows" that we don't) has zero predictive value at rank-1.

## Recommendation

**REJECT** as a model feature or blend component.

This is consistent with the efficient market hypothesis applied to BTS: if the market signal were genuinely orthogonal to our model and predictive, we would expect some nonzero partial correlation. We found r=-0.024. The betting market and our model appear to draw from the same underlying signal sources.

Note: The player-level hit props previously tested and rejected are the same dataset — rejecting team aggregation of those same props is consistent.

## Files

- Script: `scripts/validation/item_05_implied_runs.py`
- Odds data: `data/external/odds/v2/` (418 files, 2023-09-01 to 2025-09-28)
- Coverage note: Meaningful analysis limited to 2024 (183 picks) + 2025 (153 picks)
