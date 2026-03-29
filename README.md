# Beat the Streak v2

A PA-level MLB hit prediction model that beats the current state of the art on backtested data.

MLB's [Beat the Streak](https://www.mlb.com/apps/beat-the-streak) is a free contest with a $5.6 million prize: pick one player each day who you think will get at least one hit. String together 57 correct picks in a row — beating Joe DiMaggio's 56-game hitting streak — and you win. Since the contest launched in 2001, nobody has won.

This project investigates whether it's possible to build a model that makes this achievable, and what that model looks like.

## Results

Walk-forward backtested, provably leak-free, validated across 6 MLB seasons:

| Metric | BTS v2 | SOTA (Garnett 2026) |
|--------|--------|---------------------|
| P@1 (daily best pick) | 87.0% | ~85% |
| P@100 (top 100 picks) | 91.0% | 85% |
| P@500 (depth of signal) | 87.2% | 77% |

The P@500 advantage is the most robust finding — the model beats SOTA on this metric in every season tested (2020-2025).

### Multi-season validation

```
Season  Training data    P@1     P@100   P@500
2020    2019             89.6%   83.0%   83.0%
2021    2019-20          89.0%   87.0%   86.4%
2022    2019-21          90.5%   92.0%   89.0%
2023    2019-22          87.9%   95.0%   88.4%
2024    2019-23          81.6%   87.0%   85.8%
2025    2019-24          87.0%   91.0%   87.2%
```

### What this means for BTS

At 87% P@1, streak simulation over 50,000 seasons gives a median longest streak of 26 games and a 57+ game streak in ~0.6% of seasons. Still very unlikely, but meaningfully better than the mathematical baseline.

## The Story

### v1: Counting to 57 (2021)

The [original project](https://github.com/stone-ericm/MLB_counting_to_57_hits) achieved ~71.4% daily accuracy using Decision Trees on game-level Statcast data. The conclusion then: "the mathematical structure of the problem, not the quality of the model, is what makes Beat the Streak essentially unwinnable."

That conclusion was correct — at 71.4%. The question this project asks is: how much can we improve that number, and what does it take?

### What the field built (2020-2026)

In the intervening years, several researchers attacked this problem:

- **Alceo & Henriques (2020)**: MLP achieving 85% Precision@100. The academic benchmark.
- **Garnett (2026)**: MLP + LightGBM ensemble matching 85% P@100 but degrading more slowly at depth (77% P@500). Identified lineup position as the most important feature (3.8x more than #2).

All existing approaches model at the **batter-game level** — one prediction per batter per day. They aggregate pitch-level data into game-level features, losing PA-level granularity.

### v2: The PA-level insight

The game-level question decomposes naturally:

```
P(>=1 hit in game) = 1 - PROD(P(no hit in each PA))
```

If we model hit probability at the plate appearance level and aggregate, we get:

1. **More training data.** ~180K batter-game records per season become ~180K PAs per season, with ~4 PAs per batter-game giving us richer signal per record.
2. **Lineup position becomes natural.** The PA model doesn't need it as a feature — a leadoff hitter's advantage comes from having 4.5 PAs instead of 3.6, which the aggregation handles automatically.
3. **Pitcher matchup at the right level.** Platoon splits, pitcher quality, and pitch arsenal features are PA-level phenomena, not game-level.

### The leakage story

The most important part of this project was finding and fixing three separate data leakage bugs:

**Leakage #1: Static features using future data.** `platoon_hr` and `batter_gb_hit_rate` were computed over ALL data, including the test period. A batter's September 2025 performance leaked into their April 2025 predictions. Fix: expanding calculations with date-level `shift(1)`.

**Leakage #2: K-Means clustering instability.** Pitcher archetypes were clustered on the full dataset. When re-clustered on training data only, 90.8% of pitchers changed clusters — making `batter_vs_arch_hr` (our top feature by importance!) entirely dependent on future data. Fix: removed clustering features entirely.

**Leakage #3: Doubleheader contamination.** `shift(1)` on game-level features meant game 2 of a doubleheader could see game 1's results. Fix: aggregate to date level before shifting.

The impact: P@1 dropped from 94% (leaky) to 87% (clean). Every leakage bug looked like a legitimate improvement until rigorously tested. This is consistent with Garnett's warning that "data leakage via iterative test evaluation is the most common invisible failure mode."

**Verification:** A "nuclear test" manually recomputes every feature from scratch for random test PAs using only pre-date data, comparing against the pipeline's output. 260/260 checks passed.

## Architecture

Two-stage PA-level model:

```
MLB Stats API -> raw JSON -> PA-level Parquet -> 13 temporal features -> LightGBM -> P(hit|PA) -> P(>=1 hit|game)
```

### Features (13)

| Feature | Type | What it captures |
|---------|------|-----------------|
| batter_hr_{7,30,60,120}g | Rolling | Short-term form through long-term ability |
| batter_whiff_60g | Rolling | Contact quality |
| batter_count_tendency_30g | Rolling | Plate discipline |
| batter_gb_hit_rate | Expanding | Speed (infield hit proxy) |
| platoon_hr | Expanding | Batter x pitcher handedness matchup |
| pitcher_hr_30g | Rolling | Pitcher quality |
| pitcher_entropy_30g | Rolling | Arsenal diversity (harder to hit) |
| weather_temp | Context | Temperature affects ball flight |
| park_factor | Expanding | Venue effect on hit rates |
| days_rest | Context | Rust after time off |

Every feature is provably temporal — only data from dates strictly before the prediction date is used.

### What didn't work

- **Lineup position as a PA feature**: Double-counts with the aggregation step
- **Home/away**: No PA-level effect
- **Pitcher archetypes (K-Means)**: Cluster assignments unstable across train/test splits
- **MLP ensemble**: Trees handle our interaction features better
- **Umpire zone tendency**: Real effect (~3.6% CSR variance) but zero predictive power
- **Exit velocity trends, wind, career PAs, day of week**: All noise
- **Calibration (Platt/isotonic)**: Hurts P@K by reducing training data for base model
- **Training data before 2019**: Game changed enough that old data hurts (-1.1%)

## Data

- **Source**: MLB Stats API v1.1 (game feed endpoint)
- **Scope**: 9 seasons (2017-2025), 1.5M plate appearances
- **Training window**: 2019 onward (optimal)
- **Filters**: Regular season only. 7-inning COVID doubleheaders (2021) excluded.

## Usage

```bash
# Install
uv sync

# Pull game data
bts data pull --start 2019-03-20 --end 2025-10-01

# Build PA Parquet
bts data build --seasons 2019,2020,2021,2022,2023,2024,2025

# Evaluate (Python)
from bts.features.compute import compute_all_features
from bts.evaluate.backtest import walk_forward_evaluate

df = compute_all_features(pd.concat([
    pd.read_parquet(f"data/processed/pa_{y}.parquet")
    for y in range(2019, 2026)
]))
metrics, preds = walk_forward_evaluate(df, test_season=2025)
```

## Key Learnings

1. **PA-level modeling beats game-level.** More data, natural lineup position handling, pitcher matchup at the right granularity.
2. **Leakage is invisible until you test for it.** Three bugs, each discovered only through systematic verification. The "nuclear test" should be standard practice.
3. **Feature selection changes completely when leakage is present.** Our top feature under leakage (batter_vs_arch_hr) was entirely fake. Features that "hurt" under leakage (lineup_position) helped with clean data, and vice versa.
4. **More data helps, to a point.** Each additional training season improves the model, but data older than ~6 years hurts because the game changes.
5. **Simpler models with clean features beat complex models with noisy ones.** 13 features > 18. LightGBM alone > LightGBM + MLP. Default hyperparameters are fine.
6. **P@1 has wide confidence intervals.** +/-5% on a 184-day season. Multi-season validation is essential.

## References

- Garnett (2026), "Chasing $5.6 Million with Machine Learning" — current SOTA
- Alceo & Henriques (2020), "Beat the Streak: Prediction of MLB Base Hits Using ML" — academic benchmark
- Grinsztajn et al. (NeurIPS 2022), "Why do tree-based models still outperform deep learning on tabular data?"
- McElfresh et al. (NeurIPS 2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
- Nathan, "Physics of Baseball" — ball flight physics, air density effects
