# Beat the Streak v2

A PA-level MLB hit prediction model with a 12-model ensemble that beats the current state of the art on backtested data.

MLB's [Beat the Streak](https://www.mlb.com/apps/beat-the-streak) is a free contest with a $5.6 million prize: pick one player each day who you think will get at least one hit. String together 57 correct picks in a row — beating Joe DiMaggio's 56-game hitting streak — and you win. Since the contest launched in 2001, nobody has won.

This project investigates whether it's possible to build a model that makes this achievable, and what that model looks like.

**Daily picks:** [@beatthestreakbot.bsky.social](https://bsky.app/profile/beatthestreakbot.bsky.social)

## Results

Walk-forward backtested, provably leak-free, validated across 6 MLB seasons:

| Metric | BTS v2 | SOTA (Garnett 2026) |
|--------|--------|---------------------|
| P@1 (daily best pick) | **86.9% avg** (6 seasons) | ~85% |
| P@100 (top 100 picks) | 91.0% | 85% |
| P@500 (depth of signal) | 87.2% | 77% |

The 86.9% P@1 uses a densest-bucket timing strategy with 78% override threshold, 14-feature model with catcher framing, and 12-model blend — validated across all 6 test seasons (2020-2025).

### Multi-season validation

```
Season  Training data    P@1 (model)   P@1 (with strategy)
2020    2019             89.6%         88.1%
2021    2019-20          89.0%         88.5%
2022    2019-21          90.5%         88.3%
2023    2019-22          87.9%         87.4%
2024    2019-23          81.6%         86.5%
2025    2019-24          87.0%         86.4%
```

"With strategy" includes the densest-bucket timing + 78% override, which lifts the weaker seasons (2024: 81.6% → 86.5%) at the cost of slightly moderating the strongest ones.

### What this means for BTS

At ~87% P@1, raw prediction accuracy alone gives ~0.9% P(57) per season. But P(57) is dominated by **play strategy**, not model accuracy — the exponential nature of p^57 means small improvements in effective per-play accuracy compound massively.

A reachability MDP (103K states, backward induction) finds the provably optimal strategy: **P(57) = 6.66%** per season (~1 in 15), a 7.4x improvement over the baseline with zero model changes. Over 30 years of playing BTS, this gives an **88% chance of hitting 57 at least once**.

Key strategy insights (Monte Carlo validated on 912 daily profiles across 5 seasons):
- Skip days when top pick confidence is low — raises effective per-play accuracy ~86.5% → ~89%
- Double selectively through streak 0-45 — reduces total days at risk
- Stop doubling at streak 46+ — don't risk a catastrophic near-win reset
- Adapt to days remaining in season — play aggressively early, conservatively late
- Model degrades in September specifically (85% → 83%) — phase-aware bins (Sept-only late phase) capture this

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

1. **More training data.** ~180K PAs per season with ~4 PAs per batter-game giving us richer signal per record.
2. **Lineup position becomes natural.** A leadoff hitter's advantage comes from having 4.5 PAs instead of 3.6, which the aggregation handles automatically — no feature needed.
3. **Pitcher matchup at the right level.** Platoon splits, pitcher quality, and pitch arsenal features are PA-level phenomena, not game-level.

### The leakage story

The most important part of this project was finding and fixing three separate data leakage bugs:

**Leakage #1: Static features using future data.** `platoon_hr` and `batter_gb_hit_rate` were computed over ALL data, including the test period. Fix: expanding calculations with date-level `shift(1)`.

**Leakage #2: K-Means clustering instability.** Pitcher archetypes were clustered on the full dataset. When re-clustered on training data only, 90.8% of pitchers changed clusters. Fix: removed clustering features entirely.

**Leakage #3: Doubleheader contamination.** `shift(1)` on game-level features meant game 2 of a doubleheader could see game 1's results. Fix: aggregate to date level before shifting.

The impact: P@1 dropped from 94% (leaky) to 87% (clean). Every leakage bug looked like a legitimate improvement until rigorously tested.

**Verification:** A "nuclear test" manually recomputes every feature from scratch for random test PAs using only pre-date data, comparing against the pipeline's output. 260/260 checks passed.

### Catcher framing: the breakthrough

After ~40 experiments testing features, models, and blends, catcher framing was the only feature to consistently improve P@1 on both test seasons.

The insight: every PA involves three people (batter, pitcher, catcher), but the model only knew about two. A catcher who frames well steals called strikes on borderline pitches, shifting counts against the batter. The effect is massive — 11 WAR spread across MLB catchers (SABR research). We compute a proxy from the called-strike rate on borderline pitches (|pX| 0.5-1.2 or near zone edges) per pitcher, expanding with temporal shift.

Adding catcher framing improved average P@1 from 85.1% to 87.0% — a larger gain than the 12-model blend (86.2%) or any other feature tested.

### The 12-model blend

Year-to-year instability is a fundamental challenge: features that improve P@1 on one season often hurt the next. ~40 experiments confirmed this pattern. The 12-model blend (each using baseline features plus one Statcast-derived feature) improves P@1 through tie-breaking diversity. When the model's #1 pick goes hitless, the #2 pick gets a hit 84-88% of the time — the blend's diverse votes break these ties better than any single model.

### Densest bucket + override timing strategy

Not all game times are equal. Backtesting across 6 seasons showed that picking from the **densest time window** (the one with the most games) produces the best results. More games = more options = higher expected best pick.

The override: if any pick from any window exceeds **78% P(game hit)**, take it regardless of which window is densest. These high-confidence picks hit 87%+ of the time and shouldn't be missed just because they're in a smaller window.

This strategy is now managed by a **dynamic lineup scheduler** (`bts schedule`) that checks lineups 45 minutes before each game's start time, only committing to a pick when confirmed lineups show the top pick's advantage exceeds a configurable gap threshold (`early_lock_gap=0.03`, derived from backtesting: 92.8% accuracy when locking early). It improved average P@1 from 85.3% (pure densest) to 86.9%.

## Architecture

Two-stage PA-level model with daily automation:

```
MLB Stats API -> raw JSON -> PA-level Parquet -> 14 features + blend -> LightGBM -> P(hit|PA) -> P(>=1 hit|game) -> densest bucket + override -> pick
```

### Core features (14)

| Feature | Type | What it captures |
|---------|------|-----------------|
| batter_hr_{7,30,60,120}g | Rolling | Short-term form through long-term ability |
| batter_whiff_60g | Rolling | Contact quality |
| batter_count_tendency_30g | Rolling | Plate discipline |
| batter_gb_hit_rate | Expanding | Speed (infield hit proxy) |
| platoon_hr | Expanding | Batter x pitcher handedness matchup |
| pitcher_hr_30g | Rolling | Pitcher quality |
| pitcher_entropy_30g | Rolling | Arsenal diversity (harder to hit) |
| pitcher_catcher_framing | Expanding | Opposing catcher's borderline called-strike rate (framing proxy) |
| weather_temp | Context | Temperature affects ball flight |
| park_factor | Expanding | Venue effect on hit rates |
| days_rest | Context | Rust after time off |

Every feature is provably temporal — only data from dates strictly before the prediction date is used.

### Statcast features (9, used by blend variants)

Extracted from game feed hitData and pitchData. Each appears in one blend model alongside the 13 baseline features.

| Feature | Type | What it captures |
|---------|------|-----------------|
| batter_barrel_rate_30g | Rolling | Quality of hardest contact (stabilizes at ~50 BIP) |
| batter_hard_hit_rate_30g | Rolling | Hard contact rate (EV >= 95 mph) |
| batter_sweet_spot_rate_30g | Rolling | Launch angle discipline (8-32°) |
| batter_avg_ev_30g | Rolling | Consistent hard contact |
| pitcher_avg_velo_30g | Rolling | Pitch velocity |
| pitcher_avg_spin_30g | Rolling | Spin rate (pitch quality proxy) |
| pitcher_avg_extension_30g | Rolling | Release extension (perceived velocity) |
| pitcher_break_total_30g | Rolling | Pitch movement magnitude |
| batter_avg_velo_faced_30g | Rolling | Average pitch velocity faced |

These features don't improve the single model, but each adds diversity to the blend — the ensemble's power comes from models that disagree on close calls.

### Prediction features

- **Starter/reliever split**: PAs 1-2.5 use starter features, PAs 3+ use bullpen composite
- **Opener detection**: Pitchers averaging < 3 IP flagged; all PAs treated as reliever PAs
- **IL return flags**: Players with 7+ days rest flagged
- **Debut pitcher fallback**: League-average features for pitchers with no MLB history
- **Projected lineup fallback**: When lineups aren't posted yet, uses team's most recent game lineup

### MDP-optimal strategy

The play strategy is determined by a reachability MDP that finds the provably optimal action (skip/single/double) for every state of (streak, days_remaining, saver_available, quality_bin). The state space is 57 × 181 × 2 × 5 = 103K states, solved in seconds via backward induction.

**Quality bins**: Days are classified into 5 equal-frequency quintiles by top-pick confidence. Each bin has empirically measured P(hit) and P(both hit). Phase-aware bins use separate distributions for early season (Mar-Aug) vs late (Sep only, `late_phase_days=30`), capturing the model's degradation in September specifically (Aug P@1 is closer to early-season than Sept).

**Streak saver**: BTS gives one free miss when streak is 10-15. The MDP and production code track saver state and factor it into decisions.

**Policy file**: `data/models/mdp_policy.npz` (1.8KB). Generated by `bts simulate solve --save-policy`. Auto-loaded by `strategy.py` at prediction time. Falls back to heuristic thresholds if absent.

### ABS challenge data (2026)

Extracts MLB's new Automated Ball-Strike challenge data (player, role, outcome, team) for future feature development. A shadow model monitors when challenge skill features accumulate enough signal to improve predictions.

### What didn't work

**Features rejected** (tested and dropped after empirical validation):
- MiLB debut pitcher entropy — no P@1 improvement; LightGBM handles missing values well
- Team defensive efficiency (BABIP) — 30-day rolling too noisy (r=0.19)
- Granular defense (GB/FB splits, error rate, hard-hit conversion) — all park effects
- Lineup position as a PA feature — double-counts with the aggregation step
- Pitcher archetypes (K-Means) — 90.8% cluster assignment instability
- Umpire zone tendency — zero predictive power
- Exit velocity trends, wind, career PAs, day of week — all noise

**Modeling rejected** (tested and dropped):
- Hyperparameter tuning — default params are robust; tuning hurt P@1
- Recency weighting — downweighting old data removes volume the model needs
- Ranking-optimized objective — improves one season, hurts the other
- Adaptive feature selection — worse than fixed blend
- 15+ model blend — dilutes signal; 12 is the sweet spot
- Different architectures in blend (Decision Tree, Logistic Regression) — all members must be equally competent
- MLP ensemble — trees handle our interaction features better
- Calibration (Platt/isotonic) — hurts P@K
- Training data before 2019 — game changed enough that old data hurts (-1.1%)
- Sprint speed proxy (soft GB hit rate) — too noisy
- Prior-start pitch count — hurt P@1
- Nash Score proxy (pitch allocation balance) — hurt P@1
- Times through order adjustment — no effect (aggregation already captures it)
- Wind-out component — hurt 2025
- Vegas player props (market implied P(hit)) — doesn't pass both-seasons test
- Savant catcher framing (static prior-season) — expanding proxy beats the "real" data
- Eastward travel flag — helped individually but hurt in combination
- 25-PA hot hand rolling — mixed (helped 2024, flat 2025)

## Data

- **Source**: MLB Stats API v1.1 (game feed endpoint)
- **Scope**: 9 seasons (2017-2025), 1.5M plate appearances
- **Training window**: 2019 onward (optimal)
- **Filters**: Regular season only. 7-inning COVID doubleheaders (2021) excluded.
- **Statcast fields**: trajectory, hardness, total_distance, pitch velocities, spin rates, extensions, break vectors
- **ABS fields**: challenge player, role, outcome, team (2026+)

## Usage

```bash
# Install (Pi5 / orchestrator — no model deps)
UV_CACHE_DIR=/tmp/uv-cache uv sync

# Install (Mac / workers — full model)
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model

# Pull game data
UV_CACHE_DIR=/tmp/uv-cache uv run bts data pull --start 2019-03-20 --end 2025-10-01

# Build PA Parquet
UV_CACHE_DIR=/tmp/uv-cache uv run bts data build --seasons 2019,2020,2021,2022,2023,2024,2025

# Dynamic lineup scheduler (Pi5 — replaces fixed cron runs)
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml

# Preview scheduler plan without executing
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml --dry-run

# Orchestrate via SSH cascade (manual single-run alternative)
UV_CACHE_DIR=/tmp/uv-cache uv run bts orchestrate --date 2026-04-01 --config ~/.bts-orchestrator.toml

# Local prediction (Mac — predict, apply MDP strategy, save pick, post to Bluesky)
UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-04-01

# Preview without saving/posting
UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-04-01 --dry-run

# Check yesterday's results and update streak (with saver tracking)
UV_CACHE_DIR=/tmp/uv-cache uv run bts check-results --date 2026-03-31

# --- Strategy simulation ---

# Run blend backtest for strategy analysis (5 seasons, ~2-3 hours)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025

# Monte Carlo strategy comparison (10K bootstrapped seasons)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate run --trials 10000

# Solve MDP for optimal policy + generate policy file
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy.npz

# Exact P(57) for any named strategy (no Monte Carlo noise)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate exact --strategy combined
```

## Key Learnings

1. **PA-level modeling beats game-level.** More data, natural lineup position handling, pitcher matchup at the right granularity.
2. **Leakage is invisible until you test for it.** Three bugs, each discovered only through systematic verification. The "nuclear test" should be standard practice.
3. **Feature selection changes completely when leakage is present.** Our top feature under leakage was entirely fake.
4. **Year-to-year instability is fundamental.** Features that help one season hurt the next. Only the blend consistently improves both test seasons.
5. **Blend diversity > model complexity.** 12 LightGBM variants with different feature subsets beat any single model, hyperparameter tuning, different architectures, or adaptive selection. The power is in tie-breaking via diverse votes.
6. **The model's problem is ranking, not prediction.** When the top pick misses, #2 gets a hit 84-88% of the time. The model knows who's good — it struggles with who's best *today*.
7. **Simpler models with clean features beat complex models with noisy ones.** 13 features > 18. Default hyperparameters are robust to tuning.
8. **Timing strategy matters as much as the model.** Picking from the densest time window with a high-confidence override added +1.6% P@1 — more than most feature experiments.
9. **Real data isn't always better than a proxy.** Savant's calibrated catcher framing (static, prior-season) lost to our expanding proxy that updates every game. Adaptive beats precise-but-stale.
10. **The market doesn't know more than the model.** Vegas player prop odds didn't improve P@1 — the market and our model look at the same fundamentals.
11. **P@1 has wide confidence intervals.** +/-5% on a 184-day season. Multi-season validation (6 seasons) is essential — 2-season results are unreliable.
12. **Strategy >> model improvements for P(57).** MDP-optimal play strategy improved P(57) from 0.90% to 6.66% (7.4x) with zero model changes. The exponential nature of p^57 means small per-play accuracy gains from skipping bad days compound massively.
13. **Anti-correlated doubling is a dead end.** Rank-1 and rank-2 outcomes are independent (r=-0.018). P(both hit) = P1 × P2 is correct — no correlation to exploit.
14. **Model degrades in September specifically.** Sept P@1 drops to 83.1% vs Aug 85.2%. Phase-aware quality bins (Sept-only late phase, `late_phase_days=30`) capture this, adding +0.5% P(57).

## References

- Garnett (2026), "Chasing $5.6 Million with Machine Learning" — current SOTA
- Alceo & Henriques (2020), "Beat the Streak: Prediction of MLB Base Hits Using ML" — academic benchmark
- Grinsztajn et al. (NeurIPS 2022), "Why do tree-based models still outperform deep learning on tabular data?"
- McElfresh et al. (NeurIPS 2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
- Nathan, "Physics of Baseball" — ball flight physics, air density effects
