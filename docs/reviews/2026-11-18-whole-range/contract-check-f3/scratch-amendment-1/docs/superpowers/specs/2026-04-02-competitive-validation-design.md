# BTS Competitive Validation Sprint

**Date:** 2026-04-02
**Goal:** Systematically validate our BTS approach against community strategies from r/beatthestreak, verify assumptions, investigate alternatives, and measure improvements — all against a multi-metric scorecard.

## Context

Deep dive into r/beatthestreak (3,122 subscribers) revealed:
- **lokikg** (xwobiwan.com): claims P@100=89%, lineup position as #1 feature, 10K Monte Carlo sims
- **Deep_Slice875**: regression showing batting order + implied run total >> all other features
- **shefBoiRDee** (bts-inky.vercel.app): 12-factor weighted scoring, contact quality composite, AI meta-picker
- **Community consensus**: visiting team preferred, avoid high-walk batters, skip days liberally
- **Nobody does sequential decision optimization** — our MDP is unique in the ecosystem
- **2026 rule change**: reportedly can remove double-down pick after first player locks

## Approach: Scorecard-First

Build a multi-metric scorecard baseline first. Every investigation modifies one thing and produces a scorecard diff. No change is accepted without measured deltas across all metrics.

## Primary Metric

P(57) exact (absorbing chain). Secondary: P@1 stability across seasons (both-seasons test). Tertiary: full scorecard (P@K curves, streak distributions, miss patterns, calibration).

---

## Item 13: Multi-Metric Scorecard (Phase 0 — Build First)

### Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| P(57) exact | Absorbing chain | The number that matters |
| P@1 by season (2020-2025) | Walk-forward backtest | Prediction accuracy + stability |
| P@K curves (K=1,5,10,25,50,100,250,500) | Walk-forward backtest | lokikg benchmark, precision depth |
| Longest backtest streak by season | Simulation from profiles | Tail behavior |
| Mean max streak (10K trials) | Monte Carlo from profiles | Expected best case |
| 90th/99th percentile max streak | Monte Carlo from profiles | Distribution shape |
| Miss-day rank-2 hit rate | Backtest predictions | Tie-breaking quality |
| Calibration at top decile | Predicted vs actual for rank 1-10 | Overconfidence check |
| Double-down win rate by quality bin | MDP policy replay | Strategy validation |
| Skip rate by season | MDP policy replay | Policy aggressiveness |

### Implementation

- New module: `src/bts/validate/scorecard.py`
- CLI command: `bts validate scorecard`
- Reads existing backtest profiles (`data/simulation/backtest_{season}.parquet`) + MDP policy
- Outputs: formatted table to stdout + JSON artifact at `data/validation/scorecard_{timestamp}.json`
- Scorecard diff: compare two JSON artifacts to show deltas

### Compute Cost

Reads existing data. Monte Carlo and exact P(57) are seconds. No model retraining needed for baseline.

---

## Phase 1: Quick Verifications

### Item 1: 2026 Removable Double-Down Rule

**Question:** Can you now remove your second pick after your first player's game starts?

**Method:** Research official BTS rules on MLB.com and community reports.

**If confirmed:** Re-solve MDP with modified transition (item 9 becomes mandatory). Transition changes from `P(reset) = 1 - P1*P2` to `P(reset) = 1 - P1` for doubles.

**If not confirmed:** Items 9 and the real-time rescoring interaction are scoped down.

**Effort:** 30 min. **Blocks:** Item 9.

### Item 2: Official Scoring Retroactive Changes

**Question:** Does our `check_hit` logic handle MLB changing a hit to an error the next day (before 6pm)?

**Method:** Code audit of check-results pipeline. Verify whether we re-verify on subsequent runs.

**Impact:** Correctness fix, not P(57). Binary pass/fail.

**Effort:** 1 hr.

### Item 4: Visiting Team PA Guarantee

**Question:** Does our model already capture the visiting-team PA advantage, or is there residual signal?

**Method:** Analyze existing backtest profiles — compare P@1 when rank-1 is home vs away. Check average PAs per game for home vs away batters in our data.

**Impact:** Likely small, absorbed by PA aggregation. Could inform a tiebreaker.

**Effort:** 30 min.

### Item 6: P@K Benchmark vs. lokikg

**Question:** How does our P@100 compare to lokikg's claimed 89%?

**Method:** Computed for free by the scorecard (P@K curves). Compare our numbers against his claims. Note: his leakage controls are unknown and he validates on a single season.

**Impact:** Benchmarking only. Contextualizes our model quality relative to the community's best.

**Effort:** Free (scorecard produces this).

---

## Phase 2: Data Analysis (No Model Changes)

### Item 8b: Projected vs. Confirmed Lineup Impact

**Question:** How much P@1 do we lose from lineup uncertainty in production?

**Method:**
1. For each game-date in backtest (2021-2025), construct "projected" lineup: each team's lineup = most recent prior game's lineup (our PA parquet already contains per-game lineup data with batting order, so prior-game lookups are straightforward)
2. Compute predictions using projected lineups (different PA counts, different batting order)
3. Compare rank-1 pick under projected vs confirmed — how often does it differ?
4. When different: which pick got a hit more often?
5. Compute P@1 delta: projected vs confirmed predictions
6. Stratify by day-of-week (Monday after off-day = worst case)
7. Estimate P@1 ceiling with perfect lineup info vs our 3-run approach

**Impact:** Quantifies operational P@1 gap. If <0.2%, 3-run approach is fine. If >0.5%, real-time rescoring worth building.

**Effort:** 3-4 hr.

### Item 11: Miss-Day Pattern Analysis

**Question:** Is there a pattern in when our rank-1 pick fails?

**Method:**
1. Collect all miss days from backtest
2. Analyze miss-day characteristics: predicted probability, pitcher type, park, day game, time of season
3. Test confidence-gap filter: skip when rank-1 and rank-2 within 0.005
4. Check for time-of-season miss clusters
5. Verify current finding (0.818 vs 0.811 predicted probability on hit vs miss days)

**Impact:** Could inform smarter skip decisions or MDP quality bin refinement.

**Effort:** 2 hr analysis + 2 hr for skip-rule testing if patterns emerge.

### Item 12: Densest-Bucket Strategy Validation

**Question:** Does our game-time bucketing strategy actually help P(57)?

**Method:**
1. Replay all backtest days through three strategies: (a) densest-bucket as-is, (b) always pick overall rank-1 regardless of game time, (c) always pick from prime-time window (4-8pm ET)
2. Compute P@1 for each — note that backtests use confirmed lineups, so this validates the *ranking restriction concept*, not lineup uncertainty (item 8b covers that)
3. Run through exact P(57) pipeline for each
4. Grid search the 0.78 override threshold (0.75-0.85)

**Impact:** Could go either way — bucketing may cost P@1 by not always picking the best player, or the time-window constraint may act as implicit regularization.

**Effort:** 2 hr.

---

## Phase 3: Feature Investigations (Each Needs Backtest Cycle)

All feature investigations follow the same protocol:
1. Compute the new feature with proper temporal guards (date-level shift(1))
2. Walk-forward backtest on 2024 AND 2025 (both-seasons test)
3. Test as: (a) addition to baseline model, (b) new blend member, (c) replacement for existing feature
4. Scorecard diff against baseline
5. Verdict: integrate if it improves P@1 on BOTH seasons

**Combinatorial testing:** If multiple features pass individually, they must also be tested together before integration. Feature interactions can be negative — two features that each help alone may hurt in combination (redundancy or overfitting). The final integration step in Phase 4 re-runs the full backtest with all accepted features combined.

### Item 3: Batting Order Signal

**Question:** Does our PA-level aggregation capture the full batting-order signal, or is there residual value in an explicit feature?

**Method:**
1. Add explicit `batting_order` feature (1-9) to baseline model
2. Walk-forward on 2024 + 2025
3. Stratify backtest accuracy by lineup slot — is our model already better at leadoff?
4. Check if starter/reliever PA split captures times-through-order

**Hypothesis:** PA aggregation captures PA-count advantage but may miss times-through-order effect (leadoff faces starter's first time through).

**Effort:** 2-3 hr.

### Item 5: Implied Run Total as Context Feature

**Question:** Does team-level implied run total (Vegas) add signal beyond our pitcher/park/weather features?

**Method:**
1. Extract team implied run totals from `data/external/odds/v2/` (Sept 2023 - Sept 2025)
2. Compute batter's team implied run total as new feature
3. Walk-forward on 2024 + 2025
4. Test as standalone feature and as blend member

**Key distinction:** We tested and rejected player-level hit props. This is team-level run total — a different, deeper signal that captures pitcher quality + park + weather + lineup strength in one market-consensus number.

**Data coverage:** Our odds data covers Sept 2023 - Sept 2025, giving full coverage of 2024 + 2025 seasons only. Both-seasons test still applies, but we cannot validate on earlier seasons.

**Effort:** 3-4 hr.

### Item 7: Walk-Rate / BB% Feature

**Question:** Does an explicit BB% feature or hard filter help beyond our `count_tendency_30g`?

**Method:**
1. Compute `batter_bb_rate_30g` (BB/PA over last 30 game-dates)
2. Test as model feature (both-seasons backtest)
3. Test as hard filter: exclude batters with BB% > 15% from ranking
4. Compare against current count_tendency signal

**Hypothesis:** Community's Soto-avoidance heuristic might have a sharper formulation than our count tendency proxy.

**Effort:** 2 hr.

### Item 8: Contact Quality Composite

**Question:** Does a single composite contact-quality feature capture interaction effects the individual blend members miss?

**Method:**
1. Compute `contact_quality_composite`: standardized average of barrel_rate + hard_hit_rate + sweet_spot_rate + avg_ev (30g rolling)
2. Test as 13th blend member
3. Compare against existing 4 individual Statcast blend members

**Hypothesis:** Probably marginal — blend already captures these through diversity. But composite might break ties differently.

**Effort:** 1.5 hr.

---

## Phase 4: Strategy Improvements

### Item 9: MDP with Removable Double-Down (Blocked by Item 1)

**Question:** How much does P(57) improve when doubling penalty is reduced?

**Method:**
1. Modify `mdp.py` transition matrix:
   - Current: `P(reset) = 1 - P1*P2` for doubles
   - New: `P(advance+2) = P1*P2`, `P(advance+1) = P1*(1-P2)`, `P(reset) = 1-P1`
2. Re-solve with backward induction
3. Compare P(57) and policy differences vs baseline
4. Full scorecard comparison

**Impact:** Potentially the largest single P(57) improvement. Doubling becomes a near-free option.

**Effort:** 2-3 hr.

### Item 10: Phase-Aware Bin Granularity

**Question:** Would finer time bins improve the MDP?

**Method:**
1. Test monthly bins (6 phases: Apr-Sep)
2. Test quarterly bins (3 phases: Apr-May, Jun-Jul, Aug-Sep)
3. Recompute quality bins from backtest profiles for each
4. Re-solve MDP for each, scorecard comparison

**Risk:** More bins = fewer data points per bin = noisier hit rates. Bias-variance tradeoff.

**Effort:** 2 hr.

---

## Execution Plan

```
Phase 0: Scorecard baseline                          ~15 min
   Build scorecard infrastructure + compute baseline

Phase 1: Quick verifications (parallel)              ~1 hr wall
   Items 1, 2, 4, 6

Phase 2: Data analysis (parallel)                    ~4 hr wall
   Items 8b, 11, 12

Phase 3: Feature backtests (parallel via worktrees)  ~4 hr wall
   Items 3, 5, 7, 8

Phase 4: Strategy improvements (sequential)          ~3 hr wall
   Items 9 (if item 1 confirmed), 10
   Integrate all accepted Phase 3 features
   Re-solve MDP on final feature set

Phase 5: Final scorecard comparison                  ~15 min
   Baseline vs final: full metric-by-metric diff
   Document all verdicts

Total: ~12 hr wall-clock, ~26 hr CPU
```

### Parallelization

- Phase 1: all items independent, run as parallel agents
- Phase 2: all items read existing data, no conflicts
- Phase 3: each item in a git worktree, can run on Mac + Alienware simultaneously
- Phase 4: sequential — MDP must be re-solved on the final feature set

### Success Criteria

Every item gets:
1. **Verdict:** confirmed / rejected / partially-confirmed
2. **Delta:** measured change to each scorecard metric (or "N/A — correctness fix")
3. **Action:** integrate / reject / revisit later
4. **Combined P(57):** final exact calculation with all accepted changes
