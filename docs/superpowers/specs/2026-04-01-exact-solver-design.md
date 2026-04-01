# BTS Exact Solver Design (Spec A)

## Problem

The Monte Carlo simulator (10K-50K trials) gives P(57) estimates with confidence intervals and random noise. Strategy grid search requires minutes per configuration. We need:
1. **Exact P(57)** for any fixed strategy — zero noise, millisecond evaluation
2. **Provably optimal policy** — the best possible action for every (streak, days, saver, quality) state

## Approach

Two complementary tools built on Markov chain math:
1. **Absorbing chain** — exact P(57) for fixed heuristic strategies
2. **Reachability MDP** — backward induction over (streak, days_remaining, saver, quality_bin) to find the optimal policy

Both depend on empirical quality bins computed from the 912-day backtest profile pool.

## Component 1: Quality Bins

**Module**: `src/bts/simulate/quality_bins.py`

Compute empirical prediction quality tiers from backtest profiles.

**Input**: Daily profiles from `data/simulation/backtest_*.parquet`

**Process**: For each day, extract rank-1 and rank-2 predictions. Bin days into 5 equal-frequency quintiles by top-1 `p_game_hit`. For each bin, compute:
- `p_hit`: empirical P(rank-1 batter gets a hit)
- `p_both`: empirical P(rank-1 AND rank-2 both get hits)
- `frequency`: fraction of days in this bin
- `p_range`: (min, max) of `p_game_hit` in this bin

**Output**: `QualityBins` dataclass with 5 bins, each containing the above stats.

**Bin boundaries** are quintile cutpoints from the data (~0.794, 0.811, 0.824, 0.841 based on current profiles). These are recomputed from the data, not hardcoded.

Empirical values from current data:

| Bin | p_game_hit range | P(hit) | P(both) | Frequency |
|-----|-----------------|--------|---------|-----------|
| Q1  | 0.733 – 0.794   | 77.0%  | 63.4%   | 20%       |
| Q2  | 0.794 – 0.811   | 87.4%  | 74.2%   | 20%       |
| Q3  | 0.811 – 0.824   | 87.9%  | 72.0%   | 20%       |
| Q4  | 0.824 – 0.841   | 90.7%  | 77.5%   | 20%       |
| Q5  | 0.841 – 0.918   | 89.6%  | 81.4%   | 20%       |

## Component 2: Absorbing Chain (Exact P(57) for Fixed Strategies)

**Module**: `src/bts/simulate/exact.py`

For a given heuristic strategy (mapping from quality_bin → action), compute exact P(57) within a finite number of plays.

**Method**: Build the full transition matrix T (58×58) where states 0-56 are transient and 57 is absorbing. For each state s, the transition probabilities come from the strategy's action for each quality bin, weighted by bin frequency:

- If action=skip for bin q: contributes nothing (day consumed, no state change — handled by reducing effective plays)
- If action=single for bin q: `T[s, s+1] += freq[q] * p_hit[q]`, `T[s, 0] += freq[q] * (1-p_hit[q])` (with saver modification for s=10-15)
- If action=double for bin q: `T[s, s+2] += freq[q] * p_both[q]`, `T[s, 0] += freq[q] * (1-p_both[q])` (with saver)

Effective plays per season = `season_length * sum(freq[q] for q where action != skip)`.

P(reaching 57 within k plays from state 0) = `T^k[0, 57]` (k-step transition probability).

**Function**: `exact_p57(strategy: Strategy, quality_bins: QualityBins, season_length: int = 180) -> float`

Takes a `Strategy` object (from `strategies.py`). For each transition matrix row (streak state s), calls `get_thresholds(strategy, s)` to determine which quality bins lead to skip/single/double at that streak level. This correctly handles streak-dependent strategies like our optimized config (different thresholds at streak 10 vs 35 vs 50).

This replaces Monte Carlo for evaluating any heuristic strategy. Grid search becomes instantaneous.

## Component 3: Reachability MDP (Optimal Policy)

**Module**: `src/bts/simulate/mdp.py`

Find the provably optimal action for every possible game state.

### State Space

`(streak, days_remaining, saver_available, quality_bin)`:
- streak: 0-56 (57 = win terminal)
- days_remaining: 0-180
- saver_available: True/False (consumed on first miss at streak 10-15)
- quality_bin: 0-4 (Q1-Q5, observed at start of each day)

Total states: 57 × 181 × 2 × 5 = **103,170**

### Actions

{skip, single, double}

### Transitions

Given state (s, d, saver, q) and action a:

**skip**: Next state is (s, d-1, saver, q') where q' is drawn from quality bin frequency distribution. Streak holds, one day consumed.

**single**:
- Hit (prob p_hit[q]): → (s+1, d-1, saver, q')
- Miss at s=10-15 with saver available: → (s, d-1, False, q') — saver consumed, streak holds
- Miss otherwise: → (0, d-1, saver, q')

**double**:
- Both hit (prob p_both[q]): → (s+2, d-1, saver, q')
- Any miss at s=10-15 with saver: → (s, d-1, False, q') — saver consumed
- Any miss otherwise: → (0, d-1, saver, q')

### Bellman Equation

Solved by backward induction from d=0:

```
V(s, 0, saver, q) = 0        # out of days
V(57, d, saver, q) = 1       # reached 57

V(s, d, saver, q) = max over a in {skip, single, double} of:
  E_q'[ sum over outcomes of P(outcome|a,q) * V(next_s, d-1, next_saver, q') ]
```

Where `E_q'[f(q')] = sum over q' of freq[q'] * f(q')`.

### Output: MDPSolution

```python
@dataclass
class MDPSolution:
    optimal_p57: float              # V(0, 180, True, *) averaged over initial q
    value_table: np.ndarray         # V[s, d, saver, q]
    policy_table: np.ndarray        # action index for each state
    quality_bins: QualityBins

    def policy(self, streak, days_remaining, saver, quality_bin) -> str:
        """Return optimal action for a given state."""

    def compare_heuristic(self, strategy_actions) -> dict:
        """Compare MDP optimal P(57) with a heuristic strategy's exact P(57)."""

    def extract_thresholds(self) -> str:
        """Summarize the policy as human-readable threshold patterns."""
```

`extract_thresholds()` scans the policy table and reports patterns like:
- "At streak 0-9 with 100+ days: play Q2+, double Q4+"
- "At streak 35 with <30 days: play Q3+ only, never double"

This tells us how the optimal policy differs from our heuristic.

## Component 4: CLI Integration

**New commands** under `bts simulate`:

### `bts simulate solve`
```
bts simulate solve [--season-length 180]
```
Run MDP solver. Print:
- Optimal P(57)
- Heuristic P(57) (for comparison)
- Key differences between MDP policy and current heuristic
- Threshold summary

### `bts simulate exact`
```
bts simulate exact --strategy current [--season-length 180]
```
Compute exact P(57) for a named heuristic strategy via absorbing chain.

### Updates to `bts simulate run`
Add `--exact` flag: when set, use absorbing chain instead of Monte Carlo for strategy evaluation. Faster, no noise, no CI needed.

## Module Structure

```
src/bts/simulate/
    quality_bins.py    — QualityBins dataclass, compute_bins() from profiles
    exact.py           — exact_p57() via absorbing chain
    mdp.py             — solve_mdp() → MDPSolution with policy + extraction
    cli.py             — add solve + exact commands
```

## Validation

1. Cross-check: run MDP's optimal policy through Monte Carlo → should match MDP's exact P(57) within CI
2. Cross-check: absorbing chain P(57) for current heuristic should match Monte Carlo's 2.59% (within CI)
3. Sanity: MDP optimal P(57) ≥ heuristic P(57) (by definition — MDP can always replicate the heuristic)

## Testing

- Unit tests for quality bin computation (known inputs → known bins)
- Unit tests for absorbing chain (trivial 3-state chain with known solution)
- Unit tests for MDP solver (small state space, verify V(terminal)=1, V(0,0,*)=0)
- Integration: solve MDP, verify optimal ≥ heuristic
- Integration: exact_p57 for "always play, never double" matches analytical formula

## Dependencies

numpy (for matrix operations). Already available.
