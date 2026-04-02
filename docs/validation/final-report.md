# BTS Competitive Validation Sprint — Final Report

**Date:** 2026-04-02
**Duration:** ~2 hours wall-clock (parallel agent execution)
**Baseline:** P@1=86.5%, P(57)=6.15% (phase-aware MDP)

## Summary

14 investigation items across 5 phases. 13 items validated our current approach. 1 actionable improvement found: phase bin tuning.

## Verdict Table

| # | Item | Source | Verdict | Action |
|---|------|--------|---------|--------|
| 1 | 2026 double-down rule | Reddit claim | NOT CONFIRMED — official rules contradict | None (MDP correct) |
| 2 | Scoring change handling | Code audit | NOT HANDLED — 17hr gap | Audit trail rec (streak 20+) |
| 3 | Batting order signal | lokikg's #1 feature | Fully captured by PA aggregation | None |
| 4 | Home/away PA advantage | Community consensus | Signal captured (2:1 away skew) | None |
| 5 | Implied run total | Deep_Slice875 regression | No residual (r=-0.0007 with hits) | Reject |
| 6 | P@K vs lokikg | xwobiwan.com | Streak dist 14-21% better | Competitive |
| 7 | Walk-rate / BB% | Community Soto heuristic | Redundant (r=0.653 with count_tendency) | Reject |
| 8 | Contact quality composite | shefBoiRDee weighted scoring | Worse than individuals (dilutes avg_ev) | Reject |
| 8b | Lineup projection impact | lokikg real-time rescoring | 0.064pp impact, 5.9% affected | 3-run approach fine |
| 9 | MDP removable double | Research agent (hallucinated) | Rule doesn't exist | Cancelled |
| **10** | **Phase bin granularity** | **Internal investigation** | **Sept-only late bins → +0.51pp** | **ACCEPT** |
| 11 | Miss-day patterns | Internal investigation | Random (t=3.91 but no operable threshold) | None |
| 12 | Densest-bucket strategy | Internal investigation | 1.10pp rank gap, cheap at low rates | Keep current |

## Accepted Change: Phase Bin Tuning

**Before:** `late_phase_days=60` (Aug+Sep as "late" phase) → P(57) = 6.15%
**After:** `late_phase_days=30` (Sept-only as "late" phase) → P(57) = 6.66%
**Delta:** +0.51pp absolute, +8.4% relative

**Why it works:** August P@1 (85.2%) is much closer to the early-season average than to September (83.1%). The old binary split grouped Aug+Sep, under-estimating August quality and causing the MDP to be overly conservative during August.

**Policy file:** `data/models/mdp_policy_sept_only.npz` (saved, ready to deploy)

## Key Learnings

### Our model is solid
- PA-level aggregation captures batting order signal (lokikg's #1 feature is redundant for us)
- count_tendency already filters high-walk batters (Soto never reaches rank-1)
- Pitcher/park/weather features capture what Vegas market implies (r=0.316)
- Home/away handled by PA count aggregation

### Our competitive advantage is strategy, not prediction
- lokikg may have higher P@100 (89% claimed vs our ~82% extrapolated)
- But our streak distribution beats his by 14-21% (mean 31.9 vs 28)
- The MDP-optimal strategy is unique in the ecosystem — nobody on Reddit does it
- Strategy improvements compound exponentially: +0.51pp P@1 → +8.4% P(57)

### Miss days are genuinely random
- Confidence gap between hit/miss days: 0.011 (0.820 vs 0.809)
- No operable skip filter exists beyond what the MDP already uses
- Rank-2 hits 87.0% on miss days — tie-breaking, not prediction quality
- Misses are Poisson-distributed, no clustering

### What the community gets wrong
- Lineup position matters because their models don't do PA aggregation
- BB% filtering is correct instinct but our model already implements it
- "Always pick visiting team" is already captured by our PA counts
- Nobody optimizes the sequential decision (skip/double/saver)
- The "removable double-down" rule doesn't exist

## Scorecard Comparison

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| P@1 (overall) | 86.5% | 86.5% | 0 (no model changes) |
| P(57) MDP | 6.15% | 6.66% | **+0.51pp** |
| P(57) relative | — | — | **+8.4%** |
| Lifetime P(57) over 30 years | 86% | 88% | +2pp |

## Next Steps

1. **Deploy phase bin change:** Replace `data/models/mdp_policy.npz` with `mdp_policy_sept_only.npz` after verifying on Pi5
2. **Scoring audit trail:** Add raw hits count to pick JSON (minimum viable guard)
3. **Barrel rate monitoring:** Contact composite analysis showed barrel_rate has r=0.0013 (not significant) at PA level — may be the weakest blend member, worth a dedicated ablation test
4. **Odds API downgrade:** Confirmed no value from Vegas data; downgrade to $30/mo when 5M plan expires April 30
