# 2026-08-30 — Same-game double-down pairs: is there a correlation lift?

## Question

`strategy.select_pick` refuses to pair a double-down from the same game as the
primary "to avoid correlated outcomes" (`src/bts/strategy.py`, DD selection).
For a both-must-hit bet, a *positive* correlation would raise P(both) and the
rule would cut the wrong way; a *negative* one would justify it. On 2026-08-30 the
question became live: the delivered pair (Kwan + McNeil) arrived after Kwan's
cutoff, and the best remaining partner for McNeil (ATH, 16:05) by marginal
probability was Blaze Alexander (BAL, 72.5%) — in McNeil's own game — vs
Alvarez/Bohm (70.4%) in other games. Alexander's edge in product terms is
0.725/0.704 ≈ 1.03, so a same-game lift below ~0.971 would flip the call.

An earlier recommendation asserted a "mildly positive" same-game correlation
from reasoning alone. This backtest replaces that assertion.

## Method (`scripts/audit/same_game_pair_lift.py`)

- Data: `data/processed/pa_{2019,2021..2025}.parquet` (2020 excluded), batter-games
  with ≥3 PA and lineup slot ≤5 (starter-like, where BTS picks live): 140,154.
- Marginal p per batter-game: leave-one-out batter-season game-hit rate, shrunk
  toward the season league rate with k=20 pseudo-games; batters with <10 games
  dropped. The batter's own game never informs his own marginal.
- Lift **R = mean(hit_i·hit_j) / mean(p_i·p_j)** over pairs. R>1: positive
  correlation (helps a DD); R<1: negative.
- Control: same-date, *different*-game pairs (soaks up any marginal-model bias so
  the decision quantity is R_same / R_control).
- 95% CI: cluster bootstrap over dates (400 reps).

## Results

| Pair type | n | R | 95% CI |
|---|---|---|---|
| Control: same date, different game | 129,493 | 0.994 | [0.985, 1.002] |
| **Same game, opposite teams** | 340,260 | **0.9935** | [0.986, 1.001] |
| Same game, same team | 272,359 | 1.014 | [1.005, 1.022] |

Per season (opposite teams): 0.990–0.995, every CI spanning 1.0.

Strata that looked interesting were reproduced by matched controls — i.e. they
are marginal-model artifacts, not same-game effects:

| Stratum | Same game (opp. teams) | Matched control |
|---|---|---|
| both lineup ≤ 2 | 1.016 [1.003, 1.028] | 1.022 [1.006, 1.037] |
| both p ≥ 0.70 | 0.929 [0.913, 0.947] | 0.936 [0.916, 0.956] |
| exact config: home #2 × away #1 | 1.022 [1.005, 1.040] | 1.010 [0.984, 1.040] |

Same-team pairs (shared opposing pitcher) carry a small real lift: lineup ≤2
1.031 vs control 1.022 (≈ +1%); p ≥ 0.65 0.979 vs 0.961 (≈ +2%).

## Verdict

- **Opposite-team same-game pairs are independent** (relative lift 0.9995).
  There is no positive correlation to exploit and no negative correlation to
  avoid: the "avoid correlated outcomes" rationale has no measurable basis for
  opposite-team pairs, and the rule costs marginal probability for nothing.
- Under independence the higher marginal wins outright: McNeil+Alexander ≈ 0.539
  vs McNeil+Alvarez/Bohm ≈ 0.523. Bolte (same team, 72.2%, ~+1% lift) is a
  statistical tie with Alexander.
- The recommendation survived, but the stated reason ("positive correlation
  helps") was wrong. Lesson recorded: rationale is either backtested (n, result)
  or labelled reasoning-only.

## Caveats

- Marginals are batter-season rates, not the model's `p_game_hit`; the control
  absorbs most of the resulting bias, but correlation *conditional on the model's
  features* is the exact decision quantity and could differ slightly.
- Hits include resumed portions of suspended games (negligible here; the
  BTS-scoring reader is not used because this is not contest scoring).
- No claim about the DD *selection* rule's other consequences (e.g. runner-up
  semantics); a production change to allow same-game DDs is a separate decision.
