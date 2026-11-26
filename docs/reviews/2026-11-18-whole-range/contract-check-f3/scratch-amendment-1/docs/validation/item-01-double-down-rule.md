# Item 1: 2026 Removable Double-Down Rule

**Verdict:** NOT CONFIRMED

## Finding

The official MLB BTS rules (https://www.mlb.com/apps/beat-the-streak/official-rules) explicitly state:

> "You may not add a second prediction for a Double Down Pick after the Lock Time of your first prediction has passed."

> "You cannot change either of your Double Down Pick predictions after the earliest applicable Lock Time."

> "If both players get a hit, your streak improves by two. But if either of them fails to record a knock, your streak reverts back to zero."

There is no mechanism to remove or cancel a second pick after the first game starts. The rules are identical to 2025.

## MDP Implications

Our current MDP transitions are correct:
- P(+2) = P1 × P2
- P(reset) = 1 − P1×P2

There is no P(+1) partial-hit path from double down. Task 15 (MDP re-solve with removable double) is cancelled.

## Community Note

The r/beatthestreak community appears to have conflated lock-time mechanics (you can add a second pick before your first game locks) with a removal mechanic (you can remove the second pick after the first game starts). These are different — once both picks are locked, both must hit.
