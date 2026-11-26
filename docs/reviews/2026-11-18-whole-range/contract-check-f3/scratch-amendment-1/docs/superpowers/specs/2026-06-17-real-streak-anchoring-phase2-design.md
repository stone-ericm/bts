# Phase 2 — Uncertainty Layer — Design

**Date:** 2026-06-17
**Status:** Design. PART A (2a + 2c) is build-now; PART B (2b) is a deferred follow-up.
**Authors:** Claude (Opus 4.8); two design-review rounds by Codex (gpt-5.5)
**Builds on:** Phase 1 (shipped) — decision streak = real MLB `activeStreak`, `status`
fresh/lagged/stale, per-round ledger persisted to `contest_ledger.jsonl`.

## 0. Why split, and the honest caveat

Two Codex rounds showed the subtle, recurring difficulty lives entirely in **2b**
(the plausible-state set + invariance gate), whose correctness hinges on an
over-approximation invariant that is hard to get right *speculatively* — and which
bites RARELY (only when streak uncertainty straddles a decision boundary; never at
the current streak 8). **2a** (a behavior-neutral refactor) and **2c** (ledger saver
inference) are tractable, self-contained, and valuable on their own. So: build 2a + 2c
now (PART A); design 2b concretely once the decision-context and ledger parser are
real code, not speculation (PART B).

## 1. Observation model (the boundary that carries everything)

The central rule: **what MLB confirms** vs **what is only the bot's local
recommendation.** Conflating them re-introduces the Phase-1 inflation bug (the bot's
suggested pick is NOT evidence the user entered it — the 6/11 incident).

- **KNOWN (account evidence):** the ledger's settled `roundPredictions[]` — which
  rounds the account entered, each result, and per-round `streak`/`streakIncrease`.
- **OBSERVED:** the `activeStreak` counter (current, but can itself lag).
- **NOT evidence:** the bot's locally-delivered/resolved picks. They may **add or
  label** uncertainty (e.g. "a pick was delivered today, so today's round is in
  play"), but must **never narrow/prune** a branch and **never raise** the account
  streak. Only account evidence prunes.

---

# PART A — BUILD NOW

## 2a. `decide_action` refactor (behavior-neutral, foundational)

Today `select_pick` entangles the action decision with impure prep (locked-pick
short-circuit, live game-status filtering, the different-game second-pick *selection*
+ `game_pk` normalization, MDP policy IO). Split it:

- **`build_decision_context(predictions, date, picks_dir, allow_double, ...) ->
  DecisionContext`** — all the impure prep, lifted out of today's `select_pick`:
  live-status filtering, the locked-pick short-circuit, the **selected** primary
  candidate (+ its `p_game_hit`) and the **executable** different-game second-pick
  candidate (+ its `p_game_hit`), the loaded MDP policy, the date / days-remaining,
  the season-end date, and the `allow_double` operational clamp.
- **`decide_action(ctx: DecisionContext, streak: int, saver: bool) ->
  "skip"|"single"|"double"`** — PURE over `(streak, saver)` given `ctx`. Contains
  only: the MDP-or-heuristic branch, the `_DOUBLE_BY_STREAK` threshold, the
  post-MDP "double is executable" guard (a different-game second pick exists), the
  sprint singles-only rule, and the `allow_double` clamp (if `not ctx.allow_double`,
  never returns double). No file/cache/network IO.

`select_pick` = `build_decision_context(...)` then `decide_action(ctx, scalar_streak,
scalar_saver)` — **behavior preserved.** `allow_double` is KEPT (it is a global
operational clamp, distinct from any uncertainty logic; it lives in the context).

**Golden test:** over the backtest, scoped to the action-decision branch (excluding
the locked-pick / no-game / status-fetch short-circuits, which `build_decision_context`
owns), `decide_action` reproduces every current `select_pick` action. The MDP policy
is injected so `decide_action` is deterministic.

## 2c. Ledger saver inference (self-contained)

Replace Phase 1's "fall back to model-saver when contest & model streak agree" proxy
with a **ledger parser** + an inference, with explicit ambiguity.

- **New: `bts.contest_ledger` parser** — reads `contest_ledger.jsonl`, returns the
  per-round series `(round_date, entered?, result, pre_round_streak,
  post_round_streak, is_dd, dd_slot_results)`. NOTE: the raw `roundPredictions` give
  the post-round `streak`; `pre_round_streak` = the prior settled round's
  `post_round_streak`; **finality and scoring-correction state are NOT directly in
  the rows** — treat the latest row as provisional and require a stable two-fetch
  read before trusting a transition.
- **Saver consumed** iff a stable round has `result == not_hit`, `pre_round_streak ∈
  10–15`, and `post_round_streak` did NOT reset to 0 (the mulligan absorbed it).
- **Do NOT mark consumed (ambiguous):** a DD where only one slot missed, `void`/
  postponed rows, non-stable (single-read) rows, or an unrecoverable pre-round streak.
- **Output (this phase):** `saver_available = available | consumed | unknown`. Until
  2b's plausible set exists, `unknown` resolves to a **conservative point estimate =
  unavailable** for the live decision (don't rely on a saver we can't confirm). 2b
  will instead carry `unknown` as a set branch.
- Retire the model-saver-agreement proxy in `load_decision_streak_state`.

---

# PART B — DEFERRED FOLLOW-UP (design constraints, not built now)

## 2b. Plausible-state set + uncertainty-aware double-down

Designed concretely **after** 2a (`DecisionContext`) and the 2c ledger parser exist,
so the set is built from real structures. Hard constraints surfaced by review:

1. **Over-approximation invariant (load-bearing):** `plausible_states` must contain
   every truly-possible `(streak, saver)`. The invariance gate (double only if all
   states agree; else the **most-conservative** action, `skip > single > double`) is
   sound ONLY under this invariant.
2. **No lossy collapse:** a too-large gap must **force the conservative action**
   directly (not collapse to `{0, last_confirmed}` and then gate — that can drop an
   intermediate state, e.g. 8→11 crossing the .55→.60 threshold, and double when the
   true set forbids it).
3. **Inflation invariant (corrected):** a *local* hit must not raise
   `last_confirmed_streak`, nor prune the lower/no-entry/miss branches. A higher
   state may exist ONLY as an account-entry *possibility*, never selected from local
   evidence.
4. **No narrowing from local data:** local picks add/label branches; only account
   (ledger) evidence removes support.
5. **Explicit unposted-miss branch:** a settled-but-unposted *yesterday* miss (→0) is
   a distinct, date-keyed branch from today's unresolved pick (cf. Phase-1
   `_has_unconfirmed_miss`).
6. **Prune impossible `(streak, saver)` pairs** (saver only meaningful 10–15; a saver
   the ledger already shows consumed).
7. Carry 2c's `unknown` saver as a set branch (both values), then prune.

State-model additions (2b): `plausible_states`, `decision_confidence`,
`last_confirmed_streak` on `DecisionStreakState`; the orchestrator evaluates
`decide_action` over the set.

---

## Testing (PART A)

- **2a golden:** action-branch-scoped reproduction of every current `select_pick`
  decision over the backtest; `decide_action` deterministic given an injected context
  (MDP policy + real primary/second-pick candidates); `allow_double=False` still
  forces single.
- **2c saver:** consumed only on a stable `not_hit` with pre-round streak 10–15 + no
  reset; DD-one-slot-miss / void / single-read / unrecoverable-pre-streak → NOT
  consumed (→ `unknown` → conservative unavailable); the model-saver proxy is gone.
- **Ledger parser:** pre/post-round streak recovery; DD-slot handling; provisional
  (single-read) vs stable (two-read) rows.

## Out of scope

Same-day "not entered yet" nudge (needs MLB's live-round endpoint; operator is always
aware of their own entries). Auto-submission. Revisiting the MDP objective / skip
policy (the gate works *within* the existing policy).
