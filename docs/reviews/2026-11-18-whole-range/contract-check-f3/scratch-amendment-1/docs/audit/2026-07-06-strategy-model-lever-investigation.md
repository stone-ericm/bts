# 2026-07-06 — Strategy / model / PIT lever investigation (decision record)

## TL;DR

Starting from a scheduler false-alert, this thread explored whether there's a
high-value improvement in BTS strategy (the MDP) or the pick model, and whether a
point-in-time (PIT) backtest platform is worth building. **Conclusion: no big
lever, don't build PIT.** The strategy layer is ~a wash (deployed MDP ≈
always-double), the pick model is near its achievable ceiling (~0.53 AUC,
leak-free), and the existing tooling (`estimated_pa` profiles, `backtest_blend`
walk-forward, the M3 serving-parity replay) already answers the questions that are
answerable. The concrete win of the session was the shipped `check-pick-entered`
alert fix (separate doc:
`2026-07-06-deferred-dd-and-premature-entry-alert.md`).

This record exists so the next session does not re-derive all of the above.

## How we got here

`check-pick-entered` fired a premature "pick not entered" DM for a deferred
double-down (early confirmed leg + late projected leg). Fixing the alert
(shipped) raised the deeper question: *should the scheduler deliver a DD early on
a projected lineup ("deliver early on projection")?* That unpacked into: a
2-gate decision (lock-early-vs-wait + commit-projected-partner), which the objective
should be derived empirically (multi-arm backtest), which surfaced that the
deployed objective (P(57)) may be wrong for the real win condition (first-to-57
else longest streak), which implied reopening the MDP — at which point the repo's
own prior audit stopped us.

## What the evidence actually says (verified this thread)

1. **Strategy layer ≈ a wash.** On the realistic `estimated_pa` profiles
   (`data/hetzner_results/mdp_estpa_run`, 24 seeds × 5 seasons, rank-1 hit ~0.75,
   `game_pk` present), the deployed MDP vs a trivial always-double:
   - always-double 18.37 mean-max / reach-20 42.5% vs MDP 18.03 / 30.8% (reproduces
     the 2026-06-10 audit exactly).
   - **With the different-game DD constraint** (production's real rule), the
     mean-max edge shrinks to **~+0.12** (coin-flip head-to-head), and the reach-30
     tail is under-powered (MDP 2 vs AD 1 of 120). always-double wins the *low*
     milestone (reach-20) but its ~76 resets/season likely cap the *tail*.
   - Nobody reaches 57 (0/120) → P(57) is a phantom in realized replay; the
     longest-streak / milestone objective is the right one. For *competitive*
     longest-streak the high tail matters most, and there always-double is not
     clearly ahead.
   - Confirmation script: `scripts/audit/confirm_mdp_policy_replay.py` (if kept).

2. **The pick model is near ceiling — the "+9pp starter lever" was leakage.** The
   2026-06-10 audit CAPSTONE (lines 103-110) leak-corrects its own headline: the
   "+0.019 AUC / +9pp opposing-starter signal" was **mostly leakage** (bucketed by
   the pitcher's *full-season* hit-allowed, which includes games after the pick).
   Leak-free lift ≈ **+0.004-0.006 AUC**. The audit "rigorously RULED OUT three
   levers — MDP re-solve, strategy layer, and a better starter feature — as
   meaningful wins. Negative result." Model AUC ~0.53 is intrinsically hard.

3. **The existing tooling already suffices for the answerable questions.**
   - `backtest_blend` is walk-forward (trains on `train_pool + test[date<day]`,
     `backtest_blend.py:819`) and has an `estimated_pa` production-basis mode
     (`:764`) — good enough for model-vs-model and policy comparison.
   - `estimated_pa` fixes the dominant optimism: the `actual_pa` basis
     (`_actual_pa_game_predictions`, `:568`) compounds per-PA hit prob over the
     batter's *realized* PA count (hindsight → rank-1 hit 0.865); `estimated_pa`
     estimates PAs from lineup slot + starter matchup (→ realistic 0.75).
   - The M3 serving-parity replay (`replay_m3_serving_parity.py`,
     `2026-06-11-m3-serving-staleness.md`) is the right *targeted* shape for any
     serving-gap question and concluded HOLD, not a broad rebuild.

## The PIT decision: DO NOT build PIT (now)

PIT = a strict as-of-(D,T) point-in-time backtest (date-bounded training + every
feature computed only from data available at D + decision-time lineup
reconstruction). It was proposed to (a) verify policies, (b) verify the model, (c)
enable the timing question. Each justification collapsed:

- **Policy verification:** `estimated_pa` already answers it. Redundant.
- **Model verification:** the model is near ceiling (leak-free) — no headroom to
  verify toward. The primary justification is void.
- **Validation-baseline was circular:** "PIT reproduces `estimated_pa`" proves
  shared code/reconstruction bias, not truth.
- **Only the timing question** genuinely needs decision-time reconstruction — and
  that is a *targeted* piece (M3-style), not a platform.

Verdict: building a PIT platform would mostly re-derive a documented negative
result. If a leak-clean model comparison is ever needed, the cheaper path is a
targeted leak-audit MVP on top of `backtest_blend` (date-bounded feature
materialization + patch the known future-peeking lookups: `run_pipeline`
full-parquet load `predict.py:825`, `.last()` lookups `:217`, `days_rest` max,
pitcher-from-plays, weather-from-feed) — reusing the M3 pattern. Not a platform.

## Open threads (bounded value; pursue only deliberately)

Ranked by upside-per-effort:

1. **Statcast miss-distance feature** — the ONE bet with real upside vs a
   near-ceiling model, because it's an *orthogonal* signal (live contact/stuff
   quality, not recycled historical hit-rate). Caveats: coverage mid-2023+ only
   (BTS trains 2019+), swing-conditional; must beat a ~0.53-AUC model.
   **Feasibility CONFIRMED (2026-07-06):** the bat-tracking fields are already in
   local `data/processed/swing_{2023..2026}.parquet` — `miss_distance`,
   `swing_length`, `attack_angle`, `swing_path_tilt`, ball-minus-bat intercepts
   (712k swing-rows/season). So the natural feature — a **pitcher
   contact-suppression** signal (opposing starter's rolling induced miss-distance,
   aggregated to the pitcher) — is buildable with **no new Savant pulls**. There is
   already a mature **Statcast swing campaign** (`features/swing.py`, existing
   `batter_whiff_60g`, `docs/superpowers/specs/2026-06-12-statcast-swing-*`) with a
   hard controls gate (sentinels + null arms). **Next step:** its own feature-spec
   brainstorm → campaign controls gate. NOT started; deliberately deferred to a
   fresh focused session (avoid tail-of-thread over-escalation).
2. **`estimated_pa` aggregation compression** (~+0.006) — the audit's one remaining
   "live" model thread: the `1−(1−p)^n` game-aggregation compresses per-feature
   signal; un-compressing (adding raw rolling features linearly) recovers a sliver,
   possibly across features, not just pitcher. Small, data-available.
3. **DD early-lock timing reconstruction** — the original trigger. Needs targeted
   decision-time projected-vs-confirmed lineup state (not full PIT). Bounded value
   (rescues the occasional strong-early-member DD), and low leverage given the
   strategy layer is ~a wash.

## Process lesson

This thread twice "got ahead of the data" — (a) declared the MDP fine on the wrong
(hindsight `actual_pa`) profiles, and (b) cited the "+9pp model lever" without its
own leakage walk-back. Both caught by disciplined Codex checks. This is the *same*
failure mode the 2026-06-10 audit documents about itself. Lesson: in this problem
space, use leak-free / correct-basis measures from the START, and check surprising
positive results before acting on them.
