# Skip-policy shadow — settling whether the streak≥8 skip rule costs streaks (2026-06-20)

## The question

The deployed MDP **skips** (no pick) whenever, at streak ≥ 8, the top candidate's predicted
hit probability is below the **0.796** quality-bin boundary. In production this fires often —
e.g. it skipped 06-18 (best: Bo Bichette 75.0%) and 06-19 (best: Luis Arraez 78.2%), parking
the streak at 10. Is that skipping right, or is it leaving easy streak progress on the table?

## Why backtest could not answer it (the analysis that led here)

1. **Scale mismatch.** The 0.796 boundary was fit on an **actual-PA** probability scale, but
   production serves lower **estimated-PA** probabilities (≈0.04 lower). On the estimated-PA
   scale ~72–96% of top picks fall below 0.796, so they (almost) all map to the lowest quality
   bin and get skipped ("bin collapse"). The deployed threshold is mis-scaled for what it is
   applied to.
2. **Calibrated breakeven ≈ 0.744.** A calibrated estimated-PA re-solve of the MDP value
   function puts the candidate true-hit-prob at which `Q(single) == Q(skip)` at **~0.744**
   (robust 0.742–0.752 across boundaries, horizons, saver on/off; see the Q-delta diagnostic
   `/tmp/skip_qdelta.py` reproduced from the conversation, and the policy table query).
3. **The band straddles the breakeven.** The skipped candidates' realized hit rate is ~0.71–0.77
   (live production primaries ~0.706, n=34, CI [0.54, 0.83]; estimated-PA backtest ~0.744). So
   the sign of `Q(single) − Q(skip)` flips depending on the true band rate, which is **not pinned
   down** by the available data.
4. **P(57) is near-unobservable.** The objective is reaching a 57-game streak (honest P(57) ≈
   3%); in ~2 OOS seasons you expect ≈0 such streaks, and the per-policy difference is ~0.002%.
   You cannot *measure* it by replay; you can only *compute* it from per-pick hit rates through a
   model — and that computation rests on the very band calibration that is uncertain, plus
   independence assumptions, plus backtest→live transfer. (Codex reviews, 2026-06-20, pared back
   four successive over-confident readings; the durable conclusions are only "the threshold is
   mis-scaled" and "the sign is offline-unresolvable.")

**Conclusion:** the sign is only resolvable with **live data**. This shadow accumulates it.

## What this shadow is

A **counterfactual shadow POLICY** (not a shadow model — cf. the context-stack shadow model in
`shadow_eval.py` / `save_shadow_pick`). It replays the deployed decision and a **"pick-the-band"**
variant on the **same** production slate:

> shadow_action = take the **single** on days the deployed MDP would **skip** a valid top
> candidate at 8 ≤ streak < 57; otherwise mirror the deployed action.

Ground truth comes from a **decision marker the live pick path writes**, not from reconstructing
production's intent read-only. **Four review rounds** established that "the MDP evaluated an
EXECUTABLE candidate and chose to skip it" is **not** recoverable from saved slates / pick files /
the scheduler `skip_summary`: `select_pick` returns `None` for no-eligible and status-failure days
too (not only `action=="skip"`), the policy's skip region is state-dependent (not a flat
0.796/streak-8 rule), and the saved slate is pre-status-filter. So `select_pick` records the
decision at the source. This is a small **additive** cascade write — it records, it never changes a
pick.

### Mechanics (`bts/skip_policy_shadow.py` + the `select_pick` marker write)

- **Skip marker (ground truth):** at a genuine `action == "skip"`, `select_pick` calls
  `record_mdp_skip_decision` → `data/picks/<date>/skip_decision.json` (schema
  `bts_mdp_skip_decision_v1`) with the **executable** declined candidate (the post-status-filter,
  notna `best_row`) + streak/saver. Best-effort (wrapped; never raises / never affects the pick).
  It fires **only when `persist_skip_decision=True`** — which **only the scheduler's production
  `run_and_pick` passes** (NOT `bts preview`, manual `bts run`, or the shadow model) — **AND only
  for an MDP-backed decision** (`ctx.mdp` truthy; a missing policy caches to `{}` → heuristic skip →
  no marker). So provisional / non-production / heuristic skips never write a marker.
- **Final-decision resolution (delivery, not existence/chronology):** BOTH the skip marker and the
  `<date>.json` pick file are provisional/overwritten — `bts preview` pre-writes the *next* day's
  pick file, the scheduler saves candidates pre-lock, and the fallback re-delivers a cached pick
  (with a stale `run_time`, which is why a timestamp comparison was insufficient). The authoritative
  signal is **`picks.pick_was_delivered`** (a pick durably posted/DM'd to a human; set by
  `_deliver_and_lock_pick`, incl. the fallback path). `_final_decision`: a **delivered** pick →
  "pick"; else a skip marker → "skip"; else None. A skip is recorded only when it's the final
  decision; a record whose date later resolves to a delivered pick is deleted (`prune_superseded`);
  recording is idempotent. (Production runs public/DM delivery, so a real final pick sets the
  delivery fields; a private/local-only lock is intentionally not treated as "delivered".)
- **Decision files:** `data/picks/<date>.policy_shadow.json` (schema `bts_skip_policy_shadow_v1`),
  one per genuine skip: always divergent (deployed=skip, shadow=single), `rank1` = the marker's
  executable candidate, streak/saver (context), `shadow_pick_result` (filled at reconciliation).
- **Reconciliation:** the realized outcome of the skipped candidate is resolved via the MLB API
  (`picks.check_hit`) → hit/miss. **A not-yet-final game stays *pending* and is retried** (a live
  west-coast game at the nightly cron is never lost); a checker error is treated the same; a record
  still unresolved after `STALE_AFTER_DAYS` (3) is marked `void`.
- **Backfill:** `record_pending_skips` records any skip marker in the last 10 days that lacks a
  decision file, so a cron outage doesn't drop a day.
- **Status artifact:** `data/validation/skip_policy_shadow_status.json`
  (schema `bts_skip_policy_shadow_status_v1`). Headline = the realized hit rate of the skipped
  candidates with a Wilson CI, plus a **verdict** vs the 0.744 breakeven:
  - `below_breakeven` — picking the band is −EV → **the skip is validated**.
  - `above_breakeven` — picking the band is +EV → **skipping is costing streaks**.
  - `straddles_breakeven` / `insufficient_n` — not yet resolvable (default until ≥30 resolved
    divergent days).

### Surfaces (rides the existing shadow rails — grep `skip_policy_shadow` / `policy_shadow`)

- **CLI:** `bts skip-policy-shadow-update` (nightly: record today + reconcile pending + refresh
  status) and `bts skip-policy-shadow-status` (print the verdict).
- **Dashboard:** a "Skip-policy shadow" panel on `:3003` (`web.render_skip_policy_shadow_section`).
- **Cron:** 23:30 ET in `scripts/cron-setup-hetzner.sh` (records same-day; late west-coast
  outcomes reconcile the following night).
- **Docs:** this file + ARCHITECTURE.md + CLAUDE.md.

## Honest caveats (what a positive/negative verdict will and won't prove)

- The divergence is the **genuine MDP skip recorded by the marker** (with pick-precedence — a later
  pick supersedes it), not an inference from the live streak; the recorded streak/saver are context
  only and can't flip a day's classification.
- **Candidate eligibility — resolved.** The marker records the *executable* post-status-filter
  candidate the MDP actually declined (not the pre-filter slate rank-1), so the reconciled outcome
  is for the pick production could have taken. (The marker is overwritten across a day's repeated
  scheduler runs — the last genuine skip wins — and pick-precedence covers skip-early/pick-late.)
- A verdict only becomes meaningful once enough **resolved divergent days** accumulate (default
  gate: 30). Until then, `insufficient_n` is the honest state — do not read the interim rate as
  signal.
- This measures the **local per-decision EV** of picking the band (does the skipped candidate
  clear breakeven?). It does **not** simulate a fully divergent streak trajectory; the breakeven
  framing already converts the band hit rate into the policy verdict, and the saver makes a miss
  at streak 10–15 non-fatal, so the local view is the decision-relevant one.
- The 0.744 breakeven is from a calibrated estimated-PA re-solve; if the model/era changes,
  re-derive it (the value is a module constant `BREAKEVEN_P`).

## Related pre-existing bug (deferred — separate fix)

Investigating the provisional-pick-file problem surfaced a **pre-existing production bug, independent
of this shadow**: `bts check-results` scores whatever `load_pick(date)` returns, gated only on
"already resolved." On a day whose final decision was a SKIP, a stale `bts preview` / pre-lock
`<date>.json` is therefore scored and **the streak is updated from a pick production never made**.
A one-line `pick_was_delivered` gate is NOT a clean fix: `pick_was_delivered` excludes private/local
delivery (would regress private-mode scoring), and it breaks 7 existing `check-results` tests that
score undelivered fixtures. The clean fix is the **explicit end-of-day decision record** Codex's
design review identified (`{date, action, source, delivered, pick_ref, finalized_at}`) consumed by
both `check-results` and this shadow — a broader lifecycle change tracked separately, not bundled
into this shadow.
