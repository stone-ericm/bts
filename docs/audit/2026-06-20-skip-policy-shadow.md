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

Ground truth comes from a **decision record the scheduler writes to `decision.json`**, not from
reconstructing production's intent read-only. **Four review rounds** established that "the MDP
evaluated an EXECUTABLE candidate and chose to skip it" is **not** recoverable from saved slates /
pick files / the scheduler `skip_summary`: `select_pick` returns `None` for no-eligible and
status-failure days too (not only `action=="skip"`), the policy's skip region is state-dependent
(not a flat 0.796/streak-8 rule), and the saved slate is pre-status-filter. So the scheduler
records the final decision in `decision.json`. This is a small **additive** write — it records,
it never changes a pick.

### Mechanics (`bts/skip_policy_shadow.py` + `bts/daily_decision.py`)

- **Ground truth via `decision.json`:** the scheduler writes
  `data/picks/<date>/decision.json` (schema `bts_daily_decision_v1`) at each true finalization
  point. There are four write points: (1) **pick commit** — `_deliver_and_lock_pick` delivery
  branches write `scoreable=True` with `delivery_status` ∈ `{delivered, private_locked,
  locked_unconfirmed}`; (2) **classification-lock** — a pre-existing pick locked by game-start/
  status is recorded only when `pick_was_delivered` is true (non-delivered stale previews are
  silently skipped, writing nothing); (3) **crash-guard** — best-effort at abnormal exit; and
  (4) **end-of-day MDP skip** — `_write_endofday_skip` fires when `committed_pick_written`
  is still False at EOD, writing `action="skip", source="mdp", scoreable=False,
  delivery_status="not_applicable"` with the `final_skip_candidate` captured at the genuine MDP
  skip earlier in the day. The scheduler tracks `committed_pick_written` + `final_skip_candidate`
  across the day. All writes are best-effort (wrapped; never raise into the pick path) and written
  by the scheduler ONLY — `bts run`, preview, and the shadow model never write `decision.json`.
- **`check-results` scoreable gate (GH #144):** `bts check-results` reads `decision.json` first;
  if present, it scores only when `decision.get("scoreable") == True`. Fallback (no decision
  file): `picks.pick_was_delivered(daily)`. This eliminates the bug where a stale `bts preview`/
  pre-lock `<date>.json` on a skip day was scored and corrupted the streak.
- **Shadow reads `decision.json`:** `record_skip_from_decision` reads the day's `decision.json`
  and writes `{date}.policy_shadow.json` only when `action=="skip" && source=="mdp"`. Non-MDP
  skips, delivered picks, and days without `decision.json` produce no shadow record.
  `prune_superseded` removes shadow records whose `decision.json` has been overwritten to a
  delivered pick. Recording is idempotent; a cron outage is covered by `record_pending_skips`
  (scans `*/decision.json` for recent MDP skips lacking a shadow file).
- **Decision files:** `data/picks/<date>.policy_shadow.json` (schema `bts_skip_policy_shadow_v1`),
  one per genuine skip: always divergent (deployed=skip, shadow=single), `rank1` = the decision's
  executable candidate, streak/saver (context), `shadow_pick_result` (filled at reconciliation).
- **Reconciliation:** the realized outcome of the skipped candidate is resolved via the MLB API
  (`picks.check_hit`) → hit/miss. **A not-yet-final game stays *pending* and is retried** (a live
  west-coast game at the nightly cron is never lost); a checker error is treated the same; a record
  still unresolved after `STALE_AFTER_DAYS` (3) is marked `void`.
- **Backfill:** `record_pending_skips` records any MDP skip in `decision.json` within the lookback
  window that lacks a shadow file, so a cron outage doesn't drop a day.
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

- The divergence is the **genuine MDP skip recorded in `decision.json`** (with pick-precedence — a
  later delivered pick causes `prune_superseded` to remove it), not an inference from the live
  streak; the recorded streak/saver are context only and can't flip a day's classification.
- **Candidate eligibility — resolved.** The `decision.json` record contains the *executable*
  post-status-filter candidate the MDP actually declined (not the pre-filter slate rank-1), so the
  reconciled outcome is for the pick production could have taken. (`decision.json` is overwritten
  across a day's repeated scheduler runs — the last genuine skip wins — and `prune_superseded`
  covers skip-early/pick-late.)
- A verdict only becomes meaningful once enough **resolved divergent days** accumulate (default
  gate: 30). Until then, `insufficient_n` is the honest state — do not read the interim rate as
  signal.
- This measures the **local per-decision EV** of picking the band (does the skipped candidate
  clear breakeven?). It does **not** simulate a fully divergent streak trajectory; the breakeven
  framing already converts the band hit rate into the policy verdict, and the saver makes a miss
  at streak 10–15 non-fatal, so the local view is the decision-relevant one.
- The 0.744 breakeven is from a calibrated estimated-PA re-solve; if the model/era changes,
  re-derive it (the value is a module constant `BREAKEVEN_P`).

## GH #144 fix — `check-results` scoreable gate (landed on this branch)

Investigating the provisional-pick-file problem identified a **pre-existing production bug**: `bts
check-results` scored whatever `load_pick(date)` returned, gated only on "already resolved." On a
day whose final decision was a SKIP, a stale `bts preview` / pre-lock `<date>.json` was therefore
scored and **the streak was updated from a pick production never made**.

This is fixed in this branch via the `decision.json` scoreable gate: `check-results` reads
`decision.json` first and scores only when `scoreable == True`; fallback to
`picks.pick_was_delivered(daily)` when no decision file exists. The `decision.json` record is
written only by the scheduler at genuine finalization — so preview pre-writes and pre-lock
candidates no longer trigger scoring.
