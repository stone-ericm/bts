# Real-Streak Anchoring — Design

**Date:** 2026-06-17
**Status:** Design (awaiting review)
**Authors:** Claude (Opus 4.8); two adversarial review rounds by Codex (gpt-5.5)

Split into **Phase 1** (contained fix — build now) and **Phase 2** (uncertainty
layer — deferred). The split is deliberate: Phase 1 resolves the incident and
everything asked for, with no strategy-engine refactor; Phase 2 is a genuine
refactor whose payoff is a correctness guard near policy boundaries (a case the
user is nowhere near at streak 8).

## 1. Problem

The dashboard reported the user's streak as **10**; the real MLB streak was **8**.
Pick recommendations (incl. double-down) ran off the wrong number, and double-downs
were frozen.

Incident (2026-06-17): the user missed entering the bot's 6/11 suggestion (a
double-down). MLB's per-round ledger confirms no 6/11 entry; every other day's
single/double matches the bot. The bot's "model streak" (forward-replay of its own
*suggested* picks, assuming all entered + resolved) read 10; real MLB `activeStreak`
was 8. The gap = exactly the missed 6/11 double-down (+2).

## 2. Root cause

**(a) `max(model, contest)` inflation.** `contest_state.load_decision_streak_state`
branches: contest-fresh → use contest; contest-stale → `max(model_streak,
contest_streak)` + forbid doubles; no-contest → use model. The contest observation
is "stale" almost every evening (the bot resolves a game locally before MLB posts
the settled result), so the stale branch took `max(10, 7) = 10` — trusting the
inflated model over the real streak — and froze doubles.

**(b) The fetch discards a correct current streak — located in the CLI gate, not
`build_observation`.** `build_observation` persists `activeStreak` fine. The
rejection is the **currentness gate in the `fetch-contest-streak` CLI command**: it
compares `source_date` (from `contest_fetch.derive_source_date`) against the latest
local resolved pick and refuses to write when `source_date <` it. `derive_source_date`
reads the latest *settled row* in the per-round `predictions` array, which lags
MLB's own `activeStreak` counter by ~a round — so a fetch pulled `activeStreak=8`
but the gate refused it ("source_date 2026-06-15 < latest resolved pick 2026-06-16"),
leaving the stale stored 7 to feed (a).

**(b′) `not_hit` vocabulary bug — profile only.** `contest_fetch.RESOLVED =
{"hit","miss","void"}` omits MLB's actual profile value `"not_hit"`, so
`derive_source_date` skips settled *misses* → biased freshness on reset days. NOTE:
`contest_state._RESOLVED_RESULTS = {"hit","miss","void"}` is a *separate* set that
operates on **local pick files**, which use `"miss"` (verified: 6/5 and 6/9 local
files), so it is **correct** — do not change it. Two vocabularies; fix only the
profile one. (Verify local-file vocab during implementation.)

**(c) Conceptual flaw.** The system conflated "what the bot suggested" (hypothesis),
"what MLB confirms" (truth, lagged), and "what to use for tonight's decision"
(judgment under uncertainty). Phase 1 separates the first two cleanly; Phase 2
handles the third.

---

# PHASE 1 — Contained fix (build now)

Decision streak reflects the real MLB streak; the fetch stops discarding correct
data and persists the ledger; doubles respond to the real streak; the dashboard is
labeled. **No strategy refactor, no plausible-set, no saver inference** — those are
Phase 2.

## P1.1 Decision streak — kill `max`, model never raises

- Decision streak is **always** the contest (MLB) value (snapshot / last-confirmed).
  The model **can never raise** it. `max(model, contest)` is removed.
- Compute a simple `status ∈ {fresh, lagged, stale}`:
  - `fresh` — contest covers the latest decision-relevant round.
  - `lagged` — ≤1 round behind (normal overnight settlement): hold the contest
    value, **doubles stay enabled**.
  - `stale` — ≥2-round gap, fetch failing, **or the local model shows a reset the
    contest hasn't confirmed** (stale-high risk). Decision streak still = contest
    (last confirmed); `status=stale` drives the dashboard caveat + health signal.
- **Phase-1 limitation (explicit):** when `stale`, strategy still uses the contest
  number (no plausible-set yet). The stale-high-after-an-unposted-miss edge is
  surfaced (display + health) but not acted on by strategy until Phase 2; meanwhile
  it's covered operationally (the 6/17 override-guard pattern) and by the user's
  own awareness. At low streaks the action barely differs (0 vs 8 are both
  aggressive); near a boundary is exactly what Phase 2 closes.

## P1.2 Fetch fixes

- **Split snapshot from coverage.** Persist `{activeStreak, seasonBestStreak,
  recorded_at}` as a snapshot that is **always** writable; derive `source_date` /
  ledger coverage separately and allow it to be partial/None **without discarding
  the snapshot**. (Resolves the `build_observation` "source_date is None" refusal
  vs. "persist activeStreak despite predictions lag" conflict.)
- **Fix the CLI currentness gate** so a current `activeStreak` is accepted on the
  basis of the snapshot (counter value + `recorded_at`), not solely `source_date`
  vs latest local pick. Keep the fail-safe (never overwrite a good value with a
  malformed/auth-failed fetch; identity guard).
- **Normalize profile outcomes** (`hit`/`not_hit`/`void` + explicit pending/unknown)
  so `derive_source_date` counts settled misses.
- **Persist the full per-round ledger** each fetch (snapshot keyed by round +
  `recorded_at`) to a data file for analysis and Phase-2 saver inference.

## P1.3 Remove the blanket DD freeze

- `load_decision_streak_state` no longer sets `allow_double=False` on staleness. DD
  follows the (real) streak through the normal policy. The streak-based no-double in
  the sprint near 57 is separate and unchanged. (The boundary-aware uncertainty
  guard is Phase 2.)

## P1.4 Manual-override semantics (define explicitly)

The expiring `contest_streak.manual.json` override is an **operator assertion of the
real MLB streak**, treated as a confirmed contest observation: it carries a
`source_date` used for coverage, takes precedence over auto until `override_expires_at`
(existing behavior), and is assigned `status=fresh` when its `source_date` covers the
latest decision-relevant round, else `lagged`/`stale` by the same rule as auto. Stale
auto data does **not** replace an unexpired override. (This covers the override set
for the 6/17 incident.)

## P1.5 Dashboard

- Real streak is **the** number, labeled with as-of/confidence: `fresh` → "8";
  `lagged`/`stale` → "**last confirmed 8 through 6/16**" (and, when stale-high is
  possible, "current may be lower") — never an unqualified absolute below fresh.
- Model what-if streak shown as a separate, clearly-labeled research line.

## P1.6 Data preservation

- Keep computing `streak.json` (model series) as-is — research data, not deleted,
  not "corrected to reality."
- Persist the per-round ledger snapshots. Keep model + real series in parallel; the
  divergence (missed entries, single-vs-suggested-double) is signal.

## P1.7 Phase 1 explicitly EXCLUDES

Plausible-`(streak, saver)`-set; boundary-aware DD evaluation; the `select_pick`
refactor; saver inference from the ledger; `last_confirmed`/`confidence` as a
strategy input. All Phase 2.

## P1.8 Phase 1 testing (TDD, red→green)

- **Keystone regression:** model 10, contest 8 (stale) → `decision_streak == 8`,
  doubles **enabled**. Fixture from the real per-round ledger.
- Decision streak: model never raises it; `status` fresh/lagged/stale computed;
  staleness does **not** freeze doubles.
- Fetch: `not_hit` normalization (reset-day `source_date` correct); snapshot
  persisted even when `source_date` is None / predictions lag; CLI gate accepts a
  current `activeStreak`; ledger snapshot written; fail-safe + identity guard intact.
- Two-vocabulary regression: the local `"miss"` path (`_RESOLVED_RESULTS`) is
  unaffected by the profile `not_hit` fix.
- Manual override: treated as confirmed observation with its `source_date`;
  precedence/expiry intact; `status` assigned; stale auto can't replace it.
- Dashboard: labeled real streak; "last confirmed … through …" below fresh; model
  line separate.

---

# PHASE 2 — Uncertainty layer (deferred; own spec when started)

The robustness layer Codex's review showed is a real refactor, not a branch swap.

- **`last_confirmed_streak` + `decision_confidence` + a plausible `(streak, saver)`
  SET** — disjoint, not an interval (an unconfirmed local miss could mean real-miss /
  no-entry / different-entry / single-instead-of-DD / saver-consumed; a DD jumps the
  upper bound +2, a miss resets to 0). Derived from confirmed vs unconfirmed events.
- **Strategy refactor:** extract the action decision into a pure helper
  (`decide_action(p_game_hit, streak, saver, date, predictions, …)`) that both
  `select_pick` and the new layer call. Today `select_pick` takes a scalar streak and
  computes internally; `_mdp_action` returns one action per tuple and exposes no
  "boundaries" (those exist only in the heuristic `_DOUBLE_BY_STREAK`).
- **Boundary-aware DD (the real version):** evaluate `decide_action` across the
  plausible set; commit to a non-conservative action **only if it is invariant across
  the set**; otherwise single / mark uncertain. This works for both MDP and heuristic
  without enumerating analytic boundaries.
- **Principled saver handling:** infer consumption from the ledger only with
  corroboration (before/after streak delta, DD-slot semantics, void/postponement
  handling) — "`not_hit` that didn't reset ⇒ saver used" is a candidate, not proof.
  Explicitly **retire or constrain** the existing "fall back to model-saver when
  contest and model streak agree" proxy.
- **Schema/contract change** to carry the uncertainty set into strategy.
- **Phase 2 testing:** MDP action behavior; the no-second-pick double→single
  downgrade; plausible-set construction; action-invariance gate; saver inference;
  override precedence/expiry under the confidence model.

---

## Non-goals (global)

- **Same-day "not entered yet" nudge.** The profile endpoint empirically returns
  only *settled* rows (verified by the 6/12 incident), so although `has_prediction_for`
  is *coded* to detect pending rows, the data never contains them — same-day detection
  needs MLB's live-round endpoint (traffic inspection). Out of scope.
- **Missed-entry DM** ("you forgot 6/11") — the user is always aware of their entries.
- **Auto-submission** of picks to MLB.

## Rollout

- TDD on a feature branch off `main`; contest/cli/health suites green (model tests as
  available). Ship via `git push origin main:deploy` (canary + auto-rollback).
- Planning note: P1.2's currentness-gate fix lives in the `fetch-contest-streak` CLI
  command (`cli.py`) and P1.5 in the dashboard (`web.py`) — inspect both live files
  during implementation; only `contest_state`/`contest_fetch`/`strategy` were reviewed
  in detail here.
- The 6/17 manual override + box-side guard (`/tmp/override_guard_2026-06-17.py` on
  bts-hetzner, nohup) remain until Phase 1 lands; remove once deployed.

## Codex (gpt-5.5) review provenance

**Round 1 (design):** kill `max(model, contest)`; "anchor whenever contest exists"
too loose (stale can over-state after an unposted miss → last-confirmed + confidence);
don't just remove the DD freeze — replace it; freshness ≠ the `activeStreak` integer;
`RESOLVED`/`not_hit` is a real bug; saver state was missing; corrected overstatements
(model not "permanently +2"; "never over-state" only vs the model; "DD band flat 0–9"
is MDP-incident-specific).

**Round 2 (spec vs code):** §6 boundary-aware DD not implementable without a
`select_pick` refactor (scalar streak in, MDP exposes no boundaries); the
`last_confirmed`/confidence model is a real schema+contract change, not a branch swap;
the "range" is a disjoint set; saver inference too confident; the discard gate is in
the CLI not `build_observation`; two separate `RESOLVED` vocabularies; `source_date
is None` refusal conflicts with persisting `activeStreak`; manual-override semantics
undefined; dashboard wording can still overstate; testing too thin. → Drove the
Phase 1/Phase 2 split and the precision fixes above.
