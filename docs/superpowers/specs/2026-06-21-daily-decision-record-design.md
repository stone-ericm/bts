# Daily decision record — authoritative end-of-day decision (fixes GH #144)

**Status:** design approved 2026-06-21 (brainstorming + Codex design review). Implements the fix
for GH issue #144 and migrates the skip-policy shadow (commit `d73face`) onto a single authority.

## Problem

`bts check-results` scores whatever `load_pick(date)` returns (gated only on "already resolved"),
so on a day whose **final** decision was a SKIP, a stale `bts preview` / pre-lock `<date>.json` is
scored and **the streak is updated from a pick production never made** (GH #144).

Root cause (established across ~9 review rounds on the skip-policy shadow): **production has no
single authoritative record of what it finally decided each day.** Every existing artifact is
provisional — `<date>.json` is pre-written by `preview` and saved before lock; the skip marker is
overwritten; `pick_was_delivered` misses private mode; `scheduler_state.pick_locked` is spread
across paths. Both `check-results` and the shadow reverse-engineer the final decision and keep
getting it subtly wrong.

## Solution: one authoritative record, written by the scheduler at true finalization

`data/picks/<date>/decision.json` (schema `bts_daily_decision_v1`), the **single** source of truth
for "what did production finally do on `<date>`". Written **only by the scheduler**, **only at a
genuine finalization point**, so it is never provisional. Both consumers read it.

### Schema

```json
{
  "schema_version": "bts_daily_decision_v1",
  "date": "2026-06-20",
  "action": "skip" | "single" | "double",
  "source": "mdp" | "heuristic" | "unknown",
  "primary": {"batter_id":..., "batter_name":..., "team":..., "game_pk":..., "p_game_hit":...} | null,
  "double_down": {... same shape ...} | null,
  "streak": int | null,
  "saver_available": bool | null,
  "delivery_status": "delivered" | "private_locked" | "locked_unconfirmed" | "not_applicable",
  "scoreable": true | false,
  "finalized_at": "2026-06-20T23:50:00Z"
}
```

- `action` — the final action. `primary` is the executable declined candidate on `skip`, the
  chosen primary on a pick. `double_down` is populated only for `double` (a singular candidate is
  not authoritative for doubles).
- `source` — `mdp`/`heuristic` from the **resolved decision** (not inferred after the fact), or
  **`unknown`** on recovery paths (lock-by-classification, crash-guard) where no `select_pick` ran
  and `DailyPick` carries no source field. The shadow only acts on `source=="mdp"` *skips*, so
  `unknown` on a pick is harmless.
- `streak` / `saver_available` — captured on `skip` (the shadow's status rows display `streak`);
  `null` on recovery.
- `delivery_status` — the lock/delivery reason: `delivered` (posted/DM'd, `pick_was_delivered` true —
  also used when a genuine delivered pick is recovered via classification-lock), `private_locked`
  (private-mode commit — locked, not posted/DM'd), `locked_unconfirmed` (`delivery_attempted`
  crash-guard), `not_applicable` (skip). A **non-delivered** classification-lock (a stale preview file
  locked by game-start/status-failure) writes **no record** at all, so it has no status.
- `scoreable` — the gate `check-results` reads: **true for every committed-pick variant** (all
  `delivery_status` except `not_applicable`), **false for skip**.
- **Absence is meaningful:** no `decision.json` ⇒ "no authoritative production decision" (no games,
  no eligible candidates, valid-prediction/status failure *before* an action was chosen, or a
  pick selected but never committed). This is NOT a skip (Codex #6).

## Producers

### `strategy.select_pick` returns decision metadata (no file writes)

`select_pick` stops writing any file (drops the `skip_decision.json` marker and the
`persist_skip_decision` param). It returns a `SelectionResult`:

```
SelectionResult(pick_result: PickResult | None,
                action: "skip"|"single"|"double"|None,   # None when no action was reached
                source: "mdp"|"heuristic"|None,
                primary_candidate: dict | None,           # executable best_row
                double_candidate: dict | None,
                no_pick_reason: str | None)                # e.g. "no_eligible", "status_failure"
```

`source` comes from `decide_action` resolving via MDP vs heuristic (thread the actual source out of
`_mdp_action_from`/`decide_action`, do not infer from `ctx.mdp`). Callers update:
`orchestrator.run_and_pick` (returns the metadata alongside predictions/result/tier),
`cli` run/preview, the scheduler shadow-model call, and tests.

### The scheduler is the single writer, at true finalization points

`run_day` maintains two **finalization-state** variables and writes `decision.json` exactly once, at
the genuine finalization, best-effort (`try/except`, never affects the pick path):
- `final_skip_candidate` — the latest genuine MDP skip's metadata, or cleared.
- `committed_pick_written` — set True when a **scoreable pick** record is written (a genuine commit).
  Tracked explicitly because **`scheduler_state.pick_locked` is NOT a reliable "committed a pick"
  signal** (Codex r2 HIGH): `classify_pick_lock_state` also locks on `game_started_or_final` /
  `status_lookup_failed` / `fallback_status_locked`, so a lingering `bts preview` `<date>.json` can be
  classified-locked with no genuine commit.

**`final_skip_candidate` lifecycle — narrow** (Codex r2 P0: an over-broad clear erases the skip in exactly
the #144 case, where an early genuine MDP skip is followed by a stale-preview classification-lock):
- a cycle that returns a genuine MDP skip (`metadata.action=="skip" and source=="mdp"`) → **set** it
  (the primary candidate + `streak`/`saver_available`).
- a cycle that **selects or attempts a pick** (`metadata.action in {single,double}`) → **clear** it
  (the day's intent flipped to a pick).
- **every other outcome leaves it unchanged**: no selection (`sel is None` — no-predictions /
  `ContestStateError`); a **non-delivered classification-lock** (a stale/preview `<date>.json`,
  `action is None` — the #144 case, where the earlier MDP skip must survive to be recorded at end-of-day);
  a `no_pick_reason` result. Suppression of the end-of-day skip when a pick *was* committed is handled by
  `committed_pick_written` (write points 1–3), NOT by clearing the candidate.

**Write points** (a scoreable pick record also sets `committed_pick_written`):
1. **Pick committed** — every `_deliver_and_lock_pick` success branch + the two fallback call sites:
   `action=single|double` (from `daily.double_down`), `primary`+`double_down`, `source` from the
   captured metadata, `delivery_status` (`delivered` public/DM, `private_locked` private),
   `scoreable=true`. Loop breaks.
2. **Lock-by-classification** — `run_single_check` returns an existing pick locked
   (`classify_pick_lock_state`) / `run_day` sets `pick_locked=True`: write a pick record **only when
   the pick is genuinely committed** — `pick_was_delivered(daily) == true` → `action=single|double`,
   `source="unknown"`, `delivery_status="delivered"`, `scoreable=true`. A **non-delivered**
   classification-lock (a stale/preview `<date>.json` locked by `game_started_or_final` /
   `status_lookup_failed`) → **write nothing**, do NOT set `committed_pick_written` (so it cannot
   suppress a genuine skip and is never scored — this is the Codex r2 HIGH fix).
3. **Crash-guard** — `_deliver_and_lock_pick` `delivery_attempted` lock:
   `delivery_status="locked_unconfirmed"`, `source` from metadata else `"unknown"`, `scoreable=true`
   (the pick was the day's committed action; now explicit/auditable).
4. **End-of-day skip** — once, immediately **before** the end-of-day health-checks block (after final
   fallback, missed-pick handling, DH rechecks, next-day lookahead, and result polling — `run_day`
   *idles*, it does not return; Codex r2 #6), **iff `not committed_pick_written` and
   `final_skip_candidate` is set**: `action=skip`, the candidate's `primary`/`streak`/
   `saver_available`, `source="mdp"`, `delivery_status="not_applicable"`, `scoreable=false`. Else →
   **no record**. (Early no-games / dry-run returns short-circuit before this and write nothing.)

Gating the end-of-day skip on `committed_pick_written` (a genuine scoreable commit) **not**
`scheduler_state.pick_locked` means a stale preview file that got classified-locked on a real skip
day does not suppress the skip record. A day is exactly one of {committed pick, finalized MDP skip,
no record} — no stale skip, no reliance on last-write-wins.

## Consumers

### `check-results` (the #144 fix)

Precedence (Codex #7 — **already-resolved idempotency precedes the scoreable gate**, so re-runs stay
idempotent and the existing already-resolved tests are preserved; scoreable-gate-first would flip them
all to "not scoring"). The context-stack shadow reconciliation + status write run on every
*scoring-eligible* exit (already-resolved, not-scoreable, and scored) — matching today's behavior; they
do **not** run on the `No pick found` exit (no production pick to pair a shadow with):

1. Load `daily` (`load_pick`) and `decision.json`.
2. No `daily` → `No pick found`, done (unchanged — no shadow).
3. **Already resolved** (`daily.result in {hit,miss,void}`) → reconcile shadow + write status +
   `Already resolved` echo, done. Idempotency first, so a re-run never re-evaluates the gate.
4. Compute **`scoreable`** = `decision.scoreable` if a `decision.json` record exists, **else** the
   fallback `pick_was_delivered(daily)` (covers pre-feature / manual / backlog **delivered** picks and
   a committed public/DM pick whose best-effort write failed). **`scheduler_state.pick_locked` is NOT
   used** — it is True for `game_started_or_final` / status-failure classification locks of a stale
   preview file and would reintroduce #144 (Codex r2 HIGH). A preview / pre-lock / skip file is not
   delivered → not scoreable. (A *private* committed pick relies on its `decision.json` record,
   written at its `_deliver_and_lock_pick` commit; a missing record for a private pick — only on a
   rare write failure — would not score, an accepted edge since prod delivery is public/DM.)
5. **Not scoreable** (a `skip` record, or missing + uncommitted) → reconcile shadow + write status,
   then do NOT resolve slots / NOT `update_streak` / NOT save a result. Done. **(This is the #144 fix.)**
6. Score slots + `update_streak` + save result + reconcile shadow + write status + report
   (double-down / void handled by the existing slot resolver, unchanged).

### Skip-policy shadow — migrated onto `decision.json`

Folded into this change (Codex: two authorities invites another mismatch). The shadow's marker +
`pick_was_delivered` + `_final_decision` + `skip_decision.json` / `record_mdp_skip_decision` layer
is **deleted**. The nightly updater instead:
- reads `decision.json`; a date is a divergence iff `action=="skip" && source=="mdp"`, with
  `primary` as the declined candidate and `streak`/`saver_available` for the record/status rows
  (which retain `streak` — Codex #6, why those fields are in the schema).
- keeps a light **supersession re-check**: on each run, drop any existing `*.policy_shadow.json`
  whose current `decision.json` is no longer `skip&&mdp` (cheap insurance; the nightly run reads the
  day's final record, but this guards a same-day/early run).
- reconcile / Wilson status / verdict / CLI / dashboard panel are unchanged.

Net: the shadow gets **simpler** (no provisional-artifact resolution) and shares one authority with
`check-results`.

## Error handling

- All `decision.json` writes are best-effort (`try/except`, atomic) — they must never affect the
  live pick path.
- A committed **public/DM** pick whose best-effort write **fails** is still scored via the
  `check-results` `pick_was_delivered` fallback. A committed **private** pick relies on its
  `decision.json` record; a write failure there (rare) would not score — an accepted edge, since
  prod delivery is public/DM and the write is atomic/best-effort-but-usually-succeeds.
- `check-results` fallback as above; a malformed/partial `decision.json` is treated as missing.

## Testing

- **Writer** (scheduler, mocked): committed single/double → `decision.json` action/source/slots/
  scoreable; crash-guard → `locked_unconfirmed`+scoreable; end-of-day MDP skip → skip+candidate;
  end-of-day heuristic skip or no-eligible → **no** record; pick commit after an earlier skip cycle
  → final record is the pick (no stale skip).
- **`select_pick` SelectionResult**: action/source/candidates/no_pick_reason for skip/single/double/
  no-eligible; no files written.
- **`check-results`**: scoreable pick scored; skip not scored + streak untouched + stale file not
  resolved; missing+delivered → fallback scores; missing+undelivered → not scored.
- **Classification / lock-reason** (the Codex r2 HIGH): a genuine **delivered** pick recovered via
  classification-lock → scoreable record + scored; a **preview/pre-lock** `<date>.json` locked by
  `game_started_or_final` / `status_lookup_failed` → **no record**, not scored, and does **not**
  suppress that day's end-of-day MDP-skip record; a committed **private**/crash-guard pick → scoreable
  via its record.
- **Shadow**: skip&&mdp recorded with candidate; pick/heuristic/skip-without-mdp not; supersession
  drops a record whose decision flipped to a pick; reconcile/status unchanged.
- Full-suite regression incl. `TestBtsCheckResults` and `TestSelectPick` (return-shape change).

## Migration / cleanup (delete from d73face)

`skip_policy_shadow.py`: `record_mdp_skip_decision`, `load_skip_decision`, `skip_decision_path`,
`_final_decision`, `pick_was_delivered` usage, `record_skip_from_marker`'s marker read.
`strategy.select_pick`: the marker write + `persist_skip_decision` param (replaced by the metadata
return + scheduler write). Keep: reconcile/status/verdict/CLI/web/cron — but **update CLI / status /
docstring / ARCHITECTURE / CLAUDE wording away from "marker" / `skip_decision.json` / `pick_was_delivered`
terminology** to "reads `decision.json`" (Codex r2 #7); the dashboard panel + audit doc too.

## Implementation staging (Codex)

Stage and land in this order (the shadow migration depends on the schema being settled):
1. **`decision.json` writer + `SelectionResult` threading** (strategy + scheduler + orchestrator)
   — including `final_skip_candidate` and all four write points.
2. **`check-results` gate** (the #144 fix proper) + its fallback.
3. **Shadow migration** onto `decision.json` (delete the marker/`pick_was_delivered`/`_final_decision`
   layer).

## Risks / open

- **Threading the `SelectionResult`** + `final_skip_candidate` through `run_single_check`/`run_day`
  without disturbing the heartbeat/lock/fallback control flow — the highest-care area; cover with the
  writer tests above.
- **End-of-day hook placement** must be exact (Codex #5): once, after final-fallback + missed-pick
  handling, before health/idle (`run_day` *idles*, it does not return) — not a broad `finally`; and
  guarded by `final_skip_candidate` so error/abort days write nothing.
- **`source` on recovery**: `DailyPick` has no `source` field, so lock-by-classification / crash-guard
  use `source="unknown"` (the shadow ignores non-`mdp` skips, so this is harmless). If we later want
  recovered picks attributed, persist `source` on `DailyPick` — out of scope here.
- Source threading out of `decide_action` (small signature/return change).
