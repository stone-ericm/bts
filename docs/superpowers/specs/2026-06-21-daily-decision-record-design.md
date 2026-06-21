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
  "source": "mdp" | "heuristic",
  "primary": {"batter_id":..., "batter_name":..., "team":..., "game_pk":..., "p_game_hit":...} | null,
  "double_down": {... same shape ...} | null,
  "delivery_status": "delivered" | "locked_unconfirmed" | "not_applicable",
  "scoreable": true | false,
  "finalized_at": "2026-06-20T23:50:00Z"
}
```

- `action` — the final action. `primary` is the executable declined candidate on `skip`, the
  chosen primary on a pick. `double_down` is populated only for `double` (Codex: a singular
  candidate is not authoritative for doubles).
- `source` — `mdp` when the decision came from the MDP policy, `heuristic` otherwise. **From the
  resolved decision, not inferred after the fact** (Codex finding #5).
- `delivery_status` / `scoreable` — make the crash-guard case explicit (Codex #4):
  `delivered` (posted/DM'd), `locked_unconfirmed` (delivery_attempted crash-guard lock),
  `not_applicable` (skip). `scoreable` is the gate `check-results` reads (a committed pick).
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

The scheduler captures the latest `SelectionResult` each cycle and writes `decision.json` exactly
at finalization (best-effort `try/except`, never affects the pick path):

1. **Pick committed** — every `_deliver_and_lock_pick` success branch (public/DM/private, incl. the
   two fallback call sites): `action=single|double` (from `daily.double_down`), `primary`+
   `double_down`, `source` (from the captured metadata), `delivery_status=delivered`,
   `scoreable=true`. The loop then breaks.
2. **Lock-by-classification** — `run_single_check` returns an existing pick as locked
   (`classify_pick_lock_state`) and `run_day` sets `pick_locked=True` directly (Codex #3): write/
   recover `decision.json` here too (`action=pick`, `delivery_status` from `pick_was_delivered`,
   `scoreable=true`).
3. **Crash-guard** — `_deliver_and_lock_pick` `delivery_attempted` lock (Codex #4):
   `delivery_status=locked_unconfirmed`, `scoreable=true` (matches current behavior — the pick was
   the day's committed action — but now explicit/auditable).
4. **End-of-day skip finalization** — at the end of `run_day`, **if not `pick_locked`**, write
   `action=skip` **only if the latest decision was a genuine MDP skip** (`metadata.action=="skip"
   and source=="mdp"`), using `metadata.primary_candidate`. Writing the skip here — after the
   fallback, not on intermediate skip cycles — is what avoids the **stale-skip** bug (Codex #1: an
   early skip cycle followed by a cached-pick fallback delivery). If the loop ends with a
   selected-but-uncommitted pick or a no-eligible/error reason → **no record** (correct absence).

Because a pick commit breaks the loop and the skip is written once at end-of-day, a day is exactly
one of {committed pick, finalized skip, no record} — no provisional interleaving, last-write-wins is
not relied upon.

## Consumers

### `check-results` (the #144 fix)

Load `decision.json` **before** resolving production results. Then:
- `scoreable == true` → score slots + `update_streak` as today.
- `action == "skip"`, OR record missing **and** `pick_was_delivered(load_pick(date)) == false` →
  **do not** resolve slots, **do not** `update_streak`, **do not** save a result onto the stale
  `<date>.json`.
- **Fallback** (Codex): record missing **and** `pick_was_delivered(load_pick(date)) == true` →
  legacy scoreable pick (covers pre-feature / manual / backlog delivered picks). Never fall back to
  scoring an arbitrary unresolved `<date>.json`.

The existing context-stack shadow reconciliation / status paths still run regardless (they key off
`*.shadow.json`).

### Skip-policy shadow — migrated onto `decision.json`

Folded into this change (Codex: two authorities invites another mismatch). The shadow's marker +
`pick_was_delivered` + `_final_decision` + `skip_decision.json` / `record_mdp_skip_decision` layer
is **deleted**. The nightly updater instead:
- reads `decision.json`; a date is a divergence iff `action=="skip" && source=="mdp"`, with
  `primary` as the declined candidate.
- keeps a light **supersession re-check**: on each run, drop any existing `*.policy_shadow.json`
  whose current `decision.json` is no longer `skip&&mdp` (cheap insurance even though the scheduler
  writes only final decisions).
- reconcile / Wilson status / verdict / CLI / dashboard panel are unchanged.

Net: the shadow gets **simpler** (no provisional-artifact resolution) and shares one authority with
`check-results`.

## Error handling

- All `decision.json` writes are best-effort (`try/except`, atomic) — they must never affect the
  live pick path.
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
- **Shadow**: skip&&mdp recorded with candidate; pick/heuristic/skip-without-mdp not; supersession
  drops a record whose decision flipped to a pick; reconcile/status unchanged.
- Full-suite regression incl. `TestBtsCheckResults` and `TestSelectPick` (return-shape change).

## Migration / cleanup (delete from d73face)

`skip_policy_shadow.py`: `record_mdp_skip_decision`, `load_skip_decision`, `skip_decision_path`,
`_final_decision`, `pick_was_delivered` usage, `record_skip_from_marker`'s marker read.
`strategy.select_pick`: the marker write + `persist_skip_decision` param (replaced by the metadata
return + scheduler write). Keep: reconcile/status/verdict/CLI/web/cron.

## Risks / open

- **Threading the `SelectionResult`** through `run_single_check`/`run_day` without disturbing the
  heartbeat/lock/fallback control flow — the highest-care area; cover with the writer tests above.
- `run_day`'s end-of-day hook must fire on every loop-exit-without-pick path (normal end, contest-
  state abort, exceptions) — audit each `return`/`break`.
- Source threading out of `decide_action` (small signature/return change).
