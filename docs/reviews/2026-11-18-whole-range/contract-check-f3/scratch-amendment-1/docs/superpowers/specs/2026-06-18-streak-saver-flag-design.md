# Streak Saver Manual Flag — Design

**Date:** 2026-06-18
**Status:** Design
**Authors:** Claude (Opus 4.8); design reviewed by Codex (gpt-5.5)
**Replaces:** the Phase-2c `infer_saver` / `best_streak` / ledger-coverage inference. That
approach is **provably unsound**: the consuming "save" is a streak-PRESERVING transition
(e.g. `14 → 14`), so a save round can vanish from the windowed MLB `predictions` snapshot
without leaving any gap in the streak chain — you cannot tell "available" from "used" from
the snapshot. Three Codex review rounds converged on this conclusion.

## 1. Problem

The MLB Beat the Streak **Streak Saver** is a one-time, season-scoped mulligan. The decision
loop needs to know whether it is available, because the MDP policy at streak 10–15 depends on
it. But the MLB profile API does not expose saver state, and it cannot be soundly inferred
from the ledger (above). The saver changes state **at most twice per season**, so a **manual
flag** is both sound and cheap.

### The rule (deterministic, authoritative — MLB official text)
- The Streak Saver is **automatically applied the first time your streak reaches 10**.
- It saves the **first** No-Hit that occurs while your streak is **10–15 inclusive**: the streak
  is **held** at its then-current value (not reset to 0), and the saver is consumed.
- It saves **only once** per season; once consumed, it is gone for the season.
- A No-Hit while the streak is `< 10` or `> 15` resets to 0 and does **NOT** consume the saver.

### Verified influence on suggestions
The saver changes the suggested action **only when the streak is 10–15** (zero effect at any
other streak, confirmed across the policy table). Inside 10–15 it flips ≈⅓ of decisions
(single↔double, skip↔single/double) at realistic pick qualities (p ≈ 0.80–0.84+). So
correctness matters only during the days spent in the zone.

## 2. Design

### 2.1 State + persistence
The persisted file stores `state ∈ {not_earned, active, used}` at
`data/picks/account_state/saver_state.json`:
```json
{"season": 2026, "state": "active", "updated_at": "2026-06-18T15:00:00Z", "source": "manual_init"}
```
`season` is the **contest year**, derived from the live contest observation's `source_date.year`
(fallback: current ET year). `source` records who set it (`manual_init`, `auto_earn`,
`dashboard`, `cli`, `season_reset`).

**`load_saver_state` returns a fourth, loader-derived `uninitialized`** (never persisted) when
the file is missing, unparseable/invalid, or its `season` ≠ the current contest season.
`uninitialized` is **distinct from `not_earned`** — so a missing file at `best_streak ≥ 10` can
never be mistaken for "earned, unused" — and maps to **saver-unavailable** (fail-closed).

### 2.2 Authority (Codex #3)
`saver_state.json` is the **sole authority** for live saver availability:
`saver_available = (state == "active")`. The contest `saver_available` field is **retired from
the saver-decision path** — the auto-fetch writes it `None`, and the `set-contest-streak
--saver-available/--saver-unavailable` CLI option is **deprecated** (it must not be able to
silently override a `used` flag back to available). Saver changes go through the new flag only.

### 2.3 Transitions
| From | To | Trigger | Soundness |
|---|---|---|---|
| `not_earned` | `active` | **AUTO** — `maybe_auto_earn_saver` observes `best_streak ≥ 10`, but **only** when the file is already initialized `not_earned` for the current season | sound: `best_streak` is a reliable counter, and a tracked `not_earned` certifies no prior save |
| `active` | `used` | **MANUAL** — dashboard button (POST) or CLI | the unsound-to-auto-detect transition; the operator knows when their streak was saved |
| `used` | `active` | **MANUAL undo** — dashboard "undo" (visible while `used`) or CLI | mis-click recovery (Codex #2) |

There is **no automatic `→ used`** transition **for `saver_state.json`** (the live-decision
saver). A ledger-derived "likely save" only *nudges* (§2.6), never writes. NOTE: the local
model-replay saver in `streak.json` (`picks.update_streak` / `_apply_streak_day`) auto-consumes
on a replayed 10–15 miss — that is a **separate** mechanism (the model's replay of its own
suggestions, kept only for the `model_saver_available` diagnostic) and is **not** the live saver
and **not** wired into this flag.

### 2.4 Bootstrap / initialization (soundness-critical — Codex #1)
Auto-earn is sound only if we have tracked the account since *before* it first reached 10. A
missing/uninitialized (or stale-season) file is handled **fail-closed**:
- `best_streak < 10` → safe to initialize `not_earned` automatically (no save possible yet).
- `best_streak ≥ 10` → **cannot** auto-initialize `active` (the account may have earned **and
  used** the saver before the bot ever observed it). Leave **uninitialized**; live decisions
  treat an uninitialized/`>=10`-unverified file as **saver-unavailable** (fail-closed) until a
  **manual init** (CLI/dashboard) sets the true state.

**Eric's current bootstrap:** manual init to `active` — verified correct (the bot observed
`best_streak = 9` yesterday → a certificate of "no save yet" → today's climb to 10 is an
observed hit, not a save).

### 2.5 Season reset (Codex #4)
The schema carries `season`. A season mismatch makes `load_saver_state` return `uninitialized`
(§2.1) — the stale flag is never trusted. `maybe_auto_earn_saver` then re-initializes for the new
season: `best_streak < 10` → write `not_earned` (safe, no save possible yet); `best_streak ≥ 10`
→ leave `uninitialized` (fail-closed), require a manual init. Health validation flags a season
mismatch (see §4).

### 2.6 Dashboard (Codex #2, #7, #9)
- Always shows the current saver state.
- A **POST** route (`/saver/transition`) with an **atomic** write performs guarded transitions:
  - `state == active` → **"Mark Streak Saver used"** (`active → used`). **Foregrounded** when a
    likely-save is detected in the ledger.
  - `state == used` → **"Undo — mark active"** (`used → active`).
- **Likely-save nudge:** when the ledger shows a *stable* No-Hit at pre-streak 10–15 that held
  (reuse the existing stable / non-DD exclusions; ledger is "likely" evidence **only**, never a
  transition), foreground the "mark used?" prompt.
- **Persistent verification warning** whenever `state == active` AND `best_streak > 15` (or the
  streak returns to 10–15 after a prior `> 15`): "verify your saver wasn't already used" — so the
  offline-save-past-15 residual (§3) can't lie dormant until the next zone.

### 2.7 Concurrency / atomicity (Codex #5, #6, #7)
- All writes are **atomic** (temp + `os.replace` / `atomic_write_text`).
- Transitions are **guarded + monotonic**: `transition_saver_state(expected_prior, new_state)`
  re-reads and only writes if the current state equals `expected_prior`. Auto-earn may only
  write `active` after re-reading `not_earned`; it must **never** overwrite `used` or `active`.
- The dashboard is `ThreadingHTTPServer` (concurrent requests) — the POST handler uses the same
  guarded atomic write.
- **POST mutation safety:** the form body carries the `expected_prior` state (a mismatch — stale
  page, double-click — is rejected with a clear error), and the handler enforces a same-origin /
  CSRF guard. The endpoint mutates live decision state, so this holds even though the dashboard
  binds tailnet/loopback only (`web.py:1594`).

### 2.8 Helpers + placement (Codex #5)
- `load_saver_state(picks_dir) -> SaverState` — **read-only** (the decision loader calls only this).
- `transition_saver_state(picks_dir, expected_prior, new_state, source) -> bool` — guarded atomic write.
- `maybe_auto_earn_saver(picks_dir, best_streak, season) -> None` — the `not_earned → active` +
  safe new-season `not_earned` init. **Called from the fetch write path** (`fetch-contest-streak`)
  and from `set-contest-streak`, **NOT** from `load_decision_streak_state` (loaders never mutate).
- **CLI** `bts saver-state` (the manual surface used by migration, fail-closed bootstrap, and
  break-glass): `--show`; `--init {not_earned|active|used}`; `--use` (`active → used`); `--undo`
  (`used → active`). Each routes through `transition_saver_state` with the correct `expected_prior`
  (so `--use` no-ops unless currently `active`, etc.). `--init` is **guarded**: it only writes when
  the state is `uninitialized`; overwriting an existing `not_earned`/`active`/`used` requires
  explicit `--force`, so a break-glass init can never silently clobber a real `used`. This replaces
  the deprecated `set-contest-streak --saver-available`.

### 2.9 Decision wiring
`load_decision_streak_state` reads the live saver **only** from `saver_state.json`, in **both**
branches:
```python
saver = (load_saver_state(picks_dir).state == "active")
```
- **Contest-present path** (`contest_state.py:286`): replaces the `infer_saver`/`best_streak`
  block; no `contest.saver_available` branch (retired per §2.2).
- **Model-only fallback** (no contest observation, `contest_state.py:253`): also uses
  `saver_state.json`, **not** `streak.json`'s `model_saver` — so the live saver has one authority
  whether or not a contest observation exists. `model_saver_available` remains a diagnostic field
  on the returned `DecisionStreakState`.

## 3. Failure modes
- **Forgot to mark used** → stays `active` → over-aggressive at 10–15. Mitigated by the §2.6
  nudge + the `best_streak > 15` warning.
- **Mis-clicked used** → `used → active` undo (§2.3).
- **Bot offline when crossing 10** → auto-earn catches up on the next fetch (from a tracked
  `not_earned`).
- **Residual:** a save that happened while the bot was offline AND the streak then climbed past
  15 with the save windowed out → flag stays `active`, nudge can't fire. Mitigated (not
  eliminated) by the persistent `active` + `best_streak > 15` warning (§2.6). Accepted.

## 4. What's replaced / kept / retired
- **Replace:** `bts.contest_ledger.infer_saver`, the `best_streak`/coverage saver logic in
  `load_decision_streak_state`, and the saver-fallback tests (`test_decision_saver_fallback.py`).
- **Keep / repurpose:** `parse_latest_ledger` — now used **only** for the dashboard likely-save
  nudge, not for live decisions.
- **Keep:** `decide_action` (2a) + the atomic policy save — independent, ship on their own.
- **Retire:** `contest.saver_available` from the saver-decision path; deprecate the
  `set-contest-streak --saver-available` option.

## 5. Migration
1. Add `saver_state.json` + helpers + wiring + dashboard route + health check.
2. Initialize Eric's flag to **`active`** (`season=2026`, verified) via the new CLI/dashboard.
3. **Clear the temporary contest override** set 2026-06-18 (`contest_streak.manual.json` with
   `saver_available=true`) so the flag is the only saver authority.

## 6. Testing (Codex #10)
- **State machine:** each transition; guarded writes reject a wrong `expected_prior`; auto-earn
  never overwrites `used`/`active`; atomic write (no partial file).
- **Bootstrap:** missing file + `best_streak < 10` → `not_earned`; missing file + `best_streak ≥
  10` → fail-closed (saver unavailable until manual init).
- **Season:** stale-season file → reset/fail-closed by `best_streak`; health flags mismatch.
- **Decision wiring:** `state == active` → saver True; `not_earned`/`used`/uninitialized → False.
- **Migration of existing tests:** the "reached zone but unconfirmed → unavailable" assertions
  (`test_contest_state.py`, `test_decision_saver_fallback.py`) become manual-state tests; the
  ledger consumption tests (`test_contest_ledger.py`) are repurposed as nudge-evidence tests.
- **Uninitialized/stale:** missing / invalid / wrong-season file → `load_saver_state` returns
  `uninitialized` → saver unavailable (NOT `not_earned`).
- **Model-only path:** `load_decision_streak_state` with no contest observation reads the saver
  from `saver_state.json`, not `streak.json`.
- **Deprecated override inert:** `set-contest-streak --saver-available` no longer affects the live
  saver decision (and `test_cli_integration.py` saver assertions are migrated).
- **Dashboard POST guard:** a transition with a wrong `expected_prior` is rejected; same-origin is
  enforced.
- **`--init` guard:** `--init` on an existing `not_earned`/`active`/`used` errs without `--force`.

## 7. Out of scope
- Auto-detecting the `→ used` transition (provably unsound).
- The full persisted coverage/audit-trail inference (Codex's alternative) — heavier than a flag
  that changes twice a season is worth.
- Revisiting the MDP objective / the saver's effect on the policy (the policy is correct; this
  only feeds it the right saver bit).
