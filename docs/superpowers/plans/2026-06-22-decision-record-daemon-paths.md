# decision.json — daemon-path completion of the #144 fix

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Close the two daemon paths (and two robustness gaps) that let a stale projected-preview `<date>.json` be delivered or scored on a real-MDP-skip day — completing the GH #144 fix the `check-results` gate (Task 4) only partially closed.

**Architecture:** The `decision.json` record + `SelectionResult.action` now make a *genuine skip* distinguishable from a *cascade failure* and a *committed pick* from a *classification-locked stale file*. Use those signals to (C1) stop the fallback from delivering a cached pick on a genuine skip, (C2) stop result-polling from scoring a non-committed pick, (#3) survive a same-day restart, (#4) treat a malformed decision.json as missing.

**Tech Stack:** Python, pytest. `UV_CACHE_DIR=/tmp/uv-cache`. Branch `skip-policy-shadow` (continues the decision.json work; HEAD `fffb2cb`).

## Global Constraints

- Scheduler writes/reads stay **best-effort**: nothing here may raise into the live pick-delivery path.
- A **genuine MDP skip must not deliver, lock-for-polling, or score** any pick — including a leftover projected `<date>.json`.
- A **cascade failure / no-predictions** fallback must still deliver the cached pick (the existing safety net) — do NOT regress that.
- The "committed pick" test is the SAME one `check-results` uses: `decision.scoreable` if a `decision.json` record exists, else `picks.pick_was_delivered(daily)` — NEVER `scheduler_state.pick_locked`.
- Regression command (heavy model/backtest suites are orthogonal): `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q` (was 1456 green before this work).

---

## Task 1: shared "committed pick" helper (DRY the gate)

**Files:** Create `src/bts/daily_decision.py` helper `is_scoreable_commit`; Test `tests/test_daily_decision.py`.

**Interfaces:**
- Produces: `is_scoreable_commit(date, picks_dir, daily) -> bool` — `bool(load_decision(date,picks_dir).get("scoreable"))` if a (valid) record exists, else `picks.pick_was_delivered(daily)`. This is the single source of "should this pick advance the streak / be polled," shared by check-results (Task 5) + result-polling (Task 3).

- [ ] **Step 1: failing test**
```python
# tests/test_daily_decision.py
from bts.daily_decision import is_scoreable_commit, write_decision
def test_is_scoreable_commit(tmp_path):
    from bts.picks import DailyPick  # build a minimal delivered/undelivered daily via existing helpers
    # skip record -> not a commit
    write_decision("2026-06-20", tmp_path, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False)
    assert is_scoreable_commit("2026-06-20", tmp_path, _undelivered_daily()) is False
    # scoreable record -> commit (even if daily looks undelivered)
    write_decision("2026-06-21", tmp_path, action="single", source="mdp", delivery_status="delivered", scoreable=True)
    assert is_scoreable_commit("2026-06-21", tmp_path, _undelivered_daily()) is True
    # no record -> fall back to pick_was_delivered
    assert is_scoreable_commit("2026-06-22", tmp_path, _delivered_daily()) is True
    assert is_scoreable_commit("2026-06-22", tmp_path, _undelivered_daily()) is False
```
(Reuse the test file's existing daily builders; `_delivered_daily` = `bluesky_posted=True` / `notification` set, `_undelivered_daily` = neither.)
- [ ] **Step 2: verify RED** (`is_scoreable_commit` undefined).
- [ ] **Step 3: implement** in `daily_decision.py`:
```python
def is_scoreable_commit(date, picks_dir, daily) -> bool:
    from bts.picks import pick_was_delivered
    dec = load_decision(date, picks_dir)
    if dec is not None:
        return bool(dec.get("scoreable"))
    return bool(daily is not None and pick_was_delivered(daily))
```
- [ ] **Step 4: GREEN** + commit.

---

## Task 2: `load_decision` treats malformed/wrong-shape as missing (#4)

**Files:** Modify `src/bts/daily_decision.py` (`load_decision`); Test `tests/test_daily_decision.py`.

- [ ] **Step 1: failing tests** — a JSON list, a JSON string, and a dict with the wrong/absent `schema_version` all return `None`:
```python
def test_load_rejects_wrong_shape(tmp_path):
    p = decision_path("2026-06-20", tmp_path); p.parent.mkdir(parents=True, exist_ok=True)
    for bad in ("[]", "\"x\"", "{\"scoreable\": true}"):   # list, string, dict missing schema_version
        p.write_text(bad)
        assert load_decision("2026-06-20", tmp_path) is None
```
- [ ] **Step 2: verify RED** (current code returns `[]`/`"x"`/the partial dict).
- [ ] **Step 3: implement** — after `json.loads`, validate: `if not isinstance(rec, dict) or rec.get("schema_version") != DECISION_SCHEMA: return None`. Keep the existing `(JSONDecodeError, OSError)` catch.
- [ ] **Step 4: GREEN** + run `tests/test_daily_decision.py` + `tests/test_skip_policy_shadow.py` (shadow `load_decision` consumer) + commit.

---

## Task 3: result-polling + classification gate on a genuine commit (C2)

**Files:** Modify `src/bts/scheduler.py` (`run_day` result-polling gate ~2153; the classification-lock block ~1934 keeps setting `pick_locked` but polling is now gated separately); Test `tests/test_scheduler_decision_record_integration.py`.

**Interfaces:** Consumes `daily_decision.is_scoreable_commit` (Task 1).

- [ ] **Step 1: failing integration test** — drive `run_day` to a state where a **non-delivered** `<date>.json` is classification-locked (`state.pick_locked=True`) on an MDP-skip day, then assert result-polling is NOT entered / `update_streak` not called (mock `run_result_polling` and assert `assert_not_called`), and the streak is unchanged. Mirror the existing integration fixtures.
- [ ] **Step 2: verify RED** — current gate `if state.pick_locked:` enters polling.
- [ ] **Step 3: implement** — change the result-polling gate (~2153) from `if state.pick_locked:` to gate on a genuine commit:
```python
    daily_for_poll = load_pick(date, picks_dir)
    if state.pick_locked and daily_for_poll is not None and is_scoreable_commit(date, picks_dir, daily_for_poll):
        # ... existing polling block ...
```
Leave `state.pick_locked` itself unchanged (it still gates the shadow-model trigger, next-day logic, etc. — do NOT repurpose it). Add the `is_scoreable_commit` import. A non-committed classification-lock now locks (for those other purposes) but is never polled/scored.
- [ ] **Step 4: GREEN** + the scheduler regression (`tests/test_scheduler*.py tests/test_daily_decision.py` per Task 6 of the original plan) + commit.

---

## Task 4: fallback honors a genuine skip (C1)

**Files:** Modify `src/bts/scheduler.py` (`_refresh_pick_at_fallback_decision` ~1396 to carry `selection` on the no-fresh-pick return; the primary fallback ~2030 and final fallback ~2096 to NOT deliver on a genuine skip); Test `tests/test_scheduler_decision_record_integration.py` (or `tests/test_scheduler.py`).

**The distinction (load-bearing):** `_refresh_pick_at_fallback_decision` returns no fresh pick in THREE cases — (a) genuine MDP skip (`sel.action=="skip"`), (b) no-predictions (`sel is None`), (c) cascade exception (`sel` undefined). Only (a) must **suppress delivery** (honor the skip). (b)/(c) keep the safety-net "deliver cached."

- [x] **Step 1: failing tests**
  - primary-fallback skip: `_refresh_pick_at_fallback_decision` returns a skip → caller does NOT call `_deliver_and_lock_pick`, the cached `<date>.json` stays undelivered, and `final_skip_candidate` is preserved (EOD skip still recordable).
  - regression guard: a fallback on a cascade **error** (sel None) STILL delivers the cached pick (don't break the safety net).
- [x] **Step 2: verify RED** — current caller delivers (skip's `should_post=None` slips past `should_post is False`).
- [x] **Step 3: implement**
  - In `_refresh_pick_at_fallback_decision`, the `pick_result is None` return (~1396) becomes `return FallbackRefreshResult(cached_daily, None, selection=sel)` so the caller can see the action. (The exception path ~1391 stays `selection=None` → treated as safety-net deliver.)
  - Add a helper `_refresh_is_genuine_skip(refresh) -> bool`: `refresh.selection is not None and refresh.selection.action == "skip" and refresh.selection.source == "mdp"`.
  - **CAPTURE the skip candidate (Codex r2 High #1).** Honoring the skip is NOT enough: on the projected→real
    flip the earlier projected-PICK cycle already CLEARED `final_skip_candidate` (Task 3 lifecycle), and the
    skip may be detected ONLY in the fallback (never re-set via `run_single_check`'s 1906 path). So when the
    fallback honors a genuine skip, it must SET the candidate from `refresh.selection` so EOD records it:
    ```python
    def _capture_fallback_skip(state, refresh):
        sel = refresh.selection
        state.final_skip_candidate = {"primary": sel.primary_candidate,
                                      "streak": sel.streak, "saver_available": sel.saver_available}
    ```
    (operates on `state` per Task 5, which moves the field onto SchedulerState).
  - **Gate on the STANDING skip decision, not just this refresh (Codex r3 High).** A confirmed skip must survive a
    SUBSEQUENT transient refresh — e.g. the in-loop fallback confirms a skip + `continue`s, then on a single-run day
    the post-loop fallback re-runs refresh; if THAT refresh flakes to `selection=None` (cascade / no-predictions), it
    must NOT resurrect the cached pick. At each fallback delivery site compute:
    `skip_standing = _refresh_is_genuine_skip(refresh) or bool(state.final_skip_candidate and not state.committed_pick_written)`.
    When `skip_standing` and `_refresh_is_genuine_skip(refresh)`, `_capture_fallback_skip(state, refresh)` (freshest
    candidate). The cascade-error safety-net deliver-cached is preserved ONLY when `not skip_standing` (a genuine
    PICK day whose refresh errored has `final_skip_candidate is None`).
  - **Primary fallback (in-loop, ~2046):** after `daily = refresh.daily`, BEFORE the `should_post is False` block:
    `if skip_standing: <capture if _refresh_is_genuine_skip>; save_state(state, picks_dir); print("  FALLBACK: standing MDP skip — not delivering cached pick."); continue` (the `continue` lands in the `for run_info` loop → loop exhausts → EOD).
  - **Final fallback (post-loop/sequential, ~2096):** NOT inside a loop — do NOT `continue`. Structure as
    `if skip_standing: <capture if genuine>; save_state; <log>  else: _deliver_and_lock_pick(...)`, so the no-deliver
    path falls through to the subsequent EOD steps (must NOT re-enter delivery).
  - Result: on a skip day BOTH fallbacks honor the skip even if one refresh flakes; the safety-net still fires on a
    genuine PICK day whose fallback refresh errors.
- [x] **Step 4: GREEN** + scheduler regression + commit. (Add an integration test: projected-pick cycle clears the candidate, then fallback-skip RE-captures it, and EOD writes the skip record.) — commit `62cfe4e`; 5 tests added; scoped 114 passed, broad 1470 passed.

---

## Task 5: persist `final_skip_candidate` across a same-day restart (#3)

**Files:** Modify `src/bts/scheduler.py` (`SchedulerState` fields + `carry_forward_skip_state` ~930 + `run_day` `fs` init ~1863); Test `tests/test_scheduler_skip_visibility.py` (mirrors the existing carry-forward tests).

- [ ] **Step 1: failing test** — mirror `test_carry_forward_skip_state_*`: after an MDP-skip cycle sets `final_skip_candidate`, a rebuilt `SchedulerState` (same date) carries it forward so the EOD skip record is still written; a different-date previous state does NOT carry it.
- [ ] **Step 2: verify RED**.
- [ ] **Step 3: implement** — add `final_skip_candidate: dict | None = None` and `committed_pick_written: bool = False` to `SchedulerState` (defaulted → old `scheduler_state.json` still loads; `asdict` round-trips dicts cleanly — Codex confirmed). **Drop the separate `FinalizationState` and operate on `state` directly** (one finalization object — removes the dual-object sync hazard). `carry_forward_skip_state` copies the two fields when `previous_state.date == state.date`.
  - **Save-ordering (Codex r2 High #2a):** existing commit/classification paths `save_state` BEFORE recording the decision (e.g. ~548 then `_record_commit` ~553; classification ~1935). Since the flags now live on `state`, set the flag and `save_state` AFTER the decision write at each site, so a restart in between can't lose `committed_pick_written`.
  - **Overwrite-guard on `_write_endofday_skip` (Codex r2 High #2b) — the authoritative "committed today?" is the on-disk `decision.json`, not just the in-memory flag.** Before writing the EOD skip, no-op if a scoreable commit record already exists:
    ```python
    def _write_endofday_skip(picks_dir, date, state):
        if state.committed_pick_written or not state.final_skip_candidate:
            return
        existing = load_decision(date, picks_dir)        # crash between commit-write and state-save
        if existing is not None and existing.get("scoreable"):  # → a real pick already recorded; never clobber
            return
        ...
    ```
    This makes the EOD skip both restart-safe and non-clobbering regardless of flag-persistence timing.
- [ ] **Step 4: GREEN** + `tests/test_scheduler_skip_visibility.py` + scheduler regression + commit.

---

## Task 6: alert/health checks learn the decision gate (Codex r2 Medium #3)

C1 correctly NOT delivering on a genuine skip leaves an undelivered `<date>.json`, which the alert/health
layer currently reads as a failure. Gate the three checks so a deliberate skip is not a "missed/failed pick."

**Files:** Modify `src/bts/scheduler.py` (`_maybe_alert_missed_pick` ~80/2105), `src/bts/health/post_failure.py` (~39), `src/bts/health/analytics_artifacts_missing.py` (~50); Tests `tests/health/test_post_failure*.py`, `tests/health/test_analytics_artifacts_missing.py`, `tests/test_scheduler*.py`.

- [ ] **Step 1: failing tests** — (a) on a skip day (`state.final_skip_candidate` set, no committed pick), `_maybe_alert_missed_pick` does NOT fire the `missed_pick` CRITICAL; (b) `post_failure` does NOT alert when `decision.json` for the date is a skip / not scoreable; (c) `analytics_artifacts_missing` does not flag missing locked-pick artifacts for a non-committed classification-lock.
- [ ] **Step 2: verify RED** (each currently fires on the undelivered/locked stale file).
- [ ] **Step 3: implement**
  - `_maybe_alert_missed_pick` (pre-game; the EOD skip record may not exist yet): skip the alert when a skip was decided today — `if state.final_skip_candidate or state.skip_summary: return` (a genuine skip is not a missed pick). A TRUE missed pick (intended a pick, none delivered, no skip decided) still alerts.
  - `post_failure` (EOD; `decision.json` exists by now): no-op when `load_decision(date)` exists and is `action=="skip"` or `scoreable` is false — `is_scoreable_commit(date, picks_dir, daily)` is the clean predicate (don't alert a non-committed day as a failed post).
  - `analytics_artifacts_missing`: only expect locked-pick artifacts when `is_scoreable_commit(...)` (a genuine commit), not merely when `pick_locked`.
- [ ] **Step 4: GREEN** + `tests/health/ -q` + scheduler regression + commit.

---

## Task 7: check-results uses the shared helper + full regression

**Files:** Modify `src/bts/cli.py` (`check_results` reuses `is_scoreable_commit`); full regression.

- [ ] **Step 1:** replace the inline `decision.scoreable`-else-`pick_was_delivered` block in `check_results` (added in the original Task 4) with a call to `daily_decision.is_scoreable_commit(date, picks_path, daily)` (DRY — one definition of "committed"). Keep the already-resolved-before-gate ordering.
- [ ] **Step 2:** run `tests/test_cli_integration.py::TestBtsCheckResults` (must stay green).
- [ ] **Step 3: full scoped regression** (Global Constraints command) — expect green (~1456 + the new tests).
- [ ] **Step 4: commit.**

---

## Self-review
- C1 + C2 + the alert/health gates all key off the SAME `is_scoreable_commit` / genuine-skip signals (no divergent definitions).
- Coverage is complete (Codex-confirmed): 3 delivery sites (1958 fresh-only / 2046 / 2096) + 3 scoring sites (1704 / 1729 in polling / cli 1948) — all gated or safe.
- Safety-net preserved: cascade-error / no-predictions fallback still delivers cached (Task 4 regression guard); only a *genuine MDP skip* suppresses delivery.
- Fallback-skip is now CAPTURED (Task 4) so EOD records it even when the earlier projected cycle cleared the candidate and the skip surfaces only in the fallback.
- EOD skip is non-clobbering + restart-safe via the on-disk `decision.json` overwrite-guard (Task 5), so flag-persistence timing can't corrupt a real pick into a skip.
- `state.pick_locked` semantics unchanged (only polling is newly gated) — no ripple to shadow-trigger/next-day logic.
- Alert/health layer no longer false-fires on deliberate skips (Task 6).
- Best-effort intact: `is_scoreable_commit` / `load_decision` never raise (Task 2 hardens load).
