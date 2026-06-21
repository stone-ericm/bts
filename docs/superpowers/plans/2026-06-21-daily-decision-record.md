# Daily Decision Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give production one authoritative `data/picks/<date>/decision.json` — written only by the scheduler at true finalization — so `check-results` stops scoring undelivered preview/stale picks on skip days (GH #144) and the skip-policy shadow reads one signal instead of reverse-engineering provisional artifacts.

**Architecture:** `select_pick` returns decision metadata (no file writes). The scheduler is the single writer of `decision.json` at finalization points (pick commit, lock-by-classification, crash-guard, end-of-day skip), tracking `committed_pick_written` + `final_skip_candidate`. `check-results` and the shadow both read `decision.json`.

**Tech Stack:** Python 3.12, `uv`, pytest, `bts.util.atomic_write_text`. Spec: `docs/superpowers/specs/2026-06-21-daily-decision-record-design.md`.

## Global Constraints

- All commands prefixed `UV_CACHE_DIR=/tmp/uv-cache`. Run tests with `uv run pytest`.
- `decision.json` schema_version = `bts_daily_decision_v1`. Path: `data/picks/<date>/decision.json`.
- All `decision.json` writes are **best-effort** (`try/except`, atomic via `atomic_write_text`) — they must NEVER raise into or otherwise affect the live pick path.
- `source ∈ {"mdp","heuristic","unknown"}`; `action ∈ {"skip","single","double"}`; `delivery_status ∈ {"delivered","private_locked","locked_unconfirmed","not_applicable"}`.
- `scoreable=true` for every committed-pick variant; `false` for skip.
- A **non-delivered** classification-lock (`game_started_or_final`/`status_lookup_failed` on a preview file) writes **no record**.
- TDD: failing test → verify red → minimal impl → verify green → commit. Commit message trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Branch: `skip-policy-shadow` (continues the shadow work it migrates).

## File structure

- **Create** `src/bts/daily_decision.py` — schema + `write_decision`/`load_decision`/`decision_path` (Task 1).
- **Modify** `src/bts/strategy.py` — `decide_action` returns `(action, source)`; `select_pick` returns `SelectionResult`; drop the marker write + `persist_skip_decision` (Tasks 2, 5).
- **Modify** `src/bts/orchestrator.py` — `run_and_pick` returns + threads the `SelectionResult` (Task 2).
- **Modify** `src/bts/scheduler.py` — capture metadata; write `decision.json` at the 4 finalization points; `committed_pick_written`/`final_skip_candidate` (Task 3).
- **Modify** `src/bts/cli.py` — `check-results` reads `decision.json` + gate + fallback; update `select_pick` callers (Tasks 2, 4).
- **Modify** `src/bts/skip_policy_shadow.py` — read `decision.json`; delete marker layer (Task 5).
- **Modify** `ARCHITECTURE.md`, `CLAUDE.md`, `docs/audit/2026-06-20-skip-policy-shadow.md`, dashboard wording (Task 6).
- **Tests**: `tests/test_daily_decision.py` (new); modify `tests/test_strategy.py`, `tests/test_decide_action.py`, `tests/test_skip_policy_shadow.py`, `tests/test_cli_integration.py`, `tests/test_scheduler*.py`.

---

## Task 1: `decision.json` reader/writer

**Files:**
- Create: `src/bts/daily_decision.py`
- Test: `tests/test_daily_decision.py`

**Interfaces:**
- Produces:
  - `DECISION_SCHEMA = "bts_daily_decision_v1"`
  - `decision_path(date: str, picks_dir) -> Path`
  - `write_decision(date, picks_dir, *, action, source, primary=None, double_down=None, streak=None, saver_available=None, delivery_status, scoreable, now=None) -> dict | None` — best-effort; returns the record (or None on failure).
  - `load_decision(date, picks_dir) -> dict | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_daily_decision.py
import json
from pathlib import Path
from datetime import datetime, timezone
from bts.daily_decision import write_decision, load_decision, decision_path, DECISION_SCHEMA

def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}

def test_write_and_load_roundtrip(tmp_path):
    rec = write_decision("2026-06-20", tmp_path, action="skip", source="mdp",
                         primary=_cand(), streak=10, saver_available=True,
                         delivery_status="not_applicable", scoreable=False,
                         now=datetime(2026, 6, 20, tzinfo=timezone.utc))
    assert rec is not None
    assert decision_path("2026-06-20", tmp_path).exists()
    loaded = load_decision("2026-06-20", tmp_path)
    assert loaded["schema_version"] == DECISION_SCHEMA
    assert loaded["action"] == "skip" and loaded["source"] == "mdp"
    assert loaded["primary"]["batter_id"] == 1 and loaded["primary"]["p_game_hit"] == 0.78
    assert loaded["streak"] == 10 and loaded["saver_available"] is True
    assert loaded["scoreable"] is False

def test_load_missing_is_none(tmp_path):
    assert load_decision("2026-01-01", tmp_path) is None

def test_write_is_best_effort_never_raises():
    # an unwritable picks_dir must not raise (best-effort)
    assert write_decision("2026-06-20", "/proc/cannot/write/here", action="skip", source="mdp",
                          delivery_status="not_applicable", scoreable=False) is None

def test_double_carries_both_slots(tmp_path):
    write_decision("2026-06-20", tmp_path, action="double", source="mdp",
                   primary=_cand(1), double_down=_cand(2), delivery_status="delivered", scoreable=True)
    loaded = load_decision("2026-06-20", tmp_path)
    assert loaded["action"] == "double"
    assert loaded["double_down"]["batter_id"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_daily_decision.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.daily_decision'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bts/daily_decision.py
"""Authoritative end-of-day decision record (data/picks/<date>/decision.json).

The SINGLE source of truth for "what did production finally do on <date>". Written only by the
scheduler at true finalization points; read by check-results and the skip-policy shadow. See
docs/superpowers/specs/2026-06-21-daily-decision-record-design.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bts.util import atomic_write_text

DECISION_SCHEMA = "bts_daily_decision_v1"
_RANK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "p_game_hit")


def _utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _summary(cand: dict | None) -> dict | None:
    return None if cand is None else {k: cand.get(k) for k in _RANK_FIELDS}


def decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / date / "decision.json"


def write_decision(date, picks_dir, *, action, source, primary=None, double_down=None,
                   streak=None, saver_available=None, delivery_status, scoreable, now=None) -> dict | None:
    """Best-effort atomic write of the day's decision record. Returns the record, or None on any
    failure (must never raise into the live pick path)."""
    record = {
        "schema_version": DECISION_SCHEMA, "date": date,
        "action": action, "source": source,
        "primary": _summary(primary), "double_down": _summary(double_down),
        "streak": streak,
        "saver_available": (None if saver_available is None else bool(saver_available)),
        "delivery_status": delivery_status, "scoreable": bool(scoreable),
        "finalized_at": _utc_iso(now),
    }
    try:
        path = decision_path(date, picks_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(record, indent=2))
        return record
    except Exception:
        return None


def load_decision(date: str, picks_dir) -> dict | None:
    path = decision_path(date, picks_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_daily_decision.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bts/daily_decision.py tests/test_daily_decision.py
git commit -m "feat(decision): decision.json reader/writer (bts_daily_decision_v1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `select_pick` returns `SelectionResult`; `decide_action` returns source

**Files:**
- Modify: `src/bts/strategy.py` (`decide_action`, `select_pick`, add `SelectionResult`)
- Modify: `src/bts/orchestrator.py` (`run_and_pick` returns/threads metadata)
- Modify: `src/bts/cli.py` (the two `select_pick`/`run_and_pick` callers)
- Test: `tests/test_decide_action.py`, `tests/test_strategy.py`

**Interfaces:**
- Produces:
  - `decide_action(ctx, streak, saver) -> tuple[str, str]` — `(action, source)` where `source ∈ {"mdp","heuristic"}`.
  - `SelectionResult(pick_result, action, source, primary_candidate, double_candidate, no_pick_reason)` dataclass in `strategy.py`.
  - `select_pick(...) -> SelectionResult` (no longer `PickResult | None`; no file writes).
  - `run_and_pick(...) -> tuple[predictions, SelectionResult, tier_name]`.

- [ ] **Step 1: Write the failing test for `decide_action` source**

```python
# tests/test_decide_action.py — add
def test_decide_action_returns_source_heuristic():
    from bts.strategy import decide_action, DecisionContext
    ctx = DecisionContext(primary_p=0.50, second_p=None, has_diff_game=False,
                          date="2026-04-01", allow_double=True, mdp=None)
    action, source = decide_action(ctx, streak=10, saver=True)
    assert action == "skip"
    assert source == "heuristic"
```

- [ ] **Step 2: Verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decide_action.py::test_decide_action_returns_source_heuristic -q`
Expected: FAIL — `decide_action` returns a str, not a tuple (`cannot unpack`).

- [ ] **Step 3: Change `decide_action` to return `(action, source)`**

In `src/bts/strategy.py`, `decide_action` currently returns `action: str`. Change every `return`/final value to `(action, source)`:
- `_mdp_action_from(...)` returns the action or `None`. Set `source = "mdp"` when it returns non-None, else `"heuristic"`.

```python
def decide_action(ctx: DecisionContext, streak: int, saver: bool) -> tuple[str, str]:
    mdp_action = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
    if mdp_action is not None:
        action, source = mdp_action, "mdp"
    else:
        source = "heuristic"
        if ctx.primary_p < SKIP_THRESHOLD:
            action = "skip"
        elif _double_threshold(streak) is not None and ctx.has_diff_game and ctx.second_p is not None:
            p_both = ctx.primary_p * ctx.second_p
            action = "double" if p_both >= _double_threshold(streak) else "single"
        else:
            action = "single"
    if action == "double" and not ctx.allow_double:
        action = "single"
    if action == "double" and not ctx.has_diff_game:
        action = "single"
    return action, source
```

Update the existing `decide_action` call inside `select_pick`: `action = decide_action(...)` → `action, source = decide_action(ctx, streak, saver)`.

- [ ] **Step 4: Fix the existing `test_decide_action.py` assertions**

Every existing test asserting `decide_action(...) == "<x>"` becomes `decide_action(...)[0] == "<x>"` (the action). Update all ~12.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decide_action.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing test for `SelectionResult`**

```python
# tests/test_strategy.py — add inside TestSelectPick
@patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
def test_select_pick_returns_selection_result(self, _s, tmp_path):
    from bts.strategy import select_pick, SelectionResult
    from unittest.mock import patch as p2
    preds = _predictions([{"batter_name": "Weak", "p_game_hit": 0.50}])
    with p2("bts.strategy._load_mdp", return_value={"x": 1}), \
         p2("bts.simulate.mdp.lookup_action", return_value="skip"):
        sel = select_pick(preds, "2026-04-01", tmp_path, streak=10)
    assert isinstance(sel, SelectionResult)
    assert sel.pick_result is None
    assert sel.action == "skip" and sel.source == "mdp"
    assert sel.primary_candidate["batter_name"] == "Weak"
```

- [ ] **Step 6: Verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest "tests/test_strategy.py::TestSelectPick::test_select_pick_returns_selection_result" -q`
Expected: FAIL — `cannot import name 'SelectionResult'` / `select_pick` returns `PickResult|None`.

- [ ] **Step 7: Implement `SelectionResult` + wrap every `select_pick` return**

Add to `strategy.py`:

```python
@dataclass
class SelectionResult:
    pick_result: "PickResult | None"
    action: str | None              # "skip"|"single"|"double", or None if no action reached
    source: str | None              # "mdp"|"heuristic", or None
    primary_candidate: dict | None  # the executable best_row (declined on skip, chosen on pick)
    double_candidate: dict | None
    no_pick_reason: str | None      # "no_eligible"|"status_failure"|"no_valid_predictions"|None
```

Refactor `select_pick` so **every** return path returns a `SelectionResult` (drop the marker write + `persist_skip_decision` param entirely):
- locked existing pick → `SelectionResult(PickResult(daily=current, locked=True), action=None, source=None, primary_candidate=None, double_candidate=None, no_pick_reason=None)`.
- `require_detailed_statuses` with no statuses → `SelectionResult(None, None, None, None, None, "status_failure")`.
- `available.empty` (no `current`) → `SelectionResult(None, None, None, None, None, "no_eligible")`.
- `valid.empty` → `SelectionResult(None, None, None, None, None, "no_valid_predictions")`.
- after `action, source = decide_action(...)`: build `primary_candidate = _row_to_candidate(best_row)`, `double_candidate = _row_to_candidate(second_row) if second_row is not None else None`.
  - `action == "skip"` → `SelectionResult(None, "skip", source, primary_candidate, double_candidate, None)`.
  - else build the pick and → `SelectionResult(PickResult(daily=...), action, source, primary_candidate, double_candidate, None)`.

Add a `_row_to_candidate(row)` helper (native-typed): `{"batter_id": int(row["batter_id"]), "batter_name": row.get("batter_name"), "team": row.get("team"), "game_pk": (int(best_game_pk) if not pd.isna(best_game_pk) else None), "pitcher_name": row.get("pitcher_name"), "p_game_hit": float(row["p_game_hit"])}`.

- [ ] **Step 8: Update `select_pick` callers**

- `orchestrator.run_and_pick`: `result = select_pick(...)` → `sel = select_pick(...)`; the function now returns `(predictions, sel, tier_name)` (change the third tuple element from `result` to `sel`). Update its docstring/return type.
- `cli.py` (two call sites ~1118, ~1273 — `run`/`preview`): `result = select_pick(...)` → `result = select_pick(...).pick_result`.
- The scheduler shadow call (`scheduler.py:1084`, `for_shadow=True`): `result = select_pick(...)` → `result = select_pick(...).pick_result`.

(Scheduler `run_and_pick` consumers are updated in Task 3.)

- [ ] **Step 9: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py tests/test_decide_action.py -q`
Expected: PASS (update any `TestSelectPick` test that did `result = select_pick(...)` then `result.daily`/`result.locked` to use `.pick_result` — these are the assertions in `test_basic_pick`, `test_double_down_*`, `test_locked_when_game_started`, etc.; mechanical `.pick_result` insertion).

- [ ] **Step 10: Commit**

```bash
git add src/bts/strategy.py src/bts/orchestrator.py src/bts/cli.py tests/test_strategy.py tests/test_decide_action.py
git commit -m "refactor(strategy): select_pick returns SelectionResult; decide_action returns source

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: scheduler writes `decision.json` at finalization points

**Files:**
- Modify: `src/bts/scheduler.py` (`run_and_pick` consumers, `_deliver_and_lock_pick`, `run_single_check`, `run_day`)
- Modify: `src/bts/orchestrator.py` (`run_and_pick` already returns the `SelectionResult` from Task 2)
- Test: `tests/test_scheduler_decision_record.py` (new)

**Interfaces:**
- Consumes: `daily_decision.write_decision`, `SelectionResult` (Tasks 1, 2), `picks.pick_was_delivered`.
- Produces: `decision.json` at every finalization; the loop tracks `final_skip_candidate` + `committed_pick_written`.

- [ ] **Step 1: Write the failing tests** (mock the cascade; assert the record)

```python
# tests/test_scheduler_decision_record.py
import json
from pathlib import Path
from bts.daily_decision import load_decision
# Helpers: build a SchedulerState + a SelectionResult; call the small writer helpers the
# implementation exposes (see Step 3). Tests target the pure decision-writing helpers, NOT the
# whole daemon loop, to stay fast and deterministic.
from bts.scheduler import (_write_commit_decision, _write_classification_decision,
                           _write_endofday_skip, FinalizationState)

def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}

def test_commit_writes_scoreable_pick(tmp_path):
    fs = FinalizationState()
    _write_commit_decision(tmp_path, "2026-06-20", action="single", source="mdp",
                           primary=_cand(), double_down=None, delivery_status="delivered", fs=fs)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "single" and d["scoreable"] is True and d["delivery_status"] == "delivered"
    assert fs.committed_pick_written is True

def test_classification_writes_only_when_delivered(tmp_path):
    fs = FinalizationState()
    # delivered existing pick -> scoreable record
    _write_classification_decision(tmp_path, "2026-06-20", action="single", delivered=True, double_down=None, primary=_cand(), fs=fs)
    assert load_decision("2026-06-20", tmp_path)["scoreable"] is True and fs.committed_pick_written
    # NON-delivered (stale preview classified-locked) -> NO record, not committed
    fs2 = FinalizationState()
    _write_classification_decision(tmp_path, "2026-06-21", action="single", delivered=False, double_down=None, primary=_cand(), fs=fs2)
    assert load_decision("2026-06-21", tmp_path) is None and fs2.committed_pick_written is False

def test_endofday_skip_only_when_uncommitted_and_candidate(tmp_path):
    fs = FinalizationState()
    fs.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-20", fs)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "skip" and d["source"] == "mdp" and d["scoreable"] is False and d["streak"] == 10
    # committed pick suppresses the skip
    fs2 = FinalizationState(); fs2.committed_pick_written = True
    fs2.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-22", fs2)
    assert load_decision("2026-06-22", tmp_path) is None
    # no candidate -> no record
    _write_endofday_skip(tmp_path, "2026-06-23", FinalizationState())
    assert load_decision("2026-06-23", tmp_path) is None
```

- [ ] **Step 2: Verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler_decision_record.py -q`
Expected: FAIL — `cannot import name 'FinalizationState'` / the helpers don't exist.

- [ ] **Step 3: Implement the finalization helpers**

Add to `scheduler.py`:

```python
from dataclasses import dataclass, field

@dataclass
class FinalizationState:
    final_skip_candidate: dict | None = None   # {"primary":..., "streak":..., "saver_available":...}
    committed_pick_written: bool = False

def _write_commit_decision(picks_dir, date, *, action, source, primary, double_down, delivery_status, fs):
    from bts.daily_decision import write_decision
    write_decision(date, picks_dir, action=action, source=(source or "unknown"),
                   primary=primary, double_down=double_down,
                   delivery_status=delivery_status, scoreable=True)
    fs.committed_pick_written = True

def _write_classification_decision(picks_dir, date, *, action, delivered, primary, double_down, fs):
    # A genuinely DELIVERED existing pick recovered via classification-lock -> scoreable.
    # A non-delivered classification-lock (stale preview locked by game-start/status) -> nothing.
    if not delivered:
        return
    _write_commit_decision(picks_dir, date, action=action, source="unknown",
                           primary=primary, double_down=double_down, delivery_status="delivered", fs=fs)

def _write_endofday_skip(picks_dir, date, fs):
    from bts.daily_decision import write_decision
    if fs.committed_pick_written or not fs.final_skip_candidate:
        return
    c = fs.final_skip_candidate
    write_decision(date, picks_dir, action="skip", source="mdp", primary=c.get("primary"),
                   streak=c.get("streak"), saver_available=c.get("saver_available"),
                   delivery_status="not_applicable", scoreable=False)
```

- [ ] **Step 4: Verify the helper tests pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler_decision_record.py -q`
Expected: PASS

- [ ] **Step 5: Wire the helpers into the daemon control flow**

Integrate (these are existing-code edits; no new test logic, covered by the helper tests above + the full-suite scheduler regression in Step 6):
1. **Capture metadata**: `run_and_pick` returns `(predictions, sel, tier)` (Task 2). In `run_single_check`, surface `sel` (action/source/primary_candidate/double_candidate/no_pick_reason) in its returned dict.
2. **`final_skip_candidate` update** (in `run_day`, each cycle): if `sel.action=="skip" and sel.source=="mdp"` → `fs.final_skip_candidate = {"primary": sel.primary_candidate, "streak": <decision streak>, "saver_available": <saver>}`; on any pick selected/attempted, heuristic/non-mdp skip, no-action, or caught error → `fs.final_skip_candidate = None`.
3. **Commit write**: in `_deliver_and_lock_pick`, at each branch that sets `state.pick_locked=True` after a real delivery, call `_write_commit_decision(...)` with `delivery_status`: `delivered` (public/DM posted), `private_locked` (private branch), `locked_unconfirmed` (delivery_attempted crash-guard). Pass `source`/`primary`/`double_down` from the captured metadata (thread `fs`/metadata into `_deliver_and_lock_pick` via params).
4. **Classification write**: where `run_single_check`/`run_day` lock an existing pick via `classify_pick_lock_state`, call `_write_classification_decision(..., delivered=pick_was_delivered(daily), action="double" if daily.double_down else "single", primary=..., double_down=...)`.
5. **End-of-day skip**: immediately before the end-of-day health-checks block (after final fallback, missed-pick handling, DH rechecks, next-day lookahead, result polling), call `_write_endofday_skip(picks_dir, date, fs)`.

- [ ] **Step 6: Run the scheduler regression**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scheduler*.py tests/test_decision* -q`
Expected: PASS (fix any scheduler test broken by the `run_and_pick` tuple/`SelectionResult` change — update to `.pick_result` / the new dict keys).

- [ ] **Step 7: Commit**

```bash
git add src/bts/scheduler.py src/bts/orchestrator.py tests/test_scheduler_decision_record.py
git commit -m "feat(scheduler): write decision.json at finalization (commit/classification/crash/skip)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `check-results` reads `decision.json` (the #144 fix)

**Files:**
- Modify: `src/bts/cli.py` (`check_results` command)
- Test: `tests/test_cli_integration.py` (`TestBtsCheckResults` + new cases)

**Interfaces:**
- Consumes: `daily_decision.load_decision`, `picks.pick_was_delivered`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_integration.py — add to TestBtsCheckResults
from bts.daily_decision import write_decision

@patch("bts.picks.get_game_statuses_detailed", return_value={})
@patch("bts.picks.check_hit")
def test_check_results_skips_unscoreable_skip_record(self, mock_check, _s, tmp_path):
    picks_dir = tmp_path / "picks"; picks_dir.mkdir()
    save_pick(_sample_daily(bluesky_posted=False), picks_dir)   # stale preview-style file
    save_streak(5, picks_dir)
    write_decision("2026-04-01", picks_dir, action="skip", source="mdp",
                   delivery_status="not_applicable", scoreable=False)
    result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
    assert result.exit_code == 0
    assert "not scoring" in result.output.lower()
    assert load_streak(picks_dir) == 5            # untouched
    mock_check.assert_not_called()

@patch("bts.picks.get_game_statuses_detailed", return_value={})
@patch("bts.picks.check_hit", return_value=True)
def test_check_results_scores_scoreable_decision(self, _c, _s, tmp_path):
    picks_dir = tmp_path / "picks"; picks_dir.mkdir()
    save_pick(_sample_daily(bluesky_posted=True), picks_dir)
    save_streak(3, picks_dir)
    write_decision("2026-04-01", picks_dir, action="single", source="mdp",
                   delivery_status="delivered", scoreable=True)
    result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
    assert "Streak: 4" in result.output

@patch("bts.picks.get_game_statuses_detailed", return_value={})
@patch("bts.picks.check_hit", return_value=True)
def test_check_results_missing_decision_falls_back_to_delivered(self, _c, _s, tmp_path):
    picks_dir = tmp_path / "picks"; picks_dir.mkdir()
    save_pick(_sample_daily(bluesky_posted=True), picks_dir)   # delivered, no decision.json (legacy)
    save_streak(3, picks_dir)
    result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
    assert "Streak: 4" in result.output

@patch("bts.picks.get_game_statuses_detailed", return_value={})
@patch("bts.picks.check_hit")
def test_check_results_missing_decision_undelivered_not_scored(self, mock_check, _s, tmp_path):
    # the core #144 case: a stale preview <date>.json on a skip day, no decision.json, undelivered
    picks_dir = tmp_path / "picks"; picks_dir.mkdir()
    save_pick(_sample_daily(bluesky_posted=False), picks_dir)
    save_streak(5, picks_dir)
    result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
    assert load_streak(picks_dir) == 5
    mock_check.assert_not_called()
```

- [ ] **Step 2: Verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest "tests/test_cli_integration.py::TestBtsCheckResults::test_check_results_skips_unscoreable_skip_record" -q`
Expected: FAIL — current code scores it (no "not scoring"; streak changes).

- [ ] **Step 3: Restructure `check_results` precedence**

After `daily = load_pick(...)` and the helper defs, BEFORE the existing slot-scoring:

```python
from bts.daily_decision import load_decision
from bts.picks import pick_was_delivered

decision = load_decision(date, picks_path)
# shadow paths run regardless (they key off *.shadow.json):
if daily is None:
    reconcile_shadow_result(); write_shadow_status_artifact()
    click.echo(f"No pick found for {date}."); return

if decision is not None:
    scoreable = bool(decision.get("scoreable"))
else:
    scoreable = pick_was_delivered(daily)   # fallback: delivered public/DM (NOT scheduler_state.pick_locked)

if not scoreable:
    reconcile_shadow_result(); write_shadow_status_artifact()
    click.echo(f"{date}: decision was not a committed pick (skip / undelivered) — not scoring."); return
```

Keep the existing `if daily.result in (hit, miss, void): ... return` idempotency block and the slot-scoring AFTER this gate. Remove the old early `if daily is None` block if duplicated (move the shadow calls into the new structure).

- [ ] **Step 4: Update the 7 pre-existing `TestBtsCheckResults` tests**

They use `_sample_daily()` (`bluesky_posted=False`) and expect scoring. Make each reflect a committed pick: either `save_pick(_sample_daily(bluesky_posted=True), ...)` OR add `write_decision(<date>, picks_dir, action="single", source="mdp", delivery_status="delivered", scoreable=True)`. (Choose `bluesky_posted=True` — it exercises the fallback path and is minimal.)

- [ ] **Step 5: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest "tests/test_cli_integration.py::TestBtsCheckResults" -q`
Expected: PASS (new + updated existing)

- [ ] **Step 6: Commit**

```bash
git add src/bts/cli.py tests/test_cli_integration.py
git commit -m "fix(check-results): score only committed-pick decisions (GH #144)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: migrate the skip-policy shadow onto `decision.json`

**Files:**
- Modify: `src/bts/skip_policy_shadow.py`
- Test: `tests/test_skip_policy_shadow.py`

**Interfaces:**
- Consumes: `daily_decision.load_decision`.
- Removed: `record_mdp_skip_decision`, `load_skip_decision`, `skip_decision_path`, `_final_decision`, `_production_picked`/`pick_was_delivered` usage; the `skip_decision.json` marker.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skip_policy_shadow.py — replace the marker/pick_was_delivered tests
import json
from bts.daily_decision import write_decision
from bts.skip_policy_shadow import (record_skip_from_decision, record_pending_skips,
                                    prune_superseded, decision_path)

def _cand(bid=1, p=0.78): return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}

def test_records_mdp_skip_decision(tmp_path):
    write_decision("2026-06-18", tmp_path, action="skip", source="mdp", primary=_cand(1, 0.75),
                   streak=10, delivery_status="not_applicable", scoreable=False)
    rec = record_skip_from_decision("2026-06-18", tmp_path)
    assert rec is not None and rec["divergent"] is True and rec["rank1"]["batter_id"] == 1

def test_ignores_pick_and_heuristic_skip(tmp_path):
    write_decision("2026-06-19", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert record_skip_from_decision("2026-06-19", tmp_path) is None
    write_decision("2026-06-20", tmp_path, action="skip", source="heuristic", primary=_cand(),
                   delivery_status="not_applicable", scoreable=False)
    assert record_skip_from_decision("2026-06-20", tmp_path) is None

def test_prune_drops_record_when_decision_no_longer_mdp_skip(tmp_path):
    write_decision("2026-06-18", tmp_path, action="skip", source="mdp", primary=_cand(),
                   delivery_status="not_applicable", scoreable=False)
    record_skip_from_decision("2026-06-18", tmp_path)
    # decision flips to a committed pick (e.g. late delivery)
    write_decision("2026-06-18", tmp_path, action="single", source="mdp", primary=_cand(),
                   delivery_status="delivered", scoreable=True)
    assert prune_superseded(tmp_path) == ["2026-06-18"]
    assert not decision_path("2026-06-18", tmp_path).exists()
```

- [ ] **Step 2: Verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_skip_policy_shadow.py -q`
Expected: FAIL — `record_skip_from_decision` doesn't exist.

- [ ] **Step 3: Replace the marker layer with decision.json reads**

In `skip_policy_shadow.py`:
- Delete `record_mdp_skip_decision`, `load_skip_decision`, `skip_decision_path`, `SKIP_DECISION_SCHEMA`, `_final_decision`, `_production_picked`, `make_hit_checker`'s `pick_was_delivered` references.
- `record_skip_from_decision(date, picks_dir, *, now=None) -> dict | None`:

```python
def record_skip_from_decision(date, picks_dir, *, now=None):
    from bts.daily_decision import load_decision
    if decision_path(date, picks_dir).exists():       # decision_path = the policy_shadow file path
        return None
    dec = load_decision(date, picks_dir)
    if not dec or dec.get("action") != "skip" or dec.get("source") != "mdp":
        return None
    record = build_divergent_record(date, dec, now=now)   # rank1 = dec["primary"], streak = dec["streak"]
    atomic_write_text(decision_path(date, picks_dir), json.dumps(record, indent=2))
    return record
```
- `record_pending_skips`: iterate `data/picks/*/decision.json`, call `record_skip_from_decision` per date.
- `prune_superseded(picks_dir)`: for each `*.policy_shadow.json`, drop it if its date's `load_decision` is no longer `action=="skip" && source=="mdp"`.
- `build_divergent_record(date, dec, now)`: `rank1 = {fields from dec["primary"]}`, `streak = dec.get("streak")`, `saver_available = dec.get("saver_available")`, `deployed_action="skip"`, `shadow_action="single"`, `divergent=True`, `shadow_pick_result=None`.
- Keep: `reconcile_decision`, `reconcile_pending`, `build_skip_policy_shadow_status`, `write_status`, the `make_hit_checker` MLB-API checker (unchanged), `decision_path` (policy_shadow path), `BREAKEVEN_P` etc.

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_skip_policy_shadow.py tests/test_web_skip_policy_shadow.py -q`
Expected: PASS (delete/replace the obsolete marker tests; keep reconcile/status tests)

- [ ] **Step 5: Update the CLI command + strategy cleanup**

- `cli.py skip-policy-shadow-update`: `record_pending_skips` (now decision-based) + `prune_superseded` + reconcile + status — wording updated ("reads decision.json").
- `strategy.select_pick`: confirm the marker write + `persist_skip_decision` were removed in Task 2 (no `record_mdp_skip_decision` import remains). `scheduler.py:1084` shadow call already uses `.pick_result`.

- [ ] **Step 6: Commit**

```bash
git add src/bts/skip_policy_shadow.py src/bts/cli.py tests/test_skip_policy_shadow.py tests/test_web_skip_policy_shadow.py
git commit -m "refactor(shadow): read decision.json (drop marker/pick_was_delivered layer)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: docs + terminology cleanup + full regression

**Files:**
- Modify: `ARCHITECTURE.md`, `CLAUDE.md`, `docs/audit/2026-06-20-skip-policy-shadow.md`, `src/bts/web.py` (panel wording)

- [ ] **Step 1: Update docs away from marker terminology**

Replace "marker" / `skip_decision.json` / `pick_was_delivered` references in ARCHITECTURE.md (skip-policy subsection), CLAUDE.md (bullet), and the audit doc with the `decision.json` model (cascade single-writer at finalization; check-results `scoreable` gate; shadow reads `decision.json`). Add a short `decision.json` subsection to ARCHITECTURE.md. Update the dashboard panel comment/label in `web.py` if it references the marker.

- [ ] **Step 2: Full not-slow suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" -q`
Expected: PASS (green). Investigate + fix any failure before proceeding.

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md CLAUDE.md docs/audit/2026-06-20-skip-policy-shadow.md src/bts/web.py
git commit -m "docs(decision): document decision.json model; retire marker terminology

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Highest-risk task is Task 3** (scheduler control flow). Keep the decision writes best-effort and OUT of any path that could raise into pick delivery. Verify the end-of-day skip hook fires before the health/idle block on every no-pick exit (normal end + the caught-ContestStateError fallback path), and NOT on the early no-games/dry-run returns.
- After Task 2, the whole tree won't be green until Tasks 3–5 land (the `select_pick`/`run_and_pick` return-shape ripples). Run the targeted task tests at each step; run the full suite at Task 6.
- Do NOT deploy. Deploy is `git push origin main:deploy` — the user's call, after review.
