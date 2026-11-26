# Streak Saver Manual Flag — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unsound `infer_saver`/`best_streak` saver inference with a sound, manually-controlled 3-state Streak Saver flag (`saver_state.json`) that drives the live decision, auto-earns on reaching streak 10, and is marked "used" by the operator via dashboard/CLI.

**Architecture:** A new `bts.saver_state` module owns a persisted `{not_earned, active, used}` flag (with a loader-derived `uninitialized`), guarded atomic transitions, and a fetch-path auto-earn. `load_decision_streak_state` reads the flag (read-only) as the *sole* saver authority. A CLI (`bts saver-state`) and a dashboard POST route perform manual transitions; the dashboard nudges on a likely save and warns on the offline residual.

**Tech Stack:** Python 3, pytest (`UV_CACHE_DIR=/tmp/uv-cache uv run pytest`). Spec: `docs/superpowers/specs/2026-06-18-streak-saver-flag-design.md`.

**Branch:** continue on `phase2a-decide-action` (this supersedes its 2c saver commit). Create the new module + tests; the existing `infer_saver` is retired in Task 8.

**Reference (current saver wiring to replace), `contest_state.py`:** the model-only fallback returns `saver_available=model_saver` (~line 253); the contest-present path computes `contest_saver` via `infer_saver` (~line 286). Both become `load_saver_state(...).is_available`.

---

### Task 1: `saver_state` module — model + read side

**Files:**
- Create: `src/bts/saver_state.py`
- Test: `tests/test_saver_state.py` (new)

- [ ] **Step 1: Write failing tests for `load_saver_state` + season**

`tests/test_saver_state.py`:
```python
import json
from datetime import date
from bts.saver_state import load_saver_state, SaverState, season_for


def _write(picks_dir, obj):
    d = picks_dir / "account_state"; d.mkdir(parents=True, exist_ok=True)
    (d / "saver_state.json").write_text(json.dumps(obj))


def test_missing_file_is_uninitialized(tmp_path):
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized" and s.is_available is False


def test_valid_active_for_matching_season(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "active", "source": "manual_init"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "active" and s.is_available is True


def test_wrong_season_is_uninitialized_not_not_earned(tmp_path):
    _write(tmp_path, {"season": 2025, "state": "active"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized"   # stale -> fail-closed, NOT not_earned


def test_invalid_state_or_bad_json_is_uninitialized(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "bogus"})
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"
    (tmp_path / "account_state" / "saver_state.json").write_text("{not json")
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"


def test_not_earned_and_used_not_available(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "not_earned"})
    assert load_saver_state(tmp_path, season=2026).is_available is False
    _write(tmp_path, {"season": 2026, "state": "used"})
    assert load_saver_state(tmp_path, season=2026).is_available is False


def test_season_for_uses_source_date_year_else_now(tmp_path):
    assert season_for(date(2026, 6, 18), now_year=2027) == 2026
    assert season_for(None, now_year=2027) == 2027
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_saver_state.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the read side**

`src/bts/saver_state.py`:
```python
"""The Streak Saver manual flag: a sound, operator-controlled replacement for the
unsound ledger inference. Persisted at account_state/saver_state.json as one of
{not_earned, active, used}; the loader derives a fail-closed `uninitialized` for a
missing/invalid/stale-season file. See docs/.../2026-06-18-streak-saver-flag-design.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bts.util import atomic_write_text   # NB: defined in bts.util, not bts.picks

_PERSISTED = {"not_earned", "active", "used"}


@dataclass(frozen=True)
class SaverState:
    state: str                     # not_earned | active | used | uninitialized (last never persisted)
    season: int | None
    source: str | None = None
    updated_at: str | None = None

    @property
    def is_available(self) -> bool:
        return self.state == "active"


def _path(picks_dir: Path) -> Path:
    return picks_dir / "account_state" / "saver_state.json"


def season_for(source_date: date | None, *, now_year: int) -> int:
    """Contest season = the observation's calendar year, else the current year."""
    return source_date.year if source_date is not None else now_year


def load_saver_state(picks_dir: Path, *, season: int) -> SaverState:
    """Read the saver flag for `season`. Returns state='uninitialized' (fail-closed,
    DISTINCT from not_earned) when the file is missing, invalid, or for another season."""
    path = _path(picks_dir)
    if not path.exists():
        return SaverState("uninitialized", None)
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return SaverState("uninitialized", None)
    st = d.get("state")
    fseason = d.get("season") if isinstance(d.get("season"), int) else None
    if st not in _PERSISTED or fseason != season:
        return SaverState("uninitialized", fseason)
    return SaverState(st, fseason, d.get("source"), d.get("updated_at"))


def _write_state(picks_dir: Path, *, state: str, season: int, source: str) -> None:
    atomic_write_text(_path(picks_dir), json.dumps({
        "season": season,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }))
```

- [ ] **Step 4: Run — verify pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_saver_state.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/saver_state.py tests/test_saver_state.py
git commit -m "feat(saver): saver_state model + load_saver_state (uninitialized fail-closed, season)"
```

---

### Task 2: guarded transition + fetch-path auto-earn

**Files:**
- Modify: `src/bts/saver_state.py`
- Test: `tests/test_saver_state.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_saver_state.py`:
```python
from bts.saver_state import transition_saver_state, maybe_auto_earn_saver


def test_transition_guarded_by_expected_prior(tmp_path):
    # active -> used only when currently active
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active", season=2026, source="t")
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="used", season=2026, source="t") is True
    assert load_saver_state(tmp_path, season=2026).state == "used"
    # wrong expected_prior -> no-op
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="used", season=2026, source="t") is False


def test_invalid_transition_rejected_unless_forced(tmp_path):
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active", season=2026, source="t")
    # active -> not_earned is NOT an allowed transition (guards a scripted/cross-page POST)
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="not_earned", season=2026, source="t") is False
    assert load_saver_state(tmp_path, season=2026).state == "active"
    # ...but --force (CLI break-glass) may override
    assert transition_saver_state(tmp_path, expected_prior="active", new_state="not_earned", season=2026, source="t", force=True) is True


def test_auto_earn_inits_not_earned_below_10(tmp_path):
    maybe_auto_earn_saver(tmp_path, best_streak=8, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "not_earned"


def test_auto_earn_promotes_not_earned_to_active_at_10(tmp_path):
    maybe_auto_earn_saver(tmp_path, best_streak=8, season=2026)    # -> not_earned
    maybe_auto_earn_saver(tmp_path, best_streak=10, season=2026)   # -> active
    assert load_saver_state(tmp_path, season=2026).state == "active"


def test_auto_earn_will_not_init_active_from_uninitialized_at_10(tmp_path):
    # fail-closed: a fresh file at best_streak>=10 must NOT become active automatically
    maybe_auto_earn_saver(tmp_path, best_streak=12, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"


def test_auto_earn_never_overwrites_used(tmp_path):
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used", season=2026, source="t")
    maybe_auto_earn_saver(tmp_path, best_streak=14, season=2026)
    assert load_saver_state(tmp_path, season=2026).state == "used"
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_saver_state.py -k "transition or auto_earn" -q`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement transitions**

Append to `src/bts/saver_state.py`:
```python
# Allowed (prior -> new) transitions; anything else is REJECTED (so a scripted/cross-page POST
# can't do e.g. active -> not_earned). `force=True` (CLI --force only) bypasses the whitelist.
_ALLOWED = {
    ("uninitialized", "not_earned"), ("uninitialized", "active"), ("uninitialized", "used"),
    ("not_earned", "active"), ("active", "used"), ("used", "active"),
}


def transition_saver_state(picks_dir: Path, *, expected_prior: str, new_state: str,
                           season: int, source: str, force: bool = False) -> bool:
    """Guarded atomic transition: writes `new_state` ONLY if (a) `new_state` is valid, (b)
    `(expected_prior, new_state)` is an allowed transition (unless `force`), and (c) the current
    persisted state still equals `expected_prior` (re-read just before writing). Returns True iff
    written. The single monotonic-safe write path — auto-earn, CLI, and the dashboard all use it."""
    if new_state not in _PERSISTED:
        raise ValueError(f"invalid saver state: {new_state!r}")
    if not force and (expected_prior, new_state) not in _ALLOWED:
        return False
    if load_saver_state(picks_dir, season=season).state != expected_prior:
        return False
    _write_state(picks_dir, state=new_state, season=season, source=source)
    return True


def maybe_auto_earn_saver(picks_dir: Path, *, best_streak: int | None, season: int) -> None:
    """Fetch-path hook. Safe initialization + the only sound auto transition:
    - uninitialized + best_streak < 10  -> not_earned  (no save possible yet)
    - not_earned   + best_streak >= 10  -> active       (sound: best_streak is reliable)
    Never auto-inits `active` from uninitialized at >=10 (could be earned-and-used before we
    saw it -> fail-closed), and never overwrites active/used."""
    if best_streak is None:
        return
    current = load_saver_state(picks_dir, season=season).state
    if current == "uninitialized" and best_streak < 10:
        transition_saver_state(picks_dir, expected_prior="uninitialized",
                               new_state="not_earned", season=season, source="auto_earn")
    elif current == "not_earned" and best_streak >= 10:
        transition_saver_state(picks_dir, expected_prior="not_earned",
                               new_state="active", season=season, source="auto_earn")
```

- [ ] **Step 4: Run — verify pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_saver_state.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/saver_state.py tests/test_saver_state.py
git commit -m "feat(saver): guarded transition_saver_state + fetch-path maybe_auto_earn_saver"
```

---

### Task 3: decision wiring — `saver_state.json` is the sole authority

**Files:**
- Modify: `src/bts/contest_state.py`
- Test: `tests/test_decision_saver_fallback.py`, `tests/test_contest_state.py`

- [ ] **Step 1: Rewrite the saver-fallback tests for the new authority**

Replace `tests/test_decision_saver_fallback.py` body with manual-state tests:
```python
"""The live saver comes ONLY from saver_state.json (Phase 2c rev: replaces infer_saver)."""
import json
from bts.picks import save_streak
from bts.saver_state import transition_saver_state
from bts.contest_state import load_decision_streak_state


def _fresh_contest(picks_dir, best_streak=10):
    d = picks_dir / "account_state"; d.mkdir(parents=True, exist_ok=True)
    (d / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 10,
        "best_streak": best_streak, "source": "mlb_bts_profile", "source_date": "2026-06-18"}))


def test_active_flag_makes_saver_available(tmp_path):
    save_streak(10, tmp_path, saver_available=False)   # model saver irrelevant now
    _fresh_contest(tmp_path)
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    assert load_decision_streak_state(tmp_path).saver_available is True


def test_used_or_uninitialized_flag_means_unavailable(tmp_path):
    save_streak(10, tmp_path, saver_available=True)
    _fresh_contest(tmp_path)
    # no saver_state.json -> uninitialized -> unavailable
    assert load_decision_streak_state(tmp_path).saver_available is False
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used",
                           season=2026, source="t")
    assert load_decision_streak_state(tmp_path).saver_available is False


def test_model_only_fallback_reads_saver_state_not_streak_json(tmp_path):
    # no contest observation at all -> model-only path; saver still from saver_state.json
    save_streak(10, tmp_path, saver_available=True)    # streak.json says saver True...
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used",
                           season=2026, source="t")     # ...but the flag says used
    assert load_decision_streak_state(tmp_path).saver_available is False
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decision_saver_fallback.py -q`
Expected: FAIL — wiring still uses infer_saver / model saver.

- [ ] **Step 3: Wire `saver_state` into both branches of `load_decision_streak_state`**

In `src/bts/contest_state.py` add imports near the top:
```python
from zoneinfo import ZoneInfo                       # if not already imported
from bts.saver_state import load_saver_state, season_for
```
Resolve the ET year once, right after `now` is resolved (BEFORE the `contest is None` branch — the
`season` must NOT dereference `contest` until the None-check has passed):
```python
    now_year = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York")).year
```
Compute the saver **per-branch** from `saver_state.json` (the sole authority):
- **Model-only branch** (`contest is None`, AFTER the `require_contest_state` raise): use
  `load_saver_state(picks_dir, season=season_for(None, now_year=now_year)).is_available` for the
  returned `saver_available=` instead of `model_saver`.
- **Contest-present branch:** `season = season_for(contest.source_date, now_year=now_year)`;
  `contest_saver = load_saver_state(picks_dir, season=season).is_available`. DELETE the entire
  `if contest.saver_available is not None: … else: <infer_saver/parse_latest_ledger>` saver block.
- Keep `model_saver` / `model_streak` ONLY for the `model_saver_available=` / `model_streak=`
  diagnostic fields on the returned `DecisionStreakState`.

- [ ] **Step 4: Run the saver + contest-state suites**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decision_saver_fallback.py tests/test_contest_state.py -q`
Expected: PASS. Fix any `test_contest_state.py` saver assertions that referenced the old proxy — convert them to set `saver_state.json` via `transition_saver_state` and assert accordingly (the streak/status assertions are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/bts/contest_state.py tests/test_decision_saver_fallback.py tests/test_contest_state.py
git commit -m "feat(saver): load_decision_streak_state reads saver_state.json as sole authority (both branches); retire infer_saver/contest.saver_available from the decision"
```

---

### Task 4: fetch-path auto-earn + deprecate `set-contest-streak --saver-available`

**Files:**
- Modify: `src/bts/cli.py`
- Test: `tests/test_cli_integration.py`

- [ ] **Step 1: Write the failing test (auto-earn fires from the fetch path)**

Add to `tests/test_cli_integration.py` a test that runs `fetch-contest-streak` (mock the profile
to return `activeStreak=10, seasonBestStreak=10`) into a tmp `picks_dir` with no prior
`saver_state.json`, then asserts the flag is `uninitialized` (best≥10 from a cold file →
fail-closed, NOT auto-active); and a second test where a prior `not_earned` file + a fetch at
best≥10 → `active`. **Mock the function-local imports at their source modules** as the existing
`fetch-contest-streak` tests do — `bts.contest_fetch.fetch_profile`, `bts.cli._fetch_rounds`, and
the session/auth — not at the `bts.cli` re-import site.

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_integration.py -k "auto_earn or saver" -q`
Expected: FAIL.

- [ ] **Step 3: Call `maybe_auto_earn_saver` from the fetch write path**

In `src/bts/cli.py` `fetch-contest-streak`, AFTER `_atomic_write_json(out_path, observation)`
(~`cli.py:1644`), add — using the names actually in scope there (`source_date`, `observation`,
the `--picks-dir` arg):
```python
        from bts.saver_state import maybe_auto_earn_saver, season_for
        from zoneinfo import ZoneInfo
        _season = season_for(source_date, now_year=datetime.now(ZoneInfo("America/New_York")).year)
        maybe_auto_earn_saver(Path(picks_dir), best_streak=observation["best_streak"], season=_season)
```
In `set-contest-streak` (after it writes the manual override), the in-scope names are
`observed_date`, `best_streak`, `picks_dir`:
```python
        from bts.saver_state import maybe_auto_earn_saver, season_for
        from zoneinfo import ZoneInfo
        _season = season_for(observed_date, now_year=datetime.now(ZoneInfo("America/New_York")).year)
        maybe_auto_earn_saver(Path(picks_dir), best_streak=best_streak, season=_season)
```

- [ ] **Step 4: Deprecate `set-contest-streak --saver-available`**

In `set-contest-streak`, if `saver_available is not None`, `click.echo` a deprecation warning ("--saver-available no longer affects the live saver; use `bts saver-state`") and do NOT write it into the contest manual file's saver field (write `None`). Update `tests/test_cli_integration.py:254`-area saver assertions accordingly (the option is inert for the decision).

- [ ] **Step 5: Run**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_integration.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bts/cli.py tests/test_cli_integration.py
git commit -m "feat(saver): auto-earn from the fetch write path; deprecate set-contest-streak --saver-available"
```

---

### Task 5: `bts saver-state` CLI (show / init / use / undo / --force)

**Files:**
- Modify: `src/bts/cli.py`
- Test: `tests/test_cli_integration.py`

- [ ] **Step 1: Write failing tests**

Add tests: `bts saver-state --show` prints the current state; `--init active` on a fresh (uninitialized) dir sets active; `--init active` on an already-`not_earned` dir errs without `--force` and succeeds with `--force`; `--use` flips active→used and no-ops (clear message) when not active; `--undo` flips used→active. Use `CliRunner`; the season defaults to the current ET year (allow `--season` override for tests, default to ET-now).

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_integration.py -k saver_state_cli -q`
Expected: FAIL.

- [ ] **Step 3: Implement the command**

Add `@cli.command(name="saver-state")` with options `--show/--init <state>/--use/--undo`, `--force`, `--season` (default current ET year), `--picks-dir`. Route through `transition_saver_state`:
- `--init S`: `transition_saver_state(expected_prior="uninitialized", new_state=S)`; if it returns False and `--force`, re-issue with `expected_prior=<current state>`; else error "already initialized as <state>; use --force".
- `--use`: `transition_saver_state(expected_prior="active", new_state="used")`; echo result.
- `--undo`: `transition_saver_state(expected_prior="used", new_state="active")`; echo result.
- `--show`: print `load_saver_state(...).state`.

- [ ] **Step 4: Run**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_integration.py -k saver_state_cli -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/cli.py tests/test_cli_integration.py
git commit -m "feat(saver): bts saver-state CLI (show/init/use/undo, guarded --init/--force)"
```

---

### Task 6: dashboard — saver state, guarded POST, nudge, warning

**Files:**
- Modify: `src/bts/web.py`
- Test: `tests/test_web_saver.py` (new)

The dashboard has only `do_GET` + `ThreadingHTTPServer`, and the existing web tests exercise
**pure helpers**, not the `BaseHTTPRequestHandler`. So put all logic in pure, directly-tested
helpers and keep `do_POST` thin. Note: `DecisionStreakState` has **no** `best_streak`, so the
warning context must load the contest observation itself.

- [ ] **Step 1: Write failing tests on the pure helpers**

`tests/test_web_saver.py`:
```python
import json
from datetime import datetime, timezone
from bts.web import saver_dashboard_context, saver_transition_response
from bts.saver_state import transition_saver_state

NOW = datetime(2026, 6, 18, 16, 0, tzinfo=timezone.utc)


def _setup(tmp_path, state, best=10, active=10):
    d = tmp_path / "account_state"; d.mkdir(parents=True, exist_ok=True)
    (d / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": active,
        "best_streak": best, "source": "mlb_bts_profile", "source_date": "2026-06-18"}))
    if state != "uninitialized":
        transition_saver_state(tmp_path, expected_prior="uninitialized", new_state=state,
                               season=2026, source="t")


def test_button_visibility(tmp_path):
    _setup(tmp_path, "active")
    assert saver_dashboard_context(tmp_path, now=NOW).button == "mark_used"
    _setup(tmp_path, "used")
    assert saver_dashboard_context(tmp_path, now=NOW).button == "undo"
    _setup(tmp_path, "not_earned")
    assert saver_dashboard_context(tmp_path, now=NOW).button is None


def test_warning_when_active_past_15(tmp_path):
    _setup(tmp_path, "active", best=16, active=16)
    assert saver_dashboard_context(tmp_path, now=NOW).warning is True


def test_transition_rejects_wrong_expected_prior(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="not_earned",
                                        new_state="used", same_origin=True, now=NOW)
    assert code == 409


def test_transition_rejects_cross_origin(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="active",
                                        new_state="used", same_origin=False, now=NOW)
    assert code == 403


def test_transition_marks_used(tmp_path):
    _setup(tmp_path, "active")
    code, _ = saver_transition_response(tmp_path, expected_prior="active",
                                        new_state="used", same_origin=True, now=NOW)
    assert code == 200
    assert saver_dashboard_context(tmp_path, now=NOW).state == "used"
```
(A nudge test — a stable held not_hit at 10–15 in `contest_ledger.jsonl` → `context.nudge is True` —
mirrors the Task 8 `likely_save` fixtures.)

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_saver.py -q`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the helpers + a thin `do_POST`**

In `src/bts/web.py`:
- `saver_dashboard_context(picks_dir, *, now) -> SaverDashboardContext` (a small dataclass) — loads
  the contest observation (`load_contest_streak_state`, for `best_streak`/`active_streak`), the flag
  (`load_saver_state` for `season_for(contest.source_date, now_year=ET(now).year)`), and the nudge
  (`from bts.contest_ledger import parse_latest_ledger, likely_save`). Fields: `state`;
  `button ∈ {"mark_used","undo",None}` (`mark_used` iff `active`, `undo` iff `used`); `expected_prior`
  (= current state, for the form's hidden field); `nudge` (`likely_save(...)` and `state=="active"`);
  `warning` (`state=="active"` and `best_streak > 15`).
- `saver_transition_response(picks_dir, *, expected_prior, new_state, same_origin, now) -> (int, str)`
  — `403` if not `same_origin`; else `transition_saver_state(...)` (its whitelist enforces valid
  transitions); `200` on success, `409` on a guard mismatch.
- `do_POST`: route `/saver/transition`; parse the form (`expected_prior`, `new_state`); derive
  `same_origin` by comparing the `Origin`/`Referer` host to the `Host` header; call
  `saver_transition_response`; write its status code + short body. Render the
  button/nudge/warning from `saver_dashboard_context` into the page.

- [ ] **Step 4: Run**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_saver.py tests/test_web_render.py tests/test_web_streak_subtitle.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/web.py tests/test_web_saver.py
git commit -m "feat(saver): dashboard saver context + guarded POST helper (mark-used/undo) + nudge + offline warning"
```

---

### Task 7: health validation (season mismatch / uninitialized in-zone)

**Files:**
- Modify: `src/bts/health/contest_state.py` (or the appropriate health check) + register
- Test: `tests/health/test_contest_state.py`

- [ ] **Step 1: Write failing tests**

Tests: a saver-state health check WARNs when `load_saver_state` is `uninitialized` **and** the contest `active_streak` is in 10–15 (you're in the zone with no flag → decisions are wrongly conservative — operator must init); and WARNs on a stale-season file. No alert when `not_earned`/`active`/`used` for the current season.

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/health/test_contest_state.py -k saver -q`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Add the saver checks **within the existing `bts.health.contest_state.check`** (already registered
in the runner — no separate registration): load the flag for the contest season; WARN on
`uninitialized` while `active_streak ∈ [10,15]` ("Streak Saver flag uninitialized in the 10–15 zone
— run `bts saver-state --init`") and on a season mismatch (`load_saver_state` returns
`uninitialized` with a non-None `season` for the stale case — Task 1 preserves that, so the check
can distinguish stale-season from missing).

- [ ] **Step 4: Run**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/health/test_contest_state.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/health/ tests/health/test_contest_state.py
git commit -m "feat(saver): health WARN on uninitialized-in-zone / stale-season saver flag"
```

---

### Task 8: retire `infer_saver`; repurpose the ledger parser for the nudge; full suite

**Files:**
- Modify: `src/bts/contest_ledger.py`, `tests/test_contest_ledger.py`

- [ ] **Step 1: Remove `infer_saver`**

Delete `infer_saver` from `src/bts/contest_ledger.py` (no live consumer remains after Task 3). Keep `parse_latest_ledger` + `LedgerRound` (used by the Task-6 nudge). Add a small `likely_save(rounds) -> bool` helper (a stable not_hit at pre-streak 10–15 that held) for the dashboard nudge.

- [ ] **Step 2: Update tests**

In `tests/test_contest_ledger.py`: drop the `infer_saver` tests; keep the `parse_latest_ledger` tests; add `likely_save` tests (the old consumption fixtures become likely-save fixtures: a stable held not_hit at 10–15 → True; unstable/DD/out-of-zone → False).

- [ ] **Step 3: Run — verify the module + its tests pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_ledger.py -q`
Expected: PASS.

- [ ] **Step 4: Full affected-suite run (the golden guard for the whole change)**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_saver_state.py tests/test_contest_state.py tests/test_decision_saver_fallback.py tests/test_contest_ledger.py tests/test_cli_integration.py tests/test_orchestrator.py tests/test_web_saver.py tests/test_web_render.py tests/health/test_contest_state.py tests/test_strategy.py tests/test_decide_action.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/contest_ledger.py tests/test_contest_ledger.py
git commit -m "refactor(saver): retire infer_saver; parse_latest_ledger now backs the dashboard nudge only"
```

---

### Task 9: migration + ship

- [ ] **Step 1:** Read-only review of the full diff vs `main`; confirm `decide_action` (2a) + the atomic policy save are intact and the saver path now flows only through `saver_state.json`.
- [ ] **Step 2:** PR + `git push origin main:deploy` (canary + auto-rollback). The deploy is saver-neutral until the flag is initialized (uninitialized → unavailable), so it cannot make a live decision worse than today's conservative state.
- [ ] **Step 3 (on the box, after deploy):** initialize the live flag and clear the bridge override:
  ```bash
  ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/bts saver-state --init active --picks-dir data/picks'
  ssh bts-hetzner 'rm -f ~/projects/bts/data/picks/account_state/contest_streak.manual.json'   # clear today's set-contest-streak bridge
  ```
- [ ] **Step 4 (verify):** `ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python -c "from pathlib import Path; from bts.contest_state import load_decision_streak_state as L; s=L(Path(\"data/picks\")); print(s.streak, s.saver_available)"'` → `10 True`.

## Self-Review

**Spec coverage:** §2.1 model + uninitialized (Task 1); §2.3 transitions + §2.4 bootstrap + §2.5 season + §2.7 guarded atomic auto-earn (Tasks 1–2); §2.2/§2.9 sole-authority wiring both branches (Task 3); auto-earn in fetch path + deprecate override (Task 4); §2.8 CLI (Task 5); §2.6/§2.7 dashboard POST/nudge/warning (Task 6); §4 health (Task 7); retire infer_saver + nudge repurpose (Task 8); §5 migration (Task 9). **Type consistency:** `SaverState(state, season, source, updated_at)` + `.is_available`; `transition_saver_state(expected_prior, new_state, season, source) -> bool`; `maybe_auto_earn_saver(best_streak, season)`; `season_for(source_date, now_year)` — used identically across tasks. **No placeholders.** **Note:** confirm `atomic_write_text`'s signature in `picks.py` during Task 1; confirm the exact `do_GET`/server insertion points in `web.py` during Task 6.
