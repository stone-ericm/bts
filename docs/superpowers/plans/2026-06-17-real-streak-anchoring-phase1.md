# Real-Streak Anchoring — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live decision streak reflect the user's real MLB streak (never the model's hypothetical), stop the fetch discarding a current `activeStreak`, persist the per-round ledger, remove the blanket double-down freeze, and label the dashboard with confidence.

**Architecture:** `load_decision_streak_state` (in `contest_state.py`) becomes the single chokepoint: when a contest observation exists, the decision streak is *always* the contest value (model can never raise it), with a computed `status ∈ {fresh, lagged, stale}` and `allow_double` no longer frozen on staleness. The fetch path (`contest_fetch.py` + the `fetch-contest-streak` CLI command) is split so the `activeStreak` snapshot persists even when the per-round predictions array lags, and the full ledger is saved for analysis. The dashboard reads the new `status` to label "last confirmed … through …". No strategy-engine refactor (that is Phase 2).

**Tech Stack:** Python 3, `click` CLI, `pytest` (`UV_CACHE_DIR=/tmp/uv-cache uv run pytest`), `httpx` for MLB fetch. Spec: `docs/superpowers/specs/2026-06-17-real-streak-anchoring-design.md`.

**Branch:** `real-streak-anchoring` (already created off `main`; the spec is committed there as `01b1ac9`).

**Reference — current behavior being replaced** (`contest_state.py:261-273`): the stale branch returns `max(model_streak, contest.streak)` with `allow_double=False`. That is the inflation + freeze we are removing.

---

### Task 1: Decision-streak core — model never raises, status fresh/lagged/stale, no freeze

**Files:**
- Modify: `src/bts/contest_state.py` (`load_decision_streak_state`; add `_has_unconfirmed_miss`)
- Test: `tests/test_contest_state.py`

- [ ] **Step 1: Write the keystone regression test (the incident)**

Add to `tests/test_contest_state.py` (uses the existing module-level `_write_pick(path, result)` helper):

```python
def test_model_never_inflates_decision_streak(tmp_path):
    """The 2026-06-17 incident: model replay = 10 (missed 6/11 entry), real MLB = 8.
    The decision streak must be 8, NOT max(10, 8)=10, and doubles must stay enabled."""
    from bts.contest_state import load_decision_streak_state

    (tmp_path / "streak.json").write_text(json.dumps({"streak": 10, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")          # latest resolved local pick
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 8,
        "best_streak": 9, "source": "mlb_bts_profile", "source_date": "2026-06-15"}))

    state = load_decision_streak_state(tmp_path)

    assert state.streak == 8            # real MLB streak, NOT the inflated model 10
    assert state.model_streak == 10
    assert state.allow_double is True   # no staleness freeze
    assert state.status == "lagged"     # source_date 06-15 is 1 settled pick behind 06-16
```

- [ ] **Step 2: Run it — verify it fails against current code**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_state.py::test_model_never_inflates_decision_streak -v`
Expected: FAIL — current code returns `streak == 10` (`max(10, 8)`), `allow_double is False`, `status == "stale"`.

- [ ] **Step 3: Add the `_has_unconfirmed_miss` helper**

Insert after `resolved_pick_settlement_gap` (after `contest_state.py:179`):

```python
def _has_unconfirmed_miss(picks_dir: Path, source_date: date) -> bool:
    """True if a settled local pick dated strictly after ``source_date`` is a MISS.

    The bot resolves a pick locally before the contest posts it; a local miss the
    contest hasn't confirmed means the real streak may have reset (stale-high risk).
    The bot only *recommends*, so this is an uncertainty signal, not proof of a reset.
    """
    for path in picks_dir.glob("*.json"):
        if not _ISO_DATE_RE.match(path.stem):
            continue
        if date.fromisoformat(path.stem) <= source_date:
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if body.get("result") == "miss":
            return True
    return False
```

- [ ] **Step 4: Replace the post-contest body of `load_decision_streak_state`**

Replace everything from `if contest_state_is_fresh(contest, picks_dir):` (`contest_state.py:246`) to the end of the function (`:273`) with:

```python
    if contest_state_is_fresh(contest, picks_dir):
        status = "fresh"
        message = f"using fresh contest streak from {contest.source}"
    elif contest.source_date is None:
        status = "stale"
        message = "contest streak has no source_date; treating as last-confirmed (stale)"
    else:
        gap = resolved_pick_settlement_gap(picks_dir, contest.source_date)
        if gap >= 2 or _has_unconfirmed_miss(picks_dir, contest.source_date):
            status = "stale"
            message = "contest streak stale; using last confirmed value (current may be lower)"
        else:
            status = "lagged"
            message = "contest streak lagged by expected overnight settlement; using last confirmed value"

    # The decision streak is ALWAYS the contest (real MLB) value. The model is a
    # research replay of the bot's own suggestions and can NEVER raise it (the
    # 2026-06-17 inflation bug). Doubles are no longer frozen on staleness — Phase 1
    # surfaces stale-high via `status`; Phase 2 makes strategy act on the uncertainty.
    return DecisionStreakState(
        streak=contest.streak,
        saver_available=contest_saver,
        allow_double=True,
        source="contest",
        status=status,
        model_streak=model_streak,
        model_saver_available=model_saver,
        contest_streak=contest.streak,
        contest_saver_available=contest.saver_available,
        contest_source_date=contest.source_date,
        message=message,
    )
```

- [ ] **Step 5: Run the keystone — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_state.py::test_model_never_inflates_decision_streak -v`
Expected: PASS.

- [ ] **Step 6: Update the three existing tests that encode the OLD behavior**

In `tests/test_contest_state.py`, replace `test_stale_contest_state_freezes_higher_streak_and_disables_double` (`:72`), `test_stale_contest_state_never_lowers_model_streak` (`:99`), and `test_missing_source_date_disables_double` (`:203`) with:

```python
def test_lagged_contest_uses_contest_value_keeps_doubles(tmp_path):
    # Was test_stale_..._freezes...: model 5, contest 7, 1 pick behind -> lagged.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 5, "saver_available": True}))
    _write_pick(tmp_path / "2026-05-28.json", "hit")
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7, "source": "manual_screenshot",
        "source_date": "2026-05-28", "saver_available": True}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 7
    assert state.allow_double is True
    assert state.status == "lagged"
    assert state.saver_available is True   # contest saver is known (True) here


def test_stale_contest_model_does_not_inflate(tmp_path):
    # Was test_stale_..._never_lowers_model_streak: model 12 must NOT raise the streak.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 12, "saver_available": True}))
    _write_pick(tmp_path / "2026-05-29.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7, "source_date": "2026-05-28"}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 7           # contest value; model 12 cannot inflate it
    assert state.model_streak == 12
    assert state.allow_double is True
    assert state.status == "lagged"


def test_missing_source_date_is_stale_but_keeps_doubles(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 3, "saver_available": True}))
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 3,
        "best_streak": 9, "source": "mlb_bts_profile"}))  # no source_date
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 3
    assert state.allow_double is True
    assert state.status == "stale"
```

- [ ] **Step 7: Add status + override + unconfirmed-miss coverage tests**

Append to `tests/test_contest_state.py`:

```python
def test_two_pick_gap_is_stale(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")
    _write_pick(tmp_path / "2026-06-17.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "active_streak": 8, "best_streak": 9, "source": "mlb_bts_profile",
        "source_date": "2026-06-15"}))   # 2 settled picks behind -> stale
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8 and state.status == "stale" and state.allow_double is True


def test_unconfirmed_local_miss_marks_stale(tmp_path):
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 0, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-17.json", "miss")   # local reset MLB hasn't posted
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.json").write_text(json.dumps({
        "active_streak": 8, "best_streak": 9, "source": "mlb_bts_profile",
        "source_date": "2026-06-16"}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8            # last confirmed; Phase 1 doesn't lower it
    assert state.status == "stale"      # but flags the stale-high risk


def test_unexpired_override_drives_decision_with_status(tmp_path):
    # Phase 1 §P1.4: an operator override is a confirmed contest observation.
    from bts.contest_state import load_decision_streak_state
    (tmp_path / "streak.json").write_text(json.dumps({"streak": 10, "saver_available": True}))
    _write_pick(tmp_path / "2026-06-16.json", "hit")
    sd = tmp_path / "account_state"; sd.mkdir()
    (sd / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 8, "best_streak": 9, "source": "manual_cli",
        "source_date": "2026-06-16", "override_expires_at": "2099-01-01T00:00:00Z"}))
    state = load_decision_streak_state(tmp_path)
    assert state.streak == 8 and state.source == "contest" and state.allow_double is True
    assert state.status == "fresh"      # source_date 06-16 covers latest pick 06-16
```

- [ ] **Step 8: Run the whole contest_state suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_state.py tests/test_decision_saver_fallback.py tests/test_contest_state_corrupt_precedence.py -v`
Expected: PASS (all). If `test_decision_saver_fallback.py` asserts the old stale `saver_available is False`, update it to the new contest-saver behavior (saver follows `contest_saver`, not a forced False).

- [ ] **Step 9: Commit**

```bash
git add src/bts/contest_state.py tests/test_contest_state.py tests/test_decision_saver_fallback.py
git commit -m "fix(streak): decision streak = real MLB streak, never the model; status fresh/lagged/stale; no staleness DD-freeze"
```

---

### Task 2: Fetch `not_hit` normalization

**Files:**
- Modify: `src/bts/contest_fetch.py` (`RESOLVED`)
- Test: `tests/test_contest_fetch.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_contest_fetch.py`:

```python
def test_derive_source_date_counts_not_hit_rounds():
    """MLB profiles use 'not_hit' for a settled miss. derive_source_date must treat
    it as settled, else freshness is biased against reset days (contest_fetch RESOLVED bug)."""
    from datetime import date
    from bts.contest_fetch import derive_source_date
    rounds = {1: date(2026, 6, 8), 2: date(2026, 6, 9)}
    preds = [{"roundId": 1, "result": "hit"}, {"roundId": 2, "result": "not_hit"}]
    assert derive_source_date(preds, rounds) == date(2026, 6, 9)
```

- [ ] **Step 2: Run it — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_derive_source_date_counts_not_hit_rounds -v`
Expected: FAIL — returns `date(2026, 6, 8)` because `"not_hit"` is filtered out by `RESOLVED`.

- [ ] **Step 3: Fix `RESOLVED`**

In `src/bts/contest_fetch.py:10`, change:

```python
RESOLVED = {"hit", "miss", "void"}
```
to:
```python
# MLB profile settles rounds as hit / not_hit / void; "miss" kept for legacy/local safety.
RESOLVED = {"hit", "not_hit", "miss", "void"}
```

- [ ] **Step 4: Run it — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_derive_source_date_counts_not_hit_rounds -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/contest_fetch.py tests/test_contest_fetch.py
git commit -m "fix(fetch): count not_hit as a settled profile result in derive_source_date"
```

---

### Task 3: Persist the `activeStreak` snapshot even when `source_date` is None

**Files:**
- Modify: `src/bts/contest_fetch.py` (`build_observation`)
- Test: `tests/test_contest_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_observation_allows_none_source_date():
    """Snapshot (activeStreak) must persist even when ledger coverage is unknown —
    the predictions array lags the counter, so source_date can be None."""
    from datetime import datetime, timezone
    from bts.contest_fetch import build_observation
    success = {"activeStreak": 8, "seasonBestStreak": 9}
    obs = build_observation(success, None, 50311, "stonehengee",
                            datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc))
    assert obs["active_streak"] == 8
    assert obs["source_date"] is None
    assert obs["recorded_at"].endswith("Z")
```

- [ ] **Step 2: Run it — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_build_observation_allows_none_source_date -v`
Expected: FAIL — `build_observation` raises `ContestFetchError("source_date is required")`.

- [ ] **Step 3: Allow `source_date=None`**

In `src/bts/contest_fetch.py` `build_observation`, replace:

```python
    if source_date is None:
        raise ContestFetchError("source_date is required")
    validate_fetch(success)
    return {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source": "mlb_bts_profile",
        "source_date": source_date.isoformat(),
        ...
```
with:
```python
    validate_fetch(success)
    return {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source": "mlb_bts_profile",
        "source_date": source_date.isoformat() if source_date is not None else None,
        ...
```
(Keep the remaining fields — `recorded_at`, `user_id`, `username`, `saver_available` — unchanged.)

- [ ] **Step 4: Run it — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_build_observation_allows_none_source_date -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/contest_fetch.py tests/test_contest_fetch.py
git commit -m "fix(fetch): persist activeStreak snapshot even when source_date is None"
```

---

### Task 4: CLI — write the snapshot despite predictions-array lag

**Files:**
- Modify: `src/bts/cli.py` (`fetch_contest_streak` command, ~lines 1620-1660)
- Test: `tests/test_contest_fetch.py` (or `tests/test_cli_fetch_contest_streak.py` if one exists)

> **Verify against the live file first** (planning note): open `src/bts/cli.py` and locate the `fetch-contest-streak` command. The two blocks to change are (a) the `if source_date is None: _fail("profile proves no settled result date …")` guard and (b) the currentness gate that does `if latest is not None and source_date < latest: … refusing to overwrite`.

- [ ] **Step 1: Write the failing test (mocked auth + lagging profile)**

```python
def test_fetch_cli_persists_current_activestreak_despite_lag(tmp_path, monkeypatch):
    """The incident: profile activeStreak=8 is current, but the per-round predictions
    lag (latest settled row = 6/15) while a local pick is resolved through 6/16. The
    CLI must still WRITE activeStreak=8, not refuse it."""
    import json
    from datetime import date
    from click.testing import CliRunner
    from bts.cli import cli
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    import bts.cli as climod

    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))  # local resolved 6/16

    class _Sess:  # minimal AuthSession stand-in
        xsid = "x"; user_id = 50311; username = "stonehengee"
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "u"})
    monkeypatch.setattr(auth, "extract_uid", lambda c: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda uid, cookies: _Sess())
    monkeypatch.setattr(cf, "fetch_profile", lambda uid, cookies, xsid: {
        "activeStreak": 8, "seasonBestStreak": 9,
        "predictions": [{"roundId": 1, "result": "hit"}]})           # only 6/15 settled
    monkeypatch.setattr(climod, "_fetch_rounds", lambda: {1: date(2026, 6, 15)})

    res = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                   "--expected-username", "stonehengee"])

    written = json.loads((picks / "account_state" / "contest_streak.json").read_text())
    assert written["active_streak"] == 8
    assert res.exit_code == 0
```

- [ ] **Step 2: Run it — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_fetch_cli_persists_current_activestreak_despite_lag -v`
Expected: FAIL — the currentness gate exits without writing (`source_date 2026-06-15 < latest resolved pick 2026-06-16`), so `contest_streak.json` is absent.

- [ ] **Step 3: Remove the two refusal blocks**

In `fetch_contest_streak`: (a) delete the `if source_date is None: _fail("profile proves no settled result date; refusing to claim freshness")` guard — `build_observation` (Task 3) now accepts `None`; (b) delete the currentness-gate block that refuses to write when `source_date < latest` (the `latest = latest_resolved_pick_date(...)` / `resolved_pick_settlement_gap` / `_fail`/`sys.exit(0)` block). Keep the auth/identity/prior-observation guards and the atomic write. Net effect: a valid, identity-matched fetch always writes the snapshot; `contest_state`'s `status` now carries staleness downstream.

- [ ] **Step 4: Run it — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_fetch_cli_persists_current_activestreak_despite_lag -v`
Expected: PASS — `contest_streak.json` written with `active_streak == 8`.

- [ ] **Step 5: Run the fetch/cli/health suites for regressions**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py tests/ -k "contest or fetch or cli or health" -v`
Expected: PASS. Fix any test that asserted the old "refusing to overwrite" exit.

- [ ] **Step 6: Commit**

```bash
git add src/bts/cli.py tests/test_contest_fetch.py
git commit -m "fix(fetch-cli): always persist a current activeStreak; staleness now labeled downstream"
```

---

### Task 5: Persist the full per-round ledger snapshot

**Files:**
- Modify: `src/bts/cli.py` (`fetch_contest_streak`, after the snapshot write)
- Test: `tests/test_contest_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_cli_persists_per_round_ledger(tmp_path, monkeypatch):
    import json
    from datetime import date
    from click.testing import CliRunner
    from bts.cli import cli
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    import bts.cli as climod
    picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
    (picks / "2026-06-16.json").write_text(json.dumps({"result": "hit"}))
    class _Sess:
        xsid = "x"; user_id = 50311; username = "stonehengee"
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "u"})
    monkeypatch.setattr(auth, "extract_uid", lambda c: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda uid, cookies: _Sess())
    preds = [{"roundId": 1, "result": "hit", "streak": 8, "streakIncrease": 1,
              "roundPredictions": [{"playerId": 1, "result": "hit", "hits": 2, "atBats": 4}]}]
    monkeypatch.setattr(cf, "fetch_profile", lambda uid, cookies, xsid: {
        "activeStreak": 8, "seasonBestStreak": 9, "predictions": preds})
    monkeypatch.setattr(climod, "_fetch_rounds", lambda: {1: date(2026, 6, 16)})
    CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                             "--expected-username", "stonehengee"])
    ledger = (picks / "account_state" / "contest_ledger.jsonl")
    assert ledger.exists()
    row = json.loads(ledger.read_text().strip().splitlines()[-1])
    assert row["active_streak"] == 8 and len(row["predictions"]) == 1
    assert row["recorded_at"].endswith("Z")
```

- [ ] **Step 2: Run it — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_fetch_cli_persists_per_round_ledger -v`
Expected: FAIL — `contest_ledger.jsonl` does not exist.

- [ ] **Step 3: Append a ledger row after the snapshot write**

In `fetch_contest_streak`, immediately after the atomic write of `contest_streak.json` (and not in `--dry-run`), append a JSONL row. Use the already-fetched `predictions`, `success`, and `datetime.now(timezone.utc)`:

```python
    ledger_path = picks / "account_state" / "contest_ledger.jsonl"
    ledger_row = {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source_date": source_date.isoformat() if source_date is not None else None,
        "predictions": predictions,
    }
    with ledger_path.open("a") as fh:
        fh.write(json.dumps(ledger_row) + "\n")
```

- [ ] **Step 4: Run it — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py::test_fetch_cli_persists_per_round_ledger -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/cli.py tests/test_contest_fetch.py
git commit -m "feat(fetch-cli): persist full per-round MLB ledger snapshots (contest_ledger.jsonl)"
```

---

### Task 6: Dashboard — label the real streak with confidence

**Files:**
- Modify: `src/bts/web.py` (`_streak_subtitle`, `web.py:132-144`)
- Test: `tests/test_web_streak_subtitle.py` (new)

- [ ] **Step 1: Write the failing test (new file)**

```python
# tests/test_web_streak_subtitle.py
from datetime import date
from bts.web import _streak_subtitle
from bts.contest_state import DecisionStreakState


def _state(status, streak=8, model=10, src_date=date(2026, 6, 16)):
    return DecisionStreakState(
        streak=streak, saver_available=False, allow_double=True, source="contest",
        status=status, model_streak=model, model_saver_available=True,
        contest_streak=streak, contest_saver_available=None, contest_source_date=src_date)


def test_lagged_subtitle_says_last_confirmed_through_date():
    assert "Last confirmed" in _streak_subtitle(_state("lagged"))
    assert "2026-06-16" in _streak_subtitle(_state("lagged"))


def test_stale_subtitle_warns_may_be_lower():
    sub = _streak_subtitle(_state("stale"))
    assert "Last confirmed" in sub and "may be lower" in sub.lower()


def test_fresh_subtitle_is_contest_state():
    assert _streak_subtitle(_state("fresh")) == "Contest State"
```

- [ ] **Step 2: Run it — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_streak_subtitle.py -v`
Expected: FAIL — current `_streak_subtitle` returns "Contest State · Replay 10" for these (contest != model), never "Last confirmed …".

- [ ] **Step 3: Rewrite `_streak_subtitle`**

Replace `web.py:132-144` with:

```python
def _streak_subtitle(decision_state, error_message: str | None = None) -> str:
    if error_message:
        return "Streak State Error"
    if decision_state is None:
        return "Consecutive Hits"
    status = getattr(decision_state, "status", None)
    asof = decision_state.contest_source_date
    if status == "lagged" and asof is not None:
        return f"Last confirmed through {asof}"
    if status == "stale":
        tail = f" through {asof}" if asof is not None else ""
        return f"Last confirmed{tail} · current may be lower"
    if decision_state.source == "contest":
        return "Contest State"
    return "Replay State"
```

- [ ] **Step 4: Run it — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_streak_subtitle.py -v`
Expected: PASS.

- [ ] **Step 5: Add the model what-if as a separate dashboard line (optional, low-risk)**

In the streak-box HTML (`web.py:1302-1305`), add a research line under the subtitle when the model differs from the shown streak. After the `streak-sub` div, insert:

```python
                <div class="streak-sub" style="color:#8899bb">{(f"Replay {decision_state.model_streak}" if decision_state is not None and decision_state.model_streak != streak else "")}</div>
```

Manually verify by running the dashboard (see Task 7) and confirming it renders "8 / Last confirmed through 2026-06-16 / Replay 10".

- [ ] **Step 6: Commit**

```bash
git add src/bts/web.py tests/test_web_streak_subtitle.py
git commit -m "feat(dashboard): label real streak with confidence (last-confirmed/stale); show model replay separately"
```

---

### Task 7: Full suite, manual check, deploy

- [ ] **Step 1: Run the non-model suites green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -k "contest or fetch or cli or health or web or decision or streak" -v`
Expected: PASS. (Model/predict tests need `--extra model`; skip if libomp isn't installed locally — they don't exercise this change.)

- [ ] **Step 2: Manual sanity against the box's real state (read-only)**

Run: `ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python -c "from pathlib import Path; from bts.contest_state import load_decision_streak_state; import json; print(json.dumps(load_decision_streak_state(Path(\"data/picks\")).__dict__, default=str))"'`
Expected (with the override still in place): `streak: 8`, `allow_double: true`, `status` in {fresh, lagged}. Confirms the new code path against live data before shipping.

- [ ] **Step 3: Ship**

```bash
git push origin real-streak-anchoring          # PR for review
# after merge to main:
git push origin main:deploy                    # canary + auto-rollback guard it
```

- [ ] **Step 4: Post-deploy verification**

After the canary passes, re-run Step 2 against the box and confirm the dashboard (`http://bts-hetzner:3003`) shows the labeled streak. Then remove the 6/17 override + guard (they were the operational bridge): `ssh bts-hetzner 'pkill -f override_guard_2026 || true'` and let the override expire (or clear it once auto-fetch is confirmed writing the correct value).

---

## Self-Review

**Spec coverage:**
- P1.1 (kill max, model never raises, status fresh/lagged/stale) → Task 1.
- P1.2 (snapshot/coverage split, CLI gate, not_hit, ledger persist) → Tasks 2, 3, 4, 5.
- P1.3 (remove blanket DD freeze) → Task 1 (`allow_double=True`).
- P1.4 (override semantics) → Task 1 Step 7 (`test_unexpired_override_drives_decision_with_status`).
- P1.5 (dashboard labeling) → Task 6.
- P1.6 (preserve data) → model `streak.json` untouched; ledger persisted (Task 5); both series retained.

**Phase-1 limitation honored:** stale-high is *surfaced* (status `stale`, dashboard "may be lower") but strategy still uses the contest number — no plausible-set, no strategy refactor (Phase 2).

**Type consistency:** `DecisionStreakState` fields used in tests/dashboard (`streak`, `model_streak`, `status`, `contest_source_date`, `allow_double`, `source`) match the dataclass at `contest_state.py:44-58`. `status` values used: `fresh` / `lagged` / `stale` / `model_only`.

**Verify-against-live note:** Task 4/5 edit `cli.py` and Task 6 edits `web.py` — only `contest_state` / `contest_fetch` were read in full during planning; confirm the exact `fetch_contest_streak` block boundaries and the streak-box HTML before editing.
