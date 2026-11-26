# Contest-streak automation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. This build is **collaborative with Codex** (Claude implements TDD; Codex adversarially reviews each slice via the agent-room). Steps use `- [ ]` tracking.

**Goal:** Auto-fetch Eric's real MLB BTS account streak and keep `contest_streak.json` accurate, with a stale-state alert and expiring manual override so it can never silently freeze again.

**Architecture:** New narrow `bts fetch-contest-streak` command reuses existing leaderboard auth + the user-profile endpoint; writes the auto observation atomically only after identity + currentness + sanity checks; precedence makes auto the default with `set-contest-streak` as an expiring emergency override; health gains a stale-state CRITICAL alert.

**Tech Stack:** Python (existing BTS pkg), `httpx`, `click`, `pytest`. Tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-06-06-contest-streak-automation-design.md` (read for schemas, precedence, currentness semantics — not duplicated here). Codex proposal: `agent-room/docs/codex-contest-streak-design.md`.

---

## Task 0: Test environment

The loaner has no `uv`. Pick one and confirm `pytest` runs before TDD:
- [ ] **Option A (local):** `curl -LsSf https://astral.sh/uv/install.sh | sh`; `cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv sync`; smoke `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_state.py -q`.
- [ ] **Option B (hetzner):** develop locally, run tests over SSH against a synced worktree (`rsync` branch to a scratch dir on bts-hetzner, run `uv run pytest` there). Use when local `uv sync` pulls too-heavy ML deps.

Decision recorded at execution (default: A; fall back to B if `uv sync` is slow/broken).

## File structure
- Modify `src/bts/leaderboard/auth.py` — add `fetch_login_session()` (xSid + user); `fetch_xsid()` delegates.
- Create `src/bts/contest_fetch.py` — narrow fetch+parse+validate → auto-observation dict (HTTP behind an injectable client seam).
- Modify `src/bts/contest_state.py` — expiring-override-aware selection; auto-default.
- Modify `src/bts/cli.py` — add `fetch-contest-streak`; convert `set-contest-streak` to expiring override (schema v2 + `--ttl-hours`).
- Modify `src/bts/health/contest_state.py` — stale + legacy/expired-manual alerts.
- Tests: `tests/test_contest_fetch.py`, extend `tests/test_contest_state.py`, `tests/health/test_contest_state.py`, `tests/test_cli_integration.py`.
- Docs/deploy: cron entries + deploy steps (Task 8).

---

## Task 1: `fetch_login_session` (auth returns user identity)

**Files:** Modify `src/bts/leaderboard/auth.py`; Test: `tests/leaderboard/test_auth_session.py`.

- [ ] **Step 1 — failing test** (inject a fake httpx via monkeypatch on `auth.httpx.post`):
```python
def test_fetch_login_session_returns_xsid_and_user(monkeypatch):
    import bts.leaderboard.auth as a
    class R:
        status_code = 200
        def json(self): return {"success": {"xSid": "abc_123", "user": {"id": 50311, "username": "stonehengee"}}}
    monkeypatch.setattr(a.httpx, "post", lambda *args, **kw: R())
    s = a.fetch_login_session(uid="okta-uid", cookies={"oktaid": "okta-uid"})
    assert s.xsid == "abc_123" and s.user_id == 50311 and s.username == "stonehengee"

def test_fetch_xsid_still_works(monkeypatch):
    import bts.leaderboard.auth as a
    class R:
        status_code = 200
        def json(self): return {"success": {"xSid": "z_9", "user": {"id": 1, "username": "x"}}}
    monkeypatch.setattr(a.httpx, "post", lambda *args, **kw: R())
    assert a.fetch_xsid("uid", {"oktaid": "uid"}) == "z_9"
```
- [ ] **Step 2 — run, expect fail** (`fetch_login_session`/`AuthSession` undefined).
- [ ] **Step 3 — implement:** add `@dataclass(frozen=True) class AuthSession: xsid:str; user_id:int|None; username:str|None`. `fetch_login_session(uid, cookies, timeout=30.0)` = current `fetch_xsid` POST logic but returns `AuthSession(xsid, success["user"]["id"], success["user"]["username"])` (user optional/None-safe). Refactor `fetch_xsid` to `return fetch_login_session(uid, cookies, timeout).xsid`. Raise `AuthError` exactly as before on non-200/missing xSid.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(auth): fetch_login_session returns xSid + account identity`.

## Task 2: profile fetch + parse + source_date (contest_fetch core)

**Files:** Create `src/bts/contest_fetch.py`; Test: `tests/test_contest_fetch.py`.

Interface:
- `fetch_profile(user_id, cookies, xsid, *, client=httpx) -> dict` — GET `USER_PROFILE_URL_TEMPLATE`, return `success`.
- `derive_source_date(predictions, rounds: dict[int,date]) -> date|None` — max round-date among predictions whose `result` ∈ {hit,miss,void}; None if none settled.
- `RESOLVED = {"hit","miss","void"}`.

- [ ] **Step 1 — failing test** (pure parser, no network):
```python
import datetime as dt
from bts.contest_fetch import derive_source_date
def test_derive_source_date_latest_settled():
    rounds = {10: dt.date(2026,6,4), 11: dt.date(2026,6,5), 12: dt.date(2026,6,6)}
    preds = [{"roundId":10,"result":"hit"}, {"roundId":11,"result":"miss"}, {"roundId":12,"result":None}]
    assert derive_source_date(preds, rounds) == dt.date(2026,6,5)   # 12 is unsettled
def test_derive_source_date_none_when_no_settled():
    assert derive_source_date([{"roundId":1,"result":None}], {1: dt.date(2026,6,6)}) is None
```
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `derive_source_date` + `fetch_profile` (mirror `leaderboard.scraper._get_json`/`scrape_user_profile` GET with cookies + `USER_AGENT`).
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(contest_fetch): profile fetch + source_date derivation`.

## Task 3: build + validate auto observation (currentness + sanity gates)

**Files:** Modify `src/bts/contest_fetch.py`; Test: extend `tests/test_contest_fetch.py`.

Interface: `build_observation(success, source_date, user_id, username, recorded_at) -> dict` (auto schema, per spec) and `validate_fetch(success) -> None` raising `ContestFetchError` on: non-int/negative streaks, `seasonBestStreak < activeStreak`. Currentness is enforced by the CLI (Task 5) comparing `source_date` to `latest_resolved_pick_date`; `build_observation` requires a non-None `source_date`.

- [ ] **Step 1 — failing tests:** `validate_fetch` rejects `{"activeStreak":-1,...}`, `{"activeStreak":5,"seasonBestStreak":3}`, non-int; accepts `{"activeStreak":0,"seasonBestStreak":9}`. `build_observation` produces `schema_version="bts_contest_streak_auto_v1"`, `saver_available=None`, `source="mlb_bts_profile"`, correct fields.
- [ ] **Step 2 — run fail.** **Step 3 — implement.** **Step 4 — run pass.**
- [ ] **Step 5 — commit** `feat(contest_fetch): observation builder + sanity validation`.

## Task 4: precedence (expiring override + auto default)

**Files:** Modify `src/bts/contest_state.py`; Test: extend `tests/test_contest_state.py`.

Behavior (per spec §Precedence): `load_contest_streak_state` selects an **unexpired** `contest_streak.manual.json` (has `override_expires_at` > now) first; else `contest_streak.json`; an expired/legacy manual is skipped when auto exists. Add `now` param (inject for tests). Surface a `selected_path` + a `legacy_or_expired_manual: bool` signal for the health check.

- [ ] **Step 1 — failing tests:**
```python
# auto wins over legacy manual (no override_expires_at)
# unexpired override (override_expires_at in future) wins over auto
# expired override ignored -> auto used
```
(Use a fixed `now` and write both files into a tmp `account_state/`.)
- [ ] **Step 2 — run fail.** **Step 3 — implement** the selection change (keep `ContestStreakState`; add expiry parse). **Step 4 — run pass.**
- [ ] **Step 5 — commit** `feat(contest_state): expiring-override precedence, auto default`.

## Task 5: `bts fetch-contest-streak` CLI (atomic write, identity, fail-safe)

**Files:** Modify `src/bts/cli.py`; Test: extend `tests/test_cli_integration.py`.

Wiring: load cookies → `fetch_login_session` → **identity guard** (`success.user.username == --expected-username`; and match prior `contest_streak.json` username/user_id if present) → `fetch_profile` → `validate_fetch` → `derive_source_date` (fetch `rounds.json`) → **currentness gate** (`source_date` not None and `>= latest_resolved_pick_date(picks_dir)`) → atomic write `contest_streak.json` (temp + `os.replace`). Any failure: no overwrite, exit nonzero, `bts.dm.send_dm` (reuse leaderboard wording) throttled via `data/health_state/contest_streak_fetch_status.json` (≥6h cooldown). `--dry-run` prints would-write without writing.

- [ ] **Step 1 — failing tests** (monkeypatch auth + profile to inject payloads; tmp picks-dir):
```python
# happy path: writes contest_streak.json with active_streak from payload, atomic (no .tmp left)
# identity mismatch: wrong username -> exit !=0, no file written, dm called
# stale profile (source_date < latest local resolved pick) -> no write, alert
# auth failure -> no write, dm throttled (second call within cooldown does not re-dm)
```
- [ ] **Step 2 — run fail.** **Step 3 — implement** `fetch_contest_streak` command + a small `_atomic_write_json` helper + `_throttle` on the status file. **Step 4 — run pass.**
- [ ] **Step 5 — commit** `feat(cli): fetch-contest-streak (identity+currentness gated, atomic, throttled alert)`.

## Task 6: `set-contest-streak` → expiring override

**Files:** Modify `src/bts/cli.py`; Test: extend `tests/test_cli_integration.py`.

Change `set-contest-streak` to write manual schema **v2** with `override_expires_at` (default now+`--ttl-hours` 24; or explicit `--override-expires-at`), `reason` (`--reason`). Keep `--streak/--best-streak/--saver-available/--username`.

- [ ] **Step 1 — failing test:** `set-contest-streak --streak 0 --best-streak 9 --ttl-hours 24` writes `schema_version=bts_contest_streak_manual_v2` with `override_expires_at` ≈ now+24h; the value wins over auto until expiry (assert via `load_contest_streak_state`).
- [ ] **Step 2 — run fail.** **Step 3 — implement.** **Step 4 — run pass.**
- [ ] **Step 5 — commit** `feat(cli): set-contest-streak writes expiring override (v2)`.

## Task 7: health stale + legacy/expired-manual alerts

**Files:** Modify `src/bts/health/contest_state.py`; Test: extend `tests/health/test_contest_state.py`.

Add to `check(picks_dir, expected)`: keep missing/invalid; if `expected` and `not contest_state_is_fresh(state, picks_dir)` → CRITICAL (msg includes `state.path`, `state.source_date`, `latest_resolved_pick_date`). If a legacy/expired manual file is present → WARN; CRITICAL if it is the selected state while `expected`.

- [ ] **Step 1 — failing tests:** stale auto (source_date < latest pick) → CRITICAL; fresh → no alert; legacy manual present alongside auto → WARN.
- [ ] **Step 2 — run fail.** **Step 3 — implement.** **Step 4 — run pass.**
- [ ] **Step 5 — commit** `feat(health): CRITICAL on stale contest state + legacy/expired-manual alerts`.

## Task 8: full suite, PR, deploy

- [ ] **Step 1:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_fetch.py tests/test_contest_state.py tests/health/test_contest_state.py tests/test_cli_integration.py tests/leaderboard/test_auth_session.py -q` → all pass. Then full suite (or scoped) to confirm no regressions (esp. existing `fetch_xsid` callers + leaderboard scrape).
- [ ] **Step 2 — live smoke (read-only)** against the real account on hetzner via SSH: `uv run bts fetch-contest-streak --dry-run --expected-username stonehengee` → prints `active_streak=0 best_streak=9 source_date=<latest settled> would-write` without writing.
- [ ] **Step 3 — Codex final adversarial review** of the diff via the agent-room (per the collaboration); triage with receiving-code-review; fix real issues.
- [ ] **Step 4 — PR:** push branch, `gh pr create` (title: contest-streak automation; body credits Claude+Codex; links spec). 
- [ ] **Step 5 — deploy** to bts-hetzner (per `reference_bts_deploy_workflow.md`): pull merged main, add the 4 cron entries (spec §Schedule), run one real `fetch-contest-streak` (writes auto `contest_streak.json`), **archive the hotfix `contest_streak.manual.json`**, verify dashboard + `load_decision_streak_state` show the auto value, and confirm a forced auth failure ALERTS (not silently freezes).

---

## Self-Review
- **Spec coverage:** component shape (T1-T2,T5)✓; precedence/expiring override (T4,T6)✓; identity guard (T5)✓; currentness proof/source_date (T2,T5)✓; saver=null (T3)✓; fail-safe + throttle (T5)✓; stale/legacy health alert (T7)✓; schedule+deploy+archive-hotfix (T8)✓; sanity incl. accept-0 (T3)✓. No gaps.
- **Placeholders:** none — test contracts are concrete; impl described by interface + spec schemas (DRY) rather than re-transcribed.
- **Type consistency:** `AuthSession.{xsid,user_id,username}`, `derive_source_date(predictions,rounds)`, `validate_fetch`/`build_observation`/`ContestFetchError`, `fetch_profile`, `_atomic_write_json` used consistently. Schemas match spec (`bts_contest_streak_auto_v1`, `..._manual_v2`).
