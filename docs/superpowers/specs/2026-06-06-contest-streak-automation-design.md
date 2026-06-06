# Contest-streak automation — design spec (2026-06-06)

**Status:** approved (Eric, 2026-06-06). Claude + Codex collaboration (Codex proposal: `agent-room/docs/codex-contest-streak-design.md`).

## Goal

Keep BTS "contest state" — Eric's REAL MLB Beat the Streak account streak — automatically accurate, and **never silently freeze again**. Live picks are driven by this value via `bts.contest_state.load_decision_streak_state(..., require_contest_state=True)`.

## Background / the bug this fixes

Two streaks exist:
- **Model/replay** `data/picks/streak.json` — the bot's own pick-replay streak.
- **Contest** `data/picks/account_state/contest_streak.{manual.json,json}` — observation of the real MLB account.

The decision layer prefers contest state; when the contest observation is **stale** (`source_date < latest_resolved_pick_date`) it freezes effective streak at `max(model, contest)` and disables doubles. The contest observation was a **manual screenshot from 2026-05-29 (streak 7)** that nothing refreshed → it silently froze the dashboard at 7 for a week while the real streak was **0** (best **9**). The health check (`bts/health/contest_state.py`) only flagged missing/invalid state, never **stale** — so nothing alerted. Hotfixed 2026-06-06 via `bts set-contest-streak --streak 0 --best-streak 9`.

The fetch capability already exists and works (verified live 2026-06-06): leaderboard auth (cookie file on hetzner, valid) → `oktaid` uid → POST auth/login → `xSid`; `USER_PROFILE_URL_TEMPLATE` returns `success.activeStreak` + `success.seasonBestStreak`. Eric: `user_id=50311`, `username=stonehengee`. Live values: activeStreak=0, seasonBestStreak=9. The profile has **no** saver/mulligan field.

## Decisions (Eric-approved)
- **Full robust** design (currentness-proof + stale alert + expiring manual override).
- **Cadence:** after `check-results` and `reconcile`, and before the pick windows (~3-4×/day ET).

## Architecture / components

A small, well-bounded set of changes; reuse existing leaderboard auth/profile rather than new HTTP code.

1. **`bts.leaderboard.auth`** — add `fetch_login_session(cookies) -> AuthSession` returning `{xsid, user_id, username}` (parsed from auth/login `success.{xSid,user}`). Keep existing `fetch_xsid()` (call the new helper, return only xsid) for current callers — no behavior change for the leaderboard scraper.
2. **`bts.contest_fetch`** (new module) — narrow logic: load cookies → `fetch_login_session` → identity guard → fetch profile for `user_id` → parse `activeStreak`/`seasonBestStreak`/predictions → derive `source_date` (currentness proof) → sanity checks → build the auto observation dict. No file I/O or HTTP orchestration beyond the single profile call + a `rounds.json` fetch for date mapping. Pure-ish + unit-testable (HTTP behind a thin client seam so tests can inject payloads).
3. **`bts.cli`** — add `fetch-contest-streak` command (the only writer of `contest_streak.json`, atomic write). Update `set-contest-streak` to write an **expiring override** (manual schema v2 with `override_expires_at`).
4. **`bts.contest_state`** — change selection logic: expiring-override-aware precedence (below). Add `source_date` derivation contract for auto observations.
5. **`bts.health.contest_state`** — add **stale** (CRITICAL when expected + not fresh) and **legacy/expired-manual** alerts (the missing alarm).
6. **cron** (hetzner) — standalone `bts fetch-contest-streak` jobs (not in the scheduler process; not folded into `leaderboard scrape`).

## Data flow (`fetch-contest-streak`)

1. `load_session_cookies()` (existing).
2. `fetch_login_session()` → `xsid`, `user_id`, `username`.
3. **Identity guard:** require `username == --expected-username` (prod: `stonehengee`); if a prior `contest_streak.json` exists, also require its `username`/`user_id` match. On mismatch → fail (no write), DM-alert.
4. Fetch profile for `user_id` (`USER_PROFILE_URL_TEMPLATE`) → `success`.
5. Parse `activeStreak`, `seasonBestStreak`, `predictions`.
6. **Currentness proof — derive `source_date`:** the latest *settled* result date the profile proves. Map `predictions[].roundId` → date via `rounds.json` (`ROUNDS_URL`, no auth); `source_date` = max date among predictions whose `result` is settled (`hit`/`miss`/`void`). If none settled / cannot derive → `source_date = None` ⇒ **cannot claim fresh** (see step 8). `recorded_at` = UTC fetch time.
7. **Sanity checks:** `activeStreak`, `seasonBestStreak` are ints ≥ 0; `seasonBestStreak >= activeStreak`. If a prior observation exists, log/alert on large jumps but do NOT reject a valid reset backed by current evidence. (Do **not** over-guard `activeStreak=0` — a reset is the most important value to accept quickly.)
8. **Currentness gate:** if `source_date is None` or `source_date < latest_resolved_pick_date(picks_dir)` → the profile does not prove currentness → **do not overwrite**; emit WARN/CRITICAL status + DM (throttled). (This is the key correction: a 200 response is not proof of currency.)
9. **Atomic write** `data/picks/account_state/contest_streak.json` (temp file + `os.replace`) with the auto schema.

## Schemas

**Auto observation** (`contest_streak.json`):
```json
{"schema_version":"bts_contest_streak_auto_v1","active_streak":0,"best_streak":9,
 "source":"mlb_bts_profile","source_date":"2026-06-06","recorded_at":"2026-06-06T18:00:00Z",
 "user_id":50311,"username":"stonehengee","saver_available":null}
```

**Expiring manual override** (`contest_streak.manual.json`, schema v2):
```json
{"schema_version":"bts_contest_streak_manual_v2","active_streak":0,"best_streak":9,
 "source":"manual_cli","source_date":"2026-06-06","recorded_at":"2026-06-06T18:00:00Z",
 "override_expires_at":"2026-06-07T18:00:00Z","reason":"API auth unavailable","saver_available":null}
```

## Precedence / selection (`bts.contest_state.load_contest_streak_state`)

1. If `contest_streak.manual.json` exists AND is an **unexpired** override (`override_expires_at > now`) → use it.
2. Else use `contest_streak.json` (auto).
3. A manual file that is **expired** or **legacy** (no `override_expires_at`) is **ignored when an auto file exists**, and triggers a health alert (so it gets archived/removed). If it's the *only* state and contest state is expected → CRITICAL.

Rejected alternatives: "freshest wins" (a manual typo isn't more authoritative than the MLB API); "auto refuses if manual present" (a forgotten manual file would disable the fix).

`saver_available`: auto always `null` (decision layer treats `None` as false = conservative). Saver is only set inside an (expiring) manual override. No speculative endpoint probing.

## Failure handling + alert throttle

All failure modes (auth/cookie expiry, HTTP/network after brief backoff, JSON/shape, identity mismatch, not-current) → **never overwrite** the prior good state; exit nonzero; DM via existing `bts.dm.send_dm` using the leaderboard auth-failure wording. Throttle with `data/health_state/contest_streak_fetch_status.json` (last-alert timestamp + state) so hourly/3-4×-daily cron doesn't spam — alert on transition + at most once per cooldown (e.g. 6h).

## Health check additions (`bts.health.contest_state.check`)
- Keep existing missing/invalid checks.
- If contest state expected and `contest_state_is_fresh(state, picks_dir)` is false → **CRITICAL** (include `state.path`, `state.source_date`, `latest_resolved_pick_date`).
- Legacy/expired manual file present → at least WARN; CRITICAL if it is the selected state while contest state is expected.

## Schedule (hetzner cron, ET)
Standalone jobs (env-sourced like existing BTS cron), e.g.:
- After `check-results` (currently 01:00) and `reconcile` (02:00): run at ~01:10 and ~02:10.
- Before pick windows: ~10:30 and ~13:30 (ahead of typical lock windows).
Each: `cd /home/bts/projects/bts && set -a && . ./.env && set +a && uv run bts fetch-contest-streak --picks-dir data/picks --expected-username stonehengee --dm-recipient stonehengee.bsky.social >> /home/bts/logs/cron.log 2>&1`. (`stonehengee.bsky.social` = the Bluesky DM recipient from `~/.bts-orchestrator.toml [bluesky] dm_recipient`.)

## Testing (stdlib/pytest, BTS convention `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`)
- `fetch_login_session` parses xSid + user from a sample auth/login payload.
- Identity mismatch (wrong username) → no write, nonzero, alert.
- Valid profile → writes auto schema; `source_date` = latest settled prediction date.
- Stale profile evidence (latest settled < latest local pick) → no write, alert (the anti-silent-freeze guarantee).
- Auth / HTTP / shape failures → prior state untouched.
- Sanity: non-int / negative / best<active → reject; `activeStreak=0` with current evidence → accepted.
- Precedence: auto wins over legacy manual; unexpired override wins; expired override ignored + alerted.
- Health: CRITICAL when expected contest state is stale; WARN/CRITICAL on legacy/expired manual.
- Atomic write: interrupted write never yields a half-file as live state.

## Deploy
PR (Claude+Codex) → review → merge → bts-hetzner `git pull` (per `reference_bts_deploy_workflow.md`) + add cron entries. On first successful auto-fetch, **archive the current hotfix `contest_streak.manual.json`** (it's legacy v1 without expiry) so the active source is unambiguous. Verify dashboard + `load_decision_streak_state` reflect the auto value; confirm a forced auth failure alerts (not silently freezes).

## Deferred / out of scope
- Automated cookie refresh (still interactive via `scripts/capture_bts_cookies.py` on Mac; cookie expiry → alert + manual re-capture). Note: the loaner's macOS keychain was wiped, so re-capture tooling/path may need revisiting — tracked separately.
- A real saver/mulligan availability source (no profile field today).
- Deploy-gap: hetzner is at #138 vs main #141 — separate concern; this feature deploys on top of current main.
