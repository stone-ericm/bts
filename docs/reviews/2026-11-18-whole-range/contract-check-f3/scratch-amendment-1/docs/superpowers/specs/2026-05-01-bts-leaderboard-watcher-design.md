# BTS Leaderboard Watcher — Design Spec

**Date**: 2026-05-01
**Status**: Approved (brainstorm) → pending implementation plan
**Author**: Eric Stone (with Claude assist)
**Source brainstorm**: 2026-05-01 morning session

## 1. Motivation

The MLB.com Beat the Streak app exposes a **public leaderboard** with full per-user picks logs. Click any leaderboard entry → drill down to that user's complete streak history (date, batter picked, game result, streak count after). This is publicly visible for every active leaderboard participant.

Three new capabilities become possible by capturing this data daily:

1. **Wisdom-of-crowds consensus pick**: aggregate top-N performers' picks each day. If 7 of top 10 picked the same batter, that's a strong prior independent of our model.
2. **Strategy mining / adversarial pick classifier**: train a model on "would a top-N leaderboard player pick this batter today?" using historical leaderboard data. Stack as a feature in our blend.
3. **Calibration sanity at long tail**: empirical distribution of streak lengths across all leaderboard users. We claim P(57)=8.17%; the leaderboard's empirical tail tells us if that's order-of-magnitude right.

These complement (do not replace) our existing PA-level model. The scraper is fully decoupled from the production picks pipeline — its failure cannot break daily picks.

## 2. Goals & non-goals

**Goals**:
- Capture full visible leaderboard state + per-user picks logs twice daily, append-only
- Store in queryable parquet format under `data/leaderboard/`
- Surface findings (consensus pick, percentile rank) in the existing dashboard at port 3003
- Conservative ToS posture: behave like a power user, not a scraper
- Health check integration so silent failures get noticed

**Non-goals**:
- v1 does not feed scraped data into the production model
- v1 does not attempt to predict other users' picks before they lock
- v1 does not scrape any third-party analyst sites (deferred to v2 if useful)
- v1 does not implement the adversarial pick classifier (deferred to v1.5)

## 3. Architecture overview

```
┌────────────────────────────────────────────────────────────────────┐
│  bts-hetzner (production)                                           │
│                                                                      │
│  ┌─────────────────────────────────────────────────┐                │
│  │  bts-leaderboard.timer (systemd, 2× daily)      │                │
│  │  10:00 ET (morning intentions) + 01:00 ET (post-game) │           │
│  └────────────┬────────────────────────────────────┘                │
│               │                                                      │
│               ▼                                                      │
│  ┌─────────────────────────────────────────────────┐                │
│  │  src/bts/leaderboard/scraper.py                 │                │
│  │  ┌──────────────────────────────────────────┐   │                │
│  │  │  1. Load session cookies from `pass`     │   │                │
│  │  │  2. GET /leaderboard endpoints (4 tabs)  │   │                │
│  │  │  3. For each top-N user, GET picks log   │   │                │
│  │  │  4. Validate response shape              │   │                │
│  │  │  5. Append to data/leaderboard/*.parquet │   │                │
│  │  │  6. On 401/403: alert + bail (no auto)   │   │                │
│  │  └──────────────────────────────────────────┘   │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
│  ┌─────────────────────────────────────────────────┐                │
│  │  data/leaderboard/                              │                │
│  │    ├── leaderboard_snapshots/{YYYY-MM-DD}.parquet│                │
│  │    ├── user_picks/{username}.parquet (append)   │                │
│  │    └── season_stats/{YYYY-MM-DD}.parquet        │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Dashboard adapter (port 3003) — v1.5            │                │
│  │  Surfaces: today's consensus pick + diff vs ours,│                │
│  │           our percentile rank, streak histogram  │                │
│  └─────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────┘
```

**Architectural commitments**:
- **Decoupled from picks pipeline**: scraper failure cannot break production picks
- **Direct API first** (per session decision): reverse-engineer JSON endpoints; Playwright is contingency only if auth proves complex enough to require it
- **Append-only per-user picks log**: each user gets their own parquet so historical picks are preserved even after their streak ends and (presumably) MLB clears the visible log
- **No model dependency in v1**: analysis (consensus, classifier, calibration sanity) reads from the data store as separate phases
- **`pass` for session cookies**: matches existing keychain pattern (kaggle, cloudflare, etc.)

## 4. Data model

### `data/leaderboard/leaderboard_snapshots/{YYYY-MM-DD}.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `captured_at` | `timestamp[ms]` | Exact scrape time; multiple snapshots/day distinguishable |
| `tab` | `string` | `'active_streak' \| 'all_season' \| 'all_time' \| 'yesterday'` |
| `rank` | `int32` | 1-indexed |
| `username` | `string` | Public BTS handle |
| `streak` | `int32` | Null on `'yesterday'` tab |
| `hits_today` | `int32` | Only on `'yesterday'` tab; null elsewhere |

### `data/leaderboard/user_picks/{username}.parquet` (append-only)

| Column | Type | Notes |
|--------|------|-------|
| `captured_at` | `timestamp[ms]` | When we observed this pick in the user's log |
| `pick_date` | `date` | The actual game date |
| `batter_name` | `string` | As shown |
| `batter_team` | `string` | Parsed from team logo |
| `opponent_team` | `string` | Parsed from `'vs WSH'` / `'@ AZ'` |
| `home_or_away` | `string` | `'home' \| 'away'` |
| `at_bats` | `int32` | Game line: AB |
| `hits` | `int32` | Game line: H |
| `streak_after` | `int32` | Streak count showing on this pick row |
| `batter_id` | `int64?` | MLB person_id resolved via roster join (best-effort) |

### `data/leaderboard/season_stats/{YYYY-MM-DD}.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `captured_at` | `timestamp[ms]` | Scrape time |
| `username` | `string` | |
| `best_streak` | `int32` | All-time-this-season best |
| `active_streak` | `int32` | Current run |
| `pick_accuracy_pct` | `float64` | E.g., `100.0` for tombrady12 |

**Dedup logic**: `user_picks` is append-only by `captured_at`. Querying "what did tombrady12 pick on 4/30" =
```sql
SELECT * FROM user_picks/tombrady12
WHERE pick_date = '2026-04-30'
ORDER BY captured_at DESC LIMIT 1
```
This preserves the audit trail if MLB ever revises a result post-hoc (which we'd otherwise miss).

**Storage volume estimate**: top 100 users × ~35 picks visible × ~100 bytes = ~350 KB per snapshot. Daily growth: 100 users × 1 new pick = ~10 KB/day. Whole season: ~3.65 MB. Trivial.

## 5. Components

```
src/bts/leaderboard/
  ├── __init__.py
  ├── auth.py           # session cookie load/validate; Playwright contingency
  ├── endpoints.py      # discovered API URL templates (filled by Phase 1)
  ├── models.py         # pydantic schemas: LeaderboardRow, PickRow, SeasonStats
  ├── scraper.py        # core: scrape_leaderboard, scrape_user_picks, scrape_user_stats
  ├── storage.py        # parquet I/O with append + dedup helpers
  ├── cli.py            # `bts leaderboard scrape | backfill | status`
  └── analysis.py       # v1.5: consensus pick, percentile rank (stub in v1)
```

| Component | Responsibility | v1 status |
|-----------|----------------|-----------|
| `auth.py` | Load cookies from `pass`, detect 401/403, alert on auth failure | Required |
| `endpoints.py` | Holds discovered URL templates after reverse-engineering | Required |
| `models.py` | Pydantic-typed row schemas | Required |
| `scraper.py` | Orchestrates daily scrape: 4 leaderboards + N user pages | Required |
| `storage.py` | Parquet append + dedup, schema validation on write | Required |
| `cli.py` | Manual ops + status | Required |
| `analysis.py` | Daily consensus pick, percentile rank | Stubbed v1, fleshed v1.5 |

**Operational entrypoint**: `bts leaderboard scrape` (CLI). The systemd timer just calls this twice daily.

## 6. Auth flow

```
Daily run start
   │
   ▼
Load cookies from pass: `pass show mlb-bts-session-cookies`
   │
   ▼
Probe: GET leaderboard endpoint with cookies
   │
   ├── 200 OK ──► proceed with scrape
   │
   └── 401 / 403 / redirect-to-login
              │
              ▼
       Log "auth invalid"
              │
              ▼
       Send Bluesky DM to Eric: "BTS leaderboard scrape needs cookie refresh"
              │
              ▼
       Bail (no auto-Playwright on bts-hetzner — interactive login required)
```

**Cookie refresh pattern** (manual, infrequent — expected weekly to monthly):

```bash
# On Mac (interactive)
uv run python -m bts.leaderboard.auth refresh
# → Opens Playwright Chromium
# → Eric logs in to MLB.com BTS
# → Script harvests cookies, writes:
pass insert -m mlb-bts-session-cookies
# → sync to bts-hetzner pass store via existing pass sync mechanism
```

**Why no auto-refresh on the server**: Playwright on bts-hetzner would require credential storage on the server + programmatic CAPTCHA/MFA handling. Manual refresh once a month from Mac is the simpler, safer trade.

## 7. Scheduling

```ini
# ~/.config/systemd/user/bts-leaderboard.timer
[Timer]
OnCalendar=*-*-* 14:00:00 UTC   # 10:00 ET — morning, captures pick intentions before most lock-times
OnCalendar=*-*-* 05:00:00 UTC   # 01:00 ET — late-night, post-game-resolution
RandomizedDelaySec=300           # ±5 min jitter (avoid bot-pattern detection)
Persistent=true
```

**Scrape budget per run**:
- 4 leaderboard tabs × 1 GET × 2s rate limit = 8 s
- Top 100 users × 1 picks-log GET × 2s = 200 s
- Total: ~3.5 minutes wall-clock

**Why two times of day**: morning captures *intentions* (what people picked before games started); late-night captures *outcomes* (what actually happened). Diffs reveal pick changes; even if we only ever look at the late-night snapshot, having morning for free lets us reconstruct intra-day behavior post-hoc.

## 8. ToS posture

**Pre-flight checks** (one-time, must be done before first scrape):

1. **`https://www.mlb.com/robots.txt`** — read for disallowed paths under `/apps/beat-the-streak/*`
2. **MLB.com Terms of Use** — scan for explicit "no automated access" clauses
3. **MLB Privacy Policy** — confirm leaderboard usernames are intentionally public (they are surfaced in the public UX, but check anyway)

**Conservative posture if anything is ambiguous**:
- User-Agent identifying us: `bts-leaderboard-watcher/1.0 (research; contact: stone.ericm@gmail.com)`
- Honor `crawl-delay` from robots.txt; default 2s if unset
- Hard daily cap: 2 scrapes/day × ~250 requests = 500 requests/day. Power-user volume, not bot volume.
- If MLB sends 429/Retry-After: comply, back off, alert

**Fallback paths if explicitly disallowed**:
- Pivot to W2 (third-party analyst pick scraping: 57hits.com, baseballmusings.com, thebreakdownpoint.com)
- Pivot to W1-only (just leaderboard, no per-user drilldowns) — lower request count

## 9. Observability

Add `src/bts/health/leaderboard_freshness.py` — a tier-2 health check that fires:
- **WARN** if last successful scrape >12 h ago
- **CRITICAL** if last successful scrape >36 h ago (auth probably expired)
- **INFO** if scrape duration > 10 min (rate limit too aggressive, or MLB slow)

`bts leaderboard status` CLI shows:
```
Last successful scrape: 2026-05-01 05:00:42 UTC (4h 3m ago)
Latest user_picks captured: 2026-04-30 (top 100)
Auth status: valid (cookies expire in ~21 days)
Storage size: 4.2 MB
Errors in last 7d: 0
```

Integrates with existing `bts.health.runner.run_all_checks()` pipeline — appears in same health summary as `realized_calibration` / `pitcher_sparsity` / etc.

## 10. Phasing

**Phase 1 — Discovery (Mac local)**
- Use Chrome DevTools (or `superpowers-chrome` MCP) to identify JSON endpoints behind the BTS app
- Document URL templates + response shapes in `endpoints.py`
- Capture cookies via Playwright; verify single GET returns 200 with expected payload
- Manual validation: fetch tombrady12's picks log, parse, hand-compare to screenshots
- **Done when**: a Mac script pulls tombrady12's full picks log accurately

**Phase 2 — Storage (Mac local)**
- Implement `models.py` (pydantic) + `storage.py` (parquet writers with append + dedup)
- Wire parsed Phase-1 data through to actual parquet files
- Round-trip test: write fixture → read back → equal
- **Done when**: a single end-to-end scrape (1 user) produces a valid parquet under `data/leaderboard/`

**Phase 3 — Scraper integration (Mac local)**
- `scraper.py` orchestrates 4 leaderboard tabs + top-N user iteration
- Rate-limit decorator (2s default, configurable)
- Error handling per Section 3 (401/403/429/network)
- `cli.py` exposes `scrape | backfill | status` subcommands
- **Done when**: `bts leaderboard scrape` on Mac produces valid parquets for top 100 of all 4 tabs without errors

**Phase 4 — Production deploy (bts-hetzner)**
- Push to `deploy` branch (calibration-commit pattern)
- systemd `.timer` + `.service` units installed via `bash scripts/cron-setup-hetzner.sh` (extend existing)
- `leaderboard_freshness.py` added to `bts.health.runner`
- Bluesky DM hook for auth failures
- First production run; verify parquets land + health check green
- **Done when**: 24 h passes with 2 successful scrapes recorded and no false alerts

**Phase 5 — Backfill + analysis (parallel, ongoing)**
- Backfill: walk full visible picks log for every user in any current top-100
- `analysis.py` v1.5: daily consensus pick computation, percentile rank for our streak
- Dashboard adapter at port 3003 surfaces consensus pick + diff vs our pick
- **Done when**: dashboard shows today's consensus pick + diff; n≥7 days data accumulated

## 11. Testing strategy

```
tests/leaderboard/
  ├── test_parsers.py        # JSON → typed model (fixture-based)
  ├── test_storage.py        # parquet round-trip + dedup
  ├── test_scraper.py        # mocked HTTP, full orchestration
  ├── test_auth.py           # 401/403/200 paths
  ├── test_cli.py            # subcommand behavior
  ├── test_health_check.py   # leaderboard_freshness alert thresholds
  └── fixtures/
      ├── leaderboard_active_streak.json    # captured day 1, frozen
      ├── user_picks_tombrady12.json
      └── season_stats_tombrady12.json
```

| Test layer | Covers | Speed |
|------------|--------|-------|
| **Unit (parsers + storage)** | Deterministic transforms; trivially 100% coverage | Fast — ms |
| **Integration (mocked HTTP)** | Full scrape pipeline against frozen fixtures | Fast — sub-second |
| **Smoke (live API, manual)** | Sanity check against real MLB; run ~daily during dev | Slow — minutes |
| **Health-check tests** | Threshold logic, mirror existing `realized_calibration` test pattern | Fast |

**Fixture discipline**:
- Record fresh responses on day 1; freeze as JSON
- Re-record monthly via `scripts/refresh_leaderboard_fixtures.py` to catch silent schema drift
- Don't anonymize — leaderboard data is publicly visible

**Failure-mode coverage** (must-have tests):
- 401 mid-scrape → bail cleanly, log, alert (no half-written parquets)
- 429 with `Retry-After` → respect, back off, resume
- Schema drift (extra field) → log warn, write what's parseable, don't crash
- Partial network drop on user K of 100 → preserve K-1 successful writes, mark K+ as not-yet-scraped, retry next slot
- Empty leaderboard (off-season) → write empty parquet with header, don't error

**Coverage target**: 90%+ on the `bts.leaderboard` package.

## 12. Failure modes (summary)

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Auth expired | 401/403 on probe | Bluesky DM to Eric → manual `pass` refresh from Mac |
| Rate-limited | 429 | Back off per `Retry-After`, resume |
| Schema drift | Pydantic validation error | Log + write partial; continue; alert on next health check |
| Network failure mid-scrape | Connection error | Preserve completed rows, mark resume point, retry next slot |
| Disk full | Parquet write error | Existing `disk_fill` health check catches this |
| MLB blocks scraping | All requests 403 + ToS update | Pivot to W2 (third-party sites) or W1-only; manual investigation |

## 13. Phase 1 discoveries (CLOSED 2026-05-01)

All open questions resolved via `scripts/discover_bts_endpoints.py` (Playwright + cookie injection + network capture).

- ~~**Endpoint URLs**~~ — All under `https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/`:
  - `POST /auth/login` — exchange cookies + uid for short-lived xSid token
  - `GET /rank/leaderboard?season=...&ranksType=...&xSid=...` — season-wide tabs
  - `GET /rank/leaderboard/round/{N}?...&xSid=...` — round-specific (Yesterday tab)
  - `GET /rank/user/{userId}/profile?xSid=...` — picks + season stats (combined)
  - `GET /json/rounds.json` — static roundId → date mapping
- ~~**Auth scheme**~~ — Cookies (Okta-issued + MLB session) for identity. Per-call also requires an `xSid` query param obtained from `POST /auth/login` with body `{uid, platform: "web"}`. The `uid` lives in the `oktaid` cookie.
- ~~**Cookie lifetime**~~ — Not directly tested; observed `okta-access-token` JWT in cookies suggests Okta-managed sessions (typically 30-day refresh). xSid expires within hours. Safe assumption: re-auth/login on every scrape; manual cookie refresh ~monthly.
- ~~**Pagination**~~ — Leaderboard supports `page` + `limit` query params; `limit=100` returns top 100 in one response. Top-100 is the current cap; deeper paging untested.
- ~~**Picks log depth**~~ — `/rank/user/{userId}/profile` returns up to ~36 most recent rounds for an active streak (verified against tombrady12 streak=35). Coverage of OLDER streaks (concluded earlier in season) untested — likely per-user-state-dependent.

### Design adjustments from Phase 1

These differ from earlier sections; the implementation should follow these (the original sections are kept for historical context):

1. **Per-user endpoint is `profile`, not separate `picks` + `stats`.** A single `/rank/user/{userId}/profile` returns:
   - Top-level: `activeStreak`, `seasonBestStreak`, `accuracy`, `favouriteBatter`, `predictions[]`
   - Per `predictions[]` row: `roundId`, `result`, `streak`, `streakIncrease`, `roundPredictions[]`
   - Per `roundPredictions[]` row: `number` (1=primary, 2=DD), `unitId`, `playerId`, `result`, `hits`, `atBats`
   - Combine: storage layer should split this into `user_picks/<username>.parquet` (from `predictions × roundPredictions`) + `season_stats/<date>.parquet` (from top-level fields).

2. **`pick_date` is computed via roundId join.** The picks payload carries `roundId`, not `date`. Map via static `rounds.json`. The scraper should fetch + cache rounds.json once per scrape.

3. **PickRow needs `round_id` column.** Adding to the spec's data model so we can audit the join. `playerId` (MLB person_id) is preserved directly — that's stronger than the spec's `batter_name` join via roster.

4. **Auth flow**: each scrape run starts with `POST /auth/login` to mint a fresh xSid. Cache it in memory for the run; don't persist (it expires).

### Future investigation thread (separate from this spec)

**Third-party BTS historical pick databases**: there may exist a fan-maintained DB or scraper archive somewhere on the internet that holds pre-2026 BTS picks (e.g., 57hits.com archives, kaggle datasets, GitHub repos with scraped data). Worth a one-time investigation pass — if found, that's a one-time corpus injection that would be transformative for strategy mining. Out of scope for this spec; tracked as its own open thread.

## 14. Out of scope (for clarity)

- **Adversarial pick classifier**: deferred to v1.5 once data has accumulated
- **Group/private-pool data**: out of scope; only public leaderboard
- **Pick inference from streak movement**: research-grade, low EV, not pursued
- **Real-time alerting on consensus shifts**: v2+; v1 is daily snapshots
- **Cross-season historical mining via MLB.com BTS app**: confirmed not possible (pre-2026 picks not exposed). Third-party-DB hunt tracked separately (see Section 13).
