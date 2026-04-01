# BTS Daily Pick Automation — Design Spec

## Goal

Automate the daily BTS workflow: run the model twice daily, post picks to Bluesky, handle early games and lineup changes, and notify Eric of picks without any manual intervention.

## Current Manual Workflow

1. Run `bts predict --date YYYY-MM-DD` at 11am ET (early games) and 5:30pm ET (full slate)
2. Compare picks across runs — if a later game has a better pick, switch
3. Post to Bluesky via `bts post` or API
4. Submit pick in BTS app manually

## What Needs Automating

### Densest Bucket Strategy

Pick from whichever time window has the most games that day. More games = more options = better top pick. Validated at 85.1% avg P@1 across 2024-2025, statistically equivalent to "always prime" (85.4%) but handles edge cases (no-prime days) without special logic.

Three time windows:
- **Early**: before 4pm ET (day games, getaway days)
- **Prime**: 4-8pm ET (bulk of schedule — densest ~90% of days)
- **West**: 8pm+ ET (west coast)

**Two-run schedule:**
- **11:00 AM ET**: Check early games. Lock a pick ONLY if the early window is densest AND the best pick exceeds 80% P(game hit). Otherwise this run is informational.
- **4:00 PM ET**: Main run. Identify the densest window (usually prime). Lock the best pick from that window. Post to Bluesky.

On the ~6 days/season with no prime games, the densest bucket naturally falls to early or west — no fallback logic needed.

Game time distribution (2025 season, ET):
```
1-3pm:   519 games (day games, getaway days)
4-5pm:   348 games (early evening)
6-7pm:   947 games (prime time — bulk of schedule)
8-9pm:   451 games (late evening)
10pm+:   129 games (west coast)
```

**Why not three runs?** Backtesting showed "always wait for west coast" underperforms prime-only (81.1% vs 85.1%). The west pool is smaller, producing noisier picks. The 7:30pm run adds complexity without improving accuracy.

### Bluesky Posting

- Post after the **4pm run** (the main run)
- Only post once per day — don't post intermediate picks that might change
- Exception: if the 11am run locks an early game pick (>80%, densest bucket is early), post then
- Never delete posts without human approval

### Notifications

- Send Eric a notification with the pick and key details (batter, team, pitcher, P(game), double-down recommendation)
- Alert if something breaks (no games found, API errors, model failures)
- Notification channel: TBD (Telegram bot exists but is parked per memory notes, could use email, SMS, or push notification)

### Vegas Odds Integration (when ready)

- Pull live odds from the-odds-api.com ~2 hours before first pitch
- Requires the API key stored in macOS keychain as `odds-api-key`
- Feed into model as `market_p_hit` feature
- For now this is future work — the feature isn't shipped yet

## Architecture Options

### Option A: Scheduled Claude Code Agent (Recommended)

Use Claude Code's remote trigger / scheduled agent feature to run the workflow.

**Pros:**
- Can handle unexpected situations (API changes, errors, edge cases)
- Can compose Bluesky posts with context
- Can make judgment calls (e.g., skip posting if data quality is poor)
- Already has access to all tools and keychain credentials

**Cons:**
- More expensive per-run than a bare script
- Depends on Claude Code availability
- Overkill for a routine task that rarely needs judgment

### Option B: Python Cron Script on Pi5

A standalone Python script that runs on cron, executes the model, and posts.

**Pros:**
- Reliable, runs without internet dependency on AI services
- Cheap (no API costs beyond odds data)
- Fast (no LLM in the loop)

**Cons:**
- Rigid — can't handle unexpected situations
- Needs the full BTS environment installed on Pi5 (uv, LightGBM, data files)
- Pi5 may not have enough RAM/CPU for LightGBM on 1.5M PAs

### Option C: Hybrid

Python script handles routine execution (model run, Bluesky post). Claude Code agent runs weekly/on-demand for model updates, data pulls, and error recovery.

## Deployment Target

- **Pi5** (`ssh stonehengee@pi5`) — always on, already runs other scheduled tasks
- Needs: Python 3.12, uv, LightGBM (check ARM compatibility), ~16GB data files
- Alternative: Mac Mini if Pi5 can't handle the compute

## Data Requirements

The automation needs access to:
- `data/processed/pa_*.parquet` — 1.5M PAs across 9 seasons (~2GB)
- `data/external/savant_catcher_framing.csv` — catcher framing data
- `data/external/odds/v2/` — historical Vegas odds (for when the feature ships)
- `data/raw/` — only needed for rebuilding parquets (not daily operation)
- Keychain: `bluesky-bts-app-password`, `odds-api-key`

## Daily Output

The automation should produce a daily pick file at `data/picks/{date}.json`:

```json
{
  "date": "2026-04-01",
  "run_time": "2026-04-01T17:30:00-04:00",
  "pick": {
    "batter_name": "Jacob Wilson",
    "batter_id": 123456,
    "team": "ATH",
    "lineup_position": 1,
    "pitcher_name": "José Suarez",
    "pitcher_id": 654321,
    "p_game_hit": 0.763,
    "flags": [],
    "projected_lineup": false
  },
  "double_down": null,
  "runner_up": {
    "batter_name": "Jake Mangum",
    "p_game_hit": 0.726
  },
  "streak": 3,
  "bluesky_posted": true,
  "bluesky_uri": "at://did:plc:.../app.bsky.feed.post/..."
}
```

## Streak Tracking

- Maintain a streak counter in `data/picks/streak.json`
- After each game day, check if the pick got a hit (via MLB Stats API)
- Update streak: increment on hit, reset on miss
- Include streak count in Bluesky posts

## Error Handling

- If MLB API is down: retry 3 times, then notify Eric
- If no games found: skip the day, no post
- If model fails: notify Eric, don't post
- If Bluesky API fails: save the post text, retry next hour, notify Eric
- If lineups aren't posted by 5:30pm: use projected lineups with warning flag

## BTS App Submission

The actual BTS pick submission in the app still needs to be manual (the app doesn't have a public API). The automation handles everything up to the pick recommendation — Eric just opens the app and submits the name.

Future: investigate if the BTS website has a form that could be automated via browser automation.

## Security Notes

- Bluesky app password in macOS keychain (Mac) — needs equivalent on Pi5
- Odds API key in keychain — same
- No credentials in code or config files
- Pi5 SSH access via PQC keys (already configured)

## Implementation Order

1. **Minimal viable**: Python script that runs `bts predict`, saves pick JSON, posts to Bluesky. Cron on Mac for now.
2. **Pi5 deployment**: Sync data to Pi5, install dependencies, move cron there.
3. **Streak tracking**: Auto-check results, maintain streak counter.
4. **Notifications**: Send pick to Eric via preferred channel.
5. **Vegas integration**: Add odds pull to the pre-model step.
6. **Result posting**: Post next-day results with hit/miss and updated streak.
