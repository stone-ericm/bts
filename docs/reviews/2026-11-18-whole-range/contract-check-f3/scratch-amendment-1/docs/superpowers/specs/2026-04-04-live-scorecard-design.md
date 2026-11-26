# Live Scorecard — Design Spec

## Overview

Add a live baseball scorecard to the BTS dashboard that tracks picked batters' plate appearances during a game. The scorecard appears inline between the hero pick cards and pick history when the game is Live, and remains visible after the game goes Final.

Visual style follows caught-looking.app's scorecard conventions (numbered pitch grids, diamond with baserunning, trajectory lines) adapted to the dashboard's existing white/navy MLB theme.

## Data Source

MLB game feed endpoint: `GET /api/v1.1/game/{game_pk}/feed/live`

Relevant paths within the response:
- `liveData.plays.allPlays[]` — all plate appearances
- `allPlays[].matchup.batter.id` — filter to picked batters
- `allPlays[].result.event` / `.eventType` — PA result (single, strikeout, field_out, etc.)
- `allPlays[].playEvents[]` — per-pitch sequence
  - `.details.call.code` — B (ball), C (called strike), S (swinging strike), F (foul), X (in play out), D (in play no out), *B (ball in dirt)
- `allPlays[].runners[]` — base movement
  - `.movement.start` / `.end` / `.isOut`
- `allPlays[].playEvents[last].hitData` — exit velo, trajectory, coordinates
- `gameData.status.abstractGameCode` — P/L/F
- `liveData.linescore` — inning, score

No new data pipeline. Same API endpoint the scheduler and prediction pipeline already use.

## Server Side (`web.py`)

### New function: `fetch_live_scorecard(game_pk, batter_ids)`

Fetches the game feed and extracts PA data for the specified batters.

Returns:
```python
{
    "game_status": "L",  # P/L/F
    "inning": "Top 4th",
    "score": {"away": 1, "home": 0},
    "away_team": "TB",
    "home_team": "MIN",
    "batters": [
        {
            "name": "Yandy Diaz",
            "batter_id": 650490,
            "batting_hand": "R",
            "lineup_position": 2,
            "position": "DH",
            "slash_line": ".419/.486/.645",  # from boxscore
            "pas": [
                {
                    "inning": 1,
                    "result": "F9",
                    "event_type": "field_out",
                    "is_hit": false,
                    "is_out": true,
                    "out_number": 1,
                    "rbi": 0,
                    "pitches": [
                        {"number": 1, "call": "C", "is_strike": true},
                        {"number": 2, "call": "B", "is_strike": false},
                        {"number": 3, "call": "C", "is_strike": true},
                        {"number": 4, "call": "B", "is_strike": false},
                        {"number": 5, "call": "X", "is_strike": false}
                    ],
                    "runners": [
                        {"start": null, "end": null, "is_out": true}
                    ],
                    "hit_trajectory": {"x": 212.7, "y": 115.7, "type": "fly_ball"}
                }
            ]
        }
    ]
}
```

### New endpoint: `GET /api/live?date=YYYY-MM-DD`

Returns the scorecard JSON above. Used by client-side polling.

- Reads today's pick file to get `game_pk` and `batter_id`s
- Calls `fetch_live_scorecard`
- Returns JSON with `Content-Type: application/json`
- Returns `{"game_status": "P"}` if game hasn't started (no scorecard data)
- Returns `{"game_status": null}` if no pick exists for the date

### Changes to `render_page()`

- After the hero pick cards, render the scorecard section if today's game is Live or Final
- Server-renders the initial scorecard state (works without JS)
- Adds a `<script>` block (~30 lines) for client-side polling
- Adds an `id="scorecard"` wrapper div for JS to update

## Client Side

Inline vanilla JS in the page, no framework. Approximately 30 lines:

- `setInterval` polls `/api/live?date=...` every 30 seconds
- On success, rebuilds the scorecard table from the JSON response using DOM methods
- Polling stops automatically when `game_status === "F"`
- Errors fail silently (retry on next interval)

Note: the `/api/live` endpoint returns trusted, server-generated JSON (not user input). The JS rendering uses the same data flow as the server-rendered version.

## Result Code Mapping

The API provides `result.event` (human-readable) and `result.eventType` (machine code). We need traditional scorecard shorthand:

| eventType | Shorthand | Notes |
|-----------|-----------|-------|
| single | 1B | |
| double | 2B | |
| triple | 3B | |
| home_run | HR | |
| strikeout | K or backwards K | Backwards K if last pitch `call.code` is "C" (called) |
| field_out | F9, G6, L7, etc. | Derive from `runners[0].credits[0].position` (fielder) + trajectory |
| walk | BB | |
| hit_by_pitch | HBP | |
| sac_fly | SF | |
| sac_bunt | SAC | |
| force_out | FC | |
| double_play | DP | |
| field_error | E + position | |
| grounded_into_double_play | GDP | |

For `field_out`, prefix with F (fly), G (ground), L (line), P (pop) based on `hitData.trajectory`:
- fly_ball → F
- ground_ball → G  
- line_drive → L
- popup → P

Fielder position number from the first credit in `runners[].credits[]` with `credit == "f_fielded_ball"`.

## Per-PA Cell Visual Elements

Each PA cell (~100px wide, ~80px tall) contains:

1. **Result code** (top-left): F9, K, 2B, HBP, BB, etc.
   - Green text (`#16a34a`) for hits (single, double, triple, home_run)
   - Black text for other results
   - Backwards K (mirrored) for called strikeouts

2. **Pitch grid** (top-right): Numbered in 2-column layout
   - Strikes: MLB red (`#c41e3a`)
   - Balls: gray (`#aaa`)
   - Fouls: red with border (distinguishes from swinging strikes)
   - Final pitch: bold
   - In-play: bold with X marker

3. **Diamond** (bottom-right): 36x36 SVG
   - Bases as rotated squares (caught-looking style)
   - Filled black/green when occupied after the PA
   - Basepath lines showing runner advancement
   - Home plate filled gray when batter didn't reach
   - Dashed trajectory line for balls in play (red for outs, green for hits)
   - Hit trajectory direction derived from `hitData.coordinates`

4. **Out number** (bottom-left): Circled in MLB red for outs

5. **RBI dots** (bottom-left, below out number): Small filled circles

6. **Green background tint**: `rgba(34,197,94,0.08)` — only on PAs where `eventType` is a hit (single, double, triple, home_run). NOT on walks, HBP, errors, etc.

### Upcoming PAs

Empty cells with dashed border, dimmed, showing estimated inning (~5th, ~7th, etc.) based on lineup position.

## BTS Status Banner

Below the scorecard table, a status banner summarizing BTS implications:

| State | Message | Style |
|-------|---------|-------|
| Game live, no PA yet | "Waiting for first plate appearance..." | Gray |
| 0 hits so far | "0/N batters with a hit — [next batter] due up ~Xth" | Gray |
| Some hits | "1/2 batters with a hit — [other batter] due up ~Xth" | Yellow |
| All picks have hits | "Both batters have a hit — streak advances if game goes Final" | Green |
| Game final, all hit | "Streak advances to N!" | Green |
| Game final, miss | "Miss — [name] went 0-for-N" | Red |

For single picks (no double-down), simplify to "Batter has/hasn't recorded a hit".

## Lifecycle

- **Pre-game (P)**: No scorecard section. Dashboard shows hero cards as-is.
- **Live (L)**: Scorecard appears between hero cards and pick history. JS polling active (30s).
- **Final (F)**: Scorecard stays visible with final PA results. Polling stops. Status banner shows final result.
- **No pick today / skip day**: No scorecard section.

## What Stays the Same

- `web.py` remains a single file using `http.server`
- No external Python or JS dependencies added
- Dashboard fully server-rendered on initial load (JS enhances, doesn't gate)
- Pick history table, Bluesky embeds, hero cards, streak counter unchanged
- CSS remains inline in the page

## Scope Boundaries

**In scope:**
- Fetching and displaying PA data for picked batters only
- Pitch sequence, diamond, trajectory, result code
- Auto-polling during live games
- Status banner with BTS context

**Out of scope:**
- Full game scorecard (all batters)
- Pitch location heatmap / zone overlay
- Replay/challenge tracking
- Historical scorecard for past games (only today's live/recent game)
- Mobile-specific layout changes (existing responsive CSS handles width)

## Testing

- Unit test for `fetch_live_scorecard` with mocked game feed JSON
- Unit test for PA result classification (is_hit logic)
- Unit test for pitch sequence extraction
- Manual verification against caught-looking.app for visual fidelity
