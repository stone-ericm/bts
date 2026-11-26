# Live Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live baseball scorecard to the BTS dashboard that shows picked batters' plate appearances with pitch grids, diamonds, and trajectory lines during live games.

**Architecture:** New module `src/bts/scorecard.py` handles data extraction from the MLB game feed. `web.py` gets a new `/api/live` JSON endpoint and scorecard HTML rendering. Inline vanilla JS polls the endpoint every 30s during live games.

**Tech Stack:** Python stdlib (`http.server`, `json`, `urllib`), inline SVG for diamonds, vanilla JS for polling. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-04-live-scorecard-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/bts/scorecard.py` | Create | Data extraction: fetch game feed, extract PAs for specific batters, structure data |
| `src/bts/web.py` | Modify | New `/api/live` endpoint, scorecard HTML rendering, JS polling script |
| `tests/test_scorecard.py` | Create | Unit tests for data extraction and result code mapping |

---

### Task 1: Scorecard Data Extraction — Result Codes

**Files:**
- Create: `tests/test_scorecard.py`
- Create: `src/bts/scorecard.py`

- [ ] **Step 1: Write failing tests for result code mapping**

```python
# tests/test_scorecard.py
"""Tests for live scorecard data extraction."""
import pytest
from bts.scorecard import format_result_code


class TestFormatResultCode:
    def test_single(self):
        assert format_result_code("single", "field_out", None, None, None) == "1B"

    def test_double(self):
        assert format_result_code("double", "double", None, None, None) == "2B"

    def test_triple(self):
        assert format_result_code("triple", "triple", None, None, None) == "3B"

    def test_home_run(self):
        assert format_result_code("home_run", "home_run", None, None, None) == "HR"

    def test_walk(self):
        assert format_result_code("walk", "walk", None, None, None) == "BB"

    def test_hit_by_pitch(self):
        assert format_result_code("hit_by_pitch", "hit_by_pitch", None, None, None) == "HBP"

    def test_strikeout_swinging(self):
        # Last pitch was swinging strike (code "S")
        assert format_result_code("strikeout", "strikeout", "S", None, None) == "K"

    def test_strikeout_looking(self):
        # Last pitch was called strike (code "C") — backwards K
        assert format_result_code("strikeout", "strikeout", "C", None, None) == "\u042f"  # Cyrillic Ya = backwards K

    def test_flyout_to_right(self):
        assert format_result_code("field_out", "field_out", None, "fly_ball", 9) == "F9"

    def test_groundout_to_short(self):
        assert format_result_code("field_out", "field_out", None, "ground_ball", 6) == "G6"

    def test_lineout_to_center(self):
        assert format_result_code("field_out", "field_out", None, "line_drive", 8) == "L8"

    def test_popup_to_second(self):
        assert format_result_code("field_out", "field_out", None, "popup", 4) == "P4"

    def test_flyout_no_trajectory(self):
        # Missing trajectory data — fall back to generic
        assert format_result_code("field_out", "field_out", None, None, 9) == "F9"

    def test_sac_fly(self):
        assert format_result_code("sac_fly", "sac_fly", None, None, None) == "SF"

    def test_sac_bunt(self):
        assert format_result_code("sac_bunt", "sac_bunt", None, None, None) == "SAC"

    def test_double_play(self):
        assert format_result_code("double_play", "double_play", None, None, None) == "DP"

    def test_grounded_into_double_play(self):
        assert format_result_code("grounded_into_double_play", "grounded_into_double_play", None, None, None) == "GDP"

    def test_force_out(self):
        assert format_result_code("force_out", "force_out", None, None, None) == "FC"

    def test_field_error(self):
        assert format_result_code("field_error", "field_error", None, None, 6) == "E6"

    def test_field_error_no_position(self):
        assert format_result_code("field_error", "field_error", None, None, None) == "E"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.scorecard'`

- [ ] **Step 3: Implement format_result_code**

```python
# src/bts/scorecard.py
"""Live scorecard data extraction from MLB game feed.

Fetches play-by-play data for specific batters and structures it
for the dashboard's live scorecard display.
"""

from bts.data.schema import HIT_EVENTS

# Event type -> scorecard shorthand
_RESULT_MAP = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "sac_fly": "SF",
    "sac_bunt": "SAC",
    "double_play": "DP",
    "grounded_into_double_play": "GDP",
    "force_out": "FC",
    "intent_walk": "IBB",
    "catcher_interf": "CI",
}

# Hit trajectory -> prefix for field_out
_TRAJECTORY_PREFIX = {
    "fly_ball": "F",
    "ground_ball": "G",
    "line_drive": "L",
    "popup": "P",
}


def format_result_code(
    event: str,
    event_type: str,
    last_pitch_code: str | None,
    trajectory: str | None,
    fielder_position: int | None,
) -> str:
    """Convert MLB API event data to traditional scorecard shorthand."""
    if event_type == "strikeout":
        # Backwards K for called third strike
        return "\u042f" if last_pitch_code == "C" else "K"

    if event_type in ("field_out",):
        prefix = _TRAJECTORY_PREFIX.get(trajectory, "F")
        return f"{prefix}{fielder_position}" if fielder_position else f"{prefix}?"

    if event_type == "field_error":
        return f"E{fielder_position}" if fielder_position else "E"

    return _RESULT_MAP.get(event_type, event_type.upper()[:3] if event_type else "?")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/scorecard.py tests/test_scorecard.py
git commit -m "feat(scorecard): add result code mapping from MLB API events"
```

---

### Task 2: Scorecard Data Extraction — PA Parsing

**Files:**
- Modify: `tests/test_scorecard.py`
- Modify: `src/bts/scorecard.py`

- [ ] **Step 1: Write failing test for extract_batter_pas**

Add to `tests/test_scorecard.py`:

```python
import json

from bts.scorecard import extract_batter_pas


# Minimal game feed fixture — one PA for batter 650490 (Diaz)
SAMPLE_PLAY = {
    "result": {
        "type": "atBat",
        "event": "Flyout",
        "eventType": "field_out",
        "rbi": 0,
        "isOut": True,
    },
    "about": {
        "atBatIndex": 0,
        "inning": 1,
        "halfInning": "top",
        "isComplete": True,
    },
    "count": {"balls": 3, "strikes": 1, "outs": 1},
    "matchup": {
        "batter": {"id": 650490, "fullName": "Yandy Diaz"},
        "batSide": {"code": "R"},
        "pitcher": {"id": 621298, "fullName": "Joe Ryan"},
        "pitchHand": {"code": "R"},
    },
    "playEvents": [
        {
            "isPitch": True,
            "details": {"call": {"code": "B"}, "isStrike": False, "isBall": True},
            "pitchData": {"startSpeed": 94.3},
            "count": {"balls": 1, "strikes": 0},
        },
        {
            "isPitch": True,
            "details": {"call": {"code": "C"}, "isStrike": True, "isBall": False},
            "pitchData": {"startSpeed": 92.1},
            "count": {"balls": 1, "strikes": 1},
        },
        {
            "isPitch": True,
            "details": {
                "call": {"code": "X", "description": "In play, out(s)"},
                "isStrike": False, "isBall": False, "isInPlay": True,
            },
            "pitchData": {"startSpeed": 95.0},
            "hitData": {
                "trajectory": "fly_ball",
                "coordinates": {"coordX": 212.7, "coordY": 115.7},
                "launchSpeed": 84.5,
            },
            "count": {"balls": 1, "strikes": 1},
        },
    ],
    "runners": [
        {
            "movement": {"start": None, "end": None, "outBase": "1B", "isOut": True, "outNumber": 1},
            "details": {"runner": {"id": 650490}},
            "credits": [{"player": {"id": 999}, "position": {"code": "9"}, "credit": "f_fielded_ball"}],
        },
    ],
}

SAMPLE_FEED = {
    "gameData": {
        "status": {"abstractGameCode": "L", "detailedState": "In Progress"},
        "teams": {
            "away": {"abbreviation": "TB"},
            "home": {"abbreviation": "MIN"},
        },
    },
    "liveData": {
        "plays": {"allPlays": [SAMPLE_PLAY]},
        "linescore": {
            "currentInning": 4,
            "inningHalf": "Top",
            "teams": {"away": {"runs": 1}, "home": {"runs": 0}},
        },
        "boxscore": {
            "teams": {
                "away": {
                    "players": {
                        "ID650490": {
                            "person": {"id": 650490, "fullName": "Yandy Diaz"},
                            "position": {"abbreviation": "DH"},
                            "battingOrder": "200",
                            "stats": {"batting": {"avg": ".419", "obp": ".486", "slg": ".645"}},
                        },
                    },
                },
                "home": {"players": {}},
            },
        },
    },
}


class TestExtractBatterPas:
    def test_extracts_single_pa(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        assert result["game_status"] == "L"
        assert result["inning"] == "Top 4th"
        assert len(result["batters"]) == 1

        batter = result["batters"][0]
        assert batter["name"] == "Yandy Diaz"
        assert batter["batter_id"] == 650490
        assert len(batter["pas"]) == 1

        pa = batter["pas"][0]
        assert pa["result"] == "F9"
        assert pa["is_hit"] is False
        assert pa["is_out"] is True
        assert pa["out_number"] == 1
        assert pa["inning"] == 1
        assert len(pa["pitches"]) == 3

    def test_pitch_sequence(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        pitches = result["batters"][0]["pas"][0]["pitches"]
        assert pitches[0] == {"number": 1, "call": "B", "is_strike": False}
        assert pitches[1] == {"number": 2, "call": "C", "is_strike": True}
        assert pitches[2] == {"number": 3, "call": "X", "is_strike": False}

    def test_filters_to_requested_batters(self):
        result = extract_batter_pas(SAMPLE_FEED, {999999})
        assert len(result["batters"]) == 0

    def test_hit_trajectory(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        pa = result["batters"][0]["pas"][0]
        assert pa["hit_trajectory"]["type"] == "fly_ball"
        assert pa["hit_trajectory"]["x"] == 212.7

    def test_runner_movement(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        pa = result["batters"][0]["pas"][0]
        assert len(pa["runners"]) == 1
        assert pa["runners"][0]["is_out"] is True

    def test_batter_info_from_boxscore(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        batter = result["batters"][0]
        assert batter["position"] == "DH"
        assert batter["lineup_position"] == 2
        assert batter["slash_line"] == ".419/.486/.645"

    def test_score(self):
        result = extract_batter_pas(SAMPLE_FEED, {650490})
        assert result["score"] == {"away": 1, "home": 0}
        assert result["away_team"] == "TB"
        assert result["home_team"] == "MIN"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestExtractBatterPas -v`
Expected: FAIL — `ImportError: cannot import name 'extract_batter_pas'`

- [ ] **Step 3: Implement extract_batter_pas**

Add to `src/bts/scorecard.py`:

```python
def _ordinal(n: int) -> str:
    """Convert integer to ordinal string: 1 -> '1st', 2 -> '2nd', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


def _extract_fielder_position(runners: list[dict]) -> int | None:
    """Get fielder position from the first fielding credit."""
    for runner in runners:
        for credit in runner.get("credits", []):
            if credit.get("credit") == "f_fielded_ball":
                try:
                    return int(credit["position"]["code"])
                except (KeyError, ValueError, TypeError):
                    pass
    return None


def _extract_pa(play: dict) -> dict:
    """Extract a single plate appearance from an allPlays entry."""
    result = play.get("result", {})
    event = result.get("event", "")
    event_type = result.get("eventType", "")

    # Pitch sequence
    pitches = []
    last_pitch_code = None
    pitch_num = 0
    for pe in play.get("playEvents", []):
        if not pe.get("isPitch"):
            continue
        pitch_num += 1
        call_code = pe.get("details", {}).get("call", {}).get("code", "?")
        is_strike = pe.get("details", {}).get("isStrike", False)
        last_pitch_code = call_code
        pitches.append({
            "number": pitch_num,
            "call": call_code,
            "is_strike": is_strike,
        })

    # Hit trajectory
    hit_trajectory = None
    for pe in reversed(play.get("playEvents", [])):
        hd = pe.get("hitData") or pe.get("details", {}).get("hitData")
        if hd:
            coords = hd.get("coordinates", {})
            hit_trajectory = {
                "x": coords.get("coordX"),
                "y": coords.get("coordY"),
                "type": hd.get("trajectory"),
            }
            break

    # Runners
    runners_data = []
    for runner in play.get("runners", []):
        mv = runner.get("movement", {})
        runners_data.append({
            "start": mv.get("start"),
            "end": mv.get("end"),
            "is_out": mv.get("isOut", False),
        })

    # Fielder position for result code
    fielder_pos = _extract_fielder_position(play.get("runners", []))
    trajectory_type = hit_trajectory["type"] if hit_trajectory else None

    # Out number
    out_number = None
    if result.get("isOut"):
        for runner in play.get("runners", []):
            on = runner.get("movement", {}).get("outNumber")
            if on is not None:
                out_number = on
                break

    return {
        "inning": play.get("about", {}).get("inning", 0),
        "result": format_result_code(event_type, event_type, last_pitch_code, trajectory_type, fielder_pos),
        "event_type": event_type,
        "is_hit": event_type in HIT_EVENTS,
        "is_out": result.get("isOut", False),
        "out_number": out_number,
        "rbi": result.get("rbi", 0),
        "pitches": pitches,
        "runners": runners_data,
        "hit_trajectory": hit_trajectory,
    }


def extract_batter_pas(feed: dict, batter_ids: set[int]) -> dict:
    """Extract plate appearances for specific batters from a game feed.

    Args:
        feed: Full MLB game feed JSON (from /api/v1.1/game/{pk}/feed/live)
        batter_ids: Set of batter IDs to extract PAs for

    Returns:
        Structured dict with game state and per-batter PA data.
    """
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    status = game_data.get("status", {})
    linescore = live_data.get("linescore", {})
    boxscore = live_data.get("boxscore", {})

    # Game state
    inning_num = linescore.get("currentInning", 0)
    inning_half = linescore.get("inningHalf", "")
    inning_str = f"{inning_half} {_ordinal(inning_num)}" if inning_num else ""

    teams_score = linescore.get("teams", {})
    score = {
        "away": teams_score.get("away", {}).get("runs", 0),
        "home": teams_score.get("home", {}).get("runs", 0),
    }

    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("abbreviation", "?")
    home_team = teams.get("home", {}).get("abbreviation", "?")

    # Extract PAs per batter
    all_plays = live_data.get("plays", {}).get("allPlays", [])
    batter_pas: dict[int, list] = {}
    for play in all_plays:
        matchup = play.get("matchup", {})
        bid = matchup.get("batter", {}).get("id")
        if bid not in batter_ids:
            continue
        if play.get("result", {}).get("type") != "atBat":
            continue
        if not play.get("about", {}).get("isComplete", False):
            continue
        batter_pas.setdefault(bid, []).append(_extract_pa(play))

    # Build batter info from boxscore
    batters = []
    for bid in batter_ids:
        # Find batter in boxscore (either team)
        batter_info = None
        for side in ("away", "home"):
            players = boxscore.get("teams", {}).get(side, {}).get("players", {})
            key = f"ID{bid}"
            if key in players:
                batter_info = players[key]
                break

        if batter_info is None:
            continue

        person = batter_info.get("person", {})
        pos = batter_info.get("position", {}).get("abbreviation", "?")
        batting_order = batter_info.get("battingOrder", "0")
        try:
            lineup_pos = int(batting_order) // 100
        except (ValueError, TypeError):
            lineup_pos = 0

        stats = batter_info.get("stats", {}).get("batting", {})
        slash = f"{stats.get('avg', '.000')}/{stats.get('obp', '.000')}/{stats.get('slg', '.000')}"

        # Batting hand from plays
        batting_hand = "?"
        for play in all_plays:
            if play.get("matchup", {}).get("batter", {}).get("id") == bid:
                batting_hand = play["matchup"].get("batSide", {}).get("code", "?")
                break

        batters.append({
            "name": person.get("fullName", "?"),
            "batter_id": bid,
            "batting_hand": batting_hand,
            "lineup_position": lineup_pos,
            "position": pos,
            "slash_line": slash,
            "pas": batter_pas.get(bid, []),
        })

    # Sort by lineup position
    batters.sort(key=lambda b: b["lineup_position"])

    return {
        "game_status": status.get("abstractGameCode", "P"),
        "inning": inning_str,
        "score": score,
        "away_team": away_team,
        "home_team": home_team,
        "batters": batters,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/scorecard.py tests/test_scorecard.py
git commit -m "feat(scorecard): extract PA data from MLB game feed"
```

---

### Task 3: Scorecard Data Extraction — fetch_live_scorecard (network wrapper)

**Files:**
- Modify: `src/bts/scorecard.py`
- Modify: `tests/test_scorecard.py`

- [ ] **Step 1: Write failing test for fetch_live_scorecard**

Add to `tests/test_scorecard.py`:

```python
from unittest.mock import patch
from bts.scorecard import fetch_live_scorecard


class TestFetchLiveScorecard:
    @patch("bts.scorecard.retry_urlopen")
    def test_fetches_and_extracts(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(SAMPLE_FEED).encode()

        result = fetch_live_scorecard(823730, {650490})
        assert result["game_status"] == "L"
        assert len(result["batters"]) == 1
        assert result["batters"][0]["name"] == "Yandy Diaz"
        mock_urlopen.assert_called_once()

    @patch("bts.scorecard.retry_urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")
        result = fetch_live_scorecard(823730, {650490})
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestFetchLiveScorecard -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement fetch_live_scorecard**

Add to `src/bts/scorecard.py`:

```python
import json

from bts.util import retry_urlopen

API_BASE = "https://statsapi.mlb.com"


def fetch_live_scorecard(game_pk: int, batter_ids: set[int]) -> dict | None:
    """Fetch game feed and extract PA data for specific batters.

    Returns structured scorecard dict, or None on any error.
    """
    try:
        resp = retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        )
        feed = json.loads(resp.read())
        return extract_batter_pas(feed, batter_ids)
    except Exception:
        return None
```

Also add `import json` at the top of the file and import `retry_urlopen` from `bts.util`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/scorecard.py tests/test_scorecard.py
git commit -m "feat(scorecard): add fetch_live_scorecard network wrapper"
```

---

### Task 4: Dashboard — /api/live Endpoint

**Files:**
- Modify: `src/bts/web.py`

- [ ] **Step 1: Add /api/live route to Handler.do_GET**

Modify `Handler.do_GET` in `web.py` to route `/api/live` requests:

```python
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self.path)

        if parsed.path == "/api/live":
            self._handle_api_live(parse_qs(parsed.query))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_page().encode())

    def _handle_api_live(self, params):
        """Return live scorecard JSON for today's picked batters."""
        from bts.scorecard import fetch_live_scorecard

        date = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]

        # Load today's pick to get game_pk and batter_ids
        pick_path = PICKS_DIR / f"{date}.json"
        if not pick_path.exists():
            self._json_response({"game_status": None})
            return

        pick_data = json.loads(pick_path.read_text())
        pick = pick_data.get("pick", {})
        game_pk = pick.get("game_pk")
        if not game_pk:
            self._json_response({"game_status": None})
            return

        batter_ids = {pick.get("batter_id")}
        dd = pick_data.get("double_down")
        if dd:
            batter_ids.add(dd.get("batter_id"))
        batter_ids.discard(None)

        result = fetch_live_scorecard(game_pk, batter_ids)
        if result is None:
            self._json_response({"game_status": None, "error": "feed unavailable"})
            return

        self._json_response(result)

    def _json_response(self, data):
        """Send a JSON response."""
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass
```

- [ ] **Step 2: Manual test**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python -m bts.web &`
Then: `curl -s http://localhost:3003/api/live?date=2026-04-04 | python -m json.tool | head -20`
Expected: JSON with game_status, batters array (or null if no pick file locally)

Kill the server after testing.

- [ ] **Step 3: Commit**

```bash
git add src/bts/web.py
git commit -m "feat(dashboard): add /api/live JSON endpoint for scorecard polling"
```

---

### Task 5: Dashboard — Scorecard HTML Rendering

**Files:**
- Modify: `src/bts/web.py`

This is the largest task. Add scorecard HTML rendering to `render_page()`.

- [ ] **Step 1: Add render_scorecard_html function to web.py**

Add a new function that takes the scorecard data dict and returns an HTML string for the scorecard table. This follows the v5 mockup design (white theme, caught-looking pitch grid, SVG diamonds).

```python
def _render_pitch_grid(pitches: list[dict]) -> str:
    """Render pitch sequence as a 2-column numbered grid."""
    if not pitches:
        return ""
    cells = ""
    for p in pitches:
        call = p["call"]
        num = p["number"]
        is_last = num == len(pitches)
        if call in ("C", "S"):
            style = "color:#c41e3a;font-weight:700;" if is_last else "color:#c41e3a;"
        elif call == "F":
            style = "color:#c41e3a;border:1px solid #c41e3a;border-radius:1px;"
        elif call in ("X", "D"):
            style = "color:#c41e3a;font-weight:700;" if call == "X" else "color:#16a34a;font-weight:700;"
        else:  # B, *B
            style = "color:#aaa;"
        cells += f'<div style="{style}">{num}</div>'
    return f'<div style="display:grid;grid-template-columns:11px 11px;gap:1px;font-size:8px;line-height:11px;text-align:center;">{cells}</div>'


def _render_diamond(pa: dict) -> str:
    """Render SVG diamond with baserunning and trajectory."""
    # Determine which bases are occupied after this PA
    bases = {"1B": False, "2B": False, "3B": False}
    batter_reached = False
    for r in pa.get("runners", []):
        end = r.get("end")
        if end in bases:
            bases[end] = True
        if end is not None and not r.get("is_out"):
            batter_reached = True

    # Base fills
    def base_fill(key):
        if pa.get("is_hit") and bases.get(key):
            return "#16a34a"
        return "#333" if bases.get(key) else "none"

    home_fill = "#999" if not batter_reached else "none"

    # Basepath lines
    paths = ""
    if batter_reached:
        paths += '<line x1="20" y1="36" x2="36" y2="20" stroke="#666" stroke-width="1.2"/>'
        if bases.get("2B") or bases.get("3B"):
            paths += '<line x1="36" y1="20" x2="20" y2="4" stroke="#666" stroke-width="1.2"/>'
        if bases.get("3B"):
            paths += '<line x1="20" y1="4" x2="4" y2="20" stroke="#666" stroke-width="1.2"/>'

    # Hit trajectory
    traj = ""
    ht = pa.get("hit_trajectory")
    if ht and ht.get("x") is not None:
        # Map coordX/coordY (0-250 range) to SVG diamond space
        # coordX: 0=left foul, 125=center, 250=right foul
        # coordY: 0=deep outfield, 250=home plate
        tx = 4 + (ht["x"] / 250) * 32
        ty = 4 + (ht["y"] / 250) * 32
        color = "#16a34a" if pa.get("is_hit") else "#c41e3a"
        traj = f'<path d="M20,36 L{tx:.0f},{ty:.0f}" fill="none" stroke="{color}" stroke-width="0.8" stroke-dasharray="2,1.5"/>'

    return f'''<svg width="36" height="36" viewBox="0 0 40 40">
        <polygon points="20,4 36,20 20,36 4,20" fill="none" stroke="#ccc" stroke-width="1"/>
        <rect x="18" y="2" width="4" height="4" transform="rotate(45 20 4)" fill="{base_fill('2B')}" stroke="#bbb" stroke-width="0.7"/>
        <rect x="34" y="18" width="4" height="4" transform="rotate(45 36 20)" fill="{base_fill('1B')}" stroke="#bbb" stroke-width="0.7"/>
        <rect x="2" y="18" width="4" height="4" transform="rotate(45 4 20)" fill="{base_fill('3B')}" stroke="#bbb" stroke-width="0.7"/>
        <rect x="18" y="34" width="4" height="4" transform="rotate(45 20 36)" fill="{home_fill}" stroke="#bbb" stroke-width="0.7"/>
        {paths}{traj}
    </svg>'''


def _render_pa_cell(pa: dict | None, estimated_inning: str = "") -> str:
    """Render a single PA cell."""
    if pa is None:
        return f'''<td style="padding:3px;border-right:1px solid #eee;vertical-align:top;">
            <div style="min-height:80px;display:flex;align-items:center;justify-content:center;opacity:0.3;">
                <span style="font-size:10px;color:#bbb;">{estimated_inning}</span>
            </div></td>'''

    bg = "background:rgba(34,197,94,0.08);" if pa.get("is_hit") else ""
    result_color = "#16a34a" if pa.get("is_hit") else "#333"

    # Backwards K rendering
    result_text = pa["result"]
    if result_text == "\u042f":
        result_text = '<span style="display:inline-block;transform:scaleX(-1);">K</span>'

    pitch_grid = _render_pitch_grid(pa.get("pitches", []))
    diamond = _render_diamond(pa)

    # Out number
    out_html = ""
    if pa.get("is_out") and pa.get("out_number"):
        out_html = f'<div style="width:14px;height:14px;border:1.5px solid #c41e3a;border-radius:50%;font-size:8px;font-weight:700;color:#c41e3a;display:flex;align-items:center;justify-content:center;">{pa["out_number"]}</div>'

    # RBI dots
    rbi_html = ""
    rbi = pa.get("rbi", 0)
    if rbi > 0:
        dots = "".join('<div style="width:5px;height:5px;background:#333;border-radius:50;"></div>' for _ in range(rbi))
        rbi_html = f'<div style="display:flex;gap:2px;margin-top:2px;">{dots}</div>'

    return f'''<td style="padding:3px;border-right:1px solid #eee;vertical-align:top;{bg}">
        <div style="position:relative;min-height:80px;padding:3px;">
            <div style="font-weight:700;font-size:13px;color:{result_color};">{result_text}</div>
            <div style="position:absolute;top:3px;right:3px;">{pitch_grid}</div>
            <div style="position:absolute;bottom:1px;right:1px;">{diamond}</div>
            <div style="position:absolute;bottom:1px;left:1px;">{out_html}{rbi_html}</div>
        </div></td>'''


def render_scorecard_section(scorecard_data: dict | None) -> str:
    """Render the full scorecard section HTML.

    Returns empty string if no scorecard data or game not started.
    """
    if not scorecard_data or scorecard_data.get("game_status") not in ("L", "F"):
        return ""

    status = scorecard_data["game_status"]
    inning = scorecard_data.get("inning", "")
    score = scorecard_data.get("score", {})
    away = scorecard_data.get("away_team", "?")
    home = scorecard_data.get("home_team", "?")

    status_badge_color = "#22c55e" if status == "L" else "#888"
    status_text = f"LIVE — {inning}" if status == "L" else "FINAL"

    # Build batter rows
    max_pas = 5
    batter_rows = ""
    for batter in scorecard_data.get("batters", []):
        pas = batter.get("pas", [])
        pa_cells = ""
        for i in range(max_pas):
            if i < len(pas):
                pa_cells += _render_pa_cell(pas[i])
            else:
                est = f"~{batter['lineup_position'] + (i * 9) // 2}{_ordinal_suffix(batter['lineup_position'] + i * 4)}" if i > len(pas) - 1 else ""
                pa_cells += _render_pa_cell(None, f"~{(i + 1) * 3 + batter.get('lineup_position', 1)}th" if i >= len(pas) else "")

        batter_rows += f'''<tr style="border-bottom:1px solid #eee;">
            <td style="padding:6px 8px;font-weight:700;font-size:13px;color:#888;border-right:1px solid #eee;vertical-align:top;">{batter.get("lineup_position", "?")}</td>
            <td style="padding:6px 8px;border-right:1px solid #eee;vertical-align:top;">
                <div style="font-weight:700;font-size:13px;color:#1a1a2e;">{batter["name"]} <span style="font-weight:400;color:#aaa;">({batter.get("batting_hand", "?")})</span></div>
                <div style="font-size:10px;color:#bbb;margin-top:1px;">{batter.get("slash_line", "")}</div>
            </td>
            <td style="padding:6px;text-align:center;font-weight:700;font-size:12px;color:#888;border-right:1px solid #eee;vertical-align:top;">{batter.get("position", "?")}</td>
            {pa_cells}
        </tr>'''

    # PA column headers
    pa_headers = ""
    for i in range(1, max_pas + 1):
        pa_headers += f'<th style="text-align:center;padding:6px 4px;font-size:12px;font-weight:700;color:#666;border-bottom:1px solid #e0e0e0;border-right:1px solid #eee;width:100px;">{i}</th>'

    # BTS status banner
    batters = scorecard_data.get("batters", [])
    hits = [b for b in batters if any(pa.get("is_hit") for pa in b.get("pas", []))]
    total = len(batters)

    if status == "F":
        if len(hits) == total:
            banner_bg = "#f0fdf4"
            banner_border = "#bbf7d0"
            banner_color = "#16a34a"
            banner_icon = "&#9989;"
            banner_text = f"Streak advances to {load_streak() + (2 if total > 1 else 1)}!"
        else:
            no_hit = [b for b in batters if not any(pa.get("is_hit") for pa in b.get("pas", []))]
            names = ", ".join(b["name"] for b in no_hit)
            banner_bg = "#fef2f2"
            banner_border = "#fecaca"
            banner_color = "#dc2626"
            banner_icon = "&#10060;"
            banner_text = f"Miss — {names} went hitless"
    elif len(hits) == total and total > 0:
        banner_bg = "#f0fdf4"
        banner_border = "#bbf7d0"
        banner_color = "#16a34a"
        banner_icon = "&#9989;"
        banner_text = "Both batters have a hit — streak advances if game goes Final" if total > 1 else "Batter has a hit — streak advances if game goes Final"
    elif len(hits) > 0:
        remaining = [b["name"] for b in batters if b not in hits]
        banner_bg = "#fffbeb"
        banner_border = "#fde68a"
        banner_color = "#d97706"
        banner_icon = "&#9888;&#65039;"
        banner_text = f"{len(hits)}/{total} with a hit — waiting on {', '.join(remaining)}"
    else:
        banner_bg = "#f8f8f8"
        banner_border = "#e0e0e0"
        banner_color = "#888"
        banner_icon = "&#9679;"
        banner_text = "Waiting for hits..."

    return f'''
        <div class="section-header" style="display:flex;align-items:center;gap:8px;">
            LIVE SCORECARD
            <span style="background:{status_badge_color};color:#fff;font-size:9px;padding:1px 6px;border-radius:3px;letter-spacing:0;">{status_text}</span>
            <span style="margin-left:auto;font-size:11px;color:#888;font-weight:400;letter-spacing:0;text-transform:none;">{away} {score.get("away", 0)} - {home} {score.get("home", 0)}</span>
        </div>
        <div id="scorecard">
        <table style="table-layout:fixed;">
            <thead>
                <tr style="background:#f8f9fa;">
                    <th style="text-align:left;padding:6px 8px;font-size:10px;font-weight:600;color:#999;border-bottom:1px solid #e0e0e0;border-right:1px solid #eee;width:30px;">#</th>
                    <th style="text-align:left;padding:6px 8px;font-size:10px;font-weight:600;color:#999;border-bottom:1px solid #e0e0e0;border-right:1px solid #eee;">BATTERS</th>
                    <th style="text-align:center;padding:6px 6px;font-size:10px;font-weight:600;color:#999;border-bottom:1px solid #e0e0e0;border-right:1px solid #eee;width:32px;">POS</th>
                    {pa_headers}
                </tr>
            </thead>
            <tbody>
                {batter_rows}
            </tbody>
        </table>
        <div style="background:{banner_bg};border:1px solid {banner_border};border-radius:6px;padding:8px 14px;display:flex;align-items:center;gap:8px;margin-top:10px;">
            <span style="font-size:14px;">{banner_icon}</span>
            <span style="font-size:12px;color:{banner_color};font-weight:600;">{banner_text}</span>
        </div>
        </div>'''
```

- [ ] **Step 2: Integrate into render_page**

In `render_page()`, after building the `hero` variable and before the Pick History section, add the scorecard fetch and rendering:

```python
    # Live scorecard (between hero and pick history)
    scorecard_html = ""
    if today_pick:
        tp = today_pick["pick"]
        game_pk = tp.get("game_pk")
        if game_pk:
            batter_ids = {tp.get("batter_id")}
            dd = today_pick.get("double_down")
            if dd:
                batter_ids.add(dd.get("batter_id"))
            batter_ids.discard(None)
            from bts.scorecard import fetch_live_scorecard
            scorecard_data = fetch_live_scorecard(game_pk, batter_ids)
            scorecard_html = render_scorecard_section(scorecard_data)
```

Insert `{scorecard_html}` into the HTML template between `{hero}` and the Pick History section header.

- [ ] **Step 3: Add polling script**

Before the closing `</body>` tag in the HTML template, add:

```html
    <script>
    (function() {{
        var date = "{today}";
        var sc = document.getElementById("scorecard");
        if (!sc) return;
        var timer = setInterval(function() {{
            fetch("/api/live?date=" + date)
                .then(function(r) {{ return r.json(); }})
                .then(function(data) {{
                    if (data.game_status === "F") clearInterval(timer);
                    if (data.game_status === "P" || !data.game_status) return;
                    // Full page reload to get server-rendered update
                    location.reload();
                }})
                .catch(function() {{}});
        }}, 30000);
    }})();
    </script>
```

Note: for the MVP, polling triggers a full page reload when game is live. This is simpler than client-side DOM manipulation and consistent with the server-rendered approach. The reload is fast (~200ms for a LAN page). A future optimization could build the scorecard client-side from the JSON.

- [ ] **Step 4: Manual test on Pi5**

Deploy to Pi5 and verify:
```bash
git push origin main
ssh stonehengee@pi5.local "cd ~/projects/bts && git pull origin main && systemctl --user restart bts-dashboard.service"
```

Open `http://192.168.1.185:3003/` — scorecard should appear if a game is Live.
Check `http://192.168.1.185:3003/api/live?date=2026-04-04` for raw JSON.

- [ ] **Step 5: Commit**

```bash
git add src/bts/web.py
git commit -m "feat(dashboard): live scorecard with pitch grid, diamond, trajectory"
```

---

### Task 6: Polish and Edge Cases

**Files:**
- Modify: `src/bts/web.py`
- Modify: `src/bts/scorecard.py`
- Modify: `tests/test_scorecard.py`

- [ ] **Step 1: Handle pre-game state**

In `render_scorecard_section`, when `game_status == "P"`, return empty string (already handled). Verify by checking `/api/live` for a future date.

- [ ] **Step 2: Handle in-progress PA (batter currently at bat)**

Add to `extract_batter_pas`: also check for incomplete plays (`isComplete == False`). If the current batter is one of our picks, show a "live" PA cell with pitches so far but no result. Add test:

```python
def test_in_progress_pa(self):
    """Show live PA when batter is currently at bat."""
    feed = copy.deepcopy(SAMPLE_FEED)
    # Add an incomplete play
    live_play = copy.deepcopy(SAMPLE_PLAY)
    live_play["about"]["isComplete"] = False
    live_play["about"]["atBatIndex"] = 1
    live_play["about"]["inning"] = 4
    live_play["result"] = {"type": "atBat"}
    live_play["playEvents"] = [live_play["playEvents"][0]]  # Just one pitch so far
    live_play["runners"] = []
    feed["liveData"]["plays"]["allPlays"].append(live_play)

    result = extract_batter_pas(feed, {650490})
    batter = result["batters"][0]
    assert len(batter["pas"]) == 2  # 1 complete + 1 in-progress
    assert batter["pas"][1].get("in_progress") is True
```

- [ ] **Step 3: Style the in-progress PA cell**

In `_render_pa_cell`, if `pa.get("in_progress")`, show a pulsing border and "AB" label instead of a result code. Use the pitch grid to show pitches so far.

- [ ] **Step 4: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
Expected: All pass (258 existing + new scorecard tests)

- [ ] **Step 5: Commit**

```bash
git add src/bts/scorecard.py src/bts/web.py tests/test_scorecard.py
git commit -m "feat(scorecard): handle in-progress at-bat, edge cases"
```

---

### Task 7: Deploy and Verify

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Push and deploy to Pi5**

```bash
git push origin main
ssh stonehengee@pi5.local "cd ~/projects/bts && git pull origin main && systemctl --user restart bts-dashboard.service"
```

- [ ] **Step 2: Verify on dashboard**

Open `http://192.168.1.185:3003/` and check:
- Scorecard appears between hero cards and pick history when game is Live
- Pitch grids render correctly (numbered, color-coded)
- Diamonds show baserunning
- Green tint only on hit PAs
- Status banner updates appropriately
- Page auto-refreshes every 30s during live game
- Scorecard hidden when no game is live

- [ ] **Step 3: Update ARCHITECTURE.md**

Add section on live scorecard under Dashboard:
- `scorecard.py` module description
- `/api/live` endpoint
- 30s polling behavior
- Dependency on MLB game feed

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: update ARCHITECTURE.md with live scorecard"
```
