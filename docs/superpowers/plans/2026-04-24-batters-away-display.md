# Batters-Away Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's heuristic upcoming-PA inning text (`~5th`) with a real lineup-distance signal computed from the live MLB feed (`ON DECK`, `IN THE HOLE`, `N batters`, `OUT`, `Not in lineup`, or blank).

**Architecture:** Two new pure helpers in `src/bts/scorecard.py` (`_slot_from_bo`, `_compute_lineup_status`); extend `extract_batter_pas` to populate `lineup_status` + `batters_away` per batter; modify only the placeholder branch of `_render_pa_cell` in `src/bts/web.py` and its single call site. Filled-cell rendering (pitch grid, AB pulse, hit highlight) is explicitly NOT touched.

**Tech Stack:** Python 3.12, pytest 9.x.

**Spec:** `docs/superpowers/specs/2026-04-24-batters-away-display-design.md` (commits `aaaa5d8` + `d93fa75`).

**Branch:** Continue on `main`, no worktree (matches this session's pattern; change is additive, unit-tested, and the dashboard's existing 30s `/api/live-html` poll picks up the new fields automatically once deployed).

---

## File Structure

- **Modify** `src/bts/scorecard.py`:
  - NEW: `_slot_from_bo(bo_str) -> int | None` — parse "402" → 4
  - NEW: `_compute_lineup_status(batter_id, boxscore_team, current_batter_id, game_status) -> (status, batters_away)`
  - MODIFY: `extract_batter_pas` — extract `current_batter_id`, build `batter_side` map, populate `lineup_status` + `batters_away` per batter dict
  - UNCHANGED: `merge_scorecards` (generic field carry-through is sufficient)
- **Modify** `src/bts/web.py`:
  - MODIFY: `_render_pa_cell` signature: replace `estimated_inning: str = ""` with `lineup_status: str | None = None, batters_away: int | None = None`; update placeholder branch only
  - MODIFY: `render_scorecard_section` call site (web.py:486-498): drop heuristic, pass batter's `lineup_status` + `batters_away`
- **Modify** `tests/test_scorecard.py`: append new test classes
- **Create** `tests/test_web_render.py`: new file for `_render_pa_cell` placeholder + filled-cell precedence tests

---

### Task 1: Test `_slot_from_bo` parser

**Files:**
- Modify: `tests/test_scorecard.py` — append at end of file

- [ ] **Step 1: Append failing tests**

```python


# ---------------------------------------------------------------------------
# Lineup-status helpers tests (added 2026-04-24)
# ---------------------------------------------------------------------------

class TestSlotFromBo:
    def test_starter_slot_1(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("100") == 1

    def test_first_sub_slot_4(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("401") == 4

    def test_second_sub_slot_9(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("902") == 9

    def test_none_input(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo(None) is None

    def test_empty_string(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("") is None

    def test_malformed_letters(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("abc") is None

    def test_out_of_range_zero(self):
        from bts.scorecard import _slot_from_bo
        # "000" → slot 0, which isn't valid (slots are 1-9)
        assert _slot_from_bo("000") is None

    def test_out_of_range_high(self):
        from bts.scorecard import _slot_from_bo
        # Hypothetical "1000" → slot 10, invalid
        assert _slot_from_bo("1000") is None
```

- [ ] **Step 2: Run and verify RED**

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestSlotFromBo -v
```

Expected: 8 failed with `ImportError: cannot import name '_slot_from_bo'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scorecard.py
git commit -m "test(scorecard): _slot_from_bo parser cases (RED)

8 cases covering valid slot/depth strings + None/empty/malformed/
out-of-range fallbacks. Expect RED until helper exists."
```

---

### Task 2: Test `_compute_lineup_status` (live + edge cases)

**Files:**
- Modify: `tests/test_scorecard.py` — append after `TestSlotFromBo`

- [ ] **Step 1: Append failing tests**

```python


def _mk_team(battingOrder: list[int], players: dict) -> dict:
    """Helper to build a boxscore_team block from a lineup + per-player data.

    `players` is {batter_id: {"battingOrder": "100", "stats": {"batting": {"atBats": 4}}}}.
    """
    return {
        "battingOrder": list(battingOrder),
        "players": {
            f"ID{pid}": {
                "person": {"id": pid, "fullName": f"player_{pid}"},
                "battingOrder": data.get("battingOrder", "0"),
                "stats": {"batting": {"atBats": data.get("atBats", 0)}},
            }
            for pid, data in players.items()
        },
    }


class TestComputeLineupStatus:
    def test_pre_game(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team([1, 2, 3], {1: {"battingOrder": "100"}})
        status, away = _compute_lineup_status(1, team, current_batter_id=None, game_status="P")
        assert status == "pre_game"
        assert away is None

    def test_final_game(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team([1, 2, 3], {1: {"battingOrder": "100", "atBats": 4}})
        status, away = _compute_lineup_status(1, team, current_batter_id=None, game_status="F")
        assert status == "final"
        assert away is None

    def test_at_bat(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {
                1: {"battingOrder": "100"},
                2: {"battingOrder": "200"},
                3: {"battingOrder": "300"},
            },
        )
        status, away = _compute_lineup_status(2, team, current_batter_id=2, game_status="L")
        assert status == "at_bat"
        assert away == 0

    def test_on_deck(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            {i: {"battingOrder": f"{i}00"} for i in range(1, 10)},
        )
        # Current is slot 1, batter is slot 2 → on deck
        status, away = _compute_lineup_status(2, team, current_batter_id=1, game_status="L")
        assert status == "on_deck"
        assert away == 1

    def test_in_hole(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            {i: {"battingOrder": f"{i}00"} for i in range(1, 10)},
        )
        status, away = _compute_lineup_status(3, team, current_batter_id=1, game_status="L")
        assert status == "in_hole"
        assert away == 2

    def test_upcoming_distance_3(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            {i: {"battingOrder": f"{i}00"} for i in range(1, 10)},
        )
        status, away = _compute_lineup_status(4, team, current_batter_id=1, game_status="L")
        assert status == "upcoming"
        assert away == 3

    def test_upcoming_distance_8(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            {i: {"battingOrder": f"{i}00"} for i in range(1, 10)},
        )
        status, away = _compute_lineup_status(9, team, current_batter_id=1, game_status="L")
        assert status == "upcoming"
        assert away == 8

    def test_wraparound_current_slot_8_batter_slot_1(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            {i: {"battingOrder": f"{i}00"} for i in range(1, 10)},
        )
        # current=slot 8, batter=slot 1 → (1-8) % 9 = 2
        status, away = _compute_lineup_status(1, team, current_batter_id=8, game_status="L")
        assert status == "in_hole"
        assert away == 2

    def test_out_of_game_pulled_starter(self):
        from bts.scorecard import _compute_lineup_status
        # Slot 4 starter (id=4) was pulled. Replacement (id=44) now in array.
        team = _mk_team(
            [1, 2, 3, 44, 5, 6, 7, 8, 9],
            {
                1: {"battingOrder": "100"},
                2: {"battingOrder": "200"},
                3: {"battingOrder": "300"},
                4: {"battingOrder": "400", "atBats": 2},  # pulled, has bo string + ABs
                44: {"battingOrder": "401"},
                5: {"battingOrder": "500"},
                6: {"battingOrder": "600"},
                7: {"battingOrder": "700"},
                8: {"battingOrder": "800"},
                9: {"battingOrder": "900"},
            },
        )
        status, away = _compute_lineup_status(4, team, current_batter_id=1, game_status="L")
        assert status == "out_of_game"
        assert away is None

    def test_out_of_game_pulled_zero_ab(self):
        from bts.scorecard import _compute_lineup_status
        # Pulled before any PA (defensive sub immediately).
        team = _mk_team(
            [1, 2, 3, 44, 5, 6, 7, 8, 9],
            {
                4: {"battingOrder": "400", "atBats": 0},
                44: {"battingOrder": "401"},
            },
        )
        status, away = _compute_lineup_status(4, team, current_batter_id=1, game_status="L")
        assert status == "out_of_game"
        assert away is None

    def test_not_in_lineup_no_bo_string(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {
                1: {"battingOrder": "100"},
                99: {},  # in players dict but no battingOrder string
            },
        )
        status, away = _compute_lineup_status(99, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_malformed_bo_string_treats_as_not_in_lineup(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {
                1: {"battingOrder": "100"},
                99: {"battingOrder": "abc"},
            },
        )
        status, away = _compute_lineup_status(99, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_current_batter_none_during_live_game(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {1: {"battingOrder": "100"}, 2: {"battingOrder": "200"}, 3: {"battingOrder": "300"}},
        )
        # Ambiguous: live game but no current batter resolved → defensive default
        status, away = _compute_lineup_status(2, team, current_batter_id=None, game_status="L")
        assert status == "pre_game"
        assert away is None

    def test_missing_battingOrder_array(self):
        from bts.scorecard import _compute_lineup_status
        team = {"players": {"ID1": {"battingOrder": "100"}}}  # no battingOrder array key
        status, away = _compute_lineup_status(1, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_missing_player_key(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team([1, 2, 3], {1: {"battingOrder": "100"}})
        # Asking for batter 99 who isn't in players dict
        status, away = _compute_lineup_status(99, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_current_batter_id_unknown_in_players(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {1: {"battingOrder": "100"}, 2: {"battingOrder": "200"}, 3: {"battingOrder": "300"}},
        )
        # current_batter_id=999 isn't in this team's players (race during sub)
        status, away = _compute_lineup_status(2, team, current_batter_id=999, game_status="L")
        assert status == "pre_game"
        assert away is None
```

- [ ] **Step 2: Run and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestComputeLineupStatus -v
```

Expected: 16 failed with `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scorecard.py
git commit -m "test(scorecard): _compute_lineup_status 16 cases (RED)

5 distance variants (at_bat through upcoming-8), wraparound,
out_of_game (with/without ABs), not_in_lineup, malformed bo,
3 defensive fallbacks for missing data. Expect RED until helper
exists."
```

---

### Task 3: Implement both helpers

**Files:**
- Modify: `src/bts/scorecard.py` — insert after `_extract_fielder_position` (around line 92)

- [ ] **Step 1: Insert both helpers**

Find the line `def _extract_fielder_position(...)` (around line 74) and after its closing definition (around line 92, before `def _extract_pa`), insert:

```python


def _slot_from_bo(bo_str: str | None) -> int | None:
    """Parse a battingOrder string like "402" into a 1-9 lineup slot.

    The 3-digit code is `slot * 100 + depth` where depth is 0 for the
    original starter, 1 for the first sub at that slot, etc. We only
    need the slot, so integer-divide by 100.

    Returns None for missing, empty, malformed, or out-of-range inputs.
    """
    if not bo_str:
        return None
    try:
        slot = int(bo_str) // 100
    except (ValueError, TypeError):
        return None
    if slot < 1 or slot > 9:
        return None
    return slot


def _compute_lineup_status(
    batter_id: int,
    boxscore_team: dict,
    current_batter_id: int | None,
    game_status: str,
) -> tuple[str, int | None]:
    """Return (lineup_status, batters_away) for a picked batter.

    lineup_status ∈ {"pre_game", "final", "at_bat", "on_deck", "in_hole",
                     "upcoming", "out_of_game", "not_in_lineup"}.
    batters_away is 0 for at_bat, 1 for on_deck, ..., 8 for max distance,
    or None for non-active states.

    Defensive default: anything ambiguous (missing data, unparseable bo,
    unknown current batter during live) resolves to ("pre_game", None) or
    ("not_in_lineup", None) so the cell renders blank rather than wrong.
    """
    # Game not in progress
    if game_status == "P":
        return ("pre_game", None)
    if game_status == "F":
        return ("final", None)
    if game_status != "L":
        return ("pre_game", None)  # Suspended / Delayed / unknown — blank

    # Live game from here on
    players = boxscore_team.get("players", {})
    batter_entry = players.get(f"ID{batter_id}", {})
    bo_str = batter_entry.get("battingOrder")
    batter_slot = _slot_from_bo(bo_str)

    # No bo string OR malformed → never in this team's lineup
    if batter_slot is None:
        return ("not_in_lineup", None)

    current_array = boxscore_team.get("battingOrder")
    if not isinstance(current_array, list):
        return ("not_in_lineup", None)

    if batter_id not in current_array:
        return ("out_of_game", None)

    # Batter is in current lineup. Compute distance to current_batter.
    if current_batter_id is None:
        return ("pre_game", None)  # Can't compute distance without a current

    current_entry = players.get(f"ID{current_batter_id}", {})
    current_slot = _slot_from_bo(current_entry.get("battingOrder"))
    if current_slot is None:
        return ("pre_game", None)  # Unknown current — bail to blank

    distance = (batter_slot - current_slot) % 9
    if distance == 0:
        return ("at_bat", 0)
    if distance == 1:
        return ("on_deck", 1)
    if distance == 2:
        return ("in_hole", 2)
    return ("upcoming", distance)
```

- [ ] **Step 2: Run new tests, verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestSlotFromBo tests/test_scorecard.py::TestComputeLineupStatus -v
```

Expected: 24 passed (8 + 16).

- [ ] **Step 3: Run full scorecard test file (regression check)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v
```

Expected: 44 + 24 = **68 passed**, no failures or errors.

- [ ] **Step 4: Commit**

```bash
git add src/bts/scorecard.py
git commit -m "feat(scorecard): _slot_from_bo + _compute_lineup_status helpers

_slot_from_bo parses MLB Stats API battingOrder strings (3-digit
slot+depth code) into a 1-9 lineup slot. None on malformed input.

_compute_lineup_status maps a (batter_id, boxscore_team, current_id,
game_status) tuple to one of 8 status states + an optional
batters_away int. Defensive default to blank-render states for
any ambiguous input.

24 unit tests passing; full scorecard suite 68 passing."
```

---

### Task 4: Wire into `extract_batter_pas`

**Files:**
- Modify: `src/bts/scorecard.py:200-321`
- Modify: `tests/test_scorecard.py` — append integration test

- [ ] **Step 1: Append failing integration test**

Append to `tests/test_scorecard.py` after `TestComputeLineupStatus`:

```python


class TestExtractBatterPasLineupStatus:
    def test_lineup_status_populated_in_returned_batter_dict(self):
        """When extract_batter_pas processes a live feed, each returned
        batter dict carries lineup_status + batters_away derived from the
        live feed's current batter and the boxscore.
        """
        from bts.scorecard import extract_batter_pas

        # Synthetic feed: live game, batter id=2 in slot 2; current batter is id=1 in slot 1.
        # Distance = (2-1) % 9 = 1 → on_deck.
        feed = {
            "gameData": {
                "status": {"abstractGameCode": "L"},
                "teams": {
                    "away": {"abbreviation": "BOS"},
                    "home": {"abbreviation": "BAL"},
                },
            },
            "liveData": {
                "linescore": {
                    "currentInning": 1,
                    "inningHalf": "Top",
                    "teams": {"away": {"runs": 0}, "home": {"runs": 0}},
                    "offense": {"batter": {"id": 1}},
                },
                "boxscore": {
                    "teams": {
                        "away": {
                            "battingOrder": [1, 2, 3],
                            "players": {
                                "ID1": {
                                    "person": {"id": 1, "fullName": "Alpha"},
                                    "battingOrder": "100",
                                    "position": {"abbreviation": "CF"},
                                    "stats": {"batting": {"atBats": 0}},
                                },
                                "ID2": {
                                    "person": {"id": 2, "fullName": "Beta"},
                                    "battingOrder": "200",
                                    "position": {"abbreviation": "DH"},
                                    "stats": {"batting": {"atBats": 0}},
                                },
                                "ID3": {
                                    "person": {"id": 3, "fullName": "Gamma"},
                                    "battingOrder": "300",
                                    "position": {"abbreviation": "1B"},
                                    "stats": {"batting": {"atBats": 0}},
                                },
                            },
                        },
                        "home": {"battingOrder": [], "players": {}},
                    }
                },
                "plays": {"allPlays": []},
            },
        }
        result = extract_batter_pas(feed, batter_ids={2})
        assert len(result["batters"]) == 1
        b = result["batters"][0]
        assert b["batter_id"] == 2
        assert b["lineup_status"] == "on_deck"
        assert b["batters_away"] == 1
```

- [ ] **Step 2: Run, verify RED with KeyError or AssertionError**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestExtractBatterPasLineupStatus -v
```

Expected: FAIL with `KeyError: 'lineup_status'`.

- [ ] **Step 3: Modify `extract_batter_pas` to populate the new fields**

In `src/bts/scorecard.py:extract_batter_pas`:

(a) After line 226 (`inning_str = ...`), insert extraction of current batter id:

```python
    # Current batter (for lineup-distance computation)
    current_batter_id = (
        linescore.get("offense", {}).get("batter", {}).get("id")
    )
```

(b) Replace lines 235-243 (the boxscore_teams + player_lookup loop) with this version that ALSO tracks which side each batter is on:

Old:
```python
    # Build boxscore player lookup: batter_id → player entry
    boxscore_teams = live_data.get("boxscore", {}).get("teams", {})
    player_lookup: dict[int, dict] = {}
    for side in ("away", "home"):
        players = boxscore_teams.get(side, {}).get("players", {})
        for key, player_data in players.items():
            pid = player_data.get("person", {}).get("id")
            if pid is not None:
                player_lookup[pid] = player_data
```

New:
```python
    # Build boxscore player lookup: batter_id → player entry, and side mapping
    boxscore_teams = live_data.get("boxscore", {}).get("teams", {})
    player_lookup: dict[int, dict] = {}
    batter_side: dict[int, str] = {}
    for side in ("away", "home"):
        players = boxscore_teams.get(side, {}).get("players", {})
        for key, player_data in players.items():
            pid = player_data.get("person", {}).get("id")
            if pid is not None:
                player_lookup[pid] = player_data
                batter_side[pid] = side
```

(c) In the per-batter loop (lines 273-309), after computing `lineup_position` and before constructing the batter dict, insert lineup status computation:

Find the line `position = player_entry.get("position", {}).get("abbreviation", "")` (around line 292) and replace the block from there through the `batters.append(...)` call (lines 292-309) with:

```python
        position = player_entry.get("position", {}).get("abbreviation", "")
        name = person.get("fullName", "")

        # bat_side from first completed play; fall back to empty string if no PAs yet
        first_play = batter_first_play.get(batter_id)
        bat_side = first_play.get("matchup", {}).get("batSide", {}).get("code", "") if first_play else ""

        # Lineup status (distance from current batter, OUT, etc.)
        side_for_batter = batter_side.get(batter_id)
        if side_for_batter:
            lineup_status, batters_away = _compute_lineup_status(
                batter_id,
                boxscore_teams.get(side_for_batter, {}),
                current_batter_id,
                game_status,
            )
        else:
            lineup_status, batters_away = "not_in_lineup", None

        batters.append(
            {
                "batter_id": batter_id,
                "name": name,
                "position": position,
                "lineup_position": lineup_position,
                "batting_hand": bat_side,
                "slash_line": slash_line,
                "pas": batter_pas.get(batter_id, []),
                "lineup_status": lineup_status,
                "batters_away": batters_away,
            }
        )
```

- [ ] **Step 4: Run and verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py -v 2>&1 | tail -10
```

Expected: 69 passed (44 existing + 24 helpers + 1 integration).

- [ ] **Step 5: Commit**

```bash
git add src/bts/scorecard.py tests/test_scorecard.py
git commit -m "feat(scorecard): populate lineup_status + batters_away in extract_batter_pas

Each returned batter dict now carries lineup_status (one of 8 states)
and batters_away (int or None). Wired in by computing per-batter
during the existing batters-list build loop, using the new
_compute_lineup_status helper. batter_side map tracks which team
each picked batter is on so the right boxscore_team block is passed.

69 tests passing in tests/test_scorecard.py."
```

---

### Task 5: Verify `merge_scorecards` preserves new fields

**Files:**
- Modify: `tests/test_scorecard.py` — append regression test

- [ ] **Step 1: Append regression test**

Append to `tests/test_scorecard.py` after `TestExtractBatterPasLineupStatus`:

```python


class TestMergeScorecardsLineupStatus:
    def test_merge_preserves_lineup_status_per_batter(self):
        """DD-spans-two-games: each batter's lineup_status survives merge."""
        from bts.scorecard import merge_scorecards

        sc1 = {
            "game_status": "L",
            "inning": "Top 3",
            "away_team": "BOS",
            "home_team": "BAL",
            "score": {"away": 1, "home": 5},
            "batters": [
                {
                    "batter_id": 100,
                    "name": "Yoshida",
                    "position": "DH",
                    "lineup_position": 3,
                    "batting_hand": "L",
                    "slash_line": ".280/.350/.450",
                    "pas": [],
                    "lineup_status": "on_deck",
                    "batters_away": 1,
                }
            ],
        }
        sc2 = {
            "game_status": "L",
            "inning": "Top 4",
            "away_team": "PHI",
            "home_team": "ATL",
            "score": {"away": 2, "home": 0},
            "batters": [
                {
                    "batter_id": 200,
                    "name": "Turner",
                    "position": "SS",
                    "lineup_position": 1,
                    "batting_hand": "R",
                    "slash_line": ".310/.380/.520",
                    "pas": [],
                    "lineup_status": "upcoming",
                    "batters_away": 5,
                }
            ],
        }
        merged = merge_scorecards(sc1, sc2)
        assert merged is not None
        names_to_status = {b["name"]: b for b in merged["batters"]}
        assert names_to_status["Yoshida"]["lineup_status"] == "on_deck"
        assert names_to_status["Yoshida"]["batters_away"] == 1
        assert names_to_status["Turner"]["lineup_status"] == "upcoming"
        assert names_to_status["Turner"]["batters_away"] == 5
```

- [ ] **Step 2: Run — likely already GREEN since merge_scorecards passes batter dicts unchanged**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_scorecard.py::TestMergeScorecardsLineupStatus -v
```

Expected: 1 passed. (If FAIL: investigate `merge_scorecards` for any field-stripping logic and add carry-through.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_scorecard.py
git commit -m "test(scorecard): merge_scorecards preserves lineup_status per batter

Regression test for the DD-spans-two-games path. merge_scorecards
is generic enough to carry arbitrary batter fields through, so this
should pass without code change — but explicit assertion guards
against future merge-layer changes that might strip new fields."
```

---

### Task 6: Test `_render_pa_cell` placeholder branch

**Files:**
- Create: `tests/test_web_render.py`

- [ ] **Step 1: Create new test file**

Write the file `tests/test_web_render.py`:

```python
"""Tests for bts.web rendering helpers — focused on _render_pa_cell.

Separated from test_web_audit_progress.py (which is scoped to the audit
endpoint) to avoid mixing concerns.
"""
from __future__ import annotations

import pytest

from bts.web import _render_pa_cell


class TestRenderPaCellPlaceholder:
    """Placeholder branch (pa is None) — driven by lineup_status + batters_away."""

    def test_on_deck(self):
        html = _render_pa_cell(None, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" in html

    def test_in_hole(self):
        html = _render_pa_cell(None, lineup_status="in_hole", batters_away=2)
        assert "IN THE HOLE" in html

    def test_upcoming_distance_5(self):
        html = _render_pa_cell(None, lineup_status="upcoming", batters_away=5)
        assert "5 batters" in html

    def test_out_of_game(self):
        html = _render_pa_cell(None, lineup_status="out_of_game")
        assert "OUT" in html

    def test_not_in_lineup(self):
        html = _render_pa_cell(None, lineup_status="not_in_lineup")
        assert "Not in lineup" in html

    def test_at_bat_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="at_bat", batters_away=0)
        # No status text in the placeholder for at_bat — the in-progress PA cell handles it
        assert "ON DECK" not in html
        assert "OUT" not in html
        assert "batters" not in html

    def test_pre_game_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="pre_game")
        assert "ON DECK" not in html
        assert "OUT" not in html

    def test_final_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="final")
        assert "ON DECK" not in html
        assert "OUT" not in html

    def test_default_args_render_blank(self):
        # Backward compat: calling _render_pa_cell(None) with no kwargs renders blank
        html = _render_pa_cell(None)
        assert "ON DECK" not in html
        assert "OUT" not in html


class TestRenderPaCellFilledPrecedence:
    """Filled-cell branch must IGNORE lineup_status / batters_away args."""

    def test_filled_hit_ignores_lineup_status(self):
        pa = {
            "result": "Single",
            "is_hit": True,
            "rbi": 0,
            "pitches": [],
            "in_progress": False,
        }
        # Even with lineup_status="on_deck", the filled-cell rendering wins
        html = _render_pa_cell(pa, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" not in html
        assert "Single" in html

    def test_filled_out_ignores_lineup_status(self):
        pa = {
            "result": "Strikeout",
            "is_hit": False,
            "out_number": 1,
            "rbi": 0,
            "pitches": [],
            "in_progress": False,
        }
        html = _render_pa_cell(pa, lineup_status="out_of_game")
        assert "OUT" not in html or "Strikeout" in html  # at most "OUT" appears as part of strikeout
        # More precise: filled-cell rendering doesn't add the placeholder OUT badge
        assert ">OUT<" not in html

    def test_in_progress_pa_ignores_lineup_status(self):
        pa = {
            "in_progress": True,
            "pitches": [{"is_strike": True}, {"is_strike": False}],
        }
        html = _render_pa_cell(pa, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" not in html
        assert "AB" in html  # in-progress AB marker
```

- [ ] **Step 2: Run, verify RED with TypeError**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_render.py -v
```

Expected: 12 failed with `TypeError: _render_pa_cell() got an unexpected keyword argument 'lineup_status'`.

- [ ] **Step 3: Commit (red state)**

```bash
git add tests/test_web_render.py
git commit -m "test(web): _render_pa_cell tests — placeholder + filled precedence (RED)

12 tests across two classes:
- TestRenderPaCellPlaceholder: 9 cases for the placeholder branch
  driven by lineup_status (on_deck → ON DECK, in_hole → IN THE HOLE,
  upcoming → N batters, out_of_game → OUT, not_in_lineup, at_bat,
  pre_game, final, default args)
- TestRenderPaCellFilledPrecedence: 3 regression cases ensuring
  filled-cell rendering ignores lineup_status args

Expect RED until _render_pa_cell signature is updated."
```

---

### Task 7: Update `_render_pa_cell` signature + call site

**Files:**
- Modify: `src/bts/web.py:321-336` (signature + placeholder branch)
- Modify: `src/bts/web.py:486-498` (call site in render_scorecard_section)

- [ ] **Step 1: Update `_render_pa_cell` signature + placeholder branch**

In `src/bts/web.py`, find the function definition at line 321:

```python
def _render_pa_cell(pa: dict | None, estimated_inning: str = "") -> str:
    """Render a single plate appearance as a <td> element."""
    if pa is None:
        # Upcoming PA placeholder
        style = (
            "border:1px dashed #ccc;color:#bbb;font-size:10px;"
            "vertical-align:top;padding:4px;width:100px;min-width:100px;"
            "text-align:center;"
        )
        inner = ""
        if estimated_inning:
            inner = (
                f'<div style="font-size:9px;color:#bbb;margin-top:4px;">'
                f'{estimated_inning}</div>'
            )
        return f'<td style="{style}">{inner}</td>'
```

Replace with:

```python
def _render_pa_cell(
    pa: dict | None,
    lineup_status: str | None = None,
    batters_away: int | None = None,
) -> str:
    """Render a single plate appearance as a <td> element.

    PRECEDENCE: when `pa` is provided (filled or in-progress cell), the
    `lineup_status` and `batters_away` arguments are IGNORED — filled cells
    own their own visual treatment (pitch grid, AB pulse, hit highlight).
    The placeholder branch (pa is None) is the only consumer of these args.
    """
    if pa is None:
        # Upcoming PA placeholder
        style = (
            "border:1px dashed #ccc;color:#bbb;font-size:10px;"
            "vertical-align:top;padding:4px;width:100px;min-width:100px;"
            "text-align:center;"
        )
        label = ""
        if lineup_status == "on_deck":
            label = "ON DECK"
        elif lineup_status == "in_hole":
            label = "IN THE HOLE"
        elif lineup_status == "upcoming" and batters_away is not None:
            label = f"{batters_away} batters"
        elif lineup_status == "out_of_game":
            label = "OUT"
        elif lineup_status == "not_in_lineup":
            label = "Not in lineup"
        # at_bat / pre_game / final / None → label stays empty
        inner = ""
        if label:
            inner = (
                f'<div style="font-size:9px;color:#bbb;margin-top:4px;">'
                f'{label}</div>'
            )
        return f'<td style="{style}">{inner}</td>'
```

- [ ] **Step 2: Update call site in `render_scorecard_section`**

In `src/bts/web.py`, find the loop around line 486-498:

```python
        row_cells = ""
        for i in range(num_pa_cols):
            if i < len(pas):
                row_cells += _render_pa_cell(pas[i])
            else:
                # Estimate which inning this PA would occur in.
                # Each batter gets ~1 PA per 9 batters through the order.
                # First upcoming PA: current completed PAs + 1 → inning ≈ (pa_num) * 2 - 1
                pa_num = i + 1
                est_inning = pa_num * 2 - 1 if pa_num <= 5 else pa_num * 2
                est = f"~{_ordinal(est_inning)}" if i == len(pas) else ""
                row_cells += _render_pa_cell(None, estimated_inning=est)
```

Replace with:

```python
        row_cells = ""
        for i in range(num_pa_cols):
            if i < len(pas):
                row_cells += _render_pa_cell(pas[i])
            elif i == len(pas):
                # First upcoming PA: render lineup-distance badge from live data
                row_cells += _render_pa_cell(
                    None,
                    lineup_status=batter.get("lineup_status"),
                    batters_away=batter.get("batters_away"),
                )
            else:
                # Subsequent upcoming PA cells stay blank
                row_cells += _render_pa_cell(None)
```

- [ ] **Step 3: Run new render tests, verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_render.py -v
```

Expected: 12 passed.

- [ ] **Step 4: Run full test suite — expect 612 + 24 + 1 + 1 + 12 = 650 passing**

Wait — full count check:
- Existing baseline (before this plan): 612
- Task 1 added: 8 (`TestSlotFromBo`)
- Task 2 added: 16 (`TestComputeLineupStatus`)
- Task 4 added: 1 (`TestExtractBatterPasLineupStatus`)
- Task 5 added: 1 (`TestMergeScorecardsLineupStatus`)
- Task 6 added: 12 (`TestRenderPaCellPlaceholder` 9 + `TestRenderPaCellFilledPrecedence` 3)
- Total: 612 + 8 + 16 + 1 + 1 + 12 = **650**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest 2>&1 | tail -3
```

Expected: `650 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/bts/web.py
git commit -m "feat(web): _render_pa_cell consumes lineup_status + batters_away

Replaces the heuristic ~Nth inning badge in the upcoming-PA cell
with real lineup-distance copy: ON DECK / IN THE HOLE / N batters /
OUT / Not in lineup / blank.

Filled-cell rendering (pitch grid, hit highlight, in-progress AB
pulse) is unchanged — the new args are only consumed in the
placeholder branch (pa is None). Render is idempotent in the
filled case: passing lineup_status alongside a filled pa is
ignored.

render_scorecard_section call site updated: drops pa_num * 2 - 1
heuristic; passes batter['lineup_status'] + batter['batters_away']
from the scorecard payload (populated by extract_batter_pas).

Full suite: 650 passing."
```

---

## Self-Review Checklist (for the implementing engineer)

After Task 7 commits:

- [ ] **Spec coverage**: walk the spec's status-mapping table and verify each row has a matching test.
  - `pre_game`/`final`: TestComputeLineupStatus #1, #2 + render tests #7, #8
  - `at_bat`: TestComputeLineupStatus #3 + render test #6
  - `on_deck`: TestComputeLineupStatus #4 + render test #1
  - `in_hole`: TestComputeLineupStatus #5 + render test #2
  - `upcoming`: TestComputeLineupStatus #6, #7 + render test #3
  - `out_of_game`: TestComputeLineupStatus #9, #10 + render test #4
  - `not_in_lineup`: TestComputeLineupStatus #11, #12 + render test #5
- [ ] **Defensive fallbacks**: cases 13a, 13b, 13c all in TestComputeLineupStatus.
- [ ] **Filled-cell precedence**: TestRenderPaCellFilledPrecedence covers it (3 cases).
- [ ] **Merge preservation**: TestMergeScorecardsLineupStatus covers it.
- [ ] **All 38 new tests pass** (8 + 16 + 1 + 1 + 12).
- [ ] **Full suite: 650 passing**.
- [ ] **No stray placeholder strings** (`TODO`, `FIXME`, `XXX`) in code.

## Rollback

Revert is two steps:
1. `git revert` the call-site commit + the signature commit (Task 7).
2. Optionally revert the helpers + extract_batter_pas wiring (Tasks 3-5) if you also want to drop the new payload fields.

`merge_scorecards` is unchanged so no rollback needed there. No on-disk data shape changes.
