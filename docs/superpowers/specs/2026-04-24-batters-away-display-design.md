# Batters-Away Display — design

**Date**: 2026-04-24
**Author**: Eric + Claude (brainstorm)
**Scope**: `src/bts/scorecard.py`, `src/bts/web.py`, `tests/test_scorecard.py` (extend), `tests/test_web_rendering.py` (new or extend)

## Problem

The dashboard's upcoming-PA cells currently display a heuristic inning estimate (`~5th`, `~7th`) computed from PA index alone (`web.py:494-495`). It's based on `pa_num * 2 - 1`, which doesn't read any actual game state — it can't account for the batter being on deck right now, deep in the order with many outs ahead, or already pulled from the game.

Replace that text with a real "X batters away in the lineup" computation derived from the live feed.

Memory: in-flight observation 2026-04-24 19:51 ET (Yoshida 2 batters away as the 3-hole DH for BOS).

## Decisions (from brainstorming)

| # | Question | Choice |
|---|---|---|
| 1 | What goes on the cell when batter is currently at the plate? | Blank — the in-progress PA cell already shows the pitch grid + count |
| 2 | How granular should the badge text be for distance? | "ON DECK" (1), "IN THE HOLE" (2), "N batters" (3+) |
| 3 | Single OUT or per-cause OUT (PH/sub/injury)? | **Single OUT** — collapses to one consistent treatment |
| 4 | Show on second upcoming cell too? | No — first upcoming only (matches today's behavior, keeps the cell sparse) |
| 5 | Should existing past-PA / in-progress-PA rendering change? | **No.** Constraint: do not alter or deprioritize the scorecard UI (pitch grids, diamond, hit highlighting, AB pulse) |

## Battling-order schema (from MLB Stats API v1.1)

The `battingOrder` per-player STRING is a 3-digit slot+depth code:

- First 1-2 digits = lineup slot (1-9)
- Last digit = substitution depth at that slot (`0` = original starter, `1` = first sub at this slot, `2` = second sub at this slot, …)

Examples observed in `gamePk=824689` (PHI@CHC, 2026-04-23):
- Trea Turner `bo="100"` (slot 1 starter, still in)
- Felix Reyes `bo="400"` (slot 4 starter, replaced — still has battingOrder string but is NOT in current `boxscore.teams.away.battingOrder` array)
- Dylan Moore `bo="401"` (slot 4 first sub, also replaced)
- Rafael Marchán `bo="402"` (slot 4 second sub, currently in lineup array)

The `boxscore.teams.{side}.battingOrder` ARRAY is the **current 9 in the game**. A starter who's been pulled is missing from the array but still has a `battingOrder` string field.

## Status mapping

| API signal | Status | Cell text |
|---|---|---|
| Game not in progress (pre-game / Final / Suspended) | `pre_game` / `final` | *blank* |
| `batter_id == linescore.offense.batter.id` | `at_bat` | *blank* (in-progress PA cell handles it) |
| In current `battingOrder` array, distance 1 from current batter | `on_deck` | **ON DECK** |
| In current array, distance 2 | `in_hole` | **IN THE HOLE** |
| In current array, distance 3-8 | `upcoming` | **N batters** |
| Has a `battingOrder` string AND has batted (PA > 0) AND not in current array | `out_of_game` | **OUT** |
| Has a `battingOrder` string AND has not batted AND not in current array | `out_of_game` | **OUT** |
| No `battingOrder` string at all | `not_in_lineup` | **Not in lineup** |

Distance computed as `(batter_slot - current_slot) % 9` where slots are 1-9 derived from `int(battingOrder_str) // 100`.

## Architecture

```
bts/scorecard.py
├── _slot_from_bo(bo_str: str | None) -> int | None             # NEW
│       parses "402" → 4; returns None on invalid input
├── _compute_lineup_status(...) -> tuple[str, int | None]        # NEW
│       inputs: batter_id, boxscore_team_block, current_batter_id, game_status
│       returns: (status, batters_away)
└── fetch_live_scorecard(...)                                    # MODIFIED
        — extracts current_batter_id from linescore once per fetch
        — for each batter dict in returned scorecard, sets
          .lineup_status (str) and .batters_away (int | None)

bts/web.py
└── _render_pa_cell(pa, lineup_status=None, batters_away=None)   # signature change
        — replaces estimated_inning param
        — only the placeholder branch (pa is None) is touched
        — filled-cell rendering UNCHANGED
└── render_scorecard_section(...)                                # MODIFIED
        — call site no longer computes pa_num heuristic
        — passes batter['lineup_status'] / batter['batters_away']
```

`scorecard.merge_scorecards` already preserves arbitrary fields per batter, so DD-spans-two-games requires no merge-layer change — each game's batter dict carries its own status, computed during fetch from that game's boxscore.

## Component contract

```python
def _compute_lineup_status(
    batter_id: int,
    boxscore_team: dict,        # boxscore.teams.{side} — has "battingOrder" array + "players" dict
    current_batter_id: int | None,  # linescore.offense.batter.id, None if game not active
    game_status: str,           # gameData.status.detailedState
) -> tuple[str, int | None]:
    """Return (lineup_status, batters_away).

    lineup_status ∈ {"pre_game", "final", "at_bat", "on_deck", "in_hole",
                     "upcoming", "out_of_game", "not_in_lineup"}.
    batters_away is int (0 for at_bat, 1 for on_deck, …, 8 for furthest in
    9-batter cycle) or None for non-active states.

    Default-to-blank: anything ambiguous (missing data, unparseable bo string)
    resolves to ("pre_game", None) — the cell renders blank rather than
    showing wrong info.
    """
```

**Render-time mapping (in web.py)**:

```python
if pa is None:
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
    # at_bat / pre_game / final → label stays empty
    # ... render label in the placeholder cell ...
```

## Data flow — four cases

**1. Yoshida tonight @ 19:51 ET (mid-game, on deck or further):**
`current_batter_id` = Duran (slot 1). Yoshida slot 3. `(3-1) % 9 = 2` → `lineup_status="in_hole"`, `batters_away=2`. Cell renders **IN THE HOLE**.

**2. Schwarber 6th inning, just pinch-hit-for:**
Schwarber's `id` not in current `battingOrder` array; he has `battingOrder` string `"200"` and `stats.batting.atBats > 0`. → `lineup_status="out_of_game"`, `batters_away=None`. Cell renders **OUT**.

**3. Pre-first-pitch, lineup announced:**
`game_status = "Pre-Game"` (or any non-"In Progress" state). → `lineup_status="pre_game"`, `batters_away=None`. Cell blank.

**4. Game over, all PAs final:**
`game_status = "Final"` → `lineup_status="final"`, `batters_away=None`. Upcoming-PA cells blank.

**Key invariant:** the `_render_pa_cell` placeholder branch only renders text when `lineup_status` is one of the four "live, actionable" states. All other states (including ambiguous/missing-data fallback) render blank — the cell looks empty rather than showing wrong copy.

## Error handling

- **Live feed payload missing fields** (e.g., `linescore.offense.batter` not present): treat as `current_batter_id = None` → `pre_game` status → blank cell. No crash.
- **Batter ID not found in boxscore.players**: not_in_lineup. Renders **Not in lineup**.
- **Malformed `battingOrder` string** (anything `_slot_from_bo` can't parse to int 1-9): treat as if missing → `not_in_lineup` (consistent with default-to-blank-or-clear-error principle).
- **Game DD spans two games**: each game contributes its own scorecard payload; `merge_scorecards` carries through `lineup_status` + `batters_away` per batter unchanged.

## Testing

**TDD flow**: tests first in `tests/test_scorecard.py`, helper impl, then web-render unit test.

### Unit tests for `_compute_lineup_status`

| # | Scenario | Asserts |
|---|---|---|
| 1 | Game not in progress (`status="Pre-Game"`) | `("pre_game", None)` |
| 2 | Game Final | `("final", None)` |
| 3 | Batter is at the plate | `("at_bat", 0)` |
| 4 | Batter is 1 slot ahead of current | `("on_deck", 1)` |
| 5 | Batter is 2 slots ahead | `("in_hole", 2)` |
| 6 | Batter is 3 slots ahead | `("upcoming", 3)` |
| 7 | Batter is 8 slots ahead (full cycle minus one) | `("upcoming", 8)` |
| 8 | Wraparound: current slot 8, batter slot 1 → distance = (1-8) % 9 = 2 | `("in_hole", 2)` |
| 9 | Pulled batter — has bo string, has AB, NOT in current array | `("out_of_game", None)` |
| 10 | Never-in-lineup — no bo string | `("not_in_lineup", None)` |
| 11 | Pulled batter with 0 AB (defensive sub before any PA) | `("out_of_game", None)` |
| 12 | Malformed bo string (e.g., `"abc"`) | `("not_in_lineup", None)` |
| 13 | `current_batter_id=None` while game is "In Progress" (ambiguous) | `("pre_game", None)` — defensive default |

### Unit tests for `_slot_from_bo`

| # | Input | Output |
|---|---|---|
| 14 | `"100"` | `1` |
| 15 | `"402"` | `4` |
| 16 | `"902"` | `9` |
| 17 | `None` | `None` |
| 18 | `""` | `None` |
| 19 | `"abc"` | `None` |

### Integration test for `fetch_live_scorecard`

| # | Scenario | Asserts |
|---|---|---|
| 20 | Mock live feed with batter 2 slots away from current → returned scorecard's batter dict has `lineup_status="in_hole"`, `batters_away=2` |

### Render test for `_render_pa_cell` placeholder branch

| # | Scenario | Asserts |
|---|---|---|
| 21 | `pa=None, lineup_status="on_deck"` | rendered HTML contains "ON DECK" |
| 22 | `pa=None, lineup_status="in_hole"` | rendered HTML contains "IN THE HOLE" |
| 23 | `pa=None, lineup_status="upcoming", batters_away=5` | contains "5 batters" |
| 24 | `pa=None, lineup_status="out_of_game"` | contains "OUT" |
| 25 | `pa=None, lineup_status="not_in_lineup"` | contains "Not in lineup" |
| 26 | `pa=None, lineup_status="at_bat"` | placeholder cell with NO text label |
| 27 | `pa=None, lineup_status="pre_game"` | placeholder cell with NO text label |
| 28 | `pa=None, lineup_status="final"` | placeholder cell with NO text label |

### Filled-cell regression check

| # | Scenario | Asserts |
|---|---|---|
| 29 | `pa=<filled hit dict>` (any lineup_status) | renders pitch grid + green hit highlight unchanged from current behavior |
| 30 | `pa=<in-progress dict>` (any lineup_status) | renders amber pulse + AB count unchanged |

**Total: ~30 new tests.** Existing 612 tests must continue to pass.

## Out of scope (explicitly deferred)

- Showing a per-cause OUT label ("OUT — pinch-hit for", "OUT — defensive sub") — collapsed to single OUT per Q3.
- Showing the batters-away badge on the SECOND upcoming PA cell (lap-around) — keeping today's "first upcoming only" behavior per Q4.
- Rerendering or restyling past-PA / in-progress-PA cells — explicitly NOT touched per Q5.
- Live-poll cadence changes (the dashboard's existing 30s `/api/live-html` poll picks up the new info automatically).

## Rollback

Single-revert: change `_render_pa_cell` placeholder branch back to consuming `estimated_inning` and re-add the `pa_num * 2 - 1` heuristic at the call site. The new `_compute_lineup_status` and `_slot_from_bo` helpers can stay (unused) or get removed in a follow-up. No on-disk data shape changes.
