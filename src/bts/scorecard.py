"""Live scorecard data extraction from MLB game feed."""

from __future__ import annotations

import json

from bts.data.schema import HIT_EVENTS
from bts.util import retry_urlopen

API_BASE = "https://statsapi.mlb.com"

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
        return "\u042f" if last_pitch_code == "C" else "K"

    if event_type == "field_out":
        prefix = _TRAJECTORY_PREFIX.get(trajectory, "F")
        return f"{prefix}{fielder_position}" if fielder_position else f"{prefix}?"

    if event_type == "field_error":
        return f"E{fielder_position}" if fielder_position else "E"

    return _RESULT_MAP.get(event_type, event_type.upper()[:3] if event_type else "?")


# ---------------------------------------------------------------------------
# Ordinal helper
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Convert integer to ordinal string: 1 → '1st', 2 → '2nd', etc."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# PA extraction helpers
# ---------------------------------------------------------------------------

def _extract_fielder_position(runners: list[dict]) -> int | None:
    """Return the position code of the fielder who initiated the play.

    Checks f_fielded_ball first (fly outs, unassisted), then f_assist
    (assisted groundouts like 5-3, 6-3, 4-3).
    """
    for credit_type in ("f_fielded_ball", "f_assist", "f_putout"):
        for runner in runners:
            for credit in runner.get("credits", []):
                if credit.get("credit") == credit_type:
                    pos_code = credit.get("position", {}).get("code")
                    if pos_code is not None:
                        try:
                            return int(pos_code)
                        except (ValueError, TypeError):
                            pass
    return None


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
    if game_status == "P":
        return ("pre_game", None)
    if game_status == "F":
        return ("final", None)
    if game_status != "L":
        return ("pre_game", None)

    players = boxscore_team.get("players", {})
    batter_entry = players.get(f"ID{batter_id}", {})
    bo_str = batter_entry.get("battingOrder")
    batter_slot = _slot_from_bo(bo_str)

    if batter_slot is None:
        return ("not_in_lineup", None)

    current_array = boxscore_team.get("battingOrder")
    if not isinstance(current_array, list):
        return ("not_in_lineup", None)

    if batter_id not in current_array:
        return ("out_of_game", None)

    if current_batter_id is None:
        return ("pre_game", None)

    current_entry = players.get(f"ID{current_batter_id}", {})
    current_slot = _slot_from_bo(current_entry.get("battingOrder"))
    if current_slot is None:
        return ("pre_game", None)

    distance = (batter_slot - current_slot) % 9
    if distance == 0:
        return ("at_bat", 0)
    if distance == 1:
        return ("on_deck", 1)
    if distance == 2:
        return ("in_hole", 2)
    return ("upcoming", distance)


def _extract_pa(play: dict) -> dict:
    """Extract a structured plate-appearance dict from a single allPlays entry."""
    result_data = play.get("result", {})
    about = play.get("about", {})
    matchup = play.get("matchup", {})
    play_events = play.get("playEvents", [])
    runners = play.get("runners", [])

    event = result_data.get("event", "")
    event_type = result_data.get("eventType", "")

    # Pitch sequence (only events with isPitch=True)
    pitches_raw = [e for e in play_events if e.get("isPitch")]

    # Last pitch call code — needed for К vs K strikeout distinction
    last_pitch_code: str | None = None
    if pitches_raw:
        last_pitch_code = pitches_raw[-1].get("details", {}).get("call", {}).get("code")

    # Hit trajectory from the in-play pitch event
    hit_trajectory: dict | None = None
    trajectory_str: str | None = None
    for ev in play_events:
        if ev.get("isPitch") and ev.get("hitData"):
            hd = ev["hitData"]
            trajectory_str = hd.get("trajectory")
            coords = hd.get("coordinates", {})
            hit_trajectory = {
                "type": trajectory_str,
                "x": coords.get("coordX"),
                "y": coords.get("coordY"),
                "launch_speed": hd.get("launchSpeed"),
            }
            break

    # Fielder position for field_out / field_error result codes
    fielder_pos = _extract_fielder_position(runners)

    result_code = format_result_code(
        event=event,
        event_type=event_type,
        last_pitch_code=last_pitch_code,
        trajectory=trajectory_str,
        fielder_position=fielder_pos,
    )

    # Structured pitch sequence
    pitches = [
        {
            "number": i + 1,
            "call": ev.get("details", {}).get("call", {}).get("code", "?"),
            "is_strike": bool(ev.get("details", {}).get("isStrike", False)),
        }
        for i, ev in enumerate(pitches_raw)
    ]

    # Runner movements
    runner_movements = [
        {
            "runner_id": r.get("details", {}).get("runner", {}).get("id"),
            "start": r.get("movement", {}).get("start"),
            "end": r.get("movement", {}).get("end"),
            "out_base": r.get("movement", {}).get("outBase"),
            "is_out": bool(r.get("movement", {}).get("isOut", False)),
            "out_number": r.get("movement", {}).get("outNumber"),
        }
        for r in runners
    ]

    # out_number: look for the batter's own runner entry first, else first out
    batter_id = matchup.get("batter", {}).get("id")
    out_number: int | None = None
    for rm in runner_movements:
        if rm["runner_id"] == batter_id and rm["is_out"]:
            out_number = rm["out_number"]
            break
    if out_number is None:
        for rm in runner_movements:
            if rm["is_out"]:
                out_number = rm["out_number"]
                break

    is_hit = event_type in HIT_EVENTS
    is_out = bool(result_data.get("isOut", False))

    return {
        "at_bat_index": about.get("atBatIndex"),
        "inning": about.get("inning"),
        "half_inning": about.get("halfInning"),
        "result": result_code,
        "event_type": event_type,
        "is_hit": is_hit,
        "is_out": is_out,
        "out_number": out_number,
        "rbi": result_data.get("rbi", 0),
        "pitches": pitches,
        "hit_trajectory": hit_trajectory,
        "runners": runner_movements,
        "pitcher_id": matchup.get("pitcher", {}).get("id"),
        "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_batter_pas(feed: dict, batter_ids: set[int]) -> dict:
    """Extract structured PA data for the given batter IDs from a full MLB game feed.

    Args:
        feed: Full JSON response from /api/v1.1/game/{pk}/feed/live.
        batter_ids: Set of MLB player IDs whose PAs to extract.

    Returns:
        Dict with game-level context and per-batter PA lists.
    """
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})

    # Game status
    status = game_data.get("status", {})
    game_status = status.get("abstractGameCode", "")

    # Teams
    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("abbreviation", "")
    home_team = teams.get("home", {}).get("abbreviation", "")

    # Linescore — current inning
    linescore = live_data.get("linescore", {})
    current_inning = linescore.get("currentInning", 1)
    inning_half = linescore.get("inningHalf", "Top")
    inning_str = f"{inning_half} {_ordinal(current_inning)}"

    # Score
    ls_teams = linescore.get("teams", {})
    score = {
        "away": ls_teams.get("away", {}).get("runs", 0),
        "home": ls_teams.get("home", {}).get("runs", 0),
    }

    # Build boxscore player lookup: batter_id → player entry
    boxscore_teams = live_data.get("boxscore", {}).get("teams", {})
    player_lookup: dict[int, dict] = {}
    for side in ("away", "home"):
        players = boxscore_teams.get(side, {}).get("players", {})
        for key, player_data in players.items():
            pid = player_data.get("person", {}).get("id")
            if pid is not None:
                player_lookup[pid] = player_data

    # Walk allPlays, collect PAs grouped by batter
    all_plays = live_data.get("plays", {}).get("allPlays", [])
    batter_pas: dict[int, list[dict]] = {}
    batter_first_play: dict[int, dict] = {}  # stores matchup from first play for batSide

    for play in all_plays:
        about = play.get("about", {})
        batter_id = play.get("matchup", {}).get("batter", {}).get("id")
        if batter_id not in batter_ids:
            continue
        if batter_id not in batter_pas:
            batter_pas[batter_id] = []
            batter_first_play[batter_id] = play
        if about.get("isComplete", False):
            batter_pas[batter_id].append(_extract_pa(play))
        else:
            # In-progress PA — extract what we have so far
            pa = _extract_pa(play)
            pa["in_progress"] = True
            pa["result"] = "AB"
            pa["is_hit"] = False
            pa["is_out"] = False
            pa["out_number"] = None
            batter_pas[batter_id].append(pa)

    # Build batter list enriched with boxscore info.
    # Iterate over all requested batter_ids so batters with 0 completed PAs still appear.
    batters = []
    for batter_id in batter_ids:
        player_entry = player_lookup.get(batter_id, {})
        # Skip batters not in the boxscore at all (wrong game, DNP, etc.)
        if not player_entry:
            continue
        person = player_entry.get("person", {})
        stats_batting = player_entry.get("stats", {}).get("batting", {})

        avg = stats_batting.get("avg", "")
        obp = stats_batting.get("obp", "")
        slg = stats_batting.get("slg", "")
        slash_line = f"{avg}/{obp}/{slg}" if avg else ""

        batting_order_raw = player_entry.get("battingOrder", "0")
        try:
            lineup_position = int(batting_order_raw) // 100
        except (ValueError, TypeError):
            lineup_position = None

        position = player_entry.get("position", {}).get("abbreviation", "")
        name = person.get("fullName", "")

        # bat_side from first completed play; fall back to empty string if no PAs yet
        first_play = batter_first_play.get(batter_id)
        bat_side = first_play.get("matchup", {}).get("batSide", {}).get("code", "") if first_play else ""

        batters.append(
            {
                "batter_id": batter_id,
                "name": name,
                "position": position,
                "lineup_position": lineup_position,
                "batting_hand": bat_side,
                "slash_line": slash_line,
                "pas": batter_pas.get(batter_id, []),
            }
        )

    # Sort by lineup position (None last)
    batters.sort(key=lambda b: (b["lineup_position"] is None, b["lineup_position"] or 0))

    return {
        "game_status": game_status,
        "inning": inning_str,
        "away_team": away_team,
        "home_team": home_team,
        "score": score,
        "batters": batters,
    }


# ---------------------------------------------------------------------------
# Merge helper (different-game double-downs)
# ---------------------------------------------------------------------------


def merge_scorecards(sc1: dict | None, sc2: dict | None) -> dict | None:
    """Merge scorecard data from two different games.

    Used when the primary pick and double-down are in different games.
    The merged result carries batters from both games. Game status picks
    the most "watchable" state: Live wins over Final wins over Preview.
    This means the scorecard renders whenever at least one game is in
    progress, and shows Final only when BOTH games are done.
    A combined score label is stored for display.
    """
    if sc1 is None:
        return sc2
    if sc2 is None:
        return sc1

    merged = dict(sc1)
    merged["batters"] = list(sc1.get("batters", [])) + list(sc2.get("batters", []))

    # Prefer Live > Final > Preview — scorecard renders when any game is live
    status_priority = {"P": 0, "F": 1, "L": 2}
    if status_priority.get(sc2["game_status"], 0) > status_priority.get(sc1["game_status"], 0):
        merged["game_status"] = sc2["game_status"]
        merged["inning"] = sc2.get("inning", "")

    # Show both scores with innings as a combined label for the header
    s1 = sc1.get("score", {})
    s2 = sc2.get("score", {})
    inn1 = sc1.get("inning", "")
    inn2 = sc2.get("inning", "")
    label1 = f"{sc1.get('away_team', '?')} {s1.get('away', 0)}-{s1.get('home', 0)} {sc1.get('home_team', '?')}"
    label2 = f"{sc2.get('away_team', '?')} {s2.get('away', 0)}-{s2.get('home', 0)} {sc2.get('home_team', '?')}"
    if inn1:
        label1 += f" · {inn1}"
    if inn2:
        label2 += f" · {inn2}"
    merged["score_label"] = f"{label1} | {label2}"

    return merged


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------


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
