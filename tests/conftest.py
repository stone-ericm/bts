import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def stamp_saved_pick():
    """Stamp a saved pick using a required snapshot captured by its test."""
    from bts.picks import chosen_on_record

    def stamp(daily, required_state):
        daily.chosen_on = chosen_on_record(required_state, required=True)
        return daily

    return stamp


@pytest.fixture
def covered_automatic():
    """Cover a raw automatic fixture at its explicit operation clock.

    Preserve values, dates and corruptions under test; add only switch metadata.
    """
    def cover(picks, now, payload):
        from zoneinfo import ZoneInfo
        year = now.astimezone(ZoneInfo("America/New_York")).year
        identifier = "c" * 32
        directory = picks / "account_state"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "season_contract.json").write_text(json.dumps(dict(
            schema_version="bts_season_contract_v1", season=year, status="automatic",
            activation_id=identifier, switched_at=now.isoformat(), revoked_at=None, note=None)))
        path = directory / "contest_streak.json"
        path.write_text(json.dumps(payload | dict(schema_version="bts_contest_streak_auto_v2",
                                                 season=year, activation_id=identifier)))
        return path
    return cover

# Make the project root importable so `from scripts.foo import bar` works.
# (scripts/__init__.py makes scripts/ a package; the root must be on sys.path.)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture
def sample_game_feed():
    """Minimal game feed JSON with enough structure to test PA extraction."""
    return {
        "gameData": {
            "datetime": {"officialDate": "2025-06-15"},
            "teams": {
                "away": {"id": 134, "name": "Pittsburgh Pirates"},
                "home": {"id": 121, "name": "New York Mets"},
            },
            "venue": {
                "id": 3289,
                "name": "Citi Field",
                "fieldInfo": {"roofType": "Open"},
            },
            "weather": {
                "condition": "Sunny",
                "temp": "78",
                "wind": "9 mph, Out To CF",
            },
            "game": {"pk": 999999, "season": "2025", "type": "R"},
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "batters": [100001],
                        "players": {
                            "ID100001": {
                                "person": {"id": 100001, "fullName": "Test Batter"},
                                "battingOrder": "300",
                            },
                        },
                    },
                    "home": {
                        "batters": [200001],
                        "players": {
                            "ID200001": {
                                "person": {"id": 200001, "fullName": "Home Batter"},
                                "battingOrder": "100",
                            },
                        },
                    },
                },
                "officials": [
                    {
                        "official": {"id": 427215, "fullName": "Test Umpire"},
                        "officialType": "Home Plate",
                    },
                ],
            },
            "plays": {
                "allPlays": [
                    {
                        "result": {
                            "eventType": "single",
                            "description": "Test Batter singles.",
                        },
                        "about": {
                            "halfInning": "top",
                            "inning": 1,
                            "hasReview": False,
                        },
                        "matchup": {
                            "batter": {"id": 100001},
                            "batSide": {"code": "R", "description": "Right"},
                            "pitcher": {"id": 300001},
                            "pitchHand": {"code": "L", "description": "Left"},
                        },
                        "count": {"balls": 1, "strikes": 2},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "details": {
                                    "call": {"code": "B", "description": "Ball"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": -0.5, "pZ": 2.8},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                    "startSpeed": 93.5,
                                    "endSpeed": 85.0,
                                    "extension": 6.3,
                                    "breaks": {"spinRate": 2400, "breakVertical": -15.0, "breakHorizontal": 8.0},
                                },
                                "count": {"balls": 1, "strikes": 0},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "details": {
                                    "call": {"code": "C", "description": "Called Strike"},
                                    "type": {"code": "SL", "description": "Slider"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.3, "pZ": 2.1},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                    "startSpeed": 85.2,
                                    "endSpeed": 78.0,
                                    "extension": 6.1,
                                    "breaks": {"spinRate": 2700, "breakVertical": -32.0, "breakHorizontal": -2.0},
                                },
                                "count": {"balls": 1, "strikes": 1},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 3,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "CH", "description": "Changeup"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.1, "pZ": 1.9},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                    "startSpeed": 84.1,
                                    "endSpeed": 76.0,
                                    "extension": 6.2,
                                    "breaks": {"spinRate": 1800, "breakVertical": -28.0, "breakHorizontal": -12.0},
                                },
                                "count": {"balls": 1, "strikes": 2},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 4,
                                "details": {
                                    "call": {"code": "X", "description": "In play, no out"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": -0.2, "pZ": 2.5},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                    "startSpeed": 94.0,
                                    "endSpeed": 86.0,
                                    "extension": 6.4,
                                    "breaks": {"spinRate": 2350, "breakVertical": -14.0, "breakHorizontal": 9.0},
                                },
                                "hitData": {
                                    "launchSpeed": 98.3,
                                    "launchAngle": 12.0,
                                    "trajectory": "line_drive",
                                    "hardness": "hard",
                                    "totalDistance": 310.0,
                                },
                                "count": {"balls": 1, "strikes": 2},
                            },
                        ],
                    },
                    {
                        "result": {
                            "eventType": "strikeout",
                            "description": "Home Batter strikes out.",
                        },
                        "about": {
                            "halfInning": "bottom",
                            "inning": 1,
                            "hasReview": False,
                        },
                        "matchup": {
                            "batter": {"id": 200001},
                            "batSide": {"code": "L", "description": "Left"},
                            "pitcher": {"id": 400001},
                            "pitchHand": {"code": "R", "description": "Right"},
                        },
                        "count": {"balls": 0, "strikes": 3},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.0, "pZ": 2.5},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                    "startSpeed": 95.1,
                                    "endSpeed": 87.0,
                                    "extension": 6.5,
                                    "breaks": {"spinRate": 2300, "breakVertical": -13.0, "breakHorizontal": 7.0},
                                },
                                "count": {"balls": 0, "strikes": 1},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "SL", "description": "Slider"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.8, "pZ": 1.8},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                    "startSpeed": 86.0,
                                    "endSpeed": 79.0,
                                    "extension": 6.2,
                                    "breaks": {"spinRate": 2650, "breakVertical": -30.0, "breakHorizontal": -3.0},
                                },
                                "count": {"balls": 0, "strikes": 2},
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 3,
                                "details": {
                                    "call": {"code": "S", "description": "Swinging Strike"},
                                    "type": {"code": "CH", "description": "Changeup"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": 0.4, "pZ": 2.0},
                                    "strikeZoneTop": 3.3,
                                    "strikeZoneBottom": 1.6,
                                    "startSpeed": 85.5,
                                    "endSpeed": 77.0,
                                    "extension": 6.3,
                                    "breaks": {"spinRate": 1750, "breakVertical": -27.0, "breakHorizontal": -11.0},
                                },
                                "count": {"balls": 0, "strikes": 3},
                            },
                        ],
                    },
                ],
            },
        },
    }


@pytest.fixture
def sample_feed_path(sample_game_feed, tmp_path):
    """Write sample feed to a file and return its path."""
    raw_dir = tmp_path / "raw" / "2025"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "999999.json"
    path.write_text(json.dumps(sample_game_feed))
    return path


@pytest.fixture(autouse=True)
def _delivery_guard_never_reads_the_real_schedule(monkeypatch):
    """The delivery chokepoint reads MLB game statuses before every delivery (the postponed-game guard,
    incident L03). In tests that read must never reach the network: unless a test has replaced
    ``bts.picks.get_game_statuses_detailed`` or ``bts.picks.retry_urlopen`` itself, the guard sees no status
    (and therefore delivers, as it does when the schedule is unreadable)."""
    import bts.picks as picks
    import bts.scheduler as sch
    real_statuses, real_urlopen = picks.get_game_statuses_detailed, picks.retry_urlopen

    def statuses_for_the_guard(date):
        if picks.get_game_statuses_detailed is not real_statuses or picks.retry_urlopen is not real_urlopen:
            return picks.get_game_statuses_detailed(date)
        return {}

    monkeypatch.setattr(sch, "_delivery_game_statuses", statuses_for_the_guard)
