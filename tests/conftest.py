import json
import pytest


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
            "game": {"pk": 999999, "season": "2025"},
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
                            "pitcher": {"id": 300001},
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
                                },
                                "hitData": {
                                    "launchSpeed": 98.3,
                                    "launchAngle": 12.0,
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
                            "pitcher": {"id": 400001},
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
