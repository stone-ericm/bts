"""Tests for live scorecard data extraction."""
import json
import copy
import pytest
from bts.scorecard import format_result_code, extract_batter_pas


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


class TestFormatResultCode:
    def test_single(self):
        assert format_result_code("single", "single", None, None, None) == "1B"

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
        assert format_result_code("strikeout", "strikeout", "S", None, None) == "K"

    def test_strikeout_looking(self):
        assert format_result_code("strikeout", "strikeout", "C", None, None) == "\u042f"

    def test_flyout_to_right(self):
        assert format_result_code("field_out", "field_out", None, "fly_ball", 9) == "F9"

    def test_groundout_to_short(self):
        assert format_result_code("field_out", "field_out", None, "ground_ball", 6) == "G6"

    def test_lineout_to_center(self):
        assert format_result_code("field_out", "field_out", None, "line_drive", 8) == "L8"

    def test_popup_to_second(self):
        assert format_result_code("field_out", "field_out", None, "popup", 4) == "P4"

    def test_flyout_no_trajectory(self):
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
