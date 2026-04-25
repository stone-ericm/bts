"""Tests for live scorecard data extraction."""
import json
import copy
import pytest
from unittest.mock import patch
from bts.scorecard import format_result_code, extract_batter_pas, fetch_live_scorecard, merge_scorecards


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


class TestExtractFielderPosition:
    """Test _extract_fielder_position with different credit types."""

    def test_fielded_ball_credit(self):
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": [{"credit": "f_fielded_ball", "position": {"code": "9"}}]}]
        assert _extract_fielder_position(runners) == 9

    def test_assist_credit_groundout(self):
        """Assisted groundout 5-3: f_assist on 5, f_putout on 3."""
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": [
            {"credit": "f_assist", "position": {"code": "5"}},
            {"credit": "f_putout", "position": {"code": "3"}},
        ]}]
        assert _extract_fielder_position(runners) == 5

    def test_fielded_ball_preferred_over_assist(self):
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": [
            {"credit": "f_assist", "position": {"code": "6"}},
            {"credit": "f_fielded_ball", "position": {"code": "4"}},
        ]}]
        assert _extract_fielder_position(runners) == 4

    def test_putout_only_flyout(self):
        """Fly out with only f_putout credit (no f_fielded_ball)."""
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": [{"credit": "f_putout", "position": {"code": "8"}}]}]
        assert _extract_fielder_position(runners) == 8

    def test_assist_preferred_over_putout(self):
        """Groundout 6-3: f_assist (6) should win over f_putout (3)."""
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": [
            {"credit": "f_assist", "position": {"code": "6"}},
            {"credit": "f_putout", "position": {"code": "3"}},
        ]}]
        assert _extract_fielder_position(runners) == 6

    def test_no_credits(self):
        from bts.scorecard import _extract_fielder_position
        runners = [{"credits": []}]
        assert _extract_fielder_position(runners) is None


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


class TestMergeScorecard:
    def test_merge_two_scorecards(self):
        sc1 = {
            "game_status": "L",
            "inning": "Top 4th",
            "score": {"away": 1, "home": 0},
            "away_team": "TB",
            "home_team": "MIN",
            "batters": [{"name": "Diaz", "batter_id": 1, "pas": []}],
        }
        sc2 = {
            "game_status": "L",
            "inning": "Bot 3rd",
            "score": {"away": 2, "home": 1},
            "away_team": "PHI",
            "home_team": "COL",
            "batters": [{"name": "Turner", "batter_id": 2, "pas": []}],
        }
        merged = merge_scorecards(sc1, sc2)
        assert len(merged["batters"]) == 2
        names = {b["name"] for b in merged["batters"]}
        assert names == {"Diaz", "Turner"}

    def test_merge_none_first(self):
        sc = {"game_status": "L", "batters": [{"name": "X"}]}
        assert merge_scorecards(None, sc) == sc

    def test_merge_none_second(self):
        sc = {"game_status": "L", "batters": [{"name": "X"}]}
        assert merge_scorecards(sc, None) == sc

    def test_merge_score_label_set(self):
        sc1 = {
            "game_status": "L",
            "inning": "Top 2nd",
            "score": {"away": 0, "home": 1},
            "away_team": "TB",
            "home_team": "MIN",
            "batters": [],
        }
        sc2 = {
            "game_status": "F",
            "inning": "",
            "score": {"away": 3, "home": 2},
            "away_team": "PHI",
            "home_team": "COL",
            "batters": [],
        }
        merged = merge_scorecards(sc1, sc2)
        assert "score_label" in merged
        assert "TB" in merged["score_label"]
        assert "PHI" in merged["score_label"]

    def test_merge_uses_least_advanced_status(self):
        """When sc2 is Final and sc1 is Live, merged should stay Live
        (only Final when ALL games are done)."""
        sc1 = {
            "game_status": "L",
            "inning": "Top 5th",
            "score": {"away": 1, "home": 0},
            "away_team": "TB",
            "home_team": "MIN",
            "batters": [],
        }
        sc2 = {
            "game_status": "F",
            "inning": "",
            "score": {"away": 4, "home": 2},
            "away_team": "PHI",
            "home_team": "COL",
            "batters": [],
        }
        merged = merge_scorecards(sc1, sc2)
        assert merged["game_status"] == "L"

    def test_merge_stays_live_when_one_game_finished(self):
        """When sc1 is Final and sc2 is Live, merged stays Live."""
        sc1 = {
            "game_status": "F",
            "inning": "",
            "score": {"away": 4, "home": 2},
            "away_team": "TB",
            "home_team": "MIN",
            "batters": [],
        }
        sc2 = {
            "game_status": "L",
            "inning": "Top 5th",
            "score": {"away": 1, "home": 0},
            "away_team": "PHI",
            "home_team": "COL",
            "batters": [],
        }
        merged = merge_scorecards(sc1, sc2)
        assert merged["game_status"] == "L"

    def test_merge_final_when_both_final(self):
        """Only Final when ALL games are done."""
        sc1 = {
            "game_status": "F",
            "inning": "",
            "score": {"away": 4, "home": 2},
            "away_team": "TB",
            "home_team": "MIN",
            "batters": [],
        }
        sc2 = {
            "game_status": "F",
            "inning": "",
            "score": {"away": 3, "home": 1},
            "away_team": "PHI",
            "home_team": "COL",
            "batters": [],
        }
        merged = merge_scorecards(sc1, sc2)
        assert merged["game_status"] == "F"

    def test_merge_live_beats_preview_when_primary_preview(self):
        """When primary is Preview (pre-game) and double-down is Live,
        the merged status must be Live so the scorecard actually renders.

        Regression for 2026-04-12: primary was Donovan (SEA 16:10 ET, still
        P) and double-down was Anthony (BOS 14:15 ET, already L). The
        merge used "least advanced" priority which picked P, and
        render_scorecard_section returned empty because its guard requires
        game_status in (L, F). The dashboard showed no scorecard during
        Anthony's live game.
        """
        sc_primary = {
            "game_status": "P",
            "inning": "",
            "score": {"away": 0, "home": 0},
            "away_team": "HOU",
            "home_team": "SEA",
            "batters": [{"name": "Donovan", "batter_id": 1, "pas": []}],
        }
        sc_dd = {
            "game_status": "L",
            "inning": "Bot 2nd",
            "score": {"away": 1, "home": 3},
            "away_team": "BOS",
            "home_team": "STL",
            "batters": [{"name": "Anthony", "batter_id": 2, "pas": []}],
        }
        merged = merge_scorecards(sc_primary, sc_dd)
        assert merged["game_status"] == "L"
        assert merged["inning"] == "Bot 2nd"

    def test_merge_live_beats_preview_when_dd_preview(self):
        """Mirror of the regression above: primary Live, double-down Preview."""
        sc_primary = {
            "game_status": "L",
            "inning": "Top 3rd",
            "score": {"away": 1, "home": 0},
            "away_team": "BOS",
            "home_team": "STL",
            "batters": [{"name": "Anthony", "batter_id": 1, "pas": []}],
        }
        sc_dd = {
            "game_status": "P",
            "inning": "",
            "score": {"away": 0, "home": 0},
            "away_team": "HOU",
            "home_team": "SEA",
            "batters": [{"name": "Donovan", "batter_id": 2, "pas": []}],
        }
        merged = merge_scorecards(sc_primary, sc_dd)
        assert merged["game_status"] == "L"
        assert merged["inning"] == "Top 3rd"

    def test_merge_stays_preview_when_both_preview(self):
        """Both games pre-game: merged stays P (no scorecard to render yet)."""
        sc1 = {
            "game_status": "P", "inning": "", "score": {"away": 0, "home": 0},
            "away_team": "A", "home_team": "B", "batters": [],
        }
        sc2 = {
            "game_status": "P", "inning": "", "score": {"away": 0, "home": 0},
            "away_team": "C", "home_team": "D", "batters": [],
        }
        merged = merge_scorecards(sc1, sc2)
        assert merged["game_status"] == "P"


class TestRenderScorecardHeader:
    """render_scorecard_section must include a header with LIVE/FINAL badge
    and the combined score label. Without this, the #scorecard div polled
    every 30s lacks any live status indicator — the only badges on the
    dashboard come from _render_game_tags above the scorecard, and those
    aren't in the polling payload so they go stale after page load.
    """

    def _basic_scorecard(self, game_status, score_label=None):
        return {
            "game_status": game_status,
            "inning": "Bot 3rd" if game_status == "L" else "",
            "away_team": "BOS",
            "home_team": "STL",
            "score": {"away": 1, "home": 3},
            "score_label": score_label,
            "batters": [
                {"name": "Anthony", "batter_id": 1, "lineup_position": 1,
                 "position": "RF", "slash_line": ".280/.350/.450", "pas": []},
            ],
        }

    def test_live_badge_present_when_game_live(self):
        from bts.web import render_scorecard_section
        html = render_scorecard_section(self._basic_scorecard("L"))
        assert "LIVE" in html
        assert "FINAL" not in html

    def test_final_badge_present_when_game_final(self):
        from bts.web import render_scorecard_section
        html = render_scorecard_section(self._basic_scorecard("F"))
        assert "FINAL" in html
        assert "LIVE" not in html

    def test_score_label_rendered_for_double_down(self):
        """When score_label is set (double-down across games), it should
        appear in the header verbatim so the polling payload shows both
        scores + innings."""
        from bts.web import render_scorecard_section
        label = "HOU 0-0 SEA · Top 1st | BOS 3-1 STL · Bot 3rd"
        html = render_scorecard_section(self._basic_scorecard("L", score_label=label))
        assert label in html

    def test_single_game_score_fallback(self):
        """When no score_label (single pick), header should show the basic
        away/home score line."""
        from bts.web import render_scorecard_section
        html = render_scorecard_section(self._basic_scorecard("L"))
        # away team + runs + home team should all be in the header
        assert "BOS" in html
        assert "STL" in html

    def test_preview_status_still_returns_empty(self):
        """Status P must continue to return empty — no scorecard to render."""
        from bts.web import render_scorecard_section
        assert render_scorecard_section(self._basic_scorecard("P")) == ""


class TestBatterWithZeroPas:
    def test_batter_appears_with_no_pas(self):
        """Batter in boxscore but no completed PAs should still appear."""
        feed = copy.deepcopy(SAMPLE_FEED)
        feed["liveData"]["plays"]["allPlays"] = []  # No plays at all
        result = extract_batter_pas(feed, {650490})
        assert len(result["batters"]) == 1
        assert result["batters"][0]["name"] == "Yandy Diaz"
        assert result["batters"][0]["pas"] == []

    def test_batter_not_in_boxscore_excluded(self):
        """Requested batter ID not in boxscore should not appear (e.g. wrong game)."""
        feed = copy.deepcopy(SAMPLE_FEED)
        feed["liveData"]["plays"]["allPlays"] = []
        result = extract_batter_pas(feed, {999999})
        assert len(result["batters"]) == 0

    def test_zero_pa_batter_has_correct_fields(self):
        """Batter with 0 PAs should still carry boxscore info (slash line, position)."""
        feed = copy.deepcopy(SAMPLE_FEED)
        feed["liveData"]["plays"]["allPlays"] = []
        result = extract_batter_pas(feed, {650490})
        batter = result["batters"][0]
        assert batter["position"] == "DH"
        assert batter["lineup_position"] == 2
        assert batter["slash_line"] == ".419/.486/.645"
        assert batter["batting_hand"] == ""  # No plays, so no bat-side data


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
        assert _slot_from_bo("000") is None

    def test_out_of_range_high(self):
        from bts.scorecard import _slot_from_bo
        assert _slot_from_bo("1000") is None


def _mk_team(battingOrder: list[int], players: dict) -> dict:
    """Build a boxscore_team block from a lineup + per-player data."""
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
        status, away = _compute_lineup_status(1, team, current_batter_id=8, game_status="L")
        assert status == "in_hole"
        assert away == 2

    def test_out_of_game_pulled_starter(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3, 44, 5, 6, 7, 8, 9],
            {
                1: {"battingOrder": "100"},
                2: {"battingOrder": "200"},
                3: {"battingOrder": "300"},
                4: {"battingOrder": "400", "atBats": 2},
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
                99: {},
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
        status, away = _compute_lineup_status(2, team, current_batter_id=None, game_status="L")
        assert status == "pre_game"
        assert away is None

    def test_missing_battingOrder_array(self):
        from bts.scorecard import _compute_lineup_status
        team = {"players": {"ID1": {"battingOrder": "100"}}}
        status, away = _compute_lineup_status(1, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_missing_player_key(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team([1, 2, 3], {1: {"battingOrder": "100"}})
        status, away = _compute_lineup_status(99, team, current_batter_id=1, game_status="L")
        assert status == "not_in_lineup"
        assert away is None

    def test_current_batter_id_unknown_in_players(self):
        from bts.scorecard import _compute_lineup_status
        team = _mk_team(
            [1, 2, 3],
            {1: {"battingOrder": "100"}, 2: {"battingOrder": "200"}, 3: {"battingOrder": "300"}},
        )
        status, away = _compute_lineup_status(2, team, current_batter_id=999, game_status="L")
        assert status == "pre_game"
        assert away is None
