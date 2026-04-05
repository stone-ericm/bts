"""Tests for live scorecard data extraction."""
import pytest
from bts.scorecard import format_result_code


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
