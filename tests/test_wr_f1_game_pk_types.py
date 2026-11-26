"""Keep the watchdog's exact-integer game ID contract."""

import pytest

from tests.test_wr_f1_wrong_game import MATCH, PITCH, REMINDER, WRONG, row, world


@pytest.mark.parametrize("game_pk", [True, False, "101", 101.0])
def test_a1_non_integer_game_id_cannot_trigger_wrong_game(world, game_pk):
    world.commit(game=game_pk, pitch=PITCH)
    world.capture([WRONG])

    result = world.check("18:45", MATCH)

    assert result["exit"] == 0
    assert result["marker"]["status"] == "confirmed"
    assert result["dm"] == []


@pytest.mark.parametrize("game_pk", [True, False, "101", 101.0])
def test_a3_unresolved_non_integer_game_id_is_unavailable(world, game_pk):
    world.commit(game=game_pk, pitch=PITCH, dd=(1020, 201, PITCH))
    first = {"id": 77, "roundId": 1, "feedId": None}
    second = {"id": 78, "roundId": 1, "feedId": 202}
    pending = [row(), row(78, 2, 2)]
    world.capture([first, second])
    initial = world.check("18:30", pending)
    assert initial["marker"]["game_unresolved"] is True

    world.capture([first, {**second, "feedId": 201}])
    result = world.check("19:05", pending)

    assert result["exit"] == 1
    assert result["marker"]["game_unresolved"] is True
    assert result["dm"] == [REMINDER.format(
        legs="pick Batter 1010: recommended game ID unavailable (first pitch 7:40 PM ET)"
    )]
