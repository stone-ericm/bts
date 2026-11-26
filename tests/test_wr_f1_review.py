"""Round-two regressions for unresolved entry game observations."""
from itertools import product

import pytest

from bts.entry_receipt import receipts_root
from tests.test_wr_f1_wrong_game import DAY, FIRST, LEG, MATCH, PITCH, RIGHT, WRONG, World, row

ALL_CLEAR = f"✅ BTS pick entry confirmed for {DAY} (match)."
CONFLICT = f"{DAY}T230000Z"


@pytest.mark.parametrize("initial", [True, False], ids=["M11", "M12"])
@pytest.mark.parametrize("bridge", ["crosswalk", "removal", "mismatch"])
def test_failed_observation(tmp_path, initial, bridge):
    # Adopt all three intermediate results from the reviewer's failed_observation.py.
    world = World(tmp_path)
    world.commit(pitch=PITCH)
    world.capture([WRONG])
    if initial:
        missing = world.check("18:25")
        assert missing["marker"]["reason"] == "no_pick" and missing["dm"]
    failed = world.check("18:35", MATCH, dm="failure")
    assert failed["marker"]["reason"] == "wrong_game"
    assert failed["marker"]["escalations"] == (["initial"] if initial else [])
    if bridge == "crosswalk":
        intermediate = world.check("18:45", MATCH, crosswalk={})
        assert intermediate["marker"]["status"] == "present_unverified"
    else:
        intermediate = world.check("18:45", [] if bridge == "removal" else [row(player=3)])
        assert intermediate["marker"]["reason"] == ("no_pick" if bridge == "removal" else "mismatch")
    world.capture([RIGHT], CONFLICT)
    last = world.check("19:00", MATCH)
    assert last["exit"] == 1
    assert last["marker"]["reason"] == "wrong_game"
    assert last["dm"] == [FIRST.format(legs=LEG.split(", the entry")[0])]
    assert last["receipts"][-1]["qualification"]["slots"][0]["state"] == "unit_unverified"
    for result in (failed, intermediate, last):
        assert result["marker"]["game_unresolved"] is True


def test_m13_partial_double_proof(tmp_path):
    world = World(tmp_path)
    world.commit(pitch=PITCH, dd=(1020, 201, PITCH))
    second = {"id": 78, "roundId": 1, "feedId": 201}
    both = [row(), row(78, 2, 2)]
    world.capture([WRONG, second])
    world.check("18:30", both, dm="failure")
    world.check("18:40", both, crosswalk={})
    world.capture([RIGHT, {**second, "feedId": None}])
    partial = world.check("18:45", both)
    assert [slot["state"] for slot in partial["receipts"][-1]["qualification"]["slots"]] == [
        "confirmed", "unit_unverified"
    ]
    assert partial["exit"] == 1
    assert partial["marker"]["game_unresolved"] is True
    assert ALL_CLEAR not in partial["attempted_dm"]
    world.capture([RIGHT, second])
    proved = world.check("19:00", both)
    assert proved["marker"]["status"] == "confirmed"
    assert "game_unresolved" not in proved["marker"]
    assert proved["dm"] == [ALL_CLEAR]


def test_locked_delivery_names_fallback(tmp_path):
    world = World(tmp_path)
    world.commit(pitch=PITCH, dd=(1020, 201, PITCH), name=None, delivery="locked_unconfirmed")
    world.capture([WRONG, {"id": 78, "roundId": 1, "feedId": 201}])
    result = world.check("18:30", [row(), row(78, 2, 2)])
    message, = result["dm"]
    assert "The scheduler tried to send batter 1010 + DD Batter 1020." in message


def test_unavailable_recommended_game_id(tmp_path):
    world = World(tmp_path)
    world.commit(pitch=PITCH, game=None, dd=(1020, 201, PITCH))
    world.capture([RIGHT, {"id": 78, "roundId": 1, "feedId": 202}])
    result = world.check("18:30", [row(), row(78, 2, 2)])
    assert result["marker"]["reason"] == "wrong_game"
    world.capture([RIGHT, {"id": 78, "roundId": 1, "feedId": 201}])
    reminder = world.check("19:05", [row(), row(78, 2, 2)])
    message, = reminder["dm"]
    assert "pick Batter 1010: recommended game ID unavailable (first pitch 7:40 PM ET)" in message


@pytest.mark.parametrize("reason,escalations", [("wrong_game", None), ("no_pick", None),
                                                 ("no_pick", ["wrong_game"])])
def test_legacy_observation_does_not_assume_initial(tmp_path, reason, escalations):
    world = World(tmp_path)
    world.commit(pitch=PITCH)
    marker = {"date": DAY, "status": "alerted", "reason": reason}
    if escalations is None and reason == "no_pick":
        marker["game_unresolved"] = True
    elif escalations is not None:
        marker["escalations"] = escalations
    world.set_marker(marker)
    result = world.check("18:45")
    assert len(result["dm"]) == 1
    assert result["marker"]["game_unresolved"] is True
    assert result["marker"]["escalations"] == sorted(["initial", *(escalations or [])])


def test_ordered_pairs_and_triples_after_observation(tmp_path):
    cases = {
        "no entry": ([], None, WRONG),
        "batter mismatch": ([row(player=3)], None, WRONG),
        "crosswalk gap": (MATCH, {}, WRONG),
        "wrong game": (MATCH, None, WRONG),
        "unknown binding": (MATCH, None, {**RIGHT, "feedId": None}),
        "full proof": (MATCH, None, RIGHT),
    }
    outcomes = tuple(product(cases, ("success", "failure")))
    failures = []
    world = World(tmp_path)
    world.commit(pitch=PITCH)
    for length in (2, 3):
        for sequence in product(outcomes, repeat=length):
            # Restart from a real failed observation with no consumed tiers.
            world.marker.unlink(missing_ok=True)
            world.capture([WRONG])
            world.check("18:25", MATCH, dm="failure")
            proved = False
            for at, (outcome, dm) in zip(("18:35", "18:45", "18:55"), sequence):
                rows, crosswalk, unit = cases[outcome]
                world.capture([unit])
                result = world.check(at, rows, crosswalk=crosswalk, dm=dm)
                proved |= outcome == "full proof"
                if not proved:
                    marker = result["marker"]
                    if marker["status"] == "confirmed" or ALL_CLEAR in result["attempted_dm"]:
                        failures.append((sequence, at, "premature confirmation"))
                    elif marker.get("game_unresolved") is not True:
                        failures.append((sequence, at, "observation lost"))
            # The sequence only needs receipts from its own runs.
            for path in (receipts_root(world.picks) / DAY).glob("*.json"):
                path.unlink()
    assert not failures, f"{len(failures)} violations, first examples: {failures[:10]}"
