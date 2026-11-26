from scripts.audit.season_ledger.outcomes import (derived_single_result, normalize_contest, normalize_local,
                                                  slot_disagreement)


def test_local_miss_and_contest_not_hit_agree():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("not_hit")) is False


def test_void_maps_to_hold_on_both_sides():
    assert normalize_local("void") == normalize_contest("void") == "HOLD"


def test_c03_pattern_disagrees():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("hit")) is True


def test_unknown_or_missing_labels_are_never_compared():
    assert normalize_local("suspended") == normalize_local("unresolved") == "UNKNOWN"
    assert slot_disagreement("UNKNOWN", "HIT") is None and slot_disagreement(None, "HIT") is None


def test_round_labels_are_not_slot_labels():
    assert normalize_contest("used_mulligan") == "UNKNOWN"


def test_single_pick_day_result_derives_only_for_single_pick_records():
    single = {"obs_id": "o1", "has_double_down": False, "slot_result_raw": None, "day_result_raw": "hit"}
    assert derived_single_result(single) == ("hit", "day_result_of_single_pick:o1")
    assert derived_single_result(dict(single, has_double_down=True, day_result_raw="miss")) == (None, None)
    assert derived_single_result(dict(single, slot_result_raw="miss")) == (None, None)
    assert derived_single_result(None) == (None, None)
