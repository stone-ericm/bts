from scripts.audit.season_ledger.contest import match_slot, resolve_duplicate_links

ROUNDS = {971: {"2026-08-20"}}
PLAYERS = {2513: {802415}, 1300: {680757}}
SEL = {"date": "2026-08-20", "batter_id": 802415, "game_pk": 822934, "team_at_pick": "TB",
       "selection_id": "2026-08-20|primary|802415|822934"}
TB_ONE = {("2026-08-20", "TB"): {822934}}
COMPLETE = {"2026-08-20": "complete"}


def _slot(unit_id=1928, player_id=2513, round_id=971):
    return {"round_id": round_id, "unit_id": unit_id, "player_id": player_id}


def _match(*, units=None, games=None, sels=(SEL,), slot=None, status=None):
    return match_slot(slot or _slot(), rounds=ROUNDS, players=PLAYERS, units=units or {},
                      team_games=TB_ONE if games is None else games,
                      schedule_status=COMPLETE if status is None else status, local_selections=list(sels))


def test_unit_capture_gives_evidenced_match_to_the_local_selection():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {971}}}, games={})
    assert (m["match"], m["game_pk"], m["selection_id"], m["match_reason"]) == (
        "evidenced", 822934, SEL["selection_id"], "unit_capture")


def test_conflicting_unit_captures_are_ambiguous_and_never_fall_through_to_inference():
    m = _match(units={1928: {"feed_ids": {822934, 900001}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "conflicting_unit_evidence")


def test_unit_capture_seen_in_another_round_is_ambiguous():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {970, 971}}})
    assert (m["match"], m["match_reason"]) == ("ambiguous", "unit_round_contradiction")


def test_unit_capture_for_another_game_does_not_link():
    m = _match(units={1928: {"feed_ids": {900002}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("evidenced", None, "unit_capture_other_game")


def test_inference_uses_the_pick_time_team_not_the_current_one():
    # A traded player: a current-team lookup would say SEA; the pick file recorded TB.
    m = _match(games={("2026-08-20", "TB"): {822934}, ("2026-08-20", "SEA"): {900001}})
    assert (m["match"], m["game_pk"], m["match_reason"]) == ("inferred", 822934, "pick_time_team_single_scheduled_game")


def test_doubleheader_or_postponed_plus_played_game_is_ambiguous():
    m = _match(games={("2026-08-20", "TB"): {822934, 822935}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "team_schedule_not_unique")


def test_selection_without_a_recorded_game_is_ambiguous():
    # Review Focus 2: the 3/29–3/30 pick files carry game_pk = null.
    m = _match(sels=(dict(SEL, game_pk=None, selection_id="2026-08-20|primary|802415|None"),))
    assert (m["match"], m["match_reason"]) == ("ambiguous", "selection_game_pk_unrecorded")


def test_missing_or_incomplete_schedule_and_absent_team_are_ambiguous_with_distinct_reasons():
    assert _match(status={})["match_reason"] == "team_schedule_missing"
    assert _match(status={"2026-08-20": "incomplete"})["match_reason"] == "team_schedule_incomplete"
    assert _match(games={("2026-08-20", "NYY"): {5002}})["match_reason"] == "team_not_on_schedule"


def test_contest_only_slot_without_unit_capture_is_unmapped():
    m = _match(sels=())
    assert (m["match"], m["match_reason"]) == ("unmapped", "no_unit_capture_contest_only")


def test_unknown_round_is_unmapped_and_mapped_and_unmapped_slots_share_a_round():
    assert _match(slot=_slot(round_id=5))["match_reason"] == "round_date_unknown"
    known, unknown = _match(), _match(slot=_slot(unit_id=1927, player_id=42))
    assert (known["match"], unknown["match"], unknown["match_reason"]) == ("inferred", "unmapped", "player_unknown")


def test_two_contest_slots_linking_one_selection_are_both_demoted():
    a = {"round_id": 971, "unit_id": 1928, "player_id": 2513, "selection_id": "s", "match": "inferred",
         "match_reason": "x"}
    out = resolve_duplicate_links([a, dict(a, unit_id=1929), dict(a, unit_id=1930, selection_id="t")])
    assert [(m["unit_id"], m["match"], m["selection_id"]) for m in out] == [
        (1928, "ambiguous", None), (1929, "ambiguous", None), (1930, "inferred", "t")]
    assert out[0]["match_reason"] == "multiple_contest_slots_for_selection"
