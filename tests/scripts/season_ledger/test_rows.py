from scripts.audit.season_ledger.rows import day_rows
from scripts.audit.season_ledger.sources.day_records import (parse_decision, parse_lineup_evolution,
                                                             parse_scheduler_state)
from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file, pick_file_state
from tests.scripts.season_ledger.builders import cand, decision_json, evolution_jsonl, pick_json, state_json

D = "2026-08-20"
A = {"batter_id": 101, "game_pk": 5001}                    # the builders' default primary
B = {"batter_id": 202, "game_pk": 5002, "team": "NYY"}     # the builders' default double-down


def dec(**kw):
    return parse_decision(f"picks/{D}/decision.json", decision_json(D, **kw)).rows[0]


def picks(data):
    """(parsed rows, pick_file_state) for a surviving pick file, as the compiler passes them."""
    parsed = parse_pick_file(f"picks/{D}.json", data)
    return parsed.rows, pick_file_state(data, parsed)


def state(**kw):
    return parse_scheduler_state(f"picks/{D}/scheduler_state.json", state_json(D, **kw)).rows[0]


def evo(entries):
    return parse_lineup_evolution(f"picks/lineup_evolution_{D}.jsonl", evolution_jsonl(D, entries)).rows


def archived(primary):
    return parse_archive(f"picks/{D}/deferred_fallback_20260820T110000-0400.json",
                         pick_json(D, primary=primary, deferred_fallback={"reason": "r",
                                                                          "deferred_at": f"{D}T11:00:00-04:00"})).rows


def rows(decision=None, pick=None, st=None, observations=(), unusable=()):
    return day_rows(D, decision=decision, pick_rows=list(pick[0]) if pick else [], pick_file=pick[1] if pick else None,
                    state=st, observations=list(observations), unusable=frozenset(unusable))


def test_decision_double_gives_two_committed_selections():
    out = rows(decision=dec(action="double", primary=cand(101, 5001), double_down=cand(202, 5002, team="NYY")),
               pick=picks(pick_json(D, dd={}, notification_sent=True, notification_id="dm-1")))
    assert [(r["slot"], r["finalization"], r["commit_status"], r["delivery_confirmed"]) for r in out] == [
        ("primary", "decision", "committed_evidenced", True), ("double_down", "decision", "committed_evidenced", True)]
    assert out[0]["commit_basis"] == "decision:delivered;delivery:dm_notification"
    assert out[0]["lineup_position"] == 1 and out[0]["predicted_at"] == f"{D}T15:00:00.000000Z"


def test_skip_decision_with_declined_candidate_is_one_skip_day_row():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)))
    assert [(r["row_kind"], r["selection_id"], r["declined_batter_id"]) for r in out] == [("skip_day", None, 303)]


def test_skip_decision_with_commit_evidence_is_not_a_skip():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)),
               pick=picks(pick_json(D, notification_sent=True, notification_id="dm-9")))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_decision_with_commit_evidence")]


def test_skip_decision_beside_an_unparseable_but_delivered_pick_file_is_not_a_skip():
    # Codex plan r3 #5: a file-level delivery signal survives the loss of the file's only slot.
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)),
               pick=picks(pick_json(D, primary={"batter_id": "bad"}, notification_sent=True, notification_id="dm-9")))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_decision_with_commit_evidence")]


def test_a_skip_is_not_established_beside_unreadable_evidence():
    skip = dec(action="skip", primary=cand(303, 7001))
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, pick=picks(b'{"pick": {'))] == [
        ("unfinalized_day", "skip_decision_with_unreadable_pick_file")]
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, unusable={"scheduler_state"})] == [
        ("unfinalized_day", "skip_decision_with_unusable_state")]
    assert [(r["row_kind"], r["reason"]) for r in rows(decision=skip, pick=picks(pick_json(D)))] == [
        ("skip_day", "decision_skip")]          # a readable, undelivered preview does not block a skip


def test_unusable_evidence_is_never_absent_evidence():
    # Codex plan r3 #5: a quarantined decision, or quarantined history alone, is not an unobserved day.
    assert [(r["row_kind"], r["reason"]) for r in rows(pick=picks(pick_json(D)), unusable={"decision"})] == [
        ("unfinalized_day", "decision_unusable")]
    assert [(r["row_kind"], r["reason"]) for r in rows(unusable={"lineup_evolution"})] == [
        ("unfinalized_day", "unusable_evidence_only")]
    assert [(r["row_kind"], r["reason"]) for r in rows()] == [("unobserved_day", "no_evidence")]


def test_scheduler_skip_candidate_alone_is_unfinalized_intent():
    out = rows(st=state(final_skip_candidate={"primary": cand(303, 7001), "double": None}))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_intent_only")]


def test_commit_flag_names_no_selection_and_never_commits_one():
    # Codex plan r1 #3: a date-level flag is not evidence that the surviving preview was the commit.
    st = state(final_skip_candidate={"primary": cand(303, 7001), "double": None}, committed_pick_written=True)
    assert [(r["row_kind"], r["reason"]) for r in rows(st=st)] == [("unfinalized_day", "commit_flag_without_record")]
    (sel,) = rows(st=st, pick=picks(pick_json(D)))
    assert (sel["row_kind"], sel["commit_status"], sel["scheduler_commit_flag"]) == ("selection", "unconfirmed", True)


def test_archive_only_lineup_only_and_no_evidence_days():
    assert rows(observations=archived(A))[0]["reason"] == "archived_candidates_only"
    assert rows(observations=evo([(A, None)]))[0]["reason"] == "lineup_evolution_only"
    assert [(r["row_kind"], r["reason"]) for r in rows()] == [("unobserved_day", "no_evidence")]


def test_decision_only_selection_has_no_pick_file_fields():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)))
    assert (r["finalization"], r["commit_status"], r["lineup_position"], r["pick_obs_id"]) == (
        "decision", "committed_evidenced", None, None)
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (True, "decision_delivered")


def test_decision_and_pick_file_naming_different_selections_is_unresolved_without_attachment():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099}, result="hit",
                                          slot_results={"pick": "hit"})))
    assert (r["finalization"], r["pick_view_batter_id"], r["pick_view_game_pk"]) == ("unresolved", 999, 5099)
    assert r["pick_obs_id"] is None and r["delivered_at"] is None and r["lineup_position"] is None
    assert r["p_stated"] == 0.77 and r["commit_status"] == "committed_evidenced"   # the decision's own values


def test_single_decision_against_a_double_pick_file_is_unresolved_for_the_whole_set():
    # Codex plan r1 #4: the same primary must not let the double's delivery or results attach to the single.
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, dd={}, notification_sent=True, notification_id="dm-4", result="hit",
                                          slot_results={"pick": "hit", "double_down": "hit"})))
    assert (r["finalization"], r["pick_view_action"], r["pick_view_batter_id"]) == ("unresolved", "double", 101)
    assert (r["pick_obs_id"], r["delivered_at"], r["delivery_basis"]) == (None, None, "decision_delivered")
    assert (r["commit_status"], r["commit_basis"]) == ("conflicted", "decision:delivered;other:delivery:dm_notification")


def test_same_primary_with_a_different_double_down_is_unresolved():
    out = rows(decision=dec(action="double", primary=cand(101, 5001), double_down=cand(202, 5002, team="NYY")),
               pick=picks(pick_json(D, dd={"batter_id": 303, "game_pk": 5003})))
    assert [(r["slot"], r["finalization"], r["pick_view_batter_id"]) for r in out] == [
        ("primary", "unresolved", 101), ("double_down", "unresolved", 303)]


def test_delivered_other_selection_makes_the_decision_selection_conflicted():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099},
                                          notification_sent=True, notification_id="dm-2")))
    assert (r["commit_status"], r["commit_basis"]) == ("conflicted", "decision:delivered;other:delivery:dm_notification")


def test_missing_decision_with_a_delivered_pick_is_committed_via_delivery():
    (r,) = rows(pick=picks(pick_json(D, notification_sent=True, notification_id="dm-7")))
    assert (r["finalization"], r["commit_status"], r["commit_basis"]) == (
        "pick_file_only", "committed_evidenced", "delivery:dm_notification")


def test_undelivered_preview_is_unconfirmed_and_a_lock_alone_proves_nothing():
    (r,) = rows(pick=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["delivery_confirmed"], r["locked_at"]) == ("unconfirmed", None, None)


def test_private_commit_is_committed_without_a_delivery_claim():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["commit_basis"]) == ("committed_evidenced", "decision:private_locked")
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (False, "decision_private_locked")
    assert r["locked_at"] == f"{D}T21:00:00.000000Z"


def test_locked_unconfirmed_is_committed_with_unknown_delivery():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="locked_unconfirmed"),
                pick=picks(pick_json(D, delivery_attempted=True)))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == (
        "committed_evidenced", None, "decision_locked_unconfirmed")


def test_legacy_public_post_is_a_delivery_signal():
    (r,) = rows(pick=picks(pick_json(D, bluesky_posted=True, bluesky_uri="at://post/1")))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == ("committed_evidenced", True, "public_post")
    (r2,) = rows(pick=picks(pick_json(D, bluesky_posted=True)))
    assert (r2["commit_status"], r2["delivery_basis"]) == ("unconfirmed", "bluesky_posted_without_uri")


def test_positive_pick_signal_outranks_private_status_and_is_flagged():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick=picks(pick_json(D, notification_sent=True, notification_id="dm-3")))
    assert (r["delivery_confirmed"], r["delivery_basis"], r["delivery_evidence_conflict"]) == (True, "dm_notification", True)


def test_known_incomplete_needs_a_version_whose_content_is_gone():
    d, p = dec(action="single", primary=cand(101, 5001)), picks(pick_json(D))
    assert rows(decision=d, pick=p)[0]["history_status"] == "unknown"
    lost = evo([({"batter_id": 111, "game_pk": 5011}, None), (A, None)])
    assert rows(decision=d, pick=p, observations=lost)[0]["history_status"] == "known_incomplete"
    # Codex plan r1 #5: the same change with the earlier version retained in an archive proves nothing is gone.
    kept = lost + archived({"batter_id": 111, "game_pk": 5011})
    assert rows(decision=d, pick=p, observations=kept)[0]["history_status"] == "unknown"
    # A failed append for an earlier selection followed by an overwrite leaves only consistent evidence.
    assert rows(decision=d, pick=p, observations=evo([(A, None)]))[0]["history_status"] == "unknown"


def test_known_incomplete_is_withheld_while_a_record_that_might_hold_the_version_is_unusable():
    d = dec(action="single", primary=cand(101, 5001))
    lost = evo([({"batter_id": 111, "game_pk": 5011}, None), (A, None)])
    assert rows(decision=d, pick=picks(pick_json(D)), observations=lost)[0]["history_status"] == "known_incomplete"
    assert rows(decision=d, pick=picks(pick_json(D)), observations=lost,
                unusable={"archive"})[0]["history_status"] == "unknown"
    partial = picks(pick_json(D, dd={"batter_id": "bad"}))
    assert rows(decision=d, pick=partial, observations=lost)[0]["history_status"] == "unknown"


def test_discarded_double_down_preview_marks_the_single_known_incomplete():
    out = rows(decision=dec(action="single", primary=cand(101, 5001)), pick=picks(pick_json(D)),
               observations=evo([(A, B), (A, None)]))
    assert [(r["slot"], r["history_status"]) for r in out] == [("primary", "known_incomplete")]


def test_a_decision_whose_saved_pick_file_is_gone_is_known_incomplete():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)), observations=evo([(A, None)]))
    assert r["history_status"] == "known_incomplete"


def test_delivered_at_is_the_first_pick_file_signal_and_is_normalized():
    (r,) = rows(pick=picks(pick_json(D, delivered_at=f"{D}T13:36:00-04:00", notification_sent=True,
                                          notification_id="dm")))
    assert (r["delivery_basis"], r["delivered_at"]) == ("delivered_at", f"{D}T17:36:00.000000Z")


def test_a_quarantined_double_down_leg_prevents_agreement():
    # Codex plan r2 #4: an unusable DD leg must not make a double pick file look like the decision's single.
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick=picks(pick_json(D, dd={"batter_id": "not-an-id"}, notification_sent=True, notification_id="dm-5")))
    assert (r["finalization"], r["pick_view_action"], r["pick_file_complete"], r["pick_obs_id"]) == (
        "unresolved", "double", False, None)
    assert r["commit_status"] == "conflicted"


def test_an_unparseable_pick_file_is_a_conflicting_view_not_an_absence():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)), pick=picks(b'{"pick": {'))
    assert (r["finalization"], r["pick_file_complete"], r["pick_view_batter_id"]) == ("unresolved", False, None)
    assert [(x["row_kind"], x["reason"]) for x in rows(pick=picks(b'{"pick": {'))] == [
        ("unfinalized_day", "pick_file_unparseable")]
