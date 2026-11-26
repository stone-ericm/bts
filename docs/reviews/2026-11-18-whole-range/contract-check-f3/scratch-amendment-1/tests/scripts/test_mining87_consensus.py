"""#87 adapter: legal consensus pairs, ties, aliases and conservative public settlement (review r1 items 3, 11)."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit.mining87 import consensus as c


def _obs(user, slot, batter, *, name=None, unit=500, result="hit", date="2026-04-01",
         captured_at="2026-05-02T12:00:00"):
    return {"username": user, "pick_date": date, "pick_number": slot, "batter_id": batter,
            "batter_name": name or f"B{batter}", "unit_id": unit, "result": result,
            "captured_at": pd.Timestamp(captured_at)}


def _table(rows, *, users, cohort):
    """Votes always come from ``latest_observations`` (the vote identity / settlement-evidence frame)."""
    latest, _ = c.latest_observations(pd.DataFrame(rows))
    return c.consensus_table(latest, users=users, cohort=cohort)


# --- legal pair selection (amendment A2) -------------------------------------------------------------------------

def test_equal_primary_and_dd_modes_choose_the_legal_next_id_and_keep_the_slot_denominator():
    out = c.choose_slots({10: 3, 20: 1}, {10: 4, 30: 2})
    assert out["primary"]["batter_id"] == 10 and out["primary"]["tie_direct"] is False
    dd = out["dd"]
    assert dd["status"] == "chosen" and dd["batter_id"] == 30 and dd["count"] == 2
    assert dd["unconstrained_batter_id"] == 10 and dd["legal_differs_from_unconstrained"] is True
    assert dd["tie_direct"] is False and dd["tie_dependent_on_primary"] is False


def test_equal_modes_without_a_legal_alternative_leave_dd_consensus_unavailable():
    out = c.choose_slots({10: 3}, {10: 4})
    assert out["dd"]["status"] == "no_distinct_legal_id" and out["dd"]["batter_id"] is None


def test_unavailable_primary_makes_dd_consensus_unavailable_even_with_dd_votes():
    out = c.choose_slots({}, {30: 2})
    assert out["primary"]["status"] == "no_votes" and out["primary"]["batter_id"] is None
    assert out["dd"]["status"] == "primary_unavailable" and out["dd"]["batter_id"] is None


def test_a_tie_exposed_by_excluding_the_primary_breaks_to_the_lowest_id_and_counts_as_a_tie():
    out = c.choose_slots({10: 5}, {10: 6, 40: 2, 30: 2})
    dd = out["dd"]
    assert dd["batter_id"] == 30 and dd["tie_direct"] is True and dd["tie_exposed_by_exclusion"] is True
    assert dd["tied_ids"] == [30, 40]


def test_primary_tie_breaks_lowest_id_and_flags_a_dd_whose_legality_depends_on_it():
    dependent = c.choose_slots({20: 3, 10: 3}, {20: 4, 30: 1})
    assert dependent["primary"]["batter_id"] == 10 and dependent["primary"]["tie_direct"] is True
    assert dependent["dd"]["batter_id"] == 20 and dependent["dd"]["tie_direct"] is False
    assert dependent["dd"]["tie_dependent_on_primary"] is True        # primary 20 would have made the DD 30
    independent = c.choose_slots({10: 3, 20: 3}, {30: 4})
    assert independent["dd"]["batter_id"] == 30 and independent["dd"]["tie_dependent_on_primary"] is False


def test_a_dd_that_would_become_unavailable_under_the_other_tied_primary_is_dependent():
    out = c.choose_slots({10: 2, 20: 2}, {20: 3})
    assert out["dd"]["batter_id"] == 20 and out["dd"]["tie_dependent_on_primary"] is True


def test_an_unavailable_dd_that_another_tied_primary_would_make_available_is_dependent():
    """R2 finding 6: primary {1:1, 2:1}, DD {1:1}: primary 1 leaves no legal DD; primary 2 would make it 1."""
    out = c.choose_slots({1: 1, 2: 1}, {1: 1})
    assert out["primary"]["batter_id"] == 1 and out["dd"]["batter_id"] is None
    assert out["dd"]["status"] == "no_distinct_legal_id" and out["dd"]["tie_dependent_on_primary"] is True
    rows = [_obs("a", 1, 1), _obs("b", 1, 2), _obs("a", 2, 1, unit=600)]
    table, _ = _table(rows, users=None, cohort="all_tracked")
    dd = table[table["pick_number"] == 2].iloc[0]
    assert dd["consensus_available"] == False and pd.isna(dd["consensus_batter_id"])  # noqa: E712
    assert dd["consensus_tie_broken"] == True and dd["consensus_tie_reason"] == "dependent_on_primary_tie"  # noqa
    assert dd["consensus_settlement"] == "unavailable" and pd.isna(dd["consensus_hit"])


def test_dependence_flags_match_an_independent_oracle_over_every_small_count_table():
    from itertools import product
    mismatches = []
    for ac in product(range(3), repeat=3):
        a = {i + 1: n for i, n in enumerate(ac) if n}
        for bc in product(range(3), repeat=3):
            b = {i + 1: n for i, n in enumerate(bc) if n}
            got = c.choose_slots(a, b)
            winners = sorted(k for k, v in a.items() if v == max(a.values())) if a else []

            def legal(excluded):
                options = [(n, -bid, bid) for bid, n in b.items() if bid != excluded]
                return max(options)[2] if options else None

            expected = legal(winners[0]) if winners else None
            dependent = bool(b and len(winners) > 1 and any(legal(alt) != expected for alt in winners[1:]))
            if got["dd"]["batter_id"] != expected or got["dd"]["tie_dependent_on_primary"] != dependent:
                mismatches.append((a, b))
    assert mismatches == []


# --- votes by id, names as display only --------------------------------------------------------------------------

def test_votes_count_distinct_users_by_id_and_name_aliases_never_split_a_vote():
    rows = [_obs("u1", 1, 20, name="Jose Ramirez"), _obs("u2", 1, 20, name="José Ramírez"),
            _obs("u3", 1, 20, name="José Ramírez"), _obs("u4", 1, 20, name="Jose Ramirez"),
            _obs("u5", 1, 10), _obs("u6", 1, 10), _obs("u7", 1, 10)]
    table, votes = _table(rows, users=None, cohort="all_tracked")
    row = table[(table["pick_number"] == 1)].iloc[0]
    assert row["consensus_batter_id"] == 20 and row["consensus_pick_count"] == 4
    assert row["n_public_users"] == 7 and row["consensus_pick_share"] == pytest.approx(4 / 7)
    assert row["consensus_batter_name"] == "Jose Ramirez"         # deterministic: most common, then lexicographic
    assert set(votes.loc[votes["pick_number"] == 1, "batter_id"]) == {10, 20}


def test_cohort_restriction_and_dd_settlement_follow_the_legal_id_not_the_unconstrained_winner():
    rows = [_obs(u, 1, 10) for u in ("a", "b", "c")]
    rows += [_obs(u, 2, 10, result="hit") for u in ("a", "b", "c", "d")]     # unconstrained DD mode = the primary
    rows += [_obs(u, 2, 30, unit=600, result="not_hit") for u in ("e", "f")]
    table, _ = _table(rows, users=None, cohort="all_tracked")
    dd = table[table["pick_number"] == 2].iloc[0]
    assert dd["consensus_batter_id"] == 30 and dd["consensus_settlement"] == "not_hit"
    assert dd["consensus_hit"] == 0 and dd["consensus_pick_share"] == pytest.approx(2 / 6)
    fixed, _ = _table(rows, users={"a", "b", "c", "d"}, cohort="fixed_cohort")
    fdd = fixed[fixed["pick_number"] == 2].iloc[0]
    assert fdd["consensus_status"] == "no_distinct_legal_id" and pd.isna(fdd["consensus_batter_id"])
    assert fdd["consensus_available"] == False  # noqa: E712


# --- conservative settlement --------------------------------------------------------------------------------------

@pytest.mark.parametrize("results, units, status, reason", [
    (["hit", "hit", ""], [5, 5, 5], "hit", "settled"),
    (["not_hit", None], [5, 5], "not_hit", "settled"),
    (["hit", "not_hit", "hit"], [5, 5, 5], "unknown", "conflicting_results"),
    (["", None], [5, 5], "pending", "no_settled_observation"),
    (["void", "void"], [5, 5], "void", "void"),
    (["void", "hit"], [5, 5], "unknown", "conflicting_results"),
    (["hit", "hit"], [5, 6], "unknown", "multiple_units_for_batter"),
    (["hit"], [0], "unknown", "settled_observation_without_unit_identity"),
    (["postponed"], [5], "unknown", "unrecognized_result"),
    # R2 finding 2: a settled label never borrows another observation's unit
    (["hit", ""], [0, 500], "unknown", "settled_observation_without_unit_identity"),
    (["not_hit", ""], [0, 500], "unknown", "settled_observation_without_unit_identity"),
    (["void", ""], [0, 500], "unknown", "settled_observation_without_unit_identity"),
    (["hit", None], [None, 500], "unknown", "settled_observation_without_unit_identity"),
    (["hit", ""], [500, 500], "hit", "settled"),                     # hit + pending on the same unit: supported
    (["hit", ""], [500, 0], "hit", "settled"),                       # a unit-less pending observation transfers nothing
    (["", ""], [0, 500], "pending", "no_settled_observation"),
    (["", None], [0, 0], "unknown", "no_unit_identity"),
])
def test_settlement_is_conservative_and_never_a_majority_vote(results, units, status, reason):
    got_status, got_reason, unit = c.consensus_settlement(results, units)
    assert (got_status, got_reason) == (status, reason)
    if reason == "settled_observation_without_unit_identity":
        assert unit is None


def test_an_unbound_settled_label_leaves_the_selected_id_and_its_settlement_unknown():
    for label in ("hit", "not_hit", "void"):
        rows = [_obs("a", 1, 20, unit=0, result=label), _obs("b", 1, 20, unit=500, result=""),
                _obs("c", 1, 10, unit=501, result="hit")]
        table, _ = _table(rows, users=None, cohort="all_tracked")
        row = table.iloc[0]
        assert row["consensus_batter_id"] == 20 and row["consensus_pick_count"] == 2
        assert row["consensus_settlement"] == "unknown" and pd.isna(row["consensus_unit_id"])
        assert row["consensus_settlement_reason"] == "settled_observation_without_unit_identity"
        assert row["consensus_resolved"] == False  # noqa: E712


def test_consensus_hit_is_binary_only_for_settled_hits_and_misses():
    rows = [_obs("a", 1, 10, result="void"), _obs("b", 1, 10, result="void")]
    table, _ = _table(rows, users=None, cohort="all_tracked")
    row = table.iloc[0]
    assert row["consensus_settlement"] == "void" and pd.isna(row["consensus_hit"])
    assert row["consensus_resolved"] == False  # noqa: E712


# --- observation dedup ---------------------------------------------------------------------------------------------

def test_latest_observation_wins_and_conflicting_same_stamp_rows_are_ambiguous_not_votes():
    rows = [_obs("a", 1, 10, captured_at="2026-05-01T10:00:00"), _obs("a", 1, 11, captured_at="2026-05-01T12:00:00"),
            _obs("b", 1, 10, captured_at="2026-05-01T12:00:00"), _obs("b", 1, 12, captured_at="2026-05-01T12:00:00"),
            _obs("c", 1, 10), _obs("c", 1, 10)]                     # exact duplicate append: one observation
    latest, inv = c.latest_observations(pd.DataFrame(rows))
    by_user = {r["username"]: r for r in latest.to_dict("records")}
    assert by_user["a"]["batter_id"] == 11 and "b" not in by_user and by_user["c"]["batter_id"] == 10
    assert inv["ambiguous_user_slot_observations"] == 1


def test_a_result_conflict_never_changes_the_winner_and_leaves_its_settlement_unknown():
    """R2 finding 1: a,b choose 20 and c chooses 10; a second latest-stamp row for a differs only in the result."""
    rows = [_obs("a", 1, 20), _obs("b", 1, 20), _obs("c", 1, 10), _obs("a", 1, 20, result="not_hit")]
    latest, inv = c.latest_observations(pd.DataFrame(rows))
    assert sorted(latest["username"]) == ["a", "b", "c"] and inv["ambiguous_user_slot_observations"] == 0
    a = latest[latest["username"] == "a"].iloc[0]
    assert a["settlement_evidence"] == ((500, "hit"), (500, "not_hit"))     # all tied evidence, none chosen
    table, _ = c.consensus_table(latest, users=None, cohort="all_tracked")
    row = table.iloc[0]
    assert row["consensus_batter_id"] == 20 and row["consensus_pick_count"] == 2
    assert row["n_public_users"] == 3 and row["consensus_pick_share"] == pytest.approx(2 / 3)
    assert row["consensus_settlement"] == "unknown" and row["consensus_settlement_reason"] == "conflicting_results"


def test_a_unit_conflict_with_the_same_batter_keeps_the_vote_and_leaves_settlement_unknown():
    rows = [_obs("a", 1, 20), _obs("b", 1, 20), _obs("c", 1, 10), _obs("a", 1, 20, unit=600)]
    table, _ = _table(rows, users=None, cohort="all_tracked")
    row = table.iloc[0]
    assert row["consensus_batter_id"] == 20 and row["consensus_pick_count"] == 2
    assert row["consensus_settlement"] == "unknown" and row["consensus_settlement_reason"] == "multiple_units_for_batter"


def test_a_vote_needs_one_valid_batter_id_at_the_latest_stamp():
    rows = [_obs("a", 1, 20), _obs("a", 1, None),                       # valid and null id together: no vote
            _obs("b", 1, None),                                         # null id only: no vote
            _obs("c", 1, 0), _obs("d", 1, 30)]                          # nonpositive id: no vote
    latest, inv = c.latest_observations(pd.DataFrame(rows))
    assert list(latest["username"]) == ["d"]
    assert inv["ambiguous_user_slot_observations"] == 1 and inv["user_slots_without_valid_batter_id"] == 2
    assert inv["user_slot_observations"] == 4


def test_consensus_table_refuses_rows_that_did_not_come_through_latest_observations():
    with pytest.raises(ValueError, match="latest_observations"):
        c.consensus_table(pd.DataFrame([_obs("a", 1, 20)]), users=None, cohort="all_tracked")


def test_a_name_alias_for_the_same_id_at_the_same_stamp_is_not_an_ambiguity():
    rows = [_obs("a", 1, 10, name="Jose Ramirez"), _obs("a", 1, 10, name="José Ramírez")]
    latest, inv = c.latest_observations(pd.DataFrame(rows))
    assert len(latest) == 1 and inv["ambiguous_user_slot_observations"] == 0
    assert latest.iloc[0]["batter_name"] == "Jose Ramirez"


# --- inputs ---------------------------------------------------------------------------------------------------------

def _write_picks(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for r in rows:
        full.append({"captured_at": r["captured_at"], "round_id": 1, "pick_date": pd.Timestamp(r["pick_date"]).date(),
                     "pick_number": r["pick_number"], "unit_id": r["unit_id"], "bts_player_id": 1,
                     "result": r["result"], "at_bats": 4, "hits": 1, "streak_after": 1, "batter_id": r["batter_id"],
                     "batter_name": r["batter_name"], "batter_team": "BOS", "opponent_team": "NYY",
                     "home_or_away": "home"})
    pd.DataFrame(full).to_parquet(path, index=False)


def test_public_picks_are_bounded_by_the_window_and_the_capture_cutoff(tmp_path):
    d = tmp_path / "user_picks"
    _write_picks(d / "alice.parquet", [
        _obs("x", 1, 10, date="2026-03-25"),                                            # before the window
        _obs("x", 1, 10, date="2026-03-26"),
        _obs("x", 1, 10, date="2026-07-03", captured_at="2026-07-04T23:59:00"),
        _obs("x", 1, 10, date="2026-07-04", captured_at="2026-07-04T23:59:00"),         # after the window
        _obs("x", 2, 11, date="2026-07-03", captured_at="2026-07-05T00:00:00"),         # after the capture cutoff
        _obs("x", 3, 12, date="2026-06-01"),                                            # invalid slot number
    ])
    _write_picks(d / "bob.parquet", [])
    _write_picks(d / "carol.parquet", [_obs("x", 1, 10, date="2026-04-01", captured_at="2026-05-01T10:00:00"),
                                       _obs("x", 1, 11, date="2026-04-01", captured_at="2026-05-02T10:00:00"),
                                       _obs("x", 2, 12, date="2026-04-01", captured_at="2026-05-02T10:00:00"),
                                       _obs("x", 2, 13, date="2026-04-01", captured_at="2026-05-02T10:00:00")])
    obs, inv = c.load_public_picks(d, window_start="2026-03-26", window_end="2026-07-03",
                                   capture_end_exclusive="2026-07-05T00:00:00")
    alice = obs[obs["username"] == "alice"]
    assert sorted(alice["pick_date"]) == ["2026-03-26", "2026-07-03"] and set(obs["username"]) == {"alice", "carol"}
    carol = obs[obs["username"] == "carol"]                      # reduced per file to the latest observation
    assert list(carol["batter_id"]) == [11]
    assert inv["user_pick_files"] == 3 and inv["empty_user_pick_files"] == 1
    assert inv["rows_outside_window"] == 2 and inv["rows_after_capture_cutoff"] == 1
    assert inv["rows_invalid_pick_number"] == 1 and inv["ambiguous_user_slot_observations"] == 1
    assert inv["user_slot_observations"] == 4
    assert inv["pick_date_range_read"] == ["2026-03-25", "2026-07-04"]                # verifies the inventory claim
    assert inv["captured_at_range_read"] == ["2026-05-01T10:00:00", "2026-07-05T00:00:00"]


def test_cohort_is_the_pinned_snapshots_active_streak_tab_through_the_scraper_sanitizer(tmp_path):
    snap = tmp_path / "2026-07-04.parquet"
    rows = [{"captured_at": pd.Timestamp("2026-07-04T12:00:00"), "tab": "active_streak", "rank": i + 1,
             "username": u, "streak": 10, "hits_today": 0} for i, u in enumerate(["a b", "c/d", "e", "x y", "x_y"])]
    rows.append({"captured_at": pd.Timestamp("2026-07-04T12:00:00"), "tab": "all_season", "rank": 1,
                 "username": "zz", "streak": 30, "hits_today": 0})
    pd.DataFrame(rows).to_parquet(snap, index=False)
    usernames, meta = c.load_cohort(snap, tab="active_streak")
    assert usernames == {"a b", "c/d", "e", "x y", "x_y"} and meta["n_usernames"] == 5
    stems, smeta = c.cohort_stems(usernames, available_stems={"a_b", "c_d", "x_y", "other"})
    assert stems == {"a_b", "c_d", "x_y"}
    assert smeta["usernames_without_pick_file"] == 1 and smeta["sanitization_collisions"] == 1
    assert len(meta["membership_sha256"]) == 64
