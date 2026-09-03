"""bts_daily_decision_v3: objective provenance (2026-09-03).

Once 57 is unreachable the MDP switches objective (reach57 -> emax_season_best).
An action alone can no longer be reproduced without the objective, the supplied
and effective best streak, the tail artifact's sha256, and whether the forced
fallback was used. v3 persists those; readers keep accepting v1/v2, and a record
without ``objective`` means reach57 (every record before 2026-09-03).
"""
from bts.daily_decision import (
    DECISION_SCHEMA, decision_objective, is_reach57_mdp_skip, load_decision, write_decision,
)


def _cand(bid=1, p=0.72, gpk=9):
    return {"batter_id": bid, "batter_name": f"B{bid}", "team": "NYM", "game_pk": gpk, "p_game_hit": p}


def test_writer_emits_v3_with_objective_fields(tmp_path):
    rec = write_decision(
        "2026-09-04", tmp_path, action="double", source="mdp", primary=_cand(1),
        double_down=_cand(2, gpk=11), streak=0, saver_available=False,
        state_source="contest", state_status="fresh", allow_double=True,
        contest_source_date="2026-09-03", delivery_status="delivered", scoreable=True,
        objective="emax_season_best", best_streak=18, best_status="trusted",
        effective_best=18, tail_policy_sha256="t" * 64, degraded_reason=None,
    )
    assert DECISION_SCHEMA == "bts_daily_decision_v3"
    assert rec["schema_version"] == DECISION_SCHEMA
    assert rec["objective"] == "emax_season_best"
    assert (rec["best_streak"], rec["best_status"], rec["effective_best"]) == (18, "trusted", 18)
    assert rec["tail_policy_sha256"] == "t" * 64 and rec["degraded_reason"] is None
    loaded = load_decision("2026-09-04", tmp_path)
    assert loaded["objective"] == "emax_season_best" and loaded["effective_best"] == 18


def test_legacy_v2_record_loads_and_means_reach57(tmp_path):
    import json
    from bts.daily_decision import decision_path
    p = decision_path("2026-08-30", tmp_path); p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"schema_version": "bts_daily_decision_v2", "date": "2026-08-30",
                             "action": "skip", "source": "mdp", "scoreable": False}))
    rec = load_decision("2026-08-30", tmp_path)
    assert rec is not None and "objective" not in rec
    assert decision_objective(rec) == "reach57"
    assert is_reach57_mdp_skip(rec) is True


def test_tail_skip_is_not_a_reach57_mdp_skip(tmp_path):
    rec = write_decision("2026-09-19", tmp_path, action="skip", source="mdp", primary=_cand(),
                         delivery_status="not_applicable", scoreable=False,
                         objective="emax_season_best", best_streak=18, best_status="trusted",
                         effective_best=18)
    assert is_reach57_mdp_skip(rec) is False
    assert is_reach57_mdp_skip({"action": "skip", "source": "heuristic"}) is False
    assert is_reach57_mdp_skip({"action": "single", "source": "mdp", "objective": "reach57"}) is False
    assert is_reach57_mdp_skip({"action": "skip", "source": "mdp", "objective": "reach57"}) is True
