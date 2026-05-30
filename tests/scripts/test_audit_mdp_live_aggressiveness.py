from __future__ import annotations

import json

import numpy as np

from bts.simulate.mdp import ACTIONS
from scripts.audit_mdp_live_aggressiveness import run_audit


def _write_policy(path, *, season_length=20):
    policy = np.full(
        (57, season_length + 1, 2, 2),
        ACTIONS.index("skip"),
        dtype=np.int8,
    )
    policy[:, :, :, 0] = ACTIONS.index("double")
    policy[:, :, :, 1] = ACTIONS.index("single")
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([0.75]),
        season_length=np.array(season_length),
        optimal_p57=np.array(0.0),
    )


def _write_pick(path, *, date, primary_p, double_p=None):
    body = {
        "date": date,
        "run_time": f"{date}T15:00:00+00:00",
        "pick": {
            "batter_name": f"Primary {date}",
            "batter_id": 100,
            "team": "AAA",
            "lineup_position": 1,
            "pitcher_name": "Pitcher",
            "pitcher_id": 200,
            "p_game_hit": primary_p,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 900,
            "game_time": f"{date}T20:00:00Z",
            "pitcher_team": "BBB",
        },
        "double_down": None,
        "runner_up": None,
        "result": None,
        "slot_results": None,
    }
    if double_p is not None:
        body["double_down"] = {
            "batter_name": f"Double {date}",
            "batter_id": 101,
            "team": "CCC",
            "lineup_position": 1,
            "pitcher_name": "Pitcher 2",
            "pitcher_id": 201,
            "p_game_hit": double_p,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 901,
            "game_time": f"{date}T21:00:00Z",
            "pitcher_team": "DDD",
        }
    path.write_text(json.dumps(body))


def test_run_audit_flags_low_q0_double_floor_candidates(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path)
    _write_pick(picks_dir / "2026-05-01.json", date="2026-05-01", primary_p=0.70, double_p=0.70)
    _write_pick(picks_dir / "2026-05-02.json", date="2026-05-02", primary_p=0.80, double_p=0.70)
    _write_pick(picks_dir / "2026-05-02.shadow.json", date="2026-05-02", primary_p=0.60, double_p=0.60)

    result = run_audit(
        picks_dir=picks_dir,
        policy_path=policy_path,
        streak_states=[0, 4],
        q0_double_floor=0.55,
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["summary"]["n"] == 2
    assert result["summary"]["qbin_counts"] == {"0": 1, "1": 1}
    assert result["summary"]["action_counts_by_streak"]["0"] == {"double": 1, "single": 1}
    assert result["summary"]["q0_double_floor_candidate_count"] == 1
    candidate = result["guardrail_candidates"][0]
    assert candidate["date"] == "2026-05-01"
    assert candidate["p_both"] == 0.48999999999999994
    assert candidate["q0_double_floor_flagged_streaks"] == ["0", "4"]
