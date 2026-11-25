import json
from datetime import datetime

from bts import daily_decision, skip_policy_shadow
from scripts.audit.boundary_shadow_census import run_census


def test_guarded_play_and_unchanged_skip_share_census(tmp_path):
    play, skip = "2026-03-25", "2026-03-26"
    guard = dict(rule="monotone_skip_v1", bin=4, from_bin=3, table_action="skip")
    fixtures = [
        (play, "single", 50, 0.96),
        (skip, "skip", 56, 0.99),
        ("2026-03-27", "skip", 56, 0.60),
        ("2026-03-28", "skip", 56, 0.70),
        ("2026-03-29", "skip", 56, 0.80),
    ]
    for day, action, streak, probability in fixtures:
        record = daily_decision.write_decision(
            day, tmp_path, action=action, source="mdp", objective="reach57",
            primary=dict(batter_id=100, p_game_hit=probability), streak=streak,
            saver_available=False, now=datetime.fromisoformat(f"{day}T22:00:00Z"),
            delivery_status="delivered" if action == "single" else "not_applicable",
            scoreable=action == "single", policy_guard=guard if action == "single" else None)
        assert record is not None
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(
        json.dumps(dict(recorded_at=f"{day}T00:00:00Z", active_streak=streak))
        for day, _, streak, _ in fixtures
    ))
    end = fixtures[-1][0]
    result = run_census(picks_dir=tmp_path, ledger_path=ledger,
                        policy_path="data/models/mdp_policy.npz", saver_active=False,
                        start=play, end=end, as_of=end, output=tmp_path / "census.json")
    rows = {row["date"]: row for row in result["rows"]}
    assert rows[play]["parity"] == "guarded_play_excluded"
    assert rows[play]["raw_deployed"] == "skip"
    assert rows[skip]["parity"] == "parity"
    assert result["boundaries"]["primary"]["n"] == len(fixtures)
    assert rows[play]["diffs"]["primary"] == {"from": "skip", "to": "single"}
    assert result["follow_up"]["primary_n_diff"] == 0
    assert skip_policy_shadow.record_skip_from_decision(play, tmp_path) is None
    assert skip_policy_shadow.record_skip_from_decision(skip, tmp_path) is not None
    records = [daily_decision.load_decision(day, tmp_path) for day, *_ in fixtures]
    assert records[1]["policy_guard"] is None
    old_play = dict(records[0], action="skip", policy_guard=None, scoreable=False)
    assert sum(map(daily_decision.is_reach57_mdp_skip, [old_play, *records[1:]])) == 5
    assert sum(map(daily_decision.is_reach57_mdp_skip, records)) == 4
