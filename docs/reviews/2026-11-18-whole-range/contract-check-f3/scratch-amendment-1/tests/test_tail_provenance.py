"""Pick provenance carries the tail artifact's sha256 (2026-09-03, Codex r2).

``policy_npz_sha256`` identifies the reach-57 table, which is unchanged by the
tail policy; a tail-regime pick must also identify the bytes that chose it.
"""
from bts.picks import DailyPick, Pick, attach_provenance, compute_provenance, load_pick, save_pick


def _daily(date="2026-09-04"):
    pick = Pick(batter_name="A", batter_id=1, team="NYM", lineup_position=1, pitcher_name="P",
                pitcher_id=2, p_game_hit=0.72, flags=[], projected_lineup=False, game_pk=9,
                game_time="2026-09-04T23:10:00Z")
    return DailyPick(date=date, run_time="t", pick=pick, double_down=None, runner_up=None)


def test_compute_provenance_hashes_tail_when_present(tmp_path):
    tail = tmp_path / "mdp_tail_policy.npz"; tail.write_bytes(b"tail-bytes")
    prov = compute_provenance(blend_path=None, policy_path=None, tail_path=tail, cwd=tmp_path)
    import hashlib
    assert prov["tail_policy_sha256"] == hashlib.sha256(b"tail-bytes").hexdigest()
    assert compute_provenance(tail_path=tmp_path / "missing.npz", cwd=tmp_path)["tail_policy_sha256"] is None
    assert compute_provenance(cwd=tmp_path)["tail_policy_sha256"] is None


def test_attach_and_round_trip(tmp_path):
    tail = tmp_path / "mdp_tail_policy.npz"; tail.write_bytes(b"tail-bytes")
    d = attach_provenance(_daily(), tail_path=tail, cwd=tmp_path)
    assert d.tail_policy_sha256 and len(d.tail_policy_sha256) == 64
    save_pick(d, tmp_path)
    assert load_pick("2026-09-04", tmp_path).tail_policy_sha256 == d.tail_policy_sha256


def test_legacy_pick_without_field_loads_as_none(tmp_path):
    save_pick(_daily("2026-08-30"), tmp_path)
    assert load_pick("2026-08-30", tmp_path).tail_policy_sha256 is None
